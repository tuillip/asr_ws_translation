# asr_ws_final_final.py — English speech → English on-screen text
# Architecture preserved:
#   - .env for ASSEMBLYAI_API_KEY
#   - local caption WS on ws://127.0.0.1:8765
#   - spanish_caption.html consumed unchanged (it just displays text we send)
#
# Changes in this version:
#   - DeepL/glossary removed/commented out
#   - Display cadence = fast partials (debounce 160 ms) + pause-commit (700 ms)
#   - Always show <= 20 words (server trims before sending)

import os, asyncio, json, time, threading, traceback, re
from datetime import datetime

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

# -------------------- ENV / CONFIG --------------------
load_dotenv()
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY") or ""

# AssemblyAI Universal Streaming WS (your working endpoint style)
AAI_URL = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&encoding=pcm_s16le"

# Audio
RATE       = 16000
CHUNK_MS   = 100
BLOCK      = int(RATE * CHUNK_MS / 1000)

# Cadence knobs (mirrors the earlier "stt" behavior)
MAX_WORDS           = int(os.getenv("MAX_WORDS", 20))   # on-screen cap
DEBOUNCE_MS         = int(os.getenv("DEBOUNCE_MS", 160))  # partial update debounce
PAUSE_COMMIT_MS     = int(os.getenv("PAUSE_COMMIT_MS", 700))  # commit window after silence

# Logging (optional)
LOG_PATH = os.getenv("CAPTION_LOG", "captions_log.tsv")

# -------------------- RUNTIME STATE --------------------
audio_q  = asyncio.Queue()
clients  = set()

WORD_RX = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9’'--]+")

def log_caption(kind: str, text: str):
    try:
        ts = datetime.now().isoformat(timespec="seconds")
        safe_text = text.replace("\t", " ")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{kind}\t{safe_text}\n")
    except Exception:
        pass

# -------------------- CAPTION WS (unchanged) --------------------
async def caption_server():
    async def handler(ws):
        print("Caption client connected")
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
            print("Caption client disconnected")
    server = await websockets.serve(handler, "0.0.0.0", 8765, ping_interval=20, ping_timeout=20)
    print("Caption WS at ws://127.0.0.1:8765 (open spanish_caption.html)")
    return server

async def push_caption(text: str):
    if not clients:
        return
    payload = json.dumps({"text": text})
    # fire-and-forget broadcast
    await asyncio.gather(*[c.send(payload) for c in list(clients)], return_exceptions=True)

# -------------------- CADENCE DISPLAY (browser-like on the server) --------------------
class CadenceDisplay:
    """
    Mimics the earlier browser cadence:
      - Stream partial text quickly (debounced 160 ms)
      - On ~700 ms of silence (or end_of_turn), commit what's showing
      - Keep only the last MAX_WORDS visible
    """
    def __init__(self):
        self.stable_words: list[str]      = []  # committed words
        self.provisional_words: list[str] = []  # last partial's words
        self.last_change_ms: float        = 0.0
        self.last_sent_text: str          = ""
        self._debounce_until: float       = 0.0
        self._lock = asyncio.Lock()

    def _now(self) -> float:
        return time.time() * 1000.0

    def _last_n(self, words: list[str], n: int) -> str:
        if not words: return ""
        return " ".join(words[-n:])

    async def update_partial(self, transcript_text: str):
        words = [w for w in WORD_RX.findall(transcript_text)]
        async with self._lock:
            self.provisional_words = words
            self.last_change_ms = self._now()

    async def maybe_send_debounced_partial(self):
        async with self._lock:
            now = self._now()
            if now < self._debounce_until:
                return
            merged = self.stable_words + self.provisional_words
            visible = self._last_n(merged, MAX_WORDS)
            if visible and visible != self.last_sent_text:
                await push_caption(visible)
                self.last_sent_text = visible
                self._debounce_until = now + DEBOUNCE_MS

    async def commit_if_paused(self):
        async with self._lock:
            if not self.provisional_words:
                return False
            if (self._now() - self.last_change_ms) >= PAUSE_COMMIT_MS:
                # move provisional into stable and send (keeps same on-screen text)
                self.stable_words += self.provisional_words
                self.provisional_words = []
                merged = self.stable_words
                visible = self._last_n(merged, MAX_WORDS)
                if visible and visible != self.last_sent_text:
                    await push_caption(visible)
                    self.last_sent_text = visible
                return True
            return False

    async def end_of_turn(self):
        async with self._lock:
            if self.provisional_words:
                self.stable_words += self.provisional_words
                self.provisional_words = []
            merged = self.stable_words
            visible = self._last_n(merged, MAX_WORDS)
            if visible and visible != self.last_sent_text:
                await push_caption(visible)
                self.last_sent_text = visible
            # New turn: clear all words so next utterance starts fresh on screen
            self.stable_words = []
            self.provisional_words = []

display = CadenceDisplay()

# -------------------- ONE RUN (audio thread + AAI realtime) --------------------
async def run_once():
    loop = asyncio.get_running_loop()

    # --- Audio thread (unchanged) ---
    def audio_thread_func():
        print("Audio thread starting…")
        try:
            with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
                print("Mic started.")
                while True:
                    indata, overflowed = stream.read(BLOCK)
                    if overflowed:
                        print("Audio overflow!", flush=True)
                    pcm_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                    loop.call_soon_threadsafe(audio_q.put_nowait, pcm_bytes)
        except Exception as e:
            print(f"\nAudio thread error: {e}")
            loop.call_soon_threadsafe(audio_q.put_nowait, None)

    threading.Thread(target=audio_thread_func, daemon=True).start()

    # --- AAI realtime (v3) ---
    print("Connecting to AssemblyAI…")
    async with websockets.connect(
        AAI_URL,
        extra_headers=[("Authorization", AAI_KEY)],
        ping_interval=5, ping_timeout=20, max_size=10_000_000
    ) as ws:
        print("Connected to AssemblyAI.")

        async def sender():
            try:
                while True:
                    pcm = await audio_q.get()
                    if pcm is None:
                        print("[SENDER] audio thread ended")
                        break
                    # Your previous pipeline used raw PCM frames on v3 ws (encoding=pcm_s16le)
                    await ws.send(pcm)
                    audio_q.task_done()
            except Exception:
                print("\n[SENDER EXIT]")
                traceback.print_exc()
                raise

        async def receiver():
            """Drive the cadence:
               - update partials quickly
               - debounced push every ~160 ms
               - commit after 700 ms of silence or on end_of_turn
            """
            try:
                while True:
                    # Read WS but don't block cadence checks for too long
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.08)
                        msg = json.loads(raw)
                    except asyncio.TimeoutError:
                        msg = None
                    except websockets.exceptions.ConnectionClosed:
                        raise
                    except Exception:
                        # bad frame? skip
                        continue

                    # Parse transcript text across v3/v2 shapes
                    transcript = ""
                    end_turn = False
                    if isinstance(msg, dict):
                        # v3 Turn
                        if msg.get("type") == "Turn" or "transcript" in msg:
                            transcript = msg.get("transcript", "") or ""
                            end_turn = bool(msg.get("end_of_turn"))
                        # v2 fallback
                        elif msg.get("type") in ("PartialTranscript", "FinalTranscript"):
                            transcript = msg.get("text", "") or ""
                            end_turn = (msg.get("type") == "FinalTranscript")

                    # Update live partial buffer
                    if transcript:
                        await display.update_partial(transcript)
                        log_caption("partial", transcript)

                    # Debounced partial push (gives fast on-screen updates)
                    await display.maybe_send_debounced_partial()

                    # Pause commit
                    committed = await display.commit_if_paused()
                    if committed:
                        log_caption("commit", "pause")

                    # End of turn flush
                    if end_turn:
                        await display.end_of_turn()
                        log_caption("commit", "end_of_turn")

            except Exception:
                print("\n[RECEIVER EXIT]")
                traceback.print_exc()
                raise

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for p in pending: p.cancel()
        for d in done:
            if d.exception():
                raise d.exception()

# -------------------- MAIN LOOP --------------------
async def main():
    if not AAI_KEY:
        print("Missing ASSEMBLYAI_API_KEY in .env")
        return
    await caption_server()
    while True:
        try:
            # drain any stale audio on reconnect
            while not audio_q.empty():
                audio_q.get_nowait()
                audio_q.task_done()
            await run_once()
        except Exception as e:
            print(f"Run failed; retrying in 3s… {repr(e)}")
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
