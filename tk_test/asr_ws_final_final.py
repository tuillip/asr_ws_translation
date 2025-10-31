# asr_ws_final_final.py
# Mic -> AssemblyAI (your current transport) -> DeepL (ES) -> WebSocket -> spanish_caption.html
# Version 4: Early first-chunk bias for low initial latency; rest unchanged (pager + batching).

import os, asyncio, json, queue, time, traceback, re, threading
from datetime import datetime
import numpy as np
import sounddevice as sd
import websockets, aiohttp
from dotenv import load_dotenv

# -------------------- Config / Env --------------------
load_dotenv()
AAI_KEY   = os.getenv("ASSEMBLYAI_API_KEY") or ""
DEEPL_KEY = os.getenv("DEEPL_AUTH_KEY") or ""
DEEPL_URL = "https://api-free.deepl.com/v2/translate" if os.getenv("DEEPL_BASE","free").lower()=="free" \
            else "https://api.deepl.com/v2/translate"

# Keep your existing (working) AAI URL/headers
AAI_URL   = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&encoding=pcm_s16le"

# Glossary settings
GLOSSARY_NAME = os.getenv("DEEPL_GLOSSARY_NAME", "church-spanish-glossary")
GLOSSARY_PATH = os.getenv("DEEPL_GLOSSARY_TSV", "glossary.tsv")
SRC_LANG, TGT_LANG = "EN", "ES"

# Logging
LOG_PATH = os.getenv("CAPTION_LOG", "captions_log.tsv")

# Latency knobs
RATE, CHUNK_MS = 16000, 100
BLOCK = int(RATE * CHUNK_MS / 1000)
FLUSH_MS   = int(os.getenv("FLUSH_MS", 700))      # Force translate if paused >= this long

# Display / pager knobs (from previous versions)
PAGE_LIMIT      = int(os.getenv("PAGE_WORDS", 20))
PER_WORD_MS     = int(os.getenv("PER_WORD_MS", 350))
MIN_HOLD_MS     = int(os.getenv("MIN_HOLD_MS", 1200))
PAGE_MIN_HOLD_MS= int(os.getenv("PAGE_HOLD_MS", 3000))
MIN_DISPLAY_MS  = int(os.getenv("MIN_DISPLAY_MS", 3000))

# Small-batch threshold (from V3)
SMALL_BATCH_WORDS = int(os.getenv("SMALL_BATCH_WORDS", 5))

# NEW: Early first-chunk knobs
FIRST_CHUNK_MIN_WORDS   = int(os.getenv("FIRST_CHUNK_MIN_WORDS", 5))
FIRST_CHUNK_MAX_WAIT_MS = int(os.getenv("FIRST_CHUNK_MAX_WAIT_MS", 350))
FIRST_CHUNK_HOLD_MS     = int(os.getenv("FIRST_CHUNK_HOLD_MS", 900))

# Optional testing toggle from earlier
SHOW_ENGLISH_TESTING = os.getenv("SHOW_ENGLISH_TESTING", "false").lower() == "true"

audio_q = asyncio.Queue()
clients = set()

# -------------------- Utilities --------------------
def log_pair(en_text, es_text, kind):
    ts = datetime.now().isoformat(timespec="seconds")
    en_clean = (en_text or "").replace("\t", " ")
    es_clean = (es_text or "").replace("\t", " ")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{ts}\t{kind}\t{en_clean}\t{es_clean}\n")

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

async def push_caption(text):
    if not clients: return
    payload = json.dumps({"text": text})
    await asyncio.gather(*[c.send(payload) for c in list(clients)], return_exceptions=True)

# -------------------- DeepL helpers --------------------
async def translate_es(session, text_en, glossary_id=None):
    """Translate EN->ES; never block UI: on error or missing key, return English."""
    if not text_en or not text_en.strip(): return ""
    if not DEEPL_KEY:
        return text_en
    data = {"text": text_en, "target_lang": TGT_LANG}
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_KEY}"}
    if glossary_id:
        data["source_lang"] = SRC_LANG
        data["glossary_id"] = glossary_id
    try:
        async with session.post(
            DEEPL_URL, data=data, headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            r.raise_for_status()
            js = await r.json()
            return js["translations"][0]["text"]
    except Exception as e:
        print("(DeepL error; showing English) ->", repr(e))
        return text_en

def sanitize_tsv_to_entries(tsv_path: str) -> str | None:
    if not os.path.exists(tsv_path):
        print(f"(No {tsv_path}; continuing without glossary.)")
        return None
    raw = open(tsv_path, "r", encoding="utf-8-sig").read()
    lines = raw.replace("\r\n","\n").replace("\r","\n").split("\n")
    cleaned = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if "\t" not in s or s.startswith("\t"):
            print(f"(glossary.tsv line {i} invalid, skipping)")
            continue
        left, right = s.split("\t", 1)
        left, right = left.strip(), right.strip()
        if not left or not right:
            print(f"(glossary.tsv line {i} empty term, skipping)")
            continue
        if "\t" in right: right = right.split("\t",1)[0].strip()
        cleaned.append(f"{left}\t{right}")
    if not cleaned:
        print("(glossary.tsv has no valid rows; continuing without glossary.)")
        return None
    return "\n".join(cleaned)

async def ensure_glossary(session) -> str | None:
    entries_tsv = sanitize_tsv_to_entries(GLOSSARY_PATH)
    if not entries_tsv or not DEEPL_KEY:
        return None
    base = "https://api-free.deepl.com" if "api-free" in DEEPL_URL else "https://api.deepl.com"
    list_url = f"{base}/v2/glossaries"
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_KEY}"}
    existing_id = None
    try:
        async with session.get(list_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                js = await r.json()
                for g in js.get("glossaries", []):
                    if g.get("name")==GLOSSARY_NAME and g.get("source_lang","").upper()==SRC_LANG and g.get("target_lang","").upper()==TGT_LANG:
                        existing_id = g.get("glossary_id"); break
    except Exception as e:
        print("(Glossary list failed; continuing.)", e)
    if existing_id:
        put_url = f"{list_url}/{existing_id}/entries"
        async with session.put(
            put_url, headers=headers,
            data={"entries": entries_tsv, "entries_format": "tsv"},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as rr:
            if rr.status in (200, 204):
                print(f"Updated glossary '{GLOSSARY_NAME}'.")
                return existing_id
            else:
                txt = await rr.text()
                print(f"(Update glossary failed: {rr.status} {txt})")
                return existing_id
    async with session.post(
        list_url, headers=headers,
        data={"name": GLOSSARY_NAME, "source_lang": SRC_LANG, "target_lang": TGT_LANG,
              "entries": entries_tsv, "entries_format": "tsv"},
        timeout=aiohttp.ClientTimeout(total=10)
    ) as r:
        if r.status in (200, 201):
            gid = (await r.json()).get("glossary_id")
            print(f"Created glossary '{GLOSSARY_NAME}': {gid}")
            return gid
        else:
            txt = await r.text()
            print(f"(Glossary create failed: {r.status} {txt})")
            return existing_id

# -------------------- Case-insensitive normalizer --------------------
def load_glossary_normalizer(path: str) -> dict:
    if not os.path.exists(path): return {}
    terms = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#") or "\t" not in s: continue
            src = s.split("\t", 1)[0].strip()
            if not src: continue
            key = src.lower()
            if key not in terms:
                terms[key] = src
    return terms

def build_normalizer_regex(terms_ci: dict):
    if not terms_ci: return None
    parts = [re.escape(k) for k in sorted(terms_ci.keys(), key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)

def normalize_en_for_glossary(text: str, terms_ci: dict, rx) -> str:
    if not text or not rx: return text
    return rx.sub(lambda m: terms_ci.get(m.group(0).lower(), m.group(0)), text)

# -------------------- Natural pager (unchanged, with a tiny helper) --------------------
class NaturalPager:
    """Buffers Spanish words; pushes ≤ PAGE_LIMIT with a reading hold each time."""
    def __init__(self):
        self.pending = []
        self.in_hold_until_ms = 0
        self._lock = asyncio.Lock()
    def _now(self): return int(time.time()*1000)
    def _hold_for(self, n):
        base = max(MIN_HOLD_MS, n * PER_WORD_MS)
        if n >= PAGE_LIMIT: base = max(base, PAGE_MIN_HOLD_MS)
        return base
    async def add_text(self, es_text: str):
        words = [w for w in es_text.strip().split() if w]
        if not words: return
        async with self._lock:
            self.pending.extend(words)
    async def try_push(self, force=False):
        now = self._now()
        async with self._lock:
            if not self.pending: return False
            if not force and now < self.in_hold_until_ms: return False
            n = min(len(self.pending), PAGE_LIMIT)
            chunk = self.pending[:n]
            self.pending = self.pending[n:]
            await push_caption(" ".join(chunk))
            self.in_hold_until_ms = now + self._hold_for(len(chunk))
            return True
    # NEW: push immediately with a custom hold (used for first chunk)
    async def push_now_with_hold(self, hold_ms: int):
        now = self._now()
        async with self._lock:
            if not self.pending: return False
            n = min(len(self.pending), PAGE_LIMIT)
            chunk = self.pending[:n]
            self.pending = self.pending[n:]
            await push_caption(" ".join(chunk))
            self.in_hold_until_ms = now + max(0, int(hold_ms))
            return True
    async def flush_all(self):
        while True:
            async with self._lock:
                empty = (len(self.pending) == 0)
                wait_ms = max(0, self.in_hold_until_ms - self._now())
            if empty and wait_ms == 0: return
            if wait_ms > 0: await asyncio.sleep(wait_ms/1000.0)
            pushed = await self.try_push(force=True)
            if not pushed and wait_ms == 0: return
            await asyncio.sleep(0.05)

pager = NaturalPager()

# -------------------- One run cycle (with dedicated audio thread) --------------------
WORD_RX = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9’'\-]+")

async def run_once():
    loop = asyncio.get_running_loop()

    # Audio thread
    def audio_thread_func():
        print("Audio thread starting...")
        try:
            with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
                print("Mic started.")
                while True:
                    indata, overflowed = stream.read(BLOCK)
                    if overflowed:
                        print("Audio overflow!", flush=True)
                    pcm_bytes = (indata[:,0] * 32767).astype(np.int16).tobytes()
                    loop.call_soon_threadsafe(audio_q.put_nowait, pcm_bytes)
        except Exception as e:
            print(f"\nAudio thread error: {e}")
            loop.call_soon_threadsafe(audio_q.put_nowait, None)

    threading.Thread(target=audio_thread_func, daemon=True).start()

    print("Connecting to AssemblyAI…")
    async with aiohttp.ClientSession() as http, websockets.connect(
        AAI_URL, additional_headers=[("Authorization", AAI_KEY)],
        ping_interval=5, ping_timeout=20, max_size=10_000_000
    ) as ws:
        print("Connected to AssemblyAI.")

        # Glossary (optional)
        try:
            glossary_id = await ensure_glossary(http)
            if glossary_id: print("Glossary ready:", glossary_id)
        except Exception as e:
            print("(Glossary step failed; continuing without it.)", e)
            glossary_id = None

        terms_ci = load_glossary_normalizer(GLOSSARY_PATH)
        terms_rx = build_normalizer_regex(terms_ci)

        current_en = ""
        last_change_ms = 0.0
        last_sent_ms   = 0.0
        sent_word_count = 0

        # NEW: per-phrase early-first-chunk tracking
        first_word_time_ms = 0.0
        first_chunk_sent    = False

        async def sender():
            try:
                while True:
                    pcm = await audio_q.get()
                    if pcm is None:
                        print("[SENDER] Audio thread died."); break
                    await ws.send(pcm)  # unchanged transport
                    audio_q.task_done()
            except Exception:
                print("\n[SENDER EXIT]"); traceback.print_exc(); raise

        async def receiver():
            nonlocal current_en, last_change_ms, last_sent_ms, sent_word_count
            nonlocal first_word_time_ms, first_chunk_sent
            try:
                while True:
                    # Periodic timeout to allow pause logic
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        data = json.loads(raw)
                        t = data.get("transcript","") or data.get("text","")
                    except asyncio.TimeoutError:
                        data = {}; t = ""

                    if t:
                        if not current_en:
                            # phrase started
                            first_word_time_ms = time.time()*1000
                            first_chunk_sent = False
                        if t != current_en:
                            current_en = t
                            last_change_ms = time.time()*1000

                    now = time.time()*1000
                    cooled = (now - last_sent_ms) >= MIN_DISPLAY_MS

                    # Tokenize & compute unsent tail by index
                    words = WORD_RX.findall(current_en) if current_en else []
                    unsent_count = max(0, len(words) - sent_word_count)

                    send_reason = None

                    # --- NEW: early first-chunk bias ---
                    if not first_chunk_sent and unsent_count > 0:
                        time_since_first = now - (first_word_time_ms or now)
                        if (unsent_count >= FIRST_CHUNK_MIN_WORDS) or (time_since_first >= FIRST_CHUNK_MAX_WAIT_MS):
                            send_reason = "first_chunk"

                    # existing triggers
                    if send_reason is None:
                        if unsent_count >= PAGE_LIMIT and cooled:
                            send_reason = "partial_words"
                        elif unsent_count >= SMALL_BATCH_WORDS and cooled:
                            send_reason = "partial_small"
                        is_paused = current_en and (now - last_change_ms >= FLUSH_MS)
                        if unsent_count > 0 and is_paused:
                            send_reason = "pause_flush"
                        if data.get("end_of_turn"):
                            send_reason = "end_of_turn"

                    if send_reason:
                        en_slice = " ".join(words[sent_word_count:]) if unsent_count > 0 else ""
                        if en_slice:
                            en_norm = normalize_en_for_glossary(en_slice, terms_ci, terms_rx)
                            es = await translate_es(http, en_norm, glossary_id)
                            display_text = en_norm if SHOW_ENGLISH_TESTING else es
                            await pager.add_text(display_text)

                            if send_reason == "first_chunk":
                                # push immediately with a shorter initial hold
                                await pager.push_now_with_hold(FIRST_CHUNK_HOLD_MS)
                                first_chunk_sent = True
                            else:
                                # defer to normal pager cadence
                                pass

                            log_pair(en_slice, es, f"V4_{send_reason}{'_ENview' if SHOW_ENGLISH_TESTING else ''}")
                            sent_word_count = len(words)
                            last_sent_ms = now

                    if send_reason == "end_of_turn":
                        await pager.flush_all()
                        # reset for next phrase
                        current_en = ""
                        sent_word_count = 0
                        first_word_time_ms = 0.0
                        first_chunk_sent = False

            except websockets.exceptions.ConnectionClosed:
                print("\n[RECEIVER] Connection closed."); raise
            except Exception:
                print("\n[RECEIVER EXIT]"); traceback.print_exc(); raise

        async def pager_loop():
            while True:
                try:
                    await pager.try_push(force=False)
                except Exception as e:
                    print("pager error:", e)
                await asyncio.sleep(0.08)

        tasks = [asyncio.create_task(sender()),
                 asyncio.create_task(receiver()),
                 asyncio.create_task(pager_loop())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for p in pending: p.cancel()
        for d in done:
            if d.exception(): raise d.exception()

# -------------------- Main loop --------------------
async def main():
    if not AAI_KEY:
        print("Missing ASSEMBLYAI_API_KEY."); return
    await caption_server()
    while True:
        try:
            while not audio_q.empty():
                audio_q.get_nowait(); audio_q.task_done()
            await run_once()
        except Exception as e:
            print(f"Run failed; retrying in 3s… {repr(e)}")
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
