# asr_ws_final.py
# Mic -> AssemblyAI v3 -> DeepL ES (with glossary + case-insensitive normalizer)
# Broadcast Spanish-only to ws://localhost:8765
# Stabilized partials + silence flusher; logs EN/ES to captions_log.tsv; retries on errors.

import os, asyncio, json, queue, time, traceback, re
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
AAI_URL   = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&encoding=pcm_s16le"

# Glossary settings
GLOSSARY_NAME = os.getenv("DEEPL_GLOSSARY_NAME", "church-spanish-glossary")
GLOSSARY_PATH = os.getenv("DEEPL_GLOSSARY_TSV", "glossary.tsv")
SRC_LANG, TGT_LANG = "EN", "ES"

# Logging
LOG_PATH = os.getenv("CAPTION_LOG", "captions_log.tsv")

# Latency knobs
RATE, CHUNK_MS = 16000, 100          # try CHUNK_MS = 50 if you want a bit snappier
BLOCK = int(RATE * CHUNK_MS / 1000)
STABLE_MS  = 150                      # translate partial if unchanged >= this long
MIN_GAP_MS = 250                      # throttle DeepL calls
FLUSH_MS   = 700                      # force translate if paused >= this long

audio_q = queue.Queue()
clients = set()

# -------------------- Utilities --------------------
def log_pair(en_text, es_text, kind):
    ts = datetime.now().isoformat(timespec="seconds")
    en_clean = (en_text or "").replace("\t", " ")
    es_clean = (es_text or "").replace("\t", " ")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{ts}\t{kind}\t{en_clean}\t{es_clean}\n")

def mic_cb(indata, frames, t, status):
    if status: print("Audio status:", status)
    audio_q.put((indata[:,0]*32767).astype(np.int16).tobytes())

async def caption_server():
    async def handler(ws):
        print("Caption client connected")
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.remove(ws)
            print("Caption client disconnected")
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("Caption WS at ws://127.0.0.1:8765")
    return server

async def push_caption(text):
    if not clients: return
    payload = json.dumps({"text": text})
    await asyncio.gather(*[c.send(payload) for c in list(clients)], return_exceptions=True)

# -------------------- DeepL helpers --------------------
async def translate_es(session, text_en, glossary_id=None):
    if not text_en or not text_en.strip():
        return ""
    data = {"text": text_en, "target_lang": TGT_LANG}
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_KEY}"}
    # When using a glossary, DeepL requires source_lang as well
    if glossary_id:
        data["source_lang"] = SRC_LANG
        data["glossary_id"] = glossary_id
    async with session.post(
        DEEPL_URL, data=data, headers=headers,
        timeout=aiohttp.ClientTimeout(total=10),
    ) as r:
        if r.status == 403:
            raise RuntimeError("DeepL 403 (check Free vs Pro endpoint and your key)")
        r.raise_for_status()
        js = await r.json()
        return js["translations"][0]["text"]

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
            print(f"(glossary.tsv line {i} invalid, skipping: {line!r})")
            continue
        left, right = s.split("\t", 1)
        left, right = left.strip(), right.strip()
        if not left or not right:
            print(f"(glossary.tsv line {i} empty term, skipping)")
            continue
        # collapse extra tabs on right
        if "\t" in right: right = right.split("\t",1)[0].strip()
        cleaned.append(f"{left}\t{right}")
    if not cleaned:
        print("(glossary.tsv has no valid rows; continuing without glossary.)")
        return None
    return "\n".join(cleaned)

async def ensure_glossary(session) -> str | None:
    """Create/update a DeepL glossary from glossary.tsv. Return glossary_id or None. Safe if unsupported."""
    entries_tsv = sanitize_tsv_to_entries(GLOSSARY_PATH)
    if not entries_tsv:
        return None

    base = "https://api-free.deepl.com" if "api-free" in DEEPL_URL else "https://api.deepl.com"
    list_url = f"{base}/v2/glossaries"
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_KEY}"}

    # 1) Try to find existing glossary by name/langs
    existing_id = None
    try:
        async with session.get(list_url, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                js = await r.json()
                for g in js.get("glossaries", []):
                    if g.get("name")==GLOSSARY_NAME and g.get("source_lang").upper()==SRC_LANG and g.get("target_lang").upper()==TGT_LANG:
                        existing_id = g.get("glossary_id")
                        break
    except Exception as e:
        print("(Glossary list failed; will still attempt create.)", e)

    # 2) If exists, update its entries
    if existing_id:
        put_url = f"{list_url}/{existing_id}/entries"
        async with session.put(
            put_url,
            headers=headers,
            data={"entries": entries_tsv, "entries_format": "tsv"},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as rr:
            if rr.status in (200, 204):
                print(f"Updated glossary '{GLOSSARY_NAME}'.")
                return existing_id
            else:
                txt = await rr.text()
                print(f"(Update glossary failed: {rr.status} {txt})")
                return existing_id  # still usable as-is

    # 3) Otherwise, create new
    async with session.post(
        list_url,
        headers=headers,
        data={
            "name": GLOSSARY_NAME,
            "source_lang": SRC_LANG,
            "target_lang": TGT_LANG,
            "entries": entries_tsv,
            "entries_format": "tsv",
        },
        timeout=aiohttp.ClientTimeout(total=10)
    ) as r:
        if r.status in (200, 201):
            gid = (await r.json()).get("glossary_id")
            print(f"Created glossary '{GLOSSARY_NAME}':", gid)
            return gid
        elif r.status == 456:
            print("(Too many glossaries; using existing one if available.)")
            return existing_id
        else:
            txt = await r.text()
            print(f"(Glossary create failed: {r.status} {txt})")
            return existing_id

# -------------------- Case-insensitive normalizer --------------------
def load_glossary_normalizer(path: str) -> dict:
    """
    Reads glossary.tsv and returns {lower_src: canonical_src}.
    Use the exact casing you want in the TSV; we'll rewrite ASR text to that.
    """
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
                terms[key] = src  # first occurrence defines canonical casing
    return terms

def build_normalizer_regex(terms_ci: dict):
    if not terms_ci: return None
    parts = [re.escape(k) for k in sorted(terms_ci.keys(), key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)

def normalize_en_for_glossary(text: str, terms_ci: dict, rx) -> str:
    if not text or not rx: return text
    return rx.sub(lambda m: terms_ci.get(m.group(0).lower(), m.group(0)), text)

# -------------------- One run cycle --------------------
async def run_once():
    # Start mic and keep reference to avoid GC
    try:
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                blocksize=BLOCK, callback=mic_cb)
        stream.start()
        print("Mic started.")
    except Exception as e:
        print("Mic error:", repr(e))
        await asyncio.sleep(3); return

    print("Connecting to AssemblyAI…")
    async with aiohttp.ClientSession() as http, websockets.connect(
        AAI_URL, additional_headers=[("Authorization", AAI_KEY)],
        ping_interval=5, ping_timeout=20, max_size=10_000_000
    ) as ws:
        print("Connected to AssemblyAI.")

        # One-time glossary (safe)
        try:
            glossary_id = await ensure_glossary(http)
            if glossary_id: print("Glossary ready:", glossary_id)
        except Exception as e:
            print("(Glossary step failed; continuing without it.)", e)
            glossary_id = None

        # Build case-insensitive normalizer so the glossary always matches
        terms_ci = load_glossary_normalizer(GLOSSARY_PATH)
        terms_rx = build_normalizer_regex(terms_ci)

        current_en, last_en = "", ""
        last_change, last_sent = 0.0, 0.0

        async def sender():
            try:
                while True:
                    pcm = await asyncio.get_event_loop().run_in_executor(None, audio_q.get)
                    await ws.send(pcm)  # raw bytes
            except Exception:
                print("\n[SENDER EXIT]")
                traceback.print_exc(); raise

        async def receiver():
            nonlocal current_en, last_en, last_change, last_sent
            try:
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    t = data.get("transcript","")
                    if t:
                        current_en = t
                        # print("\rEN:", t[:100], end="", flush=True)  # optional
                        if t != last_en:
                            last_change = time.time()*1000

                    now = time.time()*1000
                    # stabilized partial
                    if current_en and (now-last_change>=STABLE_MS) and current_en!=last_en and (now-last_sent>=MIN_GAP_MS):
                        en_to_send = normalize_en_for_glossary(current_en, terms_ci, terms_rx)
                        es = await translate_es(http, en_to_send, glossary_id)
                        await push_caption(es)
                        log_pair(current_en, es, "partial")
                        last_en, last_sent = current_en, now

                    # final on turn end
                    if data.get("end_of_turn"):
                        if current_en.strip():
                            en_to_send = normalize_en_for_glossary(current_en, terms_ci, terms_rx)
                            es = await translate_es(http, en_to_send, glossary_id)
                            await push_caption(es)
                            log_pair(current_en, es, "final")
                        current_en, last_en = "", ""
            except Exception:
                print("\n[RECEIVER EXIT]")
                traceback.print_exc(); raise

        async def silence_flusher():
            nonlocal current_en, last_en, last_change, last_sent
            while True:
                now = time.time()*1000
                if current_en and (now-last_change>=FLUSH_MS) and current_en!=last_en and (now-last_sent>=MIN_GAP_MS):
                    en_to_send = normalize_en_for_glossary(current_en, load_glossary_normalizer(GLOSSARY_PATH), build_normalizer_regex(load_glossary_normalizer(GLOSSARY_PATH)))
                    # (rebuild in case you edit TSV while running; remove if unnecessary)
                    es = await translate_es(http, en_to_send, glossary_id)
                    await push_caption(es)
                    log_pair(current_en, es, "final")
                    last_en, last_sent = current_en, now
                await asyncio.sleep(0.1)

        tasks = [asyncio.create_task(sender()),
                 asyncio.create_task(receiver()),
                 asyncio.create_task(silence_flusher())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for p in pending: p.cancel()
        for d in done:
            if d.exception(): raise d.exception()

# -------------------- Main loop --------------------
async def main():
    if not AAI_KEY or not DEEPL_KEY:
        print("Missing API keys (ASSEMBLYAI_API_KEY / DEEPL_AUTH_KEY).")
        return
    await caption_server()
    while True:
        try:
            await run_once()
        except Exception as e:
            print("Run failed; retrying in 3s…", repr(e))
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
