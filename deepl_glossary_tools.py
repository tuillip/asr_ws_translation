import os, argparse, requests, sys
from dotenv import load_dotenv

load_dotenv()
DEEPL_KEY = os.getenv("DEEPL_AUTH_KEY")
BASE = "https://api-free.deepl.com" if os.getenv("DEEPL_BASE","free").lower()=="free" else "https://api.deepl.com"
HDRS = {"Authorization": f"DeepL-Auth-Key {DEEPL_KEY}"}

def err(s): print(s, file=sys.stderr)

def list_glossaries():
    r = requests.get(f"{BASE}/v2/glossaries", headers=HDRS, timeout=15)
    print("LIST", r.status_code, r.text)

def show_entries(gid: str):
    r = requests.get(f"{BASE}/v2/glossaries/{gid}", headers=HDRS, timeout=15)
    print("INFO", r.status_code, r.text)
    r2 = requests.get(f"{BASE}/v2/glossaries/{gid}/entries", headers=HDRS, timeout=15)
    print("ENTRIES", r2.status_code)
    print(r2.text)

def sanitize_tsv(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = []
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"): continue
            if "\t" not in s or s.startswith("\t"):
                err(f"skip line {i}: needs EN<TAB>ES : {line!r}"); continue
            left, right = s.split("\t", 1)
            left, right = left.strip(), right.strip()
            if not left or not right:
                err(f"skip line {i}: empty term"); continue
            if "\t" in right: right = right.split("\t",1)[0].strip()
            rows.append(f"{left}\t{right}")
        if not rows:
            raise SystemExit("glossary.tsv has no valid rows")
        return "\n".join(rows)

def find_by_name(name: str, src: str, tgt: str):
    r = requests.get(f"{BASE}/v2/glossaries", headers=HDRS, timeout=15)
    r.raise_for_status()
    for g in r.json().get("glossaries", []):
        if g.get("name")==name and g.get("source_lang","").upper()==src and g.get("target_lang","").upper()==tgt:
            return g.get("glossary_id")
    return None

def delete(gid: str):
    r = requests.delete(f"{BASE}/v2/glossaries/{gid}", headers=HDRS, timeout=15)
    print("DELETE", gid, r.status_code, r.text)

def replace(name: str, src: str, tgt: str, tsv: str):
    """Replace entries by delete+create. Works around 401/456 update issues."""
    entries = sanitize_tsv(tsv)
    gid = find_by_name(name, src, tgt)
    if gid:
        print("Deleting existing:", gid)
        delete(gid)
    print("Creating new glossary…")
    data = {
        "name": name,
        "source_lang": src,
        "target_lang": tgt,
        "entries": entries,
        "entries_format": "tsv",
    }
    r = requests.post(f"{BASE}/v2/glossaries", headers=HDRS, data=data, timeout=15)
    print("CREATE", r.status_code, r.text)

if __name__ == "__main__":
    if not DEEPL_KEY:
        raise SystemExit("DEEPL_AUTH_KEY missing. Check your .env.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="GLOSSARY_ID")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--name", default="church-spanish-glossary")
    ap.add_argument("--src", default="EN")
    ap.add_argument("--tgt", default="ES")
    ap.add_argument("--file", default="glossary.tsv")
    args = ap.parse_args()

    if args.list: list_glossaries()
    elif args.show: show_entries(args.show)
    elif args.replace: replace(args.name, args.src, args.tgt, args.file)
    else: ap.print_help()
