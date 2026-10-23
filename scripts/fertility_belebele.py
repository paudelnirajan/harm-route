"""Tokenizer premium: tokens per FLORES passage relative to English, per language x tokenizer.

Uses the 488 parallel FLORES passages in Belebele. srp_Latn* is a deterministic Cyrillic->Latin
conversion (FLORES+ has no srp_Latn). Run: uv run python scripts/fertility_belebele.py
First draft (2026-10-21); to be refactored into src/harm_route/ for E05.
"""
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

LANGS = ["eng_Latn","ita_Latn","vie_Latn","tur_Latn","jav_Latn","swh_Latn","yor_Latn","zho_Hans","jpn_Jpan","kor_Hang",
         "rus_Cyrl","srp_Cyrl","kaz_Cyrl","uzn_Latn","ell_Grek","arb_Arab","arb_Latn","pes_Arab","urd_Arab","urd_Latn",
         "hin_Deva","hin_Latn","npi_Deva","npi_Latn","ben_Beng","ben_Latn","sin_Sinh","sin_Latn","tha_Thai","amh_Ethi","tel_Telu","tam_Taml"]
TOKS = {"qwen3":"Qwen/Qwen3-0.6B","gemma3":"unsloth/gemma-3-1b-it","llama3.2":"unsloth/Llama-3.2-1B-Instruct"}

# Serbian Cyrillic -> Latin (deterministic, official Gaj alphabet)
CYR2LAT = dict(zip("абвгдђежзијклљмнњопрстћуфхцчџш", ["a","b","v","g","d","đ","e","ž","z","i","j","k","l","lj","m","n","nj","o","p","r","s","t","ć","u","f","h","c","č","dž","š"]))
def sr_lat(s):
    out=[]
    for ch in s:
        lo=ch.lower()
        if lo in CYR2LAT:
            t=CYR2LAT[lo]; out.append(t.capitalize() if ch!=lo else t)
        else: out.append(ch)
    return "".join(out)

texts={}
for l in LANGS:
    ds=load_dataset("facebook/belebele", l, split="test")
    seen={}
    for r in ds:
        seen.setdefault(r["link"], r["flores_passage"])
    keys=sorted(seen)
    texts[l]=[seen[k] for k in keys]
texts["srp_Latn*"]=[sr_lat(t) for t in texts["srp_Cyrl"]]
assert all(len(v)==len(texts["eng_Latn"]) for v in texts.values())
tok={n:AutoTokenizer.from_pretrained(p) for n,p in TOKS.items()}
cnt={n:{l:sum(len(t(x, add_special_tokens=False)["input_ids"]) for x in v) for l,v in texts.items()} for n,t in tok.items()}
print("n_passages", len(texts["eng_Latn"]))
print(f"{'lang':12s}" + "".join(f"{n:>20s}" for n in TOKS))
for l in texts:
    print(f"{l:12s}" + "".join(f"{cnt[n][l]/cnt[n]['eng_Latn']:>12.2f} ({cnt[n][l]//len(texts[l]):>4d}tk)" for n in TOKS))
