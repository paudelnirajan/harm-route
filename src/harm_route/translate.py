"""Machine translation backends: NLLB-200 (local, FLORES+ tags) and Google Cloud Translation (API key).

NLLB:

How the output language is chosen: the tokenizer puts the SOURCE language tag in front of the input
(`src_lang`), and generation forces the decoder's first token to be the TARGET tag
(`forced_bos_token_id`). Everything the decoder writes after that is conditioned on that one token.

Multi-line prompts are translated line by line: sentence-level MT models tend to drop or merge lines
when given several at once.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_mt(hf_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_id, dtype=dtype).to(device).eval()
    model.generation_config.max_length = None  # NLLB ships max_length=200; we pass max_new_tokens instead
    return model, tok


def lang_token_id(tok, code: str) -> int:
    i = tok.convert_tokens_to_ids(code)
    if i is None or i == tok.unk_token_id:
        raise KeyError(f"{code!r} is not a language tag of this MT model")
    return i


def translate_lines(model, tok, lines: list[str], src: str, tgt: str, num_beams: int = 4,
                    max_new_tokens: int = 256) -> list[str]:
    """One padded batch of single lines -> translations (same order)."""
    tok.src_lang = src  # the tokenizer now prepends the src tag: [src_tag, ...tokens, </s>]
    enc = tok(lines, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = enc.to(model.device)  # input_ids, attention_mask: [B, S]; right padding is fine for an encoder
    with torch.inference_mode():
        out = model.generate(**enc, forced_bos_token_id=lang_token_id(tok, tgt), num_beams=num_beams,
                             max_new_tokens=max_new_tokens)  # [B, T]: </s>, tgt_tag, translation..., pads
    return tok.batch_decode(out, skip_special_tokens=True)


def translate(model, tok, texts: list[str], src: str, tgt: str, batch_size: int = 16, **kw) -> list[str]:
    """Texts (may contain newlines) -> translations. Splits into lines, batches, re-joins per text."""
    lines, owner = [], []  # flat list of non-empty lines and which text each belongs to
    parts = [t.split("\n") for t in texts]
    for i, ps in enumerate(parts):
        for j, p in enumerate(ps):
            if p.strip():
                lines.append(p.strip())
                owner.append((i, j))
    done = []
    for b in range(0, len(lines), batch_size):
        done += translate_lines(model, tok, lines[b : b + batch_size], src, tgt, **kw)
    for (i, j), t in zip(owner, done):
        parts[i][j] = t  # empty lines stay empty -> paragraph layout is kept
    return ["\n".join(ps).strip() for ps in parts]


# ---- Google Cloud Translation (v2 REST, API key) ----
GOOGLE_URL = "https://translation.googleapis.com/language/translate/v2"


def read_key(path: str) -> str:
    """API key from a file outside the repo (never printed, never logged)."""
    return Path(path).expanduser().read_text().strip()


def google_translate(texts: list[str], src: str, tgt: str, key: str, batch_size: int = 64) -> list[str]:
    """Texts -> translations via Google. src/tgt are Google codes ('en', 'ne', 'sr'). format=text keeps
    newlines and stops HTML escaping (&#39;). Whole prompts are sent (no line splitting): Google handles
    multi-sentence input and context helps word choice."""
    out = []
    for b in range(0, len(texts), batch_size):
        body = json.dumps({"q": texts[b : b + batch_size], "source": src, "target": tgt,
                           "format": "text"}).encode()
        req = urllib.request.Request(f"{GOOGLE_URL}?key={key}", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:  # message only: the URL contains the key
            raise RuntimeError(f"Google Translate HTTP {e.code}: {e.read().decode()[:300]}") from None
        out += [t["translatedText"] for t in data["data"]["translations"]]
    return out
