"""Natural romanization (native script -> Latin letters, as people type informally) with an LLM, plus QC.

Language-agnostic: the instruction is a template filled with the language's English `name` from
configs/languages.yaml; the native-punctuation mapping (e.g. Devanagari danda -> '.') is config data.
Any OpenAI-compatible chat API works (DeepSeek, OpenAI, local vLLM): base_url + model + key file.
Never prints prompt text.
"""

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harm_route.text_stats import char_script


def chat(base_url: str, key: str, model: str, system: str, user: str, temperature: float = 0.0,
         timeout: int = 240, retries: int = 4) -> str:
    """One chat completion -> reply text. Retries with backoff on HTTP 429/5xx and network errors."""
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:  # message only, never the request (it has the key)
            msg = f"HTTP {e.code}: {e.read().decode()[:200]}"
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(msg) from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise RuntimeError(f"network: {e}") from None
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def romanize_all(texts: list[str], lang_name: str, cfg: dict, key: str) -> list[str]:
    """texts (native script) -> romanized texts, same order. cfg: base_url, model, system, temperature, workers,
    punct_map. One request per text (one prompt's error can't corrupt another's)."""
    system = cfg["system"].format(lang=lang_name)

    def one(t: str) -> str:
        if not t.strip():
            return t
        out = chat(cfg["base_url"], key, cfg["model"], system, t, cfg.get("temperature", 0.0))
        for a, b in cfg.get("punct_map", {}).items():
            out = out.replace(a, b)
        return out

    with ThreadPoolExecutor(cfg.get("workers", 8)) as ex:  # I/O-bound: threads are fine
        return list(ex.map(one, texts))


_WORD = re.compile(r"[^\W\d_]+")  # runs of letters (any script)


def load_english_words(path: str = "/usr/share/dict/words", min_len: int = 4) -> set[str]:
    """Lower-cased English dictionary words of length >= min_len (short ones collide with real romanized words)."""
    p = Path(path)
    return {w.strip().lower() for w in p.read_text().splitlines() if len(w.strip()) >= min_len} if p.exists() else set()


def qc_item(native: str, roman: str, english_words: set[str], english_src: str = "") -> dict:
    """Checks for one item. flags (any -> listed for a human look, never auto-dropped):
    word_count: whitespace word counts differ (something added, dropped, split or merged)
    non_latin:  letters of a non-Latin script left in the output
    english_words (info only, not a flag): output words in an English dictionary but not in the English source
                (meant to catch 'earth' for artha; too noisy as a flag, see below)."""
    n_nat, n_rom = len(native.split()), len(roman.split())
    non_latin = sorted({c for c in roman if char_script(c) not in (None, "LATIN")})
    src = {w.lower() for w in _WORD.findall(english_src)}
    eng = sorted({w for w in (x.lower() for x in _WORD.findall(roman)) if w in english_words and w not in src})
    # english is INFO only, not a flag: /usr/share/dict/words contains many real Hindi/Nepali words (mein, karen, naam,
    # katha, dharma), so on the first run (2026-11-25) nearly every hit was a false alarm; no 'earth'-type swap seen.
    flags = (["word_count"] if n_nat != n_rom else []) + (["non_latin"] if non_latin else [])
    return {"n_words_native": n_nat, "n_words_roman": n_rom, "non_latin": non_latin, "english_words": eng,
            "flags": flags}
