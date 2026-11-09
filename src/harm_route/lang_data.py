"""Final per-language prompt sets for multilingual experiments: MT output + human review, kept parallel.

Files (written by scripts/translate.py and scripts/review_translations.py, all git-ignored):
  {out_dir}/mt/{system}/{lang}.json                  MT / original / transform output, aligned with English by index
  {out_dir}/mt/{system}/{lang}_qc.json               per-item back-translation chrF, script
  {out_dir}/review/{system}/{lang}__{reviewer}.jsonl human decisions {"key": "set|idx", "decision", "text"}
A review belongs to the system whose text was reviewed (a fix edits that system's translation).

Rule per item, per language (a transform language, e.g. srp_Latn, inherits from its source, e.g. srp_Cyrl):
  reviewed       -> the reviewer's decision (fix replaces the text; drop drops it)
  not reviewed   -> keep, unless back-translation chrF < chrf_flag (then drop)
An item dropped in ANY language is dropped in ALL, so comparisons across languages stay paired.
"""

import json
from pathlib import Path

from harm_route.transforms import TRANSFORMS


def load_reviews(out_dir: Path, system: str, lang: str) -> dict[str, dict]:
    """key -> latest decision for this system + language, across reviewer files (later lines win)."""
    out = {}
    for p in sorted((out_dir / "review" / system).glob(f"{lang}__*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def _decide(out_dir: Path, system: str, lang: str, sets: list[str], chrf_flag: float):
    """-> (prompts {set: [text]}, dropped {set|idx}) for an `mt` or `original` language."""
    mt_dir = out_dir / "mt" / system
    raw = json.loads((mt_dir / f"{lang}.json").read_text())
    qc_path = mt_dir / f"{lang}_qc.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    rev = load_reviews(out_dir, system, lang)
    prompts, dropped = {}, set()
    for s in sets:
        prompts[s] = list(raw[s])
        for i in range(len(raw[s])):
            k = f"{s}|{i}"
            if k in rev:
                if rev[k]["decision"] == "drop":
                    dropped.add(k)
                elif rev[k]["decision"] == "fix":
                    prompts[s][i] = rev[k]["text"]
            elif qc and qc[s][i]["chrf"] is not None and qc[s][i]["chrf"] < chrf_flag:
                dropped.add(k)
    return prompts, dropped


def final_prompts(out_dir: Path, system: str, langs: list[str], lang_cfg: dict, sets: list[str],
                  chrf_flag: float):
    """-> ({lang: {set: [text]}} all aligned, {set: [kept idx]}, {lang: n_dropped_by_that_lang})."""
    prompts, dropped, why = {}, set(), {}

    def build(lang):
        if lang in prompts:
            return prompts[lang]
        L = lang_cfg[lang]
        if L["source"] == "transform":
            f = TRANSFORMS[L["transform"]]
            prompts[lang] = {s: [f(t) for t in v] for s, v in build(L["from"]).items()}
            why[lang] = 0  # inherits its source's decisions
        else:
            prompts[lang], d = _decide(out_dir, system, lang, sets, chrf_flag)
            dropped.update(d)
            why[lang] = len(d)
        return prompts[lang]

    for lang in langs:
        build(lang)
    n = {s: len(next(iter(prompts.values()))[s]) for s in sets}
    keep = {s: [i for i in range(n[s]) if f"{s}|{i}" not in dropped] for s in sets}
    return prompts, keep, why
