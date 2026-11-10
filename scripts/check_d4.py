"""Is the loop threshold (text_stats.LOOP_D4) fair for each language? Compare distinct-4 of the
generations with distinct-4 of REAL text in the same language, same tokenizer, same length.
Prints numbers only, never text.

    uv run python scripts/check_d4.py --config configs/experiments/e01_generate.yaml

Real text: the FLORES passages in Belebele (facebook/belebele), concatenated and cut into windows
of --window tokens (= max_new_tokens of the generations: distinct-4 falls with length).
`transform` languages (configs/languages.yaml) apply their transform to the source language's text.
A threshold is fair for a language if real text almost never falls below it (p1 well above LOOP_D4).
"""

import argparse
import json
import statistics as st
from pathlib import Path

import yaml
from datasets import load_dataset

from harm_route.models import load_tokenizer
from harm_route.run_config import load_config
from harm_route.text_stats import LOOP_D4, distinct_ngram_ratio
from harm_route.transforms import TRANSFORMS

ROOT = Path(__file__).resolve().parents[1]
LANGS = yaml.safe_load((ROOT / "configs" / "languages.yaml").read_text())


def real_passages(lang: str) -> list[str]:
    L = LANGS[lang]
    if L["source"] == "transform":
        return [TRANSFORMS[L["transform"]](t) for t in real_passages(L["from"])]
    ds = load_dataset("facebook/belebele", lang, split="test")  # 900 rows, 488 unique passages
    seen = {}
    for r in ds:
        seen.setdefault(r["link"], r["flores_passage"])
    return [seen[k] for k in sorted(seen)]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(int(p * len(xs)), len(xs) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="per-language generation config (e01_generate.yaml)")
    ap.add_argument("--judge", default="qwen3guard-4b", help="judge dir whose rows carry d4 per generation")
    ap.add_argument("--window", type=int, help="tokens per real-text window (default: max_new_tokens)")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.config).read_text())
    tok = load_tokenizer(base["model"])
    W = args.window or base["max_new_tokens"]
    print(f"loop threshold LOOP_D4 = {LOOP_D4}; real-text windows of {W} tokens ({base['model']} tokenizer)\n")
    print(f"{'lang':9s} {'real p1':>7s} {'p5':>5s} {'med':>5s} | {'set':12s} {'condition':13s} "
          f"{'gen med':>7s} {'<LOOP':>6s} {'<real p1':>8s}")
    for lang in base["languages"]:
        cfg = load_config(args.config, lang)
        ids = [i for t in real_passages(lang) for i in tok(t + "\n\n", add_special_tokens=False)["input_ids"]]
        real = [distinct_ngram_ratio(ids[i : i + W]) for i in range(0, len(ids) - W, W)]
        p1 = pct(real, 0.01)
        head = f"{lang:9s} {p1:7.2f} {pct(real, 0.05):5.2f} {st.median(real):5.2f}"
        flag = "  <- threshold may be UNFAIR for this language" if p1 < LOOP_D4 + 0.1 else ""
        for s in cfg["sets"]:
            for c in cfg["conditions"]:
                p = ROOT / cfg["out_dir"] / "judge" / args.judge / f"{s}__{c}.jsonl"
                if not p.exists():
                    continue
                d = [r["d4"] for r in map(json.loads, p.read_text().splitlines()) if r["d4"] is not None]
                print(f"{head} | {s:12s} {c:13s} {st.median(d):7.2f} {sum(x < LOOP_D4 for x in d):3d}/{len(d):<2d} "
                      f"{sum(x < p1 for x in d):5d}/{len(d):<2d}{flag}")
                head, flag = " " * len(head), ""


if __name__ == "__main__":
    main()
