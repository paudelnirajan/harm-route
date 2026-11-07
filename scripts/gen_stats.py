"""Counts-only quality stats for a generation run (safe to paste: prints numbers, never text).

    uv run python scripts/gen_stats.py --config configs/experiments/e00_generate.yaml

Per (set, condition): mean length, distinct-4-gram ratio over model tokens (loops -> low), how many
outputs look like loops, repeated lines, empty outputs, and the dominant output script.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import yaml

from harm_route.models import load_tokenizer
from harm_route.text_stats import distinct_ngram_ratio, dominant_script, max_line_repeat

ROOT = Path(__file__).resolve().parents[1]
LOOP = 0.5  # distinct-4 below this = degenerate repetition (normal prose is ~0.9+)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    name = cfg["model"]
    tok = load_tokenizer(name)
    out_dir = ROOT / cfg["out_dir"].format(model=name)

    print(f"{'set':13s} {'condition':16s} {'n':>3s} {'len':>6s} {'d4 mean':>7s} {'d4 med':>6s} "
          f"{'loops':>5s} {'rep>=3':>6s} {'empty':>5s}  scripts")
    for s in cfg["sets"]:
        for c in cfg["conditions"]:
            path = out_dir / f"{s}__{c}.jsonl"
            if not path.exists():
                print(f"{s:13s} {c:16s} missing")
                continue
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            d4 = [distinct_ngram_ratio(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
            d4v = [x for x in d4 if x is not None]
            reps = [max_line_repeat(r["text"]) for r in rows]
            scripts = Counter(dominant_script(r["text"])[0] or "none" for r in rows)
            print(f"{s:13s} {c:16s} {len(rows):3d} {mean(r['n_tokens'] for r in rows):6.1f} "
                  f"{mean(d4v) if d4v else float('nan'):7.2f} {median(d4v) if d4v else float('nan'):6.2f} "
                  f"{sum(x < LOOP for x in d4v):5d} {sum(r >= 3 for r in reps):6d} "
                  f"{sum(not r['text'].strip() for r in rows):5d}  "
                  + " ".join(f"{k}:{v}" for k, v in scripts.most_common()))


if __name__ == "__main__":
    main()
