"""Judge every generation file of a run; prints counts only, never text.

    uv run python scripts/judge.py --config configs/experiments/e00_generate.yaml --judge qwen3guard-4b
    uv run python scripts/judge.py --config ... --judge qwen3guard-0.6b --limit 2 --device cpu   # smoke (no files)

Reads  {out_dir}/{set}__{cond}.jsonl  (generations) + the prompts from the splits file (row idx -> prompt).
Writes {out_dir}/judge/{judge}/{set}__{cond}.jsonl, one row per output with RAW fields only:
  idx, verdict (judge text: labels, no harmful content), safety, refusal, categories,
  n_tokens, d4, script, script_share.
Classes are derived at analysis time (harm_route.judge.to_class); the summary printed here is a preview.
Existing label files are skipped (resumable).
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

from harm_route.judge import CLASSES, is_garbled, judge_batch, load_judge, parse_verdict, to_class
from harm_route.models import load_tokenizer
from harm_route.text_stats import distinct_ngram_ratio, dominant_script

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the generation config (sets, conditions, out_dir)")
    ap.add_argument("--judge", required=True, help="key in configs/judges.yaml")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N rows per file (smoke test; writes nothing)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    name = cfg["model"]
    gen_dir = ROOT / cfg["out_dir"].format(model=name)
    lab_dir = gen_dir / "judge" / args.judge
    lab_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads((ROOT / cfg["splits"].format(model=name)).read_text())
    expected_script = cfg.get("expected_script")  # E0: LATIN (English); None = don't check

    gtok = load_tokenizer(name)  # generator's tokenizer, for distinct-4 over model tokens
    dtype = None if args.device != "cpu" else torch.float32
    judge, jtok, jcfg = load_judge(args.judge, device=args.device, dtype=dtype)
    print(f"judge {args.judge} ({jcfg['hf_id']}) on {args.device}; generations {gen_dir.relative_to(ROOT)}")

    print(f"{'set':13s} {'condition':16s} {'n':>3s}  {'ref Y/N/?':>10s}  {'safe S/U/C/?':>13s}  classes")
    for s in cfg["sets"]:
        prompts = splits[s]
        for c in cfg["conditions"]:
            src, dst = gen_dir / f"{s}__{c}.jsonl", lab_dir / f"{s}__{c}.jsonl"
            if not src.exists():
                print(f"{s:13s} {c:16s} missing generations")
                continue
            if dst.exists() and not args.limit:
                print(f"skip {dst.name} (exists)")
                continue
            rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()][: args.limit]
            t = time.time()
            verdicts = []
            for i in range(0, len(rows), args.batch_size):
                chunk = rows[i : i + args.batch_size]
                verdicts += judge_batch(judge, jtok, [(prompts[r["idx"]], r["text"]) for r in chunk])

            out = []
            for r, v in zip(rows, verdicts):
                script, share = dominant_script(r["text"])
                out.append({"idx": r["idx"], "verdict": v, **parse_verdict(v, jcfg["format"]),
                            "n_tokens": r["n_tokens"],
                            "d4": distinct_ngram_ratio(gtok(r["text"], add_special_tokens=False)["input_ids"]),
                            "script": script, "script_share": round(share, 3)})
            if not args.limit:
                with dst.open("w") as f:
                    for o in out:
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")

            ref = Counter(o["refusal"] for o in out)
            saf = Counter(o["safety"] for o in out)
            cls = Counter(to_class(o, is_garbled(o["n_tokens"], o["d4"], o["script"], expected_script)) for o in out)
            print(f"{s:13s} {c:16s} {len(out):3d}  "
                  f"{ref['Yes']:3d}/{ref['No']:3d}/{ref[None]:2d}  "
                  f"{saf['Safe']:3d}/{saf['Unsafe']:3d}/{saf['Controversial']:2d}/{saf[None]:2d}  "
                  + " ".join(f"{k}:{cls[k]}" for k in CLASSES if cls[k])
                  + f"  {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
