"""E1: cache residual-stream activations at the h and r positions, every layer, for train + val.

    uv run python scripts/cache_e1_acts.py --config configs/experiments/e01_acts.yaml            # all languages
    uv run python scripts/cache_e1_acts.py --config ... --lang npi_Deva --limit 4                # smoke

Writes {out_dir}/{lang}_acts.pt:
    {"meta": {...}, "<set>": {"inst": [N, L, d], "last": [N, L, d]}}  for set in
    harmful_train, harmless_train, harmful_val, harmless_val. float32 (same as notebook 05).
    inst = resid_post at t_inst (h position); last = resid_post at the last post-instruction token (r position).
Existing files are skipped (use --overwrite). Prints shapes only, never prompt text.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from harm_route.collect import collect_resid_batched
from harm_route.models import load_bridge, load_tokenizer, model_config, pad_token_id
from harm_route.prompts import find_positions
from harm_route.run_config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lang", nargs="*", help="default: all languages in the config")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, help="first N prompts per set (smoke test; output gets a _smoke suffix)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    langs = args.lang or yaml.safe_load(Path(args.config).read_text())["languages"]
    base = load_config(args.config, langs[0])
    name = base["model"]
    tok = load_tokenizer(name)
    kw = model_config(name)["template_kwargs"]
    pad_id = pad_token_id(tok)
    bridge = None  # loaded lazily, so a run where every file exists does not load the model

    for lang in langs:
        cfg = load_config(args.config, lang)
        out = ROOT / cfg["out_dir"] / f"{lang}_acts{'_smoke' if args.limit else ''}.pt"
        if out.exists() and not args.overwrite:
            print(f"{lang}: exists, skip ({out.relative_to(ROOT)})")
            continue
        if bridge is None:
            bridge = load_bridge(name, device=args.device)
        res = {"meta": {"model": name, "lang": lang, "device": args.device,
                        "positions": {"inst": "t_inst (h)", "last": "last post-inst (r)"}}}
        for key in ("train_splits", "splits"):
            data = json.loads((ROOT / cfg[key]).read_text())
            for st, prompts in data.items():
                if st == "meta":
                    continue
                prompts = prompts[: args.limit]
                ps = [find_positions(tok, p, kw) for p in prompts]
                inst, post = collect_resid_batched(bridge, ps, pad_id, batch_size=cfg["batch_size"])
                res[st] = {"inst": inst.float().cpu().clone(),    # [N, L, d]; clone: save only this, not a parent storage
                           "last": post[:, -1].float().cpu().clone()}  # [N, P, L, d] -> [N, L, d]
                print(f"{lang} {st:15s} inst {tuple(res[st]['inst'].shape)} last {tuple(res[st]['last'].shape)}")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, out)
        print(f"{lang}: saved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
