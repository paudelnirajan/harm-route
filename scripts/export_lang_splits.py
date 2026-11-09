"""E1: write the final reviewed prompts of each language as E0-style splits files, so the E0 scripts
(generate_e0.py, judge.py, gen_stats.py) run unchanged with --lang. Prints counts only, never text.

    uv run python scripts/export_lang_splits.py --config configs/experiments/e01_translate.yaml

Writes (git-ignored) {out_dir}/splits/{lang}.json = {"meta": ..., "harmful_val": [...], "harmless_val": [...]}.
Only items kept in ALL languages are written (lang_data.final_prompts), in the same order everywhere,
so row i is the same request in every language; meta.eng_idx maps row i back to the English split index.
"""

import argparse
import json
from pathlib import Path

import yaml

from harm_route.lang_data import final_prompts

ROOT = Path(__file__).resolve().parents[1]
LANGS = yaml.safe_load((ROOT / "configs" / "languages.yaml").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the translation config (e01_translate.yaml)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = ROOT / cfg["out_dir"].format(model=cfg["model"])
    system, sets = cfg["primary"], cfg["sets"]
    prompts, keep, why = final_prompts(out_dir, system, cfg["languages"], LANGS, sets, cfg["qc"]["chrf_flag"])
    (out_dir / "splits").mkdir(exist_ok=True)
    for lang in cfg["languages"]:
        data = {s: [prompts[lang][s][i] for i in keep[s]] for s in sets}
        meta = {"lang": lang, "model_splits": cfg["model"], "system": system, "eng_idx": keep,
                "source": LANGS[lang]["source"]}
        (out_dir / "splits" / f"{lang}.json").write_text(
            json.dumps({"meta": meta, **data}, ensure_ascii=False, indent=1))
        print(f"{lang:9s} " + "  ".join(f"{s} {len(v)}" for s, v in data.items()))
    print(f"dropped because of each language: {why}\nwritten to {(out_dir / 'splits').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
