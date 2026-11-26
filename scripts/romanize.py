"""Make romanized languages (configs/languages.yaml `source: romanize`) from their native-script source.
Prints counts/flags only, never text.

    uv run python scripts/romanize.py --config configs/experiments/e01_romanize.yaml
    uv run python scripts/romanize.py --config ... --limit 3      # smoke: 3 prompts per set, writes nothing

Input = the source language's FINAL prompts (MT + human fixes, lang_data.final_prompts), full length so index i
is still English item i. Output (git-ignored), per prompt set (val, train):
  {out_dir}/mt/romanized/{lang}.json      {"meta", set: [romanized prompt_i, ...]}
  {out_dir}/mt/romanized/{lang}_qc.json   {set: [{idx, flags, ..., "back": native text, "chrf": None}]}
Same layout as MT output, so review_server.py (--system romanized) and export_lang_splits.py work unchanged.
Flagged items are never dropped automatically: they are listed for a human look. Existing files are skipped.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from harm_route.lang_data import final_prompts
from harm_route.romanize import load_english_words, qc_item, romanize_all
from harm_route.text_stats import dominant_script
from harm_route.translate import read_key

ROOT = Path(__file__).resolve().parents[1]
LANGS = yaml.safe_load((ROOT / "configs" / "languages.yaml").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="first N prompts per set; smoke test, writes nothing")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    llm, key = cfg["llm"], read_key(cfg["llm"]["key_file"])
    english_words = load_english_words()

    for tc_path in cfg["prompt_sets"]:  # translation configs: where the native prompts live
        tc = yaml.safe_load((ROOT / tc_path).read_text())
        out_dir = ROOT / tc["out_dir"].format(model=tc["model"])
        eng = json.loads((ROOT / tc["splits"].format(model=tc["model"])).read_text())
        rom_dir = out_dir / "mt" / "romanized"
        rom_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {tc_path}")
        for lang in cfg["languages"]:
            L = LANGS[lang]
            assert L["source"] == "romanize", f"{lang} is not a romanize language"
            path = rom_dir / f"{lang}.json"
            if path.exists() and not args.limit:
                print(f"{lang:9s} exists, skipped")
                continue
            src, _, _ = final_prompts(out_dir, tc["primary"], [L["from"]], LANGS, tc["sets"], tc["qc"]["chrf_flag"])
            native = {s: v[: args.limit] for s, v in src[L["from"]].items()}  # set -> [text], full length
            roman = {s: romanize_all(v, LANGS[L["from"]]["name"], llm, key) for s, v in native.items()}
            qc = {}
            for s in native:
                qc[s] = []
                for i, (n, r) in enumerate(zip(native[s], roman[s])):
                    sc, share = dominant_script(r)
                    qc[s].append({"idx": i, **qc_item(n, r, english_words, eng[s][i]), "script": sc,
                                  "script_share": share, "script_ok": sc == L["script"], "chrf": None, "back": n})
                fl = Counter(f for row in qc[s] for f in row["flags"])
                n_flag = sum(bool(row["flags"]) for row in qc[s])
                print(f"{lang:9s} {s:15s} n={len(qc[s]):3d}  flagged={n_flag:3d}  {dict(fl)}  "
                      f"script_ok={sum(r['script_ok'] for r in qc[s])}/{len(qc[s])}")
            if not args.limit:
                meta = {"source": "romanize", "from": L["from"], "lang": lang, "model_splits": tc["model"],
                        **{k: v for k, v in llm.items() if k not in ("key_file", "system")}, "system_prompt": llm["system"]}
                path.write_text(json.dumps({"meta": meta, **roman}, ensure_ascii=False, indent=1))
                (rom_dir / f"{lang}_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1))
    print("smoke run: nothing written" if args.limit else "done")


if __name__ == "__main__":
    main()
