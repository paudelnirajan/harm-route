"""Make the per-language prompt files for E1 from the English E0 splits; prints counts/scores only, never text.

    uv run python scripts/translate.py --config configs/experiments/e01_translate.yaml            # primary system
    uv run python scripts/translate.py --config ... --system nllb
    uv run python scripts/translate.py --config ... --limit 3     # smoke: 3 prompts per set, writes nothing

For each language (configs/languages.yaml, `source`):
  original  -> copy the English prompts
  mt        -> translate with the chosen system (NLLB or Google), then back-translate to English for QC
  transform -> apply a deterministic transform to another language's prompts (e.g. Serbian Cyrl -> Latn)

Writes (git-ignored) {out_dir}/mt/{system}/{lang}.json = {"meta": ..., set: [prompt_i, ...]} aligned with English by index,
and {lang}_qc.json next to it with per-item back-translation, chrF, script. Existing files are skipped.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import torch
import yaml
from sacrebleu.metrics import CHRF

from harm_route.text_stats import dominant_script
from harm_route.transforms import TRANSFORMS
from harm_route.translate import google_translate, load_mt, read_key, translate

ROOT = Path(__file__).resolve().parents[1]
LANGS = yaml.safe_load((ROOT / "configs" / "languages.yaml").read_text())
SRC = "eng_Latn"


def qc(src: list[str], out: list[str], back: list[str] | None, script: str) -> list[dict]:
    """Per item: chrF(original English, back-translation), length ratio, script of the output, untranslated."""
    chrf = CHRF()
    rows = []
    for i, (s, o) in enumerate(zip(src, out)):
        sc, share = dominant_script(o)
        rows.append({"idx": i, "chrf": chrf.sentence_score(back[i], [s]).score if back else None,
                     "len_ratio": len(o) / max(len(s), 1), "script": sc, "script_share": share,
                     "script_ok": sc == script, "untranslated": back is not None and o.strip() == s.strip(),
                     "back": back[i] if back else None})
    return rows


def report(lang: str, set_name: str, rows: list[dict], flag: float):
    ch = [r["chrf"] for r in rows if r["chrf"] is not None]
    bad = [r["idx"] for r in rows if r["chrf"] is not None and r["chrf"] < flag]
    chs = f"{mean(ch):5.1f} {median(ch):5.1f} {min(ch):5.1f}" if ch else f"{'-':>5s} {'-':>5s} {'-':>5s}"
    print(f"{lang:9s} {set_name:13s} {len(rows):3d} {chs} {len(bad):4d} "
          f"{sum(r['script_ok'] for r in rows):3d}/{len(rows):<3d} {sum(r['untranslated'] for r in rows):5d} "
          f"{median(r['len_ratio'] for r in rows):5.2f}")
    return [f"{lang}|{set_name}|{i}" for i in bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--system", help="key of mt_systems in the config (default: primary)")
    ap.add_argument("--limit", type=int, help="first N prompts per set; smoke test, writes nothing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = json.loads((ROOT / cfg["splits"].format(model=cfg["model"])).read_text())
    eng = {s: splits[s][: args.limit] for s in cfg["sets"]}  # set -> [prompt]
    system = args.system or cfg["primary"]
    out_dir = ROOT / cfg["out_dir"].format(model=cfg["model"]) / "mt" / system
    out_dir.mkdir(parents=True, exist_ok=True)
    mt_cfg, flag = cfg["mt_systems"][system], cfg["qc"]["chrf_flag"]
    mt = None  # loaded on first use: a callable (texts, src_lang, tgt_lang) -> texts

    def load_backend():
        if mt_cfg["kind"] == "nllb":  # FLORES codes are the NLLB language tags
            print(f"loading {mt_cfg['hf_id']} on {args.device}")
            model, tok = load_mt(mt_cfg["hf_id"], args.device)
            kw = {k: mt_cfg[k] for k in ("batch_size", "num_beams", "max_new_tokens")}
            return lambda texts, a, b: translate(model, tok, texts, a, b, **kw)
        if mt_cfg["kind"] == "google":
            key = read_key(mt_cfg["key_file"])
            return lambda texts, a, b: google_translate(texts, LANGS[a]["google_code"], LANGS[b]["google_code"], key)
        raise ValueError(f"unknown MT kind {mt_cfg['kind']!r}")
    made: dict[str, dict] = {}  # lang -> {set: [prompt]}

    def make(lang: str) -> dict:
        nonlocal mt
        if lang in made:
            return made[lang]
        path = out_dir / f"{lang}.json"
        if path.exists() and not args.limit:
            print(f"{lang:9s} exists, skipped")
            made[lang] = {s: v for s, v in json.loads(path.read_text()).items() if s != "meta"}
            return made[lang]
        L = LANGS[lang]
        back = None
        if L["source"] == "original":
            prompts = {s: list(v) for s, v in eng.items()}
            meta = {"source": "original"}
        elif L["source"] == "mt":
            if mt is None:
                mt = load_backend()
            prompts = {s: mt(v, SRC, lang) for s, v in eng.items()}
            back = {s: mt(v, lang, SRC) for s, v in prompts.items()}
            meta = {"source": "mt", "system": system, **{k: v for k, v in mt_cfg.items() if k != "key_file"}}
        elif L["source"] == "transform":
            f = TRANSFORMS[L["transform"]]
            prompts = {s: [f(p) for p in v] for s, v in make(L["from"]).items()}
            meta = {"source": "transform", "from": L["from"], "transform": L["transform"]}
        else:
            raise NotImplementedError(f"{lang}: source {L['source']!r} not implemented yet")
        made[lang] = prompts
        rows = {s: qc(eng[s], prompts[s], back[s] if back else None, L["script"]) for s in prompts}
        if not args.limit:
            path.write_text(json.dumps({"meta": {**meta, "lang": lang, "model_splits": cfg["model"]}, **prompts},
                                       ensure_ascii=False, indent=1))
            (out_dir / f"{lang}_qc.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        flagged.extend(k for s in rows for k in report(lang, s, rows[s], flag))
        return prompts

    flagged: list[str] = []
    print(f"system: {system}")
    print(f"{'lang':9s} {'set':13s} {'n':>3s} {'chrF mean  med   min':>17s} {f'<{flag}':>4s} {'script':>7s} "
          f"{'untr':>5s} {'len×':>5s}")
    for lang in cfg["languages"]:
        make(lang)
    print(f"\nflagged for a look (back-translation chrF < {flag}): {flagged or 'none'}")
    print("smoke run: nothing written" if args.limit else f"written to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
