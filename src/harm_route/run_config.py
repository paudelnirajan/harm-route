"""Load an experiment config, optionally for one language: fills {model} and {lang} in paths, and
takes `expected_script` from configs/languages.yaml. E0 configs (no {lang}) load unchanged."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = ("splits", "directions", "out_dir")


def load_config(path: str | Path, lang: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if lang is None:
        if any("{lang}" in str(cfg.get(k, "")) for k in PATH_KEYS):
            raise SystemExit(f"{path} is per-language: pass --lang (one of {cfg.get('languages')})")
        return cfg
    if lang not in cfg.get("languages", []):
        raise SystemExit(f"{lang} is not in this config's languages {cfg.get('languages')}")
    langs = yaml.safe_load((ROOT / "configs" / "languages.yaml").read_text())
    for k in PATH_KEYS:
        if k in cfg:
            cfg[k] = cfg[k].format(model=cfg["model"], lang=lang)
    cfg["lang"] = lang
    cfg["expected_script"] = langs[lang]["script"]  # garbled / wrong-language check in judge.py
    return cfg
