"""Human review of machine-translated prompts: ok / fix / drop, per item. Resumable.

    # review (SHOWS PROMPTS, incl. harmful ones: run it yourself in your terminal, never through Claude)
    uv run python scripts/review_translations.py --config configs/experiments/e01_translate.yaml --lang npi_Deva --reviewer nirajan
    # second look at specific items (newest decision wins)
    uv run python scripts/review_translations.py --config ... --lang npi_Deva --reviewer nirajan --keys 'harmless_val|0'
    # summary (counts + item keys only; safe to paste)
    uv run python scripts/review_translations.py --config ... --summary

Shown per item: English original, the translation, its back-translation to English and the chrF score.
  o  ok (meaning and register kept; a harmful request must stay harmful, not softened)
  f  fix: opens $EDITOR (default nano) with the translation; save and close to keep your version
  d  drop (can't be fixed quickly; dropped in ALL languages to keep comparisons paired)
  s  skip   q  quit
Reviews the primary MT system's output (--system to choose another).
Decisions go to {out_dir}/review/{system}/{lang}__{reviewer}.jsonl. The MT file itself is never modified.
"""

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from harm_route.lang_data import final_prompts, load_reviews

ROOT = Path(__file__).resolve().parents[1]
LANGS = yaml.safe_load((ROOT / "configs" / "languages.yaml").read_text())
HELP = ("o ok (meaning kept; harmful stays harmful)   f fix (opens editor)   d drop   s skip   q quit\n"
        "Check: same request? nothing added or dropped? not softened or made more harmful? natural wording?")


def edit(text: str) -> str:
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    subprocess.run([os.environ.get("EDITOR", "nano"), path], check=False)
    new = Path(path).read_text(encoding="utf-8").strip()
    os.unlink(path)
    return new


def review(cfg, out_dir: Path, system: str, lang: str, reviewer: str, keys: list[str] | None):
    if LANGS[lang]["source"] != "mt":
        raise SystemExit(f"{lang} is {LANGS[lang]['source']!r}; review its source language instead")
    eng = json.loads((ROOT / cfg["splits"].format(model=cfg["model"])).read_text())
    mt = json.loads((out_dir / "mt" / system / f"{lang}.json").read_text())
    qc = json.loads((out_dir / "mt" / system / f"{lang}_qc.json").read_text())
    (out_dir / "review" / system).mkdir(parents=True, exist_ok=True)
    path = out_dir / "review" / system / f"{lang}__{reviewer}.jsonl"
    done = {json.loads(l)["key"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
    items = [f"{s}|{i}" for s in cfg["sets"] for i in range(len(mt[s]))]
    # low chrF first: the likely problems get your freshest attention
    items.sort(key=lambda k: qc[k.split("|")[0]][int(k.split("|")[1])]["chrf"])
    todo = [k for k in items if k in set(keys)] if keys else [k for k in items if k not in done]
    print(f"{lang} ({system}): {len(done)} reviewed, {len(todo)} to go.\n{HELP}")
    input("\nEnter to start...")
    for n, k in enumerate(todo, 1):
        s, i = k.split("|")[0], int(k.split("|")[1])
        q = qc[s][i]
        os.system("clear")
        print(f"[{n}/{len(todo)}] {lang} ({system})  {k}  chrF {q['chrf']:.0f}\n\n=== ENGLISH ===\n{eng[s][i]}\n\n"
              f"=== {lang} ===\n{mt[s][i]}\n\n=== BACK-TRANSLATION (machine) ===\n{q['back']}\n\n{HELP}")
        while (c := input("review> ").strip().lower()) not in {"o", "f", "d", "s", "q"}:
            print("unknown key")
        if c == "q":
            break
        if c == "s":
            continue
        row = {"key": k, "decision": {"o": "ok", "f": "fix", "d": "drop"}[c]}
        if c == "f":
            row["text"] = edit(mt[s][i])
            if row["text"] == mt[s][i].strip():
                row["decision"] = "ok"  # opened the editor but changed nothing
        row["note"] = input("note (optional, Enter to skip)> ").strip()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.system("clear")
    print(f"saved to {path.relative_to(ROOT)}")


def summary(cfg, out_dir: Path, system: str):
    """Counts only: decisions per language, and what the final parallel set looks like."""
    langs, sets, flag = cfg["languages"], cfg["sets"], cfg["qc"]["chrf_flag"]
    for lang in langs:
        if LANGS[lang]["source"] == "mt":
            rev = load_reviews(out_dir, system, lang)
            c = Counter(r["decision"] for r in rev.values())
            n = sum(len(json.loads((out_dir / "mt" / system / f"{lang}.json").read_text())[s]) for s in sets)
            print(f"{lang:9s} reviewed {len(rev):3d}/{n}  ok {c['ok']:3d}  fix {c['fix']:3d}  drop {c['drop']:3d}"
                  + (f"   dropped: {sorted(k for k, r in rev.items() if r['decision'] == 'drop')}" if c["drop"] else ""))
    print(f"system: {system}")
    _, keep, why = final_prompts(out_dir, system, langs, LANGS, sets, flag)
    print("\nfinal parallel set: " + ", ".join(f"{s} {len(v)}" for s, v in keep.items()))
    print("items dropped because of each language (review drop, or unreviewed with chrF < "
          f"{flag}): {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lang")
    ap.add_argument("--reviewer")
    ap.add_argument("--keys", nargs="+", help="re-review only these set|idx keys")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--system", help="MT system to review (default: primary in the config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = ROOT / cfg["out_dir"].format(model=cfg["model"])
    system = args.system or cfg["primary"]
    if args.summary:
        summary(cfg, out_dir, system)
    elif args.lang and args.reviewer:
        review(cfg, out_dir, system, args.lang, args.reviewer, args.keys)
    else:
        raise SystemExit("need --summary, or --lang and --reviewer")


if __name__ == "__main__":
    main()
