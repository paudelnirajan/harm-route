"""E0 step 4e: greedy generations for each (set, condition); prints counts only, never text.

    uv run python scripts/generate_e0.py --config configs/experiments/e00_generate.yaml
    uv run python scripts/generate_e0.py --config ... --check        # batched == one-by-one (padding test)
    uv run python scripts/generate_e0.py --config ... --limit 4 --max-new-tokens 16 --device cpu   # smoke

Writes one JSONL per (set, condition) to out_dir; existing files are skipped (resumable).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from harm_route.collect import collect_resid_batched
from harm_route.interventions import ablate_hooks, add_hooks, pos_mean_ablate_hooks, span_add_hooks, t_inst_index
from harm_route.models import load_bridge, load_tokenizer, model_config, pad_token_id
from harm_route.prompts import find_positions, pad_batch
from harm_route.run_config import load_config

ROOT = Path(__file__).resolve().parents[1]


def generate(bridge, tok, prompts, kw, pad_id, hooks, max_new_tokens, batch_size):
    """Greedy continuation of each prompt under hooks. Returns list of (text, n_new_tokens).

    hooks: list of (name, fn), or a callable (positions, shift) -> list for position-specific edits.
    """
    device = bridge.cfg.device
    stop = {tok.eos_token_id, pad_id}
    out = []
    for i in range(0, len(prompts), batch_size):
        ps = [find_positions(tok, x, kw) for x in prompts[i : i + batch_size]]
        ids, mask, shift = pad_batch([p["input_ids"] for p in ps], pad_id, "left")  # [b, T]
        hk = hooks(ps, shift) if callable(hooks) else list(hooks)
        with torch.inference_mode(), bridge.hooks(fwd_hooks=hk):
            seq = bridge.generate(
                ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new_tokens,
                do_sample=False, return_type="tokens", verbose=False,
            )  # [b, T + new]
        for row in seq[:, ids.shape[1] :].tolist():  # new tokens only
            k = next((j for j, t in enumerate(row) if t in stop), len(row))  # cut at first EOS/pad
            out.append((tok.decode(row[:k], skip_special_tokens=True), k))
    return out


def t_inst_levels(bridge, tok, prompts, kw, pad_id, v):
    """Mean projection on v_hat at t_inst, per layer (resid_post), over prompts. -> [L]"""
    ps = [find_positions(tok, x, kw) for x in prompts]
    inst, _ = collect_resid_batched(bridge, ps, pad_id, batch_size=16)  # [N, L, d]
    return (inst @ (v / v.norm())).mean(0)  # [L]


PUSH_CONDS = ("topup_h", "topup_rand", "fixed_h", "fixed_rand")


def push_setup(cfg, dirs, n_layers, d_model):
    """For the h-push conditions (E1 causal stage 2): push layer, unit h there, random unit u, English targets.

    Push layer = round(push_depth * (n_layers - 1)) (chosen from the readout sweep, notebook 10). Targets come from
    the reference language's TRAIN cache at that layer: hi = harmful mean on h_hat, gap = harmful - harmless.
    """
    L = round(cfg["push_depth"] * (n_layers - 1))
    h_hat = dirs["h"][L] / dirs["h"][L].norm()  # [d]
    u = torch.randn(d_model, generator=torch.Generator().manual_seed(cfg["seed"]))
    u = u / u.norm()  # [d], same seeded random direction as scripts/causal_readout.py
    ref = load_config(cfg["config_path"], cfg["reference_lang"], cfg["model"])
    E = torch.load(ROOT / ref["acts_dir"] / f"{cfg['reference_lang']}_acts.pt")
    hi = (E["harmful_train"]["inst"][:, L] @ h_hat).mean().item()
    lo = (E["harmless_train"]["inst"][:, L] @ h_hat).mean().item()
    return {"layer": L, "h": h_hat, "u": u, "hi": hi, "gap": hi - lo}


def push_hooks(c, X, setup, bridge, tok, kw, pad_id):
    """Hooks for one push condition on prompt list X: add size * v_hat on every instruction token at the push layer.

    topup: size per prompt = max(0, English harmful level - its own h reading at t_inst), measured here in a
    no-hook pass. fixed: size = English gap for every prompt. Prompt-only: the hook skips generated tokens.
    generate() builds its own position dicts, so the size is looked up by the rendered prompt text.
    """
    L = setup["layer"]
    ps = [find_positions(tok, x, kw) for x in X]
    if c.startswith("topup"):
        inst, _ = collect_resid_batched(bridge, ps, pad_id, batch_size=16)  # [N, L, d]
        sizes = (setup["hi"] - inst[:, L] @ setup["h"]).clamp(min=0).tolist()  # [N]
    else:
        sizes = [setup["gap"]] * len(ps)
    size_of = {p["text"]: d for p, d in zip(ps, sizes)}
    inner = span_add_hooks(setup["h"] if c.endswith("_h") else setup["u"], L)

    def build(ps_, shift):
        for p in ps_:
            p["push"] = size_of[p["text"]]
        return inner(ps_, shift)

    return build, sum(sizes) / len(sizes)


def build_conditions(cfg, dirs, n_layers, d_model, levels=None):
    """Condition name -> hooks. 'add_r_x{a}' adds a * r; 'tinst_h_harmless' sets h at t_inst to the harmless level."""
    r = dirs["r"][cfg["r_pos"], cfg["r_layer"]]  # [d]
    h = dirs["h"][cfg["h_layer"]]  # [d]
    rand = torch.randn(d_model, generator=torch.Generator().manual_seed(cfg["seed"]))  # [d]
    table = {
        "base": [],
        "ablate_r": ablate_hooks(r, n_layers),
        "add_r": add_hooks(r, cfg["r_layer"]),
        "ablate_random": ablate_hooks(rand, n_layers),
        "ablate_h": ablate_hooks(h, n_layers),
    }
    table.update({c: None for c in PUSH_CONDS})  # built per set in main() (sizes depend on the prompts)
    for c in cfg["conditions"]:
        if c.startswith("add_r_x"):
            table[c] = add_hooks(float(c.removeprefix("add_r_x")) * r, cfg["r_layer"])
    if levels is not None:  # hooks depend on each batch's t_inst index -> callable
        table["tinst_h_harmless"] = lambda ps, shift: pos_mean_ablate_hooks(h, levels, t_inst_index(ps, shift))
    return {c: table[c] for c in cfg["conditions"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="first N prompts per set (smoke test)")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check", action="store_true", help="batched vs one-by-one on 4 prompts, then exit")
    ap.add_argument("--lang", help="FLORES code, for per-language configs (E1)")
    ap.add_argument("--model", help="override the config's model (key in configs/models.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.lang, args.model)
    name = cfg["model"]
    mcfg = model_config(name)
    kw = mcfg["template_kwargs"]
    max_new = args.max_new_tokens or cfg["max_new_tokens"]
    dtype = getattr(torch, cfg["dtype"]) if args.device == "cuda" else torch.float32

    tok = load_tokenizer(name)
    pad_id = pad_token_id(tok)
    bridge = load_bridge(name, device=args.device, dtype=dtype)
    n_layers, d_model = bridge.cfg.n_layers, bridge.cfg.d_model
    splits = json.loads((ROOT / cfg["splits"].format(model=name)).read_text())
    if "train_splits" in cfg:  # sets from both split files (e.g. harmful_train + harmless_val)
        splits.update({k: v for k, v in json.loads((ROOT / cfg["train_splits"]).read_text()).items() if k != "meta"})
    needs_dirs = any(c != "base" for c in cfg["conditions"])  # base-only runs need no direction file
    dirs = torch.load(ROOT / cfg["directions"].format(model=name)) if needs_dirs else None
    if needs_dirs:  # layers stored with the directions (notebook 08) unless the config pins them (E0)
        cfg.setdefault("h_layer", dirs.get("h_layer"))
        cfg.setdefault("r_layer", dirs.get("r_layer"))
        cfg.setdefault("r_pos", -1)
    setup = None
    if any(c in PUSH_CONDS for c in cfg["conditions"]):
        cfg["config_path"] = args.config
        setup = push_setup(cfg, dirs, n_layers, d_model)
        print(f"h push at L{setup['layer']} (depth {cfg['push_depth']}): English harmful level {setup['hi']:.1f}, "
              f"gap {setup['gap']:.1f}")
    levels = None
    if "tinst_h_harmless" in cfg["conditions"]:  # target from TRAIN harmless prompts (never the val set)
        levels = t_inst_levels(bridge, tok, splits["harmless_train"], kw, pad_id, dirs["h"][cfg["h_layer"]])
        print(f"harmless level on h at t_inst, L{cfg['h_layer']}: {levels[cfg['h_layer']]:.2f}")
    conds = build_conditions(cfg, dirs, n_layers, d_model, levels) if needs_dirs else {"base": []}
    where = f"r[{cfg['r_pos']}, L{cfg['r_layer']}] depth {cfg['r_layer'] / (n_layers - 1):.2f}; " if needs_dirs else ""
    print(f"{name} on {args.device} {dtype}; {where}conditions {list(conds)}")

    if args.check:  # left padding must not change greedy outputs
        X = splits["harmless_val"][:4]
        batched = generate(bridge, tok, X, kw, pad_id, [], 32, batch_size=4)
        single = [generate(bridge, tok, [x], kw, pad_id, [], 32, 1)[0] for x in X]
        same = [a[0] == b[0] for a, b in zip(batched, single)]
        print(f"batched == one-by-one: {sum(same)}/{len(same)} (bf16 can flip near-ties; expect >= 3/4)")
        return

    out_dir = ROOT / cfg["out_dir"].format(model=name)
    out_dir.mkdir(parents=True, exist_ok=True)
    subs = [s.lower() for s in cfg["refusal_substrings"]]

    for s in cfg["sets"]:
        X = splits[s][: args.limit]
        for c, hooks in conds.items():
            path = out_dir / f"{s}__{c}.jsonl"
            if path.exists() and not args.limit:
                print(f"skip {path.name} (exists)")
                continue
            t = time.time()
            if c in PUSH_CONDS:
                hooks, mean_size = push_hooks(c, X, setup, bridge, tok, kw, pad_id)
                print(f"{s} {c}: mean push {mean_size:.1f} at L{setup['layer']}")
            gens = generate(bridge, tok, X, kw, pad_id, hooks, max_new, cfg["batch_size"])
            if not args.limit:  # smoke runs don't write files
                with path.open("w") as f:
                    for i, (text, n) in enumerate(gens):
                        f.write(json.dumps({"set": s, "condition": c, "idx": i, "n_tokens": n,
                                            "text": text}, ensure_ascii=False) + "\n")
            refused = sum(any(x in g.lower() for x in subs) for g, _ in gens)
            maxed = sum(n == max_new for _, n in gens)
            print(f"{s:13s} {c:16s} substring-refusal {refused:3d}/{len(gens)}  "
                  f"hit max_len {maxed:3d}  mean len {sum(n for _, n in gens) / len(gens):5.1f}  "
                  f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
