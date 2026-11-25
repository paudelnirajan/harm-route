"""E1 causal stage 1: patch-h readout. Forward passes only, no generation, no text written.

    uv run python scripts/causal_readout.py --config configs/experiments/e01_causal.yaml --lang npi_Deva
    uv run python scripts/causal_readout.py --config ... --lang npi_Deva --limit 4 --device cpu   # smoke

Conditions (push on every instruction token, inst_start..t_inst, at blocks.{h_layer}.hook_resid_post):
    base        no hook (checked against the activation cache)
    topup_h     per prompt, push = max(0, English harmful level - its own h reading at t_inst): raises h to the
                English harmful level (Nepali gets a big push, Serbian a small one)
    topup_rand  same per-prompt push sizes along a seeded random unit direction
    fixed_h     the same push for every prompt: English harmful - harmless gap on h (the nb07 1.7B push)
    fixed_rand  that fixed push along the random direction
Readout per prompt: projection on unit h_l at t_inst and on unit r_l at the last template token, every layer.
Layer sweep: with `push_depths` in the config, every push condition runs at each layer round(f * (n_layers - 1))
plus h_layer, with that layer's h direction and English targets; keys become e.g. "topup_h@L14".

Writes {out_dir}/{lang}_readout{_smoke}.pt:
    {"meta": {...}, "<set>": {"push": {cond: [N]}, "h": {cond: [N, L]}, "r": {cond: [N, L]}}}
Raw projections (not English-scaled): notebook 10 scales them with the cache, like notebook 09.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from harm_route.collect import collect_resid_batched
from harm_route.interventions import span_add_hooks
from harm_route.models import load_bridge, load_tokenizer, model_config, pad_token_id
from harm_route.prompts import find_positions
from harm_route.run_config import load_config

ROOT = Path(__file__).resolve().parents[1]


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)  # [L, d] -> [L, d], each row length 1


def readout(bridge, ps, pad_id, H, R, batch_size, hooks=None):
    """One forward pass per batch; returns h [N, L] (t_inst on h_l) and r [N, L] (last token on r_l)."""
    inst, post = collect_resid_batched(bridge, ps, pad_id, batch_size=batch_size, hooks=hooks)  # [N, L, d], [N, P, L, d]
    h = torch.einsum("nld,ld->nl", inst, H)  # one dot product per (prompt, layer)
    r = torch.einsum("nld,ld->nl", post[:, -1], R)
    return h, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lang", nargs="*", help="default: all languages in the config")
    ap.add_argument("--model", help="override the config's model (key in configs/models.yaml)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, help="first N prompts per set (smoke; output gets a _smoke suffix)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    langs = args.lang or yaml.safe_load(Path(args.config).read_text())["languages"]
    cfg0 = load_config(args.config, langs[0], args.model)
    name = cfg0["model"]
    tok = load_tokenizer(name)
    kw = model_config(name)["template_kwargs"]
    pad_id = pad_token_id(tok)

    dirs = torch.load(ROOT / cfg0["directions"])
    H_LAYER, R_LAYER = dirs["h_layer"], dirs["r_layer"]  # chosen in notebook 08, stored with the directions
    H = unit_rows(dirs["h"])  # [L, d]
    R = unit_rows(dirs["r"][-1])  # [L, d], r at the last template token
    n_layers, d_model = H.shape

    # targets from the reference language's TRAIN cache (never val, never the language being tested), per layer
    ref_cfg = load_config(args.config, cfg0["reference_lang"], args.model)
    E = torch.load(ROOT / ref_cfg["acts_dir"] / f"{cfg0['reference_lang']}_acts.pt")
    hi = torch.einsum("nld,ld->nl", E["harmful_train"]["inst"], H).mean(0)  # [L], English harmful level on h_l
    lo = torch.einsum("nld,ld->nl", E["harmless_train"]["inst"], H).mean(0)  # [L], English harmless level
    del E
    gap = hi - lo  # [L]
    u = torch.randn(d_model, generator=torch.Generator().manual_seed(cfg0["seed"]))
    u = u / u.norm()  # [d], random control direction

    # push layers: h_layer only (stage 1), or a sweep at fixed fractions of depth + h_layer (same rule for any model)
    depths = cfg0.get("push_depths")
    push_layers = sorted({round(f * (n_layers - 1)) for f in depths} | {H_LAYER}) if depths else [H_LAYER]
    tag = lambda L: f"@L{L}" if depths else ""  # stage-1 files keep plain condition names
    print(f"{name}: L={n_layers} d={d_model} h_layer {H_LAYER} r_layer {R_LAYER} | push layers {push_layers} | "
          + " ".join(f"L{L} gap {gap[L]:.1f}" for L in push_layers) + f" | cos(u, h) {(u @ H[H_LAYER]).item():.3f}")

    bridge = None
    for lang in langs:
        cfg = load_config(args.config, lang, args.model)
        out = ROOT / cfg["out_dir"] / f"{lang}_readout{cfg.get('tag', '')}{'_smoke' if args.limit else ''}.pt"
        if out.exists() and not args.overwrite:
            print(f"{lang}: exists, skip ({out.relative_to(ROOT)})")
            continue
        if bridge is None:
            dtype = getattr(torch, cfg["dtype"]) if args.device == "cuda" else torch.float32
            bridge = load_bridge(name, device=args.device, dtype=dtype)
            assert bridge.cfg.n_layers == n_layers and bridge.cfg.d_model == d_model, "directions are for another model"
        cache = torch.load(ROOT / cfg["acts_dir"] / f"{lang}_acts.pt")
        res = {"meta": {"model": name, "lang": lang, "h_layer": H_LAYER, "r_layer": R_LAYER, "push_layers": push_layers,
                        "hi": hi, "lo": lo, "gap": gap, "seed": cfg["seed"], "dtype": str(bridge.cfg.dtype),
                        "limit": args.limit}}

        for st, split_key in cfg["sets"].items():
            prompts = json.loads((ROOT / cfg[split_key]).read_text())[st][: args.limit]
            ps = [find_positions(tok, x, kw) for x in prompts]
            bs = cfg["batch_size"]
            h0, r0 = readout(bridge, ps, pad_id, H, R, bs)  # base, [N, L] each
            c_h = cache[st]["inst"][: len(ps), H_LAYER] @ H[H_LAYER]  # [N], same prompts from the cache
            c_r = cache[st]["last"][: len(ps), R_LAYER] @ R[R_LAYER]
            print(f"{lang} {st}: base vs cache |diff| max h {(h0[:, H_LAYER] - c_h).abs().max():.2f} "
                  f"r {(r0[:, R_LAYER] - c_r).abs().max():.2f}")
            res[st] = {"push": {}, "h": {}, "r": {}}
            for L in push_layers:
                topup = (hi[L] - h0[:, L]).clamp(min=0)  # [N], 0 if a prompt is already at the English level
                pushes = {"base": torch.zeros(len(ps)), "topup_h": topup, "topup_rand": topup,
                          "fixed_h": torch.full((len(ps),), gap[L].item()),
                          "fixed_rand": torch.full((len(ps),), gap[L].item())}
                for c in cfg["conditions"]:
                    if c == "base":
                        if "base" in res[st]["r"]:
                            continue
                        key, h, r = "base", h0, r0
                    else:
                        key = c + tag(L)
                        v = H[L] if c.endswith("_h") else u
                        for p, d in zip(ps, pushes[c].tolist()):
                            p["push"] = d  # span_add_hooks reads it per prompt, so it survives batching
                        h, r = readout(bridge, ps, pad_id, H, R, bs, hooks=span_add_hooks(v, L))
                        assert torch.allclose(h[:, :L], h0[:, :L], atol=0.05 * gap[L]), "layers below the push moved"
                        if c.endswith("_h"):  # the push landed: h reading at the push layer went up by push
                            err = (h[:, L] - h0[:, L] - pushes[c]).abs().max().item()
                            assert err < 0.05 * gap[L], f"push did not land ({err:.2f})"
                    res[st]["push"][key], res[st]["h"][key], res[st]["r"][key] = pushes[c].clone(), h, r
                    print(f"{lang} {st:13s} {key:15s} mean push {pushes[c].mean():6.1f} | "
                          f"h@L{L} {h[:, L].mean():7.1f} | r@L{R_LAYER} {r[:, R_LAYER].mean():7.1f} | "
                          f"r@L{n_layers - 2} {r[:, n_layers - 2].mean():7.1f}")
        del cache
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, out)
        print(f"{lang}: saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
