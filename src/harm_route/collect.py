"""Residual-stream collection at t_inst / t_post_inst with TransformerBridge.

Verified against HF output_hidden_states on Qwen3-0.6B (notebooks/01, tests/test_collect.py):
Bridge blocks.{l}.hook_resid_post == HF hidden_states[l+1]; HF's last entry is post-final-norm.
"""

import torch
from tqdm.auto import tqdm

from harm_route.prompts import pad_batch


def resid_hook_names(bridge) -> list[str]:
    return [f"blocks.{l}.hook_resid_post" for l in range(bridge.cfg.n_layers)]


def collect_resid(bridge, ps: list[dict], pad_id: int, side: str = "left"):
    """Residual stream after every block, at t_inst and at every t_post_inst token.

    ps: outputs of find_positions (same model). All must have the same number P of t_post_inst.
    Returns acts_inst [B, L, d], acts_post [B, P, L, d] (float32, CPU).
    """
    seqs = [p["input_ids"] for p in ps]
    ids, mask, shift = pad_batch(seqs, pad_id, side)  # [B, T], [B, T], [B]
    device = bridge.cfg.device
    names = resid_hook_names(bridge)

    with torch.inference_mode():
        _, cache = bridge.run_with_cache(
            ids.to(device), attention_mask=mask.to(device), names_filter=names
        )

    resid = torch.stack([cache[n] for n in names], dim=2).cpu()  # [B, T, L, d]
    rows = torch.arange(len(ps))  # [B]
    t_inst = torch.tensor([p["t_inst"] for p in ps]) + shift  # [B]
    t_post = torch.tensor([p["t_post_inst"] for p in ps]) + shift[:, None]  # [B, P]

    acts_inst = resid[rows, t_inst]  # [B, L, d]
    acts_post = resid[rows[:, None], t_post]  # [B, P, L, d]
    return acts_inst.float(), acts_post.float()


def collect_resid_batched(
    bridge, ps: list[dict], pad_id: int, batch_size: int = 16, side: str = "left"
):
    """collect_resid over many prompts in chunks. Returns acts_inst [N, L, d], acts_post [N, P, L, d]."""
    inst, post = [], []
    for i in tqdm(range(0, len(ps), batch_size), desc="collect"):
        a_inst, a_post = collect_resid(bridge, ps[i : i + batch_size], pad_id, side)
        inst.append(a_inst)
        post.append(a_post)
    return torch.cat(inst), torch.cat(post)
