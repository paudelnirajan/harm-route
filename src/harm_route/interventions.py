"""Directional interventions on the residual stream (TransformerBridge hooks) + output metrics.

A hook is fn(resid, hook) -> tensor; TransformerBridge calls it at a named point in the forward pass and
uses the returned tensor in place of the activation. resid is [batch, seq, d_model] (seq = 1 per step when
generating with a KV cache). Verified in notebooks/03_e0_interventions.ipynb (null edit = 0, projection
after ablation ~1e-5, addition raises the projection by exactly ||v||).
"""

import torch

from harm_route.prompts import find_positions, pad_batch

ABLATE_POINTS = ("hook_resid_pre", "hook_attn_out", "hook_mlp_out")


# ---------- hooks ----------


def make_ablate(v: torch.Tensor):
    """Hook that removes the component along v: x <- x - (x . v_hat) v_hat (every token)."""
    v_hat = v / v.norm()  # [d]

    def fn(resid, hook):
        u = v_hat.to(resid.device, resid.dtype)  # no-op on CPU fp32; needed on GPU / bf16
        coef = resid @ u  # [b, T]
        return resid - coef.unsqueeze(-1) * u  # [b, T, d]

    return fn


def make_add(v: torch.Tensor):
    """Hook that adds the raw vector v to every token (length matters: v is not normalized)."""

    def fn(resid, hook):
        return resid + v.to(resid.device, resid.dtype)  # [b, T, d] + [d] broadcasts

    return fn


def ablate_hooks(v: torch.Tensor, n_layers: int):
    """Ablate v everywhere: before every block and from every attention / MLP output (Arditi et al.)."""
    fn = make_ablate(v)
    return [(f"blocks.{l}.{p}", fn) for l in range(n_layers) for p in ABLATE_POINTS]


def add_hooks(v: torch.Tensor, layer: int):
    """Add v once, at the input of one block, on all tokens."""
    return [(f"blocks.{layer}.hook_resid_pre", make_add(v))]


# ---------- forward + metrics ----------


def last_logp(
    bridge, tok, prompts: list[str], template_kwargs: dict, pad_id: int,
    fwd_hooks=(), batch_size: int = 8,
) -> torch.Tensor:
    """Next-token log-probs at the last position for each prompt, under fwd_hooks. Returns [N, V] fp32 CPU.

    Left padding so every prompt's last token sits at index -1.
    """
    device = bridge.cfg.device
    out = []
    for i in range(0, len(prompts), batch_size):
        ps = [find_positions(tok, x, template_kwargs) for x in prompts[i : i + batch_size]]
        ids, mask, _ = pad_batch([p["input_ids"] for p in ps], pad_id, "left")  # [b, T]
        with torch.inference_mode(), bridge.hooks(fwd_hooks=list(fwd_hooks)):
            logits = bridge(ids.to(device), attention_mask=mask.to(device))  # [b, T, V]
        out.append(torch.log_softmax(logits[:, -1].float(), dim=-1).cpu())  # [b, V]
    return torch.cat(out)  # [N, V]


def refusal_score(logp: torch.Tensor, refusal_ids: torch.Tensor) -> torch.Tensor:
    """Log-odds that the next token is a refusal token: logsumexp(refusal) - logsumexp(rest). [N, V] -> [N]."""
    rest = logp.clone()
    rest[:, refusal_ids] = float("-inf")
    return torch.logsumexp(logp[:, refusal_ids], -1) - torch.logsumexp(rest, -1)


def kl(logp_p: torch.Tensor, logp_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) per row from log-probs. [N, V], [N, V] -> [N]. P = clean, Q = edited."""
    return (logp_p.exp() * (logp_p - logp_q)).sum(-1)
