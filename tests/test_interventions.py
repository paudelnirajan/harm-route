"""Fast tests for interventions: fake tensors only, no model."""

import math

import torch

from harm_route.interventions import (
    ABLATE_POINTS,
    ablate_hooks,
    add_hooks,
    kl,
    make_ablate,
    make_add,
    make_pos_mean_ablate,
    pos_mean_ablate_hooks,
    t_inst_index,
    refusal_score,
)

D = 64
g = torch.Generator().manual_seed(0)
V = torch.randn(D, generator=g) * 5  # [d] not unit length on purpose
X = torch.randn(2, 7, D, generator=g)  # [b, T, d]


def test_ablate_removes_projection():
    out = make_ablate(V)(X, None)
    v_hat = V / V.norm()
    assert out.shape == X.shape
    assert (out @ v_hat).abs().max() < 1e-5


def test_ablate_keeps_orthogonal_part():
    v_hat = V / V.norm()
    out = make_ablate(V)(X, None)
    orth = X - (X @ v_hat).unsqueeze(-1) * v_hat
    assert torch.allclose(out, orth, atol=1e-6)


def test_ablate_idempotent_and_scale_invariant():
    once = make_ablate(V)(X, None)
    assert torch.allclose(make_ablate(V)(once, None), once, atol=1e-5)
    assert torch.allclose(make_ablate(-3 * V)(X, None), once, atol=1e-5)


def test_add_raises_projection_by_norm():
    out = make_add(V)(X, None)
    v_hat = V / V.norm()
    gain = (out - X) @ v_hat  # [b, T]
    assert torch.allclose(gain, torch.full_like(gain, V.norm().item()), atol=1e-4)


def test_zero_add_is_null():
    assert torch.equal(make_add(torch.zeros(D))(X, None), X)


def test_hooks_follow_dtype():
    xb = X.to(torch.bfloat16)
    assert make_ablate(V)(xb, None).dtype == torch.bfloat16
    assert make_add(V)(xb, None).dtype == torch.bfloat16


def test_hook_names():
    hooks = ablate_hooks(V, n_layers=3)
    assert len(hooks) == 3 * len(ABLATE_POINTS)
    assert hooks[0][0] == "blocks.0.hook_resid_pre"
    assert hooks[-1][0] == "blocks.2.hook_mlp_out"
    assert [n for n, _ in add_hooks(V, 5)] == ["blocks.5.hook_resid_pre"]


def test_kl_zero_and_nonneg():
    lp = torch.log_softmax(torch.randn(4, 50, generator=g), -1)
    lq = torch.log_softmax(torch.randn(4, 50, generator=g), -1)
    assert kl(lp, lp).abs().max() < 1e-6
    assert (kl(lp, lq) >= 0).all()


def test_refusal_score_known_value():
    p = torch.tensor([[0.6, 0.1, 0.3]])  # refusal ids {0, 1}: mass 0.7 vs rest 0.3
    s = refusal_score(p.log(), torch.tensor([0, 1]))
    assert math.isclose(s.item(), math.log(0.7 / 0.3), rel_tol=1e-5)


def test_pos_mean_ablate_one_token_per_row():
    idx = torch.tensor([2, 5])
    v_hat = V / V.norm()
    out = make_pos_mean_ablate(V, -3.0, idx)(X, None)
    rows = torch.arange(2)
    assert torch.allclose(out[rows, idx] @ v_hat, torch.full((2,), -3.0), atol=1e-4)  # edited tokens hit mu
    keep = torch.ones(2, 7, dtype=torch.bool); keep[rows, idx] = False
    assert torch.equal(out[keep], X[keep])  # every other token untouched
    assert not torch.equal(X[rows, idx] @ v_hat, out[rows, idx] @ v_hat)  # input not edited in place


def test_pos_mean_ablate_skips_generation_steps():
    x1 = X[:, :1]  # [b, 1, d]: a KV-cache step
    assert torch.equal(make_pos_mean_ablate(V, 0.0, torch.tensor([3, 3]))(x1, None), x1)


def test_t_inst_index_and_hook_names():
    pos = [{"t_inst": 4}, {"t_inst": 1}]
    assert t_inst_index(pos, torch.tensor([0, 3])).tolist() == [4, 4]
    hooks = pos_mean_ablate_hooks(V, torch.zeros(3), torch.tensor([0]))
    assert [n for n, _ in hooks] == [f"blocks.{l}.hook_resid_post" for l in range(3)]
