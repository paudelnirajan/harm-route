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
