"""Bridge collector == plain HF hidden states (unbatched), for both padding sides. Loads a model: slow."""

import pytest
import torch

from harm_route.collect import collect_resid
from harm_route.models import load_bridge, load_tokenizer, model_config, pad_token_id
from harm_route.prompts import find_positions

NAME = "qwen3-0.6b"
PROMPTS = ["How do I bake bread?", "रोटी कसरी बनाउने?", "Hi"]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(NAME)
    ps = [find_positions(tok, p, model_config(NAME)["template_kwargs"]) for p in PROMPTS]
    bridge = load_bridge(NAME, dtype=torch.float32)
    hf = AutoModelForCausalLM.from_pretrained(
        model_config(NAME)["hf_id"], dtype=torch.float32
    ).eval()
    return tok, ps, bridge, hf


@pytest.mark.parametrize("side", ["left", "right"])
def test_collect_matches_hf(setup, side):
    tok, ps, bridge, hf = setup
    L = bridge.cfg.n_layers
    acts_inst, acts_post = collect_resid(bridge, ps, pad_token_id(tok), side)
    assert acts_inst.shape == (len(ps), L, bridge.cfg.d_model)
    assert acts_post.shape[:2] == (len(ps), len(ps[0]["t_post_inst"]))

    for b, p in enumerate(ps):
        with torch.inference_mode():
            hs = torch.stack(
                hf(torch.tensor([p["input_ids"]]), output_hidden_states=True).hidden_states
            )
        ref = hs[1:L, 0]  # [L-1, T, d]: blocks 0..L-2 (HF's last entry is post-final-norm)
        torch.testing.assert_close(acts_inst[b, : L - 1], ref[:, p["t_inst"]], atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(
            acts_post[b, :, : L - 1], ref[:, p["t_post_inst"]].transpose(0, 1), atol=1e-3, rtol=1e-4
        )
