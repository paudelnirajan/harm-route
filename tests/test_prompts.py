"""Position and padding checks from notebooks/01, across model families and scripts (tokenizers only)."""

import pytest

from harm_route.models import load_tokenizer, model_config, pad_token_id
from harm_route.prompts import find_positions, pad_batch

MODEL_NAMES = ["qwen3-0.6b", "gemma3-1b-it", "llama3.2-1b-it"]

# Benign on purpose: positions don't depend on content.
PROMPTS = {
    "eng_Latn": "How do I bake bread?",
    "npi_Deva": "रोटी कसरी बनाउने?",
    "npi_Latn": "roti kasari banaune?",
    "rus_Cyrl": "Как испечь хлеб?",
    "zho_Hans": "如何烤面包？",
}


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request):
    name = request.param
    return load_tokenizer(name), model_config(name)["template_kwargs"]


@pytest.mark.parametrize("lang", PROMPTS)
def test_positions(model, lang):
    tok, kwargs = model
    prompt = PROMPTS[lang]
    p = find_positions(tok, prompt, kwargs)
    ids, t_inst, post = p["input_ids"], p["t_inst"], p["t_post_inst"]

    assert tok.decode([ids[t_inst]]).endswith(prompt[-1])  # t_inst not off by one
    assert p["text"].endswith(tok.decode([ids[i] for i in post]))  # tail covered exactly
    assert tok.decode(ids[p["inst_start"] : t_inst + 1]).strip() == prompt  # span complete
    if tok.bos_token_id is not None:
        assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id  # BOS once


def test_post_inst_length_is_language_independent(model):
    tok, kwargs = model
    lengths = {len(find_positions(tok, p, kwargs)["t_post_inst"]) for p in PROMPTS.values()}
    assert len(lengths) == 1


@pytest.mark.parametrize("prompt", ["hi", "user", "model", "assistant"])
def test_short_prompt_inside_template_text(model, prompt):
    # these strings also occur in some template ("</think>", role headers); must find the user's copy
    tok, kwargs = model
    p = find_positions(tok, prompt, kwargs)
    assert tok.decode(p["input_ids"][p["inst_start"] : p["t_inst"] + 1]).strip() == prompt


def test_whitespace_mismatch_raises():
    tok = load_tokenizer("gemma3-1b-it")  # Gemma's template strips whitespace
    with pytest.raises(ValueError):
        find_positions(tok, "  spaces around  ", model_config("gemma3-1b-it")["template_kwargs"])


@pytest.mark.parametrize("side", ["left", "right"])
def test_pad_batch_shifts_positions(model, side):
    tok, kwargs = model
    ps = [find_positions(tok, p, kwargs) for p in PROMPTS.values()]
    seqs = [p["input_ids"] for p in ps]
    ids, mask, shift = pad_batch(seqs, pad_token_id(tok), side)

    assert ids.shape == mask.shape == (len(seqs), max(map(len, seqs)))
    assert mask.sum(1).tolist() == [len(s) for s in seqs]
    for b, p in enumerate(ps):
        assert ids[b, p["t_inst"] + shift[b]] == seqs[b][p["t_inst"]]
        if side == "left":
            assert ids[b, -1] == seqs[b][-1]  # every row ends at T-1
