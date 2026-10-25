"""Chat-template rendering, token positions (t_inst / t_post_inst) and padding.

Positions are found from character offsets, so they work for any model's template and any script.
See notebooks/01_chat_templates_positions.ipynb for the walkthrough.
"""

import torch


def render(tok, user_content: str, template_kwargs: dict) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )


def find_positions(tok, user_content: str, template_kwargs: dict) -> dict:
    """Render a single-turn chat prompt and locate the instruction and post-instruction tokens.

    Returns:
        text:        rendered prompt string
        input_ids:   list[int], tokenized with add_special_tokens=False (template already has BOS)
        inst_start:  index of the first instruction token
        t_inst:      index of the last instruction token
        t_post_inst: list of indices after t_inst (end-of-turn + assistant header), to the end
    """
    text = render(tok, user_content, template_kwargs)

    # 1. character span of the instruction inside the rendered string. Render once with a sentinel
    # to get the template's prefix/suffix, rather than searching: a short instruction ("hi") can also
    # occur inside the template text ("</think>").
    sentinel = "\x00USER\x00"
    s_text = render(tok, sentinel, template_kwargs)
    prefix, suffix = s_text.split(sentinel)
    if text != prefix + user_content + suffix:
        # e.g. the Gemma/Llama templates strip surrounding whitespace -> strip prompts in data prep
        raise ValueError(f"user content not found verbatim in rendered template: {user_content!r}")
    start = len(prefix)
    end = start + len(user_content)

    # 2. tokenize, keeping each token's (char_start, char_end)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]  # both length seq

    # 3. characters -> tokens
    inst_start = min(i for i, (s, e) in enumerate(offsets) if e > start)
    t_inst = max(i for i, (s, e) in enumerate(offsets) if s < end)

    # 4. refuse to guess if a token mixes instruction and template text
    if offsets[t_inst][1] > end:
        raise ValueError(
            f"token {t_inst} straddles the instruction end: {tok.decode([ids[t_inst]])!r}"
        )

    return {
        "text": text,
        "input_ids": ids,
        "inst_start": inst_start,
        "t_inst": t_inst,
        "t_post_inst": list(range(t_inst + 1, len(ids))),
    }


def pad_batch(seqs: list[list[int]], pad_id: int, side: str):
    """Pad token-id lists into a rectangle.

    Returns input_ids [B, T], attention_mask [B, T], shift [B] (how far each row's tokens moved).
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    B, T = len(seqs), max(len(s) for s in seqs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)  # [B, T]
    attention_mask = torch.zeros((B, T), dtype=torch.long)  # [B, T]
    shift = torch.zeros(B, dtype=torch.long)  # [B]
    for b, s in enumerate(seqs):
        n = len(s)
        if side == "right":
            input_ids[b, :n] = torch.tensor(s)
            attention_mask[b, :n] = 1
        else:
            input_ids[b, T - n :] = torch.tensor(s)
            attention_mask[b, T - n :] = 1
            shift[b] = T - n
    return input_ids, attention_mask, shift
