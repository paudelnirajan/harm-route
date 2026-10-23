# harm-route

When a chat model fails to refuse a harmful prompt in a non-English language, did it never
*understand* the harm (comprehension failure), or understand it and never *route* it to refusal
(routing failure)? And does the prompt's script/tokenization, holding the language fixed, *cause*
the difference?

The design is language- and model-agnostic: a panel of ~12 languages spanning resource tier ×
script, with within-language script manipulations (native ↔ romanized, Serbian Cyrillic ↔ Latin,
forced segmentation), across Qwen3, Gemma 3 and Llama 3.2. Languages are FLORES+ codes and models
are HF ids from `configs/`; nothing hardcodes a language, model, layer count or chat template.

The plan lives in `notes/research_guide.md`.

## Layout

```
src/harm_route/      reusable code (directions, probes, patching, labels); import from notebooks
notebooks/           exploration and debugging, numbered in reading order
scripts/             entry points for long runs (python scripts/x.py --config ...); fertility_belebele.py
configs/             experiment configs (models, conditions, decoding settings)
data/prompts/        small curated prompt sets (redacted where needed)
data/raw, cache      downloaded / derived data, not tracked
data/generations/    model outputs (may contain harmful text), not tracked
results/             outputs (plots, tensors, CSVs), not tracked
tests/               quick checks for src/ code
external/            third-party repos we read from (refusal_direction, Minionese, ...), not tracked
notes/               private research guide and paper notes, not tracked
```

## Setup

```bash
uv sync                                    # creates .venv from pyproject.toml + uv.lock
uv run python -m ipykernel install --user --name harm-route   # Jupyter kernel (once)
uv run nbstripout --install                # strip notebook outputs from commits (once per clone)
uv add <package>                           # add a dependency (updates pyproject.toml + uv.lock)
```

Open notebooks with the **harm-route** kernel.
