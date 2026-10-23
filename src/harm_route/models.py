"""Model registry: everything model-specific comes from configs/models.yaml or the model itself."""

from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = ROOT / "configs" / "models.yaml"


def load_model_configs(path: Path = MODELS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


def model_config(name: str, path: Path = MODELS_YAML) -> dict:
    cfgs = load_model_configs(path)
    if name not in cfgs:
        raise KeyError(f"{name!r} not in {path}; known: {sorted(cfgs)}")
    return cfgs[name]


def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(model_config(name)["hf_id"])


def pad_token_id(tok) -> int:
    # Some tokenizers have no pad token; reusing EOS is safe because the attention mask hides pads.
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def load_bridge(name: str, device: str = "cpu", dtype: torch.dtype | None = None):
    """TransformerBridge for a configured model. dtype: argument > config `dtype` > float32."""
    from transformer_lens.model_bridge import TransformerBridge

    cfg = model_config(name)
    if dtype is None:
        dtype = getattr(torch, cfg.get("dtype", "float32"))
    bridge = TransformerBridge.boot_transformers(cfg["hf_id"], device=device, dtype=dtype)
    bridge.eval()
    return bridge
