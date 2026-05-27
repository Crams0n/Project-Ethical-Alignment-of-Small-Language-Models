"""Model loading — base LM for inference only. No LoRA in the RAG approach:
the model is used as-is, alignment comes from retrieved context.
"""
from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils import get_logger

log = get_logger(__name__)


def _bnb_config(cfg: dict) -> BitsAndBytesConfig | None:
    if not cfg.get("load_in_4bit", False):
        return None
    dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_tokenizer(model_cfg: dict) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_model(model_cfg: dict) -> PreTrainedModel:
    quant = _bnb_config(model_cfg)
    on_cuda = torch.cuda.is_available()
    if quant is not None and not on_cuda:
        log.warning("4-bit quantization requested but no CUDA — falling back to fp32.")
        quant = None
    if quant is None:
        dtype = torch.bfloat16 if on_cuda else torch.float32
    else:
        dtype = None
    kwargs = dict(
        quantization_config=quant,
        torch_dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if on_cuda:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)
    if not on_cuda:
        model.to("cpu")
    model.config.use_cache = True
    log.info(
        f"Loaded {model_cfg['name']} (4bit={quant is not None}, "
        f"dtype={dtype}, device={'cuda' if on_cuda else 'cpu'})"
    )
    return model
