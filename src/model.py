"""Model loading helpers — base + QLoRA adapter."""
from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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


def load_base_model(
    model_cfg: dict,
    for_training: bool = False,
) -> PreTrainedModel:
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
    model.config.use_cache = not for_training
    log.info(
        f"Loaded {model_cfg['name']} (4bit={quant is not None}, "
        f"dtype={dtype}, device={'cuda' if on_cuda else 'cpu'})"
    )
    return model


def attach_lora(model: PreTrainedModel, lora_cfg: dict) -> PreTrainedModel:
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        f"LoRA attached: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.3f}%)"
    )
    return model


def load_adapter(model: PreTrainedModel, adapter_path: str) -> PreTrainedModel:
    log.info(f"Loading LoRA adapter from {adapter_path}")
    return PeftModel.from_pretrained(model, adapter_path)
