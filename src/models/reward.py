"""Builders for the reward model and the RLOO policy.

The TRL 1.x trainers apply PEFT themselves when given a ``peft_config``, so the helpers
here are intentionally thin: they return a base model (optionally 4-bit quantized) plus
the matching tokenizer. The notebooks then hand the model and the ``LoraConfig`` to
``RewardTrainer`` / ``RLOOTrainer`` and let TRL do the PEFT wrapping.
"""
from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _default_dtype() -> torch.dtype:
    """bf16 on CUDA (much faster), fp32 elsewhere (bf16 on CPU is supported but slow)."""
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def make_bnb_config(quant_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not quant_cfg.get("load_in_4bit", False):
        return None
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_map[quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")],
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def make_lora_config(lora_cfg: dict[str, Any], task_type: TaskType) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=list(lora_cfg["target_modules"]),
        task_type=task_type,
    )


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        # Qwen ships without a dedicated pad token; reuse eos. Always-masked via
        # attention_mask, so this is safe.
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_reward_model(model_name: str, quant_cfg: dict[str, Any]):
    """Causal backbone + scalar regression head, optionally 4-bit quantized.

    The model is *not* wrapped with PEFT here — the caller passes a ``LoraConfig`` to
    ``RewardTrainer`` and TRL applies the adapter internally.
    """
    bnb = make_bnb_config(quant_cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        quantization_config=bnb,
        torch_dtype=_default_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if bnb is not None:
        model = prepare_model_for_kbit_training(model)
    tok = load_tokenizer(model_name)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_base_causal_lm(model_name: str, quant_cfg: dict[str, Any]):
    """Causal LM (policy backbone), optionally 4-bit quantized. No PEFT wrapping."""
    bnb = make_bnb_config(quant_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        torch_dtype=_default_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if bnb is not None:
        model = prepare_model_for_kbit_training(model)
    tok = load_tokenizer(model_name)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_reward_model_for_inference(
    base_model_name: str,
    adapter_path: str,
    quant_cfg: dict[str, Any] | None = None,
):
    """Reload a trained reward-model LoRA adapter for scoring."""
    from peft import PeftModel

    bnb = make_bnb_config(quant_cfg or {})
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        quantization_config=bnb,
        torch_dtype=_default_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    tok = load_tokenizer(base_model_name)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok
