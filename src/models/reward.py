"""Factory functions for the reward model and the PPO policy.

Both share the same base (``Qwen2.5-1.5B-Instruct``) and the same LoRA configuration,
but they expose different heads:

* Reward model: ``AutoModelForSequenceClassification`` with ``num_labels=1`` (a single
  scalar logit treated as the reward).
* Policy: ``AutoModelForCausalLMWithValueHead`` from TRL (causal LM head + a value head
  used by PPO for advantage estimation).
"""
from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _bnb_config(quant_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not quant_cfg.get("load_in_4bit", False):
        return None
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_map[quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")],
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def _lora_config_for_seq_cls(lora_cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=list(lora_cfg["target_modules"]),
        task_type=TaskType.SEQ_CLS,
    )


def _lora_config_for_causal_lm(lora_cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=list(lora_cfg["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        # Qwen models reuse <|endoftext|> as eos but ship without a dedicated pad
        # token; using eos as pad is safe because we always mask it via attention_mask.
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_reward_model(
    model_name: str,
    lora_cfg: dict[str, Any],
    quant_cfg: dict[str, Any],
):
    """Reward model: causal backbone + scalar regression head + LoRA adapters."""
    bnb = _bnb_config(quant_cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if bnb is not None:
        model = prepare_model_for_kbit_training(model)
    # Score head needs to be trainable in fp32 even with k-bit base weights.
    model = get_peft_model(model, _lora_config_for_seq_cls(lora_cfg))
    tok = load_tokenizer(model_name)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def build_causal_lm_with_value_head(
    model_name: str,
    lora_cfg: dict[str, Any],
    quant_cfg: dict[str, Any],
):
    """Policy used by ``PPOTrainer``: causal LM + value head, with LoRA on the LM."""
    # Imported lazily so the rest of the module is usable without TRL installed.
    from trl import AutoModelForCausalLMWithValueHead

    bnb = _bnb_config(quant_cfg)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if bnb is not None:
        base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, _lora_config_for_causal_lm(lora_cfg))

    # TRL wraps the (already-PEFT'd) causal LM and bolts on a value head.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base)
    tok = load_tokenizer(model_name)
    model.config = base.config
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_reward_model_for_inference(
    base_model_name: str,
    adapter_path: str,
    quant_cfg: dict[str, Any] | None = None,
):
    """Load a previously trained reward-model LoRA adapter ready for scoring."""
    from peft import PeftModel

    bnb = _bnb_config(quant_cfg or {})
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    tok = load_tokenizer(base_model_name)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok
