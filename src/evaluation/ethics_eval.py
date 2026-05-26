"""Evaluation logic for ETHICS subsets.

We score each example by comparing the log-probability of two candidate continuations
under the model. This is robust to sampling noise and does not require the model to
emit well-formatted text — only to assign a higher probability to the correct token.

The scoring trick (compare log P(" yes" | prompt) vs log P(" no" | prompt)) is the
standard lm-evaluation-harness approach for binary classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.ethics import EthicsExample
from src.utils.prompts import (
    A_TOKEN,
    B_TOKEN,
    NO_TOKEN,
    YES_TOKEN,
    commonsense_prompt,
    deontology_prompt,
    justice_prompt,
    utilitarianism_prompt,
    virtue_prompt,
)


@dataclass
class SubsetResult:
    subset: str
    accuracy: float
    n: int
    predictions: list[int]
    gold: list[int]


def _build_prompt(ex: EthicsExample) -> tuple[str, str, str]:
    """Return ``(prompt, positive_continuation, negative_continuation)``.

    ``positive`` is the continuation whose higher log-prob means the model predicts
    label == 1; ``negative`` corresponds to label == 0.
    """
    s = ex.subset
    f = ex.fields
    if s == "commonsense":
        return commonsense_prompt(f["scenario"]), YES_TOKEN, NO_TOKEN
    if s == "justice":
        return justice_prompt(f["scenario"]), YES_TOKEN, NO_TOKEN
    if s == "deontology":
        return deontology_prompt(f["scenario"], f["excuse"]), YES_TOKEN, NO_TOKEN
    if s == "virtue":
        return virtue_prompt(f["scenario"], f["trait"]), YES_TOKEN, NO_TOKEN
    if s == "utilitarianism":
        return utilitarianism_prompt(f["a"], f["b"]), A_TOKEN, B_TOKEN
    raise ValueError(s)


def _continuation_logprob(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    """Return the *sum* of token-level log-probs of ``continuation`` given ``prompt``.

    We compute log-probs in one forward pass by feeding ``prompt + continuation`` and
    summing the cross-entropy on the continuation tokens only.
    """
    full = prompt + continuation
    enc_full = tokenizer(full, return_tensors="pt").to(device)
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc_prompt.input_ids.shape[1]

    with torch.no_grad():
        logits = model(**enc_full).logits  # (1, seq, vocab)
    # The token at position t is predicted by the logits at position t - 1.
    target_ids = enc_full.input_ids[0, prompt_len:]
    if target_ids.numel() == 0:
        return 0.0
    shifted_logits = logits[0, prompt_len - 1 : -1, :]
    log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp.sum().item()


def evaluate_subset(
    model,
    tokenizer,
    examples: list[EthicsExample],
    device: torch.device | None = None,
    progress: bool = True,
) -> SubsetResult:
    if not examples:
        return SubsetResult(subset="empty", accuracy=0.0, n=0, predictions=[], gold=[])

    device = device or next(model.parameters()).device
    correct = 0
    preds: list[int] = []
    gold: list[int] = []
    iterator = tqdm(examples, desc=f"eval/{examples[0].subset}") if progress else examples
    for ex in iterator:
        prompt, pos, neg = _build_prompt(ex)
        lp_pos = _continuation_logprob(model, tokenizer, prompt, pos, device)
        lp_neg = _continuation_logprob(model, tokenizer, prompt, neg, device)
        pred = 1 if lp_pos > lp_neg else 0
        preds.append(pred)
        gold.append(ex.label)
        if pred == ex.label:
            correct += 1
    n = len(examples)
    return SubsetResult(
        subset=examples[0].subset,
        accuracy=correct / n,
        n=n,
        predictions=preds,
        gold=gold,
    )


def evaluate_all(
    model,
    tokenizer,
    subsets_to_examples: dict[str, list[EthicsExample]],
    device: torch.device | None = None,
) -> dict[str, SubsetResult]:
    return {
        name: evaluate_subset(model, tokenizer, exs, device=device)
        for name, exs in subsets_to_examples.items()
    }


def summarize(results: dict[str, SubsetResult]) -> dict[str, float]:
    """Flat ``{subset: accuracy, "average": ...}`` summary, handy for JSON dumps."""
    out = {name: r.accuracy for name, r in results.items()}
    if out:
        out["average"] = sum(out.values()) / len(out)
    return out


def load_causal_lm_for_eval(
    base_model_name: str,
    adapter_path: str | None = None,
):
    """Load a causal LM (with optional LoRA adapter) for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
