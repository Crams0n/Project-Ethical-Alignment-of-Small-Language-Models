"""ETHICS evaluation via log-likelihood scoring of Yes/No answers."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from src.data import EthicsExample
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class EvalResult:
    category: str
    accuracy: float
    n: int
    predictions: list[int]
    golds: list[int]
    yes_logprobs: list[float]
    no_logprobs: list[float]


def _build_chat(prompt: str, answer: str, tokenizer) -> tuple[torch.Tensor, int]:
    """Build input_ids ending in `answer`; return ids and the answer token-count
    so we can extract its log-prob from logits.
    """
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
    )
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0]
    answer_len = full_ids.shape[0] - prefix_ids.shape[0]
    if answer_len <= 0:
        raise RuntimeError(
            "Chat template produced no answer tokens — check tokenizer setup."
        )
    return full_ids, answer_len


@torch.no_grad()
def _answer_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str,
) -> float:
    ids, ans_len = _build_chat(prompt, answer, tokenizer)
    ids = ids.unsqueeze(0).to(device)

    logits = model(ids).logits  # (1, T, V)
    # Token at position t is predicted from logits[t-1]. Sum log-probs of last
    # `ans_len` tokens (the answer span).
    log_probs = logits[0, :-1].log_softmax(dim=-1)  # (T-1, V)
    target_ids = ids[0, 1:]                          # (T-1,)
    token_logp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return token_logp[-ans_len:].sum().item()


def evaluate_category(
    model,
    tokenizer,
    examples: list[EthicsExample],
    device: str = "cuda",
    yes_token: str = "Yes",
    no_token: str = "No",
) -> EvalResult:
    model.eval()
    yes_lps, no_lps, preds, golds = [], [], [], []

    for ex in tqdm(examples, desc=f"eval/{examples[0].category}"):
        lp_yes = _answer_logprob(model, tokenizer, ex.prompt, yes_token, device)
        lp_no = _answer_logprob(model, tokenizer, ex.prompt, no_token, device)
        pred = 1 if lp_yes > lp_no else 0
        yes_lps.append(lp_yes)
        no_lps.append(lp_no)
        preds.append(pred)
        golds.append(ex.gold_label)

    correct = sum(int(p == g) for p, g in zip(preds, golds))
    acc = correct / len(examples)

    return EvalResult(
        category=examples[0].category,
        accuracy=acc,
        n=len(examples),
        predictions=preds,
        golds=golds,
        yes_logprobs=yes_lps,
        no_logprobs=no_lps,
    )


def evaluate_all(
    model,
    tokenizer,
    examples_by_cat: dict[str, list[EthicsExample]],
    device: str = "cuda",
) -> dict[str, EvalResult]:
    results = {}
    for cat, examples in examples_by_cat.items():
        log.info(f"Evaluating {cat} on {len(examples)} examples")
        results[cat] = evaluate_category(model, tokenizer, examples, device)
        log.info(f"  → {cat} accuracy = {results[cat].accuracy:.3f}")
    return results


def summarise(results: dict[str, EvalResult]) -> dict:
    per_cat = {cat: {"accuracy": r.accuracy, "n": r.n} for cat, r in results.items()}
    macro = sum(r.accuracy for r in results.values()) / len(results)
    return {"per_category": per_cat, "macro_avg": macro}
