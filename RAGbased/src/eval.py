"""ETHICS evaluation via log-likelihood scoring of Yes/No answers.

Same scoring procedure as the DPO part: we compute
    logP("Yes" | prompt)  and  logP("No" | prompt)
under the chat template and predict argmax.

The only difference is that when a retriever is supplied, the user message
is augmented with retrieved ethical principles before scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from src.data import EthicsExample, query_from_example
from src.rag import build_user_message
from src.retriever import BaseRetriever, RetrievedChunk
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
    retrieved_ids: list[list[str]]   # ids of chunks retrieved per example (RAG only)


def _build_chat(prompt: str, answer: str, tokenizer) -> tuple[torch.Tensor, int]:
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
    log_probs = logits[0, :-1].log_softmax(dim=-1)  # (T-1, V)
    target_ids = ids[0, 1:]                          # (T-1,)
    token_logp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return token_logp[-ans_len:].sum().item()


def _retrieve_for(
    ex: EthicsExample,
    retriever: BaseRetriever | None,
    k: int,
) -> list[RetrievedChunk]:
    if retriever is None or k <= 0:
        return []
    return retriever.search(query_from_example(ex), k=k)


def evaluate_category(
    model,
    tokenizer,
    examples: list[EthicsExample],
    device: str = "cuda",
    retriever: BaseRetriever | None = None,
    top_k: int = 0,
    template_name: str = "principles",
    yes_token: str = "Yes",
    no_token: str = "No",
) -> EvalResult:
    """Run the Yes/No log-likelihood eval on one ETHICS category.

    If `retriever` is None or `top_k <= 0`, this is the baseline path: the
    user message is the bare ETHICS prompt. Otherwise, retrieved chunks
    are prepended via `build_user_message`.
    """
    model.eval()
    yes_lps, no_lps, preds, golds, retrieved_ids = [], [], [], [], []

    desc_tag = f"eval/{examples[0].category}{'+rag' if retriever else ''}"
    for ex in tqdm(examples, desc=desc_tag):
        chunks = _retrieve_for(ex, retriever, top_k)
        if chunks:
            user_msg = build_user_message(ex.prompt, chunks, template_name=template_name)
        else:
            user_msg = ex.prompt

        lp_yes = _answer_logprob(model, tokenizer, user_msg, yes_token, device)
        lp_no = _answer_logprob(model, tokenizer, user_msg, no_token, device)
        pred = 1 if lp_yes > lp_no else 0

        yes_lps.append(lp_yes)
        no_lps.append(lp_no)
        preds.append(pred)
        golds.append(ex.gold_label)
        retrieved_ids.append([rc.chunk.id for rc in chunks])

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
        retrieved_ids=retrieved_ids,
    )


def evaluate_all(
    model,
    tokenizer,
    examples_by_cat: dict[str, list[EthicsExample]],
    device: str = "cuda",
    retriever: BaseRetriever | None = None,
    top_k: int = 0,
    template_name: str = "principles",
) -> dict[str, EvalResult]:
    results = {}
    for cat, examples in examples_by_cat.items():
        log.info(
            f"Evaluating {cat} on {len(examples)} examples "
            f"({'RAG k=' + str(top_k) if retriever and top_k else 'baseline'})"
        )
        results[cat] = evaluate_category(
            model,
            tokenizer,
            examples,
            device=device,
            retriever=retriever,
            top_k=top_k,
            template_name=template_name,
        )
        log.info(f"  → {cat} accuracy = {results[cat].accuracy:.3f}")
    return results


def summarise(results: dict[str, EvalResult]) -> dict:
    per_cat = {cat: {"accuracy": r.accuracy, "n": r.n} for cat, r in results.items()}
    macro = sum(r.accuracy for r in results.values()) / len(results)
    return {"per_category": per_cat, "macro_avg": macro}
