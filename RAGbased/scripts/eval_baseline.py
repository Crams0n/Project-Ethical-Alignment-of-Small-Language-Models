"""Evaluate the base LM on ETHICS *without* RAG. Apples-to-apples reference
point for the RAG numbers produced by eval_rag.py.
"""
from __future__ import annotations

import argparse

from src.data import iter_all_ethics
from src.eval import evaluate_all, summarise
from src.model import load_base_model, load_tokenizer
from src.utils import cuda_summary, device, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--out", default="results/baseline_eval.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["ethics"]["seed"])
    log.info(f"Hardware: {cuda_summary()}")

    tokenizer = load_tokenizer(cfg["model"])
    model = load_base_model(cfg["model"])

    examples_by_cat = iter_all_ethics(
        cfg["ethics"]["categories"],
        cfg["ethics"]["samples_per_category"],
        cfg["ethics"]["seed"],
    )

    results = evaluate_all(model, tokenizer, examples_by_cat, device=device())
    summary = summarise(results)
    log.info(f"Baseline summary: {summary}")

    save_json(
        {
            "model": cfg["model"]["name"],
            "mode": "baseline",
            "summary": summary,
            "details": {
                cat: {
                    "accuracy": r.accuracy,
                    "n": r.n,
                    "predictions": r.predictions,
                    "golds": r.golds,
                    "yes_logprobs": r.yes_logprobs,
                    "no_logprobs": r.no_logprobs,
                }
                for cat, r in results.items()
            },
        },
        args.out,
    )
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
