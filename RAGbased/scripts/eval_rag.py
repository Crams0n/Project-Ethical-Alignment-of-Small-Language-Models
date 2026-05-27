"""Evaluate the base LM on ETHICS *with* RAG: each example is augmented with
top-k retrieved ethical principles before Yes/No scoring.

The retriever and the language model are independent — no fine-tuning here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import iter_all_ethics
from src.eval import evaluate_all, summarise
from src.model import load_base_model, load_tokenizer
from src.retriever import load_retriever
from src.utils import cuda_summary, device, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override cfg.rag.top_k (number of retrieved chunks per query).",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Override cfg.rag.template (principles | constitution | minimal).",
    )
    parser.add_argument("--out", default="results/rag_eval.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["ethics"]["seed"])
    log.info(f"Hardware: {cuda_summary()}")

    top_k = args.top_k if args.top_k is not None else cfg["rag"]["top_k"]
    template = args.template if args.template is not None else cfg["rag"]["template"]
    log.info(f"RAG settings: backend={cfg['retrieval']['backend']}, k={top_k}, template={template}")

    index_dir = cfg["retrieval"]["index_dir"]
    if cfg["retrieval"]["backend"] == "dense" and not Path(index_dir).exists():
        raise FileNotFoundError(
            f"Dense index not found at {index_dir}. "
            f"Run `python -m scripts.build_index --config {args.config}` first."
        )

    retriever = load_retriever(
        cfg["retrieval"],
        index_dir=index_dir,
        corpus_dir=cfg["corpus"]["dir"],
        device=cfg["retrieval"].get("encoder_device", "cpu"),
    )

    tokenizer = load_tokenizer(cfg["model"])
    model = load_base_model(cfg["model"])

    examples_by_cat = iter_all_ethics(
        cfg["ethics"]["categories"],
        cfg["ethics"]["samples_per_category"],
        cfg["ethics"]["seed"],
    )

    results = evaluate_all(
        model, tokenizer, examples_by_cat,
        device=device(),
        retriever=retriever,
        top_k=top_k,
        template_name=template,
    )
    summary = summarise(results)
    log.info(f"RAG summary: {summary}")

    save_json(
        {
            "model": cfg["model"]["name"],
            "mode": "rag",
            "retrieval_backend": cfg["retrieval"]["backend"],
            "encoder_name": cfg["retrieval"].get("encoder_name"),
            "top_k": top_k,
            "template": template,
            "summary": summary,
            "details": {
                cat: {
                    "accuracy": r.accuracy,
                    "n": r.n,
                    "predictions": r.predictions,
                    "golds": r.golds,
                    "yes_logprobs": r.yes_logprobs,
                    "no_logprobs": r.no_logprobs,
                    "retrieved_ids": r.retrieved_ids,
                }
                for cat, r in results.items()
            },
        },
        args.out,
    )
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
