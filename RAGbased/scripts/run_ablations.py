"""Run a small grid of RAG ablations on ETHICS.

Ablations covered:
  - retrieval backend: dense vs BM25
  - top_k:             {1, 3, 5, 10}
  - prompt template:   {principles, constitution, minimal}

The model is loaded once and reused across all runs (eval-only, no training),
which keeps the whole grid affordable: each run is just an ETHICS pass.
"""
from __future__ import annotations

import argparse
import copy
import gc
import itertools
from pathlib import Path

import torch

from src.data import iter_all_ethics
from src.eval import evaluate_all, summarise
from src.model import load_base_model, load_tokenizer
from src.retriever import BM25Retriever, DenseRetriever, load_retriever
from src.utils import cuda_summary, device, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)


GRID = {
    "backend": ["dense", "bm25"],
    "top_k": [1, 3, 5, 10],
    "template": ["principles", "constitution", "minimal"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--out", default="results/ablations.json")
    parser.add_argument(
        "--axes",
        default="top_k",
        help="Comma-separated list of axes to vary: backend, top_k, template. "
             "Other axes are pinned to the config defaults.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    set_seed(base_cfg["ethics"]["seed"])
    log.info(f"Hardware: {cuda_summary()}")

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for ax in axes:
        if ax not in GRID:
            raise SystemExit(f"Unknown axis '{ax}'. Valid: {list(GRID)}")

    # Build the cartesian product over the requested axes only.
    pinned = {
        "backend": [base_cfg["retrieval"]["backend"]],
        "top_k": [base_cfg["rag"]["top_k"]],
        "template": [base_cfg["rag"]["template"]],
    }
    values = {ax: (GRID[ax] if ax in axes else pinned[ax]) for ax in GRID}
    combos = list(itertools.product(values["backend"], values["top_k"], values["template"]))
    log.info(f"Ablation grid (over {axes}): {len(combos)} runs")

    # Load model + tokenizer once.
    tokenizer = load_tokenizer(base_cfg["model"])
    model = load_base_model(base_cfg["model"])
    examples_by_cat = iter_all_ethics(
        base_cfg["ethics"]["categories"],
        base_cfg["ethics"]["samples_per_category"],
        base_cfg["ethics"]["seed"],
    )

    # Cache retrievers across runs (instantiating dense is the expensive part).
    retriever_cache: dict[str, object] = {}

    def get_retriever(backend: str):
        if backend in retriever_cache:
            return retriever_cache[backend]
        if backend == "dense":
            idx_dir = base_cfg["retrieval"]["index_dir"]
            if Path(idx_dir).exists():
                r = DenseRetriever.load(idx_dir, device=base_cfg["retrieval"].get("encoder_device", "cpu"))
            else:
                log.info(f"No dense index at {idx_dir}; building from scratch")
                r = DenseRetriever.build(
                    corpus_dir=base_cfg["corpus"]["dir"],
                    encoder_name=base_cfg["retrieval"]["encoder_name"],
                    device=base_cfg["retrieval"].get("encoder_device", "cpu"),
                )
        elif backend == "bm25":
            r = BM25Retriever.build(base_cfg["corpus"]["dir"])
        else:
            raise ValueError(backend)
        retriever_cache[backend] = r
        return r

    runs = []
    for backend, top_k, template in combos:
        tag = f"backend={backend}_k={top_k}_tmpl={template}"
        log.info(f"=== Run {tag} ===")
        retriever = get_retriever(backend)

        results = evaluate_all(
            model, tokenizer, examples_by_cat,
            device=device(),
            retriever=retriever,
            top_k=top_k,
            template_name=template,
        )
        summary = summarise(results)
        log.info(f"{tag}: {summary}")

        runs.append({
            "tag": tag,
            "backend": backend,
            "top_k": top_k,
            "template": template,
            "summary": summary,
        })
        save_json({"runs": runs}, args.out)
        log.info(f"Updated {args.out}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
