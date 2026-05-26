"""Evaluate a DPO-trained adapter on ETHICS."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import iter_all_ethics
from src.eval import evaluate_all, summarise
from src.model import load_adapter, load_base_model, load_tokenizer
from src.utils import cuda_summary, device, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the saved LoRA adapter (e.g. checkpoints/dpo_default/final).",
    )
    parser.add_argument("--out", default="results/dpo_eval.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["ethics"]["seed"])
    log.info(f"Hardware: {cuda_summary()}")

    tokenizer = load_tokenizer(cfg["model"])
    base = load_base_model(cfg["model"], for_training=False)
    model = load_adapter(base, args.adapter)

    examples_by_cat = iter_all_ethics(
        cfg["ethics"]["categories"],
        cfg["ethics"]["samples_per_category"],
        cfg["ethics"]["seed"],
    )
    results = evaluate_all(model, tokenizer, examples_by_cat, device=device())
    summary = summarise(results)
    log.info(f"DPO summary: {summary}")

    save_json(
        {
            "model": cfg["model"]["name"],
            "adapter": args.adapter,
            "summary": summary,
            "details": {
                cat: {
                    "accuracy": r.accuracy,
                    "n": r.n,
                    "predictions": r.predictions,
                    "golds": r.golds,
                }
                for cat, r in results.items()
            },
        },
        args.out,
    )
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
