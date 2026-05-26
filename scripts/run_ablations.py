"""Run a small grid of DPO ablations and evaluate each on ETHICS.

Ablations covered (kept small for compute budget):
  - DPO beta in {0.05, 0.1, 0.3}
  - learning rate in {1e-6, 5e-6, 2e-5}
  - training set size in {5000, 20000}

For your report you can pick a subset (e.g. fix β=0.1, vary LR and size).
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

from src.data import iter_all_ethics
from src.eval import evaluate_all, summarise
from src.model import load_adapter, load_base_model, load_tokenizer
from src.train import train_dpo
from src.utils import cuda_summary, device, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)


GRID = {
    "beta": [0.05, 0.1, 0.3],
    "learning_rate": [5.0e-6],            # extend if compute allows
    "max_train_samples": [20000],          # extend if compute allows
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--out", default="results/ablations.json")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    log.info(f"Hardware: {cuda_summary()}")

    tokenizer = load_tokenizer(base_cfg["model"])
    examples_by_cat = iter_all_ethics(
        base_cfg["ethics"]["categories"],
        base_cfg["ethics"]["samples_per_category"],
        base_cfg["ethics"]["seed"],
    )

    runs = []
    combos = list(itertools.product(GRID["beta"], GRID["learning_rate"], GRID["max_train_samples"]))
    log.info(f"Total ablation runs: {len(combos)}")

    for beta, lr, n_samples in combos:
        tag = f"beta{beta}_lr{lr}_n{n_samples}"
        cfg = copy.deepcopy(base_cfg)
        cfg["dpo"]["beta"] = beta
        cfg["training"]["learning_rate"] = lr
        cfg["data"]["max_train_samples"] = n_samples
        cfg["training"]["output_dir"] = f"./checkpoints/ablation_{tag}"

        log.info(f"=== Run {tag} ===")
        adapter_path = train_dpo(cfg)

        base = load_base_model(cfg["model"], for_training=False)
        model = load_adapter(base, adapter_path)
        set_seed(cfg["ethics"]["seed"])
        results = evaluate_all(model, tokenizer, examples_by_cat, device=device())
        summary = summarise(results)
        log.info(f"{tag}: {summary}")

        runs.append({
            "tag": tag,
            "beta": beta,
            "learning_rate": lr,
            "max_train_samples": n_samples,
            "adapter": adapter_path,
            "summary": summary,
        })

        save_json({"runs": runs}, args.out)
        log.info(f"Updated {args.out}")

        # Free VRAM between runs.
        del model, base
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
