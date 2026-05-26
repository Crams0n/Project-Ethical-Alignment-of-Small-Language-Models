"""Run a single DPO training as configured by --config."""
from __future__ import annotations

import argparse

from src.train import train_dpo
from src.utils import cuda_summary, get_logger, load_config

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info(f"Hardware: {cuda_summary()}")
    log.info(f"Output dir: {cfg['training']['output_dir']}")
    adapter_path = train_dpo(cfg)
    log.info(f"Done. Adapter at {adapter_path}")


if __name__ == "__main__":
    main()
