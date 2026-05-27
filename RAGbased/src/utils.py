"""Shared utilities: seed, IO, logging, device introspection.

Duplicated from the DPO part on purpose so RAGbased/ stays a self-contained
mini-project that can be run from inside its own folder.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable — running on CPU."
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name}, {total:.1f} GB VRAM"
