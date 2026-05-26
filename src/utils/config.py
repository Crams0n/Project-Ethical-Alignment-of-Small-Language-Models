"""Tiny YAML loader so notebooks share one source of truth for hyperparameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
