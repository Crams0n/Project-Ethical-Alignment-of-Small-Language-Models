"""Detect the runtime environment so notebooks degrade gracefully on CPU-only hosts.

Real training requires CUDA — Qwen2.5-1.5B is unusable on CPU at non-trivial dataset
sizes. But contributors often *open* the notebooks on machines without a GPU to read
the code or to do quick smoke tests. This module exposes a single ``runtime_profile``
function that:

* picks the best available device (cuda > mps > cpu),
* turns off 4-bit quantization when CUDA is unavailable (bitsandbytes is CUDA-only),
* downgrades bf16 → fp32 on CPU,
* prints a clear warning so the user does not silently run a 10-day CPU job.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch


@dataclass
class RuntimeProfile:
    device: str
    has_cuda: bool
    can_quantize: bool        # bitsandbytes 4-bit requires CUDA
    use_bf16: bool


def runtime_profile(verbose: bool = True) -> RuntimeProfile:
    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if has_cuda:
        device = "cuda"
    elif has_mps:
        device = "mps"
    else:
        device = "cpu"

    profile = RuntimeProfile(
        device=device,
        has_cuda=has_cuda,
        can_quantize=has_cuda,
        use_bf16=has_cuda,
    )

    if verbose:
        if profile.device == "cuda":
            name = torch.cuda.get_device_name(0)
            print(f"[runtime] device=cuda ({name}), 4-bit quant ON, bf16 ON")
        else:
            warnings.warn(
                f"No CUDA GPU detected — running on '{profile.device}'. "
                "Training is impractical here (Qwen-1.5B + RLOO on CPU = many days). "
                "Run these notebooks on Google Colab / Kaggle / a CUDA server. "
                "Quantization and bf16 are disabled so the code at least loads.",
                stacklevel=2,
            )
            print(f"[runtime] device={profile.device}, 4-bit quant OFF, bf16 OFF")
    return profile


def adapt_quant_cfg(quant_cfg: dict, profile: RuntimeProfile) -> dict:
    """Return a copy of ``quant_cfg`` with 4-bit disabled if quantization is unsupported."""
    if profile.can_quantize:
        return quant_cfg
    out = dict(quant_cfg)
    out["load_in_4bit"] = False
    return out
