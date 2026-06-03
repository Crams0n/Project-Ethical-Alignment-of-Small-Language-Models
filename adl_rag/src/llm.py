"""
LLM wrapper
===========

Thin wrapper around Hugging Face transformers, loading
Qwen2.5-1.5B-Instruct in fp32. Runs on GPU automatically when one is
available (e.g. Kaggle's free T4), otherwise on CPU.

Two methods are exposed:

    .score_tokens(prompt, words)  → list[float]
        log-probabilities of each word's *first token* as the NEXT
        token after `prompt`. Used for ETHICS evaluation.

    .generate(prompt)             → str  (optional, for demos)

We deliberately do NOT cache `past_key_values` — every ETHICS query is
a fresh prompt, and we want each forward pass to be self-contained for
clarity. (Speed could be improved with a custom batching loop, but
this keeps the code readable.)

────────────────────────────────────────────────────────────────────────────
Performance note
────────────────────────────────────────────────────────────────────────────
On a free Colab CPU (2-vCPU Skylake, no AVX-512), Qwen2.5-1.5B fp32 runs
at roughly **2–6 seconds per ETHICS example**, so the 500-example benchmark
takes 4–8 hours. On a single GPU (e.g. Kaggle T4) the same run drops to
roughly **15–30 minutes**. Per-example checkpointing in `ethics_eval.py`
lets you resume across sessions either way.
"""
from __future__ import annotations

import math
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import config as C


# ────────────────────────────────────────────────────────────────────────────
# Dtype resolution
# ────────────────────────────────────────────────────────────────────────────

_DTYPES = {
    "float32":  torch.float32,
    "fp32":     torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
    "float16":  torch.float16,
    "fp16":     torch.float16,
}


def _torch_dtype():
    name = C.TORCH_DTYPE.lower()
    if name not in _DTYPES:
        raise ValueError(
            f"unknown TORCH_DTYPE={C.TORCH_DTYPE!r}; "
            f"expected one of {list(_DTYPES)}"
        )
    return _DTYPES[name]


# ────────────────────────────────────────────────────────────────────────────
# LLM
# ────────────────────────────────────────────────────────────────────────────

class LLM:
    """Qwen2.5-1.5B-Instruct (transformers); GPU if available, else CPU."""

    def __init__(self, model_name: str = C.MODEL_NAME):
        dtype = _torch_dtype()
        print(f"[llm] loading tokenizer  {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[llm] loading model      {model_name}  (dtype={dtype})  "
              f"— this takes 1–2 min")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Use a GPU automatically when one is available (e.g. Kaggle's free
        # T4), otherwise stay on CPU. fp32 fits easily on a 16 GB T4 (~6 GB of
        # weights) and runs ~20-40x faster than CPU. For an extra ~2x speed /
        # half the memory on GPU, set TORCH_DTYPE="float16" in config.py.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        print(f"[llm] model on device    {device}")

        self._device = next(self.model.parameters()).device
        self._name = model_name

        # Cache for first-token ids (computed once per word)
        self._token_id_cache: dict[str, int] = {}

    @property
    def name(self) -> str:
        return f"hf::{self._name} ({C.TORCH_DTYPE})"

    # ── First-token resolution ───────────────────────────────────────────

    def _first_token_id(self, word: str) -> int:
        """
        Get a sensible first-token id for `word` as it would appear
        directly after the assistant marker. We try variants with a
        leading space and capitalisation; for each we look at the
        first token. We prefer variants that tokenize to a single token.
        """
        if word in self._token_id_cache:
            return self._token_id_cache[word]

        candidates = [" " + word, " " + word.capitalize(),
                      word, word.capitalize()]
        best = None
        best_len = math.inf
        for c in candidates:
            tids = self.tokenizer.encode(c, add_special_tokens=False)
            if not tids:
                continue
            if len(tids) < best_len:
                best, best_len = tids[0], len(tids)
        if best is None:
            raise ValueError(f"could not tokenize {word!r}")
        self._token_id_cache[word] = best
        return best

    # ── Scoring ──────────────────────────────────────────────────────────

    def score_tokens(self, prompt: str, words: List[str]) -> List[float]:
        """
        Return the log-probability of each word's first token, computed
        as the next token after `prompt`.
        """
        ids = [self._first_token_id(w) for w in words]

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=C.CONTEXT_LEN,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1, :]   # (vocab,)
            logprobs = torch.log_softmax(logits, dim=-1)

        return [float(logprobs[i]) for i in ids]

    # ── Generation (optional, for demos) ─────────────────────────────────

    def generate(self, prompt: str, *, max_new_tokens: int = 128,
                 temperature: float = 0.0) -> str:
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=C.CONTEXT_LEN,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)


# ────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ────────────────────────────────────────────────────────────────────────────

def load_llm() -> LLM:
    return LLM()
