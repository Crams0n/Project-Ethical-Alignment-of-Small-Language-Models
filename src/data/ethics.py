"""Loader for the ETHICS benchmark (Hendrycks et al., ICLR 2021).

The benchmark is split into five subsets, each scored as a binary classification (the
utilitarianism subset is a pairwise ranking). The HuggingFace mirror under
``hendrycks/ethics`` exposes one config per subset. We sub-sample each subset down to
roughly 100 examples (configurable) to keep evaluation tractable as the project brief
recommends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

# Names of the configs exposed by the HuggingFace dataset.
SUBSETS = ("commonsense", "justice", "deontology", "virtue", "utilitarianism")


@dataclass(frozen=True)
class EthicsExample:
    """Subset-agnostic view of one ETHICS example."""

    subset: str
    label: int            # gold label: 1 = "yes", 0 = "no" (or 1 = A wins, 0 = B wins)
    # The raw fields needed to build a prompt. Their meaning depends on `subset`.
    fields: dict[str, str]


def _row_to_example(subset: str, row: dict[str, Any]) -> EthicsExample | None:
    """Translate a raw HuggingFace row into our ``EthicsExample`` schema.

    Returns ``None`` when the row is malformed (defensive: avoid crashing eval).
    """
    if subset == "commonsense":
        # In the HF mirror commonsense uses ``input`` + ``label`` where 1 = wrong.
        text = row.get("input") or row.get("scenario")
        if not text:
            return None
        return EthicsExample(subset, int(row["label"]), {"scenario": text})

    if subset == "justice":
        text = row.get("scenario") or row.get("input")
        if not text:
            return None
        return EthicsExample(subset, int(row["label"]), {"scenario": text})

    if subset == "deontology":
        # Two columns: ``scenario`` (request) + ``excuse`` (response).
        scenario = row.get("scenario") or row.get("input")
        excuse = row.get("excuse")
        if not scenario or not excuse:
            return None
        return EthicsExample(
            subset, int(row["label"]), {"scenario": scenario, "excuse": excuse}
        )

    if subset == "virtue":
        text = row.get("scenario") or row.get("input")
        if not text:
            return None
        # The HF "virtue" split encodes the scenario+trait as one string separated by
        # " [SEP] " on some mirrors, or exposes a separate ``trait`` field on others.
        if "[SEP]" in text:
            scenario, _, trait = text.partition("[SEP]")
        else:
            scenario = text
            trait = row.get("trait", "")
        return EthicsExample(
            subset,
            int(row["label"]),
            {"scenario": scenario.strip(), "trait": trait.strip()},
        )

    if subset == "utilitarianism":
        # Pairwise: ``baseline`` is the more-pleasant scenario, ``less_pleasant`` is the
        # less-pleasant one. We balance the side so the gold answer is roughly 50/50
        # between A and B in our subsample (see ``_balance_utilitarianism``).
        a = row.get("baseline")
        b = row.get("less_pleasant")
        if not a or not b:
            return None
        # Label convention here: 1 = "A is more pleasant", 0 = "B is more pleasant".
        # Raw rows always have A=baseline (more pleasant), so label is 1 by default.
        return EthicsExample(subset, 1, {"a": a, "b": b})

    raise ValueError(f"Unknown subset: {subset!r}")


def _balance_utilitarianism(examples: list[EthicsExample], seed: int) -> list[EthicsExample]:
    """Flip half of the utilitarianism examples so the gold label is balanced."""
    import random

    rng = random.Random(seed)
    balanced: list[EthicsExample] = []
    for ex in examples:
        if rng.random() < 0.5:
            balanced.append(
                EthicsExample(
                    ex.subset,
                    label=0,
                    fields={"a": ex.fields["b"], "b": ex.fields["a"]},
                )
            )
        else:
            balanced.append(ex)
    return balanced


def load_ethics_subset(
    subset: str,
    num_samples: int = 100,
    split: str = "test",
    seed: int = 42,
) -> list[EthicsExample]:
    """Load and sub-sample one ETHICS subset.

    The default split is ``test`` — we *never* touch the training split.
    """
    if subset not in SUBSETS:
        raise ValueError(f"{subset!r} not in {SUBSETS!r}")

    # Some mirrors expose ``test_hard`` as well; we stick to the regular ``test``.
    ds: Dataset = load_dataset("hendrycks/ethics", subset, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    examples: list[EthicsExample] = []
    for row in ds:
        ex = _row_to_example(subset, row)
        if ex is not None:
            examples.append(ex)

    if subset == "utilitarianism":
        examples = _balance_utilitarianism(examples, seed=seed)
    return examples


def load_all_ethics(
    subsets: tuple[str, ...] = SUBSETS,
    num_samples: int = 100,
    seed: int = 42,
) -> dict[str, list[EthicsExample]]:
    """Convenience wrapper that loads every subset into a dict."""
    return {s: load_ethics_subset(s, num_samples=num_samples, seed=seed) for s in subsets}
