"""Loader for the Anthropic HH-RLHF preference dataset.

HH-RLHF stores each example as two full multi-turn transcripts (`chosen` and `rejected`)
that share an identical prefix and diverge on the final assistant turn. For reward
modeling we want a single `prompt + chosen / rejected` triple, which we recover by
finding the longest common prefix of the two transcripts and treating the divergent
suffixes as the two candidate responses.
"""
from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


def _split_into_prompt_and_response(chosen: str, rejected: str) -> tuple[str, str, str]:
    """Return ``(prompt, chosen_response, rejected_response)``.

    Falls back to using the whole strings as responses with an empty prompt if the two
    transcripts share no common prefix (this is rare but defensive).
    """
    # Find the longest common character prefix.
    n = min(len(chosen), len(rejected))
    i = 0
    while i < n and chosen[i] == rejected[i]:
        i += 1
    if i == 0:
        return "", chosen, rejected

    # We want the prompt to end at a turn boundary. HH-RLHF transcripts use the literal
    # markers "\n\nHuman: " and "\n\nAssistant: " to delimit turns; we backtrack to the
    # last "Assistant:" marker that lies inside the common prefix.
    marker = "\n\nAssistant:"
    cut = chosen.rfind(marker, 0, i)
    if cut == -1:
        return chosen[:i], chosen[i:], rejected[i:]
    cut += len(marker)
    return chosen[:cut].strip(), chosen[cut:].strip(), rejected[cut:].strip()


def _format_for_reward_model(example: dict[str, Any]) -> dict[str, str]:
    prompt, chosen, rejected = _split_into_prompt_and_response(
        example["chosen"], example["rejected"]
    )
    # The TRL ``RewardTrainer`` expects "chosen" and "rejected" fields containing the
    # *full* sequence (prompt + response) — it does not concatenate for us.
    sep = "\n\n" if prompt else ""
    return {
        "prompt": prompt,
        "chosen": f"{prompt}{sep}{chosen}".strip(),
        "rejected": f"{prompt}{sep}{rejected}".strip(),
    }


def load_hh_rlhf_for_reward_model(
    num_train: int = 10_000,
    num_eval: int = 500,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Return ``(train, eval)`` reward-model splits derived from Anthropic/hh-rlhf."""
    ds = load_dataset("Anthropic/hh-rlhf")
    train = (
        ds["train"]
        .shuffle(seed=seed)
        .select(range(min(num_train, len(ds["train"]))))
        .map(_format_for_reward_model, remove_columns=ds["train"].column_names)
    )
    evalset = (
        ds["test"]
        .shuffle(seed=seed)
        .select(range(min(num_eval, len(ds["test"]))))
        .map(_format_for_reward_model, remove_columns=ds["test"].column_names)
    )
    # Drop trivially empty rows that survive the common-prefix step.
    train = train.filter(lambda r: r["chosen"] and r["rejected"])
    evalset = evalset.filter(lambda r: r["chosen"] and r["rejected"])
    return train, evalset


def load_hh_rlhf_prompts_for_rl(num_prompts: int = 2000, seed: int = 42) -> Dataset:
    """Extract just the user prompts from HH-RLHF for online RL rollouts (RLOO / PPO).

    The RL trainer does not need preference labels — only prompts to roll out
    generations for. We reuse the same dataset so the policy is optimized on the same
    distribution of human conversations the reward model was trained to score.

    The returned dataset has a single ``prompt`` column (the column name TRL 1.x
    trainers expect for the prompt-only format).
    """
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(num_prompts, len(ds))))

    def _extract(example):
        prompt, _, _ = _split_into_prompt_and_response(
            example["chosen"], example["rejected"]
        )
        # Strip the trailing "Assistant:" marker so the policy generates the assistant
        # turn itself, then keep only the last human turn — Qwen's chat template
        # expects a clean user message, not the raw HH-RLHF transcript.
        human_marker = "\n\nHuman:"
        assistant_marker = "\n\nAssistant:"
        end = prompt.rfind(assistant_marker)
        if end != -1:
            prompt = prompt[:end]
        start = prompt.rfind(human_marker)
        if start != -1:
            prompt = prompt[start + len(human_marker):].strip()
        return {"prompt": prompt}

    ds = ds.map(_extract, remove_columns=ds.column_names)
    ds = ds.filter(lambda r: 0 < len(r["prompt"]) < 2000)
    return ds


# Backwards-compatible alias (older notebook imports may reference the old name).
load_hh_rlhf_prompts_for_ppo = load_hh_rlhf_prompts_for_rl
