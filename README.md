# RLHF branch — Ethical Alignment of Small Language Models

This branch implements the **Reinforcement Learning from Human Feedback (RLHF)** track of
the project *Ethical Alignment of Small Language Models* (M1 Advanced Deep Learning).

The pipeline aligns a small (~1.5B) language model against the
[ETHICS](https://github.com/hendrycks/ethics) benchmark by:

1. Training a **reward model** on Anthropic HH-RLHF preference pairs.
2. Optimizing the **policy** with **RLOO** (REINFORCE Leave-One-Out — a lighter,
   value-head-free alternative to PPO) against this reward model, using LoRA adapters.
3. Evaluating the baseline vs. RLHF-aligned model on a held-out subset of ETHICS
   (commonsense, justice, deontology, virtue, utilitarianism).

The benchmark dataset (ETHICS) is **never** used for training — only for evaluation, as
mandated by the project specification.

## Why RLOO and not PPO?

TRL 1.x **removed** `PPOTrainer`. The library now exposes `RLOOTrainer` and
`GRPOTrainer` as the modern online-RL options. The project brief explicitly says
*"You may use PPO or lighter alternatives"*, so we use RLOO. The conceptual pipeline is
the same (reward model + policy gradient against it), only the variance-reduction
mechanism changes: RLOO samples K completions per prompt and uses the mean of the
other K−1 rewards as a baseline, replacing PPO's value head + GAE.

## Base model

`Qwen/Qwen2.5-1.5B-Instruct` — 1.5B parameters, instruction-tuned. With LoRA adapters
and 4-bit quantization (QLoRA), the pipeline fits on a single 16 GB GPU (Colab T4 /
Kaggle P100). The RLOO step is the most memory-hungry because the policy, a frozen
reference (materialised by disabling the LoRA adapters), and the reward model must all
sit on the GPU.

## Repository layout

```
.
├── configs/                       # YAML hyperparameter files
│   └── config.yaml
├── notebooks/
│   ├── 01_reward_model_training.ipynb
│   ├── 02_rloo_training.ipynb
│   └── 03_evaluation_ethics.ipynb
├── src/
│   ├── data/
│   │   ├── ethics.py              # ETHICS loader + subsetting
│   │   └── preferences.py         # HH-RLHF loader (preferences + RL prompts)
│   ├── evaluation/
│   │   └── ethics_eval.py         # log-likelihood scoring on ETHICS
│   ├── models/
│   │   └── reward.py              # base-model + LoRA/BnB config builders
│   └── utils/
│       ├── config.py              # YAML config loader
│       ├── prompts.py             # Prompt templates + chat formatter
│       └── seed.py                # Reproducibility helpers
├── outputs/                       # Trained checkpoints (git-ignored)
├── requirements.txt
└── README.md
```

## Environment

The notebooks are wired to the existing `fise-ml-global` Jupyter kernel
(`/home/elias/.venvs/fise-ml-global`). Dependencies installed there:

| Package         | Version (pinned in `requirements.txt`)      |
|-----------------|---------------------------------------------|
| Python          | 3.11                                        |
| `torch`         | 2.10                                        |
| `transformers`  | 5.6                                         |
| `trl`           | 1.5 (provides `RewardTrainer`, `RLOOTrainer`)|
| `peft`          | 0.19                                        |
| `bitsandbytes`  | 0.49                                        |
| `accelerate`    | 1.13                                        |
| `datasets`      | 4.8                                         |

To set up from scratch:

```bash
python3.11 -m venv ~/.venvs/fise-ml-global
~/.venvs/fise-ml-global/bin/pip install -U pip
~/.venvs/fise-ml-global/bin/pip install -r requirements.txt
~/.venvs/fise-ml-global/bin/python -m ipykernel install --user --name fise-ml-global --display-name "Python (fise-ml-global)"
```

## Running the pipeline

The three notebooks are sequential. Each persists its outputs in `outputs/`; the next
picks them up.

| # | Notebook                                | Output                                                |
|---|-----------------------------------------|-------------------------------------------------------|
| 1 | `01_reward_model_training.ipynb`        | `outputs/reward_model/` (LoRA adapter + tokenizer)    |
| 2 | `02_rloo_training.ipynb`                | `outputs/rloo_policy/`  (LoRA adapter + tokenizer)    |
| 3 | `03_evaluation_ethics.ipynb`            | `outputs/eval/` (per-subset accuracy JSON + plots)    |

A GPU is mandatory; the notebooks assert `torch.cuda.is_available()` and refuse to run
on CPU.

## Method summary

### Reward model

* Backbone: `Qwen2.5-1.5B-Instruct` with a regression head (single logit).
* LoRA: `r=16`, `alpha=32`, dropout `0.05`, applied to `q_proj/k_proj/v_proj/o_proj`.
* Loss: Bradley–Terry pairwise — `-log σ(r_chosen − r_rejected)` (TRL `RewardTrainer`).
* Data: 10 k preference pairs sampled from `Anthropic/hh-rlhf` (helpful + harmless).

### Policy optimisation (RLOO)

* Policy: `Qwen2.5-1.5B-Instruct` + LoRA (same shape as the reward model).
* Reference: TRL materialises it by disabling LoRA adapters — no extra memory cost.
* Reward: the trained reward model called on the concatenated `prompt + completion`.
* Prompts: ~2 k user turns from `Anthropic/hh-rlhf` (training split, **not** ETHICS).
* RLOO config: `num_generations=4` (K samples per prompt), KL coefficient `beta=0.04`,
  `learning_rate=1e-6`, 1 epoch.

### Evaluation

For each ETHICS subset we sample 100 examples from the **test** split. For every
example we build a yes/no prompt and compare the log-probability of `" yes"` vs `" no"`
under the model — the higher one is the prediction (standard lm-evaluation-harness
recipe). The utilitarianism subset is scored as a pairwise comparison instead, with
`" A"` vs `" B"`.

## Notes on choices

* **Why HH-RLHF and not UltraFeedback?** HH-RLHF is smaller (faster iteration on
  consumer GPUs) and its harmless-base slice directly targets ethical behaviour — the
  cleanest signal for ETHICS.
* **No SFT step.** The brief marks SFT as optional. Qwen2.5-1.5B-Instruct is already
  instruction-tuned, so we skip SFT and go straight to reward modelling + RL.
* **QLoRA everywhere.** 4-bit NF4 quantization on the frozen backbones, fp/bf16 LoRA
  on the adapters. Keeps memory under 16 GB for the heaviest step (RLOO).

## Reproducing the report numbers

All hyperparameters live in `configs/config.yaml` — change them there rather than in
the notebooks so the change is versioned.
