# RLHF branch — Ethical Alignment of Small Language Models

This branch implements the **Reinforcement Learning from Human Feedback (RLHF)** track of
the project *Ethical Alignment of Small Language Models* (M1 Advanced Deep Learning).

The pipeline aligns a small (~1.5B) language model against the
[ETHICS](https://github.com/hendrycks/ethics) benchmark by:

1. Training a **reward model** on the Anthropic HH-RLHF preference pairs.
2. Optimizing a **policy** with PPO against this reward model, using LoRA adapters.
3. Evaluating the baseline vs. RLHF-aligned model on a held-out subset of ETHICS
   (commonsense, justice, deontology, virtue, utilitarianism).

The benchmark dataset (ETHICS) is **never** used for training — only for evaluation, as
mandated by the project specification.

## Base model

`Qwen/Qwen2.5-1.5B-Instruct` — a 1.5B parameter instruction-tuned model. With LoRA
adapters and 4-bit quantization (QLoRA), the whole pipeline fits on a single 16 GB GPU
(Colab T4 / Kaggle P100). PPO training is the most memory-hungry step because two copies
of the policy (active + reference) and the reward model must coexist.

## Repository layout

```
.
├── configs/                       # YAML hyperparameter files
│   └── config.yaml
├── notebooks/
│   ├── 01_reward_model_training.ipynb
│   ├── 02_ppo_training.ipynb
│   └── 03_evaluation_ethics.ipynb
├── src/                           # Reusable Python modules
│   ├── data/
│   │   ├── ethics.py              # ETHICS loader + subsetting
│   │   └── preferences.py         # HH-RLHF loader
│   ├── evaluation/
│   │   └── ethics_eval.py         # log-likelihood scoring on ETHICS
│   ├── models/
│   │   └── reward.py              # Reward model factory
│   └── utils/
│       ├── prompts.py             # Prompt templates
│       └── seed.py                # Reproducibility helpers
├── outputs/                       # Trained checkpoints (git-ignored)
├── requirements.txt
└── README.md
```

## Quick start

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

GPU is mandatory. The notebooks expect CUDA — they will refuse to run on CPU.

### 2. Run the pipeline

The three notebooks are sequential. Each one persists its outputs in `outputs/`
so the next can pick them up.

| # | Notebook                                | Output                                            |
|---|-----------------------------------------|---------------------------------------------------|
| 1 | `01_reward_model_training.ipynb`        | `outputs/reward_model/` (LoRA adapter + tokenizer) |
| 2 | `02_ppo_training.ipynb`                 | `outputs/ppo_policy/`  (LoRA adapter + tokenizer)  |
| 3 | `03_evaluation_ethics.ipynb`            | `outputs/eval/` (per-subset accuracy JSON + plots) |

You can also launch the notebooks on Google Colab — point them to a runtime with a
T4 GPU or better and they should run end-to-end in ~3–5 hours total with the default
small subsets.

## Method summary

### Reward model

* Backbone: `Qwen2.5-1.5B-Instruct` with a regression head (single logit).
* LoRA: `r=16`, `alpha=32`, dropout `0.05`, applied to all attention projections.
* Loss: Bradley–Terry pairwise — `-log σ(r_chosen − r_rejected)` (TRL `RewardTrainer`).
* Data: 10 k preference pairs sampled from `Anthropic/hh-rlhf` (`harmless-base`
  + `helpful-base` mix).

### Policy optimisation (PPO)

* Policy: `Qwen2.5-1.5B-Instruct` + LoRA (same config as reward model).
* Reference: a frozen LoRA-less copy of the policy.
* Reward: the trained reward model called on the concatenated `prompt + response`.
* Prompts: ~2 k human turns from `Anthropic/hh-rlhf` (training split, **not** ETHICS).
* PPO config: 2 epochs over the dataset, batch size 4 (gradient-accum 4 → effective 16),
  KL coefficient init 0.2 with adaptive scheduling, `mini_batch_size=2`.

### Evaluation

For every ETHICS subset we sample 100 examples (the spec asks for ~100 per task) from
the **test** split. We never train on these. For each example we build a yes/no prompt
and compare the log-likelihood of the two answers under the model — the higher one is
the prediction. Accuracy is computed against the gold label. The `utilitarianism` task
is treated as a pairwise comparison instead.

## Notes on choices

* **Why PPO and not RLOO/REINFORCE?** PPO is the canonical RLHF algorithm and the one
  most directly mentioned in the project brief. The TRL `PPOTrainer` makes the
  bookkeeping (advantages, KL penalty, clipping) explicit, which is useful pedagogy.
* **Why HH-RLHF and not UltraFeedback?** HH-RLHF is smaller (so iteration is faster on
  consumer GPUs) and its `harmless-base` slice directly targets ethical behaviour —
  the cleanest signal for the ETHICS benchmark.
* **No SFT step.** The project marks SFT as optional. Qwen2.5-1.5B-Instruct is already
  instruction-tuned, so we skip SFT and go straight to reward modeling + PPO.

## Reproducing the report numbers

The hyperparameters in `configs/config.yaml` are the ones used for the report. If you
change them, update the config file rather than the notebooks so the change is
versioned.
