# RAG-Based Ethical Alignment of a Small Language Model

CPU-only study of **retrieval-augmented generation as an alignment technique**
on the [ETHICS benchmark](https://arxiv.org/abs/2008.02275), using
**Qwen2.5-1.5B-Instruct in fp32** via Hugging Face transformers.

> *M1 Advanced Deep Learning — Final project*

---

## TL;DR

- **No fine-tuning.** No DPO, no RLHF, no LoRA. Model weights are never touched.
- **A ~50-principle ethical "constitution"** indexed with FAISS over MiniLM embeddings.
- **Routed retrieval**: each ETHICS subset queries only the families relevant to its
  decision (home family + universal); the Top-K=4 principles are placed at the **end of
  the user turn**, right before the answer — where they actually move the scored token.
- Two reasoning modes: fast single-token **`score`** and reason-then-score **`cot`**.
- A **family-removal ablation** (`rag_ablate`) checks the retriever is doing real work.
- Evaluated on **100 examples × 5 ETHICS subsets = 500 total**.
- Runs end-to-end on a **free Colab CPU**.
- **4–8 hours total**, fully **resumable** if Colab disconnects (per-example checkpointing).

> **Why the gain is now larger than the original ~1pp.** In the first version, three
> abstract principles sat in the *system* prompt and the answer was read off the very next
> token — too far from the decision, and often off-topic, to matter. Three changes fix that:
> (1) **routed retrieval** so injected principles are on-topic per subset;
> (2) **placement at the end of the user turn** so recency lets them condition the answer
> token; and (3) an optional **CoT** mode that lets the model reason over the principles
> before answering. All toggled in `src/config.py`.

---

## How to run it (Colab)

1. Upload this whole project as a ZIP to your Google Drive.
2. Open `notebooks/full_pipeline.ipynb` in Colab.
3. Set runtime to **CPU** (Runtime → Change runtime type → CPU).
4. Run the cells top to bottom. Read the comments — there's a smoke test
   before the long run, and a progress monitor you can call from a second cell.

If Colab disconnects mid-run, just re-run the same cells. Checkpointing
picks up where you left off.

---

## Repository layout

```
adl_rag/
├── src/
│   ├── config.py          All hyperparameters
│   ├── llm.py             Qwen2.5-1.5B-Instruct in fp32 (transformers)
│   ├── prompts.py         ChatML formatting + baseline/RAG prompts (+ CoT helpers)
│   ├── rag.py             ~50-principle corpus + family-routed FAISS retriever
│   ├── ethics_eval.py     ETHICS evaluation WITH CHECKPOINTING
│   └── run_all.py         End-to-end orchestrator
├── scripts/
│   └── make_figures.py    Bar chart, delta chart, LaTeX results table
├── notebooks/
│   └── full_pipeline.ipynb   ← USE THIS in Colab
├── report/
│   ├── main.tex           ACL-format 8-page report
│   └── references.bib
├── outputs/               (generated)
│   ├── results/           one JSON per condition
│   └── checkpoints/       per-example JSONL for resume
├── figures/               (generated)
└── requirements.txt
```

---

## Manual execution (if you don't want the notebook)

```bash
pip install -r requirements.txt

# Build the RAG index (one-time, takes ~30 sec)
python -m src.rag --build_index

# Quick smoke test (~5 min) — baseline + RAG on 5 examples/subset
python -m src.run_all --quick

# Smoke test of EVERYTHING (baseline, rag, +CoT, +ablation)
python -m src.run_all --quick --all

# Clear smoke-test outputs before the real run
rm -rf outputs/

# Full run — baseline + RAG (score mode), 4 to 8 hours of CPU work
python -m src.run_all

# Full run including the CoT comparison and the ablation (longer)
python -m src.run_all --all

# Make figures
python scripts/make_figures.py
```

### Run conditions produced

| Tag             | Mode      | Reasoning | Notes                                   | Enabled by        |
|-----------------|-----------|-----------|-----------------------------------------|-------------------|
| `baseline`      | no RAG    | score     | minimal system prompt                   | always            |
| `rag`           | RAG       | score     | routed retrieval, guidelines in user    | always            |
| `baseline_cot`  | no RAG    | cot       | reason-then-score, no principles        | `--with_cot`      |
| `rag_cot`       | RAG       | cot       | reason over retrieved principles        | `--with_cot`      |
| `rag_ablate`    | RAG       | score     | each subset's home family removed       | `--with_ablation` |

`--all` = `--with_cot --with_ablation`. Each tag writes
`outputs/results/<tag>.json` and is skipped on re-run if it already exists
(use `--force` to recompute).

### Tuning the RAG behaviour

All levers live in `src/config.py`:

- `RAG_TOP_K` — number of principles injected (default 4).
- `RAG_FAMILY_ROUTING` / `SUBSET_TO_FAMILIES` — which families each subset may retrieve.
- `RAG_GUIDELINES_IN_USER` — `True` puts principles at the end of the user turn
  (recency, recommended); `False` reproduces the original system-prompt placement.
- `REASONING_MODE` — `"score"` or `"cot"` (the orchestrator overrides this per condition).
- `COT_MAX_NEW_TOKENS` — length of the CoT justification (default 96).

### Resuming after a crash

The same `python -m src.run_all` will detect existing checkpoints in
`outputs/checkpoints/*.jsonl` and skip the examples already done. To
force a clean re-run:

```bash
python -m src.run_all --force
```

### Running only one subset (debugging)

```bash
python -m src.ethics_eval --mode baseline --tag baseline --subsets commonsense
python -m src.ethics_eval --mode rag      --tag rag      --subsets commonsense

# CoT mode and the ablation are also exposed on the CLI:
python -m src.ethics_eval --mode rag --tag rag_cot    --reasoning cot --subsets justice
python -m src.ethics_eval --mode rag --tag rag_ablate --ablate_home_family --subsets justice

# Inspect what the routed retriever returns for a few sample queries:
python -m src.rag --demo
```

---

## Why fp32?

The assignment asked for Qwen2.5-1.5B in fp32 to keep the experiment
as faithful as possible to the original release. The trade-off:

| Dtype       | Memory  | Speed on Colab CPU      | Risk                       |
|-------------|---------|-------------------------|----------------------------|
| **fp32**    | ~6 GB   | ~2-6 s / example        | None (default, stable)     |
| bf16        | ~3 GB   | similar or slower (no AMX) | NaN / silent slowdowns  |
| int8/int4   | ~1.5 GB | 5-10× faster            | quantization noise         |

We default to fp32 and surface `TORCH_DTYPE` in `src/config.py` if you
want to experiment.

---

## Output format

After a successful run you get:

```
outputs/results/baseline.json    # baseline scores per subset + overall
outputs/results/rag.json         # RAG scores per subset + overall
outputs/results/baseline_cot.json  # (with --with_cot) baseline, reason-then-score
outputs/results/rag_cot.json       # (with --with_cot) RAG, reason-then-score
outputs/results/rag_ablate.json    # (with --with_ablation) home family removed

figures/accuracy_bars.png        # baseline vs RAG, grouped bars
figures/accuracy_delta.png       # per-subset gain horizontal bar chart
figures/results_table.tex        # LaTeX booktabs table for the report
```

Each results JSON now also records the exact run configuration under a
`"config"` key (reasoning mode, guideline placement, routing, top-K), so
results are self-documenting. The figure script reads `baseline.json` and
`rag.json`; point it at other tags by editing the `_load(...)` calls if you
want to chart the CoT or ablation conditions.

---

## Why no fine-tuning baseline (DPO / RLHF)?

The assignment requires "at least two of the three approaches" (DPO /
RLHF / RAG). We focus on RAG because (a) it's the only one that runs
on CPU at this model scale, (b) it's the only one that exposes its
"alignment data" as plain readable text — auditable, modifiable,
inspectable — and (c) it produces a clean story to tell in the report
about *what alignment data does* and *what it does not*.

If you want to add DPO or RLHF later, both would need a GPU; see
`report/main.tex` §6 (Discussion) for a sketched comparison.

---

## License

Code: MIT. Report: CC-BY-4.0.
