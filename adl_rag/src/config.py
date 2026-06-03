"""
Central configuration.

Every other module imports from here.
Override at the CLI with `--key=value` flags (see argparse in scripts).
"""
from typing import Dict, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Model — Qwen2.5-1.5B-Instruct in fp32 via transformers
# ────────────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"

# CPU inference. We default to fp32 because:
#   - CPU bf16/fp16 is sketchy in PyTorch (silent slowdowns or NaNs).
#   - The model is small enough (~6 GB in fp32) that Colab's 12 GB RAM fits it.
# Override to "bfloat16" if you know your CPU supports it (AMX, etc.).
TORCH_DTYPE: str = "float32"

CONTEXT_LEN: int = 1536     # room for ETHICS prompt + retrieved guidelines + CoT reasoning
                            # (the scored answer cue sits at the very end, so we must
                            #  not truncate it away — keep this comfortably large)

# ────────────────────────────────────────────────────────────────────────────
# RAG
# ────────────────────────────────────────────────────────────────────────────
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"   # 22 MB, CPU-fast
RAG_TOP_K: int = 4          # was 3; a slightly fuller, on-topic rule set helps

# ── Retrieval routing ───────────────────────────────────────────────────────
# The original setup ran one un-routed FAISS query over all 35 abstract
# principles. In practice that returned the same generic "harm/honesty"
# trio for most items, regardless of subset. We now *route* retrieval to the
# family/families that actually bear on each subset's yes/no decision (the
# subset's "home" family + the universal common ground). This raises the
# topical relevance of what gets injected — the single biggest lever for
# making RAG move the decision token.
RAG_FAMILY_ROUTING: bool = True

# subset → families eligible for retrieval (order is informational only)
SUBSET_TO_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "commonsense":    ("universal", "deontology", "justice"),
    "deontology":     ("deontology", "universal"),
    "justice":        ("justice", "universal"),
    "virtue":         ("virtue", "universal"),
    "utilitarianism": ("utility", "universal"),
}

# Each subset's "home" family — the one removed in the family-removal
# ablation (used to prove the retriever is doing the right thing).
SUBSET_HOME_FAMILY: Dict[str, str] = {
    "commonsense":    "universal",
    "deontology":     "deontology",
    "justice":        "justice",
    "virtue":         "virtue",
    "utilitarianism": "utility",
}

# ── Where the retrieved guidelines go ────────────────────────────────────────
# The decision is the log-prob of the first "yes"/"no" token right after the
# assistant marker. Principles buried in the system prompt are far (in tokens)
# from that decision point, so they barely shift it. Placing them at the END of
# the USER turn — immediately before "Answer: yes or no" — lets recency do the
# work and gives the retrieved content real leverage over the next token.
RAG_GUIDELINES_IN_USER: bool = True

# ── Reasoning mode ───────────────────────────────────────────────────────────
#   "score" : fast single-forward-pass scoring of yes/no (original protocol)
#   "cot"   : model first writes a short justification (optionally grounded in
#             the retrieved principles), then we score yes/no AFTER that
#             reasoning. ~2x slower but gives principles room to act.
REASONING_MODE: str = "score"
COT_MAX_NEW_TOKENS: int = 96      # short, to keep CPU cost bounded
COT_TEMPERATURE: float = 0.0      # greedy, deterministic

# ────────────────────────────────────────────────────────────────────────────
# ETHICS benchmark
# ────────────────────────────────────────────────────────────────────────────
ETHICS_REPO: str = "hendrycks/ethics"
ETHICS_SUBSETS: Tuple[str, ...] = (
    "commonsense",
    "deontology",
    "justice",
    "virtue",
    "utilitarianism",
)
ETHICS_PER_SUBSET: int = 100   # exactly 100 per subset = 500 total

# ────────────────────────────────────────────────────────────────────────────
# Misc
# ────────────────────────────────────────────────────────────────────────────
SEED: int = 42

# Output layout
OUTPUT_ROOT: str   = "outputs"
RAG_INDEX_OUT: str = f"{OUTPUT_ROOT}/rag_index"
RESULTS_DIR: str   = f"{OUTPUT_ROOT}/results"
CHECKPOINT_DIR: str = f"{OUTPUT_ROOT}/checkpoints"   # resume mid-run
FIGURES_DIR: str   = "figures"
