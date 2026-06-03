"""
End-to-end orchestrator.

Default run (in order):
    1. Baseline evaluation   (no RAG, score)          ~2-4h
    2. RAG evaluation        (routed, score)          ~2-4h
    3. Figures

Optional extra conditions (opt-in flags):
    --with_cot       adds baseline_cot + rag_cot   (reason-then-score, ~2x time)
    --with_ablation  adds rag_ablate               (home family removed)
    --all            = --with_cot --with_ablation

Every condition is checkpointed at the per-example level — if Colab
disconnects, just re-run; the script picks up where it left off.
Conditions whose results JSON already exists are skipped (use --force
to recompute).
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

from . import config as C


@contextmanager
def section(title: str):
    bar = "─" * (len(title) + 4)
    print(f"\n┌{bar}┐")
    print(f"│  {title}  │")
    print(f"└{bar}┘")
    t0 = time.time()
    yield
    print(f"[done] {title}  ·  {(time.time() - t0)/60:.1f} min elapsed")


def _have(tag: str) -> bool:
    return os.path.exists(os.path.join(C.RESULTS_DIR, f"{tag}.json"))


def main():
    ap = argparse.ArgumentParser(description="Full pipeline orchestrator.")
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoints and re-run from scratch")
    ap.add_argument("--per_subset", type=int, default=C.ETHICS_PER_SUBSET,
                    help="examples per subset (default: 100)")
    ap.add_argument("--quick", action="store_true",
                    help="quick smoke test: 5 examples per subset")
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_rag", action="store_true")
    ap.add_argument("--with_cot", action="store_true",
                    help="also run baseline_cot + rag_cot (~2x extra time)")
    ap.add_argument("--with_ablation", action="store_true",
                    help="also run rag_ablate (home family removed from retrieval)")
    ap.add_argument("--all", action="store_true",
                    help="shorthand for --with_cot --with_ablation")
    args = ap.parse_args()

    if args.all:
        args.with_cot = args.with_ablation = True

    if args.quick:
        args.per_subset = 5
        print("[quick] 5 examples per subset (25 total per condition)")

    from .llm import load_llm
    from .rag import RagWrapper
    from .ethics_eval import run_eval

    # Load the model ONCE and reuse for all conditions.
    print("\n[main] loading model (once, reused for all conditions)...")
    llm = load_llm()

    # Build/load the RAG index up-front so we can fail fast if there's
    # a network issue.
    print("\n[main] preparing RAG index...")
    rag = RagWrapper.from_disk()

    def phase(title, *, tag, mode, reasoning="score",
              ablate=False, use_rag=False):
        """Run one condition unless its result JSON already exists."""
        if args.force or not _have(tag):
            C.REASONING_MODE = reasoning           # eval reads this at call time
            with section(title):
                run_eval(mode=mode, tag=tag,
                         per_subset=args.per_subset,
                         force=args.force, llm=llm,
                         rag=(rag if use_rag else None),
                         ablate_home_family=ablate)
        else:
            print(f"[skip] {tag} results already exist")

    # ── Core comparison (fast single-token scoring) ──────────────────────
    if not args.skip_baseline:
        phase("Baseline evaluation (no RAG)",
              tag="baseline", mode="baseline", reasoning="score")
    if not args.skip_rag:
        phase("RAG evaluation (routed retrieval, guidelines in user turn)",
              tag="rag", mode="rag", reasoning="score", use_rag=True)

    # ── CoT comparison (reason first, then score) ────────────────────────
    if args.with_cot:
        if not args.skip_baseline:
            phase("Baseline + CoT", tag="baseline_cot",
                  mode="baseline", reasoning="cot")
        if not args.skip_rag:
            phase("RAG + CoT", tag="rag_cot",
                  mode="rag", reasoning="cot", use_rag=True)

    # ── Ablation: remove each subset's home family from retrieval ────────
    if args.with_ablation:
        phase("RAG ablation (home family removed)", tag="rag_ablate",
              mode="rag", reasoning="score", use_rag=True, ablate=True)

    # ── Figures ──────────────────────────────────────────────────────────
    with section("Figures"):
        try:
            from scripts.make_figures import main as make_figs
            make_figs()
        except Exception as e:
            # Try local path resolution
            import importlib.util
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mf = os.path.join(here, "scripts", "make_figures.py")
            if os.path.exists(mf):
                spec = importlib.util.spec_from_file_location("mf", mf)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.main()
            else:
                print(f"[warn] figure generation failed: {e}")


if __name__ == "__main__":
    main()
