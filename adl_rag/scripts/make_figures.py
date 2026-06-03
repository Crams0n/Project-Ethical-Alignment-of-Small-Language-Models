"""
Figure generation
=================

Produces (under figures/):
    accuracy_bars.png      grouped bar chart: baseline vs RAG, per subset
    accuracy_delta.png     horizontal bar chart showing the gain per subset
    results_table.tex      LaTeX booktabs table for the report

Reads outputs/results/{baseline,rag}.json — both must exist.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Allow running standalone, not just via run_all.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C   # noqa: E402


# ─── Styling ─────────────────────────────────────────────────────────────────

COL = {
    "baseline": "#94a3b8",
    "rag":      "#3b82f6",
}
NAME = {
    "baseline": "Baseline (no RAG)",
    "rag":      "RAG (routed, top-K=4)",
}
SUBSET_SHORT = {
    "commonsense":    "CS",
    "deontology":     "DE",
    "justice":        "JU",
    "virtue":         "VI",
    "utilitarianism": "UT",
}


def _load(tag: str) -> Optional[dict]:
    p = os.path.join(C.RESULTS_DIR, f"{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _strip_chrome(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


# ════════════════════════════════════════════════════════════════════════════
#  Plot 1: grouped bar chart
# ════════════════════════════════════════════════════════════════════════════

def plot_bars():
    base = _load("baseline")
    rag  = _load("rag")
    if base is None or rag is None:
        print("[fig] need both baseline.json and rag.json — skipping plot_bars")
        return

    subsets = list(C.ETHICS_SUBSETS) + ["overall"]

    def vals(r):
        return [r["per_subset"][s]["accuracy"] * 100 if s != "overall"
                else r["overall"] * 100 for s in subsets]

    v_base, v_rag = vals(base), vals(rag)
    delta = [r - b for r, b in zip(v_rag, v_base)]

    x = np.arange(len(subsets))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=150)
    b1 = ax.bar(x - w/2, v_base, w, label=NAME["baseline"],
                color=COL["baseline"], edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w/2, v_rag, w, label=NAME["rag"],
                color=COL["rag"], edgecolor="black", linewidth=0.4)

    for bars, vv in ((b1, v_base), (b2, v_rag)):
        for b, v in zip(bars, vv):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.6,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Highlight overall column
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="black", alpha=0.05, zorder=-1)

    # Annotate deltas above each pair
    for xi, d in zip(x, delta):
        ax.text(xi, max(v_base[xi], v_rag[xi]) + 5.5,
                f"{'+' if d >= 0 else ''}{d:.1f}",
                ha="center", va="bottom", fontsize=8.5,
                color=("#16a34a" if d > 0 else "#dc2626" if d < 0 else "#666"),
                weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in subsets])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.text(len(subsets) - 0.9, 51.2, "chance", color="grey", fontsize=7.5)
    ax.set_title("ETHICS benchmark — RAG-based alignment vs baseline\n"
                 "Qwen2.5-1.5B-Instruct (fp32, CPU)",
                 fontsize=10.5, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    _strip_chrome(ax)

    out = os.path.join(C.FIGURES_DIR, "accuracy_bars.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] → {out}")


# ════════════════════════════════════════════════════════════════════════════
#  Plot 2: delta chart (horizontal bars showing gain per subset)
# ════════════════════════════════════════════════════════════════════════════

def plot_delta():
    base = _load("baseline")
    rag  = _load("rag")
    if base is None or rag is None:
        return

    subsets = list(C.ETHICS_SUBSETS)
    deltas = [(rag["per_subset"][s]["accuracy"]
               - base["per_subset"][s]["accuracy"]) * 100
              for s in subsets]
    # Sort by gain
    pairs = sorted(zip(subsets, deltas), key=lambda p: p[1])
    sorted_subs, sorted_deltas = zip(*pairs)

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    colors = ["#16a34a" if d > 0 else "#dc2626" if d < 0 else "#94a3b8"
              for d in sorted_deltas]
    bars = ax.barh(range(len(sorted_subs)), sorted_deltas,
                   color=colors, edgecolor="black", linewidth=0.4)
    for b, d in zip(bars, sorted_deltas):
        ha = "left" if d >= 0 else "right"
        offset = 0.3 if d >= 0 else -0.3
        ax.text(b.get_width() + offset, b.get_y() + b.get_height()/2,
                f"{'+' if d >= 0 else ''}{d:.1f}",
                ha=ha, va="center", fontsize=9, weight="bold",
                color=colors[list(sorted_deltas).index(d)])

    ax.set_yticks(range(len(sorted_subs)))
    ax.set_yticklabels([s.capitalize() for s in sorted_subs])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Accuracy change (pp)")
    ax.set_title("Per-subset gain from RAG-based alignment",
                 fontsize=10.5, weight="bold")
    _strip_chrome(ax)

    out = os.path.join(C.FIGURES_DIR, "accuracy_delta.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] → {out}")


# ════════════════════════════════════════════════════════════════════════════
#  LaTeX table
# ════════════════════════════════════════════════════════════════════════════

def write_table():
    base = _load("baseline")
    rag  = _load("rag")
    if base is None or rag is None:
        return

    subsets = list(C.ETHICS_SUBSETS)
    runs = [("Baseline (no RAG)", base),
            ("RAG (routed, top-K=4)", rag)]

    lines = [
        r"\begin{tabular}{l" + "c" * len(subsets) + "|c}",
        r"\toprule",
        "Method & " + " & ".join(SUBSET_SHORT[s] for s in subsets) +
        r" & \textbf{Avg.} \\",
        r"\midrule",
    ]

    # Best per column for bolding
    cols = list(subsets) + ["overall"]
    best = {}
    for c in cols:
        best[c] = max(runs, key=lambda r:
                      (r[1]["per_subset"][c]["accuracy"] if c != "overall"
                       else r[1]["overall"]))[0]

    for name, r in runs:
        row = [name]
        for s in subsets:
            v = r["per_subset"][s]["accuracy"] * 100
            cell = f"{v:.1f}"
            if name == best[s]:
                cell = r"\textbf{" + cell + "}"
            row.append(cell)
        v = r["overall"] * 100
        avg = f"{v:.1f}"
        if name == best["overall"]:
            avg = r"\textbf{" + avg + "}"
        row.append(avg)
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(C.FIGURES_DIR, exist_ok=True)
    out = os.path.join(C.FIGURES_DIR, "results_table.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[fig] → {out}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(C.FIGURES_DIR, exist_ok=True)
    plot_bars()
    plot_delta()
    write_table()


if __name__ == "__main__":
    main()
