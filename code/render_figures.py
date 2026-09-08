#!/usr/bin/env python3
"""Render the shortcut and transfer figures from the frozen result files.

make_figures.py and transfer_fix.py both retrain models before plotting, so a
figure can silently fall behind the CSV it is supposed to show. This script
only reads data/results/refined.csv and data/results/transfer_matrix.csv and
draws paper/figures/fig_shortcut.png and paper/figures/fig_transfer.png from
them, so the numbers printed on the figures are the numbers in the CSVs (and
therefore the numbers checked into the paper by check_paper_numbers.py). The
title of the shortcut figure is computed from the data rather than typed.

Run:  python3 code/render_figures.py [extra output dir ...]
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "data", "results")
FIG = os.path.join(BASE, "paper", "figures")
NAMES = ["davidson2017", "dynahate2021", "hatexplain", "tweeteval_hate", "stormfront"]
SHORT = {"davidson2017": "Davidson", "dynahate2021": "DynaHate", "hatexplain": "HateXplain",
         "tweeteval_hate": "TweetEval", "stormfront": "Stormfront"}


def shortcut_figure(ref: pd.DataFrame, out_dirs: list[str]) -> None:
    ref = ref.set_index("dataset").loc[NAMES].reset_index()
    lo, hi = ref["retained_pct"].min(), ref["retained_pct"].max()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(ref))
    w = 0.38
    ax.bar(x - w / 2, ref["f1_full"], w, label="Full TF-IDF(1,2)", color="#3b6fb0")
    ax.bar(x + w / 2, ref["f1_top50"], w, label="Top-50 tokens only", color="#e08a2e")
    for i, r in ref.iterrows():
        ax.text(i, max(r["f1_full"], r["f1_top50"]) + 0.01, f"{r['retained_pct']:.0f}%",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[n] for n in ref["dataset"]], fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Lexical-shortcut reliance: 50 tokens retain {lo:.0f} to {hi:.0f}% of F1",
                 fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig_shortcut.png"), dpi=200)
    plt.close(fig)
    print("fig_shortcut.png:", ", ".join(f"{SHORT[n]} {p:.0f}%" for n, p in zip(ref["dataset"], ref["retained_pct"])))


def transfer_figure(M: pd.DataFrame, out_dirs: list[str]) -> None:
    M = M.loc[NAMES, NAMES]
    arr = M.values.astype(float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(arr, cmap="Blues", vmin=0.2, vmax=0.95)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([SHORT[c] for c in M.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([SHORT[i] for i in M.index], fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if arr[i, j] > 0.65 else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xlabel("tested on", fontsize=9)
    ax.set_ylabel("trained on", fontsize=9)
    ax.set_title("Cross-dataset transfer (macro-F1); diagonal = held-out", fontsize=9.5)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig_transfer.png"), dpi=200)
    plt.close(fig)
    print("fig_transfer.png:")
    print(M.round(3).to_string())


def main() -> int:
    out_dirs = [FIG] + [os.path.abspath(a) for a in sys.argv[1:]]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
    ref = pd.read_csv(os.path.join(RES, "refined.csv"))
    M = pd.read_csv(os.path.join(RES, "transfer_matrix.csv"), index_col=0)
    shortcut_figure(ref, out_dirs)
    transfer_figure(M, out_dirs)
    print("wrote to:", ", ".join(out_dirs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
