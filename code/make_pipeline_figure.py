#!/usr/bin/env python3
"""Draw the audit pipeline schematic.

One figure showing how the five raw corpora become the four measurements the
paper reports, so a reader can see where each number in the results section
comes from.

Counts in the boxes are read from the frozen artifacts rather than typed, so
the figure cannot drift away from the tables.

Run:  python3 code/make_pipeline_figure.py
Out:  paper/figures/fig_pipeline.png  and  paper/camera_ready/figures/
"""
from __future__ import annotations

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "data", "results")

plt.rcParams.update({"font.size": 7.2, "figure.dpi": 300})

INK = "#1a1a1a"
BOX = "#f4f4f4"
EDGE = "#8a8a8a"
ACCENT = "#d9e6f2"


def facts():
    nodes = pd.read_csv(os.path.join(RES, "nodes.csv"))
    pairs = pd.read_csv(os.path.join(RES, "near_dup_pairs.csv"))
    pairs["cross_dataset"] = pairs["cross_dataset"].astype(str).str.lower().eq("true")
    il = pd.read_csv(os.path.join(RES, "label_conflict_itemlevel.csv"))
    r8 = il[il["threshold"] == 0.8].iloc[0]
    r5 = il[il["threshold"] == 0.5].iloc[0]
    cross8 = pairs[(pairs["cross_dataset"]) & (pairs["jaccard"] >= 0.8)]
    return {
        "n_rows": int(nodes["n_rows"].sum()),
        "n_nodes": len(nodes),
        "cross8": len(cross8),
        "clean8_n": int(r8["n_nodes_clean"]),
        "clean8_c": int(r8["n_conflicted_clean"]),
        "clean5_n": int(r5["n_nodes_clean"]),
        "clean5_rate": float(r5["item_rate_clean_pct"]),
    }


FIG_W, FIG_H = 3.20, 3.75      # inches; single ACL column is about 3.2in
TSIZE, BSIZE = 7.0, 6.4        # title / body point size
LEAD = 1.30                    # line leading multiplier
PAD_PT = 3.0                   # vertical padding inside a box, points


def pt2ax(pt):
    """Points to axes fraction on the y axis."""
    return pt / (FIG_H * 72.0)


def box_height(n_body):
    return pt2ax(2 * PAD_PT + TSIZE * LEAD + n_body * BSIZE * LEAD)


def box(ax, x, y, w, title, lines, fc=BOX):
    """Box sized to its content: bold title line, then the body lines."""
    h = box_height(len(lines))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=0.7, edgecolor=EDGE, facecolor=fc, zorder=2))
    cx = x + w / 2
    ty = y + h - pt2ax(PAD_PT)
    ax.text(cx, ty, title, ha="center", va="top", zorder=3,
            color=INK, fontweight="bold", fontsize=TSIZE)
    if lines:
        ax.text(cx, ty - pt2ax(TSIZE * LEAD), "\n".join(lines), ha="center", va="top",
                zorder=3, color=INK, fontsize=BSIZE, linespacing=LEAD)
    return h


def arrow(ax, x, y_top, y_bot):
    ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot), arrowstyle="-|>", mutation_scale=6,
                                 linewidth=0.7, color=EDGE, zorder=1, shrinkA=0, shrinkB=0))


def main() -> int:
    f = facts()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    L, W = 0.02, 0.96
    GAP = pt2ax(7.0)
    cw = (W - 0.03) / 2
    xl, xr = L, L + cw + 0.03

    stages = [
        ("Five public English corpora", [f"{f['n_rows']:,} posts"], ACCENT),
        ("Normalize and collapse",
         ["lowercase, strip RT and URLs,",
          "@mention $\\rightarrow$ @user; unique",
          f"(corpus, text) nodes: {f['n_nodes']:,}"], BOX),
        ("Near-duplicate detection",
         ["character 5-gram MinHash,", "exact-Jaccard verification"], BOX),
        ("Split the near-duplicate graph",
         [f"cross-corpus links at $J\\geq0.8$: {f['cross8']}",
          "within-corpus links: train/test leakage"], ACCENT),
    ]
    leaves = [
        ("Contamination", ["cross-corpus overlap", "(negligible here)"]),
        ("Label conflict", [f"{f['clean8_c']}/{f['clean8_n']} at $J\\geq0.8$",
                            f"{f['clean5_rate']:.1f}% at $J\\geq0.5$"]),
        ("Leakage", ["train/test near-dups,", "effect on macro-F1"]),
        ("Shortcut, transfer", ["top-50 $\\chi^2$ tokens,", "cross-corpus F1"]),
    ]

    # total height needed, so we can start at the top and stack downward
    total = sum(box_height(len(b)) for _, b, _ in stages) \
        + box_height(len(leaves[0][1])) + box_height(len(leaves[2][1])) \
        + box_height(0) + 6 * GAP
    y = 0.5 + total / 2

    for title, lines, fc in stages:
        h = box_height(len(lines))
        y -= h
        box(ax, L, y, W, title, lines, fc)
        if title != stages[-1][0]:
            arrow(ax, 0.5, y, y - GAP)
            y -= GAP

    for row in (0, 2):
        hh = box_height(len(leaves[row][1]))
        arrow(ax, xl + cw / 2, y, y - GAP)
        arrow(ax, xr + cw / 2, y, y - GAP)
        y -= GAP + hh
        box(ax, xl, y, cw, leaves[row][0], leaves[row][1])
        box(ax, xr, y, cw, leaves[row + 1][0], leaves[row + 1][1])

    arrow(ax, xl + cw / 2, y, y - GAP)
    arrow(ax, xr + cw / 2, y, y - GAP)
    y -= GAP + box_height(0)
    box(ax, L, y, W, "Per-dataset report card", [], ACCENT)

    out_dirs = [os.path.join(ROOT, "paper", "figures"),
                os.path.join(ROOT, "paper", "camera_ready", "figures")]
    first = None
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "fig_pipeline.png")
        if first is None:
            fig.savefig(p, bbox_inches="tight", dpi=300); first = p
        else:
            shutil.copyfile(first, p)
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
