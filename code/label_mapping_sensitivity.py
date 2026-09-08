#!/usr/bin/env python3
"""Sensitivity of the cross-dataset label-conflict rate to the binary label mapping.

The question this answers: what does the abuse/not-abuse collapse destroy, and
would the strict-cutoff conflict rate change if Davidson's "offensive but not hate"
class were mapped differently?

Two datasets merge hate with generic offensive speech into the positive class:
  davidson2017  hate 1430 + offensive 19190 -> positive; neither 4163 -> negative
  hatexplain    hatespeech 5935 + offensive 6399 -> positive; normal 7814 -> negative
The other three corpora are already hate/not-hate.

We hold the near-duplicate GRAPH fixed (it depends only on text, not labels) and
vary only the mapping, so any change is attributable to the mapping alone.

  baseline   the published mapping (hate OR offensive = positive)
  hate_only_davidson    Davidson offensive -> negative
  hate_only_both        Davidson and HateXplain offensive -> negative

The script REFUSES to report variant numbers unless it first reproduces the
published baseline in data/results/label_conflict_itemlevel.csv exactly.

Run:  python3 code/label_mapping_sensitivity.py
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "data", "results")

BOOT = 10000
SEED = 20260910
BOOT_SEED = 42   # dedup.boot_node_ci uses seed=42; match it so the baseline
                 # arm reproduces the published confidence intervals too

# normalization, copied verbatim from code/dedup.py so nodes join exactly.
# 25 Aug 2026: the RT pattern here had drifted to r"^rt\s+", which fails to strip
# "rt:" and "rt!" prefixes and left 10 nodes unjoinable. Kept in sync by the
# assertion in main().
RT = re.compile(r"^rt\b[:\s]*", re.I)   # must match code/dedup.py exactly
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+")


def normalize(t):
    t = html.unescape(str(t)).lower()
    t = RT.sub(" ", t)
    t = URL.sub(" ", t)
    t = MENTION.sub("@user", t)
    t = WS.sub(" ", t).strip()
    return t


PLACEHOLDER = "@user"
PH_TOK = re.compile(r"@user")


def nonplaceholder_chars(t):
    return len(PH_TOK.sub("", t).strip())


def placeholder_tok_frac(t):
    toks = t.split()
    if not toks:
        return 1.0
    return sum(1 for x in toks if x == PLACEHOLDER) / len(toks)


def is_artifact(t):
    return placeholder_tok_frac(t) >= 0.5 or nonplaceholder_chars(t) < 15


def load_rows():
    frames = []
    for p in sorted(glob.glob(os.path.join(PROC, "*.csv"))):
        d = pd.read_csv(p)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["norm"] = df["text"].map(normalize)
    df = df[df["norm"].str.len() > 0].copy()
    return df


def remap(df, mode):
    """Return a per-row binary label under the requested mapping."""
    lab = df["label_bin"].astype(int).copy()
    raw = df["label_raw"].astype(str)
    if mode == "baseline":
        return lab
    if mode in ("hate_only_davidson", "hate_only_both"):
        m = (df["dataset"] == "davidson2017") & (raw == "offensive")
        lab = lab.mask(m, 0)
    if mode == "hate_only_both":
        m = (df["dataset"] == "hatexplain") & (raw == "offensive")
        lab = lab.mask(m, 0)
    return lab


def build_nodes(df, mode):
    df = df.copy()
    df["_lab"] = remap(df, mode)
    g = df.groupby(["dataset", "norm"])
    nodes = g.agg(n_rows=("id", "size"),
                  label=("_lab", lambda s: int(round(s.mean()))),
                  label_mean=("_lab", "mean")).reset_index()
    nodes["nid"] = nodes.index
    return nodes


def rates(nodes, pairs, thr):
    """Item-level conflict rate at a Jaccard threshold, all nodes and clean nodes."""
    lab = dict(zip(nodes["nid"], nodes["label"]))
    norm = dict(zip(nodes["nid"], nodes["norm"]))
    sub = pairs[(pairs["jaccard"] >= thr) & (pairs["cross_dataset"])]
    neigh = defaultdict(list)
    for a, b in zip(sub["nid_a"].values, sub["nid_b"].values):
        neigh[a].append(b)
        neigh[b].append(a)
    out = {}
    for tag in ("all", "clean"):
        # Match code/dedup.py exactly. dedup restricts the NODE SET to
        # non-artifact nodes but still compares each kept node against ALL of its
        # cross-dataset neighbours, artifact ones included. Filtering the
        # neighbours too is a different (stricter) estimator: it gives 197 clean
        # nodes at J>=0.5 where the published table has 212, and the variant
        # numbers would then not be comparable to the published baseline.
        nids, conf = [], []
        for nid, ns in neigh.items():
            if tag == "clean" and is_artifact(norm[nid]):
                continue
            mine = lab[nid]
            nids.append(nid)
            conf.append(1 if any(lab[x] != mine for x in ns) else 0)
        conf = np.array(conf, dtype=float)
        n = len(conf)
        pct = 100.0 * conf.mean() if n else float("nan")
        # bootstrap-over-nodes, copied from dedup.boot_node_ci (B=10000, seed=42)
        if n:
            rng = np.random.default_rng(BOOT_SEED)
            est = np.array([conf[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
            lo, hi = 100 * np.percentile(est, 2.5), 100 * np.percentile(est, 97.5)
        else:
            lo = hi = float("nan")
        out[tag] = {"n_nodes": int(n), "n_conflicted": int(conf.sum()),
                    "rate_pct": round(pct, 1),
                    "ci_lo": round(float(lo), 1), "ci_hi": round(float(hi), 1)}
    return out


def main():
    df = load_rows()
    pairs = pd.read_csv(os.path.join(RES, "near_dup_pairs.csv"))
    pairs["cross_dataset"] = pairs["cross_dataset"].astype(str).str.lower().eq("true")

    base_nodes = build_nodes(df, "baseline")

    # ---- gate: our rebuilt baseline must match the frozen nodes.csv ----
    frozen = pd.read_csv(os.path.join(RES, "nodes.csv"))
    same_n = len(frozen) == len(base_nodes)
    merged = frozen.merge(base_nodes, on=["dataset", "norm"], suffixes=("_f", "_r"))
    same_lab = bool((merged["label_f"] == merged["label_r"]).all()) if len(merged) else False
    same_nid = bool((merged["nid_f"] == merged["nid_r"]).all()) if len(merged) else False
    print(f"[gate] frozen nodes {len(frozen)}  rebuilt {len(base_nodes)}  "
          f"joined {len(merged)}  labels_match={same_lab}  nids_match={same_nid}")
    if not (same_n and len(merged) == len(frozen) and same_lab and same_nid):
        print("[gate] FAIL: rebuilt baseline does not reproduce nodes.csv. "
              "Refusing to report variant numbers.", file=sys.stderr)
        return 2

    # ---- gate: our rebuilt baseline rates must match the published table ----
    pub = pd.read_csv(os.path.join(RES, "label_conflict_itemlevel.csv"))
    ok = True
    base_rates = {}
    for thr in (0.8, 0.5):
        r = rates(base_nodes, pairs, thr)
        base_rates[str(thr)] = r
        row = pub[np.isclose(pub["threshold"], thr)].iloc[0]
        for tag, ncol, ccol, rcol in (
            ("all", "n_nodes_all", "n_conflicted_all", "item_rate_all_pct"),
            ("clean", "n_nodes_clean", "n_conflicted_clean", "item_rate_clean_pct"),
        ):
            got, want_n, want_c, want_r = r[tag], int(row[ncol]), int(row[ccol]), float(row[rcol])
            hit = (got["n_nodes"] == want_n and got["n_conflicted"] == want_c
                   and abs(got["rate_pct"] - want_r) < 0.15)
            ok &= hit
            print(f"[gate] thr={thr} {tag:5s} rebuilt n={got['n_nodes']} c={got['n_conflicted']} "
                  f"r={got['rate_pct']}  published n={want_n} c={want_c} r={want_r}  "
                  f"{'OK' if hit else 'MISMATCH'}")
    if not ok:
        print("[gate] FAIL: rebuilt baseline rates do not reproduce the published "
              "table. Refusing to report variant numbers.", file=sys.stderr)
        return 3

    # ---- gates passed: run the variants ----
    result = {
        "_provenance": {
            "script": "code/label_mapping_sensitivity.py",
            "graph_source": "data/results/near_dup_pairs.csv (labels ignored, recomputed)",
            "bootstrap_resamples": BOOT, "seed": SEED,
            "baseline_reproduced": True,
        },
        "mappings": {},
    }
    for mode in ("baseline", "hate_only_davidson", "hate_only_both"):
        nodes = build_nodes(df, mode)
        pos = int(nodes["label"].sum())
        entry = {"n_positive_nodes": pos, "thresholds": {}}
        for thr in (0.8, 0.5):
            entry["thresholds"][str(thr)] = rates(nodes, pairs, thr)
        result["mappings"][mode] = entry
        c8 = entry["thresholds"]["0.8"]["clean"]
        c5 = entry["thresholds"]["0.5"]["clean"]
        print(f"{mode:22s} thr0.8 clean {c8['n_conflicted']}/{c8['n_nodes']} = "
              f"{c8['rate_pct']}% [{c8['ci_lo']},{c8['ci_hi']}]   "
              f"thr0.5 clean {c5['n_conflicted']}/{c5['n_nodes']} = "
              f"{c5['rate_pct']}% [{c5['ci_lo']},{c5['ci_hi']}]")

    out = os.path.join(RES, "label_mapping_sensitivity.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"wrote {out}")

    # --- emit the paper table so no number is typed into the manuscript
    LABELS = {
        "baseline": "Published mapping (hate or offensive $=$ abuse)",
        "hate_only_davidson": "Davidson \\textit{offensive} $\\to$ not abuse",
        "hate_only_both": "Davidson and HateXplain \\textit{offensive} $\\to$ not abuse",
    }
    rows = []
    for mode in ("baseline", "hate_only_davidson", "hate_only_both"):
        e = result["mappings"][mode]
        c8, c5 = e["thresholds"]["0.8"]["clean"], e["thresholds"]["0.5"]["clean"]
        rows.append(
            f"{LABELS[mode]} & {c8['n_conflicted']}/{c8['n_nodes']} & "
            f"{c8['rate_pct']} \\scriptsize{{[{c8['ci_lo']}, {c8['ci_hi']}]}} & "
            f"{c5['n_conflicted']}/{c5['n_nodes']} & "
            f"{c5['rate_pct']} \\scriptsize{{[{c5['ci_lo']}, {c5['ci_hi']}]}} \\\\")
    tex = ("% GENERATED by code/label_mapping_sensitivity.py -- do not edit by hand.\n"
           "\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n"
           "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
           "& \\multicolumn{2}{c}{$J \\geq 0.8$} & \\multicolumn{2}{c}{$J \\geq 0.5$} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
           "Binary mapping & $n$ & Rate \\% & $n$ & Rate \\% \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{Sensitivity of the clean item-level cross-dataset conflict rate to the "
           "binary label mapping, with the near-duplicate graph held fixed so only the mapping "
           "varies. Bootstrap 95\\% intervals in brackets. Generated by "
           "\\texttt{code/label\\_mapping\\_sensitivity.py}, which refuses to run unless it "
           "first reproduces the published baseline row exactly.}\n"
           "\\label{tab:mapping}\n\\end{table*}\n")
    tex_out = os.path.join(ROOT, "paper", "camera_ready", "table_label_mapping.tex")
    os.makedirs(os.path.dirname(tex_out), exist_ok=True)
    with open(tex_out, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"wrote {tex_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
