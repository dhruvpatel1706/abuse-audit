#!/usr/bin/env python3
"""Build-time guard: diff key numbers in paper/main.tex against the result CSVs.

A stale confidence interval once shipped in the PDF that disagreed with the
checksummed artifact ([41.5,55.2] in the text vs [42.0,55.7] in
label_conflict_itemlevel.csv). This script re-reads the artifacts and asserts
that the load-bearing numbers actually appear in main.tex, so that mismatch
cannot recur. Run it before every PDF build (and in run_all.sh).

It is intentionally conservative: it checks the conflict-rate block, the
shortcut retained percentages, the leakage table, and the transfer gap. It does
NOT try to parse the whole paper; it asserts presence of the exact strings that
must track the CSVs. Exit code 0 = all numbers found, non-zero = a mismatch.
No fabrication: every expected string is computed from a CSV/JSON on disk.
"""
import os, re, sys, json
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "data", "results")
TEX = os.path.join(BASE, "paper", "main.tex")
if len(sys.argv) > 1:  # optional: check another copy, e.g. paper/camera_ready/main.tex
    TEX = os.path.abspath(sys.argv[1])

with open(TEX, encoding="utf-8") as f:
    tex = f.read()
# collapse whitespace so "[42.0, 55.7]" and "[42.0,55.7]" both match a single pattern
tex_flat = re.sub(r"\s+", "", tex)

problems = []

def need(label, s):
    """Assert the (whitespace-insensitive) string s appears in main.tex."""
    if re.sub(r"\s+", "", s) not in tex_flat:
        problems.append(f"{label}: expected to find '{s}' in main.tex, but it is absent")

# ---- conflict rates (label_conflict_itemlevel.csv) ----
il = pd.read_csv(os.path.join(RES, "label_conflict_itemlevel.csv")).set_index("threshold")
r5, r8 = il.loc[0.5], il.loc[0.8]
# primary item-level clean rate + CI at j>=0.5
need("conflict clean j>=0.5 rate", f"{r5['item_rate_clean_pct']:.1f}")
need("conflict clean j>=0.5 CI", f"[{r5['item_rate_clean_lo']:.1f},{r5['item_rate_clean_hi']:.1f}]")
need("conflict clean j>=0.5 n", f"{int(r5['n_nodes_clean'])}")
# strict-cutoff sensitivity rate + CI at j>=0.8
need("conflict clean j>=0.8 rate", f"{r8['item_rate_clean_pct']:.1f}")
need("conflict clean j>=0.8 CI", f"[{r8['item_rate_clean_lo']:.1f},{r8['item_rate_clean_hi']:.1f}]")
# connected-component rates (residual non-independence fix)
need("component clean j>=0.5 rate", f"{r5['component_rate_clean_pct']:.1f}")
need("component clean j>=0.5 CI", f"[{r5['component_rate_clean_lo']:.1f},{r5['component_rate_clean_hi']:.1f}]")
need("component clean j>=0.8 rate", f"{r8['component_rate_clean_pct']:.1f}")
# Davidson-excluded item-level at both thresholds
need("Davidson-excl item j>=0.5", f"{r5['item_rate_davidson_excl_pct']:.1f}")
need("Davidson-excl item j>=0.8", f"{r8['item_rate_davidson_excl_pct']:.1f}")

# ---- shortcut retained percentages (refined.csv / shortcut.csv) ----
sc = pd.read_csv(os.path.join(RES, "shortcut.csv")).set_index("dataset")
for ds in sc.index:
    need(f"shortcut retained {ds}", f"{sc.loc[ds, 'retained_pct']:.1f}")

# ---- leakage table (leakage.csv) ----
lk = pd.read_csv(os.path.join(RES, "leakage.csv")).set_index("dataset")
for ds in ("tweeteval_hate", "dynahate2021"):
    need(f"leaked pct {ds}", f"{lk.loc[ds, 'leaked_pct']:.1f}")

# ---- transfer gap (transfer_gap.json) ----
with open(os.path.join(RES, "transfer_gap.json")) as f:
    tg = json.load(f)
need("transfer gap", f"{tg['gap']:.2f}")

if problems:
    print("PAPER-NUMBER GUARD FAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("paper-number guard OK: all checked numbers in main.tex match the CSVs")
