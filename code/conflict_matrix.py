#!/usr/bin/env python3
"""Per-dataset cross-dataset label-CONFLICT severity matrix at Jaccard >= 0.8.

Companion to the contamination (near-duplicate link) matrix produced by dedup.py.
Where contamination_matrix_t80.csv counts ALL cross-dataset near-duplicate links
per dataset pair, this counts only the links whose two endpoints carry DIFFERENT
binary labels (a label conflict). The symmetric matrix shows which dataset pairs
the definitional disagreement concentrates in.

Built directly from label_conflict.csv (the cross-dataset conflicting pairs at
Jaccard >= 0.8 written by dedup.py). The result is cross-checked against
near_dup_pairs.csv so the counts are guaranteed to match the source pairs.

Output -> data/results/conflict_matrix_t80.csv  (symmetric integer matrix).
No fabrication: every cell is a count of rows in label_conflict.csv.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "data", "results")
T = 0.8

conf = pd.read_csv(os.path.join(RES, "label_conflict.csv"))
# label_conflict.csv is already filtered to cross-dataset conflicts at j>=0.8,
# but re-assert the filter so the artifact is self-checking.
conf = conf[(conf["jaccard"] >= T) & (conf["cross_dataset"]) & (conf["label_conflict"])]

ds_list = sorted(
    set(conf["dataset_a"]).union(conf["dataset_b"]).union(
        # ensure all five datasets appear even if a row never conflicts
        {"davidson2017", "dynahate2021", "hatexplain", "stormfront", "tweeteval_hate"}
    )
)
mat = pd.DataFrame(0, index=ds_list, columns=ds_list)
for r in conf.itertuples():
    mat.loc[r.dataset_a, r.dataset_b] += 1
    mat.loc[r.dataset_b, r.dataset_a] += 1
mat.to_csv(os.path.join(RES, "conflict_matrix_t80.csv"))

# ---- cross-check against the full near-dup pair table ----
allp = pd.read_csv(os.path.join(RES, "near_dup_pairs.csv"))
cross_conf = allp[(allp["jaccard"] >= T) & (allp["cross_dataset"]) & (allp["label_conflict"])]
n_from_source = len(cross_conf)
n_from_conf_csv = len(conf)
# the matrix is symmetric with a zero diagonal, so each conflict pair is counted
# twice; half the total is the number of distinct conflicting dataset pairs.
upper_sum = int(mat.values.sum() // 2)

print("conflict-severity matrix (cross-dataset label conflicts, jaccard>=%.1f):" % T)
print(mat.to_string())
print()
print("row totals (conflicts each dataset participates in):")
print(mat.sum(axis=1).to_string())
print()
print("conflict pairs in label_conflict.csv :", n_from_conf_csv)
print("conflict pairs in near_dup_pairs.csv :", n_from_source)
print("sum of upper triangle (matrix)       :", upper_sum)
assert n_from_conf_csv == n_from_source == upper_sum, "MISMATCH: counts disagree"
print("OK: all three counts agree (%d)." % upper_sum)
print("\nwrote data/results/conflict_matrix_t80.csv")
