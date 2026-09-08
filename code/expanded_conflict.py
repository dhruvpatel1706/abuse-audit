#!/usr/bin/env python3
"""ADDITIVE robustness extension: cross-dataset near-duplicate + item-level label
conflict over an EXPANDED corpus set.

This re-runs the exact overlap and item-level label-conflict logic of code/dedup.py,
but over the five frozen benchmarks PLUS the additive OLID/OffensEval-2019 corpus
(Zampieri et al. 2019). The goal is to enlarge the genuinely-near-duplicate
(Jaccard>=0.8) clean shared-item sample beyond the frozen n=18 and check whether the
Davidson-driven offensive-versus-hate conflict pattern persists or broadens.

The frozen five are read from data/processed/ (UNCHANGED); the additive corpus is read
from data/processed_expanded/. NO frozen artifact in data/results/ is overwritten:
all outputs carry the expanded_ prefix.

Outputs -> data/results/:
  expanded_near_dup_pairs.csv          cross/within near-dup node pairs (jaccard>=0.5)
  expanded_conflict.csv                item/component/pair-level conflict rates (mirrors
                                       label_conflict_itemlevel.csv schema)
  expanded_conflict_matrix_t80.csv     conflicting cross-dataset links per dataset pair (j>=0.8)
  expanded_contamination_matrix_t80.csv #cross near-dup links per dataset pair (j>=0.8)

All numbers reproduce from this script. No fabrication. The shingle/MinHash/LSH
parameters and the normalization, artifact-filter, bootstrap, and component logic are
identical to code/dedup.py so the expanded numbers are directly comparable to the frozen
five-dataset numbers.
"""
import os, re, glob, itertools, html
import numpy as np
import pandas as pd
from collections import defaultdict
from datasketch import MinHash, MinHashLSH

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
EXP = os.path.join(BASE, "data", "processed_expanded")
RES = os.path.join(BASE, "data", "results")
os.makedirs(RES, exist_ok=True)

K = 5
NUM_PERM = 128
CAND_T = 0.5
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+")
RT = re.compile(r"^rt\b[:\s]*", re.I)

def normalize(t):
    t = html.unescape(str(t)).lower()
    t = RT.sub(" ", t)
    t = URL.sub(" ", t)
    t = MENTION.sub("@user", t)
    t = WS.sub(" ", t).strip()
    return t

def shingles(t):
    if len(t) < K:
        return {t} if t else set()
    return {t[i:i + K] for i in range(len(t) - K + 1)}

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# ---- load the FROZEN five + the ADDITIVE corpora ----
files = sorted(glob.glob(os.path.join(PROC, "*.csv"))) + sorted(glob.glob(os.path.join(EXP, "*.csv")))
rows = []
for f in files:
    d = pd.read_csv(f)
    d["text"] = d["text"].astype(str)
    rows.append(d[["dataset", "id", "text", "label_bin"]])
df = pd.concat(rows, ignore_index=True)
df["norm"] = df["text"].map(normalize)
df = df[df["norm"].str.len() > 0].reset_index(drop=True)
print(f"loaded {len(df)} rows across {df['dataset'].nunique()} datasets: {sorted(df['dataset'].unique())}")

g = df.groupby(["dataset", "norm"])
nodes = g.agg(n_rows=("id", "size"),
              label=("label_bin", lambda s: int(round(s.mean()))),
              label_mean=("label_bin", "mean")).reset_index()
nodes["nid"] = nodes.index
print(f"{len(nodes)} unique (dataset, text) nodes")

norm_to_nodes = defaultdict(list)
for r in nodes.itertuples():
    norm_to_nodes[r.norm].append(r)

exact_pairs = []
for norm, ns in norm_to_nodes.items():
    ds = set(n.dataset for n in ns)
    if len(ds) > 1:
        for a, b in itertools.combinations(ns, 2):
            if a.dataset != b.dataset:
                exact_pairs.append((a.nid, b.nid, 1.0))

print("building MinHash signatures ...")
mh_by_nid = {}
sh_by_nid = {}
lsh = MinHashLSH(threshold=CAND_T, num_perm=NUM_PERM)
for i, r in enumerate(nodes.itertuples()):
    sh = shingles(r.norm)
    sh_by_nid[r.nid] = sh
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch([s.encode("utf8") for s in sh])
    mh_by_nid[r.nid] = m
    lsh.insert(str(r.nid), m)
    if (i + 1) % 20000 == 0:
        print(f"  {i+1}/{len(nodes)}")

print("querying LSH for candidate pairs ...")
nid_meta = {r.nid: (r.dataset, r.label, r.n_rows) for r in nodes.itertuples()}
seen = set()
pairs = []
for r in nodes.itertuples():
    cand = lsh.query(mh_by_nid[r.nid])
    for c in cand:
        cn = int(c)
        if cn == r.nid:
            continue
        key = (r.nid, cn) if r.nid < cn else (cn, r.nid)
        if key in seen:
            continue
        seen.add(key)
        j = jaccard(sh_by_nid[key[0]], sh_by_nid[key[1]])
        if j >= CAND_T:
            pairs.append((key[0], key[1], j))
for a, b, j in exact_pairs:
    key = (a, b) if a < b else (b, a)
    if key not in seen:
        seen.add(key)
        pairs.append((key[0], key[1], 1.0))
print(f"near/exact duplicate node pairs (jaccard>={CAND_T}): {len(pairs)}")

recs = []
for a, b, j in pairs:
    da, la, na = nid_meta[a]
    db, lb, nb = nid_meta[b]
    recs.append({"nid_a": a, "nid_b": b, "jaccard": round(j, 4),
                 "dataset_a": da, "dataset_b": db,
                 "label_a": la, "label_b": lb,
                 "cross_dataset": da != db,
                 "label_conflict": (da != db) and (la != lb),
                 "rows_a": na, "rows_b": nb})
pairs_df = pd.DataFrame(recs)
pairs_df = pairs_df.sort_values(["nid_a", "nid_b"], kind="mergesort").reset_index(drop=True)
pairs_df.to_csv(os.path.join(RES, "expanded_near_dup_pairs.csv"), index=False)

# ---- contamination matrix at t=0.8 (cross-dataset only) ----
T = 0.8
cd = pairs_df[(pairs_df["jaccard"] >= T) & (pairs_df["cross_dataset"])]
ds_list = sorted(nodes["dataset"].unique())
mat = pd.DataFrame(0, index=ds_list, columns=ds_list)
for r in cd.itertuples():
    mat.loc[r.dataset_a, r.dataset_b] += 1
    mat.loc[r.dataset_b, r.dataset_a] += 1
mat.to_csv(os.path.join(RES, "expanded_contamination_matrix_t80.csv"))

# ---- conflicting-link matrix at t=0.8 ----
cdc = cd[cd["label_conflict"]]
cmat = pd.DataFrame(0, index=ds_list, columns=ds_list)
for r in cdc.itertuples():
    cmat.loc[r.dataset_a, r.dataset_b] += 1
    cmat.loc[r.dataset_b, r.dataset_a] += 1
cmat.to_csv(os.path.join(RES, "expanded_conflict_matrix_t80.csv"))

# =====================================================================
# ITEM-LEVEL analysis (identical logic to code/dedup.py).
# =====================================================================
PLACEHOLDER = "@user"
PH_TOK = re.compile(r"@user")
nid2norm = {r.nid: r.norm for r in nodes.itertuples()}

def nonplaceholder_chars(t):
    return len(PH_TOK.sub("", t).strip())

def placeholder_tok_frac(t):
    toks = t.split()
    if not toks:
        return 1.0
    return sum(1 for x in toks if x == PLACEHOLDER) / len(toks)

def is_artifact(nid):
    t = nid2norm[nid]
    return placeholder_tok_frac(t) >= 0.5 or nonplaceholder_chars(t) < 15

def node_view(thr, exclude_ds=None):
    sub = pairs_df[(pairs_df["jaccard"] >= thr) & (pairs_df["cross_dataset"])]
    if exclude_ds is not None:
        sub = sub[(sub["dataset_a"] != exclude_ds) & (sub["dataset_b"] != exclude_ds)]
    own = {}
    neigh = defaultdict(list)
    for r in sub.itertuples():
        own[r.nid_a] = r.label_a
        own[r.nid_b] = r.label_b
        neigh[r.nid_a].append(r.label_b)
        neigh[r.nid_b].append(r.label_a)
    return {n: (own[n], neigh[n], len(neigh[n])) for n in own}

def conflicted_flags(view, keep_nids=None):
    nids = [n for n in view if (keep_nids is None or n in keep_nids)]
    flags = [int(any(l != view[n][0] for l in view[n][1])) for n in nids]
    return nids, flags

def boot_node_ci(flags, B=10000, seed=42):
    a = np.asarray(flags, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(a)
    est = np.array([a[rng.integers(0, n, n)].mean() for _ in range(B)])
    return (100 * np.percentile(est, 2.5), 100 * np.percentile(est, 97.5))

def component_flags(thr, keep_nids=None):
    sub = pairs_df[(pairs_df["jaccard"] >= thr) & (pairs_df["cross_dataset"])]
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    nodeset = set()
    for r in sub.itertuples():
        union(r.nid_a, r.nid_b)
        nodeset.add(r.nid_a)
        nodeset.add(r.nid_b)
    comp = defaultdict(list)
    for n in nodeset:
        comp[find(n)].append(n)
    flags = []
    for root in sorted(comp, key=lambda r: min(comp[r])):
        members = comp[root]
        if keep_nids is not None:
            members = [m for m in members if m in keep_nids]
        if not members:
            continue
        labels = {nid_meta[m][1] for m in members}
        flags.append(int(len(labels) > 1))
    return flags

il_rows = []
for thr in (0.8, 0.5):
    view = node_view(thr)
    nids_all, fl_all = conflicted_flags(view)
    keep_clean = {n for n in view if not is_artifact(n)}
    nids_cl, fl_cl = conflicted_flags(view, keep_clean)
    comp_all = component_flags(thr)
    comp_cl = component_flags(thr, keep_clean)
    lo_comp_all, hi_comp_all = boot_node_ci(comp_all)
    lo_comp_cl, hi_comp_cl = boot_node_ci(comp_cl)
    view_nd = node_view(thr, exclude_ds="davidson2017")
    nids_nd, fl_nd = conflicted_flags(view_nd)
    sub = pairs_df[(pairs_df["jaccard"] >= thr) & (pairs_df["cross_dataset"])]
    sub_nd = sub[(sub["dataset_a"] != "davidson2017") & (sub["dataset_b"] != "davidson2017")]
    pw_all = (sub["label_conflict"].mean() * 100) if len(sub) else float("nan")
    pw_nd = (sub_nd["label_conflict"].mean() * 100) if len(sub_nd) else float("nan")
    lo_all, hi_all = boot_node_ci(fl_all)
    lo_cl, hi_cl = boot_node_ci(fl_cl)

    def pct(num, den):
        return round(100 * num / den, 1) if den else float("nan")

    il_rows.append({
        "threshold": thr,
        "n_pairs_cross": int(len(sub)),
        "pair_rate_pct": round(pw_all, 1),
        "pair_rate_davidson_excl_pct": round(pw_nd, 1),
        "n_pairs_cross_davidson_excl": int(len(sub_nd)),
        "n_nodes_all": len(nids_all),
        "n_conflicted_all": int(sum(fl_all)),
        "item_rate_all_pct": pct(sum(fl_all), len(fl_all)),
        "item_rate_all_lo": round(lo_all, 1),
        "item_rate_all_hi": round(hi_all, 1),
        "n_nodes_clean": len(nids_cl),
        "n_conflicted_clean": int(sum(fl_cl)),
        "item_rate_clean_pct": pct(sum(fl_cl), len(fl_cl)),
        "item_rate_clean_lo": round(lo_cl, 1),
        "item_rate_clean_hi": round(hi_cl, 1),
        "n_artifact_nodes_dropped": len(nids_all) - len(nids_cl),
        "n_nodes_davidson_excl": len(nids_nd),
        "item_rate_davidson_excl_pct": pct(sum(fl_nd), len(fl_nd)),
        "n_components_all": len(comp_all),
        "n_conflicted_components_all": int(sum(comp_all)),
        "component_rate_all_pct": pct(sum(comp_all), len(comp_all)),
        "component_rate_all_lo": round(lo_comp_all, 1),
        "component_rate_all_hi": round(hi_comp_all, 1),
        "n_components_clean": len(comp_cl),
        "n_conflicted_components_clean": int(sum(comp_cl)),
        "component_rate_clean_pct": pct(sum(comp_cl), len(comp_cl)),
        "component_rate_clean_lo": round(lo_comp_cl, 1),
        "component_rate_clean_hi": round(hi_comp_cl, 1),
    })

il_df = pd.DataFrame(il_rows)
il_df.to_csv(os.path.join(RES, "expanded_conflict.csv"), index=False)
print("\nEXPANDED item-level cross-dataset label-conflict rates:\n",
      il_df.to_string(index=False))

# ---- which cross-dataset conflicting links involve OLID / Davidson (j>=0.8) ----
n_cross_t80 = int((pairs_df["cross_dataset"] & (pairs_df["jaccard"] >= 0.8)).sum())
conf_t80 = pairs_df[(pairs_df["cross_dataset"]) & (pairs_df["jaccard"] >= 0.8) & (pairs_df["label_conflict"])]
involve_dav = int(((conf_t80["dataset_a"] == "davidson2017") | (conf_t80["dataset_b"] == "davidson2017")).sum())
involve_olid = int(((conf_t80["dataset_a"] == "olid2019") | (conf_t80["dataset_b"] == "olid2019")).sum())
print(f"\nj>=0.8 cross-dataset pairs: {n_cross_t80} | conflicting links: {len(conf_t80)} "
      f"(Davidson involved: {involve_dav}, OLID involved: {involve_olid})")
print("\nDONE. expanded_* results in data/results/")
