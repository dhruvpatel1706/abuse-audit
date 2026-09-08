#!/usr/bin/env python3
"""C1 + C4: cross-dataset contamination audit.

Finds exact and near-duplicate texts WITHIN and ACROSS the public abuse datasets,
builds a contamination matrix, runs a near-duplicate threshold sensitivity sweep,
and surfaces cross-dataset label conflicts (same/near-identical text, different label).

Outputs -> data/results/:
  nodes.csv                       one row per unique (dataset, normalized_text)
  exact_cross_dataset.csv         texts that appear verbatim in >1 dataset
  near_dup_pairs.csv              cross/within near-duplicate node pairs (jaccard>=0.5)
  contamination_matrix_t80.csv    #cross-dataset near-dup links (jaccard>=0.8) per dataset pair
  contamination_rows_t80.csv      per dataset: #rows that are near-dup of ANOTHER dataset
  threshold_sensitivity.csv       pair counts at jaccard thresholds 0.5/0.7/0.8/0.9
  label_conflict.csv              cross-dataset dup pairs with differing labels
All numbers are reproduced by this script. No fabrication.
"""
import os, re, glob, itertools, html
import numpy as np
import pandas as pd
from collections import defaultdict
from datasketch import MinHash, MinHashLSH

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "data", "results")
os.makedirs(RES, exist_ok=True)

K = 5            # char shingle size
NUM_PERM = 128
CAND_T = 0.5     # LSH candidate-generation threshold
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

# ---- load all datasets ----
rows = []
for f in sorted(glob.glob(os.path.join(PROC, "*.csv"))):
    d = pd.read_csv(f)
    d["text"] = d["text"].astype(str)
    rows.append(d[["dataset", "id", "text", "label_bin"]])
df = pd.concat(rows, ignore_index=True)
df["norm"] = df["text"].map(normalize)
df = df[df["norm"].str.len() > 0].reset_index(drop=True)
print(f"loaded {len(df)} rows across {df['dataset'].nunique()} datasets")

# ---- collapse to nodes = unique (dataset, norm) ----
g = df.groupby(["dataset", "norm"])
nodes = g.agg(n_rows=("id", "size"),
              label=("label_bin", lambda s: int(round(s.mean()))),
              label_mean=("label_bin", "mean")).reset_index()
nodes["nid"] = nodes.index
print(f"{len(nodes)} unique (dataset, text) nodes")
nodes.to_csv(os.path.join(RES, "nodes.csv"), index=False)

# map norm -> list of node rows (for exact cross-dataset detection)
norm_to_nodes = defaultdict(list)
for r in nodes.itertuples():
    norm_to_nodes[r.norm].append(r)

# ---- exact cross-dataset duplicates (same norm text in >1 dataset) ----
exact_pairs = []   # (nidA, nidB) exact (jaccard=1) across datasets
exact_rows = []
for norm, ns in norm_to_nodes.items():
    ds = set(n.dataset for n in ns)
    if len(ds) > 1:
        for a, b in itertools.combinations(ns, 2):
            if a.dataset != b.dataset:
                exact_pairs.append((a.nid, b.nid, 1.0))
        exact_rows.append({"norm": norm, "datasets": ",".join(sorted(ds)),
                           "labels": ",".join(str(n.label) for n in ns),
                           "total_rows": sum(n.n_rows for n in ns)})
pd.DataFrame(exact_rows).to_csv(os.path.join(RES, "exact_cross_dataset.csv"), index=False)
print(f"exact cross-dataset duplicate texts: {len(exact_rows)} (yielding {len(exact_pairs)} cross node pairs)")

# ---- MinHash over unique node texts ----
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

# ---- candidate near-dup pairs via LSH, exact jaccard via shingles ----
print("querying LSH for candidate pairs ...")
nid_meta = {r.nid: (r.dataset, r.label, r.n_rows) for r in nodes.itertuples()}
seen = set()
pairs = []   # (nidA,nidB,jacc)
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
# add exact cross-dataset pairs (jaccard=1) not already captured
for a, b, j in exact_pairs:
    key = (a, b) if a < b else (b, a)
    if key not in seen:
        seen.add(key)
        pairs.append((key[0], key[1], 1.0))
print(f"near/exact duplicate node pairs (jaccard>={CAND_T}): {len(pairs)}")

# ---- assemble pair table ----
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
# LSH returns candidates in a nondeterministic (set-iteration) order; sort by a
# stable key so the written file is byte-reproducible across runs. Values are
# unchanged; only row order is fixed.
pairs_df = pairs_df.sort_values(["nid_a", "nid_b"], kind="mergesort").reset_index(drop=True)
pairs_df.to_csv(os.path.join(RES, "near_dup_pairs.csv"), index=False)

# ---- threshold sensitivity ----
sens = []
for t in THRESHOLDS:
    sub = pairs_df[pairs_df["jaccard"] >= t]
    sens.append({"threshold": t, "all_pairs": len(sub),
                 "cross_dataset_pairs": int(sub["cross_dataset"].sum()),
                 "within_dataset_pairs": int((~sub["cross_dataset"]).sum()),
                 "cross_label_conflicts": int(sub["label_conflict"].sum())})
sens_df = pd.DataFrame(sens)
sens_df.to_csv(os.path.join(RES, "threshold_sensitivity.csv"), index=False)
print("\nthreshold sensitivity:\n", sens_df.to_string(index=False))

# ---- contamination matrix at t=0.8 (cross-dataset only) ----
T = 0.8
cd = pairs_df[(pairs_df["jaccard"] >= T) & (pairs_df["cross_dataset"])]
ds_list = sorted(nodes["dataset"].unique())
mat = pd.DataFrame(0, index=ds_list, columns=ds_list)
for r in cd.itertuples():
    mat.loc[r.dataset_a, r.dataset_b] += 1
    mat.loc[r.dataset_b, r.dataset_a] += 1
mat.to_csv(os.path.join(RES, "contamination_matrix_t80.csv"))
print(f"\ncontamination matrix (cross-dataset near-dup links, jaccard>={T}):\n", mat.to_string())

# ---- per-dataset contaminated ROW counts (rows that dup another dataset) ----
contam_nodes = defaultdict(set)   # dataset -> set of nids that match another dataset
for r in cd.itertuples():
    contam_nodes[r.dataset_a].add(r.nid_a)
    contam_nodes[r.dataset_b].add(r.nid_b)
crows = []
tot = nodes.groupby("dataset")["n_rows"].sum()
for d in ds_list:
    contam_rows = nodes[nodes["nid"].isin(contam_nodes[d])]["n_rows"].sum()
    crows.append({"dataset": d, "total_rows": int(tot[d]),
                  "contaminated_rows": int(contam_rows),
                  "contaminated_pct": round(100 * contam_rows / tot[d], 2)})
crows_df = pd.DataFrame(crows)
crows_df.to_csv(os.path.join(RES, "contamination_rows_t80.csv"), index=False)
print("\nper-dataset contamination (rows that near-dup ANOTHER dataset, jaccard>=0.8):\n",
      crows_df.to_string(index=False))

# ---- label conflicts (cross-dataset dup pairs, differing labels) ----
conf = pairs_df[pairs_df["label_conflict"] & (pairs_df["jaccard"] >= 0.8)].copy()
conf.to_csv(os.path.join(RES, "label_conflict.csv"), index=False)
n_cross = int((pairs_df["cross_dataset"] & (pairs_df["jaccard"] >= 0.8)).sum())
rate = (len(conf) / n_cross) if n_cross else 0.0
print(f"\ncross-dataset near-dup pairs (j>=0.8): {n_cross} | label conflicts: {len(conf)} ({rate:.1%})")

# =====================================================================
# ITEM-LEVEL (per-node) cross-dataset label-conflict analysis.
# The pair-weighted rate above double counts high-degree nodes (a node in
# k cross pairs contributes k times) and is computed over non-independent
# pairs, so a binomial/Wilson CI on pairs is anti-conservative. Here we:
#   (1) report an ITEM-LEVEL rate: a node is "conflicted" if ANY of its
#       cross-dataset near-duplicate neighbors carries a different binary
#       label; the denominator is distinct nodes that have >=1 cross neighbor;
#   (2) drop placeholder/normalization artifacts (nodes that are >=50%
#       @user mention tokens, or have <15 non-placeholder characters), so the
#       rate is not driven by strings like "@user @user" or "i hate you";
#   (3) check stability to removing the highest-degree nodes;
#   (4) give the Davidson-excluded robustness rate (the broad Davidson
#       "offensive" class is the main source of conflicts);
#   (5) replace the Wilson interval with a bootstrap-over-nodes (cluster on
#       node) 95% interval.
# Primary threshold is Jaccard>=0.8 (clean matches); 0.5 is a sensitivity row.
# No existing artifact is modified; results go to new files.
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
    """Placeholder-dominated or too-short-to-be-meaningful node text."""
    t = nid2norm[nid]
    return placeholder_tok_frac(t) >= 0.5 or nonplaceholder_chars(t) < 15

def node_view(thr, exclude_ds=None):
    """Per-node cross-dataset neighbor labels at a Jaccard threshold.
    Returns {nid: (own_label, [neighbor_labels...], degree)}."""
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
    """For each node (optionally restricted to keep_nids), 1 if any cross
    neighbor disagrees on the binary label, else 0. Returns aligned arrays."""
    nids = [n for n in view if (keep_nids is None or n in keep_nids)]
    flags = [int(any(l != view[n][0] for l in view[n][1])) for n in nids]
    return nids, flags

def boot_node_ci(flags, B=10000, seed=42):
    """Bootstrap-over-nodes (cluster on node) 95% interval for the item rate."""
    a = np.asarray(flags, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(a)
    est = np.array([a[rng.integers(0, n, n)].mean() for _ in range(B)])
    return (100 * np.percentile(est, 2.5), 100 * np.percentile(est, 97.5))

def component_flags(thr, keep_nids=None):
    """Connected-component (one independent unit per shared-text cluster) flags.

    The node-level rate still treats the two endpoints of each cross-dataset
    near-duplicate as separate observations, so a single shared text contributes
    two correlated nodes and the bootstrap-over-nodes interval is a lower bound on
    the true uncertainty. Here we collapse each connected component of the
    cross-dataset near-duplicate graph to ONE unit (the cluster of texts that are
    all near-duplicates of one another) and flag the component as conflicted if its
    member nodes span more than one binary label. Restricting to keep_nids gives the
    clean (artifact-removed) variant. Components are sorted by their smallest node id
    so the flag vector, and thus the bootstrap, is byte-reproducible.
    """
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
    # all nodes
    nids_all, fl_all = conflicted_flags(view)
    # clean (artifacts removed)
    keep_clean = {n for n in view if not is_artifact(n)}
    nids_cl, fl_cl = conflicted_flags(view, keep_clean)
    # connected-component (one independent unit per shared-text cluster)
    comp_all = component_flags(thr)
    comp_cl = component_flags(thr, keep_clean)
    lo_comp_all, hi_comp_all = boot_node_ci(comp_all)
    lo_comp_cl, hi_comp_cl = boot_node_ci(comp_cl)
    # Davidson-excluded (item-level) and pair-weighted Davidson-excluded
    view_nd = node_view(thr, exclude_ds="davidson2017")
    nids_nd, fl_nd = conflicted_flags(view_nd)
    sub = pairs_df[(pairs_df["jaccard"] >= thr) & (pairs_df["cross_dataset"])]
    sub_nd = sub[(sub["dataset_a"] != "davidson2017") & (sub["dataset_b"] != "davidson2017")]
    pw_all = (sub["label_conflict"].mean() * 100) if len(sub) else float("nan")
    pw_nd = (sub_nd["label_conflict"].mean() * 100) if len(sub_nd) else float("nan")
    # bootstrap-over-nodes CIs
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
il_df.to_csv(os.path.join(RES, "label_conflict_itemlevel.csv"), index=False)
print("\nitem-level (per-node) cross-dataset label-conflict rates:\n",
      il_df.to_string(index=False))

# top-degree-removal stability (item-level), at both thresholds
stab_rows = []
for thr in (0.8, 0.5):
    view = node_view(thr)
    nids_all, fl_all = conflicted_flags(view)
    base = (sum(fl_all), len(fl_all))
    degree = {n: view[n][2] for n in view}
    order = sorted(degree, key=lambda x: -degree[x])
    for k in (0, 1, 3, 5, 10):
        drop = set(order[:k])
        # a node is dropped if it is itself top-degree; we also must drop pairs
        # touching it, which can orphan its neighbors -- recompute the view.
        sub = pairs_df[(pairs_df["jaccard"] >= thr) & (pairs_df["cross_dataset"])]
        sub = sub[~sub["nid_a"].isin(drop) & ~sub["nid_b"].isin(drop)]
        own, neigh = {}, defaultdict(list)
        for r in sub.itertuples():
            own[r.nid_a] = r.label_a; own[r.nid_b] = r.label_b
            neigh[r.nid_a].append(r.label_b); neigh[r.nid_b].append(r.label_a)
        nv = {n: (own[n], neigh[n]) for n in own}
        fl = [int(any(l != nv[n][0] for l in nv[n][1])) for n in nv]
        stab_rows.append({"threshold": thr, "drop_top_degree": k,
                          "n_nodes": len(fl), "n_conflicted": int(sum(fl)),
                          "item_rate_pct": round(100 * sum(fl) / len(fl), 1) if fl else float("nan")})
stab_df = pd.DataFrame(stab_rows)
stab_df.to_csv(os.path.join(RES, "label_conflict_topdegree_stability.csv"), index=False)
print("\ntop-degree-removal stability (item-level conflict rate):\n",
      stab_df.to_string(index=False))

print("\nDONE. results in data/results/")
