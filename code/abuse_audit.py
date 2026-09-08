#!/usr/bin/env python3
"""abuse-audit: a one-command report card for any abuse/hate-speech dataset.

This is the reusable toolkit referenced in the paper. Given a single CSV with a
text column and a binary label column, it emits the same report-card metrics the
paper computes per benchmark, using the identical normalization, near-duplicate
detection, and shortcut-score logic as code/analysis.py and code/dedup.py:

  * rows / unique texts / class balance after normalization
  * internal near-duplicate rate (distinct near-duplicate text clusters), the
    figure a user should report after deduplicating a single corpus
  * train/test near-duplicate leakage on a stratified 80/20 split, and the
    macro-F1 inflation it causes (full test vs leaked-items-removed)
  * lexical shortcut score: macro-F1 of a 50-token chi-square model divided by
    the full TF-IDF(1,2) model, with the top tokens listed

It does NOT do cross-dataset transfer (that needs >1 corpus); transfer is run by
code/analysis.py over the five bundled benchmarks. Everything here is CPU-only,
uses no paid API, and reproduces the per-dataset numbers in the paper when run on
the bundled processed CSVs (e.g. data/processed/davidson2017.csv).

Usage:
  python code/abuse_audit.py DATA.csv [--text-col text] [--label-col label_bin]
  python code/abuse_audit.py DATA.csv --json card.json   # machine-readable card

The label column may be binary already, or a free-text/multi-class column passed
with --positive-values to define the positive class (comma-separated).
"""
import os, re, html, json, argparse, sys
import numpy as np
import pandas as pd

# Reuse the EXACT pipeline functions the paper uses, so the toolkit and the
# reported numbers cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from datasketch import MinHash, MinHashLSH

K = 5
SEED = 42
URL = re.compile(r"https?://\S+|www\.\S+"); MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+"); RT = re.compile(r"^rt\b[:\s]*", re.I)


def normalize(t):
    t = html.unescape(str(t)).lower(); t = RT.sub(" ", t); t = URL.sub(" ", t)
    t = MENTION.sub("@user", t); return WS.sub(" ", t).strip()


def shingles(t):
    return {t} if len(t) < K else {t[i:i + K] for i in range(len(t) - K + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a and b) else 0.0


def leaked_mask(train_norms, test_norms, thr=0.8, num_perm=128):
    """True for each test item that exactly- or near-duplicates a train item.
    Candidates from LSH are verified by exact shingle Jaccard, so the rate is a
    conservative lower bound (a lower num_perm can only miss, never invent)."""
    uniq = list(dict.fromkeys(train_norms))
    lsh = MinHashLSH(threshold=thr, num_perm=num_perm)
    mh = {}
    for i, t in enumerate(uniq):
        m = MinHash(num_perm=num_perm); m.update_batch([s.encode() for s in shingles(t)])
        lsh.insert(str(i), m); mh[i] = shingles(t)
    trainset = set(uniq)
    out = []
    for t in test_norms:
        if t in trainset:
            out.append(True); continue
        q = MinHash(num_perm=num_perm); q.update_batch([s.encode() for s in shingles(t)])
        st = shingles(t); hit = False
        for c in lsh.query(q):
            if jaccard(st, mh[int(c)]) >= thr:
                hit = True; break
        out.append(hit)
    return np.array(out)


def internal_neardup_clusters(norms, thr=0.8, num_perm=128):
    """Number of distinct near-duplicate text clusters and the share of unique
    texts that belong to one (the residual a single-corpus deduplicator removes)."""
    uniq = list(dict.fromkeys(norms))
    sh = {i: shingles(t) for i, t in enumerate(uniq)}
    lsh = MinHashLSH(threshold=thr, num_perm=num_perm)
    mh = {}
    for i, t in enumerate(uniq):
        m = MinHash(num_perm=num_perm); m.update_batch([s.encode() for s in sh[i]])
        lsh.insert(str(i), m); mh[i] = m
    parent = list(range(len(uniq)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    in_dup = set()
    for i in range(len(uniq)):
        for c in lsh.query(mh[i]):
            j = int(c)
            if j != i and jaccard(sh[i], sh[j]) >= thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                in_dup.add(i); in_dup.add(j)
    clusters = {find(i) for i in in_dup}
    return len(uniq), len(in_dup), len(clusters)


def fit_eval(tr_text, tr_y, te_text, te_y, ngram=(1, 2)):
    vec = TfidfVectorizer(ngram_range=ngram, min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text); Xte = vec.transform(te_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, tr_y)
    return f1_score(te_y, clf.predict(Xte), average="macro")


def shortcut_score(tr, te, k=50):
    f_full = fit_eval(tr["text"], tr["y"].values, te["text"], te["y"].values, (1, 2))
    v1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True)
    Xtr = v1.fit_transform(tr["text"]); y = tr["y"].values
    ch, _ = chi2(Xtr, y)
    vocab = np.array(v1.get_feature_names_out())
    top_tokens = vocab[np.argsort(ch)[::-1][:k]]
    v2 = TfidfVectorizer(ngram_range=(1, 1), vocabulary={t: i for i, t in enumerate(top_tokens)},
                         sublinear_tf=True)
    Xtr2 = v2.fit_transform(tr["text"]); Xte2 = v2.transform(te["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr2, y)
    f_top = f1_score(te["y"].values, clf.predict(Xte2), average="macro")
    return round(f_full, 4), round(f_top, 4), round(100 * f_top / f_full, 1), list(top_tokens[:15])


def build_card(path, text_col, label_col, positive_values):
    d = pd.read_csv(path)
    if text_col not in d.columns:
        sys.exit(f"text column '{text_col}' not in {path}; columns: {list(d.columns)}")
    if label_col not in d.columns:
        sys.exit(f"label column '{label_col}' not in {path}; columns: {list(d.columns)}")
    d = d.rename(columns={text_col: "text"})
    d["text"] = d["text"].astype(str)
    if positive_values:
        pos = set(s.strip() for s in positive_values.split(","))
        d["y"] = d[label_col].astype(str).isin(pos).astype(int)
    else:
        d["y"] = pd.to_numeric(d[label_col], errors="coerce")
        if d["y"].isna().any() or not set(d["y"].dropna().unique()) <= {0, 1}:
            sys.exit(f"label column '{label_col}' is not 0/1; pass --positive-values")
        d["y"] = d["y"].astype(int)
    d["norm"] = d["text"].map(normalize)
    d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    n_rows = len(d)
    n_uniq, n_in_dup, n_clusters = internal_neardup_clusters(d["norm"].tolist())

    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["y"])
    mask = leaked_mask(tr["norm"].tolist(), te["norm"].tolist())
    f_full = fit_eval(tr["text"], tr["y"].values, te["text"], te["y"].values)
    keep = ~mask
    f_dedup = (fit_eval(tr["text"], tr["y"].values, te["text"][keep], te["y"].values[keep])
               if keep.sum() < len(te) else f_full)
    full12, top50, ret, toks = shortcut_score(tr, te)

    return {
        "dataset": os.path.basename(path),
        "n_rows": int(n_rows),
        "n_unique_texts": int(n_uniq),
        "pct_positive": round(100 * d["y"].mean(), 1),
        "internal_neardup_clusters": int(n_clusters),
        "pct_texts_in_neardup_cluster": round(100 * n_in_dup / n_uniq, 2) if n_uniq else 0.0,
        "leaked_test_pct": round(100 * mask.mean(), 2),
        "f1_full": round(f_full, 4),
        "f1_dedup": round(f_dedup, 4),
        "leakage_inflation": round(f_full - f_dedup, 4),
        "shortcut_full_f1": full12,
        "shortcut_top50_f1": top50,
        "shortcut_retained_pct": ret,
        "shortcut_top_tokens": toks,
    }


def main():
    ap = argparse.ArgumentParser(description="Emit an abuse-dataset report card.")
    ap.add_argument("data", help="path to a dataset CSV")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label_bin")
    ap.add_argument("--positive-values", default=None,
                    help="comma-separated label values that count as positive "
                         "(use when --label-col is not already 0/1)")
    ap.add_argument("--json", default=None, help="also write the card as JSON to this path")
    args = ap.parse_args()
    card = build_card(args.data, args.text_col, args.label_col, args.positive_values)
    print(f"\nabuse-audit report card: {card['dataset']}")
    print("-" * 56)
    print(f"  rows / unique texts        : {card['n_rows']} / {card['n_unique_texts']}")
    print(f"  positive class             : {card['pct_positive']}%")
    print(f"  near-dup text clusters      : {card['internal_neardup_clusters']} "
          f"({card['pct_texts_in_neardup_cluster']}% of texts in a cluster)")
    print(f"  train/test leakage          : {card['leaked_test_pct']}% of test items")
    print(f"  macro-F1 full / dedup       : {card['f1_full']} / {card['f1_dedup']} "
          f"(leakage inflation {card['leakage_inflation']:+})")
    print(f"  shortcut score (50 tokens)  : {card['shortcut_retained_pct']}% retained "
          f"(full {card['shortcut_full_f1']}, top-50 {card['shortcut_top50_f1']})")
    print(f"  top tokens                  : {' '.join(card['shortcut_top_tokens'])}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(card, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
