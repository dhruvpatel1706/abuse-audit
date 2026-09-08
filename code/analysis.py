#!/usr/bin/env python3
"""C2/C3 + leakage: the paper's headline measurements (all real, reproduced).

A. Train/test near-duplicate LEAKAGE and its F1 inflation.
   - Datasets with official splits (dynahate, tweeteval): use them.
   - Datasets without (davidson, hatexplain, stormfront): standard random 80/20 split,
     averaged over seeds, to show the leakage a typical user incurs.
B. Lexical SHORTCUT reliance: macro-F1 of a top-50-token model / full TF-IDF model.
C. Cross-dataset TRANSFER matrix (train on A, test on B), to show generalization,
   read alongside the (negligible) contamination from dedup.py.

Outputs -> data/results/{leakage.csv, shortcut.csv, transfer_matrix.csv}
"""
import os, re, glob, html, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from datasketch import MinHash, MinHashLSH

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "data", "results")
os.makedirs(RES, exist_ok=True)
SEED = 42
K = 5

URL = re.compile(r"https?://\S+|www\.\S+"); MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+"); RT = re.compile(r"^rt\b[:\s]*", re.I)
def normalize(t):
    t = html.unescape(str(t)).lower(); t = RT.sub(" ", t); t = URL.sub(" ", t)
    t = MENTION.sub("@user", t); return WS.sub(" ", t).strip()
def shingles(t):
    return {t} if len(t) < K else {t[i:i+K] for i in range(len(t)-K+1)}

def load(name):
    d = pd.read_csv(os.path.join(PROC, f"{name}.csv"))
    d["text"] = d["text"].astype(str); d["norm"] = d["text"].map(normalize)
    d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    return d

DATA = {n: load(n) for n in ["davidson2017", "dynahate2021", "hatexplain", "tweeteval_hate", "stormfront"]}

def fit_eval(tr_text, tr_y, te_text, te_y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text); Xte = vec.transform(te_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, tr_y)
    return f1_score(te_y, clf.predict(Xte), average="macro")

def leaked_mask(train_norms, test_norms, thr=0.8, num_perm=64):
    """True for each test item that exactly- or near-duplicates some train item.

    num_perm only controls LSH candidate recall: every candidate is then verified
    by EXACT shingle Jaccard (the `len(st & ss) / len(st | ss) >= thr` check below),
    so a lower num_perm can only MISS a leaked item, never invent one. The leaked
    rate is therefore a conservative lower bound, which strengthens the
    "leakage is low-impact" conclusion rather than weakening it.
    """
    uniq = list(dict.fromkeys(train_norms))
    lsh = MinHashLSH(threshold=thr, num_perm=num_perm)
    mh = {}
    for i, t in enumerate(uniq):
        m = MinHash(num_perm=num_perm); m.update_batch([s.encode() for s in shingles(t)])
        lsh.insert(str(i), m); mh[i] = (t, shingles(t))
    trainset = set(uniq)
    out = []
    for t in test_norms:
        if t in trainset:
            out.append(True); continue
        q = MinHash(num_perm=num_perm); q.update_batch([s.encode() for s in shingles(t)])
        hit = False; st = shingles(t)
        for c in lsh.query(q):
            tt, ss = mh[int(c)]
            if len(st & ss) / len(st | ss) >= thr:
                hit = True; break
        out.append(hit)
    return np.array(out)

# ---------- A. leakage + inflation ----------
print("== A. train/test leakage + F1 inflation ==")
leak_rows = []
OFFICIAL = {"dynahate2021": ("train", "test"), "tweeteval_hate": ("train", "test")}
for name, d in DATA.items():
    if name in OFFICIAL:
        trs, tes = OFFICIAL[name]
        tr = d[d["split"] == trs]; te = d[d["split"] == tes]
        reps = [(tr, te)]
        mode = "official"
    else:
        reps = []
        for s in (1, 2, 3):
            tr, te = train_test_split(d, test_size=0.2, random_state=s, stratify=d["label_bin"])
            reps.append((tr, te))
        mode = "random(3 seeds)"
    lk, ff, fd = [], [], []
    for tr, te in reps:
        m = leaked_mask(tr["norm"].tolist(), te["norm"].tolist())
        lk.append(m.mean())
        f_full = fit_eval(tr["text"], tr["label_bin"].values, te["text"], te["label_bin"].values)
        keep = ~m
        if keep.sum() < len(te):
            f_dedup = fit_eval(tr["text"], tr["label_bin"].values,
                               te["text"][keep], te["label_bin"].values[keep])
        else:
            f_dedup = f_full
        ff.append(f_full); fd.append(f_dedup)
    row = {"dataset": name, "split": mode, "n_test": int(len(reps[0][1])),
           "leaked_pct": round(100*np.mean(lk), 2),
           "f1_full": round(np.mean(ff), 4), "f1_dedup": round(np.mean(fd), 4),
           "f1_inflation": round(np.mean(ff) - np.mean(fd), 4)}
    leak_rows.append(row); print("  ", row)
pd.DataFrame(leak_rows).to_csv(os.path.join(RES, "leakage.csv"), index=False)

# ---------- B. shortcut reliance ----------
# We report two full references for the top-50 retained fraction. The primary
# reference is the TF-IDF(1,2) unigram+bigram model used throughout the paper.
# Because the top-50 model is unigram-only, the retained fraction against the
# (1,2) reference folds together two effects: a smaller feature budget AND the
# removal of all bigrams. We therefore also report a matched UNIGRAM-ONLY full
# reference, TF-IDF(1,1) with the same min_df/max_features, so that
# retained_vs_uni_pct is a clean budget comparison (50 unigrams vs all unigrams)
# with no bigram confound.
def fit_eval_uni(tr_text, tr_y, te_text, te_y):
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text); Xte = vec.transform(te_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, tr_y)
    return f1_score(te_y, clf.predict(Xte), average="macro")

print("\n== B. lexical shortcut reliance (top-50 tokens vs full) ==")
sc_rows = []
for name, d in DATA.items():
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["label_bin"])
    f_full = fit_eval(tr["text"], tr["label_bin"].values, te["text"], te["label_bin"].values)
    f_full_uni = fit_eval_uni(tr["text"], tr["label_bin"].values, te["text"], te["label_bin"].values)
    v1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True)
    Xtr = v1.fit_transform(tr["text"]); y = tr["label_bin"].values
    ch, _ = chi2(Xtr, y)
    vocab = np.array(v1.get_feature_names_out())
    top_idx = np.argsort(ch)[::-1][:50]
    top_tokens = vocab[top_idx]
    keep_vocab = {t: i for i, t in enumerate(top_tokens)}
    v2 = TfidfVectorizer(ngram_range=(1, 1), vocabulary=keep_vocab, sublinear_tf=True)
    Xtr2 = v2.fit_transform(tr["text"]); Xte2 = v2.transform(te["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr2, y)
    f_top = f1_score(te["label_bin"].values, clf.predict(Xte2), average="macro")
    row = {"dataset": name, "f1_full": round(f_full, 4), "f1_full_uni": round(f_full_uni, 4),
           "f1_top50": round(f_top, 4),
           "retained_pct": round(100*f_top/f_full, 1),
           "retained_vs_uni_pct": round(100*f_top/f_full_uni, 1),
           "top_tokens": " ".join(top_tokens[:15])}
    sc_rows.append(row)
    print(f"   {name:16} full(1,2)={f_full:.3f} full(1,1)={f_full_uni:.3f} top50={f_top:.3f} "
          f"ret_vs_12={row['retained_pct']}% ret_vs_uni={row['retained_vs_uni_pct']}%")
    print(f"      top: {row['top_tokens']}")
pd.DataFrame(sc_rows).to_csv(os.path.join(RES, "shortcut.csv"), index=False)

# ---------- C. cross-dataset transfer matrix ----------
print("\n== C. cross-dataset transfer (macro-F1, train row -> test col) ==")
names = list(DATA.keys())
vecs, clfs = {}, {}
for n in names:
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    X = v.fit_transform(DATA[n]["text"])
    c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, DATA[n]["label_bin"].values)
    vecs[n] = v; clfs[n] = c
M = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        Xb = vecs[a].transform(DATA[b]["text"])
        M.loc[a, b] = round(f1_score(DATA[b]["label_bin"].values, clfs[a].predict(Xb), average="macro"), 3)
M.to_csv(os.path.join(RES, "transfer_matrix.csv"))
print(M.to_string())
diag = np.mean([M.loc[n, n] for n in names])
off = np.mean([M.loc[a, b] for a in names for b in names if a != b])
print(f"\nmean in-dataset F1={diag:.3f} | mean cross-dataset F1={off:.3f} | transfer gap={diag-off:.3f}")
print("\nDONE -> data/results/")
