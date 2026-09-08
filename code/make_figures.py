#!/usr/bin/env python3
"""Refine stats (bootstrap CIs + fair held-out diagonal) and render publication figures.
Outputs -> paper/figures/*.png and data/results/refined.csv
"""
import os, re, html, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "data", "results")
FIG = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42; rng = np.random.default_rng(SEED)
URL = re.compile(r"https?://\S+|www\.\S+"); MENTION = re.compile(r"@\w+"); WS = re.compile(r"\s+"); RT = re.compile(r"^rt\b[:\s]*", re.I)
def normalize(t):
    t = html.unescape(str(t)).lower(); t = RT.sub(" ", t); t = URL.sub(" ", t); t = MENTION.sub("@user", t); return WS.sub(" ", t).strip()
NAMES = ["davidson2017", "dynahate2021", "hatexplain", "tweeteval_hate", "stormfront"]
DATA = {}
for n in NAMES:
    d = pd.read_csv(os.path.join(PROC, f"{n}.csv")); d["text"] = d["text"].astype(str)
    d["norm"] = d["text"].map(normalize); d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    DATA[n] = d

def boot_f1(y, p, B=1000):
    y = np.asarray(y); p = np.asarray(p); n = len(y); out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n); out[b] = f1_score(y[idx], p[idx], average="macro")
    return out

# ---- shortcut + fair diagonal with bootstrap CIs ----
rows = []
for n in NAMES:
    d = DATA[n]
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["label_bin"])
    yte = te["label_bin"].values
    vf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vf.fit_transform(tr["text"]); cf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, tr["label_bin"].values)
    pf = cf.predict(vf.transform(te["text"]))
    v1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True); X1 = v1.fit_transform(tr["text"])
    ch, _ = chi2(X1, tr["label_bin"].values); vocab = np.array(v1.get_feature_names_out())
    top = vocab[np.argsort(ch)[::-1][:50]]
    v2 = TfidfVectorizer(ngram_range=(1, 1), vocabulary={t: i for i, t in enumerate(top)}, sublinear_tf=True)
    c2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(v2.fit_transform(tr["text"]), tr["label_bin"].values)
    pt = c2.predict(v2.transform(te["text"]))
    f_full_pt = f1_score(yte, pf, average="macro")
    f_top_pt = f1_score(yte, pt, average="macro")
    bf, bt = boot_f1(yte, pf), boot_f1(yte, pt)
    ret = bt / bf
    # point estimates are the single held-out run; brackets are bootstrap 95% CIs
    rows.append({"dataset": n, "f1_full": round(f_full_pt, 3), "f1_full_lo": round(np.percentile(bf, 2.5), 3), "f1_full_hi": round(np.percentile(bf, 97.5), 3),
                 "f1_top50": round(f_top_pt, 3), "retained_pct": round(100*f_top_pt/f_full_pt, 1),
                 "retained_lo": round(100*np.percentile(ret, 2.5), 1), "retained_hi": round(100*np.percentile(ret, 97.5), 1),
                 "fair_diag_f1": round(f_full_pt, 3)})
ref = pd.DataFrame(rows); ref.to_csv(os.path.join(RES, "refined.csv"), index=False)
print("refined shortcut + fair diagonal:\n", ref.to_string(index=False))

# ---- shortcut k-sensitivity + majority-class baseline (single held-out split) ----
KS = [10, 25, 50, 100]
ksrows = []
for n in NAMES:
    d = DATA[n]
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["label_bin"])
    ytr, yte = tr["label_bin"].values, te["label_bin"].values
    vf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    cf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(vf.fit_transform(tr["text"]), ytr)
    f_full = f1_score(yte, cf.predict(vf.transform(te["text"])), average="macro")
    maj = int(round(ytr.mean()))
    f_majority = f1_score(yte, np.full(len(yte), maj), average="macro")
    v1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True)
    ch, _ = chi2(v1.fit_transform(tr["text"]), ytr)
    vocab = np.array(v1.get_feature_names_out()); order = np.argsort(ch)[::-1]
    row = {"dataset": n, "f1_majority": round(f_majority, 3), "f1_full": round(f_full, 3)}
    for k in KS:
        top = vocab[order[:k]]
        v2 = TfidfVectorizer(ngram_range=(1, 1), vocabulary={t: i for i, t in enumerate(top)}, sublinear_tf=True)
        c2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(v2.fit_transform(tr["text"]), ytr)
        f_k = f1_score(yte, c2.predict(v2.transform(te["text"])), average="macro")
        row[f"f1_top{k}"] = round(f_k, 3); row[f"ret{k}"] = round(100 * f_k / f_full, 1)
    ksrows.append(row)
ksdf = pd.DataFrame(ksrows); ksdf.to_csv(os.path.join(RES, "k_sensitivity.csv"), index=False)
print("k-sensitivity + majority baseline:\n", ksdf.to_string(index=False))

# transfer gap with fair diagonal
M = pd.read_csv(os.path.join(RES, "transfer_matrix.csv"), index_col=0)
off = np.mean([M.loc[a, b] for a in NAMES for b in NAMES if a != b])
fair_diag = ref["fair_diag_f1"].mean()
print(f"\nfair in-dataset F1={fair_diag:.3f} | cross-dataset F1={off:.3f} | transfer gap={fair_diag-off:.3f}")
json.dump({"fair_diag": float(fair_diag), "cross": float(off), "gap": float(fair_diag-off)}, open(os.path.join(RES, "transfer_gap.json"), "w"))

SHORT = {"davidson2017": "Davidson", "dynahate2021": "DynaHate", "hatexplain": "HateXplain", "tweeteval_hate": "TweetEval", "stormfront": "Stormfront"}
def heatmap(df, title, fname, fmt="{:.0f}", cmap="Reds"):
    fig, ax = plt.subplots(figsize=(5.2, 4.4)); arr = df.values.astype(float)
    im = ax.imshow(arr, cmap=cmap)
    ax.set_xticks(range(len(df.columns))); ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels([SHORT.get(c, c) for c in df.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([SHORT.get(i, i) for i in df.index], fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, fmt.format(arr[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if arr[i, j] > np.nanmax(arr)*0.6 else "black")
    ax.set_title(title, fontsize=10); fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=200); plt.close(fig); print("saved", fname)

cm = pd.read_csv(os.path.join(RES, "contamination_matrix_t80.csv"), index_col=0)
heatmap(cm, "Cross-dataset near-duplicate links (Jaccard $\\geq$ 0.8)", "fig_contamination.png", "{:.0f}", "Purples")
heatmap(M, "Cross-dataset transfer (macro-F1, train row $\\to$ test col)", "fig_transfer.png", "{:.2f}", "Blues")

# shortcut bar
fig, ax = plt.subplots(figsize=(6, 3.6)); x = np.arange(len(ref)); w = 0.38
ax.bar(x - w/2, ref["f1_full"], w, label="Full TF-IDF(1,2)", color="#3b6fb0")
ax.bar(x + w/2, ref["f1_top50"], w, label="Top-50 tokens only", color="#e08a2e")
for i, r in ref.iterrows():
    ax.text(i, max(r["f1_full"], r["f1_top50"]) + 0.01, f"{r['retained_pct']:.0f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in ref["dataset"]], fontsize=8)
ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.0); ax.set_title("Lexical-shortcut reliance: 50 tokens retain 91-95% of F1", fontsize=9.5)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_shortcut.png"), dpi=200); plt.close(fig); print("saved fig_shortcut.png")
print("\nfigures in paper/figures/")
