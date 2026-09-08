#!/usr/bin/env python3
"""Recompute the transfer matrix with a FAIR diagonal (held-out 80/20 for in-dataset),
off-diagonal = train all of A, test all of B. Overwrite csv + figure honestly."""
import os, re, html, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed"); RES = os.path.join(BASE, "data", "results"); FIG = os.path.join(BASE, "paper", "figures")
NAMES = ["davidson2017", "dynahate2021", "hatexplain", "tweeteval_hate", "stormfront"]
SHORT = {"davidson2017": "Davidson", "dynahate2021": "DynaHate", "hatexplain": "HateXplain", "tweeteval_hate": "TweetEval", "stormfront": "Stormfront"}
URL = re.compile(r"https?://\S+|www\.\S+"); MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+"); RT = re.compile(r"^rt\b[:\s]*", re.I)
def _normalize(t):
    t = html.unescape(str(t)).lower(); t = RT.sub(" ", t); t = URL.sub(" ", t)
    t = MENTION.sub("@user", t); return WS.sub(" ", t).strip()
DATA = {}
for n in NAMES:
    d = pd.read_csv(os.path.join(PROC, f"{n}.csv")); d["text"] = d["text"].astype(str)
    d["norm"] = d["text"].map(_normalize); d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    DATA[n] = d

def model(texts, y):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    X = v.fit_transform(texts); c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    return v, c

M = pd.DataFrame(index=NAMES, columns=NAMES, dtype=float)
for a in NAMES:
    for b in NAMES:
        if a == b:
            tr, te = train_test_split(DATA[a], test_size=0.2, random_state=42, stratify=DATA[a]["label_bin"])
            v, c = model(tr["text"], tr["label_bin"].values)
            M.loc[a, b] = round(f1_score(te["label_bin"].values, c.predict(v.transform(te["text"])), average="macro"), 3)
        else:
            v, c = model(DATA[a]["text"], DATA[a]["label_bin"].values)
            M.loc[a, b] = round(f1_score(DATA[b]["label_bin"].values, c.predict(v.transform(DATA[b]["text"])), average="macro"), 3)
M.to_csv(os.path.join(RES, "transfer_matrix.csv"))
print(M.to_string())
diag = np.mean([M.loc[n, n] for n in NAMES]); off = np.mean([M.loc[a, b] for a in NAMES for b in NAMES if a != b])
print(f"\nfair in-dataset={diag:.3f} cross={off:.3f} gap={diag-off:.3f}")

fig, ax = plt.subplots(figsize=(5.2, 4.4)); arr = M.values.astype(float)
im = ax.imshow(arr, cmap="Blues", vmin=0.2, vmax=0.95)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([SHORT[c] for c in M.columns], rotation=45, ha="right", fontsize=8)
ax.set_yticklabels([SHORT[i] for i in M.index], fontsize=8)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{arr[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if arr[i, j] > 0.65 else "black",
                fontweight="bold" if i == j else "normal")
ax.set_xlabel("tested on", fontsize=9); ax.set_ylabel("trained on", fontsize=9)
ax.set_title("Cross-dataset transfer (macro-F1); diagonal = held-out", fontsize=9.5)
fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_transfer.png"), dpi=200); print("saved fig_transfer.png")
