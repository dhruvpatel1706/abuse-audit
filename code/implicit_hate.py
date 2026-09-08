#!/usr/bin/env python3
"""Implicit-hate robustness check.

We run the SAME lexical-shortcut methodology used for the five lexical benchmarks
(make_figures.py / analysis.py: TF-IDF(1,2)+LogReg full model vs a 50-token
chi^2-unigram model, stratified 80/20 split, seed 42, macro-F1, majority-class
baseline) on the ElSherief et al. (2021) "Latent Hatred" implicit-hate corpus.

Implicit hate deliberately targets hate expressed WITHOUT overt slurs, so it is
the hardest case for the paper's lexical-shortcut finding. If 50 tokens retain a
much smaller fraction of full-model F1 here than on the five lexical benchmarks
(91-95%), that supports the paper's nuance: the shortcut finding is a property of
how those benchmarks are built, and an implicit-hate corpus resists it.

Data source (free, no credentials): the stage-1 binary-classification file
implicit_hate_v1_stg1_posts.tsv, mirrored publicly and ungated on the Hugging
Face Hub at tasksource/implicit-hate-stg1 (original release: SALT-NLP/implicit-hate).
Classes: not_hate (13,291), implicit_hate (7,100), explicit_hate (1,089).

Primary task (the implicit case the paper's Limitations calls out): implicit_hate
(positive) vs not_hate (negative); the lexical explicit_hate class is excluded so
the test measures detection of hate without overt slurs. We also report a
secondary "any-hate vs not_hate" framing (implicit+explicit positive) so the
full-corpus number is on record. All numbers computed identically to the paper.

Outputs -> data/results/implicit_hate_shortcut.csv and .../implicit_hate_summary.json
"""
import os, re, html, json, csv, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "implicit-hate")
RES = os.path.join(BASE, "data", "results")
os.makedirs(RES, exist_ok=True)
TSV = os.path.join(RAW, "implicit_hate_v1_stg1_posts.tsv")

SEED = 42
rng = np.random.default_rng(SEED)
KS = [10, 25, 50, 100]

# --- identical normalization to make_figures.py / analysis.py ---
URL = re.compile(r"https?://\S+|www\.\S+"); MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+"); RT = re.compile(r"^rt\b[:\s]*", re.I)
def normalize(t):
    t = html.unescape(str(t)).lower(); t = RT.sub(" ", t); t = URL.sub(" ", t)
    t = MENTION.sub("@user", t); return WS.sub(" ", t).strip()

def load():
    d = pd.read_csv(TSV, sep="\t", quoting=csv.QUOTE_MINIMAL, engine="python",
                    on_bad_lines="skip")
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={"post": "text", "class": "label_raw"})
    d["text"] = d["text"].astype(str)
    d["norm"] = d["text"].map(normalize)
    d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    return d

# --- identical model + metric to make_figures.py ---
def fit_full(tr_text, ytr, te_text, yte):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    p = clf.predict(vec.transform(te_text))
    return f1_score(yte, p, average="macro"), p

def top_tokens(tr_text, ytr, k):
    v1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True)
    X1 = v1.fit_transform(tr_text)
    ch, _ = chi2(X1, ytr)
    vocab = np.array(v1.get_feature_names_out())
    return vocab[np.argsort(ch)[::-1][:k]]

def fit_topk(tr_text, ytr, te_text, yte, toks):
    v2 = TfidfVectorizer(ngram_range=(1, 1), vocabulary={t: i for i, t in enumerate(toks)}, sublinear_tf=True)
    c2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(v2.fit_transform(tr_text), ytr)
    p = c2.predict(v2.transform(te_text))
    return f1_score(yte, p, average="macro"), p

def boot_f1(y, p, B=1000):
    y = np.asarray(y); p = np.asarray(p); n = len(y); out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n); out[b] = f1_score(y[idx], p[idx], average="macro")
    return out

def run_task(name, d, pos_classes, neg_classes):
    sub = d[d["label_raw"].isin(pos_classes + neg_classes)].copy()
    sub["label_bin"] = sub["label_raw"].isin(pos_classes).astype(int)
    tr, te = train_test_split(sub, test_size=0.2, random_state=SEED, stratify=sub["label_bin"])
    ytr, yte = tr["label_bin"].values, te["label_bin"].values

    f_full, p_full = fit_full(tr["text"], ytr, te["text"], yte)
    maj = int(round(ytr.mean()))
    f_majority = f1_score(yte, np.full(len(yte), maj), average="macro")

    # full k-sweep (same KS as paper), plus the headline top-50 with bootstrap CI on retained fraction
    ksweep = {}
    p_top50 = None
    top50_list = None
    for k in KS:
        toks = top_tokens(tr["text"], ytr, k)
        f_k, p_k = fit_topk(tr["text"], ytr, te["text"], yte, toks)
        ksweep[f"f1_top{k}"] = round(f_k, 3)
        ksweep[f"ret{k}"] = round(100 * f_k / f_full, 1)
        if k == 50:
            p_top50 = p_k
            top50_list = list(toks)

    bf, bt = boot_f1(yte, p_full), boot_f1(yte, p_top50)
    ret = bt / bf
    f_top50 = ksweep["f1_top50"]
    summary = {
        "task": name,
        "n_total": int(len(sub)),
        "n_pos": int(sub["label_bin"].sum()),
        "pos_rate": round(float(sub["label_bin"].mean()), 4),
        "n_test": int(len(te)),
        "f1_majority": round(f_majority, 3),
        "f1_full": round(f_full, 3),
        "f1_full_lo": round(float(np.percentile(bf, 2.5)), 3),
        "f1_full_hi": round(float(np.percentile(bf, 97.5)), 3),
        "f1_top50": round(f_top50, 3),
        "retained_pct": round(100 * f_top50 / f_full, 1),
        "retained_lo": round(100 * float(np.percentile(ret, 2.5)), 1),
        "retained_hi": round(100 * float(np.percentile(ret, 97.5)), 1),
        "top15_tokens": " ".join(top50_list[:15]),
    }
    summary.update(ksweep)
    return summary

if __name__ == "__main__":
    print("== Implicit-hate (Latent Hatred, ElSherief et al. 2021) shortcut check ==")
    d = load()
    dist = d["label_raw"].value_counts().to_dict()
    print(f"loaded {len(d)} rows after normalization; class distribution: {dist}")

    rows = []
    # PRIMARY: implicit vs not_hate (excludes lexical explicit_hate class)
    rows.append(run_task("implicit_vs_nothate", d, ["implicit_hate"], ["not_hate"]))
    # SECONDARY: any hate vs not_hate (full corpus, mirrors paper's "abuse vs not")
    rows.append(run_task("anyhate_vs_nothate", d, ["implicit_hate", "explicit_hate"], ["not_hate"]))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RES, "implicit_hate_shortcut.csv")
    df.to_csv(out_csv, index=False)
    out_json = os.path.join(RES, "implicit_hate_summary.json")
    json.dump({"class_distribution": dist, "n_after_norm": int(len(d)),
               "source": "tasksource/implicit-hate-stg1 (HF mirror of SALT-NLP/implicit-hate stg1)",
               "tasks": rows}, open(out_json, "w"), indent=2)

    print("\n-- results --")
    for r in rows:
        print(f"[{r['task']}] n={r['n_total']} pos_rate={r['pos_rate']:.3f} "
              f"maj={r['f1_majority']:.3f} full={r['f1_full']:.3f} "
              f"top50={r['f1_top50']:.3f} retained={r['retained_pct']}% "
              f"[{r['retained_lo']},{r['retained_hi']}]")
        print(f"     k-sweep ret: @10={r['ret10']} @25={r['ret25']} @50={r['ret50']} @100={r['ret100']}")
        print(f"     top tokens: {r['top15_tokens']}")
    print(f"\nwrote {out_csv}\nwrote {out_json}")
