#!/usr/bin/env python3
"""Transformer replication of the lexical-SHORTCUT finding (Finding B / shortcut.csv).

The body's shortcut result (analysis.py -> shortcut.csv) is linear: a TF-IDF+logreg
classifier restricted to the 50 highest-chi2 unigrams keeps 90-95% of the full
unigram model's macro-F1. The natural objection mirrors the one already
answered for leakage: maybe a contextual model would NOT lean on a handful of slur/
identity tokens, so the shortcut is a bag-of-words artifact.

This script tests that directly. It keeps the EXACT shortcut protocol of analysis.py:
  - identical text normalization (html.unescape/lower/strip RT/URL/@user),
  - identical stratified 80/20 split (random_state=42),
  - identical top-50 token selection: TF-IDF(1,1) min_df=3 vocab, top 50 by chi2 on
    the TRAIN split only (reproduces shortcut.csv's top_tokens),
and only swaps the model: fine-tune distilbert-base-uncased on (a) the FULL text and
(b) a SHORTCUT-ONLY copy of each text in which every word token NOT in the 50-token
set is deleted (order preserved). Same split, same labels, same epochs/seeds; the
only difference between the two arms is whether non-shortcut words are visible.

retained_pct = macro-F1(shortcut-only) / macro-F1(full). If distilBERT also retains
most of its F1 from just the 50 tokens, the shortcut reliance is model-agnostic, not
a linear artifact. CPU-only, deterministic per seed, frugal (subsample large train
sets to a cap, max_len 96, a few epochs). Outputs -> data/results/transformer_shortcut.{csv,json}.
"""
import os, re, html, json, gc, warnings, sys
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "data", "results")
MODEL_NAME = "distilbert-base-uncased"
SEED = 42                                   # split seed, matched to analysis.py
TRAIN_SEEDS = [42, 1]                        # model-init seeds for the CI
TRAIN_CAP = int(os.environ.get("EXP_CAP", 6000))   # subsample train for CPU frugality (test full)
MAXLEN = 96
EPOCHS = int(os.environ.get("EXP_EPOCHS", 3))
BS = 16
DATASETS = ["davidson2017", "hatexplain", "tweeteval_hate", "dynahate2021", "stormfront"]  # all five, uniform coverage
if os.environ.get("EXP_SMOKE"):
    TRAIN_SEEDS = [42]; DATASETS = ["tweeteval_hate"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182}
torch.set_num_threads(int(os.environ.get("EXP_THREADS", 6)))

URL = re.compile(r"https?://\S+|www\.\S+"); MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+"); RT = re.compile(r"^rt\b[:\s]*", re.I)
WORD = re.compile(r"\b\w\w+\b")              # TfidfVectorizer default token pattern


def normalize(t):
    t = html.unescape(str(t)).lower(); t = RT.sub(" ", t); t = URL.sub(" ", t)
    t = MENTION.sub("@user", t); return WS.sub(" ", t).strip()


def load(name):
    d = pd.read_csv(os.path.join(PROC, f"{name}.csv"))
    d["text"] = d["text"].astype(str); d["norm"] = d["text"].map(normalize)
    d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    return d


def top50_tokens(tr_text, tr_y):
    v1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True)
    X = v1.fit_transform(tr_text)
    ch, _ = chi2(X, tr_y)
    vocab = np.array(v1.get_feature_names_out())
    return set(vocab[np.argsort(ch)[::-1][:50]])


def to_shortcut(text, keep):
    toks = [w for w in WORD.findall(text.lower()) if w in keep]
    return " ".join(toks) if toks else "[empty]"


class DS(Dataset):
    def __init__(self, texts, labels, tok):
        self.e = tok(list(texts), truncation=True, max_length=MAXLEN, padding="max_length")
        self.y = list(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.e["input_ids"][i]),
                "attention_mask": torch.tensor(self.e["attention_mask"][i]),
                "labels": torch.tensor(int(self.y[i]))}


def train_eval(tr_text, tr_y, te_text, te_y, seed, tok, n_labels):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_labels)
    dl = DataLoader(DS(tr_text, tr_y, tok), batch_size=BS, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for _ in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        labels=b["labels"])
            out.loss.backward(); opt.step()
    model.eval()
    preds = []
    te = DS(te_text, te_y, tok)
    dlte = DataLoader(te, batch_size=64)
    with torch.no_grad():
        for b in dlte:
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            preds.extend(logits.argmax(-1).tolist())
    del model; gc.collect()
    return f1_score(te_y, preds, average="macro")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    rows, detail = [], []
    for name in DATASETS:
        d = load(name)
        tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["label_bin"])
        keep = top50_tokens(tr["text"], tr["label_bin"].values)
        # cap train size deterministically (stratified) for CPU frugality
        if len(tr) > TRAIN_CAP:
            tr, _ = train_test_split(tr, train_size=TRAIN_CAP, random_state=SEED,
                                     stratify=tr["label_bin"])
        n_labels = int(d["label_bin"].nunique())
        tr_full = tr["text"].tolist(); te_full = te["text"].tolist()
        tr_sc = [to_shortcut(t, keep) for t in tr_full]
        te_sc = [to_shortcut(t, keep) for t in te_full]
        ty = tr["label_bin"].values; ey = te["label_bin"].values
        full_f1, sc_f1 = [], []
        for seed in TRAIN_SEEDS:
            ff = train_eval(tr_full, ty, te_full, ey, seed, tok, n_labels)
            sf = train_eval(tr_sc, ty, te_sc, ey, seed, tok, n_labels)
            full_f1.append(ff); sc_f1.append(sf)
            detail.append({"dataset": name, "seed": seed, "f1_full": round(ff, 4),
                           "f1_shortcut": round(sf, 4), "retained_pct": round(100 * sf / ff, 1)})
            print(f"[{name} seed {seed}] full={ff:.3f} shortcut-only={sf:.3f} "
                  f"retained={100*sf/ff:.1f}%", flush=True)
        fm, sm = float(np.mean(full_f1)), float(np.mean(sc_f1))
        ret = [100 * s / f for s, f in zip(sc_f1, full_f1)]
        n = len(TRAIN_SEEDS)
        h = (T95[n - 1] * np.std(ret, ddof=1) / n ** 0.5) if n > 1 else 0.0
        rows.append({"dataset": name, "n_train": len(tr), "n_test": len(te),
                     "n_labels": n_labels, "n_top_tokens": len(keep),
                     "f1_full_mean": round(fm, 4), "f1_shortcut_mean": round(sm, 4),
                     "retained_pct_mean": round(float(np.mean(ret)), 1),
                     "retained_pct_ci95_halfwidth": round(float(h), 1)})
        print(f"  => {name}: distilBERT retains {np.mean(ret):.1f}% (+/-{h:.1f}) of full F1 "
              f"from {len(keep)} tokens", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "transformer_shortcut.csv"), index=False)
    out = {"description": "distilBERT macro-F1 on full text vs shortcut-only text (every word "
                          "outside the top-50 chi2 tokens deleted). Same split/labels/tokens as "
                          "the linear shortcut result (shortcut.csv); only the model changes.",
           "model": MODEL_NAME, "split_seed": SEED, "train_seeds": TRAIN_SEEDS,
           "train_cap": TRAIN_CAP, "max_len": MAXLEN, "epochs": EPOCHS, "device": "cpu",
           "by_dataset": rows, "per_seed": detail}
    with open(os.path.join(RES, "transformer_shortcut.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote transformer_shortcut.{csv,json}")


if __name__ == "__main__":
    main()
