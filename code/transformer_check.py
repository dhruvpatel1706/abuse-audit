#!/usr/bin/env python3
"""Transformer replication of the train/test near-duplicate LEAKAGE finding (Finding 5).

The body's leakage numbers come from TF-IDF + logistic regression, where the F1
inflation from leaked test items is at most 0.012 in magnitude. The obvious
objection is that this low-impact result might be an artifact of a linear
bag-of-words model. This script repeats the IDENTICAL leakage protocol with a
fine-tuned distilbert-base-uncased on the two highest-leakage datasets that ship
official splits (dynahate2021, leaked 8.7%; tweeteval_hate, leaked 9.8%).

Protocol (matched to code/analysis.py):
  - Same text normalization (lowercase, strip RT/URLs, @user mentions).
  - Same leaked_mask: a test item is "leaked" if it exactly- or near-duplicates
    (MinHash LSH, 5-char shingles, Jaccard >= 0.8) some TRAIN item.
  - Same official train/test split (split column).
  - Metric: macro-F1 on label_bin.
  - LEAKAGE INFLATION = macro-F1(full test) - macro-F1(test with leaked items removed).
    The train set is identical in both; we only drop leaked items from the test set,
    exactly as the paper does (analysis.py lines 97-104).

Only the model changes (distilbert instead of TF-IDF+LogReg), so any difference in
inflation is attributable to model capacity, not protocol. Frugal and CPU-only:
batch 16, max_len 128, a few epochs, run over several seeds, mean +/- std.

Outputs -> data/results/transformer_leakage.csv (+ .json with per-seed detail).
"""
import os, re, html, json, gc, warnings, argparse, sys
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score
# Use the IDENTICAL datasketch MinHash LSH the paper used in code/analysis.py so the
# leaked-item definition is byte-for-byte the same. The torch venv has no datasketch,
# so we import it from the paper's own venv by APPENDING that site-packages to the
# path (lowest priority, so this venv's numpy/scipy still win). Verified to reproduce
# the paper's leaked_pct exactly (tweeteval 9.80%, dynahate 8.69%).
_PAPER_VENV_SP = os.environ.get(
    "ABUSE_AUDIT_VENV_SITE_PACKAGES",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 ".venv", "lib", "python3.12", "site-packages"))
if os.path.isdir(_PAPER_VENV_SP) and _PAPER_VENV_SP not in sys.path:
    sys.path.append(_PAPER_VENV_SP)
from datasketch import MinHash, MinHashLSH

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "data", "results")
os.makedirs(RES, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
# CPU-only by design, to match the rest of the audit (everything else runs on a
# laptop CPU). distilBERT fine-tuning on CPU is slow but feasible for the two
# datasets here; we do not use MPS or CUDA.
DEVICE = "cpu"
MAX_LEN = 128
BATCH = 16
EVAL_BATCH = 32
LR = 2e-5
K_SHINGLE = 5  # matches analysis.py

# ----- text normalization (identical to code/analysis.py) -----
URL = re.compile(r"https?://\S+|www\.\S+"); MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+"); RT = re.compile(r"^rt\b[:\s]*", re.I)
def normalize(t):
    t = html.unescape(str(t)).lower(); t = RT.sub(" ", t); t = URL.sub(" ", t)
    t = MENTION.sub("@user", t); return WS.sub(" ", t).strip()
def shingles(t):
    return {t} if len(t) < K_SHINGLE else {t[i:i+K_SHINGLE] for i in range(len(t)-K_SHINGLE+1)}

def load(name):
    d = pd.read_csv(os.path.join(PROC, f"{name}.csv"))
    d["text"] = d["text"].astype(str); d["norm"] = d["text"].map(normalize)
    d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    return d

def leaked_mask(train_norms, test_norms, thr=0.8, num_perm=64):
    """True for each test item that exactly- or near-duplicates some train item.
    Identical logic to code/analysis.py leaked_mask()."""
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

# ----- torch dataset -----
class TextDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=False)
        self.labels = list(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["labels"] = self.labels[i]
        return item

def collate(batch, tok):
    keys = [k for k in batch[0] if k != "labels"]
    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {}
    pad_id = tok.pad_token_id
    for k in keys:
        seqs = []
        for b in batch:
            s = b[k]
            pad_val = pad_id if k == "input_ids" else 0
            seqs.append(s + [pad_val] * (maxlen - len(s)))
        out[k] = torch.tensor(seqs, dtype=torch.long)
    out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return out

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

@torch.no_grad()
def predict(model, texts, tok):
    model.eval()
    preds = []
    for i in range(0, len(texts), EVAL_BATCH):
        chunk = list(texts[i:i+EVAL_BATCH])
        enc = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)

def train_one(tr_text, tr_y, tok, seed, epochs):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)
    ds = TextDS(tr_text, tr_y, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    collate_fn=lambda b: collate(b, tok))
    # class weights for imbalance (mirrors LogReg class_weight="balanced")
    classes, counts = np.unique(tr_y, return_counts=True)
    w = (len(tr_y) / (len(classes) * counts)).astype(np.float32)
    cw = torch.tensor(w, device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(dl) * epochs
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    model.train()
    step = 0
    for ep in range(epochs):
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
        print(f"      epoch {ep+1}/{epochs} done (last loss {loss.item():.4f})", flush=True)
    return model

def run_dataset(name, seeds, epochs):
    print(f"\n=== {name} (device={DEVICE}, epochs={epochs}, seeds={seeds}) ===", flush=True)
    d = load(name)
    tr = d[d["split"] == "train"].reset_index(drop=True)
    te = d[d["split"] == "test"].reset_index(drop=True)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    # leaked mask is deterministic (does not depend on seed): compute once
    mask = leaked_mask(tr["norm"].tolist(), te["norm"].tolist())
    keep = ~mask
    leaked_pct = round(100 * mask.mean(), 2)
    print(f"  leaked test items: {int(mask.sum())}/{len(te)} = {leaked_pct}%", flush=True)

    te_text = te["text"].tolist()
    te_y = te["label_bin"].values
    te_text_keep = [t for t, k in zip(te_text, keep) if k]
    te_y_keep = te_y[keep]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    per_seed = []
    for s in seeds:
        model = train_one(tr["text"].tolist(), tr["label_bin"].values, tok, s, epochs)
        pred_full = predict(model, te_text, tok)
        f_full = f1_score(te_y, pred_full, average="macro")
        # dedup test = same predictions, subset to kept indices (identical model)
        pred_keep = pred_full[keep]
        f_dedup = f1_score(te_y_keep, pred_keep, average="macro")
        infl = f_full - f_dedup
        per_seed.append({"seed": s, "f1_full": round(float(f_full), 4),
                         "f1_dedup": round(float(f_dedup), 4),
                         "f1_inflation": round(float(infl), 4)})
        print(f"   seed {s}: full={f_full:.4f} dedup={f_dedup:.4f} "
              f"inflation={infl:+.4f}", flush=True)
        del model
        gc.collect()

    ff = np.array([r["f1_full"] for r in per_seed])
    fd = np.array([r["f1_dedup"] for r in per_seed])
    inf = np.array([r["f1_inflation"] for r in per_seed])
    summary = {
        "dataset": name, "model": MODEL_NAME, "split": "official",
        "n_test": int(len(te)), "leaked_pct": leaked_pct,
        "n_seeds": len(seeds), "epochs": epochs,
        "f1_full_mean": round(float(ff.mean()), 4), "f1_full_std": round(float(ff.std()), 4),
        "f1_dedup_mean": round(float(fd.mean()), 4), "f1_dedup_std": round(float(fd.std()), 4),
        "f1_inflation_mean": round(float(inf.mean()), 4),
        "f1_inflation_std": round(float(inf.std()), 4),
        "per_seed": per_seed,
    }
    print(f"  >> {name}: full={summary['f1_full_mean']:.4f}+/-{summary['f1_full_std']:.4f} "
          f"dedup={summary['f1_dedup_mean']:.4f} "
          f"INFLATION={summary['f1_inflation_mean']:+.4f}+/-{summary['f1_inflation_std']:.4f}",
          flush=True)
    del tok
    gc.collect()
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dynahate2021", "tweeteval_hate"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    print(f"torch {torch.__version__} | device {DEVICE} | model {MODEL_NAME}", flush=True)
    summaries = []
    for ds in args.datasets:
        summaries.append(run_dataset(ds, args.seeds, args.epochs))
    # tidy CSV
    rows = [{k: v for k, v in s.items() if k != "per_seed"} for s in summaries]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RES, "transformer_leakage.csv")
    df.to_csv(out_csv, index=False)
    with open(os.path.join(RES, "transformer_leakage.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print("\n=== SUMMARY (transformer leakage inflation) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"\nsaved -> {out_csv}", flush=True)

if __name__ == "__main__":
    main()
