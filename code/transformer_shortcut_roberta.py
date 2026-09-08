#!/usr/bin/env python3
"""GPU (Apple MPS) extension of the lexical-shortcut transformer check.

The body result (transformer_shortcut.py) fine-tunes a 66M distilBERT on full vs
shortcut-only text and is run CPU-only. The objection that remains is scale:
does the shortcut survive in a LARGER pretrained model than distilBERT, the kind that
tops modern leaderboards? This script answers that on the same five datasets with the
IDENTICAL protocol (same stratified split, same per-dataset top-50 chi2 token set, same
shortcut-only construction, same training budget), swapping only the model to
roberta-base (125M) and the device to Apple-Silicon MPS, with three model-init seeds
for a tighter interval. It writes a SEPARATE artifact
(transformer_shortcut_roberta.json); the CPU distilBERT artifact and the paper's
CPU-only headline are untouched. Numbers are reported as an MPS-run robustness check,
not as a CPU-reproducible result.

Run: python3 code/transformer_shortcut_roberta.py   (uses MPS if available)
"""
import os, sys, json, gc, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# reuse the EXACT data/feature/shortcut pipeline from the canonical script
import transformer_shortcut as TS

MODEL_NAME = os.environ.get("EXP_MODEL", "roberta-base")
TRAIN_SEEDS = [42, 1, 7]                       # 3 seeds -> df=2, tighter CI than the 2-seed distilBERT run
DATASETS = ["davidson2017", "hatexplain", "tweeteval_hate", "dynahate2021", "stormfront"]
MAXLEN, BS, EPOCHS = TS.MAXLEN, TS.BS, TS.EPOCHS
TRAIN_CAP = TS.TRAIN_CAP
SEED = TS.SEED
RES = TS.RES
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
if os.environ.get("EXP_SMOKE"):
    TRAIN_SEEDS = [42]; DATASETS = ["tweeteval_hate"]; EPOCHS = 1; TRAIN_CAP = 800

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
from sklearn.model_selection import train_test_split


def train_eval(tr_text, tr_y, te_text, te_y, seed, tok, n_labels):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_labels).to(DEVICE)
    dl = DataLoader(TS.DS(tr_text, tr_y, tok), batch_size=BS, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for _ in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            out = model(input_ids=b["input_ids"].to(DEVICE),
                        attention_mask=b["attention_mask"].to(DEVICE),
                        labels=b["labels"].to(DEVICE))
            out.loss.backward(); opt.step()
    model.eval()
    preds = []
    dlte = DataLoader(TS.DS(te_text, te_y, tok), batch_size=64)
    with torch.no_grad():
        for b in dlte:
            logits = model(input_ids=b["input_ids"].to(DEVICE),
                           attention_mask=b["attention_mask"].to(DEVICE)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    del model; gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return f1_score(te_y, preds, average="macro")


def main():
    print(f"model={MODEL_NAME} device={DEVICE} seeds={TRAIN_SEEDS} datasets={DATASETS}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    rows, detail = [], []
    for name in DATASETS:
        d = TS.load(name)
        tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["label_bin"])
        keep = TS.top50_tokens(tr["text"], tr["label_bin"].values)
        if len(tr) > TRAIN_CAP:
            tr, _ = train_test_split(tr, train_size=TRAIN_CAP, random_state=SEED, stratify=tr["label_bin"])
        n_labels = int(d["label_bin"].nunique())
        tr_full = tr["text"].tolist(); te_full = te["text"].tolist()
        tr_sc = [TS.to_shortcut(t, keep) for t in tr_full]
        te_sc = [TS.to_shortcut(t, keep) for t in te_full]
        ty = tr["label_bin"].values; ey = te["label_bin"].values
        full_f1, sc_f1 = [], []
        for seed in TRAIN_SEEDS:
            ff = train_eval(tr_full, ty, te_full, ey, seed, tok, n_labels)
            sf = train_eval(tr_sc, ty, te_sc, ey, seed, tok, n_labels)
            full_f1.append(ff); sc_f1.append(sf)
            detail.append({"dataset": name, "seed": seed, "f1_full": round(ff, 4),
                           "f1_shortcut": round(sf, 4), "retained_pct": round(100 * sf / ff, 1)})
            print(f"[{name} seed {seed}] full={ff:.3f} shortcut-only={sf:.3f} retained={100*sf/ff:.1f}%", flush=True)
        fm, sm = float(np.mean(full_f1)), float(np.mean(sc_f1))
        ret = [100 * s / f for s, f in zip(sc_f1, full_f1)]
        n = len(TRAIN_SEEDS)
        h = (T95[n - 1] * np.std(ret, ddof=1) / n ** 0.5) if n > 1 else 0.0
        rows.append({"dataset": name, "n_train": len(tr), "n_test": len(te), "n_labels": n_labels,
                     "n_top_tokens": len(keep), "f1_full_mean": round(fm, 4),
                     "f1_shortcut_mean": round(sm, 4), "retained_pct_mean": round(float(np.mean(ret)), 1),
                     "retained_pct_ci95_halfwidth": round(float(h), 1)})
        print(f"  => {name}: {MODEL_NAME} retains {np.mean(ret):.1f}% (+/-{h:.1f}) of full F1 from {len(keep)} tokens", flush=True)
    out = {"description": f"{MODEL_NAME} (MPS-run GPU extension) macro-F1 on full vs shortcut-only text "
                          "(every word outside the per-dataset top-50 chi2 tokens deleted). Identical split/"
                          "labels/tokens/budget as the CPU distilBERT result; only the model and device change. "
                          "Reported as a larger-model robustness check, not a CPU-reproducible number.",
           "model": MODEL_NAME, "device": str(DEVICE), "split_seed": SEED, "train_seeds": TRAIN_SEEDS,
           "train_cap": TRAIN_CAP, "max_len": MAXLEN, "epochs": EPOCHS, "by_dataset": rows, "per_seed": detail}
    tag = MODEL_NAME.split("/")[-1].replace("-", "")
    with open(os.path.join(RES, f"transformer_shortcut_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote transformer_shortcut_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
