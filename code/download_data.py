#!/usr/bin/env python3
"""Download + normalize public, no-login abuse/hate/toxicity datasets into a common schema.

Schema (CSV per dataset in data/processed/):
  dataset, id, text, label_raw, label_bin (1=abuse/hate, 0=not), split

All sources are public (git-cloned shallow). No API keys, no logins.
Idempotent: skips clones/outputs that already exist.
"""
import os, subprocess, json, glob, sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
os.makedirs(RAW, exist_ok=True)
os.makedirs(PROC, exist_ok=True)

REPOS = {
    "hate-speech-and-offensive-language": "https://github.com/t-davidson/hate-speech-and-offensive-language",
    "Dynamically-Generated-Hate-Speech-Dataset": "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset",
    "HateXplain": "https://github.com/hate-alert/HateXplain",
    "tweeteval": "https://github.com/cardiffnlp/tweeteval",
    "hate-speech-dataset": "https://github.com/Vicomtech/hate-speech-dataset",
    # ADDITIVE (robustness extension): OLID / OffensEval-2019 (Zampieri et al. 2019),
    # public, no-login mirror carrying the full tweet text and gold subtask-A labels.
    "OLID": "https://github.com/OceanSnape/OLID",
}

def clone(name, url):
    dst = os.path.join(RAW, name)
    if os.path.isdir(dst):
        print(f"  [skip clone] {name}")
        return dst
    print(f"  [clone] {name} ...")
    r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    !! clone failed: {r.stderr.strip()[:200]}")
        return None
    return dst

def save(df, name):
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    out = os.path.join(PROC, f"{name}.csv")
    df.to_csv(out, index=False)
    pos = df["label_bin"].mean() if len(df) else float("nan")
    print(f"  [ok] {name:14} n={len(df):>7}  %abuse={pos:.2%}  -> {os.path.relpath(out, BASE)}")
    return len(df)

def load_davidson():
    p = os.path.join(RAW, "hate-speech-and-offensive-language", "data", "labeled_data.csv")
    d = pd.read_csv(p)
    out = pd.DataFrame({
        "dataset": "davidson2017", "id": d.index.astype(str), "text": d["tweet"],
        "label_raw": d["class"].map({0: "hate", 1: "offensive", 2: "neither"}),
        "label_bin": d["class"].isin([0, 1]).astype(int), "split": ""})
    return out

def load_dynahate():
    cands = glob.glob(os.path.join(RAW, "Dynamically-Generated-Hate-Speech-Dataset", "*.csv"))
    cands = sorted(cands, key=lambda x: ("v0.2.3" not in x, "v0.2.2" not in x, x))
    if not cands:
        raise FileNotFoundError("dynahate csv not found")
    d = pd.read_csv(cands[0])
    tcol = "text" if "text" in d.columns else d.columns[d.columns.str.contains("text", case=False)][0]
    lcol = "label" if "label" in d.columns else d.columns[d.columns.str.contains("label", case=False)][0]
    scol = "split" if "split" in d.columns else None
    out = pd.DataFrame({
        "dataset": "dynahate2021", "id": d.index.astype(str), "text": d[tcol],
        "label_raw": d[lcol].astype(str),
        "label_bin": d[lcol].astype(str).str.lower().str.replace(" ", "").eq("hate").astype(int),
        "split": d[scol].astype(str) if scol else ""})
    return out

def load_hatexplain():
    p = os.path.join(RAW, "HateXplain", "Data", "dataset.json")
    raw = json.load(open(p))
    rows = []
    for pid, rec in raw.items():
        labs = [a["label"] for a in rec.get("annotators", []) if "label" in a]
        if not labs:
            continue
        maj = max(set(labs), key=labs.count)
        text = " ".join(rec.get("post_tokens", []))
        rows.append((pid, text, maj, int(maj in ("hatespeech", "offensive"))))
    out = pd.DataFrame(rows, columns=["id", "text", "label_raw", "label_bin"])
    out.insert(0, "dataset", "hatexplain")
    out["split"] = ""
    return out

def load_tweeteval():
    base = os.path.join(RAW, "tweeteval", "datasets", "hate")
    frames = []
    for split in ["train", "val", "test"]:
        tf = os.path.join(base, f"{split}_text.txt")
        lf = os.path.join(base, f"{split}_labels.txt")
        if not (os.path.exists(tf) and os.path.exists(lf)):
            continue
        texts = open(tf, encoding="utf-8").read().splitlines()
        labs = [int(x) for x in open(lf).read().split()]
        n = min(len(texts), len(labs))
        frames.append(pd.DataFrame({"dataset": "tweeteval_hate",
            "id": [f"{split}_{i}" for i in range(n)], "text": texts[:n],
            "label_raw": ["hate" if l == 1 else "not" for l in labs[:n]],
            "label_bin": labs[:n], "split": split}))
    return pd.concat(frames, ignore_index=True)

def load_stormfront():
    root = os.path.join(RAW, "hate-speech-dataset")
    meta = pd.read_csv(os.path.join(root, "annotations_metadata.csv"))
    txtdir = os.path.join(root, "all_files")
    rows = []
    for _, r in meta.iterrows():
        lab = str(r["label"]).strip()
        if lab not in ("hate", "noHate"):
            continue
        fp = os.path.join(txtdir, f"{r['file_id']}.txt")
        if not os.path.exists(fp):
            continue
        rows.append((str(r["file_id"]), open(fp, encoding="utf-8").read().strip(),
                     lab, int(lab == "hate")))
    out = pd.DataFrame(rows, columns=["id", "text", "label_raw", "label_bin"])
    out.insert(0, "dataset", "stormfront")
    out["split"] = ""
    return out

def load_olid():
    """OLID / OffensEval-2019 subtask A (Zampieri et al. 2019).
    Train tsv has columns id, tweet, subtask_a (OFF/NOT). The official test split
    is text (testset-levela.tsv) + gold labels (labels-levela.csv) keyed by id.
    Binary mapping: OFF -> 1 (abusive/offensive), NOT -> 0."""
    root = os.path.join(RAW, "OLID")
    tr = pd.read_csv(os.path.join(root, "olid-training-v1.0.tsv"), sep="\t")
    train = pd.DataFrame({
        "dataset": "olid2019", "id": "train_" + tr["id"].astype(str),
        "text": tr["tweet"], "label_raw": tr["subtask_a"].astype(str),
        "label_bin": tr["subtask_a"].astype(str).str.upper().eq("OFF").astype(int),
        "split": "train"})
    tte = pd.read_csv(os.path.join(root, "testset-levela.tsv"), sep="\t")
    lab = pd.read_csv(os.path.join(root, "labels-levela.csv"), header=None,
                      names=["id", "subtask_a"])
    te = tte.merge(lab, on="id", how="inner")
    test = pd.DataFrame({
        "dataset": "olid2019", "id": "test_" + te["id"].astype(str),
        "text": te["tweet"], "label_raw": te["subtask_a"].astype(str),
        "label_bin": te["subtask_a"].astype(str).str.upper().eq("OFF").astype(int),
        "split": "test"})
    return pd.concat([train, test], ignore_index=True)

LOADERS = {"davidson2017": load_davidson, "dynahate2021": load_dynahate,
           "hatexplain": load_hatexplain, "tweeteval_hate": load_tweeteval,
           "stormfront": load_stormfront}

# ADDITIVE corpora written to a SEPARATE directory (data/processed_expanded/) so the
# frozen 5-dataset artifacts produced by code/dedup.py (which globs data/processed/*.csv)
# are never touched. The expanded analysis (code/expanded_conflict.py) reads the frozen
# five from data/processed/ plus these.
EXP_LOADERS = {"olid2019": load_olid}

def save_expanded(df, name):
    """Save an additive corpus to data/processed_expanded/ (NOT data/processed/)."""
    exp = os.path.join(BASE, "data", "processed_expanded")
    os.makedirs(exp, exist_ok=True)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    out = os.path.join(exp, f"{name}.csv")
    df.to_csv(out, index=False)
    pos = df["label_bin"].mean() if len(df) else float("nan")
    print(f"  [ok] {name:14} n={len(df):>7}  %abuse={pos:.2%}  -> {os.path.relpath(out, BASE)}")
    return len(df)

if __name__ == "__main__":
    print("== cloning repos ==")
    for n, u in REPOS.items():
        clone(n, u)
    print("== normalizing ==")
    total = 0
    summary = []
    for name, fn in LOADERS.items():
        try:
            df = fn()
            total += save(df, name)
            summary.append((name, len(df), float(df["label_bin"].mean())))
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\nTOTAL rows: {total} across {len(summary)} datasets")
    print("== normalizing ADDITIVE corpora (data/processed_expanded/) ==")
    for name, fn in EXP_LOADERS.items():
        try:
            df = fn()
            save_expanded(df, name)
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
