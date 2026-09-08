# abuse-audit

A small, reproducible toolkit that audits public online-abuse / hate-speech / toxicity
datasets and emits a per-dataset "report card" covering three axes:

1. **Overlap / contamination**: exact and near-duplicate texts within and across datasets
   (character 5-gram MinHash + LSH), plus item-level label conflict on shared texts.
2. **Lexical shortcuts**: how much macro-F1 a 50-token model retains vs. a full TF-IDF model.
3. **Cross-dataset transfer**: train on one corpus, test on every other.

It accompanies the paper *"Fifty Tokens of Signal: A Cross-Dataset Audit of Overlap,
Shortcuts, and Label Divergence in Online-Abuse Benchmarks"* (WOAH 2026). Every number and
figure in the paper is produced by this code. No GPU and no paid API are required.

## Datasets
All five are public and downloadable without login, API key, or tweet rehydration. They are
fetched (shallow git clone) and normalized to a common schema by `code/download_data.py`.
We redistribute no raw or normalized text: `data/raw/` and `data/processed/` are gitignored
and regenerated locally by running `python code/download_data.py`. The raw corpora contain
verbatim user content (Twitter and Gab handles, forum posts, and some email addresses), so
each is used only under its own source license and the originating platform's terms:

- **Davidson (2017)** (`t-davidson/hate-speech-and-offensive-language`): MIT license; tweet
  text under Twitter/X terms. Cite Davidson et al. (2017).
- **DynaHate / Vidgen (2021)** (`bvidgen/Dynamically-Generated-Hate-Speech-Dataset`):
  CC BY 4.0; human-generated (not scraped), no third-party PII.
- **HateXplain (2021)** (`hate-alert/HateXplain`): MIT license; posts from Twitter and Gab,
  under those platforms' terms. Cite Mathew et al. (2021).
- **TweetEval-Hate (2020)** (`cardiffnlp/tweeteval`): tweet text governed by Twitter/X
  Developer terms; the hate subset derives from Basile et al. (2019, SemEval).
- **Stormfront / de Gibert (2018)** (`Vicomtech/hate-speech-dataset`): CC BY-SA 3.0 ES;
  forum posts that include some user email addresses.

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
bash run_all.sh          # or: PY=python ./run_all.sh
```
This runs, in order:
- `code/download_data.py`     -> `data/processed/*.csv`
- `code/dedup.py`             -> contamination matrix, near-dup pairs, item- and component-level label conflicts
- `code/conflict_matrix.py`   -> cross-dataset conflict-severity matrix (per dataset pair)
- `code/analysis.py`          -> leakage, shortcut, transfer
- `code/transfer_fix.py`      -> transfer matrix with a fair held-out diagonal
- `code/make_figures.py`      -> bootstrap CIs + figures in `paper/figures/`
- `code/check_paper_numbers.py` -> build guard: diffs every number in `paper/main.tex` against the result CSVs (fails loudly on any mismatch)

Results land in `data/results/`; figures in `paper/figures/`.

### Report card for your own dataset
`code/abuse_audit.py` is the one-command tool: give it a single CSV with a text column
and a binary (or mappable) label column and it prints the report card (rows / unique texts /
class balance, internal near-duplicate clusters, train/test leakage and its macro-F1 inflation,
and the 50-token lexical-shortcut score with its top tokens). CPU-only, no paid API.
```bash
python code/abuse_audit.py data/processed/davidson2017.csv --text-col text --label-col label_bin
python code/abuse_audit.py YOUR.csv --text-col text --label-col category --positive-values hate,abuse --json card.json
```

## Outputs (the report card)
The released `data/results/` artifacts are derived statistics and report cards only, with no
verbatim user text: `contamination_matrix_t80.csv`, `near_dup_pairs.csv` (indices and metrics),
`label_conflict.csv`, `threshold_sensitivity.csv`, `leakage.csv`, `shortcut.csv`,
`refined.csv`, `transfer_matrix.csv`. Two intermediate files that embed verbatim text
(`nodes.csv`, `exact_cross_dataset.csv`) are gitignored and kept local only; they regenerate
from the processed corpora via `code/dedup.py`.

## Reproducibility
Fixed seeds throughout. Runs on a CPU in a few minutes. Re-running regenerates every table
and figure in the paper.

## Ethics
The datasets contain slurs and harmful content. This toolkit is for evaluating and improving
abuse-mitigation systems. High-association tokens are reported because they are central to the
shortcut finding; the strongest slurs are masked in the paper.

## License
MIT (see `LICENSE`).
