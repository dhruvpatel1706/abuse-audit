#!/usr/bin/env bash
# Reproduce the full audit. Override the interpreter with: PY=python ./run_all.sh
set -e
PY=${PY:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
echo "[1/6] download + normalize datasets"
$PY code/download_data.py
echo "[2/6] contamination + item/component label conflict"
$PY code/dedup.py
echo "[3/6] cross-dataset conflict-severity matrix"
$PY code/conflict_matrix.py
echo "[4/6] leakage + shortcut + transfer"
$PY code/analysis.py
echo "[5/6] fair transfer matrix"
$PY code/transfer_fix.py
echo "[6/6] bootstrap CIs + figures"
$PY code/make_figures.py
echo "guard: diff paper numbers against result CSVs"
$PY code/check_paper_numbers.py
echo "Done. Results in data/results/ ; figures in paper/figures/."
