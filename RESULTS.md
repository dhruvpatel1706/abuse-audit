# Results (reproduced 2026-06-04)
All numbers produced by `code/{dedup,analysis,transfer_fix,make_figures}.py` over 5 public datasets (109,698 cleaned rows; 108,946 unique texts). All scripts drop empty-normalized rows so every number is on the same cleaned data. No fabrication. Thesis was revised to match evidence.

## What the data actually says (the honest pivot)
My initial hypothesis (large cross-dataset contamination inflating transfer) was **not supported**. The evidence instead supports a stronger, more useful "myth-busting + mechanism" story.

### Finding 1 — Cross-dataset contamination is NEGLIGIBLE (counters the merge-without-care assumption)
- Only **17** cross-dataset near-duplicate text pairs (Jaccard >= 0.8) out of 108,946 unique texts, **of which 5 are exact duplicates** (the 5 exact are a SUBSET of the 17, not additional).
- Overlap is dominated by, but not confined to, the Twitter-sourced sets (Davidson/HateXplain/TweetEval). At j>=0.8 DynaHate (synthetic) still appears in **6 of 17** cross pairs and **3 of 5** exact cross-duplicates (mostly short generic strings, e.g. "i hate you"). Stormfront (forum) is the genuinely disjoint case: 0 cross pairs at j>=0.8, only 15 of 272 at j>=0.5. [verified against near_dup_pairs.csv]
- Implication: efforts that merge corpora are not silently double-counting; but they are also not as redundant as feared.

### Finding 2 — Where texts DO overlap across datasets, labels CONFLICT about half the time
**Now reported ITEM-LEVEL (per-node), not pair-weighted.** A node is "conflicted" if any cross-dataset near-dup neighbor carries a different binary label; denominator = distinct nodes with >=1 cross neighbor. This replaces the earlier pair-weighted figure: the old 47.8% was pair-weighted over non-independent pairs (95 of 281 nodes at j>=0.5 appear in >1 pair, one in 16), so the tight Wilson CI was misleading. We also drop placeholder/normalization artifacts (nodes >=50% @user tokens OR <15 non-placeholder chars, e.g. `@user @user`, "i hate you"). [code/dedup.py -> data/results/label_conflict_itemlevel.csv, label_conflict_topdegree_stability.csv]

- **PRIMARY (clean item-level, j>=0.5, the largest usable sample):** 103/212 = **48.6%** (95% bootstrap-over-nodes CI [42.0, 55.7], 10k resamples). [NOTE: the j>=0.5 clean CI was corrected from a stale [41.5,55.2] in an earlier PDF to the checksummed-artifact value [42.0,55.7]; a build-time guard code/check_paper_numbers.py now diffs every paper number against the CSVs so this cannot recur.]
- **Sensitivity (clean item-level, j>=0.8, genuine near-dups):** 10/18 = **55.6%** (CI [33.3, 77.8]); all-nodes j>=0.5 = 151/281 = 53.7% (CI [48.0, 59.4]); all-nodes j>=0.8 = 14/27 = 51.9%.
- **Residual non-independence:** the two endpoints of each cross-dataset near-dup are correlated nodes, so the bootstrap-over-nodes interval is a LOWER bound. Collapsing each connected component of the near-dup graph to ONE independent unit (cluster conflicts if its members span both labels) gives clean **42.5%** (31/73, CI [31.5, 53.4]) at j>=0.5 and **50.0%** (4/8, CI [12.5, 87.5]) at j>=0.8. The point estimate slides down and the CI widens, but node-level 48.6% sits inside the component CI, so "about half" survives. New columns in label_conflict_itemlevel.csv (component_rate_clean_*, computed by code/dedup.py).
- **Robustness (defect-driven):**
  - Davidson-excluded (pair-weighted, j>=0.5): **47.6%** (49/103), essentially unchanged from 47.8% over all pairs. Item-level Davidson-excluded depends on cutoff: **58.7%** at j>=0.5 (n=109), but **40.0%** (4/10) at j>=0.8 (noisy at small n); stable robustness statement is at j>=0.5. So the headline is not a Davidson artifact.
  - Top-degree stability (item-level, j>=0.5): drop top 1/3/5/10 highest-degree nodes -> 48.9 / 51.0 / 48.7 / 49.8%. Stable.
- OLD->NEW headline: pair-weighted 47.8% (Wilson [41.9,53.7]) -> item-level clean **48.6% (j>=0.5, primary, n=212)** / 55.6% (j>=0.8, n=18), bootstrap-over-nodes CI [42.0,55.7], plus a component-level row (42.5% j>=0.5) for the residual pairing. Abstract now LEADS with the n=212 figure; 55.6%/n=18 demoted to the strict-cutoff sensitivity row. Pair-weighted rows retained in Table 2 for reference.
- Concrete, item-level evidence of definitional divergence (the abstract claim of Fortuna 2020, now measured directly). Sec 4.3 transfer attribution softened to a "lower bound on the definitional component" (it only sees the texts that happen to recur).

### Finding 3 — HEADLINE: these benchmarks are overwhelmingly lexical shortcuts
Macro-F1 recoverable from just **50 chi2-selected unigrams** vs the full TF-IDF(1,2) model:

| dataset | full F1 | top-50 F1 | retained |
|---|---|---|---|
| davidson2017 | 0.899 | 0.816 | **90.8%** |
| dynahate2021 | 0.640 | 0.610 | **95.4%** |
| hatexplain | 0.737 | 0.677 | **91.9%** |
| tweeteval_hate | 0.734 | 0.692 | **94.4%** |
| stormfront | 0.681 | 0.645 | **94.7%** |

- 50 tokens retain ~91-95% of performance everywhere. Top tokens are slurs/identity terms/keywords. Abuse "detection" on these benchmarks is largely keyword spotting.
- **Unigram-only full reference added (defect-driven, shortcut.csv: f1_full_uni, retained_vs_uni_pct).** The retained-% vs the (1,2) reference conflated feature budget with bigram removal. Against a matched UNIGRAM-ONLY (1,1) full reference (clean 50-unigrams-vs-all-unigrams budget comparison), retained = davidson 90.5 / dynahate 91.5 / hatexplain 91.5 / tweeteval 94.5 / stormfront 93.8%, band **90.5-94.5%** (vs 90.8-95.4% on (1,2)). Qualitative pattern holds and is tighter. The only mover is DynaHate (95.4 vs 91.5%): adding bigrams slightly LOWERED its full model (0.640 vs 0.667), so the (1,2) comparison was understating concentration there. Both columns in Table 2 (now Table 3).
- Robustness (Appendix B / k_sensitivity.csv): budget sweep k=10/25/50/100 shows retention climbs smoothly (63-93% at k=10 to 93-100% at k=100), so 50 is representative not cherry-picked; majority-class macro-F1 (0.35-0.47) is far below, so the retained signal is real.
- **Implicit-hate robustness (Appendix C / implicit_hate_shortcut.csv, code/implicit_hate.py).** Hardest case for the lexical-shortcut claim. Ran the IDENTICAL pipeline on the ElSherief 2021 "Latent Hatred" stage-1 corpus (HF mirror `tasksource/implicit-hate-stg1`, redistribution of SALT-NLP/implicit-hate stg1; 21,480 posts: not_hate 13,291 / implicit_hate 7,100 / explicit_hate 1,089). Data obtained free, no credentials (the GitHub repo gates the zip behind a survey/Dropbox, but the HF mirror is public+ungated).
  - PRIMARY (implicit_hate vs not_hate, explicit excluded): majority F1 **0.395**, full F1 **0.712**, top-50 F1 **0.652**, **retained 91.5%** (95% CI [88.7, 94.2]). k-sweep ret @10/25/50/100 = 81.8/90.1/91.5/93.1.
  - SECONDARY (any-hate vs not_hate, full corpus): majority 0.382, full **0.720**, top-50 0.674, **retained 93.6%** (CI [90.9, 96.4]).
  - INTERPRETATION (honest, reported as-is): retained fraction is NOT lower than the five lexical benchmarks (90.8-95.4% band) -- it sits inside it. But absolute full F1 (0.712) is below 4 of 5 benchmarks and the full-minus-majority gap (0.317) is mid-pack, so the task is genuinely harder in absolute terms; that is where implicitness shows up. The shortcut TOKENS shift from raw slurs to coded/dog-whistle terms (whitegenocide, diversity, genocide, civilization, send, illegals). Conclusion: even a corpus designed to be non-lexical leaves a recoverable lexical fingerprint for a linear TF-IDF model, in a different vocabulary. This SHARPENS (does not overturn) the body claim and turns the old Limitations hedge into a measurement. Limitations + Sec 4.3 now cross-reference Appendix C.

### Finding 4 — Poor cross-dataset transfer, NOT explained by contamination
- Cross-dataset (off-diagonal) macro-F1 mean ~**0.54**; worst pairs ~0.29-0.46. Models do not generalize across corpora.
- Since contamination is negligible (Finding 1), the transfer failure is attributable to distribution + **definitional** divergence (Finding 2), not leakage.
- DONE: diagonal now uses a held-out 80/20 split (fair). Mean in-dataset F1 = 0.738, mean cross-dataset = 0.537, gap = 0.20. Paper attribution softened to distributional AND definitional divergence; the label-conflict measurement isolates only the definitional part.

### Finding 5 — Train/test near-dup leakage is common but low-impact for linear models (nuance)
- Leaked test fraction: tweeteval 9.8%, dynahate 8.7%, davidson 2.4%, stormfront 2.2%, hatexplain 0.4%.
- F1 inflation from leakage: at most 0.012 in magnitude (largest is -0.0113 on DynaHate; negligible) for TF-IDF+LogReg. [leakage.csv] Honest counter to the assumption that any leakage badly inflates scores (the strong Arango effect was author-level / specific setups). Flag: may matter more for high-capacity neural models; we scope our claim to linear baselines.
- Transformer check (distilBERT, Appendix D): inflation -0.0025 (TweetEval-Hate) / -0.0060 (DynaHate), within noise of zero over 3 seeds. SCOPED: the body no longer says "leakage does not inflate transformer scores either" as a general claim; it now reads "for a fine-tuned distilBERT on these two corpora," and notes a 66M/3-epoch model is the case LEAST likely to memorize (weakest test of the worry), so it extends rather than settles the question. transformer_check.py is now CPU-only (device hard-set to "cpu"; mps branches and the time.time() print removed); RE-RUN 2026-06-16: torch==2.12.0 + transformers==5.12.1 installed in the audit venv; distilBERT re-run from scratch (3 seeds, CPU, ~3.5h). Numbers updated to the reproduced values above (DynaHate inflation moved -0.0039 -> -0.0060 with the newer transformers; both still within noise of zero, std <=0.0013). transformer_leakage.csv/.json re-frozen in CHECKSUMS (20/20 OK), requirements.txt now pins these exact versions, and the paper-number build guard passes. Appendix D is now reproducible from the shipped environment.

## Reframed contributions
1. A released, runnable tool (`abuse-audit`, code/abuse_audit.py) that emits a per-dataset report card (near-dup rate, train/test leakage + F1 inflation, shortcut score + top tokens) for any CSV; reproduces the per-dataset paper numbers (e.g. Davidson shortcut 90.8%, full F1 0.899) on CPU. No longer future-tense: "we release," not "we will release."
2. Evidence that cross-corpus contamination is negligible yet, where present, labels conflict ~half the time (primary item-level rate 48.6% at j>=0.5, n=212; definitional divergence).
3. A single comparable **shortcut score** showing ~91-95% of F1 is lexical across 5 datasets.
4. A cross-dataset transfer audit attributing the gap to divergence, not contamination/leakage.
5. Myth-busting nuance: near-dup train/test leakage is common but low-impact for linear models.


## Reproduce
Reproduce: `./run_all.sh` (uses a Python 3.12 venv from requirements.txt). Run a report card on any CSV with `python code/abuse_audit.py DATA.csv --text-col text --label-col label_bin`.
