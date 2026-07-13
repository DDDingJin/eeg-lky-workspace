# Auditory neural-decoding benchmark — consolidation branch

`codex/open-source-latest` is the integration branch for the current latest EEG and MEG benchmark code.  It contains the leaf branches for MEG-SCANS, within-dataset fixed holdout, cross-dataset transfer, LOSO, subject-holdout fine-tuning, and DECAF, without raw data, checkpoints, or prediction dumps.

It is the staging branch for a future public release, not yet a public-release candidate.  Start with:

- [`docs/OPEN_SOURCE_BRANCH_STATUS.md`](docs/OPEN_SOURCE_BRANCH_STATUS.md) for the branch scope, what was integrated, and the remaining release work.
- [`docs/meg_scans_single_subject_training_walkthrough_zh.md`](docs/meg_scans_single_subject_training_walkthrough_zh.md) for the first code-reading path: the successfully completed MEG-SCANS single-subject training run.

The remainder of this README retains the earlier audit-package description and reading references.  It is useful context, but it does not replace the two consolidation documents above.

This workspace remains a compact audit bundle intended to let an external reviewer judge:

- whether the current reproduction workflow is methodologically reasonable
- whether the current implementations and results look internally consistent
- what should be standardized or improved next before claiming a paper-grade benchmark

## Scope

Current focus:

- auditory EEG
- speech envelope reconstruction / related decoding
- baseline reproduction and benchmark unification
- explicit audit of subject-conditioned versus non-conditioned deep models

Current non-goals of this branch:

- storing raw datasets
- storing full training artifacts
- storing environment snapshots
- serving as the final public benchmark release

## What Is Included

This branch currently exposes four kinds of material.

### 1. Review documents

- `docs/CURRENT_EVALUATION_STATUS.md`
- `docs/reproduction_audit_note.tex`
- `docs/GITHUB_REVIEW_LOOP.md`
- `docs/NEUROCONFORMER_AUDIT.md`

These explain:

- which datasets are already in the unified pipeline
- which methods are currently comparable
- how train/val/test is being done
- what is already methodologically correct
- what is still not fully standardized
- how the GitHub review loop should be handled
- how the remote `NeuroConformer` code differs from the current local benchmark

### 2. Result tables and figures

- `experiments/summary_figures/unified_reference_main_summary.csv`
- `experiments/summary_figures/unified_reference_main_overview.png`
- `experiments/summary_figures/sample_all_methods_summary.csv`
- `experiments/summary_figures/sample_all_methods_overview.png`
- `experiments/summary_figures/happyquokka_conditioning_summary.csv`
- `experiments/summary_figures/happyquokka_conditioning_overview.png`
- `experiments/summary_figures/happyquokka_training_curves.png`
- `experiments/summary_figures/null_conditioning_summary.csv`
- `experiments/summary_figures/null_conditioning_overview.png`
- `experiments/summary_figures/null_training_curves.png`
- `experiments/summary_figures/reproduction_audit_training_curves.csv`
- `experiments/summary_figures/reproduction_audit_training_curves.png`
- `experiments/summary_figures/reproduction_audit_model_protocols.csv`
- `experiments/summary_figures/reproduction_audit_run_metadata.csv`

These cover:

- unified benchmark means on current datasets
- development-sample method panorama
- conditioned versus non-conditioned `HappyQuokka` runs
- conditioned versus non-conditioned `NULL` runs
- validation curves for audit
- protocol metadata for each model family
- run metadata such as requested epochs, completed epochs, and best-validation epoch

### 3. Core implementation files

- `src/repro/reference_baselines.py`
- `src/repro/adt_exact.py`
- `src/repro/vlaai_exact.py`
- `src/repro/happyquokka_reference.py`

These are the most important code files for judging whether the current benchmark logic is plausible.

### 4. Audit package generator

- `scripts/generate_reproduction_audit_package.py`

This script regenerates the current audit tables, curves, and LaTeX note from local experiment outputs.

## Current Benchmark Picture

At the moment, the most important distinction is:

- subject-specific baselines:
  - Ridge
  - CCA
  - FCNN
  - CNN
  - EEGNet
- pooled exact structural ports:
  - ADT-exact
  - VLAAI-exact
- subject-conditioned deep model:
  - HappyQuokka (`g_con=True`)
- matched non-conditioned reference:
  - HappyQuokka (`g_con=False`)
- local integrated conformer-family reference:
  - NULL (`g_con=True`)
  - NULL (`g_con=False`)

This means the current main comparison is informative, but not yet a perfectly apples-to-apples final leaderboard.

That caveat is explicit and intentional in the audit materials.

## HappyQuokka And `g_con`

`HappyQuokka` includes an optional `global conditioner`, exposed as `g_con`.

- `g_con=True` means the model receives an explicit subject identity input
- `g_con=False` means the model only receives EEG input and must behave like a pooled non-conditioned model

Interpretation:

- `g_con=True` is a stronger within-subject configuration because the network can adapt to stable subject-specific EEG differences
- `g_con=False` is the fairer comparison row when placing `HappyQuokka` next to baselines that do not receive subject identity

Current 100-epoch benchmark results:

- `weissbart_tf64`
  - `g_con=True`: `0.1577`
  - `g_con=False`: `0.1434`
- `etard_tf64`
  - `g_con=True`: `0.1287`
  - `g_con=False`: `0.1118`

These paired runs are summarized in:

- `experiments/summary_figures/happyquokka_conditioning_summary.csv`
- `experiments/summary_figures/happyquokka_conditioning_overview.png`
- `experiments/summary_figures/happyquokka_training_curves.png`

## NULL

The local integrated benchmark name for the remote `NeuroConformer` family is `NULL`.

Current 10-epoch benchmark results:

- `hugo_sample_tf64`
  - `g_con=True`: `0.2103`
  - `g_con=False`: `0.1957`
- `weissbart_tf64`
  - `g_con=True`: `0.1993`
  - `g_con=False`: `0.1751`
- `etard_tf64`
  - `g_con=True`: `0.1480`
  - `g_con=False`: `0.1244`

These paired runs are summarized in:

- `experiments/summary_figures/null_conditioning_summary.csv`
- `experiments/summary_figures/null_conditioning_overview.png`
- `experiments/summary_figures/null_training_curves.png`

## Datasets Currently Reflected In This Branch

The unified pipeline currently covers:

- `hugo_sample_tf64`
- `weissbart_tf64`
- `etard_tf64`

Important separation note:

- these are the datasets currently used in the local benchmark tables
- they are not the same thing as the controlled `ICASSP 2023 challenge split` tracked separately under `data/raw/challenge_2023`
- `weissbart_tf64` and `etard_tf64` are public HDF5-aligned dataset releases used as independent benchmark datasets in this branch

Interpretation:

- `hugo_sample_tf64` is mainly a development / smoke-test style dataset
- `weissbart_tf64` and `etard_tf64` are the current more serious reconstruction benchmarks

Dataset lineage shorthand:

- `hugo_sample_tf64`
  - tutorial/development data from the `Thornton / mldecoders` line
  - not `SparrKULee`
  - not the `ICASSP 2023 challenge split`
- `weissbart_tf64`
  - public HDF5-aligned release from the Weissbart line
- `etard_tf64`
  - public HDF5-aligned release from the Etard/Reichenbach line
- `SparrKULee` and the `ICASSP 2023/2024 challenge` splits
  - separate KU Leuven / challenge ecosystem benchmark line
  - not yet integrated into the current branch tables

Datasets downloaded or planned but not yet fully unified into the same benchmark adapter are outside the scope of this branch.

## Recommended Reading Order

If you are reviewing this branch, read in this order:

1. `docs/CURRENT_EVALUATION_STATUS.md`
2. `docs/reproduction_audit_note.tex`
3. `docs/GITHUB_REVIEW_LOOP.md`
4. `docs/NEUROCONFORMER_AUDIT.md`
5. `experiments/summary_figures/unified_reference_main_summary.csv`
6. `experiments/summary_figures/unified_reference_main_overview.png`
7. `experiments/summary_figures/reproduction_audit_training_curves.png`
8. `experiments/summary_figures/reproduction_audit_model_protocols.csv`
9. `experiments/summary_figures/reproduction_audit_run_metadata.csv`
10. `experiments/summary_figures/happyquokka_conditioning_summary.csv`
11. `experiments/summary_figures/happyquokka_conditioning_overview.png`
12. `experiments/summary_figures/null_conditioning_summary.csv`
13. `experiments/summary_figures/null_conditioning_overview.png`
14. `src/repro/reference_baselines.py`
15. `src/repro/adt_exact.py`
16. `src/repro/vlaai_exact.py`
17. `src/repro/happyquokka_reference.py`
18. `src/repro/neuroconformer_reference.py`

## What A Reviewer Should Judge

The most useful feedback at this stage is not "is the score high enough?"

The most useful feedback is:

1. Is the current workflow structurally reasonable for an auditory EEG benchmark?
2. Are any of the reported results obviously suspicious or internally inconsistent?
3. Are disagreements between methods more likely to come from implementation bugs, protocol mismatch, or genuine task difficulty?
4. What should be standardized next to turn this into a stronger benchmark paper?

## Known Caveats

The current branch is intentionally transparent about the main limitations:

- subject-specific baselines and pooled exact ports are not trained under identical regimes
- scalar-sample reconstruction baselines and sequence-to-sequence deep models do not optimize exactly the same target
- some sample-dataset runs are development-grade rather than final article-grade runs
- more datasets still need to be integrated into the same adapter and evaluation layer
- subject-conditioned and non-conditioned comparisons are currently explicit only for `HappyQuokka`, not for every deep model family

## What Is Not Included

This branch does not include:

- raw EEG / stimulus data
- large HDF5 files
- downloaded archives
- environment directories
- temporary caches
- all local exploratory scripts and outputs

That is deliberate. The goal here is reviewability, not full archival completeness.

## Suggested External Review Prompt

You can send reviewers the branch link together with a prompt like this:

```text
Please review this branch as a methodological audit package for an auditory EEG decoding benchmark workspace.

I want feedback on:
1. whether the current reproduction / benchmark workflow is structurally reasonable;
2. whether the current implementations and results look internally consistent;
3. which parts are already acceptable as baseline benchmark code;
4. which parts should be standardized or redesigned next before claiming a paper-grade benchmark.

Please prioritize the following files:
- docs/CURRENT_EVALUATION_STATUS.md
- docs/reproduction_audit_note.tex
- docs/GITHUB_REVIEW_LOOP.md
- docs/NEUROCONFORMER_AUDIT.md
- experiments/summary_figures/unified_reference_main_summary.csv
- experiments/summary_figures/unified_reference_main_overview.png
- experiments/summary_figures/happyquokka_conditioning_summary.csv
- experiments/summary_figures/happyquokka_conditioning_overview.png
- experiments/summary_figures/happyquokka_training_curves.png
- experiments/summary_figures/null_conditioning_summary.csv
- experiments/summary_figures/null_conditioning_overview.png
- experiments/summary_figures/null_training_curves.png
- experiments/summary_figures/reproduction_audit_training_curves.png
- experiments/summary_figures/reproduction_audit_model_protocols.csv
- experiments/summary_figures/reproduction_audit_run_metadata.csv
- src/repro/reference_baselines.py
- src/repro/adt_exact.py
- src/repro/vlaai_exact.py
- src/repro/happyquokka_reference.py
- src/repro/neuroconformer_reference.py

Please distinguish clearly between:
- possible implementation bugs
- protocol mismatch / unfair comparison issues
- expected performance differences caused by task difficulty or model family

I care more about whether the workflow and comparison logic are correct than about whether the current scores are already optimal.
```

## Branch Intent

This branch should be read as:

- a serious intermediate benchmark audit
- not yet the final benchmark release
- not yet the final paper table

It exists to make the current state inspectable and criticizable before more datasets and methods are added.
