# Incremental Comparison

- previous fix commit: `c32114a1d919604341458ca3da5d9288814f0107`
- current branch: `fix/ar-20260625-161300-a43831b-full-subject-single-seed-v1`
- this round adds the first full-subject single-seed v1 run on `weissbart_tf64` and `etard_tf64`.
- this round does not rerun historical summary figures, pilot-only artifacts, multi-seed experiments, or specialized models outside `ridge / cca / fcnn / adt`.

## New in this round
- `configs/benchmark/gate0_gate2_full_subject_single_seed.json`
- `scripts/run_gate0_gate2_full_subject_single_seed.py`
- full-subject per-job ledgers and aggregated dataset/condition metrics

## Job scope
- `etard_tf64`: 20 subjects
- `weissbart_tf64`: 13 subjects
- cumulative successful jobs available after this round: `132`
- actually run jobs: `0`
- skipped existing jobs: `132`
- failed jobs: `0`

## Not rerun in this round
- `experiments/summary_figures/*` historical summaries
- `experiments/gate0_gate2_article_pilot/*`
- `experiments/gate0_gate2_etard_p00_focused_rerun/*`
- VLAAI / HappyQuokka / NULL / CNN / EEGNet bundles
- any multi-seed or cross-dataset transfer experiment
