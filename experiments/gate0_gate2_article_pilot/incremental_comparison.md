# Incremental Comparison

- previous fix branch: `fix/ar-20260625-161300-a43831b-article-pilot-weissbart-etard-p00`
- previous fix commit: `827e66d9c703595ec65c2abda2e6e7c1a6da8807`
- current fix branch: `fix/ar-20260625-161300-a43831b-etard-p00-pilot-closure`

## Newly run in this round
- `etard_tf64_p00` pilot closure under `experiments/gate0_gate2_article_pilot/`
- the unified runner now reads Etard `P00` directly from the full `etard_tf64` export instead of requiring a separate `etard_tf64_p00` alias
- `ridge`, `cca`, `fcnn`, `adt` under the unified pilot runner

## Not rerun in this round
- historical `experiments/summary_figures/*` outputs
- old Hugo sample pilot outputs
- full-subject Weissbart runs
- full-subject Etard runs
- historical HappyQuokka and VLAAI result bundles

## Etard pilot closure
- `etard_tf64_p00` now succeeds by reading participant `P00` from `data/processed/reference_splits/etard_tf64`.
- `etard_tf64_p00_test` remains only a test fixture and was not substituted silently as article pilot evidence.
- `failure_report.json` is no longer needed once the Etard pilot closure succeeds.
