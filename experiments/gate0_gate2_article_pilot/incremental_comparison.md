# Incremental Comparison

- previous fix branch: `fix/ar-20260625-161300-a43831b-gate0-gate2-real-evidence-and-paper`
- previous fix commit: `39b78b6a9f2cab94f7665a80efe794d3c5b99671`
- current fix branch: `fix/ar-20260625-161300-a43831b-article-pilot-weissbart-etard-p00`

## Newly run in this round
- `weissbart_tf64_p00` pilot outputs under `experiments/gate0_gate2_article_pilot/`
- `ridge`, `cca`, `fcnn`, `adt` under the unified pilot runner

## Not rerun in this round
- historical `experiments/summary_figures/*` outputs
- old Hugo sample pilot outputs
- full-subject Weissbart runs
- full-subject Etard runs
- historical HappyQuokka and VLAAI result bundles

## Missing-dataset handling
- `etard_tf64_p00` was requested but not found locally.
- The existing local `etard_tf64_p00_test` export was not substituted silently.
- A failure report was emitted in `run_manifest.json` instead of fabricating Etard pilot metrics.
