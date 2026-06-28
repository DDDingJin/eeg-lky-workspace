# Incremental Comparison

- previous fix branch: `fix/ar-20260625-161300-a43831b-etard-p00-pilot-closure`
- previous fix commit: `85fbd5a22dcad9df19d51d8d2369ccd1c27bce9b`
- current fix branch: `fix/ar-20260625-161300-a43831b-article-pilot-diagnostic`

## This round
- diagnostic-only; no full-subject rerun
- old summary figures were audited but not regenerated
- no new benchmark claim should be made from this round

## New diagnostic artifacts
- `result_consistency_check.md`
- `legacy_vs_current_diagnostic.csv`
- `legacy_vs_current_diagnostic.md`
- `etard_condition_diagnostic.csv`
- `etard_condition_diagnostic.md`
- `etard_split_sanity.md`

## Preliminary conclusion
- The Etard gap is more plausibly explained by single-subject pilot scope, condition mixture, and FCNN/ADT protocol/budget mismatch than by an immediately obvious pipeline bug.
- Recommended next step: `C` (focused rerun before full-subject).

## Focused rerun follow-up
- This round did not rerun old summary figures.
- This round added a focused P00 rerun for Etard FCNN and ADT only.
- The focused rerun changes the interpretation: ADT low pilot values are now largely explainable by training budget, while FCNN still appears protocol-limited.

