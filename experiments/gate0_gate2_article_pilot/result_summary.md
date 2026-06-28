# Article-Grade Dataset Pilot Validation

This directory now contains the original P00 pilot outputs plus a diagnostic-only reconciliation layer.

## Diagnostic scope
- This round did not rerun full-subject benchmarks.
- This round did not rerun old summary figures.
- This round only audited the current P00 pilot against existing summary files and split metadata.

## Current pilot values
- `weissbart_tf64_p00`: `success`
  - `ridge` subject-level Pearson: `0.132547`
  - `cca` subject-level Pearson: `0.064111`
  - `fcnn` subject-level Pearson: `0.050608`
  - `adt` subject-level Pearson: `0.162951`
- `etard_tf64_p00`: `success`
  - `ridge` subject-level Pearson: `0.088560`
  - `cca` subject-level Pearson: `0.063781`
  - `fcnn` subject-level Pearson: `0.030900`
  - `adt` subject-level Pearson: `0.056455`

## Diagnostic interpretation
- The current diagnostic is preliminary and should not be written as a paper conclusion.
- Ridge and CCA remain close to the historical scale, which argues against a gross scorer or alignment failure.
- FCNN and ADT on Etard P00 are lower than the historical full-subject summaries and need focused protocol review before full-subject rollout.

## Recommended next step
- Recommendation `C`: pause full-subject for now and do a focused rerun that matches FCNN/ADT protocol or budget more closely on Etard P00.
