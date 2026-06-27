# Article-Grade Dataset Pilot Validation

This directory is a pilot validation layer for article-grade candidate datasets, not a full-subject benchmark and not a final paper conclusion.

## Requested datasets
- `weissbart_tf64_p00`
- `etard_tf64_p00`

## Requested models
- `ridge`
- `cca`
- `fcnn`
- `adt`

## Pilot status
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

## Interpretation guardrail
- These P00 pilot outputs only validate that the current unified runner/scorer stack can be extended to article-grade candidate datasets.
- They must not be reported as full benchmark conclusions.
