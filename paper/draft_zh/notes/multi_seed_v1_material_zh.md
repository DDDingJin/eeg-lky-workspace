# Multi-Seed v1 Writing Material

This note is writing material only.
It is not final paper prose.

## Multi-seed scope

- seeds: `0`, `42`, `2026`
- datasets: `weissbart_tf64`, `etard_tf64`
- models covered in multi-seed v1: `ridge`, `cca`, `fcnn`, `adt`
- total jobs: `396`
- successful jobs: `396`
- failed jobs: `0`

## Writing-use observations

- `ridge` and `cca` are effectively deterministic in the current unified full-subject setting across these three seeds.
- `adt` and `fcnn` show seed-to-seed variation, but the aggregate magnitude is currently small relative to the mean dataset-level signal.
- The present evidence supports a preliminary stability statement for the current four-model multi-seed bundle.
- The present evidence does not yet justify a final all-model ranking statement.

## Dataset-level references

- `weissbart_tf64`
  - `adt`: mean across seeds about `0.1314`, seed std about `0.0036`
  - `fcnn`: mean across seeds about `0.0932`, seed std about `0.0028`
- `etard_tf64`
  - `adt`: mean across seeds about `0.1009`, seed std about `0.0021`
  - `fcnn`: mean across seeds about `0.0575`, seed std about `0.0033`

## Intended usage

- suitable for results-structure drafting
- suitable for reviewer-facing summary notes
- not suitable as final polished discussion text
