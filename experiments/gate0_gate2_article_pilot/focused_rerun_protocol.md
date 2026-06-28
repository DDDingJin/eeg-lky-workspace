# Focused Rerun Protocol

This round is a protocol and training-budget diagnostic for the current Etard `P00` article pilot. It is not a full benchmark and it is not intended to replace the historical summary tables.

## Comparison targets

1. Current article pilot
- `FCNN`: `5` max epochs in `gate0_gate2_article_pilot`
- `ADT`: `3` max epochs in `gate0_gate2_article_pilot`

2. Focused rerun
- dataset: `etard_tf64_p00`
- subject: `P00`
- models: `fcnn`, `adt`
- same train/val/test split
- same unified scorer
- same result schema
- increased training budget closer to the older summaries

3. Historical summaries
- `experiments/summary_figures/unified_reference_main_summary.csv`
- `experiments/summary_figures/exact_reference_dataset_summary.csv`

## Scope and constraints

- This is a training-budget and protocol diagnostic, not a final benchmark.
- This round does not change data splits.
- This round does not change the scorer.
- This round does not change the result schema.
- This round only adjusts training budget and clearly declared training protocol parameters.
- This round does not treat old full-subject means and current `P00` results as the same evaluation layer.

## Focused rerun budget

- `FCNN`
  - `max_epochs`: `100`
  - `early_stopping_patience`: `10`
  - record:
    - `best_epoch`
    - `best_val_score`
    - `epochs_completed`

- `ADT`
  - `max_epochs`: `100`
  - `early_stopping_patience`: `10`
  - record:
    - `best_epoch`
    - `best_val_metric`
    - `epochs_completed`

## Expected diagnostic value

The main question is whether Etard `P00` low values in the current pilot are mostly explained by short training budget and protocol mismatch, or whether they continue to suggest a deeper issue such as implementation mismatch, alignment, windowing, or aggregation.
