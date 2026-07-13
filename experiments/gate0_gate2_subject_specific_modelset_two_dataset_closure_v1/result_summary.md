# Two-Dataset Subject-Specific Modelset Closure

## Scope
- Weissbart: 13 subjects x 12 models x seed0
- Etard: 20 subjects x 12 models x seed0
- Planned/completed/failed: 396/396/0

## Protocol Markers
- linear/lasso/elasticnet: budgeted linear-family baseline
- vlaai: local adaptation
- happyquokka: seeded local 10s-chunk adaptation
- Full DECAF TwoBranch is not part of this pure EEG-only model set

## Validation
- planned_jobs_396: `true`
- completed_jobs_396: `true`
- failed_jobs_zero: `true`
- subject_metrics_unique_dataset_subject_model_seed: `true`
- dataset_metrics_unique_dataset_model_seed: `true`
- dataset_metrics_n_subjects_correct: `true`
- recording_metrics_cover_all_completed_jobs: `true`
- source_schema_all_passed: `true`
- no_decaf_twobranch_in_pure_eeg_modelset: `true`

## Source Artifacts
- weissbart_reference_subject_rows: 91
- weissbart_local_subject_rows: 52
- weissbart_seeded_happyquokka_subject_rows: 13
- etard_subject_rows: 240

## Dataset Metrics
See `combined_dataset_metrics.csv` for raw mean/std/median/min/max per dataset/model/seed.
