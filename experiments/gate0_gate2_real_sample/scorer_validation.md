# Scorer Validation

- Dataset: `hugo_sample_tf64_p00`
- Subject: `P00`
- Artifact scope: `pipeline_validation_real_sample`
- All benchmark rows in `recording_metrics.csv` and `subject_metrics.csv` were produced by the shared scorer in `src/benchmark/scoring.py`.
- Model-specific direct metrics were recorded only as auxiliary metadata in `run_manifest.json` and were not used as the final reported schema output.
- `prediction_samples.csv` files were written per model under `experiments/gate0_gate2_real_sample/model_outputs/`.

## Direct-vs-unified checks

- `ridge` direct Pearson: `0.186105`
- `cca` direct reconstruction correlation: `0.153673`
- `fcnn` direct Pearson: `0.197046`
