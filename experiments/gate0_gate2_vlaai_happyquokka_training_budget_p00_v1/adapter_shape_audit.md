# Adapter Shape Audit

- protocol: `gate0_gate2_vlaai_happyquokka_training_budget_p00_v1`
- dataset: `weissbart_tf64`
- subject: `P00`
- seed: `0`

## Status Overview
- `vlaai`: `success`
- `happyquokka`: `success`

## `vlaai`

- status: `success`
- model_family_contract: `vlaai_local_adapter`
- input_shape: `[1, 64, 50]`
- raw_output_shape: `[1]`
- postprocessed_prediction_shape: `[990]`
- target_shape: `[990]`
- scorer_input_shape: `[990]`
- best_val_score: `0.108968`
- subject_metric: `0.081639`

## `happyquokka`

- status: `success`
- model_family_contract: `10s_chunk`
- input_shape: `[1, 640, 64]`
- raw_output_shape: `[1, 640, 1]`
- postprocessed_prediction_shape: `[640]`
- target_shape: `[640]`
- scorer_input_shape: `[640]`
- best_val_score: `0.134845`
- subject_metric: `0.085697`

