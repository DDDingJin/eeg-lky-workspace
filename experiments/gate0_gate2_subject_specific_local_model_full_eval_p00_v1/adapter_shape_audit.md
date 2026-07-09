# Adapter Shape Audit

- protocol: `gate0_gate2_subject_specific_local_model_full_eval_p00_v1`
- dataset: `weissbart_tf64`
- subject: `P00`
- seed: `0`

## Status Overview
- `linear`: `success`
- `lasso`: `success`
- `elasticnet`: `success`
- `vlaai`: `success`
- `happyquokka`: `success`

## `linear`

- status: `success`
- model_family_contract: `lag_matrix_trf`
- input_shape: `[990, 3200]`
- raw_output_shape: `[990]`
- postprocessed_prediction_shape: `[990]`
- target_shape: `[990]`
- scorer_input_shape: `[990]`
- train_fit_samples: `12000`
- val_fit_samples: `12000`
- best_val_score: `0.087728`
- subject_metric: `0.087138`

## `lasso`

- status: `success`
- model_family_contract: `lag_matrix_trf`
- input_shape: `[990, 3200]`
- raw_output_shape: `[990]`
- postprocessed_prediction_shape: `[990]`
- target_shape: `[990]`
- scorer_input_shape: `[990]`
- train_fit_samples: `12000`
- val_fit_samples: `12000`
- best_val_score: `0.124385`
- subject_metric: `0.105440`

## `elasticnet`

- status: `success`
- model_family_contract: `lag_matrix_trf`
- input_shape: `[990, 3200]`
- raw_output_shape: `[990]`
- postprocessed_prediction_shape: `[990]`
- target_shape: `[990]`
- scorer_input_shape: `[990]`
- train_fit_samples: `12000`
- val_fit_samples: `12000`
- best_val_score: `0.124577`
- subject_metric: `0.105726`

## `vlaai`

- status: `success`
- model_family_contract: `vlaai_local_adapter`
- input_shape: `[1, 64, 50]`
- raw_output_shape: `[1]`
- postprocessed_prediction_shape: `[990]`
- target_shape: `[990]`
- scorer_input_shape: `[990]`
- best_val_score: `0.060682`
- subject_metric: `0.089347`

## `happyquokka`

- status: `success`
- model_family_contract: `10s_chunk`
- input_shape: `[1, 640, 64]`
- raw_output_shape: `[1, 640, 1]`
- postprocessed_prediction_shape: `[640]`
- target_shape: `[640]`
- scorer_input_shape: `[640]`
- best_val_score: `-0.068566`
- subject_metric: `-0.031750`

