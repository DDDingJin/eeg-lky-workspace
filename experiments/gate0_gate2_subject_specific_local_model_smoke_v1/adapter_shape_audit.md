# Adapter Shape Audit

- protocol: `gate0_gate2_subject_specific_local_model_smoke_v1`
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
- input_shape: `[256, 3200]`
- raw_output_shape: `[256]`
- postprocessed_prediction_shape: `[256]`
- target_shape: `[256]`
- scorer_input_shape: `[256]`
- selected_hyperparameters: `{"end_lag": 50, "max_eval_windows_per_recording": 256, "max_fit_samples_per_split": 12000, "start_lag": 0}`
- validation_score: `0.087728`
- subject_metric: `0.081507`

## `lasso`

- status: `success`
- input_shape: `[256, 3200]`
- raw_output_shape: `[256]`
- postprocessed_prediction_shape: `[256]`
- target_shape: `[256]`
- scorer_input_shape: `[256]`
- selected_hyperparameters: `{"alpha": 0.001, "end_lag": 50, "max_eval_windows_per_recording": 256, "max_fit_samples_per_split": 12000, "start_lag": 0}`
- validation_score: `0.124385`
- subject_metric: `0.101177`

## `elasticnet`

- status: `success`
- input_shape: `[256, 3200]`
- raw_output_shape: `[256]`
- postprocessed_prediction_shape: `[256]`
- target_shape: `[256]`
- scorer_input_shape: `[256]`
- selected_hyperparameters: `{"alpha": 0.001, "end_lag": 50, "l1_ratio": 0.8, "max_eval_windows_per_recording": 256, "max_fit_samples_per_split": 12000, "start_lag": 0}`
- validation_score: `0.124577`
- subject_metric: `0.101458`

## `vlaai`

- status: `success`
- input_shape: `[1, 64, 50]`
- raw_output_shape: `[1]`
- postprocessed_prediction_shape: `[990]`
- target_shape: `[990]`
- scorer_input_shape: `[990]`
- selected_hyperparameters: `{"batch_size": 64, "best_epoch": 0, "learning_rate": 0.0001, "max_epochs": 1, "max_eval_windows_per_recording": 256, "weight_decay": 0.0001, "window_size": 50}`
- best_val_score: `0.059850`
- subject_metric: `0.105685`

## `happyquokka`

- status: `success`
- input_shape: `[1, 640, 64]`
- raw_output_shape: `[1, 640, 1]`
- postprocessed_prediction_shape: `[1039]`
- target_shape: `[1039]`
- scorer_input_shape: `[1039]`
- selected_hyperparameters: `{"batch_size": 4, "best_epoch": 1, "dropout": 0.3, "g_con": false, "input_length": 640, "lamda": 0.2, "learning_rate": 0.0005, "max_epochs": 1}`
- best_val_score: `-0.068566`
- subject_metric: `-0.031750`

