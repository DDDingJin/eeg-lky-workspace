# Etard Subject-Specific Modelset Protocol Audit

- dataset: `etard_tf64`
- seed: `0`
- scope: pure EEG-only subject-specific models
- excluded: complete DECAF TwoBranch and LSTM
- scorer: `benchmark.scoring.pearson_on_valid`
- aggregation: full recording prediction aggregation -> recording Pearson -> subject mean

## `linear`

- input_contract: lagged EEG matrix, lags 0..50, target aligned after trim_valid_range
- normalization: linear-family StandardScaler fitted on train split only
- training_budget: max_fit_samples_per_split=12000
- checkpoint_selection: no regularization; validation Pearson recorded as auxiliary metadata
- identity_status: `budgeted baseline / local adaptation`
- uses_true_envelope_context_as_input: `False`

## `ridge`

- input_contract: lagged EEG matrix, lags 0..50, target aligned after trim_valid_range
- normalization: linear-family StandardScaler fitted on train split only
- training_budget: alphas=[0.001,0.01,0.1,1,10,100]
- checkpoint_selection: choose alpha by validation Pearson after lag trimming
- identity_status: `reference-compatible local subject-specific run`
- uses_true_envelope_context_as_input: `False`

## `lasso`

- input_contract: lagged EEG matrix, lags 0..50, target aligned after trim_valid_range
- normalization: linear-family StandardScaler fitted on train split only
- training_budget: alphas=[0.001,0.01,0.1], max_iter=5000, max_fit_samples_per_split=12000
- checkpoint_selection: choose alpha by validation Pearson
- identity_status: `budgeted baseline / local adaptation`
- uses_true_envelope_context_as_input: `False`

## `elasticnet`

- input_contract: lagged EEG matrix, lags 0..50, target aligned after trim_valid_range
- normalization: linear-family StandardScaler fitted on train split only
- training_budget: alphas=[0.001,0.01,0.1], l1_ratios=[0.2,0.5,0.8], max_iter=5000, max_fit_samples_per_split=12000
- checkpoint_selection: choose alpha and l1_ratio by validation Pearson
- identity_status: `budgeted baseline / local adaptation`
- uses_true_envelope_context_as_input: `False`

## `cca`

- input_contract: lagged EEG matrix, lags 0..50, target aligned after trim_valid_range
- normalization: CCA x/y scalers fitted on train split only
- training_budget: reference CCA hyperparameter search
- checkpoint_selection: choose PCA/component/reconstruction alpha by validation reconstruction correlation
- identity_status: `reference-compatible local subject-specific run`
- uses_true_envelope_context_as_input: `False`

## `fcnn`

- input_contract: 50-sample EEG window, target_index=last
- normalization: model-local train split handling; no test-fitted scaler
- training_budget: max_epochs=100, patience=10, batch_size=256
- checkpoint_selection: best validation Pearson checkpoint
- identity_status: `reference-compatible local subject-specific run`
- uses_true_envelope_context_as_input: `False`

## `dnn`

- input_contract: 50-sample EEG window, target_index=last
- normalization: model-local train split handling; no test-fitted scaler
- training_budget: max_epochs=100, patience=10, batch_size=256
- checkpoint_selection: best validation Pearson checkpoint
- identity_status: `local adaptation / not yet reference-protocol parity`
- uses_true_envelope_context_as_input: `False`

## `cnn`

- input_contract: 50-sample EEG window, target_index=last
- normalization: model-local train split handling; no test-fitted scaler
- training_budget: max_epochs=100, patience=10, batch_size=256
- checkpoint_selection: best validation Pearson checkpoint
- identity_status: `local adaptation / not yet reference-protocol parity`
- uses_true_envelope_context_as_input: `False`

## `eegnet`

- input_contract: 50-sample EEG window, target_index=last
- normalization: model-local train split handling; no test-fitted scaler
- training_budget: max_epochs=100, patience=10, batch_size=256
- checkpoint_selection: best validation Pearson checkpoint
- identity_status: `reference-compatible local subject-specific run`
- uses_true_envelope_context_as_input: `False`

## `adt`

- input_contract: 320-sample EEG window, hop_length=64
- normalization: model-local train split handling; no test-fitted scaler
- training_budget: max_epochs=100, patience=10, batch_size=32
- checkpoint_selection: best validation checkpoint
- identity_status: `reference-compatible local subject-specific run`
- uses_true_envelope_context_as_input: `False`

## `vlaai`

- input_contract: 50-sample EEG window, target_index=last
- normalization: model-local train split handling; no test-fitted scaler
- training_budget: max_epochs=100, patience=10, batch_size=64
- checkpoint_selection: best validation Pearson checkpoint
- identity_status: `local adaptation / not yet reference-protocol parity`
- uses_true_envelope_context_as_input: `False`

## `happyquokka`

- input_contract: 10s EEG chunk at 64 Hz, g_con=false, no true envelope context input
- normalization: model-local train split handling; no test-fitted scaler
- training_budget: max_epochs=100, patience=10, batch_size=4
- checkpoint_selection: best validation Pearson checkpoint
- identity_status: `local adaptation / not yet reference-protocol parity`
- uses_true_envelope_context_as_input: `False`
