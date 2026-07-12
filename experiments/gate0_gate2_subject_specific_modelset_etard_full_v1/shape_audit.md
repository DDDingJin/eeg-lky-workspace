# Shape Audit

- This is a contract preflight; no model training was started.

## `linear`
- category: `lagged_linear_family`
- EEG input: lagged EEG design matrix from 64 channels x lags 0..50
- target: trimmed envelope vector aligned after lag valid-range trim
- output: 1D envelope prediction vector
- true_envelope_context_input: `False`

## `ridge`
- category: `lagged_linear_family`
- EEG input: lagged EEG design matrix from 64 channels x lags 0..50
- target: trimmed envelope vector aligned after lag valid-range trim
- output: 1D envelope prediction vector
- true_envelope_context_input: `False`

## `lasso`
- category: `lagged_linear_family`
- EEG input: lagged EEG design matrix from 64 channels x lags 0..50
- target: trimmed envelope vector aligned after lag valid-range trim
- output: 1D envelope prediction vector
- true_envelope_context_input: `False`

## `elasticnet`
- category: `lagged_linear_family`
- EEG input: lagged EEG design matrix from 64 channels x lags 0..50
- target: trimmed envelope vector aligned after lag valid-range trim
- output: 1D envelope prediction vector
- true_envelope_context_input: `False`

## `cca`
- category: `lagged_linear_family`
- EEG input: lagged EEG design matrix from 64 channels x lags 0..50
- target: trimmed envelope vector aligned after lag valid-range trim
- output: 1D envelope prediction vector
- true_envelope_context_input: `False`

## `fcnn`
- category: `50_sample_window`
- EEG input: [batch, 64, 50] or adapter-equivalent 50-sample EEG window
- target: scalar last-sample envelope target
- output: scalar envelope prediction per window
- true_envelope_context_input: `False`

## `dnn`
- category: `50_sample_window`
- EEG input: [batch, 64, 50] or adapter-equivalent 50-sample EEG window
- target: scalar last-sample envelope target
- output: scalar envelope prediction per window
- true_envelope_context_input: `False`

## `cnn`
- category: `50_sample_window`
- EEG input: [batch, 64, 50] or adapter-equivalent 50-sample EEG window
- target: scalar last-sample envelope target
- output: scalar envelope prediction per window
- true_envelope_context_input: `False`

## `eegnet`
- category: `50_sample_window`
- EEG input: [batch, 64, 50] or adapter-equivalent 50-sample EEG window
- target: scalar last-sample envelope target
- output: scalar envelope prediction per window
- true_envelope_context_input: `False`

## `adt`
- category: `320_sample_window_hop64`
- EEG input: [batch, 64, 320] ADT window
- target: window-level envelope target under ADT exact adapter
- output: window-level envelope prediction
- true_envelope_context_input: `False`

## `vlaai`
- category: `50_sample_window`
- EEG input: [batch, 64, 50] or adapter-equivalent 50-sample EEG window
- target: scalar last-sample envelope target
- output: scalar envelope prediction per window
- true_envelope_context_input: `False`

## `happyquokka`
- category: `10s_chunk`
- EEG input: [batch, 640, 64] 10s EEG chunk at 64 Hz
- target: [batch, 640, 1] 10s envelope chunk
- output: [batch, 640, 1] envelope prediction
- true_envelope_context_input: `False`
