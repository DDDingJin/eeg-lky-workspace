# Dataloader Contract Report

- output_dir: `experiments/gate0_gate2_loso_eegnet_full_manual_v1_clean`
- dataset: `weissbart_tf64`
- model: `eegnet`
- seed: `0`
- protocol: `full-scale pure LOSO`

## Contract statement

LOSO dataloader does not need to reuse the same subject split as subject-specific runs, but it must keep the same EEGNet batch/input/target/scorer contract.

## Subject-specific vs LOSO

- EEGNet model adapter: same `src/repro/mldecoders/models.py::EEGNetRegressor`
- Window dataset contract:
  `src/repro/reference_baselines.py::ReferenceWindowDataset`
  and
  `scripts/run_gate0_gate2_loso_resumable_runner_closure_v1.py::PreloadedWindowDataset`
  both return `(x, y)`, where `x` is an EEG window and `y` is a single-point envelope target
- Split difference: only subject membership
  subject-specific: train/val/test come from the same heldout subject
  LOSO: heldout subject is test only, all non-heldout subjects feed train/val
- Split difference must not change model input protocol, target definition, postprocessing, or scorer entry

## EEGNet contract details

- window_size:
  subject-specific = `50`
  LOSO = `50`
- target_index:
  subject-specific = `last`
  LOSO = `last`
- input tensor shape:
  subject-specific = `[batch, 64, 50]`
  LOSO = `[batch, 64, 50]`
- target tensor shape:
  subject-specific = `[batch]`
  LOSO = `[batch]`
- raw model output shape:
  subject-specific = `[batch]`
  LOSO = `[batch]`
- postprocessed prediction shape:
  subject-specific = `[n_windows]`
  LOSO = `[n_windows]`
- scorer input shape:
  subject-specific = `[n_windows]` prediction vs `[n_windows]` target
  LOSO = `[n_windows]` prediction vs `[n_windows]` target

## Evidence

- Subject-specific dataset class: `src/repro/reference_baselines.py::ReferenceWindowDataset`
  `x = eeg[start:start+window_size].T`
  `y = env[start + window_size - 1]` when `target_index == "last"`
- LOSO dataset class: `scripts/run_gate0_gate2_loso_resumable_runner_closure_v1.py::PreloadedWindowDataset`
  `x = eeg[start:start+window_size].T`
  `y = env[start + window_size - 1]` when `target_index == "last"`
- Subject-specific training/eval loaders:
  `src/repro/reference_baselines.py::train_dnn_reference_logged`
  `src/repro/reference_baselines.py::evaluate_dnn_reference_on_test`
- LOSO training/eval loaders:
  `scripts/run_gate0_gate2_loso_resumable_runner_closure_v1.py::fit_job`
  `scripts/run_gate0_gate2_loso_resumable_runner_closure_v1.py::predict_window_model`

## Conclusion

- The current EEGNet LOSO runner keeps the same batch/input/target/scorer contract as the subject-specific EEGNet route
- The only allowed difference is subject membership in the split
- The current LOSO engineering changes only add manual-run, resume, progress, profiling, and incremental-save behavior without changing the scientific protocol
