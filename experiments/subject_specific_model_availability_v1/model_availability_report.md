# Subject-Specific Model Availability Audit v1

Date: 2026-07-09

This audit is intentionally read-only. No training was started, no benchmark jobs were launched, and no model weights were added to Git.

## Scope

Target model set:

- `linear`
- `ridge`
- `lasso`
- `elasticnet`
- `cca`
- `fcnn`
- `dnn`
- `cnn`
- `eegnet`
- `adt`
- `vlaai`
- `happyquokka`
- `decaf`

Audit questions:

- Does the implementation already exist in this repo or current environment?
- Is there an exact registry row?
- Does the current subject-specific runner already support it?
- Can the model be smoke-tested with only a small adapter closure?
- Does it require an external dependency?

## Current subject-specific execution surface

Primary subject-specific runner:

- `scripts/run_gate0_gate2_full_subject_single_seed.py`

Current native branches inside `run_job(...)`:

- `ridge`
- `cca`
- `fcnn`
- `dnn`
- `cnn`
- `eegnet`
- `adt`

This means the current subject-specific runner does **not** natively support:

- `linear`
- `lasso`
- `elasticnet`
- `vlaai`
- `happyquokka`
- `decaf`

## Reusable accepted subject-specific results

Accepted compact subject-specific results are already present for these 7 models and should be reused rather than rerun:

- `ridge`
- `cca`
- `fcnn`
- `dnn`
- `cnn`
- `eegnet`
- `adt`

Evidence:

- `experiments/gate0_gate2_model_expansion_v1/subject_metrics.csv` contains rows for `ridge, cca, fcnn, dnn, cnn, eegnet, adt`.
- `experiments/gate0_gate2_full_subject_single_seed/subject_metrics.csv` contains earlier accepted rows for `ridge, cca, fcnn, adt`.

## Per-model status

### Ready now without new adapter

These are already supported by the current subject-specific runner and already have accepted compact subject-specific results:

- `ridge`
- `cca`
- `fcnn`
- `dnn`
- `cnn`
- `eegnet`
- `adt`

Dependency note:

- `dnn` and `cnn` rely on shared upstream code at `E:/decode/external/upstream/mldecoders/pipeline/dnn.py`.
- That dependency is present in the current environment, so both remain `ready_for_smoke=true`.

Registry note:

- `eegnet` has implementation and runner support, but no exact `eegnet` row was found in `registry/model_registry.csv`.

### Likely ready after a small subject-specific adapter

#### `linear`

Local implementation exists via:

- `src/repro/reference_baselines.py::fit_reference_linear_variant`
- `src/repro/reference_baselines.py::evaluate_reference_linear_variant`

Assessment:

- Can reuse the existing ridge/TRF-style subject-specific flow.
- No new external dependency is needed.
- Current blocker is runner integration, not algorithm availability.

Status:

- `needs_new_adapter=true`
- `ready_for_smoke=true`

#### `lasso`

Local lasso-family code exists via:

- `src/repro/mldecoders/linear_baselines.py::fit_avgdec_lasso_model_fast`
- `src/repro/mldecoders/linear_baselines.py::fit_avgcorr_lasso_model`

Assessment:

- The sparse linear family is locally present.
- It can likely reuse the same subject-specific ridge/TRF execution pattern.
- The current subject-specific runner has no `lasso` branch.
- There is also no exact `lasso` row in `registry/model_registry.csv`.

Important caveat:

- `scripts/run_reference_baselines.py` exposes `lasso` and `elastic_net` CLI tokens, but only the lasso-family local fit functions were confirmed in this branch.

Status:

- `needs_new_adapter=true`
- `ready_for_smoke=true`

### Not ready: implementation or contract gap remains

#### `elasticnet`

Findings:

- No exact `elasticnet` registry row.
- No exact local `ElasticNet`-based fit/eval implementation was found in this branch.
- `scripts/run_reference_baselines.py` includes an `elastic_net` token, but the corresponding local fit/eval implementation was not found in `src/repro/reference_baselines.py`.

Assessment:

- This is not just a missing adapter.
- It first needs a real local implementation closure or restoration of the missing sparse-decoder functions.

Status:

- `implementation_exists=false`
- `ready_for_smoke=false`

#### `vlaai`

Local code exists:

- `src/repro/vlaai_exact.py::VLAAIExactOfficial`
- `src/repro/mldecoders/models.py::VLAAIExactOfficialRegressor`

Observed contract:

- Exact reference model forward expects `(batch, time, channels)` and returns a sequence `(batch, time, output_dim)`.
- The current subject-specific runner is built around scalar-window prediction with final alignment equivalent to `[batch, 64, window] -> [batch]`.

Assessment:

- This is available in-repo, not unavailable.
- But it needs a dedicated subject-specific adapter to reconcile the sequence-output reference path with the scalar-window subject-specific contract.

Status:

- `implementation_exists=true`
- `subject_specific_runner_supported=false`
- `needs_new_adapter=true`
- `ready_for_smoke=false`

#### `happyquokka`

Local wrapper code exists:

- `src/repro/happyquokka_reference.py`

External dependency:

- `external/upstream/HappyQuokka_system_for_EEG_Challenge/models/FFT_block.py`

Observed contract:

- Training wrapper uses 10-second chunks by default: `win_len_seconds * sample_rate = 10 * 64 = 640`.
- Evaluation is segment-based rather than the current subject-specific scalar 50-sample window contract.

Assessment:

- This is available in the current environment, but not natively integrated into the subject-specific runner.
- It needs both a subject-specific adapter and continued access to the upstream HappyQuokka dependency.

Status:

- `implementation_exists=true`
- `subject_specific_runner_supported=false`
- `needs_new_adapter=true`
- `external_dependency_needed=true`
- `ready_for_smoke=false`

#### `decaf`

External audit only; no code was copied into this repo.

Accessible repositories:

- Stub repo: `https://github.com/JHU-LCAP/DECAF`
- Actual code repo linked by the stub README: `https://github.com/carankt/DECAF`

Observed DECAF metadata:

- License: MIT
- Top-level training entry points:
  - `train_model.py`
  - `train_all_models.sh`
  - `fine_tune_subjects.py`
  - `train_and_finetune.py`
- Repo structure includes:
  - `src/data`
  - `src/models`
  - `src/train`

README-declared dependencies:

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `wandb`

Model-family note from DECAF README:

- `HappyQuoka`
- `VLAAI`
- baseline decoder
- dynamic GRU/LSTM variants

Assessment:

- DECAF appears to build its own model stack and registry via `src.models.model_factory`.
- It references HappyQuoka as one of the model families, so there is conceptual dependency/reuse in the external project design.
- There is no in-repo DECAF implementation in the current benchmark worktree.
- Compatibility with the current 64 Hz / 50-sample scalar-window subject-specific contract is **not** immediate:
  - DECAF uses its own dataloader/trainer/config stack.
  - The README and training entry points indicate a subject-independent / fine-tuning workflow, but not direct compatibility with the current `ReferenceWindowDataset` contract.

Status:

- `implementation_exists=false` in this repo
- `external_dependency_needed=true`
- `needs_new_adapter=true`
- `ready_for_smoke=false`

## Registry summary

Exact rows found in `registry/model_registry.csv`:

- `linear`
- `ridge`
- `cca`
- `fcnn`
- `dnn`
- `cnn`
- `adt`
- `vlaai`
- `happyquokka`

Exact rows not found:

- `lasso`
- `elasticnet`
- `eegnet`
- `decaf`

## Recommended interpretation of readiness

### Reuse only, do not rerun

- `ridge`
- `cca`
- `fcnn`
- `dnn`
- `cnn`
- `eegnet`
- `adt`

### Small adapter candidates

- `linear`
- `lasso`

### Engineering closure still needed before any smoke

- `elasticnet`
- `vlaai`
- `happyquokka`
- `decaf`

## Large-file and safety check

- No training was started.
- No checkpoint files were committed.
- No raw data or prediction dumps were generated by this audit.
- Current `git status` showed an untracked local directory: `local_checkpoints/`
- This audit does not stage or commit `local_checkpoints/`.
