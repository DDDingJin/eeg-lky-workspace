# Start Here

Last updated: 2026-07-13

This file is the primary handoff note for the current worktree. If chat memory conflicts with this file or current branch artifacts, trust the files.

## Current Main Task

- branch: `fix/ar-20260625-161300-a43831b-within-dataset-fixed-holdout-modelset-v1`
- worktree: `E:\decode\_fix_fixed_split_pooled20_modelset_v1`
- status: independent within-dataset training runner retired; dry-run planning and reuse audit only
- training status: no new smoke, zero-shot, pooled10, or formal training is authorized from this branch
- linear-family status: `linear`, `ridge`, `lasso`, `elasticnet`, and `cca` remain in the roster but are deferred pending validation-fix scope

## New Policy

Do not maintain a separate within-dataset fixed-holdout training implementation in this branch.

Future fixed subject-holdout work must be composed from two accepted chains:

- subject-specific full training code/configs, which own each model's adaptor, input contract, windowing, target, loss, optimizer, learning rate, batch size, max epochs, patience, early stopping, and scorer
- fixed subject-holdout zero-shot / pooled10 runners, which own split manifests, job/resume behavior, atomic artifact writes, checkpoint provenance, test-subject aggregation, pooled10 calibration, and progress output

The only fixed-holdout logic retained here is data selection:

- train: `train_subjects` train recordings
- val: `val_subjects` val recordings
- test: `test_subjects` test recordings
- windows/chunks must be generated only within their own recording and split

This branch must not add or restore a model factory, train loop, forward/evaluate function, model adapter, loss, optimizer, epoch/patience policy, scorer, or formal parameter loading from smoke configs.

## Fixed Splits

The accepted fixed subject-holdout manifests remain:

- `splits/subject_holdout_fixed_split_v1/weissbart_tf64.json`
- `splits/subject_holdout_fixed_split_v1/etard_tf64.json`

Subject lists:

- `weissbart_tf64`
  - train: `P00,P09,P02,P07,P11,P06,P04,P03`
  - val: `P12,P10`
  - test: `P08,P05,P01`
- `etard_tf64`
  - train: `P05,P02,P00,P01,P13,P04,P11,P07,P18,P06,P08,P12`
  - val: `P09,P15,P14,P19`
  - test: `P17,P16,P10,P03`

No new subject split should be generated.

## Current Script

- script: `scripts/run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py`
- config: `configs/benchmark/gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json`

The script is now a dry-run/planning/reuse-audit surface only. Supported stages:

- `smoke`
- `zero_shot`
- `pooled10`

It writes job plans, execution plans, data selection audits, reuse audits, schema reports, failure reports, run state, and manual command templates. It returns no-op status if invoked without `--dry-run-plan`, `--startup-only`, or `--verify-existing-output-only`.

## Accepted Chain References

- fixed-holdout zero-shot runner: `E:\decode\_fix_subject_holdout_pooled_finetune_v1\scripts\run_gate0_gate2_subject_holdout_fixed_split_v1.py`
- pooled10 runner: `E:\decode\_fix_subject_holdout_pooled_finetune_v1\scripts\run_gate0_gate2_subject_holdout_pooled_finetune_v1.py`

## Subject-Specific Source Configs

- `fcnn`: `scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_fcnn`; `configs/benchmark/gate0_gate2_model_expansion_v1.json::fcnn`
- `cnn`: `scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_cnn`; `configs/benchmark/gate0_gate2_model_expansion_v1.json::cnn`
- `eegnet`: `scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_eegnet`; `configs/benchmark/gate0_gate2_model_expansion_v1.json::eegnet`
- `adt`: `scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_adt`; `configs/benchmark/gate0_gate2_model_expansion_v1.json::adt`
- `vlaai`: `scripts/run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1.py::run_vlaai_full_eval`; `configs/benchmark/gate0_gate2_vlaai_happyquokka_training_budget_p00_v1.json::vlaai`
- `happyquokka`: `scripts/run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1.py::run_happyquokka_full_eval`; `configs/benchmark/gate0_gate2_subject_specific_happyquokka_seeded_weissbart_full_v1.json::happyquokka`

VLAAI and HappyQuokka must use their accepted full-training budgets: `max_epochs=100`, `early_stopping_patience=10`. VLAAI uses the accepted strict deterministic policy; HappyQuokka uses the seeded full-training policy. Do not use any `max_epochs=1` smoke config for formal parameters.

## Reuse Audit Status

Static reuse targets are `eegnet`, `fcnn`, and `adt`.

- EEGNet Weissbart: currently marked `lossless_reuse_ok` because accepted fixed-holdout and pooled10 manifests exist in `E:\decode\_fix_subject_holdout_pooled_finetune_v1`
- FCNN: not marked reusable; local checkpoints exist, but complete accepted run artifacts/config provenance were not found in this branch
- ADT: not marked reusable; local checkpoints exist, but complete accepted run artifacts/config provenance were not found in this branch
- EEGNet Etard: not marked reusable for the same provenance reason

Do not rerun or silently mix in non-lossless results. If reuse evidence differs later, update `reuse_audit.csv` by static verification first.

## Generated Dry-Run Artifacts

- zero-shot plan/audit: `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot`
- pooled10 plan/audit: `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_pooled10_plan`
- smoke plan/audit may be regenerated with `--stage smoke --dry-run-plan`, but no generic smoke execution should be started from this branch

Deprecated independent-runner execution artifacts were removed from Git, including old smoke completed jobs/logs/results and zero-shot engineering-smoke metrics.

## Manual Commands

Dry-run checks from this worktree:

```powershell
F:\miniconda\envs\decode-torch\python.exe -m py_compile scripts\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py --config configs\benchmark\gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json --startup-only
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py --config configs\benchmark\gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json --stage zero_shot --dry-run-plan
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py --config configs\benchmark\gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json --stage pooled10 --models cnn,vlaai,happyquokka --dry-run-plan
```

Accepted-chain zero-shot template for CNN/VLAAI/HappyQuokka on both datasets:

```powershell
cd E:\decode\_fix_subject_holdout_pooled_finetune_v1
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_fixed_split_v1.py --config <accepted_fixed_holdout_config_for_dataset_model_seed0.json> --device auto --dry-run-plan
```

Accepted-chain pooled10 template after accepted zero-shot checkpoint provenance exists:

```powershell
cd E:\decode\_fix_subject_holdout_pooled_finetune_v1
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_pooled_finetune_v1.py --config <accepted_pooled10_config_for_dataset_model_seed0.json> --device auto --dry-run-plan
```

## Resume Rule

- Continue on the current branch; do not create a new branch.
- Do not start training from this branch.
- Do not generate new subject splits.
- Do not commit raw data, checkpoints, model weights, `.pt/.pth/.npy/.npz/.h5/.mat`, `local_checkpoints/`, `external/`, `jobs/`, or `__pycache__`.
- Submit this branch for review after compile/startup/dry-run verification and push.
