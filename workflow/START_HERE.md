# Start Here

Last updated: 2026-07-13

This file is the primary handoff note for the current worktree. If chat memory conflicts with this file or current branch artifacts, trust the files.

## Current Main Task

- branch: `fix/ar-20260625-161300-a43831b-within-dataset-fixed-holdout-modelset-v1`
- local HEAD: `ee5e118`
- worktree: `E:\decode\_fix_fixed_split_pooled20_modelset_v1`
- task status: runner implementation and bounded engineering evidence completed; ready for review
- smoke status: 22-job roster planned; 12 nonlinear jobs completed; 10 linear-family jobs deferred; 0 failures
- zero-shot status: formal zero-shot not started; isolated 2-job FCNN engineering smoke completed
- remote push status: pending at this handoff

## What This Branch Is For

Build a unified within-dataset cross-subject modelset runner on top of the accepted fixed subject-holdout split manifests.

Datasets:

- `weissbart_tf64`
- `etard_tf64`

Main 11-method roster:

- `linear`
- `ridge`
- `lasso`
- `elasticnet`
- `cca`
- `fcnn`
- `cnn`
- `eegnet`
- `adt`
- `vlaai`
- `happyquokka`

Do not add `dnn` to this roster. `dnn` remains excluded because it was already audited as `functional_alias_of=fcnn`.

## Fixed Subject-Holdout Splits To Reuse

These manifests were copied into this worktree and must be reused exactly as-is:

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

No new subject split should be generated. Do not modify these manifests.

## New Runner And Config

- runner:
  - `scripts/run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py`
- config:
  - `configs/benchmark/gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json`

Output dirs:

- smoke preflight artifacts:
  - `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_smoke`
- zero-shot preflight artifacts:
  - `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot`
- isolated zero-shot engineering-smoke artifacts:
  - `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot_engineering_smoke`

Checkpoint root reserved for future real runs only:

- `local_checkpoints/within_dataset_fixed_holdout_modelset_v1`

## What The Runner Currently Supports

CLI flags already implemented:

- `--datasets`
- `--models`
- `--stage smoke|zero_shot`
- `--resume`
- `--max-jobs`
- `--startup-only`
- `--dry-run-plan`
- `--engineering-smoke`
- `--device`

Important current behavior:

- `--stage smoke` can execute real selected nonlinear jobs
- smoke loads fixed train/val/test subject splits and performs train forward/backward/optimizer step, val forward, and held-out test aggregation
- smoke writes smoke-only artifacts and does not write formal subject/dataset metrics or checkpoints
- `--stage zero_shot` has a real execution path for nonlinear jobs, with metrics and local-only checkpoint writes
- formal zero-shot has not been started
- `--engineering-smoke` isolates capped zero-shot engineering evidence from formal zero-shot artifacts
- linear-family models remain in roster/plan but real execution is deferred pending validation-fix scope confirmation

## Preflight And Engineering Result Already Achieved

Both stages passed preflight:

- smoke:
  - planned jobs: `22`
  - schema passed: `true`
  - training started: `true`
  - completed nonlinear jobs: `12`
  - deferred linear-family jobs: `10`
  - failures: `0`
- zero_shot:
  - planned jobs: `22`
  - schema passed: `true`
  - training started: `false`
- zero_shot engineering smoke:
  - isolated output dir: `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot_engineering_smoke`
  - completed jobs: `2` (`weissbart_tf64:fcnn`, `etard_tf64:fcnn`)
  - subject metric rows: `7`
  - recording metric rows: `189`
  - dataset metric rows: `2`
  - failures: `0`
  - checkpoint path: `local_checkpoints/within_dataset_fixed_holdout_modelset_v1/zero_shot_engineering_smoke/...`

The 22 jobs for each stage are:

- `2 datasets x 11 models = 22`

Current dry-run job identity format:

- `dataset:model:seed0:stage=<stage>`

Examples:

- `weissbart_tf64:vlaai:seed0:stage=smoke`
- `etard_tf64:happyquokka:seed0:stage=zero_shot`

## Zero-Shot Defer Rule

Formal zero-shot jobs currently deferred pending separate confirmation of linear-family validation-fix scope:

- `linear`
- `ridge`
- `lasso`
- `elasticnet`
- `cca`

This defer applies to both datasets, so currently deferred zero-shot jobs count is:

- `10`

Zero-shot jobs not deferred after this preflight:

- `fcnn`
- `cnn`
- `eegnet`
- `adt`
- `vlaai`
- `happyquokka`

Across both datasets, that leaves:

- `12` non-deferred neural/nonlinear zero-shot jobs

Do not silently remove the linear-family models from the roster. They remain in smoke planning and future official zero-shot scope, but they are not yet allowed to start.

## Determinism Note

Do not retrofit strict deterministic execution into this runner in this branch.

Read:

- `workflow/DETERMINISM_POLICY_PENDING.md`
- `workflow/START_HERE_1.md`

Those files document the accepted VLAAI deterministic reproducibility closure and the pending policy integration that should happen later for formal neural zero-shot jobs.

## Files Generated In This Round

Smoke preflight:

- `smoke_job_plan.csv`
- `smoke_execution_plan.csv`
- `split_preflight.csv`
- `shape_checkpoint_preflight.csv`
- `schema_validation_report.json`
- `failure_report.json`
- `deferred_jobs.json`
- `run_state.json`
- `completed_jobs.json`
- `smoke_job_results.csv`
- `logs/runner.log`
- `preflight_report.md`

Zero-shot preflight:

- `zero_shot_job_plan.csv`
- `zero_shot_execution_plan.csv`
- `split_preflight.csv`
- `shape_checkpoint_preflight.csv`
- `schema_validation_report.json`
- `failure_report.json`
- `run_state.json`
- `preflight_report.md`

Zero-shot engineering smoke:

- `zero_shot_job_plan.csv`
- `zero_shot_execution_plan.csv`
- `split_preflight.csv`
- `shape_checkpoint_preflight.csv`
- `schema_validation_report.json`
- `failure_report.json`
- `run_state.json`
- `completed_jobs.json`
- `subject_metrics.csv`
- `recording_metrics.csv`
- `dataset_metrics.csv`
- `logs/runner.log`

## What Must Happen Next

1. Push this branch if it is not already on origin.
2. Let review/decision happen on:
   - runner execution correctness
   - split reuse correctness
   - smoke accounting correctness: 12 completed nonlinear jobs + 10 deferred linear-family jobs
   - smoke artifact boundary: no formal subject/dataset metrics and no checkpoint
   - zero-shot engineering-smoke isolation from formal zero-shot artifacts
   - zero-shot defer policy for linear-family models
3. Do not start formal zero-shot until review approves the runner and artifact boundaries.
4. Only after smoke review passes, implement a separate deterministic-policy patch for neural zero-shot jobs if required.
5. Only then begin approved formal `zero_shot` execution, keeping the linear-family defer rule unless separately lifted.

## Current Resume Rule

If a future Codex session opens only this file, the intended behavior is:

1. Treat this branch as ready for review.
2. Do not start formal zero-shot automatically.
3. First review smoke and zero-shot engineering-smoke artifacts.
4. If this task is resumed, continue from reviewer findings or formal zero-shot approval.

## Manual Commands To Re-Establish Context

From `E:\decode\_fix_fixed_split_pooled20_modelset_v1`:

```powershell
git status --short --branch
git rev-parse HEAD
Get-Content workflow\START_HERE.md
Get-Content workflow\START_HERE_1.md
Get-Content experiments\gate0_gate2_within_dataset_fixed_holdout_modelset_v1_smoke\schema_validation_report.json
Get-Content experiments\gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot\schema_validation_report.json
Get-Content experiments\gate0_gate2_within_dataset_fixed_holdout_modelset_v1_smoke\split_preflight.csv
Get-Content experiments\gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot\shape_checkpoint_preflight.csv
```

## Manual Push Command

If this branch still has not been pushed:

```powershell
git push -u origin fix/ar-20260625-161300-a43831b-within-dataset-fixed-holdout-modelset-v1
```

## Manual Smoke Command

Use:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py --config configs\benchmark\gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json --stage smoke --device auto --resume
```

At this handoff, this command should report no pending work except already deferred linear-family jobs because the 12 nonlinear smoke jobs have completed.

## Resume Rule

- Trust the current branch files over chat memory.
- Do not start training from this branch unless the user explicitly asks.
- Do not generate new subject splits.
- Do not commit raw data, prediction dumps, checkpoints, model weights, `.pt/.pth/.npy/.npz/.h5/.mat`, `local_checkpoints/`, `jobs/`, or `__pycache__`.
