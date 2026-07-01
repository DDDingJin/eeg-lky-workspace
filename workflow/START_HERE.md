# Start Here

Last updated: 2026-07-01

This is the single entry file for resuming work in this worktree.

## Current Active Task

- active round: `model-expansion-v1`
- requested base branch: `fix/ar-20260625-161300-a43831b-multi-seed-v1`
- requested base commit: `dee590b26aca180637b920b9579a59daa9ba176c`
- requested new branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1`
- current status:
  - local branch `fix/ar-20260625-161300-a43831b-model-expansion-v1` created from `dee590b26aca180637b920b9579a59daa9ba176c`
  - model-expansion config, runner, smoke configs, schema validation, writing-material note, and model identity note added locally
  - smoke runs passed for `cnn`, `dnn`, and `eegnet` under the unified full-subject schema
  - full model-expansion v1 run is complete locally
  - aggregate output directory: `experiments/gate0_gate2_model_expansion_v1`
  - schema validation passed for the final aggregate output
  - current user request is to commit and push only compact reviewable artifacts for this round

## Current Worktree

- worktree: `E:\decode\_fix_gate0_gate2_real`
- branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1`
- latest commit: `dee590b26aca180637b920b9579a59daa9ba176c`
- remote branch target: `origin/fix/ar-20260625-161300-a43831b-model-expansion-v1`

## Current State

- The full-subject multi-seed benchmark v1 is complete and serves as the reused base.
- The completed seed set is `0, 42, 2026`.
- The covered datasets are `weissbart_tf64` and `etard_tf64`.
- The current model-expansion coverage is `ridge`, `cca`, `fcnn`, `adt`, `dnn`, `cnn`, and `eegnet`.
- Final job accounting for `model-expansion-v1` is `693/693` successful and `0` failed.

## What Just Happened

- The previous interrupted session did not lose the benchmark logic.
- The main interruption point for this round was after local model-expansion artifacts were produced but before a clean reviewable commit was prepared.
- The branch now contains the local model-expansion code, configs, summaries, ledgers, schema validation, and resume log updates.

## Read In This Order

1. `workflow/START_HERE.md`
2. `workflow/skill_alignment.md`
3. `experiments/gate0_gate2_model_expansion_v1/run_manifest.json`
4. `experiments/gate0_gate2_model_expansion_v1/result_summary.md`
5. `experiments/gate0_gate2_model_expansion_v1/model_identity_check.md`
6. `README.md`

## Key Files

- machine-readable run state:
  - `experiments/gate0_gate2_model_expansion_v1/run_manifest.json`
- human-readable summary:
  - `experiments/gate0_gate2_model_expansion_v1/result_summary.md`
- model identity audit note:
  - `experiments/gate0_gate2_model_expansion_v1/model_identity_check.md`
- aggregate metrics:
  - `experiments/gate0_gate2_model_expansion_v1/dataset_metrics_across_seeds.csv`
- review-loop lock:
  - `workflow/skill_lock.json`
- local skill alignment note:
  - `workflow/skill_alignment.md`

## Commands To Re-Establish Context

Run these from `E:\decode\_fix_gate0_gate2_real`:

```powershell
git status --short --branch
git rev-parse HEAD
Get-Content workflow\START_HERE.md
Get-Content experiments\gate0_gate2_model_expansion_v1\run_manifest.json
Get-Content experiments\gate0_gate2_model_expansion_v1\result_summary.md
Get-Content experiments\gate0_gate2_model_expansion_v1\model_identity_check.md
```

## Benchmark Rebuild Command

If the model-expansion aggregate summary ever needs to be rebuilt from existing per-job artifacts, use:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts/run_gate0_gate2_model_expansion_v1.py --config configs/benchmark/gate0_gate2_model_expansion_v1.json --device cuda --resume --skip-existing
```

## Resume Rule

- Do not assume chat memory is authoritative.
- Reconstruct state from this file, the current git commit, and the benchmark manifests.
- Before any new rerun or reviewer response, verify that the branch and commit in this file still match the checked-out state.

## Expected Next Step

- Current next step is to commit and push the compact `model-expansion-v1` artifacts for reviewer inspection.
- Reuse of `ridge`, `cca`, `fcnn`, and `adt` has already been completed; no rerun is needed for those base models.
- Do not upload prediction dumps, checkpoints, `.pt/.pth/.npy/.npz/.h5/.mat`, or large per-job directories.
- Keep the review/execution trace version-locked to the existing local skill lock and GitHub rule commit.
