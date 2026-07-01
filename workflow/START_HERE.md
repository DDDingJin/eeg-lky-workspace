# Start Here

Last updated: 2026-07-01

This is the single entry file for resuming work in this worktree.

## Current Active Task

- active round: `model-expansion-v1-metadata-closure`
- requested base branch: `fix/ar-20260625-161300-a43831b-multi-seed-v1`
- requested base commit: `dee590b26aca180637b920b9579a59daa9ba176c`
- published result branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1`
- published result commit: `68f87a4c0005e4556a51b83cd42e73908edc44ad`
- requested new branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure`
- current status:
  - published branch `fix/ar-20260625-161300-a43831b-model-expansion-v1` exists on `origin` at `68f87a4c0005e4556a51b83cd42e73908edc44ad`
  - `model-expansion-v1` experiment results are already pushed and can be used as the next analysis baseline
  - current round is a metadata-only closure for reviewable artifact consistency
  - no model rerun is allowed in this round
  - target aggregate output directory remains `experiments/gate0_gate2_model_expansion_v1`
  - current task is to align manifest-referenced compact artifacts with what is actually published for review

## Current Worktree

- worktree: `E:\decode\_fix_gate0_gate2_real`
- branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure`
- published baseline branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1`
- published baseline commit: `68f87a4c0005e4556a51b83cd42e73908edc44ad`
- remote branch target: `origin/fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure`

## Current State

- The full-subject multi-seed benchmark v1 is complete and serves as the reused base.
- The completed seed set is `0, 42, 2026`.
- The covered datasets are `weissbart_tf64` and `etard_tf64`.
- The current model-expansion coverage is `ridge`, `cca`, `fcnn`, `adt`, `dnn`, `cnn`, and `eegnet`.
- Final job accounting for `model-expansion-v1` is `693/693` successful and `0` failed.
- The current closure task is about metadata publication consistency, not result regeneration.

## What Just Happened

- The previous interrupted session did not lose the benchmark logic.
- The main metadata mismatch after publication was that `run_manifest.json` referenced compact ledger and metric files that were still local-only.
- This closure round only reconciles reviewable artifact publication and handoff metadata against the already-pushed experiment commit.

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
- compact ledgers and dense metrics:
  - `experiments/gate0_gate2_model_expansion_v1/job_ledger.csv`
  - `experiments/gate0_gate2_model_expansion_v1/job_ledger.json`
  - `experiments/gate0_gate2_model_expansion_v1/recording_metrics.csv`
  - `experiments/gate0_gate2_model_expansion_v1/subject_metrics.csv`
  - `experiments/gate0_gate2_model_expansion_v1/skipped_models.md`
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
git show 68f87a4c0005e4556a51b83cd42e73908edc44ad --stat --no-patch
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
- Before any new rerun or reviewer response, verify that the published baseline branch and commit in this file still match the intended review target.

## Expected Next Step

- Current next step is to publish the missing compact manifest-referenced artifacts and the metadata closure branch for reviewer inspection.
- Reuse of `ridge`, `cca`, `fcnn`, and `adt` has already been completed; no rerun is needed for any model in this round.
- Do not upload prediction dumps, checkpoints, `.pt/.pth/.npy/.npz/.h5/.mat`, or large per-job directories.
- The reviewer owns the branch register; do not edit it locally from the execution side.
- Keep the review/execution trace version-locked to the existing local skill lock and GitHub rule commit.
