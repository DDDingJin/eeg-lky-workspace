# Start Here

Last updated: 2026-07-01

This is the single entry file for resuming work in this worktree.

## Current Active Task

- active round: `loso-ridge-full-v1`
- requested base branch: `fix/ar-20260625-161300-a43831b-loso-pilot-v1`
- requested base commit: `fb3494d6edb3b54d793288e739d6c37c4fb34e5f`
- published result branch: `fix/ar-20260625-161300-a43831b-loso-pilot-v1`
- published result commit: `fb3494d6edb3b54d793288e739d6c37c4fb34e5f`
- requested new branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- current status:
  - published branch `fix/ar-20260625-161300-a43831b-loso-pilot-v1` exists on `origin` at `fb3494d6edb3b54d793288e739d6c37c4fb34e5f`
  - the LOSO pilot was accepted as subject-independent interface validation
  - this round expands only the validated LOSO ridge path to all subjects on Weissbart and Etard
  - no deep models, no multi-seed, and no paper prose updates are allowed in this round
  - target output directory is `experiments/gate0_gate2_loso_ridge_full_v1`
  - full LOSO ridge run is complete locally with compact artifacts generated
  - next step is commit + push for reviewer inspection

## Current Worktree

- worktree: `E:\decode\_fix_loso_pilot_v1_clean`
- branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- published baseline branch: `fix/ar-20260625-161300-a43831b-loso-pilot-v1`
- published baseline commit: `fb3494d6edb3b54d793288e739d6c37c4fb34e5f`
- remote branch target: `origin/fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`

## Current State

- The accepted baseline is the published LOSO pilot branch.
- The current round is the first full all-subject LOSO ridge expansion under the validated subject-independent interface.
- Scope:
  - `weissbart_tf64`: all subjects
  - `etard_tf64`: all subjects
  - `ridge` only
  - seed `0` only
- The current round must publish only compact artifacts.
- Current local LOSO full output status:
  - planned jobs `33`
  - successful jobs `33`
  - failed jobs `0`
  - schema validation `passed`

## What Just Happened

- This round starts from the accepted LOSO pilot baseline.
- The main new task is to scale the already-validated pure LOSO ridge procedure across all Weissbart and Etard subjects.
- The current deliverable is a reviewable compact LOSO ridge full package with dataset-level, Etard condition-level, and subject-specific comparison outputs.
- The current local results are ready for review once committed and pushed.

## Read In This Order

1. `workflow/START_HERE.md`
2. `workflow/skill_alignment.md`
3. `configs/benchmark/gate0_gate2_loso_ridge_full_v1.json`
4. `scripts/run_gate0_gate2_loso_ridge_full_v1.py`
5. `experiments/gate0_gate2_loso_ridge_full_v1/run_manifest.json`
6. `README.md`

## Key Files

- config:
  - `configs/benchmark/gate0_gate2_loso_ridge_full_v1.json`
- runner:
  - `scripts/run_gate0_gate2_loso_ridge_full_v1.py`
- machine-readable run state:
  - `experiments/gate0_gate2_loso_ridge_full_v1/run_manifest.json`
- compact outputs:
  - `experiments/gate0_gate2_loso_ridge_full_v1/recording_metrics.csv`
  - `experiments/gate0_gate2_loso_ridge_full_v1/subject_metrics.csv`
  - `experiments/gate0_gate2_loso_ridge_full_v1/dataset_metrics.csv`
  - `experiments/gate0_gate2_loso_ridge_full_v1/etard_condition_metrics.csv`
  - `experiments/gate0_gate2_loso_ridge_full_v1/leakage_summary.md`
  - `experiments/gate0_gate2_loso_ridge_full_v1/loso_vs_subject_specific_comparison.csv`
  - `experiments/gate0_gate2_loso_ridge_full_v1/loso_vs_subject_specific_comparison.md`
  - `experiments/gate0_gate2_loso_ridge_full_v1/result_summary.md`
  - `experiments/gate0_gate2_loso_ridge_full_v1/schema_validation_report.json`
  - `experiments/gate0_gate2_loso_ridge_full_v1/run_manifest.json`
- review-loop lock:
  - `workflow/skill_lock.json`
- local skill alignment note:
  - `workflow/skill_alignment.md`

## Commands To Re-Establish Context

Run these from `E:\decode\_fix_loso_pilot_v1_clean`:

```powershell
git status --short --branch
git rev-parse HEAD
Get-Content workflow\START_HERE.md
git show fb3494d6edb3b54d793288e739d6c37c4fb34e5f --stat --no-patch
Get-Content configs\benchmark\gate0_gate2_loso_ridge_full_v1.json
Get-Content experiments\gate0_gate2_loso_ridge_full_v1\run_manifest.json
Get-Content experiments\gate0_gate2_loso_ridge_full_v1\leakage_summary.md
Get-Content experiments\gate0_gate2_loso_ridge_full_v1\result_summary.md
Get-Content experiments\gate0_gate2_loso_ridge_full_v1\loso_vs_subject_specific_comparison.md
```

## Benchmark Rebuild Command

If the LOSO ridge full package needs to be rebuilt, use:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts/run_gate0_gate2_loso_ridge_full_v1.py --config configs/benchmark/gate0_gate2_loso_ridge_full_v1.json
```

## Resume Rule

- Do not assume chat memory is authoritative.
- Reconstruct state from this file, the current git commit, and the benchmark manifests.
- Before any new rerun or reviewer response, verify that the published baseline branch and commit in this file still match the intended review target.

## Expected Next Step

- Current next step is to commit and push the completed full all-subject LOSO ridge package for reviewer inspection.
- Do not upload prediction dumps, checkpoints, `.pt/.pth/.npy/.npz/.h5/.mat`, or per-job directories.
- The reviewer owns the branch register; do not edit it locally from the execution side.
- Keep the review/execution trace version-locked to the current shared rule requirements.
