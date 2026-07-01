# Start Here

Last updated: 2026-07-01

This is the single entry file for resuming work in this worktree.

## Current Active Task

- active round: `loso-pilot-v1`
- requested base branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure`
- requested base commit: `d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`
- published result branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure`
- published result commit: `d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`
- requested new branch: `fix/ar-20260625-161300-a43831b-loso-pilot-v1`
- current status:
  - published branch `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure` exists on `origin` at `d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`
  - `model-expansion-v1` remains the accepted result baseline; this round only adds a minimal LOSO interface pilot
  - scope is restricted to `weissbart_tf64`, `ridge`, held-out `P00`, `P01`, `P02`
  - no new models and no paper prose updates are allowed in this round
  - target output directory is `experiments/gate0_gate2_loso_pilot_v1`
  - LOSO pilot run is complete locally with compact artifacts generated
  - next step is commit + push for reviewer inspection

## Current Worktree

- worktree: `E:\decode\_fix_loso_pilot_v1_clean`
- branch: `fix/ar-20260625-161300-a43831b-loso-pilot-v1`
- published baseline branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure`
- published baseline commit: `d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`
- remote branch target: `origin/fix/ar-20260625-161300-a43831b-loso-pilot-v1`

## Current State

- The accepted baseline is the published model-expansion metadata-closure branch.
- The current round is the first subject-independent LOSO pilot under the existing scorer/schema style.
- The pilot is intentionally minimal:
  - dataset `weissbart_tf64`
  - model `ridge`
  - held-out subjects `P00`, `P01`, `P02`
- The current round must publish only compact artifacts.
- Current local LOSO output status:
  - planned jobs `3`
  - successful jobs `3`
  - failed jobs `0`
  - schema validation `passed`

## What Just Happened

- This round starts from the already-published metadata-closure baseline.
- The main new task is to prove that a pure LOSO subject-independent interface can run without target-subject leakage.
- The current deliverable is an execution-ready and reviewable compact LOSO pilot package, not a large experiment campaign.
- The current local results show the LOSO interface is working end-to-end for the three requested held-out subjects.

## Read In This Order

1. `workflow/START_HERE.md`
2. `workflow/skill_alignment.md`
3. `configs/benchmark/gate0_gate2_loso_pilot_v1.json`
4. `scripts/run_gate0_gate2_loso_pilot_v1.py`
5. `experiments/gate0_gate2_loso_pilot_v1/run_manifest.json`
6. `README.md`

## Key Files

- config:
  - `configs/benchmark/gate0_gate2_loso_pilot_v1.json`
- runner:
  - `scripts/run_gate0_gate2_loso_pilot_v1.py`
- machine-readable run state:
  - `experiments/gate0_gate2_loso_pilot_v1/run_manifest.json`
- compact outputs:
  - `experiments/gate0_gate2_loso_pilot_v1/recording_metrics.csv`
  - `experiments/gate0_gate2_loso_pilot_v1/subject_metrics.csv`
  - `experiments/gate0_gate2_loso_pilot_v1/leakage_check.md`
  - `experiments/gate0_gate2_loso_pilot_v1/result_summary.md`
  - `experiments/gate0_gate2_loso_pilot_v1/schema_validation_report.json`
  - `experiments/gate0_gate2_loso_pilot_v1/run_manifest.json`
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
git show d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae --stat --no-patch
Get-Content configs\benchmark\gate0_gate2_loso_pilot_v1.json
Get-Content experiments\gate0_gate2_loso_pilot_v1\run_manifest.json
Get-Content experiments\gate0_gate2_loso_pilot_v1\leakage_check.md
Get-Content experiments\gate0_gate2_loso_pilot_v1\result_summary.md
```

## Benchmark Rebuild Command

If the LOSO pilot package needs to be rebuilt, use:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts/run_gate0_gate2_loso_pilot_v1.py --config configs/benchmark/gate0_gate2_loso_pilot_v1.json
```

## Resume Rule

- Do not assume chat memory is authoritative.
- Reconstruct state from this file, the current git commit, and the benchmark manifests.
- Before any new rerun or reviewer response, verify that the published baseline branch and commit in this file still match the intended review target.

## Expected Next Step

- Current next step is to commit and push the completed minimal LOSO pilot for reviewer inspection.
- Do not upload prediction dumps, checkpoints, `.pt/.pth/.npy/.npz/.h5/.mat`, or per-job directories.
- The reviewer owns the branch register; do not edit it locally from the execution side.
- Keep the review/execution trace version-locked to the current shared rule requirements.
