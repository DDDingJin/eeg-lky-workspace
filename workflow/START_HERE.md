# Start Here

Last updated: 2026-07-02

This is the single entry file for resuming work in this worktree.

## Current Active Task

- active round: `loso-nonridge-data-interface-smoke-v1`
- requested base branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- requested base commit: `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- current branch: `fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1`
- current status:
  - this round is a local adapter smoke matrix only
  - do not enter full LOSO run from this state
  - protocol is pure LOSO with heldout subject excluded from train, val, scaler or normalization fitting, hyperparameter selection, and checkpoint selection
  - dataset is `weissbart_tf64`
  - heldout subject is `P00`
  - seed is `0`
  - target output directory is `experiments/gate0_gate2_loso_nonridge_data_interface_smoke_v1`
  - required success gate is `eegnet + fcnn + cnn`
  - `cca` is allowed to land as a documented scalability-specific skip

## Current Worktree

- worktree: `E:\decode\_fix_loso_pilot_v1_clean`
- branch: `fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1`
- base branch for reporting: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- base commit for reporting: `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- remote branch target if publish succeeds: `origin/fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1`

## Current State

- A prior local-only single-model smoke for `eegnet` already succeeded on `weissbart_tf64 / P00 / seed 0`.
- That prior smoke verified:
  - heldout `P00` excluded from train and val
  - lazy recording-level window generation
  - compact artifacts only
  - passed schema validation
- The current runner/config were expanded from that single-model smoke into an adapter smoke matrix for:
  - `eegnet`
  - `fcnn`
  - `cnn`
  - `adt`
  - `dnn`
  - `cca`
- The matrix runner had been rewritten locally but not yet published at the time this note was updated.

## What Just Happened

- The earlier all-model LOSO branch was intentionally paused because only Ridge was reusable there and non-Ridge models were not on a scalable data path.
- This round isolates the scalable lazy data interface problem.
- Existing `eegnet` smoke evidence was preserved locally before matrix execution so the first successful smoke is not lost even if final matrix artifacts overwrite the same output directory.

## Read In This Order

1. `workflow/START_HERE.md`
2. `workflow/skill_alignment.md`
3. `configs/benchmark/gate0_gate2_loso_nonridge_data_interface_smoke_v1.json`
4. `scripts/run_gate0_gate2_loso_nonridge_data_interface_smoke_v1.py`
5. `experiments/gate0_gate2_loso_nonridge_data_interface_smoke_v1/run_manifest.json`
6. `experiments/gate0_gate2_loso_nonridge_data_interface_smoke_v1/schema_validation_report.json`

## Key Files

- config:
  - `configs/benchmark/gate0_gate2_loso_nonridge_data_interface_smoke_v1.json`
- runner:
  - `scripts/run_gate0_gate2_loso_nonridge_data_interface_smoke_v1.py`
- output directory:
  - `experiments/gate0_gate2_loso_nonridge_data_interface_smoke_v1/`
- prior standalone smoke evidence:
  - `experiments/gate0_gate2_loso_nonridge_data_interface_smoke_v1/subject_metrics.csv`
  - `experiments/gate0_gate2_loso_nonridge_data_interface_smoke_v1/recording_metrics.csv`
  - `experiments/gate0_gate2_loso_nonridge_data_interface_smoke_v1/schema_validation_report.json`

## Commands To Re-Establish Context

Run these from `E:\decode\_fix_loso_pilot_v1_clean`:

```powershell
git status --short --branch
git rev-parse HEAD
Get-Content workflow\START_HERE.md
Get-Content configs\benchmark\gate0_gate2_loso_nonridge_data_interface_smoke_v1.json
Get-Content scripts\run_gate0_gate2_loso_nonridge_data_interface_smoke_v1.py
Get-ChildItem experiments\gate0_gate2_loso_nonridge_data_interface_smoke_v1
```

## Smoke Matrix Command

If the adapter smoke matrix needs to be built or resumed, use:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts/run_gate0_gate2_loso_nonridge_data_interface_smoke_v1.py --config configs/benchmark/gate0_gate2_loso_nonridge_data_interface_smoke_v1.json --device auto
```

## Resume Rule

- Do not assume chat memory is authoritative.
- Reconstruct state from this file, the current git commit, and the output manifest.
- Before any push, confirm the branch is still a smoke-only branch and not a full LOSO benchmark branch.
- Do not upload raw data, prediction dumps, checkpoints, model weights, per-job directories, smoke caches, or `.pt/.pth/.npy/.npz/.h5/.mat`.

## Expected Next Step

- Run the local adapter smoke matrix.
- Confirm:
  - `eegnet`, `fcnn`, and `cnn` succeed
  - leakage checks all pass
  - prediction-target alignment is explicitly recorded
  - `cca` is either documented as a scalability-specific issue or succeeds without violating the compact-artifact rule
- Only after that should this branch be committed and a push attempted.
