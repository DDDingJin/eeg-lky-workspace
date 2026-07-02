# Start Here

Last updated: 2026-07-01

This is the single entry file for resuming work in this worktree.

## Current Active Task

- active round: `loso-all-models-single-seed-v1`
- requested base branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- requested base commit: `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- published baseline branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- published baseline commit: `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- requested new branch: `fix/ar-20260625-161300-a43831b-loso-all-models-single-seed-v1`
- current status:
  - this round extends validated pure LOSO from ridge-only to the current unified subject-specific benchmark model set
  - target output directory is `experiments/gate0_gate2_loso_all_models_single_seed_v1`
  - `ridge` must be reused from `loso-ridge-full-v1`, not rerun
  - `cca`, `fcnn`, `dnn`, `cnn`, `eegnet`, and `adt` are tracked in this round with implementation identity and explicit runtime blockage status
  - target subject must be excluded from train, val, scaler/normalization, and model/checkpoint selection
  - only compact artifacts may be committed and pushed

## Current Worktree

- worktree: `E:\decode\_fix_loso_pilot_v1_clean`
- branch: `fix/ar-20260625-161300-a43831b-loso-all-models-single-seed-v1`
- published baseline branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- published baseline commit: `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- remote branch target: `origin/fix/ar-20260625-161300-a43831b-loso-all-models-single-seed-v1`

## Current State

- The accepted baseline is the published LOSO ridge full branch.
- Scope:
  - datasets: `weissbart_tf64`, `etard_tf64`
  - held-out subjects: all available test subjects in each dataset
  - models: `ridge`, `cca`, `fcnn`, `dnn`, `cnn`, `eegnet`, `adt`
  - seed: `0`
- Comparison target:
  - subject-specific reference protocol: `gate0_gate2_full_subject_single_seed_v1`
  - ridge LOSO reference protocol: `gate0_gate2_loso_ridge_full_v1`
- Current local code status:
  - config created
  - runner created
  - compact result package generated locally
  - ridge reused successfully for all subjects
  - non-ridge models recorded as `skipped_with_reason` pending a scalable LOSO data/training path

## What Just Happened

- The LOSO pilot and LOSO ridge full rounds were accepted as the subject-independent baseline path.
- This round aligns LOSO with the existing subject-specific benchmark model set so the two protocols can be compared model-by-model.
- The current deliverable is a compact all-model LOSO package with dataset-level, subject-level, and Etard condition-level comparison artifacts.

## Read In This Order

1. `workflow/START_HERE.md`
2. `workflow/skill_alignment.md`
3. `configs/benchmark/gate0_gate2_loso_all_models_single_seed_v1.json`
4. `scripts/run_gate0_gate2_loso_all_models_single_seed_v1.py`
5. `experiments/gate0_gate2_loso_ridge_full_v1/run_manifest.json`
6. `experiments/gate0_gate2_model_expansion_v1/run_manifest.json`

## Key Files

- config:
  - `configs/benchmark/gate0_gate2_loso_all_models_single_seed_v1.json`
- runner:
  - `scripts/run_gate0_gate2_loso_all_models_single_seed_v1.py`
- reused LOSO ridge reference:
  - `experiments/gate0_gate2_loso_ridge_full_v1/recording_metrics.csv`
  - `experiments/gate0_gate2_loso_ridge_full_v1/subject_metrics.csv`
- subject-specific reference:
  - `experiments/gate0_gate2_model_expansion_v1/subject_metrics.csv`
- target output directory:
  - `experiments/gate0_gate2_loso_all_models_single_seed_v1/`

## Commands To Re-Establish Context

Run these from `E:\decode\_fix_loso_pilot_v1_clean`:

```powershell
git status --short --branch
git rev-parse HEAD
Get-Content workflow\START_HERE.md
git show f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a --stat --no-patch
Get-Content configs\benchmark\gate0_gate2_loso_all_models_single_seed_v1.json
Get-Content scripts\run_gate0_gate2_loso_all_models_single_seed_v1.py
Get-Content experiments\gate0_gate2_loso_ridge_full_v1\run_manifest.json
Get-Content experiments\gate0_gate2_model_expansion_v1\run_manifest.json
```

## Benchmark Rebuild Command

If the LOSO all-model package needs to be built or resumed, use:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts/run_gate0_gate2_loso_all_models_single_seed_v1.py --config configs/benchmark/gate0_gate2_loso_all_models_single_seed_v1.json --resume
```

## Resume Rule

- Do not assume chat memory is authoritative.
- Reconstruct state from this file, the current git commit, and the benchmark manifests.
- Before any new rerun or reviewer response, verify that the published baseline branch and commit in this file still match the intended review target.

## Expected Next Step

- Current next step is to commit and push the compact LOSO all-model single-seed package.
- Do not upload raw data, prediction dumps, checkpoints, model weights, `.pt/.pth/.npy/.npz/.h5/.mat`, or per-job directories.
- The reviewer owns the branch register; do not edit it locally from the execution side.
- Under the current shared rule, this round is reviewable only after commit plus successful push with remote commit verification.
