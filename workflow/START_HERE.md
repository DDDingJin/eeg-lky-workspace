# Start Here

Last updated: 2026-07-06 03:25:00 +08:00

This is the single entry file for resuming work in this worktree.

## Current Active Task

- active round: `etard-eegnet-loso-p00-pilot-publish`
- accepted baseline branch: `fix/ar-20260625-161300-a43831b-loso-resumable-runner-closure-v1`
- accepted baseline commit: `4e365bc`
- accepted result: `experiments/gate0_gate2_loso_eegnet_full_manual_v1_clean`
- accepted protocol: `Weissbart / EEGNet / LOSO / seed0`
- requested new branch: `fix/ar-20260625-161300-a43831b-loso-eegnet-etard-full-v1`
- current branch status:
  - Etard EEGNet LOSO preflight changes are committed locally and pushed for review
  - current Etard prep commit: `9c8720f Prepare Etard EEGNet LOSO preflight`
  - this round must not start long training from Codex
  - a manual bounded run on `P00` has completed
  - do not continue `P01/P02` yet

## Current Worktree

- worktree: `E:\decode\_fix_loso_resumable_runner_closure_v1`
- branch: `fix/ar-20260625-161300-a43831b-loso-eegnet-etard-full-v1`
- accepted baseline branch: `fix/ar-20260625-161300-a43831b-loso-resumable-runner-closure-v1`
- accepted baseline commit: `4e365bc`
- current prep commit: `9c8720f`
- current local pilot result state: `P00 completed`

## Current State

- Goal dataset: `etard_tf64`
- Goal model: `eegnet`
- Goal seed: `0`
- Goal protocol: `pure LOSO`
- Goal output directory: `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean`
- Scientific protocol must remain identical to accepted Weissbart EEGNet LOSO baseline:
  - `sampling_rate=64`
  - `window_size=50`
  - `target_index=last`
  - input tensor contract `[batch, 64, 50]`
  - raw output contract `[batch]`
  - scorer and aggregation unchanged
  - split and leakage unchanged
  - checkpoint selection unchanged
  - `max_epochs` and patience unchanged
  - `batch_size` and `eval_batch_size` unchanged
  - `num_workers=0`
  - `persistent_workers=false`
  - raw recording preload only, no full window-matrix materialization
  - completed job key remains `dataset + subject + model + seed`

## Etard Subject List

- The Etard subject list was confirmed from `E:\decode\data\processed\reference_splits\etard_tf64\export_summary.json`
- Confirmed subjects:
  - `P00 P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19`
- Total subjects: `20`

## What Just Happened

- Added Etard config:
  - `configs/benchmark/gate0_gate2_loso_eegnet_etard_manual_v1.json`
- Added minimal runner preflight support:
  - `--job-plan-only`
- Cleaned partial chunk schema logging so partial chunks do not misleadingly imply global schema failure.
- Fixed first-run `--resume` behavior for a brand-new empty Etard output directory so `startup-only` and `dry-run-plan` do not incorrectly enter corruption recovery.
- Updated:
  - `workflow/RUN_COMMANDS.md`
- A human-run bounded pilot on `etard_tf64 / P00 / eegnet / seed0` completed successfully.
- Pilot result summary:
  - `metric=0.039849916028856534`
  - `best_epoch=22`
  - `epochs_completed=33`
  - `total_job_seconds=12610.2312274`
  - `recording_rows=24`
  - `completed_jobs_count=1`
  - `chunk_schema_validation_passed=true`
  - `global_schema_validation_passed=true`

## Preflight Status

- `py_compile`: passed
- `dry-run-plan`: passed
- `startup-only`: passed
- `job-plan-only`: passed
- Pilot compact artifacts now present under:
  - `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean`
- Current pilot artifact status:
  - `subject_metrics.csv`: `1` row for `P00`
  - `recording_metrics.csv`: `24` rows for `P00`
  - `completed_jobs.json`: `1` success job
  - `failure_report.json`: empty
  - `schema_validation_report.json`: passed `true`

## First Job Plan Snapshot

- heldout subject: `P00`
- train_subjects / val_subjects: `P01-P19`
- test_subject: `P00`
- train_recording_count: `648`
- val_recording_count: `648`
- test_recording_count: `24`
- train_window_count: `5527937`
- val_window_count: `662920`
- train_batches_per_epoch: `21594`
- val_batches_per_epoch: `2590`
- batch_size / eval_batch_size: `256 / 1024`
- input tensor contract: `[batch, 64, 50] -> [batch]`
- target/scorer alignment:
  - `target_index=last`
  - train target is scalar per window
  - eval scorer compares aligned per-window prediction and target arrays
  - aggregation remains recording Pearson then subject mean Pearson

## Current Next Step

- Publish only the compact artifacts for `etard_tf64 / P00 / eegnet / seed0`.
- Do not continue `P01/P02`.
- Do not enter full Etard LOSO yet.

## Read In This Order

1. `workflow/START_HERE.md`
2. `workflow/RUN_COMMANDS.md`
3. `configs/benchmark/gate0_gate2_loso_eegnet_etard_manual_v1.json`
4. `scripts/run_gate0_gate2_loso_resumable_runner_closure_v1.py`
5. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/run_state.json`
6. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/heartbeat.json`
7. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/completed_jobs.json`
8. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/failure_report.json`
9. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/subject_metrics.csv`
10. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/model_run_entries.json`
11. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/schema_validation_report.json`
12. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/leakage_summary.md`
13. `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean/logs/`

## Key Files

- start-here anchor:
  - `workflow/START_HERE.md`
- command pack:
  - `workflow/RUN_COMMANDS.md`
- Etard config:
  - `configs/benchmark/gate0_gate2_loso_eegnet_etard_manual_v1.json`
- runner:
  - `scripts/run_gate0_gate2_loso_resumable_runner_closure_v1.py`
- accepted Weissbart final result:
  - `experiments/gate0_gate2_loso_eegnet_full_manual_v1_clean`
- Etard output directory:
  - `experiments/gate0_gate2_loso_eegnet_etard_manual_v1_clean`

## Commands To Re-Establish Context

Run these from `E:\decode\_fix_loso_resumable_runner_closure_v1`:

```powershell
git status --short --branch
git rev-parse HEAD
Get-Content workflow\START_HERE.md
Get-Content workflow\RUN_COMMANDS.md
Get-Content configs\benchmark\gate0_gate2_loso_eegnet_etard_manual_v1.json
Get-Content experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\run_state.json -Raw
Get-Content experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\heartbeat.json -Raw
Get-Content experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\completed_jobs.json -Raw
Get-Content experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\failure_report.json -Raw
Import-Csv experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\subject_metrics.csv | Select-Object dataset,subject_id,model,seed,metric_value,checkpoint_id
Get-Content experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\model_run_entries.json -Raw
Get-Content experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\schema_validation_report.json -Raw
Get-Content experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\leakage_summary.md
Get-ChildItem experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean\logs | Sort-Object LastWriteTime -Descending | Select-Object Name,LastWriteTime,Length -First 20
```

## Resume Rule

- Do not trust chat memory.
- Reconstruct state from this file, the current git commit, and the Etard output artifacts.
- On crash, reboot, sleep, Codex context loss, or terminal interruption:
  - read `workflow/START_HERE.md` first
  - then inspect the Etard output directory artifacts listed above
  - then decide whether the last subject completed cleanly, partially, or failed

## Expected Next Step

- The immediate next step is review/publish of the `P00` pilot compact artifacts.
- Confirm that `P01/P02` were not started.
- Do not start new models.
- Do not change scorer, split, target alignment, checkpoint rule, or training budget.
- Do not upload raw data, prediction dumps, checkpoints, weights, `.pt/.pth/.npy/.npz/.h5/.mat`, or per-job heavy directories.
