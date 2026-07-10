# Start Here

Last updated: 2026-07-10

This file is the authoritative handoff note for the current worktree. Ignore older chat memory if it conflicts with the files and run artifacts described here.

## Current Active Task

- active round: `subject-specific-local-models-weissbart-full-v1`
- current branch: `fix/ar-20260625-161300-a43831b-subject-specific-local-models-weissbart-full-v1`
- current HEAD at handoff write time: `40ed8716a558dad7b1828bc3a1359451b1f736be`
- base branch requested by user: `fix/ar-20260625-161300-a43831b-vlaai-happyquokka-training-budget-p00-v1`
- base commit requested by user: `40ed8716a558dad7b1828bc3a1359451b1f736be`
- current task status: full Weissbart subject-specific local-model expansion completed; post-run closure checks passed locally; compact artifacts are ready for review publication

## Current Worktree

- worktree: `E:\decode\_fix_fixed_split_pooled20_modelset_v1`
- branch: `fix/ar-20260625-161300-a43831b-subject-specific-local-models-weissbart-full-v1`
- remote push has not been done yet for this branch

## Scope Of This Round

- dataset: `weissbart_tf64`
- subjects: `P00-P12`
- seed: `0`
- new local candidate models:
  - `linear`
  - `lasso`
  - `elasticnet`
  - `vlaai`
  - `happyquokka`
- protocol:
  - per-subject training on subject `train`
  - selection on subject `val`
  - final full evaluation on subject `test`
  - no `256-window` eval cap
  - each model keeps its own input contract
- explicitly not in scope:
  - no DECAF integration
  - no all-dataset benchmark
  - no rerun of already accepted 7 reference models in their original result directory

## Accepted Reference Status

The following 7 subject-specific models already have accepted Weissbart `seed=0` full-subject results and are reused as reference only:

- `ridge`
- `cca`
- `fcnn`
- `dnn`
- `cnn`
- `eegnet`
- `adt`

Reference source:

- `experiments/gate0_gate2_model_expansion_v1/subject_metrics.csv`

Verified coverage:

- Weissbart `P00-P12`
- `7 x 13 = 91` subject rows present in the accepted reference artifact

These 7 models do not need rerun in this round.

## New Runner And Config

- runner:
  - `scripts/run_gate0_gate2_subject_specific_local_models_weissbart_full_v1.py`
- config:
  - `configs/benchmark/gate0_gate2_subject_specific_local_models_weissbart_full_v1.json`
- output dir:
  - `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1`

Important config facts:

- planned jobs: `13 subjects x 5 models = 65`
- `linear/lasso/elasticnet` are intentionally marked as `budgeted linear-family baseline`
- linear-family fit budget:
  - `max_fit_samples_per_split = 12000`
- `vlaai`:
  - `max_epochs = 100`
  - `early_stopping_patience = 10`
- `happyquokka`:
  - `max_epochs = 100`
  - `early_stopping_patience = 10`
  - `model_family_contract = 10s_chunk`
  - `g_con = false`

## Protocol Audit Conclusion

Generated files:

- `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/accepted_protocol_audit.csv`
- `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/accepted_protocol_audit.md`

Audit summary:

- `linear`: `partial`
- `lasso`: `partial`
- `elasticnet`: `partial`
- `vlaai`: `true`
- `happyquokka`: `partial`

Interpretation:

- `linear/lasso/elasticnet` share split/scorer/aggregation/full-test-eval with accepted reference, but use fit sample cap on train/val, so they are not fully identical to accepted `ridge`
- `vlaai` is considered protocol-comparable
- `happyquokka` keeps distinct `10s_chunk` contract and coverage behavior, but still uses aligned split/scorer/final test logic

No model was `comparable=false`, so runtime execution is allowed.

## Manual Long Run Status

The user manually ran the real long run from PowerShell and it completed successfully. Do not start a duplicate run unless a future task explicitly requests a rerun.

Manual command in use:

```powershell
F:\miniconda\envs\decode-torch\python.exe E:\decode\_fix_fixed_split_pooled20_modelset_v1\scripts\run_gate0_gate2_subject_specific_local_models_weissbart_full_v1.py --config E:\decode\_fix_fixed_split_pooled20_modelset_v1\configs\benchmark\gate0_gate2_subject_specific_local_models_weissbart_full_v1.json --device auto --resume
```

## Runtime Bugfixes Already Applied

Two runtime issues were found and fixed in the Weissbart full runner:

1. `subject_id` summary bug
   - old failure: reused single-subject summary writer assumed `config['dataset']['subject_id']` always existed
   - fix: aggregate summary paths now use a synthetic summary config with `subject_id = all_subjects`

2. mixed-model training-summary/schema bug
   - old failure: summary/schema assumed every success model had epoch metadata
   - this is false for `linear/lasso/elasticnet`
   - fix: runner now uses mixed-model summary/schema logic
   - deep-model training curve checks apply only to `vlaai` and `happyquokka`

These fixes live only in:

- `scripts/run_gate0_gate2_subject_specific_local_models_weissbart_full_v1.py`

Do not revert them.

## Final Runtime Snapshot

Observed artifact state after completion:

- `completed_jobs.json` contains all `65` planned jobs
- `failure_report.json` is empty
- `schema_validation_report.json` has `passed = true`
- `run_state.json` has `completed_job_count = 65`, `pending_job_count = 0`, and `last_completed_job_key = weissbart_tf64:P12:happyquokka:seed0`
- `schema_validation_report.json` reports `subject_metric_rows = 65`, `recording_metric_rows = 975`, `training_curve_rows = 652`, and `failure_count = 0`
- aggregate `dataset_metrics.csv` reports `mean_subject_metric = 0.1016683617285833`

Per-model subject metric means:

- `elasticnet`: `0.112541` over `13` subjects
- `happyquokka`: `0.107702` over `13` subjects
- `lasso`: `0.112112` over `13` subjects
- `linear`: `0.091647` over `13` subjects
- `vlaai`: `0.084339` over `13` subjects

Reference of the final log source:

- `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/logs/run.log`

## Non-Fatal Warning Seen During Runtime

The user reported `sklearn` `ConvergenceWarning` for `lasso/elasticnet`, e.g.:

- `Objective did not converge ... increase number of iterations ... consider increasing regularisation`

Interpretation:

- not a crash
- expected sometimes for lag-matrix linear-family models with coordinate descent
- current run should continue
- if final results look suspicious, follow-up cleanup can raise `max_iter` or simplify the hyperparameter grid

Do not treat this warning alone as a failure.

## Preflight Results Already Passed

- `py_compile` for the new runner: passed
- `startup-only`: passed
- `dry-run-plan`: passed
- `shape-check-only`: passed
- planned jobs confirmed: `65`

## Files Expected From This Round

Main output directory:

- `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1`

Expected compact artifacts:

- `run_manifest.json`
- `completed_jobs.json`
- `run_state.json`
- `subject_metrics.csv`
- `recording_metrics.csv`
- `dataset_metrics.csv`
- `model_run_entries.json`
- `full_eval_matrix.csv`
- `training_curve.csv`
- `training_summary.md`
- `coverage_summary.csv`
- `accepted_protocol_audit.csv`
- `accepted_protocol_audit.md`
- `schema_validation_report.json`
- `failure_report.json`
- `result_summary.md`
- `logs/*.log`

## Git State At Handoff

Current untracked work relevant to this round:

- `configs/benchmark/gate0_gate2_subject_specific_local_models_weissbart_full_v1.json`
- `scripts/run_gate0_gate2_subject_specific_local_models_weissbart_full_v1.py`
- `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/`

Also present locally and should remain untracked:

- `local_checkpoints/`
- `__pycache__/`

Do not stage or commit:

- `local_checkpoints/`
- `.pt/.pth/.npy/.npz/.h5/.mat`
- raw data
- prediction dumps
- large per-job directories
- `__pycache__`

## What The Next Codex Should Do

If the user comes back after updating Codex:

1. Read this file first.
2. Read:
   - `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/run_state.json`
   - `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/completed_jobs.json`
   - `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/schema_validation_report.json`
   - `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/failure_report.json`
   - `experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1/logs/run.log`
3. First determine whether the user-manual run is still in progress or already finished.
4. If still running:
   - do not start another copy
   - only monitor or answer user questions
5. If already finished:
   - do not rerun
   - verify `completed_jobs.json` still contains all `65` Weissbart jobs
   - verify `failure_report.json` remains empty
   - verify `schema_validation_report.json` remains `passed = true`
   - check large-file / no-checkpoint-commit hygiene before any new commit
   - publish only compact artifacts and necessary code/config/docs for review

## Closure Target Status

The run has:

- `65` completed Weissbart jobs
- `0` failures
- full-eval metrics for:
  - `linear`
  - `lasso`
  - `elasticnet`
  - `vlaai`
  - `happyquokka`
- accepted-reference audit retained in the same output dir

## Commands To Re-Establish Context

Run from `E:\decode\_fix_fixed_split_pooled20_modelset_v1`:

```powershell
git status --short --branch
git rev-parse HEAD
Get-Content workflow\START_HERE.md
Get-Content experiments\gate0_gate2_subject_specific_local_models_weissbart_full_v1\run_state.json
Get-Content experiments\gate0_gate2_subject_specific_local_models_weissbart_full_v1\completed_jobs.json
Get-Content experiments\gate0_gate2_subject_specific_local_models_weissbart_full_v1\schema_validation_report.json
Get-Content experiments\gate0_gate2_subject_specific_local_models_weissbart_full_v1\failure_report.json
Get-Content experiments\gate0_gate2_subject_specific_local_models_weissbart_full_v1\logs\run.log -Tail 60
```

## Resume Rule

- Do not trust chat memory over file state.
- Do not restart the Weissbart full run if the user-manual process is still alive.
- Do not delete or overwrite the current output directory.
- Use `--resume` only if the user explicitly wants to continue after interruption.
- Preserve the already completed jobs.
