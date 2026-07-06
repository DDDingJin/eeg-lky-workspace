# LOSO Checkpoint Saving Closure

This change adds local checkpoint persistence for the LOSO runner without changing the scientific protocol.

## Scope

- save the per-job best checkpoint selected by the existing validation rule
- write checkpoint files only to local gitignored storage
- write compact provenance metadata to result artifacts

## Non-Goals

- no change to train/val/test membership
- no change to scorer or aggregation
- no change to target alignment
- no change to early stopping
- no change to checkpoint selection rule
- no change to batch size
- no change to `num_workers` or `persistent_workers`
- no long training started by this closure
- no fine-tuning started by this closure

## Local Checkpoint Layout

Checkpoint files are written under:

```text
local_checkpoints/loso/<model>/<dataset>/<subject_id>/seed<seed>/best_epoch_<best_epoch>.pt
```

Example:

```text
local_checkpoints/loso/eegnet/weissbart_tf64/P01/seed0/best_epoch_24.pt
```

These files are local only and must not be committed to GitHub.

## Checkpoint Contents

Each checkpoint contains at least:

- `model_state_dict`
- `dataset`
- `subject_id`
- `model`
- `seed`
- `best_epoch`
- `best_val_score`
- `protocol`
- `config_path`
- `branch`
- `commit_sha` when available
- `created_at`

## Compact Provenance

Each real result output directory writes `checkpoint_manifest.json` with per-job entries including:

- `dataset`
- `subject_id`
- `model`
- `seed`
- `source_job_key`
- `checkpoint_local_id`
- `checkpoint_relative_path`
- `checkpoint_artifact_status=local_only_not_committed`
- `best_epoch`
- `best_val_score`
- `protocol`
- `config_path`
- `branch`
- `commit_sha` when available

`model_run_entries.json` and in-memory `run_state.json` also record `checkpoint_local_id`.

## Engineering Smoke Artifacts

Mock save/load validation must not write into a real scientific result directory.

Engineering-only smoke artifacts are written under:

```text
experiments/gate0_gate2_loso_checkpoint_saving_closure_smoke/
```

These smoke artifacts are:

- mock only
- not a scientific result
- not a real LOSO checkpoint record
- expected to use `checkpoint_artifact_status=local_only_not_committed`

## Important Limitation

Previously completed LOSO subjects that were run before this closure do not automatically gain checkpoint files.

They cannot be used directly for future fine-tuning unless the target LOSO job is rerun after this closure, so that a real local checkpoint is produced.
