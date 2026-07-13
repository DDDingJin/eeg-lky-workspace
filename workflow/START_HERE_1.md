# Start Here 1

Last updated: 2026-07-13

This is the secondary handoff note that remains relevant to the current worktree because the next cross-subject neural zero-shot phase is expected to reuse its deterministic conclusion.

## Secondary Task / Reference

- branch: `fix/ar-20260625-161300-a43831b-vlaai-deterministic-reproducibility-p06-v1`
- accepted closure commit: `ae53d62`
- scope: `weissbart_tf64 / P06 / vlaai / seed0`
- status: completed and reproducible under strict deterministic settings
- usage status: reference only; not an active execution task unless deterministic rollout is explicitly reopened

## Why This Still Matters

The current main branch does not yet integrate strict deterministic execution into the new within-dataset fixed-holdout neural zero-shot runner.

Before formal neural zero-shot execution starts for:

- `fcnn`
- `cnn`
- `eegnet`
- `adt`
- `vlaai`
- `happyquokka`

the deterministic-policy rollout should be patched separately, and this closure is the engineering reference for how to do it.

## Accepted Deterministic Settings

The reproducible VLAAI path required:

- `random.seed(seed)`
- `numpy.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `DataLoader` with fixed `torch.Generator(seed)`
- `num_workers=0`
- `torch.backends.cudnn.benchmark=False`
- `torch.backends.cudnn.deterministic=True`
- `torch.use_deterministic_algorithms(True, warn_only=False)`
- process environment set before Python launch:
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`

## Closure Result

Two repeats were identical on:

- `initial_state_hash`
- `first_train_batch_hash`
- `after_first_optimizer_step_hash`
- `best_state_hash`
- `prediction_hash`
- `best_epoch`
- `epochs_completed`
- `best_val_score`
- `test_metric`

Final matching values:

- `best_epoch = 4`
- `epochs_completed = 14`
- `best_val_score = 0.09118621892606218`
- `test_metric = 0.028696070905947592`

Artifacts live in:

- `experiments/gate0_gate2_vlaai_deterministic_reproducibility_p06_v1`

Key files:

- `reproducibility_runs.csv`
- `environment_and_seed_audit.json`
- `single_batch_smoke_report.json`
- `post_run_closure.md`

## What The Next Codex Should Do With This

If the user asks to make neural zero-shot execution deterministic:

1. Read this file.
2. Read `workflow/DETERMINISM_POLICY_PENDING.md`.
3. Read:
   - `experiments/gate0_gate2_vlaai_deterministic_reproducibility_p06_v1/reproducibility_runs.csv`
   - `experiments/gate0_gate2_vlaai_deterministic_reproducibility_p06_v1/environment_and_seed_audit.json`
   - `experiments/gate0_gate2_vlaai_deterministic_reproducibility_p06_v1/post_run_closure.md`
4. Apply the same strict deterministic policy to the future neural zero-shot runner.
5. Do not claim deterministic reproducibility for the new within-dataset cross-subject runner until that patch is actually implemented and revalidated.

## What Not To Do

- Do not rerun this VLAAI P06 reproducibility task unless explicitly requested.
- Do not mix these artifacts into the within-dataset fixed-holdout modelset directories.
- Do not assume linear-family validation issues are solved by this deterministic result. They are separate issues.

## Why This File Exists Next To START_HERE.md

These are the current two workflow notes:

- `START_HERE.md` = active paused main task for within-dataset fixed-holdout modelset preflight
- `START_HERE_1.md` = deterministic reference task that the future neural zero-shot patch may need

If a future Codex session only reads the workflow notes, it should understand that the main task is paused and this file is supporting context rather than an immediate execution target.
