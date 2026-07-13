# Execution And Resume

## Contents

1. Full-run entry
2. Job identity
3. Incremental artifacts
4. Resume and retry
5. Checkpoints
6. Long-run command mode
7. Implementer output
8. Completion states

## Full-Run Entry

Start a full run only when:

1. the scientific protocol is frozen;
2. the code, data, split, and config versions are recorded;
3. the exact path passed smoke or engineering closure;
4. the domain gate is `approved_for_full_run`;
5. the formal audit-loop publication and role state is valid when applicable;
6. output directories are new or verified resumable;
7. old blocked artifacts are isolated.

## Job Identity

Use at least:

```text
dataset_id + subject_or_fold_id + model_id + seed
```

Add protocol, target, or adaptation identifiers when one directory contains multiple scientific definitions. Generate a stable serialized job key and use it in metrics, logs, failures, checkpoints, and completed state.

## Incremental Artifacts

Write:

- `run_manifest.json` before execution;
- `run_state.json` before and during execution;
- `heartbeat.json` at a bounded interval;
- one job log under `logs/`;
- `recording_metrics.csv` after each completed job when required;
- `subject_metrics.csv` after each completed job;
- leakage and checkpoint-selection metadata;
- `failure_report.json` for failed jobs;
- `completed_jobs.json` only after required metrics and metadata are durable.

Use atomic writes for mutable JSON and CSV indexes. Close temporary files before replace. Use unique temporary names. Keep heartbeat frequency low enough to avoid write contention.

Do not mark completion from the existence of a checkpoint alone.

## Resume And Retry

Before resume:

1. load the active resolved config;
2. load completed jobs;
3. validate required metric rows for each completed key;
4. validate model-run and leakage metadata;
5. inspect matching failure entries;
6. classify planned jobs as complete, failed, missing, invalid, or pending;
7. refuse to mix results from another config or code path.

Resume only missing jobs. Retry failed jobs only through an explicit selection. Preserve the original failure record and append the retry reason and outcome. Never silently clear failures.

If a completed key lacks metric rows, mark it invalid and repair the state before running further jobs.

## Checkpoints

After each independent run:

1. save the best checkpoint selected by validation data;
2. save the final checkpoint only when needed for recovery or analysis;
3. store model, optimizer, scheduler, epoch, scaler, and RNG state required by the resume contract;
4. store resolved config, seed, split, data, model, and code identity;
5. store selection metric and epoch;
6. compute a checksum;
7. load-test the checkpoint;
8. prevent cross-job overwrite;
9. keep large weights outside normal Git history;
10. record the local or external storage path in a checkpoint manifest.

## Long-Run Command Mode

Do not start a command expected to exceed 30 minutes in a foreground Codex tool session. Provide:

1. working directory;
2. exact command;
3. environment activation;
4. resolved config path;
5. expected output directory;
6. progress and heartbeat inspection commands;
7. safe stop behavior after the current job;
8. expected files after every job;
9. resume command;
10. estimated job and chunk duration.

Choose chunk size from measured full-scale job timing. Use a no-new-job threshold for bounded sessions: finish the active job, persist artifacts, then stop before launching another job when the threshold is reached.

## Implementer Output

Upload only necessary reviewable material:

1. source and configuration changes;
2. resolved config;
3. split manifest;
4. exact command;
5. environment summary;
6. compact key logs;
7. raw recording and subject metrics;
8. fold and seed results;
9. run state and completed jobs;
10. failure report;
11. checkpoint manifest without large weights;
12. schema and leakage reports;
13. changed-file summary;
14. branch, base, and full commit.

Do not require the implementer to calculate final means and variances. Do not upload caches, full prediction dumps, raw restricted data, per-job directories, or large weights for routine review.

## Completion States

For formal endpoint handoff, use the audit-loop states:

- `published_for_review`: committed, pushed, remote commit verified, compact evidence available;
- `local_committed_push_blocked`: committed locally, one push attempt failed, manual user help required;
- `local_only_incomplete`: intended artifacts remain uncommitted or the run is incomplete.

The implementer may report preliminary local observations in the latter states but may not ask for formal acceptance or start the next dependent round.
