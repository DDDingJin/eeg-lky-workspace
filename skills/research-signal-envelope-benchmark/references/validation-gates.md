# Validation Gates

## Contents

1. Gate sequence
2. Leakage audit
3. Code and contract checks
4. Smoke test
5. Smoke-to-full invariance
6. Engineering closure
7. Runtime-specific gates
8. Gate outputs

## Gate Sequence

Use these statuses:

- `approved_for_full_run`;
- `approved_as_smoke_only`;
- `engineering_closure_required`;
- `rejected_until_reproduced_cleanly`.

Apply the most restrictive applicable status. A successful command does not imply approval.

## Leakage Audit

Verify:

1. raw recording, trial, or subject membership was assigned before windowing;
2. no source samples or overlapping windows cross splits;
3. normalization was fitted only on training data;
4. learned filters, feature selection, dimensionality reduction, and augmentation policy did not use held-out data;
5. validation data alone selected lag, hyperparameters, thresholds, early stopping, and checkpoint;
6. test data did not influence development;
7. held-out subjects are absent from cross-subject training and fitting;
8. target datasets are absent from zero-shot source training;
9. cached artifacts are tied to the correct split version;
10. exclusions are declared and stable.

Produce a leakage report traceable to `dataset + subject_or_fold + model + seed` or to a coarser shared protocol artifact with an explicit mapping.

## Code And Contract Checks

Inspect code, not only reports. Check:

1. input, target, raw output, postprocessed output, and scorer shapes;
2. target alignment and crop rules;
3. overlap aggregation;
4. finite loss and gradients;
5. optimizer and scheduler order;
6. train/eval mode and no-grad behavior;
7. metric reset and reduction;
8. batch, recording, subject, and fold aggregation;
9. label or target mapping;
10. final incomplete batch;
11. checkpoint selection and load target;
12. restored optimizer, scheduler, epoch, and scaler state when resuming;
13. DataLoader seed and runtime settings;
14. device and dtype;
15. exceptions and skipped failures;
16. formal epochs not forced to one;
17. smoke flags not leaking into formal config.

## Smoke Test

Use:

- one dataset;
- one or two subjects or recordings;
- one representative trained model;
- one reliable or reused baseline when relevant;
- one seed;
- the exact full-run code path and runtime settings;
- reduced scope only.

Exercise:

1. train, validation, and test;
2. logging;
3. checkpoint save and reload;
4. recording and subject metric writes;
5. failure reporting;
6. completed-job state;
7. interruption and resume;
8. schema validation;
9. leakage reporting;
10. time and resource measurement.

Smoke evidence is permission to continue engineering, not article evidence.

## Smoke-To-Full Invariance

Require a new smoke after changing:

1. model adapter or input shape;
2. target alignment;
3. scorer or aggregation;
4. split or leakage rule;
5. epochs, early stopping, batch size, or training budget;
6. DataLoader `num_workers`, `pin_memory`, or `persistent_workers`;
7. window length, hop, lag range, or target index;
8. checkpoint selection;
9. resume semantics;
10. reused-result filtering;
11. result schema.

Full execution may expand only dataset, subject, model, seed, and resulting wall-clock scope.

## Engineering Closure

Require a closure when a runner was edited during or after an attempted full run.

Use:

- one stable dataset;
- one or two held-out subjects;
- one reference model when relevant;
- one newly trained representative model;
- one seed;
- intended full-run training budget and runtime settings.

Require:

1. run manifest;
2. schema validation report;
3. leakage report;
4. failure report;
5. recording and subject metric rows;
6. completed-job state;
7. checkpoint-selection evidence;
8. resume proof;
9. engineering report explaining the change.

Do not count closure metrics as final scientific evidence.

## Runtime-Specific Gates

### DataLoader

Treat a worker-setting change as a new execution path. If `num_workers=0` passed smoke, use it for full until another setting passes its own smoke. Reject multi-worker paths with worker crashes, permission failures, shutdown errors, or missing outputs.

### Recording Cache

When lazy global window sampling repeatedly loads and evicts recordings, log cache hit rate, load requests, unique recordings, repeated loads, batch time, and memory estimates. Prefer raw-recording preload or recording-aware sampling after measurement. Do not materialize a full unfolded window matrix without a safe memory estimate.

### Timeout

Classify a timeout as `still_running`, `process_failed`, `timeout_no_partial_state`, or `stalled`. For `timeout_no_partial_state`, reject the monolithic path and require resumable incremental closure.

### Result Reuse

Filter reused rows to the active dataset, subject or fold, model, and seed set. Record source branch, commit, path, protocol, and filter. Mark the job `reused`, not newly successful.

## Gate Outputs

Report:

1. gate status;
2. accepted evidence;
3. blockers;
4. exact next action;
5. configuration identity;
6. scope tested;
7. rows and artifacts produced;
8. whether the audit-loop publication state permits review.
