# Review And Paper Evidence

## Contents

1. Intake
2. Code review
3. Result validation
4. Aggregation and statistics
5. Claim support
6. Review decisions
7. Outputs

## Intake

Require enough evidence to identify:

1. branch, commit, base, and publication state;
2. resolved config;
3. data and split versions;
4. exact command;
5. environment;
6. planned, completed, failed, skipped, reused, and pending jobs;
7. raw recording and subject metrics;
8. checkpoint and failure manifests;
9. schema and leakage reports;
10. whether the run is smoke, closure, pilot, or article-grade.

Request only missing material that can change the review. Prefer structured evidence to screenshots or long narrative reports.

## Code Review

Inspect the implementation for:

- epochs or budgets forced to smoke values;
- train/eval and no-grad behavior;
- optimizer, scheduler, gradient, and reset order;
- validation-only selection and early stopping;
- test isolation;
- split-before-windowing;
- training-only fitted preprocessing;
- subject and dataset isolation;
- input, target, prediction, and scorer shapes;
- target alignment and overlap aggregation;
- loss reduction and metric reset;
- batch, recording, subject, and fold aggregation;
- final incomplete batch;
- checkpoint selection, load, and resume state;
- DataLoader seeds and settings;
- device, dtype, and nondeterminism;
- swallowed exceptions or silently removed failures.

Do not accept a report that says code was checked without identifying the inspected files and evidence.

## Result Validation

1. Verify completed keys against metric rows.
2. Check empty files, header-only CSVs, duplicates, and missing rows.
3. Check metric values for NaN, Inf, impossible ranges, and suspicious constants.
4. Check reused rows against the active matrix and provenance.
5. Check failed jobs and retry history.
6. Check smoke rows are excluded from final aggregation.
7. Check old or blocked directories are excluded.
8. Check result counts against datasets, subjects or folds, models, and seeds.
9. Check that prediction and target observation counts agree.
10. Check that checkpoints and metrics share the same version identity.

## Aggregation And Statistics

Let the reviewer recompute aggregates from raw results.

1. Preserve raw rows.
2. Aggregate recordings to the declared subject or fold unit.
3. Distinguish macro, micro, and weighted aggregation.
4. Compute mean and standard deviation where appropriate.
5. Compute confidence intervals or other uncertainty estimates when planned.
6. Use paired subject-level tests for paired scientific comparisons.
7. Report effect size.
8. Apply the planned multiple-comparison correction.
9. Keep seed variation and subject variation distinguishable.
10. Apply the predeclared missing-run policy.
11. Preserve a mapping from raw rows to every table value.
12. Generate tables and figures through scripts.

## Claim Support

For every article claim:

1. locate its claim ID;
2. locate the exact experiment config;
3. locate raw and aggregate evidence;
4. verify protocol comparability;
5. verify statistical support;
6. separate practical effect from significance;
7. check computational-cost qualifications;
8. record contradictory or negative evidence;
9. record limits and unavailable comparisons;
10. classify the claim as supported, partially supported, unsupported, or exploratory.

Do not revise a predeclared primary metric after viewing test results. Create a clearly labeled exploratory analysis instead.

## Review Decisions

Use one decision:

- pass;
- conditional pass;
- fail;
- blocked;
- deferred to backlog.

Bind every decision to evidence. Give each required correction a priority, exact scope, acceptance check, and stop condition. Under a formal audit loop, use the reviewer-owned issue and verification statuses instead of inventing a parallel state machine.

## Outputs

Produce:

1. validated raw-result inventory;
2. aggregate tables;
3. statistical results;
4. figures;
5. raw-to-summary mapping;
6. claim-to-evidence status;
7. limitations and negative results;
8. review decision;
9. next safe action.
