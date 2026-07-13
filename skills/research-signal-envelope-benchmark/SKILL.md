---
name: research-signal-envelope-benchmark
description: Build, validate, run, review, and release paper-grade multi-model, multi-dataset benchmarks for neural-signal-to-envelope reconstruction. Use when designing benchmark questions, integrating sourced models or datasets, defining subject-specific, cross-subject, cross-dataset, or cross-modal protocols, auditing leakage and prediction contracts, gating smoke or full runs, resuming interrupted jobs, reviewing statistical evidence, consolidating experiment branches, preparing reproducible releases, or recording recurring errors and learnings. Apply together with research-audit-loop for formal reviewer/implementer Git review rounds.
---

# Research Signal Envelope Benchmark

## Purpose

Build an auditable `models × datasets × protocols × seeds` benchmark that can support a scientific article. Keep scientific validity, execution reliability, review ownership, and public reproducibility distinct but connected.

Do not force every request through the full lifecycle. Route the task to the smallest sufficient set of modules.

## Start Every Task

1. Determine the repository root.
2. Determine the active role: `reviewer`, `implementer`, or `standalone`.
3. Read `workflow/START_HERE.md` and `workflow/state.json` when they exist.
4. Compare recorded branch, commit, config, split, and gate state with the repository.
5. Read active error patterns before repeating a failed path.
6. Classify the request with the routing table below.
7. Load only the required references.
8. State which phase is active and which phases are not required.
9. Continue from the last verified state; do not reconstruct state from chat memory.

If project state does not exist, run `scripts/init_workflow_state.py <project-root>` before long or multi-round work.

## Compose With The Audit Loop

For a formal two-endpoint review, fix, verification, branch, tag, PR, or publication round:

1. Read the sibling `../research-audit-loop/SKILL.md` and its required references.
2. Let `research-audit-loop` own roles, branches, commits, tags, PRs, issue states, review artifacts, and branch-register ownership.
3. Let this Skill own scientific design, data and model contracts, leakage, metrics, runner gates, statistical evidence, and release reproducibility.
4. Record the applicable domain gate in the audit manifest.
5. Require both gates to pass. If either Skill blocks, do not start a full run, verify the round, or promote results.
6. Never duplicate or weaken audit-loop role ownership in project-local instructions.

Read `references/audit-loop-integration.md` whenever both Skills apply.

## Route The Task

| Task | Required references |
|---|---|
| Define article questions, claims, matrix, fairness, or statistics | `study-design.md`, `envelope-reconstruction.md` |
| Add or inspect a model or dataset | `model-data-contracts.md`, `envelope-reconstruction.md` |
| Change split, sliding windows, lag, scorer, target, or protocol | `study-design.md`, `envelope-reconstruction.md`, `validation-gates.md` |
| Run or approve a smoke or engineering closure | `validation-gates.md`, `execution-and-resume.md` |
| Start, monitor, stop, retry, or resume a long run | `execution-and-resume.md`, `validation-gates.md` |
| Review raw metrics, statistics, tables, figures, or claims | `review-and-paper.md`, `study-design.md` |
| Hand work between reviewer and implementer | `collaboration.md`, `audit-loop-integration.md` |
| Consolidate branches or prepare open source release | `release-and-convergence.md`, `audit-loop-integration.md` |
| Diagnose an error or add a learning | `learning-and-error-memory.md` plus the affected phase reference |
| Create or validate structured artifacts | `schemas.md` |

## Global Scientific Invariants

1. Separate model implementations from dataset implementations.
2. Keep model, dataset, protocol, runner, metric, and configuration responsibilities distinct.
3. Use explicit input, output, target, scorer, alignment, and overlap-aggregation contracts.
4. Source model implementations from author or credible open implementations; record paper, repository, commit, license, dependencies, and deviations.
5. Split raw recordings, trials, or subjects before windowing.
6. Fit normalization, feature selection, dimensionality reduction, and learned preprocessing on training data only.
7. Use validation data only for tuning, lag selection, checkpoint selection, thresholds, and early stopping.
8. Keep test data out of all model and protocol selection.
9. Preserve subject separation for cross-subject work and source separation for cross-dataset work.
10. Predeclare the primary metric, aggregation level, baseline set, comparison budget, and missing-run policy.
11. Store recording-, subject-, fold-, and seed-level results before aggregation.
12. Bind every result to code commit, resolved config, data version, split version, model version, and seed.
13. Distinguish development evidence, smoke evidence, engineering closure, and article-grade evidence.
14. Keep exploratory analyses separate from confirmatory claims.

## Global Execution Invariants

1. Pass a bounded smoke before a full run.
2. Keep the accepted smoke and full run identical in code path, adapters, target alignment, scorer, aggregation, split rules, training budget, batch size, DataLoader settings, checkpoint selection, resume semantics, and reused-row filtering.
3. Allow a full run to differ only by a larger dataset, subject, model, seed, or wall-clock scope.
4. Define a minimum job key as `dataset + subject_or_fold + model + seed`.
5. Write metrics incrementally after each job.
6. Mark a job complete only after required metric rows and metadata exist.
7. Keep successful results when later jobs fail.
8. Resume only verified missing or explicitly retried jobs.
9. Save and load-test checkpoints immediately after an independent run.
10. Never patch a runner during a claimed final run. Stop, isolate results, close engineering, re-smoke, and restart affected evidence.
11. Use a persistent user terminal for commands expected to exceed 30 minutes.
12. Never treat a tool timeout with no partial state as permission to rerun the same monolithic command.

## Hard Stops

Do not approve or continue a full run when:

- train, validation, or test membership is ambiguous;
- sliding windows were created before the split;
- held-out subjects or datasets influence fitted preprocessing or selection;
- prediction and target alignment contracts are missing or unvalidated;
- full settings differ from the accepted smoke;
- the formal config still contains smoke-only limits such as forced `epochs=1`;
- a completed job has no required metric rows;
- completed-job keys omit dataset, subject or fold, model, or seed;
- reused rows are not filtered to the active config;
- old blocked results are mixed into the active result directory;
- the runner changed after smoke without a new closure;
- the process timed out without usable run state, heartbeat, logs, metrics, or failures;
- the audit-loop role, branch, commit, or publication state is unresolved.

## State And Evidence

Maintain project state under `workflow/`:

- `START_HERE.md`: concise human-readable current state and next safe action;
- `state.json`: machine-readable phase, version identity, gates, and jobs;
- `decisions.md`: accepted scientific and engineering decisions;
- `backlog.md`: non-blocking deferred work;
- `knowledge/incidents.jsonl`: factual error records;
- `knowledge/error-patterns.md`: validated reusable project patterns;
- `knowledge/learned-rules.md`: accepted project rules;
- `knowledge/review-queue.md`: candidates for promotion.

Update state after a commit, formal config change, gate decision, completed or failed job, closed incident, and handoff. If state conflicts with checked artifacts, treat the artifacts as authoritative and repair state before proceeding.

## Learn From Errors

On every error:

1. Preserve the traceback, command, environment, branch, commit, config, job key, and affected paths.
2. Search existing error patterns.
3. Reuse a verified fix only when the signature and context match.
4. Diagnose root cause separately from symptoms.
5. Apply the smallest scoped fix.
6. Reproduce the original failure and verify the fix.
7. Re-run the affected contract check or smoke gate.
8. Record the incident with `scripts/record_incident.py`.
9. Add an automated check or regression test when practical.
10. Promote a pattern after three recurrences, or immediately for a verified high-severity leakage or evidence-corruption failure.

Keep one-off incidents in the project. Promote cross-project rules into this Skill only after deduplication and validation. Read `references/learning-and-error-memory.md` for the lifecycle.

## Role Summary

### Reviewer

- inspect scientific design, code, configs, raw evidence, statistics, and claim support;
- compute aggregate statistics from raw results;
- request only necessary artifacts;
- issue staged actions with inputs, outputs, acceptance checks, and stop conditions;
- own reviewer evidence and final verification states under `research-audit-loop`;
- defer low-contribution work to the backlog.

### Implementer

- integrate sourced models and datasets;
- run short preflight and smoke checks;
- provide complete commands for user-run long jobs;
- upload necessary code, configs, raw results, manifests, and compact logs;
- report facts, files, commits, and blockers without claiming independent verification;
- attempt push once, then stop and report `local_committed_push_blocked` with the exact manual command if it fails.

Read `references/collaboration.md` for complete role handoffs.

## Finish A Task

1. Validate the affected artifacts and schemas.
2. Update project state and error knowledge.
3. Report the active phase and gate status.
4. Report branch, full commit, resolved config, result path, completed, failed, and pending jobs when applicable.
5. Report which references and checks were applied.
6. Give only the next safe action.
7. For a formal audit round, finish under the publication and role rules of `research-audit-loop`.

## Maintain This Skill

Use imperative instructions. Keep this file as the router and invariant layer. Put stage-specific detail in one-level references, deterministic repeated work in scripts, and output templates in assets. Do not add a new rule until it is classified as global, stage-specific, scriptable, project-specific, or archival. Validate the Skill and forward-test realistic tasks after material changes.
