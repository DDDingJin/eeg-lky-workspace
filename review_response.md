# Review Response

Review round: `AR-20260625-161300-a43831b`  
Role: `implementer`  
Target branch: `audit/reproduction-note`  
Target commit: `a43831b83598c82520be320c21b56e92b73b7dcd`  
Fix branch: `fix/ar-20260625-161300-a43831b-gate0-gate2`  
Updated at: `2026-06-27T00:00:00+08:00`

This response is scoped to Gate 0 through Gate 2 only.

- Gate 0: review package completeness and traceability
- Gate 1: model identity and reproduction boundary clarity
- Gate 2: unified schema, scorer, and lag-boundary validation

It does not claim article-grade benchmark completion.
It does not claim multi-seed or multi-dataset benchmark closure.
Smoke-test outputs below are validation artifacts, not paper results.

## WORKPACKAGE-01

Status: `partially completed`

What is completed:

- Restored key code paths explicitly flagged as missing in the review:
  - [happyquokka_reference.py](/E:/decode/_fix_gate0_gate2/src/repro/happyquokka_reference.py)
  - [run_reference_baselines.py](/E:/decode/_fix_gate0_gate2/scripts/run_reference_baselines.py)
  - [run_reference_exact_suite.py](/E:/decode/_fix_gate0_gate2/scripts/run_reference_exact_suite.py)
  - [cca.py](/E:/decode/_fix_gate0_gate2/src/repro/mldecoders/cca.py)
  - [linear_baselines.py](/E:/decode/_fix_gate0_gate2/src/repro/mldecoders/linear_baselines.py)
- Added a registry that records model identity, source, runner path, config path, comparable scope, and current validation status:
  - [model_registry.csv](/E:/decode/_fix_gate0_gate2/registry/model_registry.csv)
- Added a validator that checks whether the registry rows point to real runner/config files:
  - [validate_result_schema.py](/E:/decode/_fix_gate0_gate2/scripts/validate_result_schema.py)
- Added a minimal identity and validation note:
  - [MODEL_IDENTITY_AND_VALIDATION.md](/E:/decode/_fix_gate0_gate2/docs/MODEL_IDENTITY_AND_VALIDATION.md)

Evidence:

- Registry smoke output will be written to:
  - `experiments/gate0_gate2_smoke/registry_validation.json`
- Validation command:
  - `python scripts/validate_result_schema.py --check-model-registry --output experiments/gate0_gate2_smoke/registry_validation_report.json`

What is not yet completed:

- Full environment lockfiles, full config matrix coverage, and per-run log manifests for all historical results are not closed in this branch.
- This branch restores traceability for core rows and adds a registry/validator baseline, but it is not yet a complete publication release package.

## WORKPACKAGE-02

Status: `partially completed`

What is completed:

- Introduced explicit model identity categories required by the review:
  - `official_reference`
  - `faithful_port`
  - `architecture_baseline`
  - `local_exploratory`
- Recorded current known deviations and current validation scope for:
  - `ridge`
  - `cca`
  - `fcnn`
  - `cnn`
  - `eegnet`
  - `adt_exact`
  - `vlaai_exact`
  - `happyquokka_gcon`
  - `null_gcon`
- Restored the HappyQuokka implementation path so the row is no longer only a reported result without code:
  - [happyquokka_reference.py](/E:/decode/_fix_gate0_gate2/src/repro/happyquokka_reference.py)

Evidence:

- Model identity source:
  - [model_registry.csv](/E:/decode/_fix_gate0_gate2/registry/model_registry.csv)
- Human-readable model cards:
  - [MODEL_IDENTITY_AND_VALIDATION.md](/E:/decode/_fix_gate0_gate2/docs/MODEL_IDENTITY_AND_VALIDATION.md)

What is not yet completed:

- Full parity artifact packs for ADT, VLAAI, and HappyQuokka are still pending.
- This branch does not claim that `adt_exact` or `vlaai_exact` has passed full official-framework parity.
- Those rows are deliberately labeled `faithful_port` and `partial`, not article-grade exact reproductions.

Reason for partial completion:

- The review asked for model identity and reproducibility boundaries first.
- Full parity work is a Gate 1 continuation, but not required to close Gate 0 package completeness or Gate 2 schema/scorer closure.

## WORKPACKAGE-03

Status: `completed for smoke-test scope`

What is completed:

- Added a unified prediction schema:
  - [result_schema.py](/E:/decode/_fix_gate0_gate2/src/benchmark/result_schema.py)
- Added a unified reconstruction scorer that:
  - reconstructs continuous recording-level predictions from overlapping windows
  - computes recording-level Pearson only on valid samples
  - aggregates recording rows into subject rows
  - writes `prediction_samples.csv`, `recording_metrics.csv`, and `subject_metrics.csv`
  - [scoring.py](/E:/decode/_fix_gate0_gate2/src/benchmark/scoring.py)
- Added a boundary utility that demonstrates the exact cross-recording lag problem and the safe alternative:
  - [boundaries.py](/E:/decode/_fix_gate0_gate2/src/benchmark/boundaries.py)
- Added a smoke script that generates schema-valid synthetic prediction artifacts and a lag-boundary report:
  - [smoke_test_gate0_gate2.py](/E:/decode/_fix_gate0_gate2/scripts/smoke_test_gate0_gate2.py)

Evidence:

- Smoke output directory:
  - `experiments/gate0_gate2_smoke/`
- Generated files after running the smoke script:
  - `prediction_samples.csv`
  - `recording_metrics.csv`
  - `subject_metrics.csv`
  - `boundary_validation.json`
  - `scorer_smoke_summary.json`
  - `smoke_manifest.json`
- Validation command:
  - `python scripts/smoke_test_gate0_gate2.py --phase all`
- Schema check command:
  - `python scripts/validate_result_schema.py --check-metrics-dir experiments/gate0_gate2_smoke --output experiments/gate0_gate2_smoke/schema_validation_report.json`

Important limitation:

- The current completion is intentionally a smoke-test closure, not a full historical re-scoring of all benchmark runs.
- It proves that the branch now has a runnable common schema, common scorer, and boundary validation path.

## WORKPACKAGE-04

Status: `not executed in this round`

Reason:

- This round is intentionally restricted to Gate 0 through Gate 2.
- Multi-seed fairness, budget harmonization, and formal statistics require the unified scorer outputs to exist first.

Planned next step:

- Freeze protocol families after the Gate 2 schema is accepted.
- Then add per-seed, per-protocol result manifests and formal statistical scripts.

## WORKPACKAGE-05

Status: `not executed in this round`

Reason:

- This round is intentionally restricted to Gate 0 through Gate 2.
- The long-term benchmark blueprint was read and used as a scoping constraint, but task hierarchy, dataset roles, EEG/MEG layering, and manuscript figure/table completion are not closed here.

Planned next step:

- Keep the current branch as the technical traceability foundation.
- Use the accepted schema and model identities as the base for the blueprint-driven benchmark expansion.

## Commands Run

- `python scripts/smoke_test_gate0_gate2.py --phase all`
- `python scripts/validate_result_schema.py --check-model-registry --output experiments/gate0_gate2_smoke/registry_validation_report.json`
- `python scripts/validate_result_schema.py --check-metrics-dir experiments/gate0_gate2_smoke --output experiments/gate0_gate2_smoke/schema_validation_report.json`

## Summary

- WORKPACKAGE-01: partially completed
- WORKPACKAGE-02: partially completed
- WORKPACKAGE-03: completed for smoke-test scope
- WORKPACKAGE-04: deferred by scope
- WORKPACKAGE-05: deferred by scope

This branch should be reviewed as a Gate 0 to Gate 2 repair branch, not as a completed benchmark branch.
