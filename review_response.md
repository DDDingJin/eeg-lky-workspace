# Review Response

Review round: `AR-20260625-161300-a43831b`  
Role: `implementer`  
Target branch: `audit/reproduction-note`  
Target commit: `a43831b83598c82520be320c21b56e92b73b7dcd`  
Fix branch: `fix/ar-20260625-161300-a43831b-gate0-gate2-real-evidence-and-paper`  
Updated at: `2026-06-27T00:00:00+08:00`

This response is still scoped around Gate 0 through Gate 2, but it advances the previous framework-only round into a minimal real-sample closure and a manuscript scaffold.

- Gate 0: review package completeness and traceability
- Gate 1: model identity and reproduction boundary clarity
- Gate 2: unified schema, scorer, and lag-boundary validation on a real sample

It does not claim article-grade benchmark completion.  
It does not claim multi-seed or multi-dataset benchmark closure.  
Artifacts under `experiments/gate0_gate2_real_sample/` are pipeline-validation evidence, not final benchmark claims.

## WORKPACKAGE-01

Status: `mostly complete`

What is completed:

- Registry fields were rewritten to match the requested minimum audit schema:
  - `model_name`
  - `implementation_type`
  - `source_paper`
  - `source_repo_or_local_path`
  - `runner_path`
  - `config_path`
  - `supported_tasks`
  - `current_validation_status`
  - `known_deviations`
- The required core rows are now present in:
  - `registry/model_registry.csv`
  - `linear`
  - `dnn`
  - `fcnn`
  - `cnn`
  - `ridge`
  - `cca`
  - `vlaai`
  - `adt`
  - `happyquokka`
- The registry validator was updated to validate the new field names and runner/config existence:
  - `src/benchmark/registry.py`
  - `scripts/validate_result_schema.py`
- The previous path pollution in `review_response.md` was removed. This branch no longer uses committed absolute Windows paths in the review response or fix manifest.
- A dedicated real-sample config and runner were added:
  - `configs/benchmark/gate0_gate2_real_sample.json`
  - `scripts/run_gate0_gate2_real_sample.py`

Evidence:

- Registry file:
  - `registry/model_registry.csv`
- Validation command:
  - `python scripts/validate_result_schema.py --check-model-registry`

What is not yet completed:

- Not every legacy benchmark script has been fully normalized into the new registry contract.
- Some older documentation files elsewhere in the repository still contain historical local path notes and dataset-location notes. Those are legacy operational documents, not the current audit response layer.

## WORKPACKAGE-02

Status: `partial`

What is completed:

- The model families now have explicit audit-facing identities in the registry and manuscript scaffold:
  - `architecture_baseline`
  - `faithful_port`
  - `local_exploratory`
- The current comparison scope is now explicitly separated for:
  - classical baseline rows
  - simple neural baseline rows
  - target faithful-port rows
- The current real-sample minimal closure includes:
  - `ridge`
  - `fcnn`
  - `adt`
- `cca` was retained as a boundary and metric-separation row:
  - reconstruction correlation is kept separate from canonical correlation
  - match-mismatch is treated as a distinct derived task output

Evidence:

- Registry:
  - `registry/model_registry.csv`
- Real-sample runner:
  - `scripts/run_gate0_gate2_real_sample.py`
- Manuscript tables:
  - `paper/draft_zh/tables/model_inventory_table.tex`

What is not yet completed:

- No official parity artifact pack was produced for `vlaai`, `adt`, or `happyquokka`.
- This branch still does not claim official-framework parity, TensorFlow-to-PyTorch parity, or official benchmark-number parity.
- `dnn` and `cnn` remain listed for coverage, but the branch-level minimal runnable closure currently prioritizes `fcnn` as the simple neural baseline because the original upstream baseline runner is not self-contained in this worktree.

Reason for remaining partial:

- The review required identity clarity before full parity closure.
- This round advances traceability and minimal real evidence, but not full official reproduction parity.

## WORKPACKAGE-03

Status: `real-sample minimal complete`

What is completed:

- The synthetic scorer/schema smoke path remains available:
  - `scripts/smoke_test_gate0_gate2.py`
- A real-sample pipeline-validation run was added on:
  - `hugo_sample_tf64_p00`
- The real-sample run uses a shared result schema and shared scorer for multiple models:
  - `ridge`
  - `fcnn`
  - `adt`
- All reported metrics in the real-sample output directory come from the shared scorer:
  - `experiments/gate0_gate2_real_sample/recording_metrics.csv`
  - `experiments/gate0_gate2_real_sample/subject_metrics.csv`
- Real-length lag-boundary validation was added:
  - `experiments/gate0_gate2_real_sample/boundary_validation.md`
- Scorer re-computation provenance was documented:
  - `experiments/gate0_gate2_real_sample/scorer_validation.md`

Evidence:

- Config:
  - `configs/benchmark/gate0_gate2_real_sample.json`
- Runner:
  - `scripts/run_gate0_gate2_real_sample.py`
- Output directory:
  - `experiments/gate0_gate2_real_sample/`

Important limitation:

- This is still only a minimal real-sample closure on a small public sample-like split.
- These outputs cannot be promoted to article-grade benchmark conclusions.

## WORKPACKAGE-04

Status: `planning`

Current state:

- The current engineering closure now supports the next phase:
  - multi-subject reruns
  - multi-dataset reruns
  - seed-aware scoring
  - protocol-level fairness checks

What remains:

- Multi-seed harmonization
- Budget harmonization across architectures
- Statistical testing layer
- Cross-dataset benchmark schedule

## WORKPACKAGE-05

Status: `manuscript scaffold complete`

What is completed:

- A paper scaffold was added under:
  - `paper/draft_zh/`
- The manuscript now includes:
  - benchmark motivation
  - task framing
  - dataset-task matrix
  - model inventory table
  - metric schema table
  - minimal result wording as pipeline validation only

Evidence:

- Main TeX file:
  - `paper/draft_zh/main.tex`
- Section files:
  - `paper/draft_zh/sections/introduction_zh.tex`
  - `paper/draft_zh/sections/methods_zh.tex`
  - `paper/draft_zh/sections/benchmark_design_zh.tex`
  - `paper/draft_zh/sections/results_placeholder_zh.tex`
  - `paper/draft_zh/sections/discussion_zh.tex`

## Commands Run

- `python scripts/validate_result_schema.py --check-model-registry --output experiments/gate0_gate2_smoke/registry_validation_report.json`
- `python scripts/smoke_test_gate0_gate2.py --phase all`
- `python scripts/run_gate0_gate2_real_sample.py --config configs/benchmark/gate0_gate2_real_sample.json`
- `python scripts/check_paper_draft.py`

## Summary

- WORKPACKAGE-01: mostly complete
- WORKPACKAGE-02: partial
- WORKPACKAGE-03: real-sample minimal complete
- WORKPACKAGE-04: planning
- WORKPACKAGE-05: manuscript scaffold complete

This branch should be reviewed as a Gate 0 through Gate 2 real-evidence branch. It upgrades the previous framework layer into a minimal real-sample benchmark closure plus a paper-ready scaffold, but it is not yet the final benchmark branch.
