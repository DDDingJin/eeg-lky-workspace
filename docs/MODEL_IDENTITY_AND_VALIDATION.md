# Model Identity And Validation

This document is the current Gate 0 through Gate 2 model-card layer for the benchmark branch.

It does not claim article-grade benchmark validity.  
It records what each model row currently means, what it is comparable to, and what evidence exists in this branch.

## Scope Rules

- `official_reference`: the original framework or repository is used directly as the anchor implementation.
- `faithful_port`: the local port intends to preserve the original model definition, but official parity evidence is still required before stronger labels are used.
- `architecture_baseline`: a representative implementation of a model family under the unified benchmark pipeline, not a claim of exact paper reproduction.
- `local_exploratory`: a local model kept for development or innovation, not an official benchmark anchor.

## Registry Coverage

The current registry file is:

- `registry/model_registry.csv`

The minimum audit-facing model list currently covered is:

- `linear`
- `dnn`
- `fcnn`
- `cnn`
- `ridge`
- `cca`
- `vlaai`
- `adt`
- `happyquokka`

## Current Interpretation By Family

### Classical baselines

- `ridge`
  - type: `architecture_baseline`
  - task support: `reconstruction`
  - current validation status: `real_sample_minimal_complete`
  - evidence:
    - `scripts/run_gate0_gate2_real_sample.py`
    - `experiments/gate0_gate2_real_sample/recording_metrics.csv`
    - `experiments/gate0_gate2_real_sample/boundary_validation.md`

- `cca`
  - type: `architecture_baseline`
  - task support: `reconstruction|match_mismatch`
  - current validation status: `real_sample_boundary_complete`
  - evidence:
    - `scripts/run_gate0_gate2_real_sample.py`
    - `experiments/gate0_gate2_real_sample/boundary_validation.md`
  - reporting rule:
    - canonical correlation, reconstruction correlation, and match-mismatch accuracy must remain separate columns

### Simple neural baselines

- `linear`
  - type: `architecture_baseline`
  - task support: `reconstruction`
  - current validation status: `registry_only`
  - current role:
    - registry completeness row
    - not yet promoted to this round's minimal real-sample execution set

- `dnn`
  - type: `architecture_baseline`
  - task support: `reconstruction`
  - current validation status: `registry_only`
  - current limitation:
    - still tied to an upstream external implementation path in historical runners

- `fcnn`
  - type: `architecture_baseline`
  - task support: `reconstruction`
  - current validation status: `real_sample_minimal_complete`
  - evidence:
    - `src/repro/simple_models.py`
    - `scripts/run_gate0_gate2_real_sample.py`
    - `experiments/gate0_gate2_real_sample/subject_metrics.csv`

- `cnn`
  - type: `architecture_baseline`
  - task support: `reconstruction`
  - current validation status: `registry_only`
  - current limitation:
    - historical runner remains available, but this branch did not yet upgrade `cnn` into the self-contained real-sample execution set

### Target auditory models

- `adt`
  - type: `faithful_port`
  - task support: `reconstruction`
  - current validation status: `real_sample_minimal_complete`
  - evidence:
    - `src/repro/adt_exact.py`
    - `scripts/run_gate0_gate2_real_sample.py`
    - `experiments/gate0_gate2_real_sample/recording_metrics.csv`
  - known limitation:
    - this branch still does not claim full TensorFlow-to-PyTorch parity evidence

- `vlaai`
  - type: `faithful_port`
  - task support: `reconstruction`
  - current validation status: `partial`
  - evidence:
    - `src/repro/vlaai_exact.py`
    - `scripts/run_vlaai_exact_reference.py`
  - known limitation:
    - parity artifacts are still incomplete

- `happyquokka`
  - type: `faithful_port`
  - task support: `reconstruction`
  - current validation status: `partial`
  - evidence:
    - `src/repro/happyquokka_reference.py`
    - `scripts/run_happyquokka_reference.py`
  - known limitation:
    - conditioned and unconditioned settings must remain distinct
    - optimizer and protocol parity are not closed in this round

## What This Document Does Not Claim

- It does not claim full official parity for `adt`, `vlaai`, or `happyquokka`.
- It does not claim that every registry row has already been rerun under the new unified scorer.
- It does not convert pipeline-validation outputs into paper-grade comparative findings.
