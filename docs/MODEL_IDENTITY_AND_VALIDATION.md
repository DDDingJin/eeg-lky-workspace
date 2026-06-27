# Model Identity And Validation

This document is the minimal Gate 0 to Gate 2 model card layer for the benchmark branch.

It does not claim article-grade benchmark validity.
It records what each current row is, what it is comparable to, and what evidence exists today.

## Scope Rules

- `official_reference`: use the original framework or repository as the anchor implementation.
- `faithful_port`: local port intended to preserve the original model definition, but parity evidence is still required before using stronger labels such as `exact`.
- `architecture_baseline`: a representative implementation of a model family under the unified benchmark pipeline, not a claim of exact paper reproduction.
- `local_exploratory`: local model kept for development and future innovation, not an official benchmark anchor.

## Model Summary

| Model | Type | Source repo | Source ref | Current task support | Comparable scope | Current validation evidence |
|---|---|---|---|---|---|---|
| `ridge` | `architecture_baseline` | local baseline suite | `local-repo` | reconstruction | unified reconstruction scorer only | model registry, boundary smoke, scorer smoke |
| `cca` | `architecture_baseline` | local baseline suite | `local-repo` | reconstruction, optional match-mismatch | keep canonical / reconstruction / decision metrics separate | model registry, boundary smoke, scorer smoke |
| `fcnn` | `architecture_baseline` | `mldecoders`-style adapted baseline | `adapted-local` | reconstruction | unified reconstruction pipeline only | model registry and runner path |
| `cnn` | `architecture_baseline` | `mldecoders`-style adapted baseline | `adapted-local` | reconstruction | unified reconstruction pipeline only | model registry and runner path |
| `eegnet` | `architecture_baseline` | adapted local implementation | `local-repo` | reconstruction | unified reconstruction pipeline only | model registry and runner path |
| `adt_exact` | `faithful_port` | `ruix6/ADT_Network` | `e31ad252960e05856abb9c54669c3f65ebfa49ac` | reconstruction | do not claim article-grade exact parity yet | runner path plus documented known deviations and pending parity status |
| `vlaai_exact` | `faithful_port` | `vlaai` | `9af13054bf73188e9c248f249f44ecc0cd02f6c5` | reconstruction | do not claim article-grade exact parity yet | runner path plus documented known deviations and pending parity status |
| `happyquokka_gcon` | `faithful_port` | `jkyunnng/HappyQuokka_system_for_EEG_Challenge` | `upstream-main` | reconstruction | conditioned runs must stay separate from plain pooled baselines | code path restored, runner path documented, parity still partial |
| `null_gcon` | `local_exploratory` | local | `local-repo` | reconstruction | local exploratory only | existing runner and audit note |

## Minimal Model Cards

### `ridge`

- Model name: `ridge`
- Implementation type: `architecture_baseline`
- Source paper / source repo / commit or ref: backward TRF-style linear decoder family / local benchmark implementation / `local-repo`
- Input-output definition: lagged EEG samples to single-channel envelope prediction
- Current task support: reconstruction
- Known deviations: this branch does not claim one fixed upstream package parity; it standardizes a benchmark baseline
- Comparable scope: compare only on unified reconstruction scorer outputs
- Current validation evidence:
  - registry row in [model_registry.csv](/E:/decode/_fix_gate0_gate2/registry/model_registry.csv)
  - boundary smoke artifact path documented in `experiments/gate0_gate2_smoke/boundary_validation.json` after running the smoke script

### `cca`

- Model name: `cca`
- Implementation type: `architecture_baseline`
- Source paper / source repo / commit or ref: CCA reconstruction family / local benchmark implementation / `local-repo`
- Input-output definition: lagged EEG and lagged envelope mapped into canonical subspaces, then reconstruction readout
- Current task support: reconstruction and optional match-mismatch style analysis
- Known deviations: canonical correlation and reconstruction correlation are not the same task metric and must remain separate in reporting
- Comparable scope: use as architecture baseline unless a single upstream package is later frozen and parity-tested
- Current validation evidence:
  - registry row in [model_registry.csv](/E:/decode/_fix_gate0_gate2/registry/model_registry.csv)
  - boundary smoke artifact path documented in `experiments/gate0_gate2_smoke/boundary_validation.json` after running the smoke script

### `fcnn`, `cnn`, `eegnet`

- Model name: `fcnn`, `cnn`, `eegnet`
- Implementation type: `architecture_baseline`
- Source paper / source repo / commit or ref:
  - `fcnn` and `cnn`: `mldecoders`-style adapted local baselines
  - `eegnet`: adapted local implementation of the EEGNet family
- Input-output definition: windowed EEG to scalar envelope target
- Current task support: reconstruction
- Known deviations: not claimed as exact paper reproductions
- Comparable scope: unified reconstruction pipeline only
- Current validation evidence:
  - registry rows in [model_registry.csv](/E:/decode/_fix_gate0_gate2/registry/model_registry.csv)
  - runnable baseline entrypoint [run_reference_baselines.py](/E:/decode/_fix_gate0_gate2/scripts/run_reference_baselines.py)

### `adt_exact`

- Model name: `adt_exact`
- Implementation type: `faithful_port`
- Source paper / source repo / commit or ref: ADT / `ruix6/ADT_Network` / `e31ad252960e05856abb9c54669c3f65ebfa49ac`
- Input-output definition: recording-aligned EEG sequence to sequence-aligned envelope prediction
- Current task support: reconstruction
- Known deviations:
  - this branch documents the model path and runner but does not yet ship full TensorFlow-to-PyTorch parity artifacts
  - do not use `exact` as a paper claim without additional parity evidence
- Comparable scope: compare only within the unified reconstruction protocol and label as faithful port
- Current validation evidence:
  - source implementation [adt_exact.py](/E:/decode/_fix_gate0_gate2/src/repro/adt_exact.py)
  - runner [run_adt_exact_reference.py](/E:/decode/_fix_gate0_gate2/scripts/run_adt_exact_reference.py)

### `vlaai_exact`

- Model name: `vlaai_exact`
- Implementation type: `faithful_port`
- Source paper / source repo / commit or ref: VLAAI / `vlaai` / `9af13054bf73188e9c248f249f44ecc0cd02f6c5`
- Input-output definition: recording-aligned EEG sequence to sequence-aligned envelope prediction
- Current task support: reconstruction
- Known deviations:
  - LeakyReLU slope and LayerNorm epsilon parity to the official TensorFlow anchor still need explicit artifact-level confirmation
  - do not use `exact` as a paper claim without additional parity evidence
- Comparable scope: compare only within the unified reconstruction protocol and label as faithful port
- Current validation evidence:
  - source implementation [vlaai_exact.py](/E:/decode/_fix_gate0_gate2/src/repro/vlaai_exact.py)
  - runner [run_vlaai_exact_reference.py](/E:/decode/_fix_gate0_gate2/scripts/run_vlaai_exact_reference.py)

### `happyquokka_gcon`

- Model name: `happyquokka_gcon`
- Implementation type: `faithful_port`
- Source paper / source repo / commit or ref: HappyQuokka challenge system / `jkyunnng/HappyQuokka_system_for_EEG_Challenge` / `upstream-main`
- Input-output definition: recording-aligned EEG sequence with optional subject conditioner to sequence-aligned envelope prediction
- Current task support: reconstruction
- Known deviations:
  - current branch restores the code path and runner but does not yet ship a full parity artifact pack for optimizer, loss, and windowing
  - `g_con=True` conditioned runs must stay separate from plain pooled baselines
- Comparable scope: conditioned within-subject or seen-subject comparisons only unless an unconditioned counterpart is used
- Current validation evidence:
  - source implementation [happyquokka_reference.py](/E:/decode/_fix_gate0_gate2/src/repro/happyquokka_reference.py)
  - runner [run_happyquokka_reference.py](/E:/decode/_fix_gate0_gate2/scripts/run_happyquokka_reference.py)

### `null_gcon`

- Model name: `null_gcon`
- Implementation type: `local_exploratory`
- Source paper / source repo / commit or ref: local only / local / `local-repo`
- Input-output definition: recording-aligned EEG sequence with optional subject conditioner
- Current task support: reconstruction
- Known deviations: not an official benchmark anchor and not a reproduction target
- Comparable scope: local exploratory comparisons only
- Current validation evidence:
  - runner [run_neuroconformer_reference.py](/E:/decode/_fix_gate0_gate2/scripts/run_neuroconformer_reference.py)
  - existing audit note [NEUROCONFORMER_AUDIT.md](/E:/decode/_fix_gate0_gate2/docs/NEUROCONFORMER_AUDIT.md)
