# Auditory Benchmark Roadmap

This file separates the **task definitions** from the **model implementations** so the workspace does not become a pile of unrelated scripts.

## Benchmark Narrative

The benchmark should not become a flat collection of dataset-by-dataset scores.

It should answer a small number of research questions that remain useful both for:

- building a publishable benchmark paper
- learning the practical operating boundary of each model family

The current intended narrative is:

1. in-dataset comparison
   - which models are strongest when trained and tested within the same dataset
2. same-language cross-dataset generalization
   - whether models transfer across datasets without merely memorizing one acquisition setup
3. cross-language generalization
   - whether language changes strongly degrade speech-tracking performance
4. cross-modality generalization
   - whether models trained or tuned in EEG still retain value on MEG-style data

This means the benchmark should be organized around research questions, not brute-force traversal of every possible train/test pair.

## Evaluation Axes

Every dataset and every result row should eventually be tagged along these axes:

### By task

- `reconstruction`
- `match_mismatch`
- `aad`

### By evaluation regime

- `in_dataset`
- `cross_dataset_same_language`
- `cross_language`
- `cross_modality`

### By subject setting

- `subject_specific`
- `pooled_multi_subject`
- `leave_one_subject_out`

### By implementation mode

- `reference`
- `exact_structural_port`
- `local_baseline`
- `approximate_variant`

## Task Layers

### 1. Envelope Reconstruction

Input:
- EEG segment

Target:
- speech envelope

Typical metric:
- Pearson `r`
- MSE or `R^2` as secondary metrics

This is the cleanest first task for:
- ridge
- CCA-style reconstruction
- FCNN
- CNN
- VLAAI-like models

### 2. Match / Mismatch

Input:
- EEG segment
- one matched envelope segment
- one mismatched envelope segment

Target:
- choose which envelope matches the EEG

Typical metric:
- accuracy
- score margin

This can already be built from a single-speaker dataset by pairing the EEG window with:
- its true envelope window
- another envelope window from a shifted time region

### 3. Auditory Attention Decoding

Input:
- EEG segment
- attended candidate envelope
- unattended candidate envelope

Target:
- decide which candidate was attended

Typical metric:
- accuracy
- AUC

This is the classic AAD setup from many papers. It needs true competing-speaker candidate streams, not just a single-speaker envelope.

## Relationship Between Tasks

The benchmark should not treat reconstruction, match/mismatch, and AAD as unrelated silos.

The intended relationship is:

1. reconstruction is the primary low-level speech-tracking task
2. match/mismatch can be derived from reconstruction quality using matched and mismatched candidate envelopes
3. true AAD should be evaluated when datasets provide attended and unattended candidate streams

This gives:

- continuous-value metrics
  - Pearson `r`
  - `R^2`
  - MSE
- decision metrics
  - match/mismatch accuracy
  - AAD accuracy
  - AUC

This layered structure is stronger than reporting only envelope correlation.

## What The Current Hugo Sample Supports

Current local `mldecoders` sample supports:
- envelope reconstruction
- match / mismatch

It does **not** yet support full attended-vs-unattended AAD.

## What The Current Weissbart HDF5 Likely Supports

The Weissbart release is a much better candidate for larger-scale auditory benchmarking because it has:
- all participants in aligned `h5`
- story-wise organization
- official preprocessing helpers

After all participant files are downloaded, it should become the main dataset for:
- subject-specific reconstruction
- subject-independent reconstruction
- match / mismatch
- later reference-model runs

## Evaluation Axes
The minimal near-term evaluation plan is:

1. in-dataset reconstruction comparison on every currently unified public dataset
2. in-dataset match/mismatch where the dataset supports a clean matched-vs-shifted envelope construction
3. true AAD only on datasets with real attended/unattended candidate streams
4. only after that, cross-dataset and cross-language generalization layers

This order keeps the benchmark interpretable.

## Dataset Roles

- `hugo_sample_tf64`
  - secondary public evaluation dataset
  - usable for in-dataset reconstruction and match/mismatch
  - should no longer be described only as a smoke-test dataset
- `weissbart_tf64`
  - main public reconstruction benchmark
- `etard_tf64`
  - public reconstruction benchmark with multilingual and multi-condition value
- `SparrKULee`
  - future core benchmark for large-scale AAD/challenge-style evaluation
- `DTU / Fuglsang`
  - external generalization dataset family
- future `MEG`
  - cross-modality benchmark layer

## Language Strategy

The datasets should not be treated as if they all speak the same language or support the same linguistic claim.

The intended language-aware benchmark logic is:

- use same-language transfer to isolate dataset/protocol effects
- use cross-language transfer to estimate whether language shift matters
- prefer controlled comparisons inside one family when available
  - for example English versus Dutch conditions inside the Etard line

The desired scientific outcome is not to assume language does not matter.
The desired outcome is to test whether performance remains stable enough that language is not the dominant factor.

## Recommended Build Order

1. finish strong in-dataset comparisons on currently unified public datasets
2. complete the missing classical baseline family
3. complete the exact-port deep-model family
4. add task-aware reporting
   - reconstruction
   - match/mismatch
   - AAD
5. add same-language and cross-language generalization
6. add external and cross-modality evaluation layers

## Practical Rule

Do not ask one dataset or one metric to answer every question.

For now:
- use all currently unified public datasets for in-dataset comparison
- use Etard language structure for the first cross-language tests
- use later multi-speaker datasets for true AAD
- use DTU / Fuglsang and future MEG for external generalization, not for the first fair in-dataset leaderboard
