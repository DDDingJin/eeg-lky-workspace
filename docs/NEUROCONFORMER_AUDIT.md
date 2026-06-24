# NeuroConformer Audit

Last updated: 2026-06-24

This note audits the `happy/NeuroConformer` code on `origin/master` and explains how it relates to the current local benchmark.

## Scope

The audited code lives on the remote branch:

- `origin/master:happy/NeuroConformer/`

It is not currently checked out into the working tree of the review branch, so this note summarizes the inspected remote code and the implications for integration.

## Main Conclusion

`NeuroConformer` is not an entirely separate benchmark family.

It is structurally a descendant of the `HappyQuokka` / `FFT_block` line, with a Conformer-style encoder and several training-time modifications layered on top.

So the right interpretation is:

- not "brand-new unrelated model"
- but "new variant built on the same reconstruction pipeline family"

## What It Reuses

The remote code keeps the same broad reconstruction setup as the `HappyQuokka`-style family:

- `64 Hz` input
- `10 s` windows
- pooled multi-subject training
- optional subject conditioning via `g_con`
- sequence-to-sequence envelope prediction

Shared family traits:

- explicit subject embedding path when `g_con=True`
- EEG-to-envelope regression
- train / val / test split folders with `train_-_*`, `val_-_*`, `test_-_*` naming

## What Changes Relative To HappyQuokka

### 1. Encoder family

`HappyQuokka` uses its original FFT / transformer-style decoder stack.

`NeuroConformer` replaces that stack with a Conformer-style encoder.

### 2. Default model width and depth

Observed defaults in `train.py`:

- `d_model = 256`
- `n_head = 4`
- `n_layers = 4`

Current local `HappyQuokka` reference wrapper uses:

- `d_model = 128`
- `n_head = 2`
- `n_layers = 8`

So the architecture is not just "Conformer instead of transformer".
It also changes capacity allocation:

- wider hidden size
- more heads
- fewer layers

### 3. Input frontend

`NeuroConformer v2` defaults to:

- `skip_cnn = True`

This means the model can bypass the original 3-layer CNN frontend and use a direct linear projection from raw EEG to `d_model`.

That is a major protocol difference from the current local `HappyQuokka` wrapper, which still uses the official upstream CNN-style frontend.

### 4. Output head

`NeuroConformer v2` adds:

- optional MLP output head

instead of the simple final linear projection.

### 5. Residual design

`NeuroConformer v2` adds:

- global residual around the Conformer stack
- optional gated residual fusion

These do not exist in the current local `HappyQuokka` reference wrapper.

### 6. Loss and optimization logic

The remote `train.py` uses a custom mixed objective:

- multi-scale Pearson loss
- Huber-style loss
- hand-tuned weighting between frequency scales

This differs from the current local `HappyQuokka` reference wrapper, which uses:

- Pearson loss
- L1 loss

This is not a minor implementation detail. It can substantially change both optimization and final scores.

### 7. Epoch budget

Remote `NeuroConformer` default:

- `500` epochs

Current local `HappyQuokka` formal runs:

- `100` epochs

This difference must be normalized before claiming a fair comparison.

### 8. Window sampling

Remote `NeuroConformer` training dataset introduces:

- `windows_per_sample = 20`

That means each recording contributes multiple random windows per epoch.

Current local wrapper does not currently mirror this exact training-sample multiplication policy.

## Quantified Protocol Differences

The most important measurable differences are:

1. hidden width doubled
   - `128 -> 256`
2. attention heads doubled
   - `2 -> 4`
3. depth changed
   - `8 -> 4`
4. training budget increased
   - `100 -> 500` epochs
5. training sample policy changed
   - single random crop logic vs `20` windows per sample
6. loss family changed
   - `Pearson + L1` vs `multi-scale Pearson + Huber`

These differences are large enough that raw leaderboard comparison would be misleading without a matched reproduction.

## Critical Fairness Warning

There is a serious fairness issue in the remote comparison utility.

In `happy/NeuroConformer/compare_all_models.py`, the script contains explicit logic that adds `0.02` to the `ADT` model's Pearson values before comparison.

That means any statistics or plots produced by that script are not acceptable as clean benchmark evidence without removing that adjustment and rerunning.

This should be treated as:

- a reporting distortion
- not a valid baseline comparison procedure

## What This Means For Local Integration

The correct local plan is:

1. add `NeuroConformer` as a separate named model family
2. port its architecture faithfully enough for local controlled testing
3. run it under the same local dataset adapters
4. compare it against:
   - `HappyQuokka g_con=False`
   - `HappyQuokka g_con=True`
   - `VLAAI-exact`
   - `ADT-exact`
   - strong classical baselines
5. separate:
   - architecture gain
   - conditioning gain
   - optimization / loss gain

## Recommended Quantification Plan

To make the effect sizes interpretable, the first local `NeuroConformer` integration should report:

### A. Family-level comparison

- `HappyQuokka g_con=False`
- `HappyQuokka g_con=True`
- `NeuroConformer g_con=False`
- `NeuroConformer g_con=True`

### B. Controlled ablations

- same loss, different encoder
- same encoder, with / without conditioning
- same encoder, with / without skip-CNN

### C. Standardized report

For each run:

- mean test correlation
- std across subjects
- best validation epoch
- epochs completed
- training curve
- parameter count

## Immediate Recommendation

The next local benchmark step should not be:

- directly trusting the remote comparison plots

It should be:

- bringing `NeuroConformer` into the local standardized evaluation layer
- removing ad hoc reporting distortions
- quantifying architecture and protocol differences under one benchmark adapter
