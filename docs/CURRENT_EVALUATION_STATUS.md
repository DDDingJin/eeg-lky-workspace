# Current Evaluation Status

Last updated: 2026-06-17

This note records:

- what has already been trained and evaluated
- how the current train/val/test procedure works
- which parts are method-development grade
- which parts are closer to article-grade
- what is still not fully standardized

## 1. Datasets Currently In The Unified Pipeline

The current unified `reference_splits` pipeline covers:

- `hugo_sample_tf64`
  - small development dataset
  - used for smoke tests, parity checks, and fast method comparisons
- `weissbart_tf64`
  - 13 participants
  - first serious article-oriented reconstruction benchmark
- `etard_tf64`
  - 20 participants
  - second serious reconstruction benchmark with more condition diversity

The following datasets are present or being downloaded, but are not yet behind the same unified benchmark adapter:

- `DTU / Fuglsang`
- `KUL AAD 4004271`
- `SparrKULee`

## 2. Methods Currently Available

### 2.1 Sample-only method panorama

The local `mldecoders`-style benchmark suite is currently available on `hugo_sample_tf64`:

- `ridge`
- `avgdec_ridge`
- `avgcorr_ridge`
- `cca_recon`
- `fcnn`
- `cnn`
- `eegnet`
- `adt_lite`
- `vlaai_lite`
- `adt_exact`
- `vlaai_exact`

Important:

- these methods are not yet all wired into `weissbart_tf64` and `etard_tf64`
- therefore the current full cross-method figure is valid only on the sample dataset

### 2.2 Exact structural ports on unified datasets

The following methods are currently integrated into the unified `reference_splits` pipeline:

- `adt_exact`
- `vlaai_exact`

These can now run on:

- `hugo_sample_tf64`
- `weissbart_tf64`
- `etard_tf64`

## 3. Current Results

### 3.1 Sample dataset: all currently comparable methods

Source file:

- [sample_all_methods_summary.csv](/E:/decode/experiments/summary_figures/sample_all_methods_summary.csv)

Current mean full-subject results:

- `Ridge`: `0.1459`
- `AvgDec-Ridge`: `0.1220`
- `AvgCorr-Ridge`: `0.1449`
- `CCA`: `0.1139`
- `FCNN`: `0.2058`
- `CNN`: `0.2176`
- `EEGNet`: `0.2171`
- `ADT-lite`: `0.2120`
- `VLAAI-lite`: `0.1177`
- `ADT-exact`: `0.1447`
- `VLAAI-exact`: `0.1477`

Associated figure:

- [sample_all_methods_overview.png](/E:/decode/experiments/summary_figures/sample_all_methods_overview.png)

### 3.2 Exact structural ports across unified datasets

Source file:

- [exact_reference_dataset_summary.csv](/E:/decode/experiments/summary_figures/exact_reference_dataset_summary.csv)

Current `100`-epoch-requested, early-stopped runs:

- `ADT-exact`
  - `hugo_sample_tf64`: `0.1447`
  - `weissbart_tf64`: `0.1147`
  - `etard_tf64`: `0.0860`
- `VLAAI-exact`
  - `hugo_sample_tf64`: `0.1477`
  - `weissbart_tf64`: `0.1389`
  - `etard_tf64`: `0.1128`

Associated figure:

- [exact_reference_dataset_overview.png](/E:/decode/experiments/summary_figures/exact_reference_dataset_overview.png)

## 4. How The Current Training Procedure Works

For the exact structural ports, the current procedure is:

1. load `train` split recordings
2. generate sliding windows dynamically from `train`
3. train one epoch on those windows
4. evaluate on `val`
5. keep the best checkpoint according to validation loss / validation Pearson metric
6. continue training until either:
   - `epochs_requested` is reached, or
   - early stopping is triggered after `patience` stale validation epochs
7. restore the best validation checkpoint
8. run final evaluation once on the `test` split

This means:

- `train` is used for optimization
- `val` is used for model selection
- `test` is used only for the final report

This part is correct in principle.

## 5. Important Clarification About `best_epoch`

The field `best_epoch` in the exact-port summaries does **not** mean the model stopped training immediately at that epoch.

It means:

- that epoch had the best validation score
- training may have continued for several more epochs
- early stopping only happened after enough later epochs failed to improve validation

Examples from the current runs:

- `VLAAI + weissbart`
  - `best_epoch = 6`
  - actual recorded history length = `17`
- `ADT + etard`
  - `best_epoch = 5`
  - actual recorded history length = `16`
- `VLAAI + etard`
  - `best_epoch = 5`
  - actual recorded history length = `16`
- `ADT + weissbart`
  - `best_epoch = 14`
  - actual recorded history length = `25`

So the current summaries are easy to misread.

The current code should later be improved to also store:

- `epochs_completed`
- `stopped_early`
- `selection_metric`

## 6. Is This Evaluation Procedure Reasonable?

### 6.1 What is already correct

The following parts are methodologically correct:

- separate `train`, `val`, and `test`
- choose checkpoints using `val`, not `test`
- report `test` only after model selection
- use the same reconstruction-style metric across the exact-port datasets

### 6.2 What is still unsatisfactory for a final paper

The following issues remain:

- not all baseline models are yet evaluated on the same dataset family
- sample-only methods and unified-dataset exact ports are still partially separated
- subject-independent evaluation is not yet standardized
- external generalization datasets are not yet in the same pipeline
- exact-port logs do not currently expose `epochs_completed`, which makes the stopping behavior harder to audit

### 6.3 The main remaining ambiguity

The main ambiguity is not whether `train/val/test` exists.

The main ambiguity is:

- which methods are being compared on exactly the same dataset
- which methods are only compared on the sample dataset
- which results are implementation checks versus benchmark conclusions

That is why the current outputs should be interpreted as:

- useful and technically meaningful
- but not yet the final article table

## 7. What Would Be More Article-Grade

The next more standardized evaluation layer should look like this:

1. choose one reconstruction dataset family for fair full-method comparison
   - ideally `weissbart_tf64`
   - then also `etard_tf64`
2. wire the same method set into the same unified pipeline
   - `ridge`
   - `cca`
   - `fcnn`
   - `cnn`
   - `eegnet`
   - `adt_exact`
   - `vlaai_exact`
3. keep one fixed split rule
4. keep one fixed metric definition
5. keep one fixed checkpoint-selection rule
   - best validation checkpoint
6. later add:
   - subject-independent evaluation
   - external generalization
   - true AAD datasets

## 8. Practical Bottom Line

Right now:

- the current exact-port evaluation logic is not fundamentally wrong
- the current results are useful
- but the current benchmark is still not fully standardized across methods

Therefore:

- the exact-port results on `weissbart_tf64` and `etard_tf64` can be trusted as internal benchmark results
- the sample full-method panorama can be trusted as a development comparison
- but the project still needs one more round of unification before claiming a final paper-grade leaderboard
