# Current Evaluation Status

Last updated: 2026-06-23

This note records:

- what has already been trained and evaluated
- how the current train/val/test procedure works
- which parts are method-development grade
- which parts are closer to article-grade
- what is still not fully standardized

## 1. Datasets Currently In The Unified Pipeline

The current unified `reference_splits` pipeline covers:

- `hugo_sample_tf64`
  - 13 participants
  - secondary public evaluation dataset
  - usable for in-dataset reconstruction and match/mismatch
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

## 1.1 Intended benchmark questions

The benchmark is not intended to be a flat leaderboard.

The current intended question structure is:

1. which methods are strongest in-dataset
2. which methods generalize across datasets within the same language regime
3. which methods are robust to language shift
4. which methods transfer to external or future cross-modality settings

Therefore future train/test comparisons should be added because they answer one of these questions, not because every pairwise dataset traversal must be exhausted.

## 1.2 Task layers

The benchmark should eventually report three connected task layers:

- `reconstruction`
- `match_mismatch`
- `aad`

Interpretation:

- reconstruction is the main continuous speech-tracking layer
- match/mismatch is the first decision layer derived from candidate-envelope comparison
- true AAD requires real attended and unattended candidate streams

## 1.3 Language-aware evaluation

The current datasets should not be treated as if language is irrelevant by construction.

Near-term language-aware logic should be:

- same-language cross-dataset transfer
  - isolates dataset/protocol effects more cleanly
- cross-language transfer
  - estimates whether language shift is a dominant degradation source
- the Etard family is especially valuable here because it already contains both English and Dutch conditions

Desired scientific use:

- not to assume language has no effect
- but to test whether language shift is small enough that model conclusions remain stable

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

## 2.3 Classical baseline family that still needs to be completed

The benchmark is still missing some classical rows that are important for interpretation:

- explicit lagged backward TRF / eTRF-style reconstruction baseline
- explicit lagged forward TRF baseline
- direct sparse `lasso` decoder
- `elastic_net` decoder
- stronger CCA variants
  - multi-lag CCA
  - forward/backward CCA-style variants when supported by the task protocol

These are not cosmetic additions.
They are necessary if later claims about deep-model advantage are going to be defensible.

### 2.2 Exact structural ports on unified datasets

The following methods are currently integrated into the unified `reference_splits` pipeline:

- `adt_exact`
- `vlaai_exact`
- `happyquokka_gcon`
- `null_gcon`

The following remote method has been audited but is not yet locally integrated into the unified benchmark runner:

- `NeuroConformer`
  - audited from `origin/master:happy/NeuroConformer/`
  - documented in `docs/NEUROCONFORMER_AUDIT.md`

Important naming note:

- the local integrated benchmark row for the `NeuroConformer` family is reported as `NULL`

These can now run on:

- `hugo_sample_tf64`
- `weissbart_tf64`
- `etard_tf64`

Important:

- `happyquokka_gcon` is not a plain pooled model
- the current run uses `g_con=True`
- this means subject identity is provided as an auxiliary conditioner
- this is acceptable for the current within-subject split protocol
- it should be reported separately from non-conditioned baselines

## 3. Current Results

### 3.1 Sample dataset: all currently comparable methods

Source file:

- `experiments/summary_figures/sample_all_methods_summary.csv`

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
- `HappyQuokka (g-con)`: `0.1105`
- `NULL`: `0.2103`

Associated figure:

- `experiments/summary_figures/sample_all_methods_overview.png`

### 3.2 Exact structural ports across unified datasets

Source file:

- `experiments/summary_figures/exact_reference_dataset_summary.csv`

Current `100`-epoch-requested, early-stopped runs:

- `ADT-exact`
  - `hugo_sample_tf64`: `0.1447`
  - `weissbart_tf64`: `0.1147`
  - `etard_tf64`: `0.0860`
- `VLAAI-exact`
  - `hugo_sample_tf64`: `0.1477`
  - `weissbart_tf64`: `0.1389`
  - `etard_tf64`: `0.1128`
- `HappyQuokka (g-con)`
  - `hugo_sample_tf64` is still a `20`-epoch fixed-budget reference run
  - `hugo_sample_tf64`: `0.1105`
  - `weissbart_tf64`: `0.1577` using the new `100`-epoch run
  - `etard_tf64`: `0.1287` using the new `100`-epoch run
- `NULL`
  - `hugo_sample_tf64`: `0.2103` using the current `10`-epoch local integration run
  - `weissbart_tf64`: `0.1993` using the current `10`-epoch local integration run
  - `etard_tf64`: `0.1480` using the current `10`-epoch local integration run

### 3.3 HappyQuokka conditioning check on article-oriented datasets

Source files:

- `experiments/summary_figures/happyquokka_conditioning_summary.csv`
- `experiments/summary_figures/happyquokka_conditioning_overview.png`
- `experiments/summary_figures/happyquokka_training_curves.png`

Current `100`-epoch runs:

- `weissbart_tf64`
  - `g_con=True`: `0.1577`
  - `g_con=False`: `0.1434`
- `etard_tf64`
  - `g_con=True`: `0.1287`
  - `g_con=False`: `0.1118`

Interpretation:

- the model benefits from subject conditioning on both datasets
- the gain is about `+0.0143` on `weissbart_tf64`
- the gain is about `+0.0169` on `etard_tf64`
- for a strict cross-method comparison, `g_con=False` is the fairer row to place next to non-conditioned baselines
- for a within-subject system comparison, `g_con=True` is a valid and stronger configuration

### 3.4 NULL conditioning check on unified datasets

Source files:

- `experiments/summary_figures/null_conditioning_summary.csv`
- `experiments/summary_figures/null_conditioning_overview.png`
- `experiments/summary_figures/null_training_curves.png`

Current `10`-epoch runs:

- `hugo_sample_tf64`
  - `g_con=True`: `0.2103`
  - `g_con=False`: `0.1957`
- `weissbart_tf64`
  - `g_con=True`: `0.1993`
  - `g_con=False`: `0.1751`
- `etard_tf64`
  - `g_con=True`: `0.1480`
  - `g_con=False`: `0.1244`

Interpretation:

- `NULL` is already strong under the current unified input and split interface
- `NULL` benefits from subject conditioning on all three currently integrated datasets
- for fairer cross-method comparison, `g_con=False` is the more conservative row
- for a strong within-subject reference result, `g_con=True` is the stronger row

Associated figure:

- `experiments/summary_figures/exact_reference_dataset_overview.png`

## 4. How The Current Training Procedure Works

For the exact structural ports and current HappyQuokka reference runs, the current procedure is:

1. load `train` split recordings
2. generate sliding windows dynamically from `train`
3. train one epoch on those windows
4. evaluate on `val`
5. keep the best checkpoint according to validation loss / validation Pearson metric
6. for `adt_exact` and `vlaai_exact`, continue until:
   - `epochs_requested` is reached, or
   - early stopping is triggered after `patience` stale validation epochs
7. for `happyquokka_gcon`, continue for the fixed requested epoch budget
8. restore the best validation checkpoint
9. run final evaluation once on the `test` split

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

For `happyquokka_gcon`, `epochs_completed` is already stored, and the current benchmark now also includes:

- a `100`-epoch conditioned run
- a matched `100`-epoch non-conditioned comparison run

For `null_gcon`, the current benchmark now includes:

- matched conditioned and non-conditioned runs on all three currently integrated datasets
- but only at the first local `10`-epoch integration budget
- so the current `NULL` row is already informative, but not yet final-budget tuned

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
- subject-conditioned and non-conditioned deep models are now separated for `HappyQuokka`, but not yet for the whole model family
- `NULL` is now also separated into conditioned and non-conditioned reporting rows
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
   - `happyquokka_gcon`
   - `null_gcon`
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
- the current `HappyQuokka` integration is operational and GPU-trainable
- the current results are useful
- but the current benchmark is still not fully standardized across methods

Therefore:

- the exact-port results on `weissbart_tf64` and `etard_tf64` can be trusted as internal benchmark results
- the sample full-method panorama can be trusted as a development comparison
- but the project still needs one more round of unification before claiming a final paper-grade leaderboard
