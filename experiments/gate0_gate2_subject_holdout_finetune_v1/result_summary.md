# Subject-Holdout Fine-Tuning v1

- dataset: `weissbart_tf64`
- model: `eegnet`
- seed: `0`
- split_id: `subject_holdout_fixed_split_v1`
- source_checkpoint_local_id: `subject_holdout:eegnet:weissbart_tf64:subject_holdout_fixed_split_v1:seed0:best_epoch_5`
- target_subjects: `P01, P05, P08`
- n_subjects: `3`
- mean_fine_tuned_pearson: `0.08248697260175615`
- std_fine_tuned_pearson: `0.03159187053267058`
- min_fine_tuned_pearson: `0.057729342231511424`
- max_fine_tuned_pearson: `0.1270739269673292`

## Zero-Shot vs Fine-Tune

### P01
- zero_shot_metric: `0.06733708950404578`
- fine_tuned_metric: `0.06265764860642783`
- delta: `-0.004679440897617945`
- actual_calibration_seconds: `318.09375`
- fine_tune_best_epoch: `41`
- fine_tune_best_val_score: `0.12646214850246906`

### P05
- zero_shot_metric: `0.04903786329944087`
- fine_tuned_metric: `0.057729342231511424`
- delta: `0.008691478932070555`
- actual_calibration_seconds: `318.09375`
- fine_tune_best_epoch: `1`
- fine_tune_best_val_score: `0.18614593148231506`

### P08
- zero_shot_metric: `0.1270739269673292`
- fine_tuned_metric: `0.1270739269673292`
- delta: `0.0`
- actual_calibration_seconds: `318.09375`
- fine_tune_best_epoch: `0`
- fine_tune_best_val_score: `0.08105507958680391`

