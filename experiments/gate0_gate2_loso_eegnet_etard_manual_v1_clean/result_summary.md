# Etard EEGNet Sampled LOSO Seed0 v1

This directory is a compact scientific result snapshot for Etard EEGNet sampled LOSO seed0 v1.

- dataset: `etard_tf64`
- model: `eegnet`
- protocol: `pure LOSO`
- seed: `0`
- sampled subjects: `P00, P05, P10, P15`
- sampled subject count: `4`
- note: this is **not** the full Etard 20-subject LOSO package
- note: `P05/P10/P15` are part of a fixed preset sampled subset; they were **not** selected based on model results
- failure_report: `empty`
- sampled-scope schema: `passed`
- full 20-subject global schema: `not claimed in this snapshot`

## Subject Metrics

- `P00`: metric=`0.039849916028856534`, checkpoint_id=`eegnet_epoch_22`, epochs_completed=`33`, best_epoch=`22`, best_val_score=`0.08803662887368781`, num_recordings=`24`
- `P05`: metric=`0.14239045302534853`, checkpoint_id=`eegnet_epoch_7`, epochs_completed=`18`, best_epoch=`7`, best_val_score=`0.08864196114512242`, num_recordings=`24`
- `P10`: metric=`0.11035483543097183`, checkpoint_id=`eegnet_epoch_9`, epochs_completed=`20`, best_epoch=`9`, best_val_score=`0.08619034108096385`, num_recordings=`40`
- `P15`: metric=`0.03243815797473619`, checkpoint_id=`eegnet_epoch_10`, epochs_completed=`21`, best_epoch=`10`, best_val_score=`0.0916321649071975`, num_recordings=`40`

## Scope Guard

- This snapshot is intended for review of sampled Etard EEGNet LOSO behavior only.
- It must not be interpreted as a complete 20-subject Etard LOSO benchmark closure.
