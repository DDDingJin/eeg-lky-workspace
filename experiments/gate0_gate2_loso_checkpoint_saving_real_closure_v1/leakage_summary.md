# Leakage Summary

- protocol: `gate0_gate2_loso_checkpoint_saving_real_closure_v1`
- seed: `0`
- pure LOSO required: heldout subject only in test, never in train/val/selection

## Per-job leakage checks
- dataset=`weissbart_tf64` subject=`P01` model=`eegnet` seed=`0` train_exclude=`True` val_exclude=`True` selection_exclude=`True` normalization_on_target=`False` test_subjects=`P01`
