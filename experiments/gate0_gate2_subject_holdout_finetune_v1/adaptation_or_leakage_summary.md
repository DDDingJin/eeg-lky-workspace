# Adaptation / Leakage Summary

- protocol: `gate0_gate2_subject_holdout_finetune_v1`
- split_id: `subject_holdout_fixed_split_v1`
- target_subjects: `P01, P05, P08`
- source_checkpoint_local_id: `subject_holdout:eegnet:weissbart_tf64:subject_holdout_fixed_split_v1:seed0:best_epoch_5`
- calibration uses target subject train split recordings only for fine-tune train budget
- calibration uses target subject val split recordings only for fine-tune validation budget
- fixed test recordings remain unchanged from zero-shot and do not enter fine-tune train, validation, normalization fitting, or checkpoint selection

## P01
- train calibration recordings: `train_-_P01_-_AUNP01, train_-_P01_-_AUNP02`
- val calibration recordings: `val_-_P01_-_AUNP01, val_-_P01_-_AUNP02, val_-_P01_-_AUNP03, val_-_P01_-_AUNP04`
- fixed test recordings: `test_-_P01_-_AUNP01, test_-_P01_-_AUNP02, test_-_P01_-_AUNP03, test_-_P01_-_AUNP04, test_-_P01_-_AUNP05, test_-_P01_-_AUNP06, test_-_P01_-_AUNP07, test_-_P01_-_AUNP08, test_-_P01_-_BROP01, test_-_P01_-_BROP02, test_-_P01_-_BROP03, test_-_P01_-_FLOP01, test_-_P01_-_FLOP02, test_-_P01_-_FLOP03, test_-_P01_-_FLOP04`
- actual_calibration_seconds: `318.09375`
- no_overlap: `True`

## P05
- train calibration recordings: `train_-_P05_-_AUNP01, train_-_P05_-_AUNP02`
- val calibration recordings: `val_-_P05_-_AUNP01, val_-_P05_-_AUNP02, val_-_P05_-_AUNP03, val_-_P05_-_AUNP04`
- fixed test recordings: `test_-_P05_-_AUNP01, test_-_P05_-_AUNP02, test_-_P05_-_AUNP03, test_-_P05_-_AUNP04, test_-_P05_-_AUNP05, test_-_P05_-_AUNP06, test_-_P05_-_AUNP07, test_-_P05_-_AUNP08, test_-_P05_-_BROP01, test_-_P05_-_BROP02, test_-_P05_-_BROP03, test_-_P05_-_FLOP01, test_-_P05_-_FLOP02, test_-_P05_-_FLOP03, test_-_P05_-_FLOP04`
- actual_calibration_seconds: `318.09375`
- no_overlap: `True`

## P08
- train calibration recordings: `train_-_P08_-_AUNP01, train_-_P08_-_AUNP02`
- val calibration recordings: `val_-_P08_-_AUNP01, val_-_P08_-_AUNP02, val_-_P08_-_AUNP03, val_-_P08_-_AUNP04`
- fixed test recordings: `test_-_P08_-_AUNP01, test_-_P08_-_AUNP02, test_-_P08_-_AUNP03, test_-_P08_-_AUNP04, test_-_P08_-_AUNP05, test_-_P08_-_AUNP06, test_-_P08_-_AUNP07, test_-_P08_-_AUNP08, test_-_P08_-_BROP01, test_-_P08_-_BROP02, test_-_P08_-_BROP03, test_-_P08_-_FLOP01, test_-_P08_-_FLOP02, test_-_P08_-_FLOP03, test_-_P08_-_FLOP04`
- actual_calibration_seconds: `318.09375`
- no_overlap: `True`
