# Leakage / Adaptation Summary

- protocol: `gate0_gate2_loso_finetune_5min_closure_v1`
- source_checkpoint_local_id: `loso:eegnet:weissbart_tf64:P01:seed0:best_epoch_18`
- train_recording_ids: `train_-_P01_-_AUNP01, train_-_P01_-_AUNP02`
- val_recording_ids: `val_-_P01_-_AUNP01, val_-_P01_-_AUNP02, val_-_P01_-_AUNP03, val_-_P01_-_AUNP04`
- test_recording_ids: `test_-_P01_-_AUNP01, test_-_P01_-_AUNP02, test_-_P01_-_AUNP03, test_-_P01_-_AUNP04, test_-_P01_-_AUNP05, test_-_P01_-_AUNP06, test_-_P01_-_AUNP07, test_-_P01_-_AUNP08, test_-_P01_-_BROP01, test_-_P01_-_BROP02, test_-_P01_-_BROP03, test_-_P01_-_FLOP01, test_-_P01_-_FLOP02, test_-_P01_-_FLOP03, test_-_P01_-_FLOP04`
- no_overlap: `True`
- test recordings do not enter fine-tune train, fine-tune val, or checkpoint selection
- source checkpoint is same-dataset same-subject LOSO checkpoint
