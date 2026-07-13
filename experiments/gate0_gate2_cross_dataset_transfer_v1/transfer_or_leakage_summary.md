# Transfer / Leakage Summary

- protocol: `gate0_gate2_cross_dataset_transfer_v1`
- models covered: `eegnet`, `fcnn`, `adt`
- source training uses only source dataset train subjects
- source checkpoint selection uses only source dataset val subjects
- target pooled10 calibration uses only target original train/val recordings from target fixed test subjects
- final evaluation uses only target original test split
- calibration train / calibration val / final test recording IDs have no overlap in either transfer direction

## weissbart_tf64 -> etard_tf64
- models: `eegnet`, `fcnn`, `adt`
- source_train_subjects: `P00, P09, P02, P07, P11, P06, P04, P03`
- source_val_subjects: `P12, P10`
- target_test_subjects: `P17, P16, P10, P03`
- target_pooled_train_seconds: `1951.3125`
- target_pooled_val_seconds: `504.46875`
- target_final_test_seconds: `2419.53125`
- calibration_source_scope: `target original train/val only`
- final_eval_scope: `target original test only`
- calibration_test_overlap: `none`

## etard_tf64 -> weissbart_tf64
- models: `eegnet`, `fcnn`, `adt`
- source_train_subjects: `P05, P02, P00, P01, P13, P04, P11, P07, P18, P06, P08, P12`
- source_val_subjects: `P09, P15, P14, P19`
- target_test_subjects: `P08, P05, P01`
- target_pooled_train_seconds: `1466.953125`
- target_pooled_val_seconds: `394.875`
- target_final_test_seconds: `715.359375`
- calibration_source_scope: `target original train/val only`
- final_eval_scope: `target original test only`
- calibration_test_overlap: `none`

