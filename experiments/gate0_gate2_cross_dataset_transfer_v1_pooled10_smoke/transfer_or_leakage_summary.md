# Transfer / Leakage Summary

- protocol: `gate0_gate2_cross_dataset_transfer_v1`
- source training uses only source dataset train subjects
- source checkpoint selection uses only source dataset val subjects
- target pooled calibration uses only target original train/val recordings of target fixed test subjects
- target original test split remains final test only

## weissbart_tf64 -> etard_tf64
- source_train_subjects: `P00, P09, P02, P07, P11, P06, P04, P03`
- source_val_subjects: `P12, P10`
- target_test_subjects: `P17, P16, P10, P03`
- target_pooled_train_seconds: `1951.3125`
- target_pooled_val_seconds: `504.46875`
- target_final_test_seconds: `2419.53125`

