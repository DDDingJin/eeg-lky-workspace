# Leakage / Split Summary

- protocol: `gate0_gate2_subject_holdout_fixed_split_v1`
- split_id: `subject_holdout_fixed_split_v1`
- train_subjects: `P00, P09, P02, P07, P11, P06, P04, P03`
- val_subjects: `P12, P10`
- test_subjects: `P08, P05, P01`
- checkpoint selection uses val subjects only
- test subjects do not enter train, val, normalization fitting, or checkpoint selection
