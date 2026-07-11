# DECAF Data Interface Audit

- dataset_dir: `E:\decode\data\processed\reference_splits\weissbart_tf64`
- scaler_fit_split: `train`
- scaler_fit_recording_count: `15`
- split_recording_counts: `{'train': 15, 'val': 15, 'test': 15}`
- window_counts: `{'train': 528, 'val': 52, 'test': 52}`
- normalization_policy: `EEG channel z-score and envelope z-score fitted on train recordings only; train-fitted scalers are applied to train/val/test.`
- interval_rule: `For target [t,t+224), EEG uses [t,t+224), envelope_context uses [t-128,t), and TwoBranchModel forward trims to [t-96,t).`
- no_leakage: `True`
- no_cross_recording: `True`
- no_cross_split: `True`
- test_context_strictly_past: `True`
- test_initial_128_samples_excluded: `True`
