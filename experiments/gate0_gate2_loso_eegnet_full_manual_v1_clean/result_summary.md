# EEGNet Weissbart LOSO Final Result Summary

- dataset: `weissbart_tf64`
- model: `eegnet`
- protocol: `LOSO subject-independent`
- seed: `0`
- n_subjects: `13`
- mean Pearson: `0.074045137678765`
- std Pearson: `0.043325203745608`
- min Pearson: `0.002031838814616`
- max Pearson: `0.127525475494850`
- failure status: `empty`
- note: `P09 was rerun from scratch after reboot recovery`

## Per-subject Metrics

| subject_id | metric | checkpoint_id | epochs_completed | best_epoch | best_val_score |
| --- | ---: | --- | ---: | ---: | ---: |
| P00 | 0.094792332895198 | eegnet_epoch_18 | 29 | 18 | 0.126156111088366 |
| P01 | 0.079584946761874 | eegnet_epoch_18 | 29 | 18 | 0.128408760522100 |
| P02 | 0.072719568371340 | eegnet_epoch_20 | 31 | 20 | 0.132722900981411 |
| P03 | 0.124048855998062 | eegnet_epoch_20 | 31 | 20 | 0.129418268507374 |
| P04 | 0.071331286914106 | eegnet_epoch_18 | 29 | 18 | 0.133714338165437 |
| P05 | 0.098352146633604 | eegnet_epoch_38 | 49 | 38 | 0.124370781332340 |
| P06 | 0.002031838814616 | eegnet_epoch_11 | 22 | 11 | 0.132350786756989 |
| P07 | 0.010386371754707 | eegnet_epoch_25 | 36 | 25 | 0.133029843549042 |
| P08 | 0.127525475494850 | eegnet_epoch_25 | 36 | 25 | 0.131439870579839 |
| P09 | 0.009737123785294 | eegnet_epoch_13 | 24 | 13 | 0.126080740265196 |
| P10 | 0.063060325605729 | eegnet_epoch_11 | 22 | 11 | 0.130203467014294 |
| P11 | 0.085108146401620 | eegnet_epoch_32 | 43 | 32 | 0.137460950381544 |
| P12 | 0.123908370392950 | eegnet_epoch_30 | 41 | 30 | 0.124105905098772 |

## Publish Checks

- completed_jobs count: `13`
- recording_metrics rows: `195`
- leakage checks: `all passed`
- failure_report: `empty`
- schema_validation_report passed: `true`
