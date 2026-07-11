# MEG-SCANS Official Replication START HERE

- Branch: `fix/ar-20260711-meg-scans-official-preprocessing-sub03-v1`
- Base: `fix/ar-20260711-meg-scans-representation-preflight-v1` / `69ed86096951e40728348dc1f7fc9d0b59c8556c`
- Worktree: `E:\decode\_meg_scans_official_preprocessing_sub03_v1`
- Local-only output root: `E:\decode\data\derived\meg_scans_official_replication_v1`

## Resume order

1. Read `docs/meg_scans_official_replication_sub03_zh.md`.
2. Read `experiments/meg_scans_official_preprocessing_sub03_v1/official_preprocessing_provenance.json`.
3. Read `experiments/meg_scans_official_preprocessing_sub03_v1/sub03_input_precheck.csv`.
4. Read `experiments/meg_scans_official_preprocessing_sub03_v1/sub03_official_trial_validation.csv`.
5. Read `experiments/meg_scans_official_preprocessing_sub03_v1/sub03_official_olsa_trial_validation.csv`.
6. Read `experiments/meg_scans_official_preprocessing_sub03_v1/sub03_official_decoding_anchor_summary.json`.
7. Read `experiments/meg_scans_official_preprocessing_sub03_v1/preprocessing_validation_report.md`.

## Commands

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_official_replication\precheck_sub03.py
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "run('E:\decode\_meg_scans_official_preprocessing_sub03_v1\scripts\meg_scans_official_replication\run_official_preprocessing_sub03.m'); run_official_preprocessing_sub03"
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "run('E:\decode\_meg_scans_official_preprocessing_sub03_v1\scripts\meg_scans_official_replication\validate_official_preprocessing_sub03.m'); validate_official_preprocessing_sub03"
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_official_replication\precheck_official_anchor_sub03.py
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "addpath('E:\decode\_meg_scans_official_preprocessing_sub03_v1\scripts\meg_scans_official_replication'); run_official_olsa_preprocessing_sub03"
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "addpath('E:\decode\_meg_scans_official_preprocessing_sub03_v1\scripts\meg_scans_official_replication'); validate_official_olsa_sub03"
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "addpath('E:\decode\_meg_scans_official_preprocessing_sub03_v1\scripts\meg_scans_official_replication'); run_official_training_sub03"
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "addpath('E:\decode\_meg_scans_official_preprocessing_sub03_v1\scripts\meg_scans_official_replication'); extract_official_decoding_anchor_sub03"
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_official_replication\validate_compact_artifacts.py
```

Do not commit `.mat`, FIF, MRI, envelope arrays, checkpoints, predictions, or large logs.
