# MEG-SCANS Canonical START HERE

- Branch: `fix/ar-20260712-meg-scans-canonical-05-08hz-64hz-sub03-v1`
- Worktree: `E:\decode\_meg_scans_canonical_05_08hz_64hz_sub03_v1`
- Base: `fix/ar-20260711-meg-scans-official-preprocessing-sub03-v1` / `8e8c694ee3cca1121b2031b0d00ddcfc0395d6bf`
- Local-only output root: `E:\decode\data\derived\meg_scans_canonical_05_08hz_64hz_v1`

## Resume order

1. Read `docs/meg_scans_canonical_05_08hz_64hz_sub03_zh.md`.
2. Read `experiments/meg_scans_canonical_05_08hz_64hz_sub03_v1/protocol_lock.json`.
3. Read `experiments/meg_scans_canonical_05_08hz_64hz_sub03_v1/run_manifest.json`.
4. Read `experiments/meg_scans_canonical_05_08hz_64hz_sub03_v1/trial_manifest.csv`.
5. Read `experiments/meg_scans_canonical_05_08hz_64hz_sub03_v1/psd_summary.csv`.
6. Read `experiments/meg_scans_canonical_05_08hz_64hz_sub03_v1/validation_report.md`.

## Commands

```powershell
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "addpath('E:\decode\_meg_scans_canonical_05_08hz_64hz_sub03_v1\scripts\meg_scans_canonical'); run_canonical_sub03_05_08hz_64hz"
& 'F:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "addpath('E:\decode\_meg_scans_canonical_05_08hz_64hz_sub03_v1\scripts\meg_scans_canonical'); validate_canonical_sub03_05_08hz_64hz"
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_canonical\validate_compact_artifacts.py
```

Do not commit MAT, FIF, raw data, predictions, checkpoints, weights, `.pt/.pth/.npy/.npz/.h5/.mat`, or large logs.
