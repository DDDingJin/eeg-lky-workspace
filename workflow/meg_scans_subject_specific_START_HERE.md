# MEG-SCANS subject-specific START_HERE

Branch:

`fix/ar-20260712-meg-scans-subject-specific-sub03-v1`

Base:

`fix/ar-20260712-meg-scans-canonical-model-readiness-v1 @ b58142827336b48ce7011b9a31e4522b39356e27`

Worktree:

`E:\decode\_meg_scans_subject_specific_sub03_v1`

Run:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_subject_specific\run_sub03_v1.py --config configs\benchmark\meg_scans_subject_specific\sub03_v1.json
```

Validate:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_subject_specific\validate_sub03_v1.py
F:\miniconda\envs\decode-torch\python.exe -m py_compile scripts\meg_scans_subject_specific\run_sub03_v1.py scripts\meg_scans_subject_specific\validate_sub03_v1.py
git diff --check
```

Output:

`experiments/meg_scans_subject_specific_sub03_v1/`

Do not commit MAT/FIF/raw/prediction/checkpoint/weight artifacts.
