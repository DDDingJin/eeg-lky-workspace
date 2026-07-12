# MEG-SCANS canonical model readiness START_HERE

Branch:

`fix/ar-20260712-meg-scans-canonical-model-readiness-v1`

Base:

`fix/ar-20260712-meg-scans-canonical-05-08hz-64hz-sub03-v1 @ 2837b68b590592add506f1353ebefe24dc063032`

Worktree:

`E:\decode\_meg_scans_canonical_model_readiness_v1`

Primary command:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_canonical_model_readiness\run_model_readiness_v1.py --config configs\benchmark\meg_scans_canonical_model_readiness\sub03_v1.json
```

Validation:

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\meg_scans_canonical_model_readiness\validate_model_readiness_v1.py
F:\miniconda\envs\decode-torch\python.exe -m py_compile scripts\meg_scans_canonical_model_readiness\run_model_readiness_v1.py scripts\meg_scans_canonical_model_readiness\validate_model_readiness_v1.py
git diff --check
```

Output directory:

`experiments/meg_scans_canonical_model_readiness_v1/`

Important constraints:

- Do not rerun canonical preprocessing.
- Do not modify local paired/envelope MAT.
- Do not use OLSA.
- Do not commit MAT/FIF/raw/prediction/checkpoint/weight arrays.
- Treat outputs as native-MEG readiness only, not a paper benchmark.
