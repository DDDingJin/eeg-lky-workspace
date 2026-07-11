# Preflight Validation Report

- Base branch: `origin/fix/ar-20260625-161300-a43831b-subject-specific-model-availability-v1`
- Base commit: `41a9c458731962b3dc1da575a159abb6a6c211a8`
- Subjects checked: `sub-01`, `sub-03`
- sub-03 block rows: `16`
- Unique block IDs: `16`
- R3 mapping status: blocked, because target EEG64 coordinates are unavailable from accepted EEG data.
- R4 status: interface only, blocked on target coordinates.
- R5 status: workflow only, required MRI/coreg/noise inputs present for inspected subjects.

## Validation summary

- Channel-count check: expected 306/102/204 from representative maxfilter FIF.
- Coordinate availability check: MEG magnetometer device coordinates available; EEG64 target coordinates unavailable.
- One-to-one mapping uniqueness check: not applicable while R3 is blocked.
- Block overlap/leakage check: block IDs are unique and 120 s windows are non-overlapping within each run.
- Raw/large-file check: artifacts are JSON/CSV/MD/PY only; no raw MEG/MRI/envelope arrays/checkpoints/predictions are added.
