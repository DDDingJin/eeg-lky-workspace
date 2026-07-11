# MEG-SCANS sub-03 Official Preprocessing Validation Report

- Branch: `fix/ar-20260711-meg-scans-official-preprocessing-sub03-v1`
- Base: `fix/ar-20260711-meg-scans-representation-preflight-v1` / `69ed86096951e40728348dc1f7fc9d0b59c8556c`
- Official MEG-SCANS commit: `32bfc690e28e7591b45d96615c59b2d53b6a7165`
- Official function: `speech/decoding/preprocessing_audiobooks_decoding.m`
- Official settings: `speech/settings_speech.m`
- Official trialfun: `helper_functions/my_trialfun_audiobook.m`
- Local-only output MAT: `E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_preprocessed_audiobooks_decoding.mat`

## Input Precheck

All four sub-03 maxfiltered audiobook runs exist and are readable:

- `task-audiobook1_run-01`: 306 MEG channels, 102 mag, 204 grad.
- `task-audiobook1_run-02`: 306 MEG channels, 102 mag, 204 grad.
- `task-audiobook2_run-01`: 306 MEG channels, 102 mag, 204 grad.
- `task-audiobook2_run-02`: 306 MEG channels, 102 mag, 204 grad.

Raw event FIF files, events TSV files, and `sub-others_preprocessed_audiobook_envelopes_decoding.mat` are present. sub-03 T1w MRI, coreg transform, empty-room/noise files, calibration, and crosstalk are present.

## Protocol Confirmation

- `use_maxfilter = true`: confirmed from saved `results.settings`.
- `apply_latency_correction = true`: confirmed from saved `results.settings`.
- `audio_latency = 0.003 s`: confirmed from saved `results.settings`.
- `bandpass = [0.5, 4] Hz`: confirmed from saved `results.settings`.
- `trialdur = 120 s`: confirmed from saved `results.settings`.
- `fs_neuro = 1000 Hz`: confirmed from saved `results.settings`.
- `fs_down = 64 Hz`: confirmed from saved `results.settings` and `results.epochs_neuro.fsample`.

## Paired Trial Validation

`results.epochs_neuro`, `results.epochs_audio`, `results.mapping_epochs`, and `results.mapping_label` are present.

Mapping labels match the official audio labels:

- `task-audiobook1_run-01`
- `task-audiobook1_run-02`
- `task-audiobook2_run-01`
- `task-audiobook2_run-02`

Each run has 5 neuro trials defined by the official trialfun and 5 original audio trials. The official cleanup retains 4 paired trials per run, 16 paired trials total. The retained trials have 306 MEG channels and are truncated to 7680 samples at 64 Hz, exactly 120 seconds.

## Trial Removal

Each run removes 1 neuro/audio pair during the official cleanup because the last neuro trial is shorter than 99% of the 120 second target. No extra audio-only trials were removed. No audio/neuro run mismatch, empty retained run, or post-truncation length mismatch was observed.

## Compact Artifact Check

Only wrapper/settings, provenance, CSV/JSON/MD validation reports, and text summaries are intended for Git. The generated `.mat` file and raw/processed FIF/MRI/envelope arrays remain outside Git.
