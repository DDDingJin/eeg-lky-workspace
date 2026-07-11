# MEG-SCANS sub-03 Official Decoding Replication Anchor Report

- Branch: `fix/ar-20260711-meg-scans-official-preprocessing-sub03-v1`
- Base: `fix/ar-20260711-meg-scans-representation-preflight-v1` / `69ed86096951e40728348dc1f7fc9d0b59c8556c`
- Official MEG-SCANS commit: `32bfc690e28e7591b45d96615c59b2d53b6a7165`
- Official function: `speech/decoding/preprocessing_audiobooks_decoding.m`
- Official OLSA function: `speech/decoding/preprocessing_olsa_decoding.m`
- Official training function: `speech/decoding/training_decoding.m`
- Official settings: `speech/settings_speech.m`
- Official trialfun: `helper_functions/my_trialfun_audiobook.m`
- MATLAB: `24.2.0.2712019 (R2024b)`
- mTRF Toolbox: local toolbox path `E:/decode/external/toolboxes/mTRF-Toolbox-master/mTRF-Toolbox-master/mtrf`; user-installed package recorded as mTRF-Toolbox 2.7, version metadata not exposed by the toolbox files inspected.
- Local-only output MAT: `E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_preprocessed_audiobooks_decoding.mat`
- Local-only OLSA MAT: `E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_preprocessed_olsa_decoding.mat`
- Local-only decoding MAT: `E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_decoding.mat`

This is an official replication anchor only; it is not comparable to this project's unified Pearson benchmark or EEG results.

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

## OLSA Validation

OLSA preprocessing used the official `preprocessing_olsa_decoding.m` with `use_maxfilter=true`, `audio_latency=0.003`, `bandpass=[0.5,4] Hz`, `fs_neuro=1000`, `fs_down=64`, `prestim=0`, and `poststim=0.5`.

The OLSA output has 120 neuro/audio pairs. Every OLSA trial has 306 MEG channels, every neuro/audio pair is equal-length after truncation, and `SNR`, `intelligibility`, and `playlist` each contain 120 entries. OLSA and audiobook MEG channel labels/order match exactly.

## Official Decoding Anchor

Official `training_decoding('sub-03', settings)` completed successfully.

- `n_trials = 16`
- `n_trials_train = 13`
- `n_trials_test = 3`
- selected lambda: `0.01`
- correlation metric: `Spearman`
- split policy: official `rng("shuffle")`, random 80/20 audiobook split; this run is non-deterministic.
- z-score policy: official global z-score; audiobook and OLSA normalized separately; MEG mag and grad normalized separately.

Compact result summaries are in:

- `sub03_official_decoding_anchor_summary.json`
- `sub03_official_decoding_anchor_metrics.csv`
- `sub03_official_decoding_anchor_report.md`
