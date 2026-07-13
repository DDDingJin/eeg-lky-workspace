# Model Identity Audit

- protocol: `gate0_gate2_subject_specific_local_models_weissbart_full_v1`
- dataset: `weissbart_tf64`
- seed: `0`

## VLAAI

- current implementation: `src/repro/mldecoders/models.py::VLAAIExactOfficialRegressor`
- current contract: within-subject local adaptation, `64 x 50` EEG window input, last-target scalar regression output, validation-selected checkpoint, full subject test aggregation.
- reference path: repository reference VLAAI paths such as `src/repro/vlaai_exact.py` and `scripts/run_vlaai_exact_reference.py` represent reference/exact structural runs with their own dataset construction and benchmark protocol.
- identity conclusion: this run is a local adaptation / not yet reference-protocol parity result. It should not be described as an official equivalent VLAAI reproduction.

## HappyQuokka

- current implementation: `src/repro/happyquokka_reference.py` helpers plus upstream `external/upstream/HappyQuokka_system_for_EEG_Challenge` decoder.
- current contract: within-subject local adaptation, non-overlapping `10s_chunk` windows, `g_con=false`, one local subject id, validation-selected checkpoint, and full available test chunks per recording.
- reference path: repository reference HappyQuokka paths such as `src/repro/happyquokka_reference.py`, `scripts/run_happyquokka_reference.py`, and archived `happyquokka_gcon` summaries use reference-style dataset packaging and commonly evaluate `g_con=true` subject-conditioned variants.
- seed control after this fix: HappyQuokka training receives the config seed explicitly and sets Python, NumPy, Torch, CUDA-if-available, and DataLoader shuffle generator seeds.
- identity conclusion: this run is a local adaptation / not yet reference-protocol parity result. It preserves the current 10s_chunk contract and must not be claimed as an official equivalent HappyQuokka reproduction.
