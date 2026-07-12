# MEG-SCANS canonical 0.5-8 Hz / 64 Hz sub-03 validation

- Local envelope MAT: `E:\decode\data\derived\meg_scans_canonical_05_08hz_64hz_v1\stimuli\sub-others_preprocessed_audiobook_envelopes_decoding.mat`
- Local paired MAT: `E:\decode\data\derived\meg_scans_canonical_05_08hz_64hz_v1\sub-03\speech\sub-03_preprocessed_audiobooks_decoding.mat`
- Valid paired trials: `16`
- Samples per trial: `7680`
- MEG channels: `306`
- MEG band: `0.5-8 Hz`
- Envelope band: `0.5-8 Hz`
- Final fs: `64 Hz`
- Audio latency correction: `3 ms`
- Normalization: none in preprocessing; future runner must fit scaler on training split only.
- Official 0.5-4 Hz anchor files were checked and not overwritten by this wrapper.
- Validation status: passed.

Invalid trials are recorded in `invalid_trials.csv`; each run has one short final trial removed by the official cleanup rule.
