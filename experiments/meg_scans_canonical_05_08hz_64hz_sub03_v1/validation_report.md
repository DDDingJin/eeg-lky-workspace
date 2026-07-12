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
- Upstream post-hoc commit check: `32bfc690e28e7591b45d96615c59b2d53b6a7165`.
- Channel inventory: `306` total, `102` mag, `204` grad; labels unique.
- mag102 channel-order SHA256: `9e1c53ae8d218548df09c606ea51e6eeecd1eb3190aa8010b2c1eca49aea4756`.
- Channelwise PSD summary separates mag and grad before aggregation.
- Data provenance hashes record existing local MAT files without modifying them.
- Validation status: passed.

Invalid trials are recorded in `invalid_trials.csv`; each run has one short final trial removed by the official cleanup rule.
