# EEG/MEG Protocol Compatibility

## Compatible

- Sampling target can be aligned at 64 Hz.
- Recording-level unit can be represented as `recording_id` rows and scored with `pearson_on_valid`.
- R1/R2 can preserve time x channels arrays and existing target alignment conventions.

## Not directly compatible

- EEG accepted model paths assume `channels=range(64)` in the runner.
- MEG native has 306 channels; magnetometer-only has 102 channels.
- Existing EEG HDF5 metadata does not provide confirmed EEG64 sensor coordinates.
- Existing accepted runner consumes preprocessed `*_envelope.npy`; it does not lock an envelope extraction implementation.

## Decision

For the next smoke, train from scratch on MEG representation outputs. Do not transfer EEG-trained weights directly.
