# DECAF TwoBranch Interface Audit

- source: `external/upstream/DECAF/src/models/two_branch_model.py::TwoBranchModel`
- pinned upstream commit: `f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332`
- status: `DECAF local adaptation / not yet reference-protocol parity`
- true forward signature: `forward(eeg, envelope_context)`.
- EEG input interval: target/prediction interval, length `224` samples.
- envelope_context input interval: immediately preceding context interval, forward input length `128` samples.
- envelope forecaster expected context after trim: `96` samples.
- forward trims first `32` context samples via `envelope_context = envelope_context[:,32:,:]` before forecasting.
- prediction output interval: same duration as EEG input / target interval.
- target interval: same sample interval as model output; used only for evaluation, not for synthetic shape-check context.
- fusion strategy: combines envelope-branch and EEG-branch predictions; `weighted` uses softmax-normalized learned two-element weights.
- upstream trainer/data path indicates context models can receive ground-truth past envelope context; this changes task class relative to pure EEG reconstruction.

- shape_check_context: `synthetic_zero_context_only_no_target_envelope_used`
- eeg_input_shape: `[1, 224, 64]`
- envelope_context_input_shape: `[1, 128, 1]`
- forward_context_trim_samples: `32`
- forecaster_expected_context_shape: `[1, 96, 1]`
- effective_envelope_context_shape_after_forward_trim: `[1, 96, 1]`
- output_keys: `['eeg_branch', 'envelope', 'envelope_branch', 'fusion_weights']`
- envelope_shape: `[1, 224, 1]`
- envelope_branch_shape: `[1, 224, 1]`
- eeg_branch_shape: `[1, 224, 1]`
- fusion_weights_shape: `[2]`
- scientific_metric_computed: `False`
