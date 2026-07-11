# Adapter Shape Audit

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
