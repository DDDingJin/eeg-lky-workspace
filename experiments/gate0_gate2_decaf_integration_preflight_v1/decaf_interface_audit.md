# DECAF Interface Audit

- model family: `HappyQuoka`
- implementation: `external/upstream/DECAF/src/models/happyquoka.py::HappyQuoka`
- native HappyQuoka input: `(batch, time, channels)` EEG tensor.
- native HappyQuoka output: dict with `envelope` shaped `(batch, time, 1)` and `dec_output` shaped `(batch, time, d_model)`.
- two-branch DECAF variants additionally require envelope context and fusion logic; this preflight does not claim reference-protocol parity for those variants.
- adapter stance: align output/target/scorer after auditing native context/input/output; do not force existing 50-sample windows.
- status: `DECAF local adaptation / not yet reference-protocol parity`

- recording_id: `test_-_P00_-_AUNP01`
- input_shape: `[1, 640, 64]`
- raw_output_type: `dict`
- raw_output_keys: `['dec_output', 'envelope']`
- raw_envelope_shape: `[1, 640, 1]`
- postprocessed_prediction_shape: `[640]`
- target_shape: `[640]`
- scorer_input_shape: `[640]`
- scorer_smoke_metric_name: `pearson_r`
- native_contract: `HappyQuoka: EEG tensor (batch,time,channels) -> dict with envelope (batch,time,1)`
- adapter_note: `Shape/scorer smoke uses an untrained model only; no scientific metric is reported.`
