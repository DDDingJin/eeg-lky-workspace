# Envelope Reconstruction Protocol

## Contents

1. Target definition
2. Temporal alignment
3. Windowing and aggregation
4. Prediction contract
5. Metrics
6. Baselines and sanity checks
7. Cross-dataset and cross-modal alignment

## Target Definition

Record for every dataset and protocol:

1. source audio or signal;
2. envelope extraction method;
3. rectification, Hilbert, filterbank, or power-law steps;
4. filter type and passband;
5. resampling method and target rate;
6. anti-aliasing behavior;
7. causal or acausal processing;
8. normalization method and fitting scope;
9. missing and boundary sample handling;
10. target dtype, scale, and shape;
11. target implementation version.

Use one target definition for directly compared models. If datasets require different definitions, make the difference explicit and limit claims accordingly.

## Temporal Alignment

1. Define neural-to-envelope lag and sign convention.
2. Define whether lag is fixed, searched, or model-internal.
3. Select lag using training or validation data only.
4. Keep target-index and crop rules explicit.
5. Record filter delay and resampling delay.
6. Check that prediction and target describe the same physical time interval.
7. Validate one synthetic or easily inspected alignment case.
8. Treat a changed lag range, target index, crop, or delay rule as a protocol change requiring a new smoke.

## Windowing And Aggregation

1. Split recordings, trials, or subjects before creating windows.
2. Record window length, hop length, padding, and final-window behavior.
3. Preserve raw recording, trial, and window identifiers.
4. Prevent overlapping source samples from crossing train, validation, or test.
5. Define overlap aggregation for continuous reconstruction.
6. Define whether overlapping predictions are averaged, weighted, selected, or discarded.
7. Reconstruct continuous recording predictions before the primary recording-level metric when appropriate.
8. Keep window-level metrics as diagnostics unless the article claim is explicitly window-level.
9. Do not concatenate unrelated recordings before computing a correlation.
10. Declare how short recordings and incomplete windows are handled.

## Prediction Contract

Store a contract per model family with:

```text
input tensor shape
raw model output shape
postprocessed prediction shape
target shape
scorer input shape
alignment rule
overlap aggregation rule
mask and padding rule
variable-length behavior
```

Require:

1. the same postprocess and scorer path across subject-specific, LOSO, transfer, and evaluation modes where possible;
2. batched inference to match an accepted single-recording path;
3. explicit squeezing or channel selection for outputs such as `(batch, time, 1)`;
4. mask-aware scoring for padded samples;
5. checkpoint-loaded predictions to match pre-save predictions within tolerance;
6. deterministic output shape tests.

Use `scripts/validate_prediction_contract.py` to validate the contract artifact. Add model-specific numerical checks in the project test suite.

## Metrics

Candidate reconstruction metrics include:

- Pearson correlation;
- Spearman correlation;
- mean squared error;
- mean absolute error;
- coefficient of determination;
- task-specific spectral or perceptual measures when scientifically justified.

Rules:

1. Select one primary metric before final testing.
2. Keep scale-invariant and scale-sensitive metrics distinct.
3. Define all normalization used before a scale-sensitive metric.
4. Compute the primary metric at the declared recording, trial, or subject level.
5. Store the observation count for every metric row.
6. Define behavior for constant predictions, NaN, Inf, and empty masks.
7. Do not silently replace undefined correlations with zero.
8. Preserve per-recording values before subject aggregation.
9. Distinguish macro, micro, and weighted aggregation.
10. Use the same metric implementation for all comparable models.

## Baselines And Sanity Checks

Include as appropriate:

1. mean or zero predictor;
2. reliable linear Ridge or temporal response function baseline;
3. shuffled target or permutation null;
4. time-shifted null outside the plausible response range;
5. identity or oracle-style pipeline checks that test alignment but are not scientific baselines;
6. data reliability or noise-ceiling analysis when repeated measurements allow it.

Require a model to outperform meaningful null behavior before interpreting small positive correlations as reconstruction.

## Cross-Dataset And Cross-Modal Alignment

Record:

1. input modality;
2. channel or sensor mapping;
3. coordinate or region mapping when used;
4. sampling-rate alignment;
5. envelope definition alignment;
6. preprocessing alignment;
7. label and metadata alignment;
8. model input dimensionality adaptation;
9. target-domain data used during adaptation;
10. incompatible features or claims.

Report same-modality and cross-modality transfer separately. Do not attribute all performance loss to model generalization when target construction or sensor geometry also changed.
