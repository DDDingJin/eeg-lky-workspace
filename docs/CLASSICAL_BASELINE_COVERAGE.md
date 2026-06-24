# Classical Baseline Coverage

Last checked: 2026-06-12

This file maps paper-style baseline names to local code names.

## Already implemented in local code

### Backward / envelope reconstruction family

- `ridge`
  - standard backward linear decoder
- `avgdec_ridge`
  - MMSE-Avgdec-Ridge
- `avgdec_lasso`
  - MMSE-Avgdec-LASSO
  - code exists
  - results not complete
- `avgcorr_ridge`
  - MMSE-Avgcorr-Ridge
- `avgcorr_lasso`
  - MMSE-Avgcorr-LASSO
  - code exists
  - results not complete

### CCA family

- `cca_canonical`
  - selects by validation canonical correlation
  - canonical-correlation style baseline
- `cca_recon`
  - CCA plus ridge readout for envelope reconstruction
- `cca_match_mismatch`
  - CCA model scored as match/mismatch accuracy

### Forward / encoding family

- `forward_ridge`
  - supervised forward encoding baseline
  - predicts EEG channels from lagged envelope
  - metric is mean channel correlation

## High-priority baseline additions

These are the baseline families that should be added next because they matter for interpretation, not just for table size.

### TRF / eTRF family

- `backward_trf_ridge`
  - lagged envelope reconstruction
  - should be reported as the main linear reconstruction baseline when explicit lag structure is used
- `forward_trf_ridge`
  - explicit lagged forward encoding model
- `etrf`
  - if implemented through an established package or faithful recipe, should be reported as the package-style TRF baseline

Important:

- plain `ridge` and lagged `backward_trf_ridge` should not be silently merged
- the lag definition is part of the method

### Sparse / regularized linear family

- `lasso`
  - direct sparse backward decoder
- `elastic_net`
  - compromise between ridge and lasso

### Expanded CCA family

- `cca_forward_backward`
  - paired forward/backward CCA-style baseline when the protocol supports it
- `cca_multi_lag`
  - explicit multi-lag CCA reconstruction setup
- `cca_match_mismatch`
  - already present conceptually, but should be kept as a first-class task row in benchmark reporting

## Reporting rule for the classical family

The classical baselines should not be thrown into a single undifferentiated chart.

They should be grouped at least into:

- backward reconstruction
- forward encoding
- canonical-correlation style
- decision-style scoring

This is required if the benchmark is going to support strong claims about where deep models actually add value.

## Important comparability note

Not all metrics are directly comparable:

- `ridge`, `avgdec_*`, `avgcorr_*`, `cca_recon`
  - envelope reconstruction correlation
- `cca_match_mismatch`
  - classification-style accuracy
- `cca_canonical`
  - canonical correlation
- `forward_ridge`
  - mean EEG-channel correlation

So these should be grouped by task/metric, not all thrown into one bar chart.

## Main code files

- `src/repro/mldecoders/linear_baselines.py`
- `src/repro/mldecoders/cca.py`
- `scripts/run_linear_variant_baselines.py`
- `scripts/run_classical_baselines.py`
