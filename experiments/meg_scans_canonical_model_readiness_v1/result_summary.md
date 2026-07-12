# MEG-SCANS Canonical Model Readiness v1

Scope: native-MEG sub-03 audiobook modeling readiness only; not a paper benchmark and not cross-modal transfer.

## Model status
- `ridge_mag102`: status=`actual`, metric=`0.15330705320745663`, delta=`0.183547911110158`
- `ridge_all306`: status=`actual`, metric=`0.16978736802255778`, delta=`0.17837577719943892`
- `cca_mag102`: status=`skipped`, metric=`None`, delta=`None`
  reason: skipped_existing_interface_unavailable: accepted CCA path src/repro/mldecoders/cca.py::fit_cca_reconstruction accepts concatenated x/y and internally constructs lags, which would cross canonical recording boundaries; this round forbids cross-recording lag construction and forbids silently replacing the CCA definition.
- `eegnet_mag102_smoke`: status=`smoke_only`, metric=`0.11356901800351529`, delta=`None`

## Sorted vs mismatched
Fixed derangement: 4 -> 8 -> 12 -> 16 -> 4.
