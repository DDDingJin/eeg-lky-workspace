# Leakage Summary

- `protocol`: `gate0_gate2_loso_all_models_single_seed_v1`
- `seed`: `0`
- `models_requested`: `ridge, cca, fcnn, dnn, cnn, eegnet, adt`

Model identity and training/selection protocol:

## `ridge`
- family: `linear_ridge`
- implementation: `scripts/run_gate0_gate2_loso_ridge_full_v1.py::fit_loso_ridge_for_dataset_subject`
- selection rule: `reuse verified loso-ridge-full-v1 results under identical split/scorer/schema`
- training protocol: `no rerun; import subject and recording metrics from loso-ridge-full-v1 and relabel to current protocol`

## `cca`
- family: `cca_reconstruction`
- implementation: `src/repro/mldecoders/cca.py::fit_cca_reconstruction`
- selection rule: `choose pooled non-heldout validation reconstruction correlation over PCA/component/alpha grid`
- training protocol: `pooled non-heldout train split for fit; pooled non-heldout val split for model selection; heldout test only for evaluation`

## `fcnn`
- family: `mlp_fcnn_family`
- implementation: `src/repro/simple_models.py::FCNNBaseline`
- selection rule: `choose checkpoint by pooled non-heldout validation Pearson`
- training protocol: `pooled non-heldout train windows; pooled non-heldout val windows; heldout test windows only for evaluation`

## `dnn`
- family: `mlp_fcnn_family`
- implementation: `external/upstream/mldecoders/pipeline/dnn.py::FCNN`
- selection rule: `choose checkpoint by pooled non-heldout validation Pearson`
- training protocol: `pooled non-heldout train windows; pooled non-heldout val windows; heldout test windows only for evaluation`

## `cnn`
- family: `cnn_family`
- implementation: `external/upstream/mldecoders/pipeline/dnn.py::CNN`
- selection rule: `choose checkpoint by pooled non-heldout validation Pearson`
- training protocol: `pooled non-heldout train windows; pooled non-heldout val windows; heldout test windows only for evaluation`

## `eegnet`
- family: `eegnet_family`
- implementation: `src/repro/mldecoders/models.py::EEGNetRegressor`
- selection rule: `choose checkpoint by pooled non-heldout validation Pearson`
- training protocol: `pooled non-heldout train windows; pooled non-heldout val windows; heldout test windows only for evaluation`

## `adt`
- family: `transformer_adt_family`
- implementation: `src/repro/adt_exact.py::ADTExactRegressor`
- selection rule: `choose checkpoint by pooled non-heldout validation loss; report pooled non-heldout validation Pearson alongside loss`
- training protocol: `pooled non-heldout train sequence windows; pooled non-heldout val sequence windows; heldout test windows only for evaluation`

Pure LOSO guarantees for successful jobs:
- held-out target subject excluded from train
- held-out target subject excluded from val
- held-out target subject excluded from scaler / normalization / alpha or checkpoint selection
- held-out target subject used only for test evaluation

- `weissbart_tf64` / `ridge` / `P00`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P00`
- `weissbart_tf64` / `ridge` / `P01`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P01`
- `weissbart_tf64` / `ridge` / `P02`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P02`
- `weissbart_tf64` / `ridge` / `P03`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P03`
- `weissbart_tf64` / `ridge` / `P04`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P04`
- `weissbart_tf64` / `ridge` / `P05`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P05`
- `weissbart_tf64` / `ridge` / `P06`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P06`
- `weissbart_tf64` / `ridge` / `P07`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P07`
- `weissbart_tf64` / `ridge` / `P08`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P08`
- `weissbart_tf64` / `ridge` / `P09`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P09`
- `weissbart_tf64` / `ridge` / `P10`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P10`
- `weissbart_tf64` / `ridge` / `P11`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P11`
- `weissbart_tf64` / `ridge` / `P12`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P12`
- `etard_tf64` / `ridge` / `P00`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P00`
- `etard_tf64` / `ridge` / `P01`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P01`
- `etard_tf64` / `ridge` / `P02`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P02`
- `etard_tf64` / `ridge` / `P03`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P03`
- `etard_tf64` / `ridge` / `P04`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P04`
- `etard_tf64` / `ridge` / `P05`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P05`
- `etard_tf64` / `ridge` / `P06`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P06`
- `etard_tf64` / `ridge` / `P07`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P07`
- `etard_tf64` / `ridge` / `P08`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P08`
- `etard_tf64` / `ridge` / `P09`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P09`
- `etard_tf64` / `ridge` / `P10`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P10`
- `etard_tf64` / `ridge` / `P11`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P11`
- `etard_tf64` / `ridge` / `P12`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P12`
- `etard_tf64` / `ridge` / `P13`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P13`
- `etard_tf64` / `ridge` / `P14`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P14`
- `etard_tf64` / `ridge` / `P15`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P15`
- `etard_tf64` / `ridge` / `P16`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P16`
- `etard_tf64` / `ridge` / `P17`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P17`
- `etard_tf64` / `ridge` / `P18`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P18`
- `etard_tf64` / `ridge` / `P19`: train_exclude=`True`, val_exclude=`True`, selection_exclude=`True`, test_only=`P19`
