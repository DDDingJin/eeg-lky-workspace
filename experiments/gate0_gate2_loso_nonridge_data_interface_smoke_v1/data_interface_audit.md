# LOSO Non-Ridge Data Interface Audit

- protocol: `gate0_gate2_loso_nonridge_adapter_smoke_matrix_v1`
- branch: `fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1`
- base branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- base commit: `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- blocked branch context: `fix/ar-20260625-161300-a43831b-loso-all-models-single-seed-v1`
- blocked branch commit: `7b79e0b4b8b86926b1dd5e275ee014dda596eea0`
- device: `cuda`
- dataset: `weissbart_tf64`
- heldout subject: `P00`

## Shared window adapter
- train subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- train recording count: `180`
- val recording count: `180`
- train window count estimate: `1454736`
- val window count estimate: `174048`
- lazy generation: `True`
- full window materialization avoided: `True`
- hypothetical full train materialization bytes: `18626439744`

## Shared sequence adapter
- train recording count: `180`
- val recording count: `180`
- train window count estimate: `22080`
- val window count estimate: `2040`
- lazy generation: `True`
- full window materialization avoided: `True`
- hypothetical full train materialization bytes: `1837056000`

## Model-by-model note
- `eegnet`: status=`success`, input_shape=`[256, 64, 50]`, output_shape=`[990]`, target_shape=`[990]`, alignment=`aligned_window_target_last_index`
- `fcnn`: status=`success`, input_shape=`[256, 64, 50]`, output_shape=`[990]`, target_shape=`[990]`, alignment=`aligned_window_target_last_index`
- `cnn`: status=`success`, input_shape=`[256, 64, 50]`, output_shape=`[990]`, target_shape=`[990]`, alignment=`aligned_window_target_last_index`
- `adt`: status=`success`, input_shape=`[32, 320, 64]`, output_shape=`[320]`, target_shape=`[320]`, alignment=`aligned_sequence_window_prediction`
- `dnn`: status=`success`, input_shape=`[256, 64, 50]`, output_shape=`[990]`, target_shape=`[990]`, alignment=`aligned_window_target_last_index`
- `cca`: status=`skipped_with_reason`, input_shape=`[1454736, 3200]`, output_shape=`[1454736]`, target_shape=`[1454736]`, alignment=`blocked_before_fit_due_to_lag_memory_scaling`
