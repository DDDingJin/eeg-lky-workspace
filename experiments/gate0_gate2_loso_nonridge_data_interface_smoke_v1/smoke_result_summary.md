# LOSO Non-Ridge Adapter Smoke Matrix Summary

- protocol: `gate0_gate2_loso_nonridge_adapter_smoke_matrix_v1`
- dataset: `weissbart_tf64`
- heldout subject: `P00`
- device: `cuda`

| model | status | subject_metric | note |
| --- | --- | ---: | --- |
| eegnet | success | 0.02235298908320685 | best_epoch=0; best_val_score=-0.06025776546448469 |
| fcnn | success | 0.02035225335175803 | best_epoch=0; best_val_score=-0.044400437735021114 |
| cnn | success | 0.00883514714026327 | best_epoch=0; best_val_score=-0.034001109190285206 |
| adt | success | 0.01310487321855682 | best_epoch=0; best_val_loss=0.006101077422499657; best_val_metric=-0.006101077422499657 |
| dnn | success | 0.02035225335175803 | best_epoch=0; best_val_score=-0.044400437735021114 |
| cca | skipped_with_reason |  | CCA-specific scalability issue: estimated pooled LOSO lag-matrix memory exceeds guard; train_estimated_bytes=37241241600, val_estimated_bytes=4455628800, guard=16000000000 |

- deep-model full LOSO readiness suggestion: `yes`
- This is an adapter smoke matrix, not a full benchmark run.
