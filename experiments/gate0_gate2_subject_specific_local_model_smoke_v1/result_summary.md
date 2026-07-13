# Subject-Specific Local Model Smoke v1

- protocol: `gate0_gate2_subject_specific_local_model_smoke_v1`
- dataset: `weissbart_tf64`
- subject: `P00`
- seed: `0`

| model | status | subject_metric | notes |
| --- | --- | ---: | --- |
| linear | success | 0.08150697514454137 | validation Pearson recorded as auxiliary metadata: 0.087728 |
| lasso | success | 0.10117742573038128 | validation Pearson selected alpha=0.001 |
| elasticnet | success | 0.10145778483787096 | validation Pearson selected alpha and l1_ratio |
| vlaai | success | 0.1056852678883428 | best_val_score=0.059850; max_eval_windows_per_recording=256 |
| happyquokka | success | -0.031749759892344194 | HappyQuokka smoke uses non-overlapping 10-second chunks and aggregates them back to recording-level scorer input. |

## Scope Notes
- This is a local candidate-model adapter smoke matrix, not a full benchmark.
- Accepted subject-specific models (`ridge / cca / fcnn / dnn / cnn / eegnet / adt`) were intentionally not rerun here.
- `decaf` remains external-only and was not integrated in this round.
