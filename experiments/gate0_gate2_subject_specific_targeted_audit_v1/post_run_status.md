# Subject-Specific Targeted Audit Post-Run Status

Status: manual targeted diagnostics completed and closed without rerun by this post-run closure.

## Scope
- Branch artifact scope: `subject_specific_targeted_audit`.
- Output directory: `experiments/gate0_gate2_subject_specific_targeted_audit_v1`.
- Published Weissbart/Etard two-dataset result directories were not rewritten by this closure.
- No checkpoint, raw data, prediction dump, model weights, cache, or large binary artifact is part of this closure.

## Completed Manual Diagnostics
- `weissbart_tf64 / P06 / vlaai / seed0`: two repeatability runs recorded in `vlaai_p06_repeatability.csv`.
- `etard_tf64 / P11 / elasticnet / seed0`: train-only StandardScaler diagnostic recorded in `elasticnet_p11_standardized_summary.json` and `elasticnet_p11_standardized_recording_metrics.csv`.
- `etard_tf64 / P11 / happyquokka / seed0`: read-only audit remains available in `hq_p11_audit.md` and `hq_p11_audit.csv`.

## VLAAI P06 Repeatability
- run 1: best_epoch=`2`, epochs_completed=`12`, best_val_score=`0.11765319084127744`, test_metric=`0.02166360125528517`, independent_pearson_metric=`0.02166360125528517`.
- run 2: best_epoch=`2`, epochs_completed=`12`, best_val_score=`0.09747411018858353`, test_metric=`0.01565069587715324`, independent_pearson_metric=`0.01565069587715324`.
- first_test_input_hash was identical across repeats: `f761f700196e2e2cb35dc101ec3bcbedc5c2db28716a4e5967c5a83815d63289`.
- prediction_hash differed across repeats, so this records observed current-path repeatability rather than enforcing new deterministic CUDA behavior.

## ElasticNet P11 Standardized Diagnostic
- selected_alpha=`0.1`.
- selected_l1_ratio=`0.2`.
- best_validation_score=`0.022883231752525882`.
- final_test_metric=`-0.0021333661870536714`.
- convergence_warning_count=`3`.
- standardizer_fit_scope=`P11 train lag matrix only`.
- published_result_replaced=`false`.
- recording_rows=`16`.
