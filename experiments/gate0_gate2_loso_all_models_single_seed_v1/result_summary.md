# Gate0-Gate2 LOSO All-Models Single-Seed v1

This directory extends the validated LOSO interface from ridge to the current unified subject-specific benchmark model set.
It is a subject-independent LOSO single-seed round under the same scorer/schema style and does not add raw dumps or final paper prose.

## Protocol
- `protocol`: `gate0_gate2_loso_all_models_single_seed_v1`
- `seed`: `0`
- `models_requested`: `ridge, cca, fcnn, dnn, cnn, eegnet, adt`

## Subject inventory
- `weissbart_tf64` (13 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- `etard_tf64` (20 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`

## Dataset metrics
| dataset | model | n_subjects | mean_pearson | std_pearson | min_pearson | max_pearson |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| etard_tf64 | ridge | 20 | 0.062338 | 0.035003 | -0.010371 | 0.132972 |
| weissbart_tf64 | ridge | 13 | 0.065466 | 0.034009 | 0.014096 | 0.117471 |

## LOSO vs subject-specific dataset summary
| dataset | model | family | loso_mean_pearson | subject_specific_mean_pearson | delta_loso_minus_subject_specific |
| --- | --- | --- | ---: | ---: | ---: |
| etard_tf64 | ridge | linear_ridge | 0.062338 | 0.103414 | -0.041076 |
| weissbart_tf64 | ridge | linear_ridge | 0.065466 | 0.133181 | -0.067715 |

## Etard condition metrics
| model | condition | n_subjects | n_recordings | mean_pearson | std_pearson | negative_recording_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ridge | clean | 18 | 72 | 0.115288 | 0.109829 | 11 |
| ridge | fM | 18 | 72 | 0.111881 | 0.103854 | 11 |
| ridge | fW | 18 | 72 | 0.078365 | 0.090376 | 12 |
| ridge | hb | 18 | 72 | 0.067879 | 0.097260 | 15 |
| ridge | lb | 18 | 72 | 0.079901 | 0.063636 | 7 |
| ridge | mb | 18 | 72 | 0.040476 | 0.098838 | 26 |

## Failures and skips
- `etard_tf64` / `adt`: `20` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO ADT path would require subject-wise repeated pooled sequence training on the largest dataset under the existing exact sequence-window pipeline; this round does not fabricate results without a dedicated scalable LOSO training path.`
- `etard_tf64` / `cca`: `20` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO CCA path materializes dataset-scale multi-subject lagged train/val arrays and did not complete a single Weissbart held-out smoke within the allotted runtime after aggregate bug fix; this branch records the implementation identity and runtime blockage without fabricating results.`
- `etard_tf64` / `cnn`: `20` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO upstream CNN path shares the same exact pooled sliding-window expansion bottleneck as the other windowed deep models; this round records the implementation identity but does not claim a successful scalable LOSO result.`
- `etard_tf64` / `dnn`: `20` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO upstream FCNN path shares the same exact pooled sliding-window expansion bottleneck as FCNN; this round records the implementation identity but does not claim a successful scalable LOSO result.`
- `etard_tf64` / `eegnet`: `20` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO EEGNet path shares the same exact pooled sliding-window expansion bottleneck as the other windowed deep models; this round records the implementation identity but does not claim a successful scalable LOSO result.`
- `etard_tf64` / `fcnn`: `20` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO FCNN path would train over millions of pooled sliding windows per held-out subject under the existing exact window-expansion dataset implementation; not completed in this round without first introducing a separate scalable LOSO data pipeline.`
- `weissbart_tf64` / `adt`: `13` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO ADT path would require subject-wise repeated pooled sequence training on the largest dataset under the existing exact sequence-window pipeline; this round does not fabricate results without a dedicated scalable LOSO training path.`
- `weissbart_tf64` / `cca`: `13` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO CCA path materializes dataset-scale multi-subject lagged train/val arrays and did not complete a single Weissbart held-out smoke within the allotted runtime after aggregate bug fix; this branch records the implementation identity and runtime blockage without fabricating results.`
- `weissbart_tf64` / `cnn`: `13` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO upstream CNN path shares the same exact pooled sliding-window expansion bottleneck as the other windowed deep models; this round records the implementation identity but does not claim a successful scalable LOSO result.`
- `weissbart_tf64` / `dnn`: `13` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO upstream FCNN path shares the same exact pooled sliding-window expansion bottleneck as FCNN; this round records the implementation identity but does not claim a successful scalable LOSO result.`
- `weissbart_tf64` / `eegnet`: `13` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO EEGNet path shares the same exact pooled sliding-window expansion bottleneck as the other windowed deep models; this round records the implementation identity but does not claim a successful scalable LOSO result.`
- `weissbart_tf64` / `fcnn`: `13` jobs, statuses=`skipped_with_reason`, example_subjects=`P00, P01, P02, P03, P04`, reason=`Current pooled LOSO FCNN path would train over millions of pooled sliding windows per held-out subject under the existing exact window-expansion dataset implementation; not completed in this round without first introducing a separate scalable LOSO data pipeline.`

Note: `dnn` and `fcnn` remain two implementations within the same `MLP/FCNN family` and are not described as separate architecture families in this report.
