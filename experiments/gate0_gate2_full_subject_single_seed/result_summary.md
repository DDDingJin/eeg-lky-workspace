# Full-Subject Single-Seed v1

This directory contains the first article-grade full-subject single-seed run under the unified runner, schema, and scorer.
It is not a multi-seed benchmark, not a cross-dataset transfer study, and not a final paper conclusion.

## Protocol
- `protocol`: `gate0_gate2_full_subject_single_seed_v1`
- `device`: `cuda`
- `seed`: `0`
- `models`: `ridge, cca, fcnn, adt`

## Subject inventory
- `etard_tf64` (20 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- `weissbart_tf64` (13 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`

## Job accounting
- expected jobs: `132`
- cumulative successful jobs available: `132`
- skipped existing jobs: `132`
- protocol mismatch jobs: `0`
- actually run jobs: `0`
- failed jobs: `0`

## Dataset-level Pearson summary
| dataset | model | n_subjects | mean_pearson | std_pearson | median_pearson | min_pearson | max_pearson | fisher_z_mean | backtransformed_mean_r |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| etard_tf64 | adt | 20 | 0.101376 | 0.043114 | 0.097827 | 0.029016 | 0.186245 | 0.101936 | 0.101585 |
| etard_tf64 | cca | 20 | 0.069850 | 0.022950 | 0.067010 | 0.016975 | 0.105258 | 0.070000 | 0.069886 |
| etard_tf64 | fcnn | 20 | 0.062121 | 0.020510 | 0.058597 | 0.024304 | 0.107262 | 0.062228 | 0.062148 |
| etard_tf64 | ridge | 20 | 0.103414 | 0.039370 | 0.096879 | 0.044848 | 0.175744 | 0.103956 | 0.103583 |
| weissbart_tf64 | adt | 13 | 0.135267 | 0.043033 | 0.118415 | 0.069682 | 0.206214 | 0.136372 | 0.135533 |
| weissbart_tf64 | cca | 13 | 0.089385 | 0.024977 | 0.092182 | 0.043354 | 0.136374 | 0.089679 | 0.089440 |
| weissbart_tf64 | fcnn | 13 | 0.091861 | 0.039136 | 0.093017 | 0.002300 | 0.177043 | 0.092263 | 0.092002 |
| weissbart_tf64 | ridge | 13 | 0.133181 | 0.029443 | 0.141483 | 0.066734 | 0.173728 | 0.134087 | 0.133289 |

## Etard condition summary
| model | condition | n_subjects | n_recordings | mean_pearson | median_pearson | std_pearson | negative_recording_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adt | clean | 18 | 72 | 0.121771 | 0.116841 | 0.130719 | 13 |
| adt | fM | 18 | 72 | 0.144164 | 0.130522 | 0.123347 | 7 |
| adt | fW | 18 | 72 | 0.105059 | 0.114579 | 0.116590 | 14 |
| adt | hb | 18 | 72 | 0.144878 | 0.134035 | 0.109732 | 8 |
| adt | lb | 18 | 72 | 0.114962 | 0.108181 | 0.087187 | 4 |
| adt | mb | 18 | 72 | 0.116744 | 0.113284 | 0.108104 | 8 |
| cca | clean | 18 | 72 | 0.112550 | 0.099920 | 0.107746 | 11 |
| cca | fM | 18 | 72 | 0.116915 | 0.121607 | 0.125626 | 13 |
| cca | fW | 18 | 72 | 0.107228 | 0.112664 | 0.096027 | 11 |
| cca | hb | 18 | 72 | 0.089475 | 0.084541 | 0.099057 | 12 |
| cca | lb | 18 | 72 | 0.078403 | 0.089112 | 0.073408 | 11 |
| cca | mb | 18 | 72 | 0.058018 | 0.056650 | 0.101341 | 23 |
| fcnn | clean | 18 | 72 | 0.105754 | 0.106111 | 0.101863 | 10 |
| fcnn | fM | 18 | 72 | 0.105285 | 0.104276 | 0.092423 | 10 |
| fcnn | fW | 18 | 72 | 0.075866 | 0.084068 | 0.079341 | 12 |
| fcnn | hb | 18 | 72 | 0.068544 | 0.067191 | 0.092899 | 13 |
| fcnn | lb | 18 | 72 | 0.066322 | 0.069550 | 0.080437 | 16 |
| fcnn | mb | 18 | 72 | 0.043311 | 0.046501 | 0.074353 | 23 |
| ridge | clean | 18 | 72 | 0.171230 | 0.176688 | 0.106735 | 4 |
| ridge | fM | 18 | 72 | 0.160014 | 0.155412 | 0.114767 | 6 |
| ridge | fW | 18 | 72 | 0.136593 | 0.143037 | 0.080029 | 3 |
| ridge | hb | 18 | 72 | 0.120753 | 0.114647 | 0.083519 | 4 |
| ridge | lb | 18 | 72 | 0.117044 | 0.114700 | 0.069114 | 2 |
| ridge | mb | 18 | 72 | 0.083247 | 0.080322 | 0.084034 | 12 |

## Scope notes
- Historical `experiments/summary_figures/*` were not rerun and remain reference-only.
- This run keeps a single unified scorer and aggregation rule across all four models.
- Per-job outputs are stored incrementally under `experiments/gate0_gate2_full_subject_single_seed/jobs/` to enable resume and protocol-mismatch detection without overwriting prior evidence.
- Etard condition-level `n_subjects` is condition-specific rather than always 20 because the processed split itself is not perfectly balanced across condition families for every participant. The full dataset-level subject summary still covers all 20 Etard subjects.
