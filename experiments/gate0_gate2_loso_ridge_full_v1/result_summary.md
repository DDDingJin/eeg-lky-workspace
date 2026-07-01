# Gate0-Gate2 LOSO Ridge Full v1

This directory extends the validated LOSO ridge interface to all subjects on `weissbart_tf64` and `etard_tf64` with seed `0` only.
It is a pure subject-independent LOSO ridge round and does not add deep models or paper prose.

## Protocol
- `protocol`: `gate0_gate2_loso_ridge_full_v1`
- `model`: `ridge`
- `seed`: `0`

## Subject inventory
- `weissbart_tf64` (13 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- `etard_tf64` (20 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`

## Dataset metrics
| dataset | model | n_subjects | mean_pearson | std_pearson | median_pearson | min_pearson | max_pearson |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| etard_tf64 | ridge | 20 | 0.062338 | 0.035003 | 0.058201 | -0.010371 | 0.132972 |
| weissbart_tf64 | ridge | 13 | 0.065466 | 0.034009 | 0.059734 | 0.014096 | 0.117471 |

## Etard condition metrics
| model | condition | n_subjects | n_recordings | mean_pearson | median_pearson | std_pearson | negative_recording_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | clean | 18 | 72 | 0.115288 | 0.123116 | 0.109829 | 11 |
| ridge | fM | 18 | 72 | 0.111881 | 0.101084 | 0.103854 | 11 |
| ridge | fW | 18 | 72 | 0.078365 | 0.087416 | 0.090376 | 12 |
| ridge | hb | 18 | 72 | 0.067879 | 0.058880 | 0.097260 | 15 |
| ridge | lb | 18 | 72 | 0.079901 | 0.082615 | 0.063636 | 7 |
| ridge | mb | 18 | 72 | 0.040476 | 0.047782 | 0.098838 | 26 |

## LOSO vs subject-specific ridge
- compared rows: `33`
- mean delta (LOSO - subject-specific): `-0.051570`
- min delta: `-0.118886`
- max delta: `0.010144`
