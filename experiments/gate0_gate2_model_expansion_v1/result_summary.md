# Gate0-Gate2 Model Expansion v1

This directory extends the full-subject multi-seed benchmark with additional baseline models under the same split, scorer, and aggregation schema.
It is structured execution evidence and writing material rather than final paper prose.

## Protocol
- `protocol`: `gate0_gate2_model_expansion_v1`
- `device`: `cuda`
- `seeds`: `0, 42, 2026`
- reused base models: `ridge, cca, fcnn, adt`
- requested expansion models: `dnn, cnn, eegnet`
- effective models in this run: `ridge, cca, fcnn, adt, dnn, cnn, eegnet`

## Job accounting
- total jobs across effective models: `693`
- successful jobs available: `693`
- failed jobs: `0`
- skipped/reused jobs: `693`

## Subject inventory
- `etard_tf64` (20 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- `weissbart_tf64` (13 subjects): `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`

## Dataset metrics by seed
| dataset | model | seed | n_subjects | mean_pearson | std_pearson | min_pearson | max_pearson |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| etard_tf64 | adt | 0 | 20 | 0.101376 | 0.043114 | 0.029016 | 0.186245 |
| etard_tf64 | adt | 42 | 20 | 0.103153 | 0.039205 | 0.041127 | 0.208813 |
| etard_tf64 | adt | 2026 | 20 | 0.098140 | 0.040659 | 0.037819 | 0.192668 |
| etard_tf64 | cca | 0 | 20 | 0.069850 | 0.022950 | 0.016975 | 0.105258 |
| etard_tf64 | cca | 42 | 20 | 0.069850 | 0.022950 | 0.016975 | 0.105258 |
| etard_tf64 | cca | 2026 | 20 | 0.069850 | 0.022950 | 0.016975 | 0.105258 |
| etard_tf64 | cnn | 0 | 20 | 0.090604 | 0.040121 | 0.009873 | 0.149420 |
| etard_tf64 | cnn | 42 | 20 | 0.095340 | 0.039974 | -0.000823 | 0.167275 |
| etard_tf64 | cnn | 2026 | 20 | 0.095174 | 0.035960 | 0.013316 | 0.156403 |
| etard_tf64 | dnn | 0 | 20 | 0.062110 | 0.020645 | 0.024948 | 0.107263 |
| etard_tf64 | dnn | 42 | 20 | 0.056367 | 0.027220 | -0.000384 | 0.107688 |
| etard_tf64 | dnn | 2026 | 20 | 0.055562 | 0.029842 | -0.019553 | 0.109604 |
| etard_tf64 | eegnet | 0 | 20 | 0.102895 | 0.041844 | 0.009582 | 0.179666 |
| etard_tf64 | eegnet | 42 | 20 | 0.104746 | 0.037603 | 0.030859 | 0.169677 |
| etard_tf64 | eegnet | 2026 | 20 | 0.105266 | 0.041197 | 0.017260 | 0.185996 |
| etard_tf64 | fcnn | 0 | 20 | 0.062121 | 0.020510 | 0.024304 | 0.107262 |
| etard_tf64 | fcnn | 42 | 20 | 0.054506 | 0.030075 | -0.001780 | 0.107630 |
| etard_tf64 | fcnn | 2026 | 20 | 0.055899 | 0.030033 | -0.021596 | 0.109604 |
| etard_tf64 | ridge | 0 | 20 | 0.103414 | 0.039370 | 0.044848 | 0.175744 |
| etard_tf64 | ridge | 42 | 20 | 0.103414 | 0.039370 | 0.044848 | 0.175744 |
| etard_tf64 | ridge | 2026 | 20 | 0.103414 | 0.039367 | 0.044848 | 0.175744 |
| weissbart_tf64 | adt | 0 | 13 | 0.135267 | 0.043033 | 0.069682 | 0.206214 |
| weissbart_tf64 | adt | 42 | 13 | 0.132315 | 0.043026 | 0.073702 | 0.204706 |
| weissbart_tf64 | adt | 2026 | 13 | 0.126521 | 0.044353 | 0.042914 | 0.206588 |
| weissbart_tf64 | cca | 0 | 13 | 0.089385 | 0.024977 | 0.043354 | 0.136374 |
| weissbart_tf64 | cca | 42 | 13 | 0.089385 | 0.024977 | 0.043354 | 0.136374 |
| weissbart_tf64 | cca | 2026 | 13 | 0.089385 | 0.024977 | 0.043354 | 0.136374 |
| weissbart_tf64 | cnn | 0 | 13 | 0.127511 | 0.033446 | 0.065869 | 0.200751 |
| weissbart_tf64 | cnn | 42 | 13 | 0.115930 | 0.038696 | 0.041636 | 0.199850 |
| weissbart_tf64 | cnn | 2026 | 13 | 0.130527 | 0.032096 | 0.073003 | 0.202475 |
| weissbart_tf64 | dnn | 0 | 13 | 0.091771 | 0.039228 | 0.002032 | 0.177043 |
| weissbart_tf64 | dnn | 42 | 13 | 0.093033 | 0.040671 | 0.002758 | 0.179109 |
| weissbart_tf64 | dnn | 2026 | 13 | 0.096246 | 0.035402 | 0.015697 | 0.155357 |
| weissbart_tf64 | eegnet | 0 | 13 | 0.145987 | 0.033854 | 0.077860 | 0.210597 |
| weissbart_tf64 | eegnet | 42 | 13 | 0.139259 | 0.035288 | 0.072120 | 0.214084 |
| weissbart_tf64 | eegnet | 2026 | 13 | 0.141412 | 0.033075 | 0.074166 | 0.200795 |
| weissbart_tf64 | fcnn | 0 | 13 | 0.091861 | 0.039136 | 0.002300 | 0.177043 |
| weissbart_tf64 | fcnn | 42 | 13 | 0.090692 | 0.041787 | 0.002611 | 0.179108 |
| weissbart_tf64 | fcnn | 2026 | 13 | 0.097036 | 0.034825 | 0.014364 | 0.155355 |
| weissbart_tf64 | ridge | 0 | 13 | 0.133181 | 0.029443 | 0.066734 | 0.173728 |
| weissbart_tf64 | ridge | 42 | 13 | 0.133181 | 0.029443 | 0.066734 | 0.173728 |
| weissbart_tf64 | ridge | 2026 | 13 | 0.133181 | 0.029443 | 0.066734 | 0.173728 |

## Dataset mean across seeds
| dataset | model | n_seeds | mean_of_seed_means | std_of_seed_means | min_seed_mean | max_seed_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| etard_tf64 | adt | 3 | 0.100890 | 0.002076 | 0.098140 | 0.103153 |
| etard_tf64 | cca | 3 | 0.069850 | 0.000000 | 0.069850 | 0.069850 |
| etard_tf64 | cnn | 3 | 0.093706 | 0.002194 | 0.090604 | 0.095340 |
| etard_tf64 | dnn | 3 | 0.058013 | 0.002916 | 0.055562 | 0.062110 |
| etard_tf64 | eegnet | 3 | 0.104302 | 0.001018 | 0.102895 | 0.105266 |
| etard_tf64 | fcnn | 3 | 0.057509 | 0.003310 | 0.054506 | 0.062121 |
| etard_tf64 | ridge | 3 | 0.103414 | 0.000000 | 0.103414 | 0.103414 |
| weissbart_tf64 | adt | 3 | 0.131368 | 0.003633 | 0.126521 | 0.135267 |
| weissbart_tf64 | cca | 3 | 0.089385 | 0.000000 | 0.089385 | 0.089385 |
| weissbart_tf64 | cnn | 3 | 0.124656 | 0.006292 | 0.115930 | 0.130527 |
| weissbart_tf64 | dnn | 3 | 0.093683 | 0.001884 | 0.091771 | 0.096246 |
| weissbart_tf64 | eegnet | 3 | 0.142219 | 0.002805 | 0.139259 | 0.145987 |
| weissbart_tf64 | fcnn | 3 | 0.093196 | 0.002757 | 0.090692 | 0.097036 |
| weissbart_tf64 | ridge | 3 | 0.133181 | 0.000000 | 0.133181 | 0.133181 |
