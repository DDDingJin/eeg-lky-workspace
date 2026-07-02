# LOSO vs Subject-Specific Comparison

This file compares the current LOSO result against the existing subject-specific seed-0 baseline from `gate0_gate2_full_subject_single_seed_v1`.
Note: `dnn` and `fcnn` remain two implementations within the same `MLP/FCNN family`; they are not described here as separate architecture families.

| dataset | model | subject | loso_metric | subject_specific_metric | delta_loso_minus_subject_specific |
| --- | --- | --- | ---: | ---: | ---: |
| etard_tf64 | ridge | P00 | 0.027889 | 0.088560 | -0.060671 |
| etard_tf64 | ridge | P01 | 0.006295 | 0.093488 | -0.087193 |
| etard_tf64 | ridge | P02 | 0.049877 | 0.077094 | -0.027218 |
| etard_tf64 | ridge | P03 | 0.057472 | 0.104055 | -0.046583 |
| etard_tf64 | ridge | P04 | 0.093388 | 0.175744 | -0.082356 |
| etard_tf64 | ridge | P05 | 0.132972 | 0.165606 | -0.032634 |
| etard_tf64 | ridge | P06 | 0.058930 | 0.074566 | -0.015635 |
| etard_tf64 | ridge | P07 | 0.038626 | 0.053228 | -0.014602 |
| etard_tf64 | ridge | P08 | 0.099142 | 0.139684 | -0.040541 |
| etard_tf64 | ridge | P09 | 0.093474 | 0.100271 | -0.006796 |
| etard_tf64 | ridge | P10 | 0.109191 | 0.144008 | -0.034817 |
| etard_tf64 | ridge | P11 | -0.010371 | 0.044848 | -0.055219 |
| etard_tf64 | ridge | P12 | 0.044461 | 0.051038 | -0.006577 |
| etard_tf64 | ridge | P13 | 0.048698 | 0.082365 | -0.033666 |
| etard_tf64 | ridge | P14 | 0.038149 | 0.071046 | -0.032898 |
| etard_tf64 | ridge | P15 | 0.035003 | 0.067240 | -0.032237 |
| etard_tf64 | ridge | P16 | 0.088469 | 0.118127 | -0.029659 |
| etard_tf64 | ridge | P17 | 0.069021 | 0.147982 | -0.078961 |
| etard_tf64 | ridge | P18 | 0.072030 | 0.107870 | -0.035840 |
| etard_tf64 | ridge | P19 | 0.094038 | 0.161450 | -0.067412 |
| weissbart_tf64 | ridge | P00 | 0.105327 | 0.132547 | -0.027221 |
| weissbart_tf64 | ridge | P01 | 0.044541 | 0.084190 | -0.039649 |
| weissbart_tf64 | ridge | P02 | 0.048566 | 0.146138 | -0.097572 |
| weissbart_tf64 | ridge | P03 | 0.099467 | 0.153287 | -0.053820 |
| weissbart_tf64 | ridge | P04 | 0.059734 | 0.156419 | -0.096686 |
| weissbart_tf64 | ridge | P05 | 0.056507 | 0.153185 | -0.096678 |
| weissbart_tf64 | ridge | P06 | 0.014096 | 0.066734 | -0.052637 |
| weissbart_tf64 | ridge | P07 | 0.019698 | 0.138584 | -0.118886 |
| weissbart_tf64 | ridge | P08 | 0.113465 | 0.103321 | 0.010144 |
| weissbart_tf64 | ridge | P09 | 0.026733 | 0.141483 | -0.114750 |
| weissbart_tf64 | ridge | P10 | 0.065418 | 0.131127 | -0.065708 |
| weissbart_tf64 | ridge | P11 | 0.080036 | 0.150607 | -0.070570 |
| weissbart_tf64 | ridge | P12 | 0.117471 | 0.173728 | -0.056257 |
