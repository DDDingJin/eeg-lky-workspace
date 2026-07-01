# Leakage Check

- `protocol`: `gate0_gate2_loso_pilot_v1`
- `dataset`: `weissbart_tf64`
- `model`: `ridge`

This pilot uses pure LOSO subject-independent fitting:
- train uses only non-heldout subjects' `train` split
- validation uses only non-heldout subjects' `val` split
- test uses only heldout subject `test` split
- heldout subject train/val/test data are excluded from fit, scaler, and hyperparameter selection

## Held-out `P00`
- train subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P00`
- excluded target from train: `True`
- excluded target from val: `True`
- test recording count: `15`
- selected alpha by pooled validation: `0.001`

## Held-out `P01`
- train subjects: `P00, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P01`
- excluded target from train: `True`
- excluded target from val: `True`
- test recording count: `15`
- selected alpha by pooled validation: `0.001`

## Held-out `P02`
- train subjects: `P00, P01, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P01, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P02`
- excluded target from train: `True`
- excluded target from val: `True`
- test recording count: `15`
- selected alpha by pooled validation: `0.001`
