# Leakage Summary

- `protocol`: `gate0_gate2_loso_ridge_full_v1`
- `seed`: `0`
- `model`: `ridge`

Pure LOSO guarantees in this round:
- held-out target subject is excluded from train
- held-out target subject is excluded from val
- held-out target subject is excluded from alpha selection
- held-out target subject appears only in test

## `weissbart_tf64` / `P00`
- train subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P00`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P01`
- train subjects: `P00, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P01`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P02`
- train subjects: `P00, P01, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P01, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P02`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P03`
- train subjects: `P00, P01, P02, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P01, P02, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P03`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P04`
- train subjects: `P00, P01, P02, P03, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P01, P02, P03, P05, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P04`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P05`
- train subjects: `P00, P01, P02, P03, P04, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P01, P02, P03, P04, P06, P07, P08, P09, P10, P11, P12`
- test subjects: `P05`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P06`
- train subjects: `P00, P01, P02, P03, P04, P05, P07, P08, P09, P10, P11, P12`
- val subjects: `P00, P01, P02, P03, P04, P05, P07, P08, P09, P10, P11, P12`
- test subjects: `P06`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P07`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P08, P09, P10, P11, P12`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P08, P09, P10, P11, P12`
- test subjects: `P07`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P08`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P09, P10, P11, P12`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P09, P10, P11, P12`
- test subjects: `P08`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P09`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P10, P11, P12`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P10, P11, P12`
- test subjects: `P09`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P10`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P11, P12`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P11, P12`
- test subjects: `P10`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P11`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P12`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P12`
- test subjects: `P11`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `weissbart_tf64` / `P12`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11`
- test subjects: `P12`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P00`
- train subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P00`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P01`
- train subjects: `P00, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P01`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P02`
- train subjects: `P00, P01, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P02`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P03`
- train subjects: `P00, P01, P02, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P03`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P04`
- train subjects: `P00, P01, P02, P03, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P04`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P05`
- train subjects: `P00, P01, P02, P03, P04, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P05`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P06`
- train subjects: `P00, P01, P02, P03, P04, P05, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P06`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P07`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P07`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P08`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P08`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P09`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P09`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P10`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P11, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P10`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P11`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P12, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P12, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P11`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P12`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P13, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P13, P14, P15, P16, P17, P18, P19`
- test subjects: `P12`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P13`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P14, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P14, P15, P16, P17, P18, P19`
- test subjects: `P13`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P14`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P15, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P15, P16, P17, P18, P19`
- test subjects: `P14`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P15`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P16, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P16, P17, P18, P19`
- test subjects: `P15`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P16`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P17, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P17, P18, P19`
- test subjects: `P16`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P17`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P18, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P18, P19`
- test subjects: `P17`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P18`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P19`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P19`
- test subjects: `P18`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`

## `etard_tf64` / `P19`
- train subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18`
- val subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18`
- test subjects: `P19`
- excluded target from train: `True`
- excluded target from val: `True`
- excluded target from alpha selection: `True`
- selected alpha: `0.001`
