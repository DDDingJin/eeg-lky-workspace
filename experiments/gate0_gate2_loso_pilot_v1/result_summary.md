# Gate0-Gate2 LOSO Pilot v1

This directory contains a minimal subject-independent LOSO pilot for `weissbart_tf64` using only `ridge` on held-out `P00`, `P01`, and `P02`.
It is execution evidence for the LOSO interface and does not add new models or paper prose.

## Protocol
- `protocol`: `gate0_gate2_loso_pilot_v1`
- `dataset`: `weissbart_tf64`
- `model`: `ridge`
- `heldout_subjects`: `P00, P01, P02`
- `seed`: `0`

## Subject metrics
| subject | mean_recording_pearson_r | num_recordings | checkpoint_id |
| --- | ---: | ---: | --- |
| P00 | 0.105327 | 15 | ridge_alpha_0.001 |
| P01 | 0.044541 | 15 | ridge_alpha_0.001 |
| P02 | 0.048566 | 15 | ridge_alpha_0.001 |

## Recording rows
- total recording metric rows: `45`
- total subject metric rows: `3`

## Leakage status
- `P00`: train/val exclude target=`True`, test subjects=`P00`
- `P01`: train/val exclude target=`True`, test subjects=`P01`
- `P02`: train/val exclude target=`True`, test subjects=`P02`
