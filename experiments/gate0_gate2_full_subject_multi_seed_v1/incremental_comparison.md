# Incremental Comparison

- previous branch: `fix/ar-20260625-161300-a43831b-full-subject-single-seed-v1`
- previous commit: `e459f2ca4807c9f72334c20fa93b2d3dfc67c253`
- current branch: `fix/ar-20260625-161300-a43831b-multi-seed-v1`
- this round extends the full-subject benchmark from single-seed to multi-seed v1.
- seed 0 was reused from the existing single-seed run.
- only missing seed jobs for 42 and 2026 were executed.

## New in this round
- `configs/benchmark/gate0_gate2_full_subject_multi_seed_v1.json`
- `scripts/run_gate0_gate2_full_subject_multi_seed_v1.py`
- seed-specific dataset metrics and seed-averaged summaries
- Etard condition-level seed-averaged summary

## Historical results not rerun
- `experiments/summary_figures/*`
- pilot / focused rerun diagnostic bundles
- specialized models outside `ridge / cca / fcnn / adt`
