# FCNN Protocol Audit

This audit is intentionally small. It does not rerun full-subject training and it does not try to tune FCNN further. The goal is only to judge whether the current FCNN in the focused rerun is directly comparable to the historical Etard FCNN summary.

## Files compared

Current FCNN:
- `src/repro/simple_models.py`
- `scripts/run_gate0_gate2_focused_rerun.py`
- `configs/benchmark/gate0_gate2_etard_p00_focused_rerun.json`

Historical FCNN summary:
- `experiments/summary_figures/unified_reference_main_summary.csv`
- historical metric source:
  - `experiments/reference_baselines/etard_tf64/fcnn/metrics.csv`
- recoverable generic historical runner:
  - `scripts/run_reference_baselines.py`
- recoverable upstream model class:
  - `external/upstream/mldecoders/pipeline/dnn.py`

## Direct answers

1. Current FCNN and the old summary FCNN should **not** be treated as directly comparable.

2. The main differences are:
- the current focused rerun uses local `src/repro/simple_models.py::FCNNBaseline`
- the older summary path points to the upstream `pipeline.dnn.FCNN` family
- the local class and upstream class are structurally similar, but they are not literally the same implementation identity
- the current result is a single-subject `P00` pilot, while the old summary is a 20-subject Etard mean
- the current reporting path uses the unified scorer and subject-level `mean_recording_pearson_r`, while the old summary is derived from participant-level `test_pearson` files and then aggregated into the dataset summary table

3. Yes. The current FCNN should be labeled more conservatively as a `local_fcnn_baseline` or `architecture_baseline`, not as proof of parity with the historical FCNN summary.

4. There is **no strong evidence** here for a pipeline bug.
- ridge and cca already stayed in the historical scale
- ADT recovered to the historical scale after budget increase
- FCNN staying low after budget increase is more consistent with protocol or implementation non-equivalence than with a global scorer or alignment failure

5. No large FCNN tuning campaign is justified at this stage.
- The branch goal is benchmark stabilization, not FCNN score maximization
- Repeated FCNN tuning would likely produce a model-specific optimization detour without improving the auditability of the benchmark

6. Yes. Full-subject single-seed can still continue **if FCNN is explicitly downgraded in interpretation**.
- It should be retained only as a weak/simple/local baseline
- It should not be used as a parity anchor against the historical FCNN summary

## Recoverability note

The old FCNN protocol is only partially recoverable from the current branch.
- We can recover the generic runner family and the saved metrics path
- We can recover the upstream FCNN class that the generic runner imports
- We cannot prove from the current branch alone that the exact historical run used no extra local edits, no environment differences, and no branch-specific wrapper behavior

Therefore:
- `old FCNN protocol not fully recoverable from current branch`

## Recommendation

Recommendation: `C`

- FCNN is not shown to have a pipeline bug
- FCNN is also not shown to be directly comparable with the old summary
- keep it as a `local/simple baseline`
- do not let it block the next full-subject single-seed phase for the stronger baseline family and ADT
