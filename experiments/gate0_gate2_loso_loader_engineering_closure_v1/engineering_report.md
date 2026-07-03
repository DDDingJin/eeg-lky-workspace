# Engineering Report

This package is an engineering closure for the single-process LOSO loader path. It is not a full LOSO benchmark package.

## Fixed Constraints
- num_workers: `0`
- persistent_workers: `False`
- model adapter changed: `false`
- target alignment changed: `false`
- scorer changed: `false`
- split / LOSO rule changed: `false`
- training budget changed: `false`
- optimization type: `raw recording preload only; no full window matrix materialization`

## Profiling Comparison
- train old seconds for 200 batches: `102.764626`
- train optimized seconds for 200 batches: `1.205648`
- val old seconds for 50 batches: `0.269383`
- val optimized seconds for 50 batches: `0.217479`
- old train load requests: `50039`
- optimized train preload loads: `180`
- old train recordings loaded more than once: `180`
- optimized train recordings loaded more than once: `0`
- train batch shape unchanged: `True`
- val batch shape unchanged: `True`

## Mini Closure Coverage
- subject_metrics rows expected: `4`
- ridge / P00: `0.105327`
- ridge / P01: `0.044541`
- eegnet / P00: `0.022353`
- eegnet / P01: `0.024808`

## Artifact Hygiene
- raw data uploaded: `false`
- prediction dump uploaded: `false`
- checkpoint uploaded: `false`
- model weights uploaded: `false`
- per-job directories uploaded: `false`
- large array / weight files uploaded: `false`
