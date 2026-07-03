# Loader Engineering Closure

This report profiles the existing single-process lazy window loader against a raw-recording preload optimization.
The optimization does not materialize the full window matrix and does not change split rules, target alignment, scorer, or model adapter.

## Scope
- protocol: `gate0_gate2_loso_loader_engineering_closure_v1`
- dataset: `weissbart_tf64`
- heldout subject for profiling: `P00`
- train subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- val subjects: `P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- num_workers: `0`
- persistent_workers: `False`

## Train Loader (200 batches)
- old elapsed seconds: `102.764626`
- optimized elapsed seconds: `1.205648`
- old batch shape: `[256, 64, 50]`
- optimized batch shape: `[256, 64, 50]`
- old cache hit rate: `0.02267578125`
- old load requests: `50039`
- optimized preload loads: `180`
- old recordings loaded more than once: `180`
- optimized recordings loaded more than once: `0`

## Val Loader (50 batches)
- old elapsed seconds: `0.269383`
- optimized elapsed seconds: `0.217479`
- old batch shape: `[256, 64, 50]`
- optimized batch shape: `[256, 64, 50]`
- old cache hit rate: `0.99890625`
- old load requests: `14`
- optimized preload loads: `180`

## Memory Estimate
- old train estimated raw recording bytes: `380524560`
- old train peak cache bytes: `13067600`
- optimized train preloaded bytes: `380524560`
- old val estimated raw recording bytes: `47545680`
- old val peak cache bytes: `1459380`
- optimized val preloaded bytes: `47545680`

## Alignment Invariants
- train batch shape unchanged: `True`
- val batch shape unchanged: `True`
- train target batch shape unchanged: `True`
- val target batch shape unchanged: `True`
- target alignment rule unchanged: `window target index = last sample`
- scorer unchanged: `pearson_on_valid`
- pure LOSO split unchanged: `heldout subject excluded from train and val`
