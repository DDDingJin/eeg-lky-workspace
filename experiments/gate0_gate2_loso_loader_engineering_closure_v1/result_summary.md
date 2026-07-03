# LOSO Loader Engineering Closure v1

This package is not a full benchmark. It closes the `num_workers=0` loader-engineering question before any broader LOSO expansion.

## Profiling
- train old seconds: `102.764626`
- train optimized seconds: `1.205648`
- val old seconds: `0.269383`
- val optimized seconds: `0.217479`
- train batch shape unchanged: `True`
- val batch shape unchanged: `True`

## Mini Closure Subject Metrics
- ridge / P00: `0.105327`
- ridge / P01: `0.044541`
- eegnet / P00: `0.022353`
- eegnet / P01: `0.024808`

## Status
- failure count: `0`
- ridge rows are reused from the accepted ridge LOSO package and relabeled to the current engineering-closure protocol.
- eegnet rows are newly run under the optimized single-process preloaded-raw-recording loader.
