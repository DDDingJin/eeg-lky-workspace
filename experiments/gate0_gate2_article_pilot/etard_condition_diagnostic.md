# Etard Condition Diagnostic

This file diagnoses the current Etard P00 pilot without rerunning any model.

## Direct answers
1. The low Etard P00 mean is condition-driven rather than uniformly low. The weakest cells are `fcnn:fW=0.002, fcnn:fM=0.003, adt:fW=0.006, fcnn:mb=0.009`.
2. The `clean` condition is not the main problem. It remains moderate for all four models and is generally above the hardest mixed/noisy conditions.
3. The hardest conditions are concentrated in `fM, fW` style noisy or distractor-heavy splits, with condition-specific differences by model.
4. Condition sensitivity differs by model: ridge degrades but stays relatively stable; CCA is comparatively flat but low; FCNN shows the strongest collapse on the hardest conditions; ADT is also sensitive and loses more than ridge on some noisy conditions.
5. Yes, later full-subject analysis or paper figures should report Etard by condition, not only a single pooled mean.
6. If results are not stratified by condition, the overall mean can hide that a model is adequate on `clean` but unstable on `fM/fW/hb/lb/mb`, or conversely that a low pooled average is driven by a small subset of difficult recordings.

## Preliminary recommendation
- Condition stratification should be included before making strong claims about model ranking on Etard.
