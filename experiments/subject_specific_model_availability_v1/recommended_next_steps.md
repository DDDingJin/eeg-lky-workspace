# Recommended Next Steps

## Priority 1: metadata and low-risk adapter closure

1. Add exact registry rows for:
   - `eegnet`
   - `lasso`
   - `elasticnet` only after a real implementation exists
   - `decaf` only after external integration is intentionally approved

2. Add the smallest possible subject-specific runner branches for:
   - `linear`
   - `lasso`

Rationale:

- Both can reuse the existing per-subject train/val/test flow used by ridge-style models.
- Neither needs a new external dependency.
- This is the fastest path to enlarge the subject-specific model set without changing scientific protocol.

## Priority 2: repair or define sparse linear coverage

1. Resolve the `elastic_net` gap:
   - either restore the missing sparse-decoder fit/eval functions referenced by `scripts/run_reference_baselines.py`
   - or explicitly remove/rename the stale CLI token until implementation exists

2. Decide naming policy:
   - use `elasticnet` everywhere
   - or use `elastic_net` everywhere
   - avoid mixed naming across registry, runner, and reports

## Priority 3: sequence-model adapter audit

1. For `vlaai`, decide whether subject-specific benchmarking should use:
   - a scalar regressor adapter over `[batch, 64, window] -> [batch]`
   - or a sequence-output protocol with a separate reporting path

2. For `happyquokka`, decide whether subject-specific benchmarking should:
   - preserve the upstream 10-second chunk contract
   - or add a benchmark-local adapter to the current 64 Hz / 50-sample window contract

Without that decision, both models remain engineering-risky rather than smoke-ready.

## Priority 4: DECAF external integration gate

Do not start integration yet. If DECAF becomes a target model, do this first:

1. Lock the external source repo and commit SHA.
2. Record the MIT license and dependency set.
3. Audit `src/models/model_factory` and data loaders for exact input/output contract.
4. Decide whether DECAF enters the benchmark as:
   - a source-faithful external stack
   - or a benchmark-local adapter over the existing 64 Hz / 50-sample contract

## Suggested smoke order

1. `linear`
2. `lasso`
3. `vlaai`
4. `happyquokka`
5. `elasticnet`
6. `decaf`

This order minimizes protocol risk and external dependency churn.
