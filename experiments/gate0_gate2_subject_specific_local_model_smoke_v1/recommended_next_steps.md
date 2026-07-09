# Recommended Next Steps

- successful local smoke models: `linear, lasso, elasticnet, vlaai, happyquokka`
- failed local smoke models: `none`
- skipped local smoke models: `none`

## Suggested follow-up order
1. Promote successful smoke adapters to a small multi-subject closure before any wider benchmark run.
2. If `elasticnet` failed, isolate whether the issue is solver stability or interface mismatch before expanding the alpha/l1 grid.
3. If `happyquokka` succeeded, decide whether the 10-second chunk contract is scientifically acceptable as a separate model family rather than forcing 50-sample parity.
4. Keep `decaf` on a separate external-integration branch.
