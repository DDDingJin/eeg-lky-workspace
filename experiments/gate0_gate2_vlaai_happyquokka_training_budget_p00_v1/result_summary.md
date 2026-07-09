# Subject-Specific Local Model Full-Eval P00 v1

- protocol: `gate0_gate2_vlaai_happyquokka_training_budget_p00_v1`
- dataset: `weissbart_tf64`
- subject: `P00`
- seed: `0`

| model | family | metric | note |
| --- | --- | ---: | --- |
| happyquokka | local_full_eval | 0.08569685795111738 | local full-eval |
| vlaai | local_full_eval | 0.08163938835281748 | local full-eval |

## Scope Notes
- Test evaluation is no longer capped to 256 windows/points.
- Train/validation fitting may still be subsampled for local diagnostic runtime, but test evaluation covers the full available P00 test recording contract for each model family.
- HappyQuokka remains a `10s_chunk` family and is not forced into the 50-sample window contract.
