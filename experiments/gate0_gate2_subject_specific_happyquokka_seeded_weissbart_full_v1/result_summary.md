# Subject-Specific Local Model Full-Eval Weissbart Full v1

- protocol: `gate0_gate2_subject_specific_happyquokka_seeded_weissbart_full_v1`
- dataset: `weissbart_tf64`
- subject: `all_subjects`
- seed: `0`

| model | family | metric | note |
| --- | --- | ---: | --- |
| happyquokka | local_full_eval | 0.12398048733468973 | local full-eval |

## Scope Notes
- Test evaluation is no longer capped to 256 windows/points.
- Train/validation fitting may still be subsampled for local diagnostic runtime, but test evaluation covers the full available Weissbart test recording contract for each subject and model family.
- HappyQuokka remains a `10s_chunk` family and is not forced into the 50-sample window contract.
