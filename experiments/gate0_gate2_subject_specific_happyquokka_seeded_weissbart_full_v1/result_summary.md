# Subject-Specific Local Model Full-Eval Weissbart Full v1

- protocol: `gate0_gate2_subject_specific_happyquokka_seeded_weissbart_full_v1`
- dataset: `weissbart_tf64`
- subject: `all_subjects`
- seed: `0`

| model | family | metric | note |
| --- | --- | ---: | --- |
| happyquokka | local_full_eval | 0.12398048733468973 | local full-eval |
| happyquokka | local_full_eval | 0.11985516571673505 | local full-eval |
| happyquokka | local_full_eval | 0.08948923630210734 | local full-eval |
| happyquokka | local_full_eval | 0.09016431516507921 | local full-eval |
| happyquokka | local_full_eval | 0.06618615447026523 | local full-eval |
| happyquokka | local_full_eval | 0.17538350771711173 | local full-eval |
| happyquokka | local_full_eval | 0.010693212790145965 | local full-eval |
| happyquokka | local_full_eval | 0.14078891437872484 | local full-eval |
| happyquokka | local_full_eval | 0.08780414224816877 | local full-eval |
| happyquokka | local_full_eval | 0.19395309012394638 | local full-eval |
| happyquokka | local_full_eval | 0.08068507484370747 | local full-eval |
| happyquokka | local_full_eval | 0.19741273404829585 | local full-eval |
| happyquokka | local_full_eval | 0.2142479501669975 | local full-eval |

## Scope Notes
- Test evaluation is no longer capped to 256 windows/points.
- Train/validation fitting may still be subsampled for local diagnostic runtime, but test evaluation covers the full available Weissbart test recording contract for each subject and model family.
- HappyQuokka remains a `10s_chunk` family and is not forced into the 50-sample window contract.
