# Subject-Specific Local Model Full-Eval P00 v1

- protocol: `gate0_gate2_subject_specific_local_model_full_eval_p00_v1`
- dataset: `weissbart_tf64`
- subject: `P00`
- seed: `0`

| model | family | metric | note |
| --- | --- | ---: | --- |
| adt | accepted_reference | 0.20621424426632606 | accepted reference |
| cca | accepted_reference | 0.06411140220705791 | accepted reference |
| cnn | accepted_reference | 0.154926323613149 | accepted reference |
| dnn | accepted_reference | 0.06553855512055622 | accepted reference |
| eegnet | accepted_reference | 0.1558581841991332 | accepted reference |
| fcnn | accepted_reference | 0.06643642328319224 | accepted reference |
| ridge | accepted_reference | 0.13254733446979064 | accepted reference |
| elasticnet | local_full_eval | 0.10572577740907098 | local full-eval |
| happyquokka | local_full_eval | -0.031749759892344194 | local full-eval |
| lasso | local_full_eval | 0.10543951402510784 | local full-eval |
| linear | local_full_eval | 0.08713776516933823 | local full-eval |
| vlaai | local_full_eval | 0.08934658952215918 | local full-eval |

## Scope Notes
- Test evaluation is no longer capped to 256 windows/points.
- Train/validation fitting may still be subsampled for local diagnostic runtime, but test evaluation covers the full available P00 test recording contract for each model family.
- HappyQuokka remains a `10s_chunk` family and is not forced into the 50-sample window contract.
