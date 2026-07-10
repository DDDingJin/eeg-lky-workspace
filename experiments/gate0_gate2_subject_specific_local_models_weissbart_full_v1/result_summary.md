# Subject-Specific Local Model Full-Eval Weissbart Full v1

- protocol: `gate0_gate2_subject_specific_local_models_weissbart_full_v1`
- dataset: `weissbart_tf64`
- subject: `all_subjects`
- seed: `0`

| model | family | metric | note |
| --- | --- | ---: | --- |
| elasticnet | local_full_eval | 0.10572577740907098 | local full-eval |
| elasticnet | local_full_eval | 0.08367361771471674 | local full-eval |
| elasticnet | local_full_eval | 0.09831205753491948 | local full-eval |
| elasticnet | local_full_eval | 0.13065897188938982 | local full-eval |
| elasticnet | local_full_eval | 0.12294387139998404 | local full-eval |
| elasticnet | local_full_eval | 0.1304355936780162 | local full-eval |
| elasticnet | local_full_eval | 0.06002709546499239 | local full-eval |
| elasticnet | local_full_eval | 0.10969724371258649 | local full-eval |
| elasticnet | local_full_eval | 0.08322809154812218 | local full-eval |
| elasticnet | local_full_eval | 0.13459838178511133 | local full-eval |
| elasticnet | local_full_eval | 0.10275775702201671 | local full-eval |
| elasticnet | local_full_eval | 0.13331598979201378 | local full-eval |
| elasticnet | local_full_eval | 0.16766181410204234 | local full-eval |
| happyquokka | local_full_eval | 0.08569685795111738 | local full-eval |
| happyquokka | local_full_eval | 0.03946283373509691 | local full-eval |
| happyquokka | local_full_eval | 0.10791564070468398 | local full-eval |
| happyquokka | local_full_eval | 0.1767197783543857 | local full-eval |
| happyquokka | local_full_eval | 0.09180382032832883 | local full-eval |
| happyquokka | local_full_eval | 0.19886248301210638 | local full-eval |
| happyquokka | local_full_eval | 0.01472662286508464 | local full-eval |
| happyquokka | local_full_eval | 0.12446668873757773 | local full-eval |
| happyquokka | local_full_eval | 0.01923296943669545 | local full-eval |
| happyquokka | local_full_eval | 0.19232913054379605 | local full-eval |
| happyquokka | local_full_eval | 0.08862443804057271 | local full-eval |
| happyquokka | local_full_eval | 0.052175547321480144 | local full-eval |
| happyquokka | local_full_eval | 0.20810665133825332 | local full-eval |
| lasso | local_full_eval | 0.10543951402510784 | local full-eval |
| lasso | local_full_eval | 0.07935923374432896 | local full-eval |
| lasso | local_full_eval | 0.10910393989759545 | local full-eval |
| lasso | local_full_eval | 0.12861212396959545 | local full-eval |
| lasso | local_full_eval | 0.11973510888137297 | local full-eval |
| lasso | local_full_eval | 0.13072518490740795 | local full-eval |
| lasso | local_full_eval | 0.054726549959807066 | local full-eval |
| lasso | local_full_eval | 0.08105590851383755 | local full-eval |
| lasso | local_full_eval | 0.10744044016658015 | local full-eval |
| lasso | local_full_eval | 0.13295334864061484 | local full-eval |
| lasso | local_full_eval | 0.10720019917467097 | local full-eval |
| lasso | local_full_eval | 0.13196177833280837 | local full-eval |
| lasso | local_full_eval | 0.16914155612113527 | local full-eval |
| linear | local_full_eval | 0.08713776516933823 | local full-eval |
| linear | local_full_eval | 0.045803683113870355 | local full-eval |
| linear | local_full_eval | 0.09132738143915838 | local full-eval |
| linear | local_full_eval | 0.11395624406070433 | local full-eval |
| linear | local_full_eval | 0.10658886322030268 | local full-eval |
| linear | local_full_eval | 0.09859802794889115 | local full-eval |
| linear | local_full_eval | 0.051471282986620376 | local full-eval |
| linear | local_full_eval | 0.09184350723991225 | local full-eval |
| linear | local_full_eval | 0.06336599695831904 | local full-eval |
| linear | local_full_eval | 0.10312302386354676 | local full-eval |
| linear | local_full_eval | 0.09145687722601076 | local full-eval |
| linear | local_full_eval | 0.11542583465695497 | local full-eval |
| linear | local_full_eval | 0.13131834168122572 | local full-eval |
| vlaai | local_full_eval | 0.08132087080623955 | local full-eval |
| vlaai | local_full_eval | 0.08182630281933458 | local full-eval |
| vlaai | local_full_eval | 0.11144597470627263 | local full-eval |
| vlaai | local_full_eval | 0.05708163502378872 | local full-eval |
| vlaai | local_full_eval | 0.10292963913186143 | local full-eval |
| vlaai | local_full_eval | 0.13152276223639855 | local full-eval |
| vlaai | local_full_eval | -0.00432094181859644 | local full-eval |
| vlaai | local_full_eval | 0.045203574271906195 | local full-eval |
| vlaai | local_full_eval | 0.055231307701817195 | local full-eval |
| vlaai | local_full_eval | 0.13681227583680147 | local full-eval |
| vlaai | local_full_eval | 0.06537472826821693 | local full-eval |
| vlaai | local_full_eval | 0.05027721199330533 | local full-eval |
| vlaai | local_full_eval | 0.181706730058688 | local full-eval |

## Scope Notes
- Test evaluation is no longer capped to 256 windows/points.
- Train/validation fitting may still be subsampled for local diagnostic runtime, but test evaluation covers the full available Weissbart test recording contract for each subject and model family.
- HappyQuokka remains a `10s_chunk` family and is not forced into the 50-sample window contract.
