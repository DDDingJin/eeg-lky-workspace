# Accepted Protocol Audit

- dataset: `weissbart_tf64`
- subject count: `13`
- seed: `0`

## `linear`

- comparable: `partial`
- reason: same split/scorer/aggregation/full-test eval, but train/val fit uses sample cap unlike accepted ridge full-fit baseline
- model_family_contract: `lag_matrix_trf`
- checkpoint_selection_rule: `no checkpoint; OLS fit on train, validation score recorded only`
- known_caveats: budgeted linear-family baseline with max_fit_samples_per_split=12000 on train/val

## `lasso`

- comparable: `partial`
- reason: same split/scorer/aggregation/full-test eval, but train/val fit uses sample cap and lasso regularization differs from accepted ridge baseline
- model_family_contract: `lag_matrix_trf`
- checkpoint_selection_rule: `choose alpha by validation Pearson`
- known_caveats: budgeted linear-family baseline with max_fit_samples_per_split=12000 on train/val

## `elasticnet`

- comparable: `partial`
- reason: same split/scorer/aggregation/full-test eval, but train/val fit uses sample cap and elasticnet regularization differs from accepted ridge baseline
- model_family_contract: `lag_matrix_trf`
- checkpoint_selection_rule: `choose alpha and l1_ratio by validation Pearson`
- known_caveats: budgeted linear-family baseline with max_fit_samples_per_split=12000 on train/val

## `vlaai`

- comparable: `true`
- reason: same split/scorer/aggregation/full-test eval as accepted deep subject-specific models; only model family contract differs
- model_family_contract: `vlaai_local_adapter`
- checkpoint_selection_rule: `best validation Pearson`
- known_caveats: distinct VLAAI window contract with last-target regression output

## `happyquokka`

- comparable: `partial`
- reason: same split/scorer/aggregation/final-test usage, but model family contract and coverage differ from accepted 50-sample window families
- model_family_contract: `10s_chunk`
- checkpoint_selection_rule: `best validation Pearson`
- known_caveats: distinct 10s_chunk contract; per-recording coverage can be <1 due to tail shorter than 10s chunk
