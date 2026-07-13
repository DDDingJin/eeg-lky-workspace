# Subject-Specific Targeted Audit Preflight

Status: implementation/preflight only unless an explicit manual diagnostic flag is supplied.

## Scope
- VLAAI repeatability job: `weissbart_tf64:P06:vlaai:seed0`
- HappyQuokka read-only job: `etard_tf64:P11:happyquokka:seed0`
- ElasticNet standardized diagnostic job: `etard_tf64:P11:elasticnet:seed0`

## ElasticNet Runtime Normalization
- formal_config_normalization_claim: `linear-family StandardScaler fitted on train split only`
- published_runtime_standardization_applied: `False`
- targeted diagnostic will fit `StandardScaler` on P11 train lag matrix only, then transform train/val/test.

## Policy
- published subject metrics are read-only.
- all new outputs stay in this independent output directory.
- no checkpoint, raw data, prediction dump, or model weights are written by preflight/HQ audit.
