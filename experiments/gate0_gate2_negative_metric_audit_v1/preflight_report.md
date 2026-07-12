# Negative Metric Diagnostic Preflight

Status: preflight only. No diagnostic rerun, checkpoint, prediction dump, raw data export, or cross-subject training was executed.

## Jobs
- `weissbart_tf64 / P06 / vlaai / seed0` source_metric=`-0.00432094181859644`
- `etard_tf64 / P11 / happyquokka / seed0` source_metric=`-0.0001975521606870697`
- `etard_tf64 / P11 / elasticnet / seed0` source_metric=`-0.0021333631940971577`

## Required Manual Diagnostic Command
```powershell
cd E:\decode\_fix_fixed_split_pooled20_modelset_v1
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_negative_metric_audit_v1.py --config configs\benchmark\gate0_gate2_negative_metric_audit_v1.json --device auto --run-diagnostics
```

## FCNN/DNN Identity
- conclusion: `dnn not proven alias_of=fcnn; hashes differ`

## Policy
- lag sweep is diagnostic only: -128..+128 samples; it must not select checkpoint, hyperparameter, or final score.
- current scorer metric and independent Pearson must both be reported in the manual diagnostic run.
