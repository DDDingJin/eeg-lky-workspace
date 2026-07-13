# Manual Commands

These commands are dispatch templates only. This branch does not implement or run training.

## Accepted Chain Roots

- zero-shot runner: `E:/decode/_fix_subject_holdout_pooled_finetune_v1/scripts/run_gate0_gate2_subject_holdout_fixed_split_v1.py`
- pooled10 runner: `E:/decode/_fix_subject_holdout_pooled_finetune_v1/scripts/run_gate0_gate2_subject_holdout_pooled_finetune_v1.py`

## CNN / VLAAI / HappyQuokka Two-Dataset Zero-Shot

Create model-specific accepted-chain configs first, using the source config paths recorded in `job_plan.csv`.
Then run the accepted fixed-holdout zero-shot runner from its owning worktree, for each dataset/model config:

```powershell
cd E:\decode\_fix_subject_holdout_pooled_finetune_v1
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_fixed_split_v1.py --config <accepted_fixed_holdout_config_for_dataset_model_seed0.json> --device auto --dry-run-plan
```

## CNN / VLAAI / HappyQuokka Follow-Up Pooled10

After zero-shot artifacts and local-only checkpoints exist with accepted provenance, run pooled10 through the accepted pooled runner:

```powershell
cd E:\decode\_fix_subject_holdout_pooled_finetune_v1
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_pooled_finetune_v1.py --config <accepted_pooled10_config_for_dataset_model_seed0.json> --device auto --dry-run-plan
```

## Current Command Matrix

- `weissbart_tf64 / cnn`: zero-shot config must reuse `configs/benchmark/gate0_gate2_model_expansion_v1.json::cnn`; pooled10 config must point to the accepted zero-shot checkpoint provenance.
- `weissbart_tf64 / vlaai`: zero-shot config must reuse `configs/benchmark/gate0_gate2_vlaai_happyquokka_training_budget_p00_v1.json::vlaai`; pooled10 config must point to the accepted zero-shot checkpoint provenance.
- `weissbart_tf64 / happyquokka`: zero-shot config must reuse `configs/benchmark/gate0_gate2_subject_specific_happyquokka_seeded_weissbart_full_v1.json::happyquokka`; pooled10 config must point to the accepted zero-shot checkpoint provenance.
- `etard_tf64 / cnn`: zero-shot config must reuse `configs/benchmark/gate0_gate2_model_expansion_v1.json::cnn`; pooled10 config must point to the accepted zero-shot checkpoint provenance.
- `etard_tf64 / vlaai`: zero-shot config must reuse `configs/benchmark/gate0_gate2_vlaai_happyquokka_training_budget_p00_v1.json::vlaai`; pooled10 config must point to the accepted zero-shot checkpoint provenance.
- `etard_tf64 / happyquokka`: zero-shot config must reuse `configs/benchmark/gate0_gate2_subject_specific_happyquokka_seeded_weissbart_full_v1.json::happyquokka`; pooled10 config must point to the accepted zero-shot checkpoint provenance.
