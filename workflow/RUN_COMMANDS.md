# EEGNet LOSO Manual Commands

## Working directory

```powershell
cd E:\decode\_fix_loso_resumable_runner_closure_v1
```

## Accepted Weissbart closure

- branch: `fix/ar-20260625-161300-a43831b-loso-resumable-runner-closure-v1`
- commit: `4e365bc`
- config: `configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json`
- output_dir: `E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean`
- completed subjects: `P00 P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12`
- failure_report: `empty`
- note: `P09 was rerun from scratch after reboot recovery`

## Etard prep target

- branch: `fix/ar-20260625-161300-a43831b-loso-eegnet-etard-full-v1`
- config: `configs\benchmark\gate0_gate2_loso_eegnet_etard_manual_v1.json`
- output_dir: `E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_etard_manual_v1_clean`
- dataset: `etard_tf64`
- model: `eegnet`
- seed: `0`
- subject list: `P00 P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19`

## Etard dry-run plan

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_etard_manual_v1.json --device auto --resume --models eegnet --subjects P00 P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19 --max-jobs 20 --dry-run-plan
```

## Etard startup-only

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_etard_manual_v1.json --device auto --resume --models eegnet --subjects P00 P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19 --max-jobs 20 --startup-only
```

## Etard first job plan check

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_etard_manual_v1.json --device auto --resume --models eegnet --subjects P00 P01 P02 --max-jobs 3 --job-plan-only
```

## Etard first manual training chunk

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_etard_manual_v1.json --device auto --resume --models eegnet --subjects P00 P01 P02 --max-jobs 3 --max-runtime-start-new-job-seconds 10800
```

## Real checkpoint closure test

- branch: `fix/ar-20260625-161300-a43831b-loso-checkpoint-saving-closure-v1`
- config: `configs\benchmark\gate0_gate2_loso_checkpoint_saving_real_closure_v1.json`
- output_dir: `E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_checkpoint_saving_real_closure_v1`
- dataset: `weissbart_tf64`
- subject: `P01`
- model: `eegnet`
- seed: `0`
- note: `single real LOSO checkpoint closure job; do not start fine-tuning`

```powershell
cd E:\decode\_fix_loso_resumable_runner_closure_v1

F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_checkpoint_saving_real_closure_v1.json --device auto --resume --models eegnet --subjects P01 --max-jobs 1
```

## Subject-holdout fixed split zero-shot closure

- branch: `fix/ar-20260625-161300-a43831b-subject-holdout-fixed-split-v1`
- config: `configs\benchmark\gate0_gate2_subject_holdout_fixed_split_v1_weissbart_eegnet_seed0.json`
- output_dir: `E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_subject_holdout_fixed_split_v1`
- split manifest: `splits\subject_holdout_fixed_split_v1\weissbart_tf64.json`
- dataset: `weissbart_tf64`
- model: `eegnet`
- seed: `0`
- note: `long zero-shot training; manual run only`

```powershell
cd E:\decode\_fix_loso_resumable_runner_closure_v1

F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_fixed_split_v1.py --config configs\benchmark\gate0_gate2_subject_holdout_fixed_split_v1_weissbart_eegnet_seed0.json --device auto
```

## Subject-holdout 5min fine-tuning closure

- branch: `fix/ar-20260625-161300-a43831b-subject-holdout-finetune-v1`
- config: `configs\benchmark\gate0_gate2_subject_holdout_finetune_v1_weissbart_eegnet_seed0.json`
- output_dir: `E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_subject_holdout_finetune_v1`
- dataset: `weissbart_tf64`
- model: `eegnet`
- seed: `0`
- split_id: `subject_holdout_fixed_split_v1`
- target subjects: `P01 P05 P08`
- source checkpoint: `local_checkpoints\subject_holdout\eegnet\weissbart_tf64\subject_holdout_fixed_split_v1\seed0\best_epoch_5.pt`

## Subject-holdout fine-tuning preflight

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_finetune_v1.py --config configs\benchmark\gate0_gate2_subject_holdout_finetune_v1_weissbart_eegnet_seed0.json --device auto --checkpoint-load-preflight
```

## Subject-holdout fine-tuning startup-only

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_finetune_v1.py --config configs\benchmark\gate0_gate2_subject_holdout_finetune_v1_weissbart_eegnet_seed0.json --device auto --startup-only
```

## Subject-holdout fine-tuning job-plan-only

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_finetune_v1.py --config configs\benchmark\gate0_gate2_subject_holdout_finetune_v1_weissbart_eegnet_seed0.json --device auto --job-plan-only --subjects P01
```

## Subject-holdout fine-tuning full closure

```powershell
cd E:\decode\_fix_loso_resumable_runner_closure_v1

F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_holdout_finetune_v1.py --config configs\benchmark\gate0_gate2_subject_holdout_finetune_v1_weissbart_eegnet_seed0.json --device auto --resume --subjects P01 P05 P08 --max-jobs 3
```
