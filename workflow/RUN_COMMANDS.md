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
