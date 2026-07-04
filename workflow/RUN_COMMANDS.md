# EEGNet Weissbart LOSO Manual Commands

## Working directory

```powershell
cd E:\decode\_fix_loso_resumable_runner_closure_v1
```

## Active config

```powershell
configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json
```

## Active output_dir

```powershell
E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean
```

## Completed subjects in clean output_dir

`P01`

## Chunk 1 startup-only

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P00 P02 P03 --max-jobs 3 --startup-only
```

## Chunk 1 run

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P00 P02 P03 --max-jobs 3
```

## Chunk 2 startup-only

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P04 P05 P06 --max-jobs 3 --startup-only
```

## Chunk 2 run

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P04 P05 P06 --max-jobs 3
```

## Chunk 3 startup-only

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P07 P08 P09 --max-jobs 3 --startup-only
```

## Chunk 3 run

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P07 P08 P09 --max-jobs 3
```

## Chunk 4 startup-only

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P10 P11 P12 --max-jobs 3 --startup-only
```

## Chunk 4 run

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P10 P11 P12 --max-jobs 3
```

## Dry-run example

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P00 P02 P03 --max-jobs 3 --dry-run-plan
```

## Live artifact checks

```powershell
Get-Content "E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean\run_state.json" -Raw
```

```powershell
Get-Content "E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean\completed_jobs.json" -Raw
```

```powershell
Import-Csv "E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean\subject_metrics.csv" | Select-Object dataset,subject_id,model,seed,metric_value,checkpoint_id
```

```powershell
Get-Content "E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean\logs\weissbart_tf64_P01_eegnet_seed0.log" -Tail 50
```

## Safe stop

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Select-Object ProcessId, CommandLine
Stop-Process -Id <PID>
```
