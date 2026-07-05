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

## Final status

- completed subjects: `P00 P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12`
- pending subjects: `none`
- failure_report: `empty`
- note: `P09 was rerun from scratch after reboot recovery`

## Final resume check

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_loso_resumable_runner_closure_v1.py --config configs\benchmark\gate0_gate2_loso_eegnet_full_manual_v1.json --device auto --resume --models eegnet --subjects P00 P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 --max-jobs 13 --startup-only
```

## Final artifact checks

```powershell
Get-Content "E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean\run_state.json" -Raw
```

```powershell
Get-Content "E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean\completed_jobs.json" -Raw
```

```powershell
Import-Csv "E:\decode\_fix_loso_resumable_runner_closure_v1\experiments\gate0_gate2_loso_eegnet_full_manual_v1_clean\subject_metrics.csv" | Select-Object dataset,subject_id,model,seed,metric_value,checkpoint_id
```
