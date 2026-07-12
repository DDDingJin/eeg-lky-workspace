# Manual Chunk Commands

These commands are preflight chunk checks only and do not start training. Formal execution remains disabled in this branch until the reviewer explicitly opens long-run execution.

```powershell
cd E:\decode\_fix_fixed_split_pooled20_modelset_v1
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_specific_modelset_etard_full_v1.py --config configs\benchmark\gate0_gate2_subject_specific_modelset_etard_full_v1.json --device auto --dry-run-plan --subjects P00,P01,P02,P03,P04  # P00-P04
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_specific_modelset_etard_full_v1.py --config configs\benchmark\gate0_gate2_subject_specific_modelset_etard_full_v1.json --device auto --dry-run-plan --subjects P05,P06,P07,P08,P09  # P05-P09
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_specific_modelset_etard_full_v1.py --config configs\benchmark\gate0_gate2_subject_specific_modelset_etard_full_v1.json --device auto --dry-run-plan --subjects P10,P11,P12,P13,P14  # P10-P14
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_subject_specific_modelset_etard_full_v1.py --config configs\benchmark\gate0_gate2_subject_specific_modelset_etard_full_v1.json --device auto --dry-run-plan --subjects P15,P16,P17,P18,P19  # P15-P19
```
