# Within-Dataset Fixed Holdout Modelset zero_shot

- stage: `zero_shot`
- planned jobs: `2`
- pending jobs after resume/max-jobs filtering: `2`
- training started: `false`
- metrics written: `false`
- engineering smoke: `true`
- DNN excluded: `functional_alias_of=fcnn`
- linear-family zero-shot deferred jobs: `0`

## Split Subjects
- `weissbart_tf64` train=`P00,P09,P02,P07,P11,P06,P04,P03` val=`P12,P10` test=`P08,P05,P01`
- `etard_tf64` train=`P05,P02,P00,P01,P13,P04,P11,P07,P18,P06,P08,P12` val=`P09,P15,P14,P19` test=`P17,P16,P10,P03`

## Manual Smoke Command
```powershell
cd E:\decode\_fix_fixed_split_pooled20_modelset_v1
F:\miniconda\envs\decode-torch\python.exe scripts\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py --config configs\benchmark\gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json --stage smoke --device auto --models fcnn,cnn,eegnet,adt,vlaai,happyquokka --resume
```
