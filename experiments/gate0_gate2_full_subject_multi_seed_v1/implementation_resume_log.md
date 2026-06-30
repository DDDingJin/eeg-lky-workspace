## Multi-Seed Resume Log

- branch: `fix/ar-20260625-161300-a43831b-multi-seed-v1`
- base single-seed commit: `e459f2ca4807c9f72334c20fa93b2d3dfc67c253`
- log updated: `2026-06-30T00:00:00+08:00`

### Context

- This worktree contained uncommitted multi-seed code, configs, and partial local results for seeds `42` and `2026`.
- The pushed Git history only contained the single-seed v1 checkpoint at commit `e459f2c`.
- Local seed coverage before resume:
  - `seed 0`: complete and already committed
  - `seed 42`: complete locally
  - `seed 2026`: incomplete locally

### Resume Checks

- Verified the locked review-loop skill references:
  - `workflow/skill_lock.json`
  - `workflow/skill_alignment.md`
- Verified local worktree branch and reflog lineage for the multi-seed branch.
- Verified local result presence under:
  - `experiments/gate0_gate2_full_subject_single_seed/jobs/.../seed_42`
  - `experiments/gate0_gate2_full_subject_single_seed/jobs/.../seed_2026`

### Missing Jobs Found Before Resume

- `seed 2026` was missing `23` jobs, all on `etard_tf64`:
  - `P14`: `adt`, `cca`, `fcnn`
  - `P15`: `ridge`, `cca`, `fcnn`, `adt`
  - `P16`: `ridge`, `cca`, `fcnn`, `adt`
  - `P17`: `ridge`, `cca`, `fcnn`, `adt`
  - `P18`: `ridge`, `cca`, `fcnn`, `adt`
  - `P19`: `ridge`, `cca`, `fcnn`, `adt`

### Commands Run

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts/run_gate0_gate2_full_subject_multi_seed_v1.py --config configs/benchmark/gate0_gate2_full_subject_multi_seed_v1_seed2026_etard.json --device cuda --resume --skip-existing
```

- First long run reduced the missing set from `23` jobs to `4` jobs.
- Second run completed the remaining `P19` jobs.

```powershell
F:\miniconda\envs\decode-torch\python.exe scripts/run_gate0_gate2_full_subject_multi_seed_v1.py --config configs/benchmark/gate0_gate2_full_subject_multi_seed_v1.json --device cuda --resume --skip-existing
```

- Rebuilt the full multi-seed summary from existing local jobs across seeds `0`, `42`, and `2026`.

### Environment Used

- interpreter: `F:\miniconda\envs\decode-torch\python.exe`
- `scipy`: `1.15.3`
- `torch`: `2.11.0+cu128`
- `cuda available`: `True`

### Final State

- `seed 42`: complete
- `seed 2026`: complete
- final full inventory:
  - datasets: `weissbart_tf64`, `etard_tf64`
  - models: `ridge`, `cca`, `fcnn`, `adt`
  - seeds: `0`, `42`, `2026`
  - total jobs: `396`
  - successful jobs: `396`
  - failed jobs: `0`

### Key Output Files

- `experiments/gate0_gate2_full_subject_multi_seed_v1/run_manifest.json`
- `experiments/gate0_gate2_full_subject_multi_seed_v1/job_ledger.csv`
- `experiments/gate0_gate2_full_subject_multi_seed_v1/dataset_metrics_by_seed.csv`
- `experiments/gate0_gate2_full_subject_multi_seed_v1/dataset_metrics_across_seeds.csv`
- `experiments/gate0_gate2_full_subject_multi_seed_v1/etard_condition_metrics_by_seed.csv`
- `experiments/gate0_gate2_full_subject_multi_seed_v1/etard_condition_metrics_across_seeds.csv`
- `experiments/gate0_gate2_full_subject_multi_seed_v1/result_summary.md`

### Notes

- The first attempted resume using the default `python` failed because the base environment lacked `scipy`.
- No benchmark failures were recorded after rerunning with the `decode-torch` environment.
