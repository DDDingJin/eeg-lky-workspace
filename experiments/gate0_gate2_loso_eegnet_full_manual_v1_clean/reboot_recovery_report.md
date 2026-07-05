# Reboot Recovery Report

- LastBootUpTime: `2026/7/5 4:50:01`
- System evidence confirms an abnormal interruption rather than a clean stop:
  - `2026/7/5 4:50:05` `Kernel-Power 41`
  - `2026/7/5 4:50:11` `EventLog 6008`
  - no `1074` planned restart event was captured in the earlier diagnostic pass
- Windows Update activity existed within the same 24-hour window, but no direct evidence was captured that it issued the restart.

## Corrupt State Files
- `run_state.json` had become all `0x00` bytes and was quarantined to `experiments/gate0_gate2_loso_eegnet_full_manual_v1_clean/run_state.corrupt_20260705_111141.json`
- `heartbeat.json` had become all `0x00` bytes and was quarantined to `experiments/gate0_gate2_loso_eegnet_full_manual_v1_clean/heartbeat.corrupt_20260705_111141.json`
- repaired `run_state.json` and `heartbeat.json` are now valid JSON and represent reconstructed runtime state only

## Recovery Facts
- completed subjects in the requested `P04-P12` slice: `P04, P05, P06, P07, P08`
- pending subjects in the requested `P04-P12` slice: `P09, P10, P11, P12`
- earlier completed history `P00-P03` remains preserved in `completed_jobs.json` and `subject_metrics.csv`
- `P09` has only a partial log and does not count as completed
- `P09` must be rerun from scratch on the next real resume
- recovery source of truth is `completed_jobs.json` + `subject_metrics.csv` + `recording_metrics.csv` + `failure_report.json`
- `run_state.json` and `heartbeat.json` are not authoritative for recovery decisions
