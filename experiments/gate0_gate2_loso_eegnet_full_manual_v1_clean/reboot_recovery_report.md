# Reboot Recovery Report

- LastBootUpTime: `2026/7/5 4:50:01`
- Kernel-Power 41 / EventLog 6008 evidence:
  - /Date(1783198205402)/ | Id=41 | Provider=Microsoft-Windows-Kernel-Power | Message=系统已在未先正常关机的情况下重新启动。如果系统停止响应、发生崩溃或意外断电，则可能会导致此错误。
  - /Date(1783198211521)/ | Id=6008 | Provider=EventLog | Message=上一次系统的 4:24:25 在 ‎2026/‎7/‎5 上的关闭是意外的。
- Windows Update evidence (past 24h):
  - /Date(1783220473141)/ | Id=19 | Message=安装成功: Windows 成功安装了下列更新: Microsoft Defender Antivirus 的安全智能更新 - KB2267602（版本 1.453.429.0）- 当前频道（广泛）
  - /Date(1783220467269)/ | Id=43 | Message=安装已启动: Windows 已开始安装以下更新: Microsoft Defender Antivirus 的安全智能更新 - KB2267602（版本 1.453.429.0）- 当前频道（广泛）
  - /Date(1783220466375)/ | Id=44 | Message=Windows 更新已开始下载更新。
  - /Date(1783220446581)/ | Id=19 | Message=安装成功: Windows 成功安装了下列更新: Microsoft Defender Antivirus 的安全智能更新 - KB2267602（版本 1.453.422.0）- 当前频道（广泛）
  - /Date(1783220446581)/ | Id=44 | Message=Windows 更新已开始下载更新。
  - /Date(1783194885101)/ | Id=19 | Message=安装成功: Windows 成功安装了下列更新: Microsoft Defender Antivirus 的安全智能更新 - KB2267602（版本 1.453.426.0）- 当前频道（广泛）
  - /Date(1783194878595)/ | Id=43 | Message=安装已启动: Windows 已开始安装以下更新: Microsoft Defender Antivirus 的安全智能更新 - KB2267602（版本 1.453.426.0）- 当前频道（广泛）
  - /Date(1783194878595)/ | Id=44 | Message=Windows 更新已开始下载更新。
  - /Date(1783192898211)/ | Id=19 | Message=安装成功: Windows 成功安装了下列更新: 9MSMLRH6LZF3-Microsoft.WindowsNotepad
  - /Date(1783192897209)/ | Id=43 | Message=安装已启动: Windows 已开始安装以下更新: 9MSMLRH6LZF3-Microsoft.WindowsNotepad
  - /Date(1783192896695)/ | Id=19 | Message=安装成功: Windows 成功安装了下列更新: 9P1J8S7CCWWT-Clipchamp.Clipchamp
  - /Date(1783192894555)/ | Id=43 | Message=安装已启动: Windows 已开始安装以下更新: 9P1J8S7CCWWT-Clipchamp.Clipchamp
  - /Date(1783192880252)/ | Id=44 | Message=Windows 更新已开始下载更新。
  - /Date(1783192880252)/ | Id=44 | Message=Windows 更新已开始下载更新。

## Corrupt State Files
- run_state.json status: `all_zero`
- run_state.json error: `file contains only 0x00 bytes`
- run_state.json quarantine: `experiments/gate0_gate2_loso_eegnet_full_manual_v1_clean/run_state.corrupt_20260705_111141.json`
- heartbeat.json status: `all_zero`
- heartbeat.json error: `file contains only 0x00 bytes`
- heartbeat.json quarantine: `experiments/gate0_gate2_loso_eegnet_full_manual_v1_clean/heartbeat.corrupt_20260705_111141.json`

## Recovery Facts
- completed subjects: `P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12`
- pending subjects: `none`
- P09 partial log exists and does not count as a completed job.
- P09 must be rerun from scratch on the next real resume.
- source of truth for recovery: `completed_jobs.json` + `subject_metrics.csv` + `recording_metrics.csv` + `failure_report.json`.
- `run_state.json` and `heartbeat.json` are treated as non-authoritative runtime state.
