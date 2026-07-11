# MEG-SCANS sub-03 官方 paired-trial preprocessing replication v1

本模块复现官方 sub-03 audiobook/OLSA paired MEG-envelope preprocessing，并生成一个官方 `training_decoding.m` anchor。它是 official replication only，不是本项目统一 EEG/MEG benchmark，不可与统一 Pearson benchmark 或 EEG 结果直接比较。

## 官方协议锁定

- 官方函数：`E:/decode/external/upstream/MEG-SCANS/speech/decoding/preprocessing_audiobooks_decoding.m`
- 官方设置来源：`E:/decode/external/upstream/MEG-SCANS/speech/settings_speech.m`
- 官方 trialfun：`E:/decode/external/upstream/MEG-SCANS/helper_functions/my_trialfun_audiobook.m`
- `use_maxfilter = true`
- `apply_latency_correction = true`
- `audio_latency = 3 ms`
- `bandpass = 0.5-4 Hz`
- `trialdur = 120 s`
- `fs_neuro = 1000 Hz`
- `fs_down = 64 Hz`
- envelope：`derivatives/stimuli/sub-others_preprocessed_audiobook_envelopes_decoding.mat`

## 输出位置

`.mat` 输出均保存在 Git 外部：

`E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_preprocessed_audiobooks_decoding.mat`

`E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_preprocessed_olsa_decoding.mat`

`E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_decoding.mat`

Git 中只提交 wrapper、provenance、CSV/JSON/MD validation reports 和文本日志摘要。

## 官方 decoding anchor

- 官方训练函数：`training_decoding('sub-03', settings)`
- 官方代码使用全局 z-score；audiobook 与 OLSA 分别归一化，MEG mag 与 grad 分别归一化。
- audiobook split 使用 `rng("shuffle")` 后随机 80/20；本次结果是非确定性官方复现 run。
- metric 为 Spearman。
- 本轮只提交 compact 统计，不提交模型、prediction dump 或 `.mat`。
