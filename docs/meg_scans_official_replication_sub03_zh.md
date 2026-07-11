# MEG-SCANS sub-03 官方 paired-trial preprocessing replication v1

本轮只复现官方 `preprocessing_audiobooks_decoding.m` 的 sub-03 paired MEG-envelope trial 生成步骤。不训练 mTRF，不运行统一 benchmark，不做 R1-R5，不做 source inverse/forward，不做 EEG 权重迁移。

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

paired-trial `.mat` 保存在 Git 外部：

`E:/decode/data/derived/meg_scans_official_replication_v1/sub-03/speech/sub-03_preprocessed_audiobooks_decoding.mat`

Git 中只提交 wrapper、provenance、CSV/JSON/MD validation reports 和文本日志摘要。
