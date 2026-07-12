# MEG-SCANS canonical 0.5-8 Hz / 64 Hz sub-03

本模块只生成 sub-03 audiobook canonical 预处理派生物，不训练任何模型，不做 Ridge/EEGNet/source/R3/R4/R5。

本地数据本体写入 Git 外部：

`E:/decode/data/derived/meg_scans_canonical_05_08hz_64hz_v1/sub-03/`

协议：

- MEG source: tSSS + motion-corrected FIF。
- MEG band-pass: 0.5-8 Hz。
- Envelope source: 原始 audiobook WAV，不能从旧 0.5-4 Hz envelope 再滤波。
- Envelope method: AMT auditory filterbank 50-5000 Hz -> `abs(.)^0.6` -> 跨频带均值。
- Envelope temporal band-pass: 1000 Hz intermediate 上做 0.5-8 Hz。
- Final fs: 64 Hz。
- Trial duration: 120 s。
- Audio latency correction: 3 ms，沿用官方 paired-trial preprocessing。
- Preprocessing 不做 global z-score；未来训练 runner 必须只在训练 split 上拟合 scaler。

官方 0.5-4 Hz anchor 文件和 official replication 结果禁止覆盖，继续作为 official replication only。
