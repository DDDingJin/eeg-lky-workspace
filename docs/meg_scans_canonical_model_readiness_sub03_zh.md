# MEG-SCANS canonical model readiness sub-03

本目录记录 canonical MEG-SCANS sub-03 audiobook 数据的 native-MEG modeling readiness 检查。

范围限制：

- 只使用 canonical `0.5-8 Hz / 64 Hz` audiobook paired MAT。
- 不使用 OLSA。
- 不重跑 preprocessing。
- 不修改 paired MAT、envelope MAT 或 official `0.5-4 Hz` anchor。
- 不加载 EEG checkpoint，不做 EEG 到 MEG 权重迁移。
- 结果是 sanity/readiness，不是论文 benchmark。

输入数据：

- paired MAT: `E:\decode\data\derived\meg_scans_canonical_05_08hz_64hz_v1\sub-03\speech\sub-03_preprocessed_audiobooks_decoding.mat`
- envelope MAT: `E:\decode\data\derived\meg_scans_canonical_05_08hz_64hz_v1\stimuli\sub-others_preprocessed_audiobook_envelopes_decoding.mat`
- expected paired SHA256: `d7c189af6ba44031ce67745f1511162259af0a1ef07ea19eb8618b33367bf91e`
- expected envelope SHA256: `0aed00fa9a9990d45378627911255d98cd09afc70cdff07372837d296e2b269d`

固定 split：

- train: `[1, 2, 5, 6, 9, 10, 13, 14]`
- val: `[3, 7, 11, 15]`
- test: `[4, 8, 12, 16]`

模型状态：

- `ridge_mag102`: actual sanity run。
- `ridge_all306`: actual sanity run。
- `cca_mag102`: skipped，原因是现有 accepted CCA 接口会在内部对 concatenated recording 构造 lag，无法同时满足本轮禁止跨 recording lag 的约束；未静默替换算法。
- `eegnet_mag102_smoke`: smoke-only，输入实际为 `[batch, 102, 50] -> [batch]`，从头初始化，1 epoch。

主指标：

- 使用统一 scorer 语义：per-recording Pearson，subject metric 为 `mean_recording_pearson_r`。
- Ridge actual 模型额外输出 sorted-vs-mismatched sanity，对 test trials 使用固定 derangement `4 -> 8 -> 12 -> 16 -> 4`。
