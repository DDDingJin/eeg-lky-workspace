# MEG-SCANS subject-specific sub-03 v1

本轮是 canonical MEG-SCANS sub-03 audiobook 的第一轮正式 native-MEG subject-specific pilot。

边界：

- 不重跑或修改 canonical `0.5-8 Hz / 64 Hz` paired/envelope MAT。
- 不使用 OLSA、LOSO、CCA、多 seed、跨模态迁移或 EEG checkpoint。
- 16 个 120 s trial 仅作为 train/val/test 原子 recording 单位；lag/window 只在 recording 内构造。
- checkpoint 仅保留在内存中，不提交权重、prediction dump 或大数组。

固定 split：

- train: `[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]`
- val: `[4, 12]`
- test: `[8, 16]`

模型：

- `recording_safe_ridge_adapter_mag102`
- `recording_safe_ridge_adapter_all306`
- `eegnet_mag102`

EEGNet 使用 CUDA，输入为 `[batch, 102, 50] -> [batch]`，每 epoch 跑 validation，以 validation Pearson 选择 best checkpoint，最后只评估一次 test。
