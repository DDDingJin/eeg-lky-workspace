# MEG-SCANS 跨模态传感器表示 preflight v1

本模块是隔离新增模块，base 为 `origin/fix/ar-20260625-161300-a43831b-subject-specific-model-availability-v1` / `41a9c458731962b3dc1da575a159abb6a6c211a8`。本轮只生成 compact artifacts，不训练深度模型，不生成 source inverse、EEG forward 或 virtual EEG。

## 目标

为后续五种表示建立可审阅前置检查：

- R1: MEG306-native，使用 306 个 MEG 通道。
- R2: MEG102-mag-native，只使用 102 个 magnetometer。
- R3: MEG64-select，需要目标 EEG64 坐标；当前 blocked。
- R4: MEG64-interp，只定义接口和权重生成方案；当前因目标 EEG64 坐标 blocked。
- R5: virtual-EEG64，只定义 source inverse + EEG forward 工作流；本轮不执行。

## 当前结论

- sub-03 是正式 preflight subject，包含 4 个 audiobook run。
- sub-01 仅登记为后续 EEG32 + MEG102 bridge validation subject。
- sub-03 生成 `16` 个 120 秒 block manifest rows。
- 既有 EEG 数据无法确认目标 EEG64 坐标，因此 R3/R4 不能给出有效距离映射。

## 后续最小 smoke

建议先使用 R1 或 R2 在 sub-03 的 120 秒 block 上导出 `time x channel` / envelope split，再运行现有 unified scorer 的 recording-level Pearson 聚合。
