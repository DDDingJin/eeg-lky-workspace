# 从 MEG-SCANS 单数据集、单被试训练读代码

这份指南对应已经完成的 `sub-03` native-MEG 11 模型运行。它是理解整个仓库最合适的起点：数据规模小、划分固定、训练入口集中，而且结果和审计产物齐全。

## 从哪里开始

- 配置：`configs/benchmark/meg_scans_subject_specific_11models/sub03_v1.json`
- 总 runner：`scripts/meg_scans_subject_specific_11models/run_11models_v1.py`
- 结果目录：`experiments/meg_scans_subject_specific_11models_all_canonicalized_v1/`
- 结果审计：`reviews/meg_scans_sub03_11model_result_audit_2026-07-13.md`（如果该审计文件在工作树中可用）

正式运行的命令形态为：

```bash
python scripts/meg_scans_subject_specific_11models/run_11models_v1.py \
  --config configs/benchmark/meg_scans_subject_specific_11models/sub03_v1.json \
  --resume
```

`--preflight-only` 只做数据和形状检查；`--smoke-gate` 会用缩小的训练预算跑完整个模型名单；`--resume --max-jobs N` 用于可中断的增量执行。CUDA 不可用时，深度模型会主动报错，而不是静默切到 CPU。

## 一次运行的完整数据流

```text
JSON 配置
  -> load_paired_mat：读取 16 个 MEG/包络 trial
  -> validate_and_select：SHA256、64 Hz、形状、通道顺序检查
  -> split_trials：12 train / 2 validation / 2 test，按整段 recording 切分
  -> run_model：按模型分发到经典模型或 PyTorch 训练
  -> validation 选择超参数或最佳 epoch
  -> held-out test 仅评估一次
  -> recording_metrics.csv -> subject_metrics.csv -> schema_validation_report.json
```

每个 trial 是 120 秒、64 Hz，即 7680 个时间点；原始输入为 `[7680, 306]`。runner 通过固定清单选择 102 个 magnetometer 通道，并对通道顺序和数据文件的 SHA256 做校验。它拒绝把 trial 切碎后再随机分到不同集合，因此 train/validation/test 不会共享同一个原始 recording。

固定划分为 train 的 12 个 trial、validation 的 2 个 trial（4、12）和 test 的 2 个 trial（8、16）。`StandardScaler` 只在 train recording 上拟合，然后用于三组数据；这是防止统计量泄漏的关键一行逻辑。

## 两条训练路径

### 1. 经典模型：`run_linear_family` 与 `run_cca`

`linear`、`ridge`、`lasso`、`elasticnet` 共享 50 个时间滞后。对时刻 `t` 的输出，输入包含 `t, t-1, ..., t-49` 的 102 通道特征，因此每条 recording 的前 49 个样本不参与评分。Ridge/Lasso/ElasticNet 在 validation 的两条 recording 上选超参数，最后才读取 test。

CCA 训练时会使用 train 的 EEG/MEG 滞后矩阵和包络滞后矩阵建立共同表示，再以 Ridge 把 CCA 表示重建为中心时刻的包络。validation/test 推理只走 `X -> scaler/PCA -> CCA.transform(X) -> Ridge -> y_hat`；真实 test 包络只在末尾计算 Pearson r，不能进入预测路径。

### 2. 深度模型：`run_torch_model`

`WindowDataset` 只在每条 recording 内产生窗口，不会跨 recording 拼接窗口。训练循环的共同结构是：

1. 使用 train 的窗口更新参数；
2. 每个 epoch 将 validation windows 重组成完整 recording，算两条 recording 的平均 Pearson r；
3. 保存 validation 最好的内存权重；
4. 达到耐心阈值后停止，恢复最佳权重；
5. 对 test 的两条 recording 只运行一次重建和评分。

模型接口并不完全相同，这一点必须在解释结果时保留：

| 模型 | 输入/目标 | 窗口 |
| --- | --- | --- |
| FCNN、CNN、EEGNet | `[batch, 102, 50] -> 当前窗口最后一个包络样本` | 50，步长 1 |
| VLAAI adapter | 先学得 `102 -> 64` 投影，再走 VLAAI | 50，步长 1 |
| ADT | `[batch, 320, 102] -> 320 点包络序列` | 320，步长 64 |
| HappyQuokka adapter | 先学得 `102 -> 64` 投影，再走 chunk 解码器 | 640，步长 640 |

因此它们共享 train/val/test、归一化和最终的 recording-level Pearson 指标，但并不共享完全相同的时间上下文或优化目标。当前结果适合离线重建比较；若要做严格实时/因果比较，需要另建统一的 endpoint/decision-window 协议。

## 看哪些输出能判断一次训练是否真的成功

- `data_integrity_check.json`：数据文件哈希、采样率、trial 数、102 通道顺序。
- `split_manifest.json`：每个 trial 属于 train/validation/test 的证据。
- `shape_preflight.json`：每个模型的输入/输出形状和 CUDA 可用性。
- `validation_diagnostics.csv` 与 `training_history.csv`：经典模型的候选选择、深度模型的逐 epoch 轨迹。
- `completed_jobs.json`、`failure_report.json`、`run_state.json`：11 个 job 是否完整、可恢复且无失败。
- `recording_metrics.csv`：每个模型两条 held-out recording 的 Pearson r。
- `subject_metrics.csv`：上述两条 recording 的算术平均，这是主指标 `mean_recording_pearson_r`。
- `schema_validation_report.json`：检查 job 唯一性、每个成功 job 恰有两条 recording 指标和一条 subject 指标，以及不提交数据/权重等禁用文件。

## 已完成的单被试结果应如何理解

`sub-03`、seed 0、固定划分的 11 个 job 均已完成且审计通过。它是对 runner 的成功验证，不是模型总排名证据：只有一个被试、一个随机种子和两条测试 recording；并且不同架构的窗口/输出协议不同。下一步扩展时，应保持本配置的输入表示、划分和评分端点不变，再逐步增加 canonicalized 被试。
