# EEG：从已完成的被试内任务到跨被试任务

这条是 EEG 主线，不应与 MEG-SCANS 的 `sub-03` 示例混淆。它先在每位被试内部训练和测试；随后改为固定的“训练被试 / 验证被试 / 测试被试”划分，训练一个 pooled 模型并在未见被试上测试。

## 先校正术语

“单数据集、单被试”在这里指 **subject-specific（被试内）协议**：一次 job 只使用一位被试自己的 train/validation/test recordings，不把别人的数据用于这次模型拟合。它不是只跑了一个人。

已闭环的纯 EEG subject-specific 汇总为：

- `weissbart_tf64`：13 位被试；`etard_tf64`：20 位被试。
- 每位被试运行 12 个模型、seed 0，共 `33 × 12 = 396` 个 job。
- 汇总检查为 `396/396` 成功、`0` 失败，且每个 job 都有 recording 指标和 subject 指标。
- 汇总证据：`experiments/gate0_gate2_subject_specific_modelset_two_dataset_closure_v1/closure_validation_report.json`。

所以，EEG 的被试内阶段已经不是“单个 pilot”，而是可复查的完整阶段；当前新工作是跨被试泛化，而不是重新做这 396 个 job。

## 被试内 runner 如何工作

主要入口：

- 配置：`configs/benchmark/gate0_gate2_subject_specific_modelset_etard_full_v1.json`
- 编排器：`scripts/run_gate0_gate2_subject_specific_modelset_etard_full_v1.py`
- 基础四模型历史入口：`scripts/run_gate0_gate2_full_subject_single_seed.py`

`run_gate0_gate2_subject_specific_modelset_etard_full_v1.py` 本身主要是“任务调度器”，不是把 12 个模型重新实现一遍。它为每个 `(subject, model, seed)` 建立独立 job 目录，再按模型类型委派：

- `ridge / cca / fcnn / dnn / cnn / eegnet / adt`：复用 `run_gate0_gate2_full_subject_single_seed.py` 的 `run_job`。
- `linear / lasso / elasticnet / vlaai / happyquokka`：复用 `run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1.py` 的完整评估函数。

每个成功 job 生成两类东西：

1. `recording_metrics.csv`：每条完整测试 recording 的 Pearson r；
2. `subject_metrics.csv`：该被试所有测试 recording 的算术平均，即主指标 `mean_recording_pearson_r`。

最后调度器把所有 job 合并成数据集级 `dataset_metrics.csv`。这层的关键价值是：训练实现可以不同，但汇总端点统一为“完整 recording 重建 -> recording Pearson -> 被试平均”。

## 跨被试 runner 改变了什么

入口改为：

- 配置：`configs/benchmark/gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json`
- runner：`scripts/run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py`

每个 job 不再是 `subject:model`，而是 `dataset:model:seed:stage`。也就是说，每个模型对一个数据集只训练一次 pooled 网络，再测试该数据集的全部 held-out 被试。

固定被试划分为：

| 数据集 | Train subjects | Validation subjects | Test subjects |
| --- | ---: | ---: | ---: |
| Weissbart | 8 | 2 | 3 |
| Etard | 12 | 4 | 4 |

运行前 `validate_split` 检查三组被试是否不相交。深度模型的数据集类也会按被试逐个读取 recording，再在 recording 内产生窗口，因此不会跨 recording 产生窗口，更不会让同一被试同时进入 train 和 test。

## 代码最值得读的 6 个位置

1. `build_jobs`：生成 `2 datasets × 11 models = 22` 个跨被试任务；`dnn` 被排除，因为它已审计为 `fcnn` 的功能别名。
2. `MultiSubjectWindowDataset`：FCNN/CNN/EEGNet/VLAAI 的 50 点 endpoint windows；输入为 `[64, 50]`，目标为窗口最后一个包络样本。
3. `MultiSubjectSequenceDataset`：ADT 的 320 点序列窗口，hop 为 64。
4. `MultiSubjectHappyQuokkaDataset`：HappyQuokka 的 10 秒 chunk；当前 `g_con=false`，不使用被试 ID 或真实包络作为额外输入。
5. `train_model`：只用 train subjects 优化；每 epoch 在 validation subjects 上选最佳权重；zero-shot 才把 checkpoint 写到本地 `local_checkpoints/`，不提交进 Git。
6. `evaluate_model`：仅对 test subjects 做完整 recording 重建，再还原为 recording / subject / dataset 三层指标。

一个容易忽略的细节：跨被试深度模型的 checkpoint 选择使用 validation batches 上的相关系数；最终报告则是 test recording Pearson 后按被试平均。两者方向一致，但聚合粒度不同，论文和结果说明中应明确。

## 当前跨被试状态（此仓库快照）

这不是“22 个正式 zero-shot 结果已完成”。当前已留存的证据是：

- `smoke`：22 个计划任务中，12 个非线性模型完成真实前向/反向/优化和验证；5 个线性族模型在两个数据集上共 10 个任务被明确延后。Smoke 不写正式指标或 checkpoint。
- `zero_shot`：正式 22-job 运行尚未开始；线性族仍因 validation-fix 范围待确认而延后。
- `zero_shot engineering smoke`：FCNN 在两个数据集各完成一个真实跨被试 job，共 2 个；会写本地 checkpoint 和测试指标，但标记为工程证据，不应当作为科学结果。

对应证据分别在：

- `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_smoke/`
- `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot/`
- `experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot_engineering_smoke/`

因此，阅读 EEG 代码的顺序应该是：先把已经完成且稳定的 subject-specific 数据流读懂，再看固定被试划分如何把“同一个人内训练”替换成“多训练被试 pooled 训练、未见被试测试”。不要把 smoke 的数值当成正式跨被试 benchmark，也不要把被试内的高分直接解释为泛化能力。
