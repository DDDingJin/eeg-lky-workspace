# 执行端复现报告审阅版

审阅日期：2026-06-25  
远端分支：`audit/reproduction-note`  
审阅 commit：`a43831b83598c82520be320c21b56e92b73b7dcd`  
原始报告：`docs/reproduction_audit_note.tex`

## 1. 文档身份

执行端没有在 GitHub 分支中提交编译后的 PDF。当前提交的是 LaTeX 源码、Markdown 状态说明、CSV、PNG 和部分模型代码。

本审阅版用于：

- 把执行端报告的主要信息整理成可直接阅读的中文 PDF；
- 区分执行端已经证明的内容和仍未证明的内容；
- 给出进入正式论文实验前必须完成的任务。

它不是执行端原始 PDF，也不替代原始代码和结果文件。

## 2. 执行端当前声称的评价设置

当前统一 reconstruction 数据包括：

- `hugo_sample_tf64`：开发与快速验证；
- `weissbart_tf64`：13 名被试；
- `etard_tf64`：20 名被试。

当前模型分为：

- subject-specific：Ridge、CCA、FCNN、CNN、EEGNet；
- pooled multi-subject：ADT、VLAAI；
- pooled subject-conditioned：HappyQuokka `g_con=True`；
- pooled unconditioned：HappyQuokka `g_con=False`；
- 当前分支另有 NULL/NeuroConformer 相关结果，但原 LaTeX 报告尚未同步。

## 3. 当前汇总结果

以下数值来自执行端提交的汇总 CSV，只能视为 preliminary。

| Dataset | Ridge | CCA | FCNN | CNN | EEGNet | ADT | VLAAI | HappyQuokka g-con |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weissbart | 0.1332 | 0.0931 | 0.0905 | 0.1228 | 0.1465 | 0.1147 | 0.1389 | 0.1577 |
| Etard | 0.0982 | 0.0680 | 0.0621 | 0.0956 | 0.1003 | 0.0860 | 0.1128 | 0.1287 |

这些数值不能形成最终排名，因为：

- 训练制度不同；
- 输出和 Pearson 聚合方式不同；
- HappyQuokka 使用被试 ID；
- 部分实现与运行代码没有提交；
- 深度模型多数只有一个 seed。

## 4. 执行端图形

![当前统一结果柱状图](/Users/macbookair/Documents/解码与快速写文的探索/tmp/eeg-lky-rule-work/experiments/summary_figures/unified_reference_main_overview.png)

图中混合 subject-specific、pooled 和 conditioned 模型，误差线为不统一定义的 SD。该图适合开发检查，不适合论文主图。

![当前训练曲线](/Users/macbookair/Documents/解码与快速写文的探索/tmp/eeg-lky-rule-work/experiments/summary_figures/reproduction_audit_training_curves.png)

subject-specific 曲线在后期只对仍有该 epoch 记录的被试求平均，因此样本数随 epoch 改变。部分后段标准差变为 0，不代表全部被试稳定收敛。

![HappyQuokka conditioning](/Users/macbookair/Documents/解码与快速写文的探索/tmp/eeg-lky-rule-work/experiments/summary_figures/happyquokka_conditioning_overview.png)

该图能说明 conditioning 可能提高同批被试上的表现，但不能直接证明对未见被试有效。

## 5. 已经合理的部分

1. train、validation 和 test 被区分；
2. checkpoint 由 validation 选择；
3. test 原则上只用于最终评价；
4. 报告主动承认当前不是 apples-to-apples leaderboard；
5. 已记录 requested epochs、completed epochs 和 best epoch；
6. 已比较 HappyQuokka conditioned 与 unconditioned；
7. 已开始保存训练曲线和协议 metadata。

## 6. 代码审阅发现

### 6.1 审阅包缺失关键代码

报告或 README 引用了以下路径，但 commit 中不存在：

- `src/repro/happyquokka_reference.py`；
- `scripts/run_reference_baselines.py`；
- `scripts/run_reference_exact_suite.py`；
- `src/repro/mldecoders/cca.py`；
- `src/repro/mldecoders/linear_baselines.py`。

因此 HappyQuokka、CCA、Ridge 变体和基础网络目前不能完整复核或运行。

### 6.2 ADT mask 与官方方向一致

官方 ADT commit：

`e31ad252960e05856abb9c54669c3f65ebfa49ac`

官方和本地均采用上三角允许矩阵。原先“mask 方向可能写反”的怀疑不再作为主要问题。

剩余核验：

- PyTorch 与 Keras 对全屏蔽行的行为；
- mask 广播；
- 固定权重/输入前向输出；
- 参数量和逐层 shape。

### 6.3 VLAAI 目前不能称为 exact

官方 VLAAI commit：

`9af13054bf73188e9c248f249f44ecc0cd02f6c5`

当前明确差异：

- 官方 TensorFlow LeakyReLU 默认斜率与本地 PyTorch 默认值不同；
- Keras 和 PyTorch LayerNorm 默认 epsilon 不同。

官方仓库提供：

- 预训练权重；
- ONNX；
- DTU `S1_000` 固定测试；
- 预期 reconstruction correlation `0.244431957...`。

执行端应以该测试作为 VLAAI port 的验收锚点。

### 6.4 Ridge 与 CCA 可能跨 recording 构造 lag

外围代码先调用 `concatenate_reference_split`，再构造 lag。这样 recording A 末尾和 B 开头可能形成不存在的连续上下文。

正确做法：

1. 每段 recording 内独立构造 lag；
2. 删除边缘无效点；
3. 合并合法样本；
4. 训练样本再随机打乱。

### 6.5 当前 Pearson 不是统一指标

- Ridge/FCNN/CNN/EEGNet：偏向拼接标量预测后计算；
- ADT/VLAAI/HQ：偏向窗口或分段 Pearson 后平均；
- CCA：包含 reconstruction correlation 和 canonical correlation。

最终必须导出 recording-level 连续预测，由同一个 scorer 重算。

## 7. 图表审阅

当前图表存在：

- 只显示 mean±SD；
- 没有被试点和配对关系；
- 不同训练制度混图；
- 误差线定义不统一；
- 训练曲线参与平均的被试数随 epoch 改变；
- 模型标签过长；
- 没有统计检验和 effect size；
- 没有跨协议、跨数据集和跨模态结果。

正式图应采用：

- subject-level points；
- bootstrap 95% CI；
- protocol 分面；
- paired lines；
- transfer matrix；
- ranking stability；
- performance-resource Pareto。

## 8. 审阅结论

> 当前报告可作为开发进度说明，不可作为论文结果章节的基石。

现有分数应保留并标记为：

`preliminary / non-comparable / single-seed`

在以下三项完成前，不建议继续大规模增加模型：

1. 审阅包完整；
2. 核心模型 parity 与 recording 边界修复；
3. 统一 reconstruction scorer。

## 9. 执行端下一次报告必须包含

1. 当前 commit 和生成时间；
2. 所有引用文件的存在检查；
3. 每个模型的来源、身份和 parity 状态；
4. 数据集 manifest 和 split 检查；
5. per-recording 连续预测；
6. per-subject × seed 长表；
7. 统一 scorer 结果；
8. subject-specific、pooled seen 和 LOSO 分表；
9. 训练预算、steps、seed 和资源；
10. 自动生成的论文图草图。

## 10. 建议决定

审阅状态：

`MAJOR REVISION`

允许执行端解释或反驳每条意见，但必须提供：

- 代码位置；
- 配置；
- 日志；
- 测试；
- 结果差异。

仅用“分数更高”“原论文也这样做”或“目前能运行”不能关闭问题。
