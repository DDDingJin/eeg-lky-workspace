# 听觉神经语音解码评估论文：执行蓝图

版本日期：2026-06-25  
状态：供用户、审阅端与执行端共同确认的初步冻结框架  
对应执行分支：`audit/reproduction-note`  
对应审阅 commit：`a43831b83598c82520be320c21b56e92b73b7dcd`

## 1. 论文的控制性问题

本文不以“提出一个模型并在所有数据集上获胜”为前提，而回答一个领域级问题：

> 在统一的数据边界、训练协议和评分标准下，现有听觉神经语音解码模型的优势能否跨被试、跨数据集和跨模态保持？连续包络重建能力能否进一步转化为可靠的语音匹配与注意力判别能力？

这条主线同时容纳：

- 某模型在一个数据集上好、另一个数据集上差；
- 线性模型在小数据或迁移场景中更稳；
- 深度模型在大数据和同分布场景中上限更高；
- reconstruction 高但 match-mismatch/AAD 不一定高；
- EEG 与 MEG 上模型绝对分数不同，但相对排名或归纳偏置可能一致。

## 2. 推荐题目方向

中文工作题目：

> 跨数据集、跨被试与跨模态的听觉神经语音解码基准：从包络重建到决策级解码

英文工作题目：

> Benchmarking Generalization in Auditory Neural Speech Decoding Across Datasets, Subjects, and Modalities

更聚焦的英文备选：

> A Fidelity-Audited Benchmark for Generalizable Auditory Neural Speech Reconstruction and Decision Decoding

## 3. 论文的五项贡献

1. **复现可信度贡献**  
   对进入主表的官方模型进行来源锁定、实现差异记录和最小 parity test，避免把移植错误误判为模型迁移失败。

2. **统一评估贡献**  
   建立 recording-aware 数据边界、统一 split、连续预测输出和模型无关 scorer，使不同窗口、loss 和架构仍能在同一正式指标下比较。

3. **泛化评价贡献**  
   系统比较 subject-specific、pooled seen-subject、LOSO unseen-subject 和跨数据集设置，分析模型排名是否随训练制度变化。

4. **任务关联贡献**  
   以 envelope reconstruction 为主任务，进一步检验重建分数能否转化为 match-mismatch 和 reconstruction-based AAD 决策表现。

5. **开放科学贡献**  
   开源 adapter、split manifest、模型实现、配置、预测、统计脚本、图表和 LaTeX，使结果可重新评分而不必重新训练。

## 4. 任务层级

### 4.1 主任务：Envelope Reconstruction

输入：

- 一段 EEG 或 MEG 阵列时间序列。

输出：

- 与刺激时间对齐的单通道 speech envelope。

正式主指标：

- recording 内 Pearson `r`；
- subject 聚合后的 Fisher-z；
- subject bootstrap 95% CI。

该任务承担全文主要模型比较和泛化分析。

### 4.2 派生任务：Match-Mismatch

输入：

- 神经信号窗口；
- 一个时间对齐候选 envelope；
- 一个或多个预先固定的不匹配候选。

输出：

- 正确候选的匹配分数或排序。

正式主指标：

- binary accuracy；
- score margin；
- AUC；
- 5-way/10-way retrieval 作为补充。

该任务回答：较高的连续重建相关是否具有决策价值。

### 4.3 独立扩展：Auditory Attention Decoding

仅用于真实 competing-speaker、具有 attended/unattended 标签的数据集。

比较两类方法：

- reconstruction-based AAD；
- direct AAD classifier，例如 AADNet 或任务兼容网络。

正式主指标：

- accuracy；
- attended-unattended margin；
- AUC；
- decision-window curve。

AAD 结果不与单说话人 match-mismatch 混成同一个排行榜。

### 4.4 可选扩展：Mel Reconstruction

Mel reconstruction 不阻塞第一篇文章。只有满足以下条件才进入正文：

- 至少两个任务兼容数据集；
- 音频和时间对齐可靠；
- 至少一个简单 baseline 和两个代表性深度模型；
- 计算成本不挤占 reconstruction 主实验。

否则放入补充材料或后续论文。

## 5. 数据集角色

最终数据集名单应在 adapter 审计后冻结。当前建议按角色选择，而不是单纯追求数量。

| Role | Candidate | Modality | Main use | Priority |
|---|---|---:|---|---|
| Development fixture | Hugo/mldecoders sample | EEG | smoke test、parity、快速调试 | 不进入主要结论 |
| Core reconstruction A | Weissbart | EEG | in-dataset reconstruction、subject regimes | Tier A |
| Core reconstruction B | Etard/Reichenbach | EEG | reconstruction、语言/条件分析 | Tier A |
| Large-scale reconstruction | SparrKULee/ICASSP | EEG | 大规模 reconstruction、MM、LOSO | Tier A/B |
| External EEG | DTU/Fuglsang compatible layer | EEG | 外部泛化、AAD | Tier B |
| Classic AAD | KUL AAD | EEG | reconstruction-based AAD、CCA、direct AAD | Tier B |
| Confound audit | AV-GC-AAD | EEG/EOG | gaze robustness | Tier C |
| Language extension | Chinese/Mandarin AAD dataset | EEG | 跨语言或多方向 AAD | Tier C |
| MEG reconstruction A | MEG-SCANS | MEG | 模态扩展、speech reconstruction | Tier B |
| MEG reconstruction B | MEG-MASC or compatible dataset | MEG | MEG 内部泛化 | 待兼容性核验 |

### 5.1 最低可投稿数据集组合

最低版本：

- Weissbart；
- Etard；
- SparrKULee；
- 一个外部 EEG/AAD 数据集。

更强版本：

- 上述四个；
- 一个 MEG 数据集；
- 一个真实 AAD 数据集。

若加入 MEG：

- EEG 与 MEG 分层报告；
- 不直接比较绝对相关谁更高；
- 比较模型排序、相对增益、条件敏感性和迁移下降。

## 6. 模型集合

### 6.1 Reconstruction 核心模型

| Family | Model | Benchmark role | Required status |
|---|---|---|---|
| Linear | Ridge/mTRF backward decoder | 低容量、可解释基线 | Core |
| Correlation | CCA reconstruction | 相关性与低参数基线 | Core |
| Shallow nonlinear | FCNN | 最简单非线性对照 | Core/Optional |
| Convolutional | CNN | 局部时空建模 | Core |
| Compact spatial-temporal | EEGNet regressor | 轻量网络基线 | Core |
| Deep convolutional | VLAAI | 强非线性重建模型 | Core after parity |
| Transformer | ADT | attention-based reconstruction | Core after parity |
| Challenge system | HappyQuokka | 长上下文、subject conditioning | Core after code audit |
| Local exploratory | NULL | 探索与未来创新起点 | Supplementary initially |

第一版不必机械凑满十个。主表应优先保证 7 至 8 个模型完整覆盖，而不是 12 个模型各缺一部分。

### 6.2 AAD 模型

| Family | Candidate | Role |
|---|---|---|
| Reconstruction-based | Ridge/CCA/CNN/VLAAI/ADT/HQ subset | 从重建包络选择 attended stream |
| Correlation-based | CCA/eCCA | 经典 AAD 基线 |
| Direct classifier | AADNet | 端到端 AAD |
| Dataset-specific | 一种公开 CNN/attention baseline | 补充直接分类家族 |

### 6.3 模型复现身份

正文模型表仅保留一列简明身份：

- Official implementation；
- Faithful PyTorch port；
- Architecture baseline；
- Local exploratory model。

详细 parity 报告放仓库和补充材料。

## 7. 评价协议

### 7.1 核心被试协议

#### P1 Subject-Specific

- 每名被试单独训练；
- recording/story 级 train/val/test；
- 回答个性化建模上限。

#### P2 Pooled Seen-Subject

- 所有训练被试共享一个模型；
- 不输入 subject ID；
- 测试同一批被试的未见 recording；
- 回答共享模型能否利用跨被试数据。

#### P3 LOSO Unseen-Subject

- 每一折完全留出一名被试；
- 留出被试不参与 normalization、调参和 checkpoint 选择；
- 回答新用户零校准泛化。

### 7.2 机制协议

#### P4 Subject Conditioning

- 只对支持 `g_con` 或 subject embedding 的模型；
- 与相同架构、相同预算的无条件版本配对；
- 不与普通 pooled 模型混为架构排名。

#### P5 New-Subject Adaptation

- 从 LOSO checkpoint 出发；
- 使用目标被试 0、1、5、10 分钟校准数据；
- 只选择代表模型，不要求完整模型池。

### 7.3 跨数据集协议

只选择科学上可解释的方向：

- 相同任务、相同模态的 EEG-to-EEG；
- 同语言数据集用于隔离设备/协议差异；
- 跨语言数据集用于分析语言与材料变化；
- EEG-to-MEG 或 MEG-to-EEG 作为探索，不直接与同模态迁移等同。

不建议遍历所有数据集两两组合。

## 8. 输入、输出和评分规范

### 8.1 Canonical Recording

每个 adapter 至少输出：

```text
dataset
modality
subject_id
recording_id
condition
language
sampling_rate
channel_names
eeg_or_meg[T, C]
envelope[T, 1]
split
preprocessing_hash
```

### 8.2 模型输入

模型可以保留不同窗口：

- lagged linear input；
- scalar-target CNN window；
- sequence-to-sequence long window。

所有窗口必须在单一 recording 内生成。

### 8.3 模型正式输出

```text
dataset
model
task
protocol
seed
subject_id
recording_id
sampling_rate
time_index
prediction
target
valid_mask
checkpoint_id
```

### 8.4 Reconstruction Scorer

1. 恢复每段 recording 的连续预测；
2. 重叠窗口位置求平均；
3. 标记 lag、padding 或不完整窗口产生的无效位置；
4. 在模型共同有效时间点计算 recording Pearson；
5. recording 先聚合为 subject 分数；
6. subject 作为主要统计单位。

### 8.5 Match-Mismatch Scorer

- 使用固定、可复用的 mismatch manifest；
- 匹配候选和错误候选长度一致；
- 规定最小时间间隔和是否允许同故事；
- 所有模型使用同一候选列表；
- 候选生成与模型训练完全分离。

## 9. 训练公平性

### 9.1 Benchmark Budget

不强制相同 epoch，固定：

- 最大 optimizer steps；
- 每次验证间隔；
- early-stopping patience；
- 每个模型可用的超参数 trial 数；
- 每个模型可见的训练窗口预算；
- 随机 seed 列表。

### 9.2 模型可保留的内容

- 原模型必要的 loss；
- 与架构强关联的 optimizer；
- 原始上下文长度；
- 必要的 subject conditioner。

这些差异必须报告，但不应强迫统一到破坏模型定义。

### 9.3 Budget Sensitivity

对核心模型运行：

- 25% budget；
- 50% budget；
- 100% budget。

该实验回答某模型低分是否仅因为收敛更慢。

### 9.4 Seed 方案

- 开发全矩阵：1 seed；
- 正文核心深度模型：3 seeds；
- 关键或接近结论：5 seeds；
- 线性确定性模型无需重复无意义 seed。

## 10. 统计分析

### 10.1 统计单位

主要单位：

- subject。

次要单位：

- recording；
- condition；
- seed。

segment/window 不能被当作独立生物样本。

### 10.2 Reconstruction

- recording Pearson；
- subject 内先聚合；
- Fisher-z 后检验；
- 表格报告 back-transformed `r`、sample SD；
- 图形报告 subject bootstrap 95% CI 和所有被试点。

### 10.3 单数据集完整模型块

当所有模型覆盖同一批被试：

1. Friedman omnibus；
2. 配对 Wilcoxon signed-rank；
3. Holm correction；
4. rank-biserial effect size；
5. 模型差值 bootstrap CI。

### 10.4 跨数据集与不完整覆盖

主模型：

```text
Fisher-z ~ model * protocol + dataset + modality
            + model:dataset + (1 | dataset:subject)
```

重点解释：

- `model × protocol`；
- `model × dataset`；
- `model × modality`。

若模型和数据覆盖不足，不强行拟合过度复杂交互。

### 10.5 迁移指标

主要使用 Fisher-z 下降：

```text
generalization_drop = z_in_dataset - z_cross_dataset
```

辅助报告：

- relative retention；
- 跨数据集模型排名 Kendall tau；
- 每模型跨数据集方差。

### 10.6 Match-Mismatch 与 AAD

- subject bootstrap 95% CI；
- permutation test against chance；
- 配对模型比较；
- accuracy 与 decision window 的混合模型或重复测量分析；
- 多方向任务报告 confusion matrix 和 macro-F1。

### 10.7 多重比较

- 预先定义的主要比较使用 Holm；
- 大规模探索矩阵使用 Benjamini-Hochberg FDR；
- EEG 主分析与 MEG 探索分析分开校正。

## 11. 论文逐节架构

## 11.1 Abstract

必须包含：

1. 领域问题：不同数据、任务和协议导致模型不可直接比较；
2. 方法：建立 fidelity-audited、recording-aware、多协议 benchmark；
3. 范围：模型数、数据集数、EEG/MEG、reconstruction/MM/AAD；
4. 主要发现：等待结果后填入，不能预写冠军；
5. 意义：给出模型适用边界和可复现资源。

## 11.2 Introduction

### Paragraph 1：为什么听觉神经解码重要

- speech tracking；
- AAD、BCI、助听场景；
- EEG/MEG 的非侵入式价值。

### Paragraph 2：方法已经快速扩展

- linear/TRF；
- CCA；
- CNN/EEGNet/VLAAI；
- Transformer/HappyQuokka/ADT；
- direct AAD。

### Paragraph 3：现有数字不可直接比较

- 数据集不同；
- single-speaker 与 competing-speaker 不同；
- split 不同；
- subject-specific 与 pooled 不同；
- 指标聚合不同；
- gaze、语言、模态等混杂。

### Paragraph 4：现有文献的空白

- 缺少模型复现可信度审计；
- 缺少 recording-aware 统一 scorer；
- 缺少同一架构跨被试、跨数据集、跨模态的系统压力测试；
- reconstruction 与决策表现之间关系不清楚。

### Paragraph 5：本文研究问题

- RQ1：官方模型能否在统一框架中被忠实实现？
- RQ2：模型在同数据、同协议下如何排序？
- RQ3：排名是否随被试协议改变？
- RQ4：哪些模型具有更小跨数据集下降？
- RQ5：reconstruction 是否转化为 MM/AAD？
- RQ6：模型趋势在 EEG/MEG、语言和条件变化下是否稳定？

### Paragraph 6：贡献

压缩为前述五项贡献，不写尚未由结果支持的结论。

## 11.3 Related Work

按问题而非模型名单组织：

1. Envelope reconstruction and TRF；
2. Deep nonlinear reconstruction；
3. Match-mismatch and challenge systems；
4. Reconstruction-based and direct AAD；
5. Cross-subject/cross-dataset generalization；
6. EEG/MEG and confound-aware evaluation；
7. 本文与现有工作的区别。

## 11.4 Benchmark Design

这是现有草稿缺少的独立核心章节。

包含：

- benchmark principles；
- research questions；
- task hierarchy；
- dataset role assignment；
- model-task compatibility；
- fairness boundaries；
- preregistered experiment matrix。

## 11.5 Datasets and Preprocessing

每个数据集统一介绍：

- 来源与许可证；
- modality；
- subjects；
- recordings/hours；
- language；
- paradigm；
- stimulus；
- channel/sensor setting；
- sampling rate；
- conditions；
- task compatibility；
- preprocessing；
- split；
- 本文角色。

正文表格给汇总，逐 ID manifest 放仓库。

## 11.6 Models and Fidelity Audit

先介绍家族差异：

- 输入输出；
- 容量；
- 时间上下文；
- 归纳偏置；
- 预期优势；
- 预期失败模式。

随后报告：

- 来源；
- 框架；
- port 修改；
- 参数量；
- parity 状态；
- 是否进入主表。

## 11.7 Evaluation Protocols

包含：

- subject-specific；
- pooled seen-subject；
- LOSO；
- conditioning ablation；
- cross-dataset；
- adaptation；
- reconstruction scorer；
- MM/AAD scorer；
- budget；
- seeds；
- statistics。

## 11.8 Results

结果章节必须按研究问题组织，而不是按数据集逐个堆表。

### R1 Reproduction and Audit Readiness

- 哪些模型通过 parity；
- 哪些只能作为 architecture baseline；
- 数据与 scorer 验证结果。

### R2 In-Dataset Reconstruction

- 同协议下的主模型比较；
- 被试分布；
- 模型排名；
- 线性与深度差异。

### R3 Subject Generalization

- subject-specific；
- pooled seen；
- LOSO；
- conditioning gain；
- 模型与协议交互。

### R4 Cross-Dataset Generalization

- 选择的迁移矩阵；
- generalization drop；
- ranking stability；
- 数据规模、语言、通道和条件解释。

### R5 Decision-Level Utility

- reconstruction 与 MM accuracy 的关系；
- binary 和 retrieval；
- reconstruction-based AAD 与 direct AAD；
- decision window。

### R6 Modality, Condition and Resource Robustness

- EEG/MEG 趋势；
- 语言/噪声/条件；
- seed 稳定性；
- 性能、参数量、训练时间 Pareto。

## 11.9 Discussion

围绕结果模式：

- 复杂模型是否只在大数据/同分布有优势；
- 低容量模型是否在迁移中更稳；
- 被试条件器的收益是否伴随泛化代价；
- reconstruction 与决策是否一致；
- EEG/MEG 模型趋势是否共通；
- 哪些数据属性导致排名改变；
- 对未来模型设计的启示。

## 11.10 Limitations

- 公开数据的预处理和许可差异；
- 并非所有模型支持所有任务；
- MEG 与 EEG 不能完全控制变量；
- 单 GPU 限制 seed 和调参；
- 某些官方实现缺少权重或完整环境；
- 结果是公开数据现实泛化，而非理想实验室控制。

## 11.11 Data and Code Availability

必须明确：

- 原始数据不重复分发；
- 下载地址和版本；
- adapter、config、manifest、测试；
- 模型来源和 commit；
- 预测级结果；
- 统计和作图脚本；
- LaTeX 和最终 PDF。

## 12. 正文图计划

### Figure 1：Benchmark Pipeline

**目的：** 展示从公开数据、复现审计、模型训练到统一评分和统计的闭环。

子图：

- (a) Dataset adapters and manifests；
- (b) Fidelity-audited model zoo；
- (c) Subject and transfer protocols；
- (d) Unified prediction/scoring/statistics。

### Figure 2：Comparison Design Map

**目的：** 作为全文“怎么比、比什么”的总图。

形式：

- 行：模型家族；
- 列：task × protocol × modality；
- 单元格：Core、Extension、Not applicable；
- 右侧：每类比较对应的正式指标；
- 下方：RQ1-RQ6 与后续结果图的连接。

该图应在正式跑实验前冻结，执行端按空白单元格补结果。

### Figure 3：Dataset and Model Landscape

子图：

- (a) 数据规模、语言、模态、范式散点图；
- (b) dataset-task compatibility heatmap；
- (c) 模型容量、上下文长度和输出形式；
- (d) model fidelity status。

### Figure 4：In-Dataset Reconstruction

子图：

- (a) Weissbart subject-level；
- (b) Etard subject-level；
- (c) SparrKULee subject-level；
- (d) dataset-specific rank and interaction。

图形优先使用 raincloud/box/point，不使用只有 mean±SD 的柱状图。

### Figure 5：Subject and Dataset Generalization

子图：

- (a) subject-specific vs pooled seen；
- (b) pooled seen vs LOSO；
- (c) conditioning gain；
- (d) cross-dataset generalization matrix；
- (e) generalization drop；
- (f) ranking stability。

### Figure 6：From Reconstruction to Decision

子图：

- (a) reconstruction `r` vs binary MM accuracy；
- (b) MM window-duration curves；
- (c) 5/10-way retrieval；
- (d) reconstruction-based AAD vs direct AAD；
- (e) AAD decision-window curves。

### Figure 7：Modality, Language and Condition Robustness

子图：

- (a) EEG 内模型相对排名；
- (b) MEG 内模型相对排名；
- (c) EEG/MEG rank concordance；
- (d) language/condition effect；
- (e) gaze/noise robustness when available。

### Figure 8：Stability and Efficiency

子图：

- (a) seed variance；
- (b) budget sensitivity；
- (c) performance vs parameters；
- (d) performance vs training time；
- (e) accuracy-generalization Pareto frontier。

## 13. 正文表计划

### Table 1：Dataset Inventory

包含数据规模、语言、模态、任务、split、本文角色和下载地址。

### Table 2：Model and Fidelity Inventory

包含来源、官方框架、输入输出、参数量、上下文、训练模式、port 修改和 parity 状态。

### Table 3：Pre-Registered Experiment Matrix

包含 experiment ID、dataset、model family、task、protocol、seed、metric 和 figure destination。

### Table 4：Main Reconstruction Results

按 dataset × protocol 分层，报告 `r`、95% CI、效应量和显著性。

### Table 5：Generalization and Decision Summary

报告 LOSO、cross-dataset drop、MM、AAD 和资源指标。

完整 per-subject 和 pairwise statistics 放补充材料。

## 14. 执行实验矩阵

| ID | Question | Datasets | Models | Protocol | Seeds | Priority |
|---|---|---|---|---|---:|---|
| E0 | 实现是否可信 | fixture + official test | ADT, VLAAI, HQ | parity | fixed | Gate |
| E1 | 同数据谁表现好 | Weissbart, Etard, SparrKULee | core recon set | subject-specific | 1→3 | A |
| E2 | 共享模型是否有效 | two core datasets | feasible core set | pooled seen | 1→3 | A |
| E3 | 新被试泛化 | two core datasets | feasible core set | LOSO | 1→3 | A |
| E4 | 被试 ID 带来什么 | two core datasets | HQ, NULL, compatible models | conditioned ablation | 3 | B |
| E5 | 跨数据集是否稳定 | selected compatible directions | 4-6 representative | cross-dataset | 3 | A/B |
| E6 | 重建是否支持决策 | reconstruction-compatible datasets | all recon outputs | binary MM | scorer only | A |
| E7 | 更难检索 | selected datasets | all recon outputs | 5/10-way | scorer only | B |
| E8 | 真实 AAD | KUL/DTU/compatible | CCA, recon subset, AADNet | within + LOSO | 3 | B |
| E9 | MEG 趋势 | compatible MEG datasets | modality-compatible subset | in-dataset | 1→3 | B |
| E10 | 少样本适配 | one EEG dataset | 3-4 models | 0/1/5/10 min | 3 | C |
| E11 | 资源与稳定性 | core datasets | core deep models | budget curve | 3 | B |

箭头 `1→3` 表示先用一个 seed 完成系统验证，通过后再扩展到正式三个 seeds。

## 15. 结果文件最低规范

每次运行必须产生：

```text
run_config.yaml
environment.json
training_history.csv
checkpoint_manifest.json
predictions.parquet
subject_metrics.csv
recording_metrics.csv
resource_metrics.json
```

汇总脚本只读取这些标准文件，不再针对每个模型临时解析不同 JSON。

## 16. 停止无效扩展的门禁

### Gate 0

报告引用文件全部存在。

### Gate 1

ADT、VLAAI、HappyQuokka 复现状态明确；Ridge/CCA recording 边界修复。

### Gate 2

统一 scorer 对所有模型生成相同 schema。

### Gate 3

两个数据集、核心模型、一个 seed 跑通，所有正文主图能从结果 schema 自动生成。

### Gate 4

再扩展 3 seeds、LOSO、跨数据集、SparrKULee 和 MEG。

### Gate 5

最后增加 MM、AAD、few-shot、gaze 和 Mel。

## 17. 当前不应继续做的事情

- 不在 scorer 未统一时继续扩大排行榜；
- 不将 `exact` 当作未经测试的宣传词；
- 不把不同训练制度放入一个总冠军图；
- 不用窗口数量冒充独立样本量；
- 不为了凑十个模型纳入缺少来源或无法运行的方法；
- 不在数据兼容性未确认时穷举跨数据集组合；
- 不先写“模型 A 显著优于模型 B”的结论。

## 18. 当前可以立即写入论文的内容

- 研究问题与贡献；
- benchmark principles；
- task hierarchy；
- dataset selection criteria；
- model family taxonomy；
- subject protocols；
- output/scoring specification；
- statistical analysis plan；
- figure/table blueprint；
- data and code availability plan；
- limitations 中的预期边界。

所有具体性能、显著性、最佳模型和失败原因必须等待统一实验结果。

## 19. 下一次人工确认

用户与审阅端下一轮只需确认五件事：

1. 是否接受 envelope reconstruction 为绝对主任务；
2. 是否接受 MM 为派生任务、AAD 为独立扩展；
3. 是否接受三个核心被试协议；
4. 是否接受 EEG 主层 + MEG 分层扩展；
5. 是否接受先完成 Gate 3，再允许执行端批量运行完整实验矩阵。

确认后，这份蓝图即可转换为执行端的 experiment registry 和论文 `main.tex` 章节骨架。
