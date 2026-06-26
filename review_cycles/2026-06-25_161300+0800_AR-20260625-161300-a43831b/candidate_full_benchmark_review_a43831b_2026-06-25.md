# 听觉 EEG Benchmark 候选审阅确认表

审阅日期：2026-06-25

当前状态：候选意见，尚未上传 GitHub

审阅对象：

- 仓库：`DDDingJin/eeg-lky-workspace`
- 分支：`audit/reproduction-note`
- Commit：`a43831b83598c82520be320c21b56e92b73b7dcd`

## 使用方法

请在每条意见的“用户决定”和“用户注释”位置填写内容。

可用决定：

- `接受`：进入正式审阅。
- `修改`：保留问题，但修改严重程度、范围或要求。
- `补充调查`：先继续核查证据。
- `暂缓`：问题成立，但本轮不要求执行。
- `驳回`：不进入正式审阅，并说明原因。

正式上传前，本文件中的 `DRAFT-*` 编号不会转为正式 `AUDIT-*` 问题。

---

## 一、模型复现正确性

### DRAFT-A01：审阅包缺少关键实现文件

- 严重程度：`Blocker`
- 置信度：高
- 类别：复现完整性

**问题**

README 和审阅文档提到了 `happyquokka_reference.py`、`mldecoders` 相关模块及若干运行脚本，但当前审阅分支没有提供全部依赖文件。外部审阅者无法仅根据该分支完整运行或核查模型。

**影响**

- 无法验证模型代码与汇总结果是否对应。
- 无法判断部分 baseline 是否正确实现。
- 当前 GitHub 包不能作为最终可复现代码发布。

**建议动作**

上传论文结果所依赖的模型代码、运行入口、配置和必要工具模块；或者生成明确的代码依赖 manifest，指向固定 commit。

**用户决定：**

> 

**用户注释或问题：**

> 你的意思是他的相关代码没有完全上传上来，而是只给了它相关的结果？？我觉得这里你的审阅思路没有问题，因为这个工作后面最好要将代码全部开源，因此我们在审阅的时候也需要对这个代码进行规范化的显示与检查，这里后期也应该规范一下，到底整个上传文件夹的分布应该是怎么样的，才既方便检查又符合一贯的开源架构。

**审阅端回复（2026-06-25）：**

> 是的。当前分支不是完全“只有结果”，但确实只上传了部分核心代码和汇总结果。例如 README 提到了 `happyquokka_reference.py`，当前分支实际没有该文件；`reference_baselines.py` 还依赖未包含在审阅包中的 `mldecoders` 模块。因而外部人员无法只克隆这一分支就完整复现实验。
>
> 最终开源仓库建议至少形成以下结构：
>
> ```text
> configs/                 每个数据集、模型和实验协议的配置
> src/
>   data/                  下载、预处理、适配与 split
>   models/                模型实现与统一 adapter
>   evaluation/            统一 scorer 和统计分析
> scripts/                 可直接运行的训练、测试和汇总入口
> tests/                   mask、shape、边界、parity 等测试
> manifests/               数据、split、模型来源和版本清单
> results/
>   summaries/             可公开的小型 CSV/JSON 汇总
> docs/
>   model_fidelity/        各模型复现说明
>   protocols/             评估协议
> environment/             requirements、lockfile 或容器配置
> ```
>
> 原始 EEG、大 checkpoint 和大规模中间文件不必放进 Git；但产生论文结果的代码、配置、每被试结果、数据下载说明和校验信息必须存在。该意见建议保留为 `Blocker`。

---

### DRAFT-A02：缺少系统的模型复现忠实度证明

- 严重程度：`Blocker`
- 置信度：高
- 类别：模型复现

**问题**

当前使用了 `exact`、`official wrapper` 和 `upstream-style` 等描述，但没有为每个模型系统记录：

- 原论文和官方仓库；
- 上游 commit；
- 原始超参数；
- 本地参数映射；
- 修改过的结构或训练流程；
- 与上游实现的 parity test。

**影响**

无法区分“忠实复现”“结构移植”“受启发的本地 baseline”。模型间性能差异可能来自实现偏差，而不是模型能力。

**建议动作**

建立 model fidelity table，并为高优先级模型增加最小输入输出对照、参数量对照和小数据 parity test。

**用户决定：**

> 

**用户注释或问题：**

> 我觉得你说的这个很有道理，而且最后在写文章的时候 其实应该有一个benchmark或者关于数据的介绍，这些介绍里是不是都应该放着这些对比的方法或者对比数据的下载仓库或者别的什么，但是这里需要注明 哪些的仓库用的tensorflow实现，或者哪些是什么。。。但是我们为了尽可能的统一评估，将他们都改为pytorch下的对比。改了什么你这里确实也需要说一下，最好在细节上都和原来一样，因为我们这里是为了评估，当然以一切以完整复现为主，而且做完评估 进行下一个创新网络的时候 应该也用的是这些。

**审阅端回复（2026-06-25）：**

> 论文和仓库都应提供模型来源表，至少包含论文、官方代码链接、上游 commit、原框架、许可证、原任务、原采样率、输入输出形式，以及本地修改。
>
> 但这里需要纠正一个思路：**统一评估不等于必须把所有模型都改写成 PyTorch。** 全部重写虽然方便调用，却会引入“框架移植误差”，反而损害复现可信度。
>
> 更正规的做法是保留两层：
>
> 1. **Reference/Fidelity track**：尽量使用官方框架和官方代码，证明原结果或行为可以复现。
> 2. **Unified PyTorch track**：在统一 adapter、split 和 scorer 下实现 PyTorch 版本，用于受控比较和以后开发新模型。
>
> 如果算力和维护成本不允许保留两套，至少应对核心模型保留官方实现作为锚点，并且在 PyTorch 移植通过 parity test 前不要称为 `exact`。未来创新网络可以建立在统一 PyTorch pipeline 上，但 benchmark 仍应保留官方锚点，避免“大家都被改错了却相互比较”。

（这里我的进一步意见是：所以我请你来审阅 ，其中的一个工作就是看看他是否复现正确，而不是在这里瞎弄。你看看你是否可以完成这个任务，我既然都做评估了 我也没必要偏袒某一方，架构直接用他们开源出来就行啊。改成torch只要不改错就行吧？？）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 可以，而且“核实模型是否复现正确”应当是审阅端的核心职责之一，而不能只让执行端自称 `exact`。我已经开始把本地实现与官方仓库逐项对照，目前得到的不是泛泛结论，而是以下可定位结果：
>
> 1. **ADT**：当前上三角 mask 的数值方向与 ADT 官方仓库一致。官方实现同样使用 `1 - lower_triangle`，Keras 中非零位置表示允许注意。因此原 A03 中“mask 可能写反”的怀疑需要降级，后续重点是验证 PyTorch 对全屏蔽行和 mask 广播的行为是否与 Keras 一致。
> 2. **VLAAI**：整体共享模块、卷积方向和 padding 结构接近官方实现，但当前 PyTorch 版使用 `nn.LeakyReLU()` 默认斜率 `0.01`，官方 TensorFlow `LeakyReLU` 默认斜率通常为 `0.3`；PyTorch `LayerNorm` 默认 `eps=1e-5`，官方 Keras 默认通常为 `1e-3`。这两项属于明确的实现差异，所以当前版本不宜继续称为 `VLAAI-exact`。
> 3. **HappyQuokka**：报告声称存在 `src/repro/happyquokka_reference.py`，但该文件没有提交到审阅分支，因此目前无法核实它是否保留了官方的 Pearson+L1 loss、Adam betas、epsilon、全局被试条件器和数据窗口逻辑。
> 4. **Ridge/CCA/FCNN/CNN/EEGNet**：`reference_baselines.py` 依赖的 `src/repro/mldecoders/` 没有提交，运行入口也缺失，因此只能审阅外围调用，无法核实核心线性、CCA 和网络定义。
>
> 把架构改成 PyTorch 本身没有问题，前提是通过具体复现验收。需要严格区分两件事：
>
> - **实现保真度**：PyTorch 代码是否仍然表达官方模型；
> - **迁移表现**：忠实模型在新数据集上是否仍然有效。
>
> 如果代码本身改错，低分不能解释为“原模型迁移性差”；只有先通过保真核验，跨数据集下降才是可信的迁移结论。审阅端会保持中立，不以分数高低决定哪个版本“正确”。

---

### DRAFT-A03：ADT attention mask 语义尚未被证明

- 严重程度：`Major`
- 置信度：中高
- 类别：实现正确性

**问题**

当前 ADT mask 将上三角标记为可见位置。需要与原始 ADT 实现确认其所谓 `anti-causal` 的准确含义以及 Keras/PyTorch mask 的 0/1 语义。

**影响**

如果移植时 mask 方向反转，ADT 的结构和结果均不可信；如果当前方向正确，也需要证据消除疑问。

**建议动作**

对照官方代码，增加 4x4 mask 单元测试，并报告当前 mask 与反向 mask 的验证结果。

**用户决定：**

> 

**用户注释或问题：**

> 这里确实需要补充一个这个实验，但是这个实验应该是本地跑的，相当于只是模型复现层面的问题。这里的正反难道对结果的影响很大？万一用错了结果反而提高了？我又应该怎么确定哪样的才是作者原来真正想要表达的？

**审阅端回复（2026-06-25）：**

> 正反 mask 可能显著改变结果，因为它决定某个时间点能看到过去、未来，还是不包含自身。如果错误版本允许使用本不应可见的未来 EEG，分数确实可能更高；但“分数更高”不能证明它正确。
>
> 判断作者真实意图应按证据优先级：
>
> 1. 原论文对 causal/anti-causal 和时间延迟的文字及公式；
> 2. 官方代码生成的 mask 数值；
> 3. 官方框架中 `1/0` 到底表示允许还是屏蔽；
> 4. 对固定 4 点序列打印注意力可见矩阵；
> 5. 做扰动测试：只改变未来输入，观察过去输出是否变化。
>
> 如果论文、代码和命名彼此矛盾，应将“官方代码行为”作为复现版本，同时把另一方向作为消融实验报告。正式结论应依据忠实版本，而不是依据得分较高的版本。该实验本地完成即可，但单元测试、结论和必要日志需要上传。

（这里你需要更明确执行端需要做的东西，你现在这样泛泛的说我觉得不好，任务一点都不具体 并不是一个可实施的任务条。）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 同意。A03 应改写成下面这条可直接验收的执行任务：
>
> **任务 ADT-PARITY-01：验证 PyTorch ADT 与官方 TensorFlow ADT 的 attention mask 和前向行为。**
>
> 执行端必须完成：
>
> 1. 固定官方来源为 `ruix6/ADT_Network` commit `e31ad252960e05856abb9c54669c3f65ebfa49ac`。
> 2. 新增 `tests/parity/test_adt_mask.py`，断言长度 4 时允许矩阵为：
>
>    ```text
>    0 1 1 1
>    0 0 1 1
>    0 0 0 1
>    0 0 0 0
>    ```
>
> 3. 分别在 TensorFlow 和 PyTorch 中记录最后一行“没有可见 key”时的输出，确认不会出现 NaN，且处理策略一致；若不一致，PyTorch 版必须模拟官方框架行为。
> 4. 使用同一小型固定输入，关闭 dropout，至少比较 feature extraction、第一层 attention 输出和最终输出的 shape、均值、标准差及最大绝对误差。
> 5. 增加未来扰动测试：仅改变输入的后半段，确认前半段哪些输出应当变化，证明时间方向与官方代码一致。
> 6. 输出 `artifacts/model_fidelity/adt_parity.json`，记录官方 commit、本地 commit、配置、参数量、测试结果和已知差异。
>
> 验收标准是 mask 数值、广播维度和时间方向一致、无 NaN、参数量差异有解释。若不能完成逐层数值转换，名称改为 `ADT-faithful-port`，不能继续无条件使用 `exact`。
>
> 这项测试不需要成为正文主图，但测试代码和一页汇总应开源，论文模型表中报告 `parity test passed/partial`。
---

### DRAFT-A04：VLAAI 缺少结构和时序方向 parity test

- 严重程度：`Major`
- 置信度：中
- 类别：实现正确性

**问题**

VLAAI 的卷积 padding、共享模块、时间方向和输出聚合方式需要与官方实现逐项对照。目前仅凭结构名称不足以证明 `VLAAI-exact`。

**影响**

可能出现代码能够运行，但感受野、因果方向或参数共享方式与原模型不同的情况。

**建议动作**

增加参数量、层级配置、输入输出 shape 和固定权重/固定输入 parity test。

**用户决定：**

> 

**用户注释或问题：**

> 这是什么意思？我不是很理解？能否详细讲讲，这个所谓的VLAAI-exact是我让ai通过原版VLAAI的代码中转的，确实会出现你说的这个 实现正确性的问题，而且也可能被审稿人提及，帮我看看应该如何测试，才能认为这个模型和原来的模型尽可能的相似。因为很多参数的变化以及实现架构的变化 让我们很难做到100%复现。这个应该如何解决呢？或者作出怎么样的实验才可以判断说可以了？

**审阅端回复（2026-06-25）：**

> `parity test` 的含义是：不给两个实现自由训练，而是尽量控制输入、权重和配置，检查它们是否执行同一个数学映射。
>
> VLAAI 建议分四层验证：
>
> 1. **静态结构一致性**：逐层列出 kernel、padding、激活、归一化、skip connection、模块是否共享、参数量和输入输出 shape。
> 2. **感受野与方向一致性**：对单点脉冲输入，观察哪些输出时间点发生变化，确认原版和 PyTorch 版使用相同过去/未来上下文。
> 3. **数值 parity**：如果能转换官方权重，将同一权重放入两种实现，对同一小输入比较逐层输出；允许浮点小误差，但不允许结构性偏差。
> 4. **行为 parity**：在固定的小数据、固定 split、固定 seed 下训练少量步骤，比较 loss 变化、输出分布和验证相关，而不是要求最终分数完全相等。
>
> 不要求跨 TensorFlow/PyTorch 达到每一个浮点数 100% 相同。真正要求的是：已知差异被列明，主要结构和时序行为一致，偏差不会改变模型定义。若无法转换权重，就应把名称改为 `VLAAI structural port`，不要使用容易暗示逐数值一致的 `exact`。审稿时主动提供 fidelity table 和 parity test，比笼统声称“完全复现”更可信。

（这一步应该和2是一个问题吧，说的都是 是否复现正确的意思？？这里我觉得确实需要这个对比，但是我不确定这个对比是否有必要开源出来？还是只要自己知道就行？如果只要自己知道的话，其实没必要弄那么多）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 是的，A02、A03、A04 和 A05 最终应合并为一个总问题：**模型来源、实现保真度与复现证据不完整**。ADT 和 VLAAI 只是其中两个具体子任务，不应各自成为一套彼此重复的审稿逻辑。
>
> 这部分需要开源，但不需要把每次调试产生的巨大中间张量都上传。最小公开范围是：官方仓库链接与固定 commit、本地实现和修改清单、自动 parity test、参数量/shape/感受野/关键数值差异汇总、一条可重跑命令，以及机器可读的通过结果。
>
> 对 VLAAI，执行任务应具体为：
>
> 1. 把 `LeakyReLU` 斜率和 `LayerNorm eps` 调整为官方 TensorFlow 值，或明确记录为何不同。
> 2. 使用官方预训练 `vlaai.h5`、SavedModel 或 ONNX 作为锚点。
> 3. 复现官方测试中的 DTU `S1_000` reconstruction correlation `0.244431957...`。
> 4. 在固定输入上比较官方模型与 PyTorch 模型的逐层输出；若权重转换困难，至少通过 ONNX 输出和最终相关系数进行对照。
> 5. 通过后称为 `VLAAI-faithful-port`；只有逐层数值误差也满足预先设定阈值时才使用 `exact`。
>
> 这些公开材料不仅服务审稿，也防止未来创新模型建立在一个悄悄改错的 baseline 上。一次做好，后续实验可以反复复用。

---

### DRAFT-A05：FCNN、CNN 和 EEGNet 的复现身份不清楚

- 严重程度：`Major`
- 置信度：高
- 类别：模型分类

**问题**

当前将这些模型描述为 `upstream-style` 或 `EEGNet-style`。这并不能说明它们复现了哪篇论文、哪个任务版本或哪些超参数。

**影响**

论文读者无法判断这些是：

- 忠实论文复现；
- 官方代码适配；
- 常见架构 baseline；
- 本地修改模型。

**建议动作**

给每个模型明确标注实现类型、引用来源、修改内容和不可比边界。

**用户决定：**

> 

**用户注释或问题：**

> 你这里的意见我认为很对，因为这个工作是完全奔着评估、发文、开源去的，因此最好每个步骤都应该清晰，并且有支撑，不是我随便说一个方法拿一个数据集就行的，而是应该尽量详细。 

**审阅端回复（2026-06-25）：**

> 同意。建议将模型统一分为四类并贯穿论文、配置和图表：`official reference`、`faithful port`、`architecture baseline`、`local exploratory model`。每个模型只能有一个明确身份，不能在不同文档中切换称呼。该意见建议接受。


（我觉得不用这样，我没看到有人做文章的时候 是这么麻烦的做的，你直接自己复现就行了，而且这些基础架构不少都是torch写的，很容易迁移，没那么精准。而且比的就是这些架构的可用性，和在不同数据集上的迁移性，如果只要稍微一点不对就完全不行，那说明迁移性就不行，只能适合他自己的数据）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 你的担心是合理的：正文不应被复杂的“身份标签”占满，也不需要对每个普通 CNN 做法医式复现。这里我收缩要求：
>
> - 有官方代码且主张复现某篇具体模型的 ADT、VLAAI、HappyQuokka，需要保真核验；
> - FCNN、CNN、EEGNet 若只是通用基础架构，可以直接标为 `architecture baseline`，无需声称复现某篇论文的精确结果；
> - NULL 是本地探索模型，直接标为本地模型即可。
>
> 最终正文只需要一张模型表，用一列写“官方实现、PyTorch 移植、通用 baseline 或本地模型”，再用一列概括主要修改，不会显得繁琐。
>
> 但“实现差异”和“迁移性差”仍不能混为一谈。例如把 VLAAI 的激活斜率从 `0.3` 改为 `0.01` 后性能下降，不能据此说 VLAAI 不能迁移，因为测试的已经不是相同模型。我们的标准不是要求每个浮点数都一样，而是不能改动足以改变模型定义的关键结构后仍归因于原方法。
---

### DRAFT-A06：NeuroConformer/NULL 的名称与上游配置不够清晰

- 严重程度：`Major`
- 置信度：高
- 类别：复现与报告

**问题**

本地模型以 `NULL` 命名，但实际对应 NeuroConformer family；本地默认配置与上游默认配置也存在差异，例如 positional encoding 设置。

**影响**

- 模型名称不可追踪。
- 容易将本地变体误认为官方 NeuroConformer。
- 无法判断结果来自架构、配置还是训练目标变化。

**建议动作**

采用可追踪名称，例如 `NeuroConformer-local-port`，并分别报告 upstream-parity 配置和本地改进配置。

**用户决定：**

> 

**用户注释或问题：**

> 不要说什么 NeuroConformer family，这个其实是我乱篡的一个架构，没有特定的名字，只是改成这个，你直接都叫NULL就行了。这个架构你不用多管，这是类似是测试架构，不需要过多寻找支撑和引经据典的。

**审阅端回复（2026-06-25）：**

> 接受你的事实纠正：NULL 不是需要忠实复现的 NeuroConformer，也不应继续被仓库描述成“NeuroConformer family 的本地集成”。因此原意见需要修改，而不是要求它寻找官方支撑。
>
> 建议将 NULL 明确标为 `local exploratory model`。内部阶段可以继续使用 `NULL`，但它不应进入“公开方法复现是否正确”的考核；只需要保证代码、配置和实验可复现。
>
> 如果最终论文是纯 benchmark，NULL 应放在探索性结果或补充材料中，不参与“现有方法公平排名”的核心结论。如果最终变成“benchmark + 新模型”，则需要给 NULL 一个可解释的正式名称、设计动机和消融。当前建议把 A06 从“上游复现问题”改为“身份与论文定位问题”，严重程度降为 `Minor` 或暂缓。

---

## 二、数据输入与输出规范

### DRAFT-A07：缺少完整的数据适配和预处理实现

- 严重程度：`Blocker`
- 置信度：高
- 类别：数据可复现性

**问题**

当前审阅包没有完整提供从公开数据到 `reference_splits` 的处理链，包括：

- 滤波；
- 重采样；
- 坏道处理；
- 通道映射；
- 包络提取；
- 对齐；
- 标准化；
- split 生成。

**影响**

模型使用的数据是否一致无法核查，也无法排除全数据归一化、时间错位或 split 泄漏。

**建议动作**

上传数据 adapter、预处理配置和可执行生成脚本；原始数据本身不必上传。

**用户决定：**

> 

**用户注释或问题：**

> 这里你说的也很对，另外你帮我确定一下 这部分最后在开源的时候是不是也需要上传呢？这里的数据切分应该是一样的。而且本质上来说 如果我不是为了刷指标，只是为了评估的话，这里最关键的一点其实是让每个架构的输入一样，并不一定说处理的方式有多先进，只需要必要的都存在就行，而且很多开源都是处理好的。这里你到时候也可以帮我核实一下（但是核实的话就需要告诉执行端 后面最好同时上传什么？）

**审阅端回复（2026-06-25）：**

> 这部分最终必须开源，但上传的是处理代码、配置、split 和校验信息，不是受许可证限制的原始数据。
>
> 你的核心判断基本正确：benchmark 的预处理不需要追求复杂，而应追求固定、必要、透明、无泄漏。不过“每个架构输入一样”需要更准确地表述：
>
> - 所有模型应从同一份 canonical EEG/envelope、同一 split 和同一 train-only normalization 出发；
> - 模型可以按照原论文要求切成 50、320 或 640 点窗口；
> - 不应为了表面统一而破坏原模型必要的时间上下文。
>
> 建议要求执行端后续同时上传：
>
> 1. 数据下载脚本或官方链接及版本号；
> 2. 每个数据集 adapter；
> 3. 预处理配置；
> 4. split 生成脚本和固定 split 文件；
> 5. normalization 参数由哪个 split 拟合的说明；
> 6. 通道映射与缺失通道策略；
> 7. 一个小型非敏感样例或 synthetic fixture；
> 8. 预处理输出 shape、时长和哈希检查；
> 9. 从预处理到模型输入的自动测试。
>
> 已经处理好的公开数据可以作为输入，但仍需锁定版本，并核实它是否对全数据做过归一化或使用了未知 split。

**审阅端进一步回复（2026-06-25，第二轮）：**

> 这里可以进一步压缩成执行端必须交付的五类材料：
>
> 1. `datasets/<dataset>/adapter.py`：将公开数据转换成统一的 recording 级 EEG、envelope 和 metadata。
> 2. `configs/datasets/<dataset>.yaml`：采样率、滤波、通道、包络提取和 normalization 设置。
> 3. `splits/<dataset>/<protocol>.json`：train/val/test 的 subject、recording 或 story 归属。
> 4. `manifests/<dataset>.csv`：匿名 ID、时长、shape、条件、语言、采样率和 split。
> 5. `tests/data/test_<dataset>_adapter.py`：检查长度对齐、split 不重叠、归一化只拟合训练集、窗口不跨 recording。
>
> 论文正文只汇总预处理原则和数据规模，详细文件放仓库及补充材料。它们应在大规模训练前完成，因为如果 split 或对齐有错，后面所有模型都要重跑。

---

### DRAFT-A08：缺少统一的数据集与 split manifest

- 严重程度：`Blocker`
- 置信度：高
- 类别：数据规范

**问题**

每个数据集缺少机器可读的 manifest，不能清楚核对 subject、recording、condition、language 和 split 的对应关系。

**影响**

- 难以验证 train/val/test 独立性。
- 无法进行条件、语言和跨数据集分析。
- 后续新增数据集时容易产生隐式协议差异。

**建议动作**

为每个数据集输出 manifest，至少包含：

- subject ID；
- recording ID；
- condition；
- language；
- split；
- duration；
- sampling rate；
- channel set；
- EEG/envelope shape；
- preprocessing version/hash。

**用户决定：**

> 

**用户注释或问题：**

> 这个部分我觉得和之前提到的对数据集的介绍应该是放在一起。即详细的介绍训练数据的构成。这里你说的加上每个ID有点夸张了，我只要保证训练验证测试不泄漏就行吧？？

**审阅端回复（2026-06-25）：**

> 论文正文不需要列出每一个 ID，正文只需汇总 subject 数、recording 数、总时长、条件和 split 规则。但仓库里的机器可读 manifest 最好保留每个 subject/recording 的归属，因为“保证不泄漏”必须有可检查的依据。
>
> 这不是为了增加论文篇幅，而是为了让程序自动断言：
>
> - 同一 recording 不跨 train/val/test；
> - subject-independent 协议中测试被试不进入训练；
> - 同一故事或重复刺激是否跨 split；
> - 条件和语言是否均衡；
> - 每次重跑是否仍使用同一 split。
>
> ID 可以是匿名 ID，也可以只保存公开数据已有的 ID。建议把 A08 保留为 `Blocker`，但明确“逐 ID 清单放仓库，不放正文”。

（这里你究竟想表达什么，我有点不太理解，你帮我重新想想，你说的这些是只服务于开源文件，还是服务于写的文章内容呢？如果是前者的话 可以后面再说）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 它同时服务文章和开源，但呈现层次不同：
>
> - **论文正文**：写每个数据集有多少被试、recording、时长、条件、语言，采用什么 split 原则，以及如何避免同故事、同 recording 或同被试泄漏。
> - **仓库 manifest**：保存每个匿名 ID 实际分到哪个 split，供程序自动检查。
>
> 逐 ID 表不需要放进论文，也不是为了增加工作量。它是“文章里声称无泄漏”的证据。这个部分不能完全拖到最后，因为当前结果是否可信正取决于 split；但第一阶段只需生成最小 manifest 和自动不重叠检查，不需要马上建设复杂数据管理系统。

---

### DRAFT-A09：线性模型和 CCA 可能产生跨 recording 的 lag 窗口

- 严重程度：`Major`
- 置信度：高
- 类别：数据边界

**问题**

Ridge 和 CCA 当前先将同一 split 的多段 recording 拼接，再构造 lag matrix。录音边界附近可能形成不存在的跨 recording 时间窗口。

**影响**

可能污染训练、验证和测试特征，尤其是在短 recording 较多时。

**建议动作**

对每段 recording 独立构造 lag 特征并裁剪无效边界，最后再拼接有效样本。

**用户决定：**

> 

**用户注释或问题：**

> 这个是不是有问题？这里的拼接就有点不太对吧？CCA和这个ridge需要多长的数据呢？我记得只需要延迟100ms左右就行，那现在数据集的每个输入是多长呢？拼接的话 是不是每个数据就不能打乱？？还是说需要加长CCA和ridge这几个方法的输入长度？？那这样的话验证和测试是不是也要加长 这和其他方法是不是就不一样了？这里你的想法是什么呢？

**审阅端回复（2026-06-25）：**

> 问题不在于 recording 本身不够长，而在于构造 lag 特征的顺序。当前配置写的是 `lags 0:50`；如果单位是 64 Hz 下的采样点，那么最多约为 `50/64=0.781` 秒，并不是 100 ms。这个单位需要执行端确认。
>
> 正确流程应是：
>
> 1. 对每一段 recording 单独生成 lagged samples；
> 2. 丢弃该 recording 起始或末尾没有完整上下文的位置；
> 3. 将生成好的有效训练样本拼接；
> 4. 训练样本此后可以打乱。
>
> 错误流程是先把 recording A 和 B 的原始时间序列首尾相接，再让 A 末尾与 B 开头组成一个虚假的 0.8 秒上下文。
>
> 不需要为了修正该问题而加长整段数据，也不需要让验证和测试的 recording 更长。各模型的上下文长度本来就可以不同，这是模型设计的一部分；需要统一的是 split 和最终 scorer。论文应报告每个模型实际使用的时间上下文，并可额外做“匹配上下文长度”消融，但不必强迫全部相同。
（这里我理解的意思是 需要训练的数据集大于0.8就行是吧？即我只要不拼接就能完成 就不影响对比？？即现在A和B需要接起来才能完成它现在的延迟构建？？）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 基本理解正确，但需要说得更精确：
>
> - 如果 `0:50` 确实表示 64 Hz 下的 51 个 lag 点，那么每段 recording 只要长于约 0.8 秒，就能产生有效 lag 样本；实际听觉实验 recording 通常远长于此。
> - A 和 B **不需要也不应该**先首尾连接来构造 lag。应分别在 A、B 内部构造 lag，丢弃各自边缘，再把已经合法的样本行合并训练。
>
> 当前代码确实存在先拼接再构造 lag 的路径：`fit_reference_ridge`、`evaluate_reference_ridge` 和 `fit_reference_cca` 都先调用 `concatenate_reference_split`。因此 A 的末尾与 B 的开头可能被当作连续 EEG。由于核心 `trim_valid_range` 没有提交，我还不能确认 `end_lag=50` 是包含还是不包含 50，但跨 recording 风险已经成立。
>
> 修正后不会破坏公平比较，也不需要加长其他模型输入。它只是删除不存在的边界样本。执行端还应报告修正前后分数差异，证明该 bug 对现有结论影响多大。

---

### DRAFT-A10：需要统一模型的测试输出接口

- 严重程度：`Major`
- 置信度：高
- 类别：输出规范

**问题**

不同模型可以保留原论文需要的窗口长度、目标形式和训练损失，但测试阶段没有统一输出格式。

**影响**

无法通过一个独立 scorer 对所有模型进行完全相同的评价。

**建议动作**

要求所有模型导出：

- subject ID；
- recording ID；
- prediction；
- target；
-采样率；
-有效时间索引；
-模型和 checkpoint 标识。

统一 scorer 只读取这些输出计算正式论文指标。

**用户决定：**

> 

**用户注释或问题：**

> 你的意思是说 这些模型都可以保留自己原来设置的loss函数，这里的需要窗口长度是什么意思？你指的是对输入的数据作增强的切分窗口长度吗？？然后只要统一输出就行？？你多次提到这个采样率还有什么各种ID，输出这个的意义是什么呢？我不理解，能否给我解释一下。

**审阅端回复（2026-06-25）：**

> 是的，模型可以保留原始 loss，因为 loss 是模型训练方法的一部分。训练 loss 不需要统一；正式测试指标需要统一。
>
> 这里的“窗口长度”是一次送给模型的连续时间上下文，例如：
>
> - FCNN/CNN：约 50 个采样点；
> - ADT/VLAAI：320 个采样点；
> - HappyQuokka/NULL：640 个采样点。
>
> 它与数据增强有关，但不是同一概念。数据增强描述“如何随机抽窗口”；窗口长度描述“一个输入样本覆盖多长时间”。
>
> 统一输出不是只保存一个最终分数，而是保存可重新评分的预测。各字段作用如下：
>
> - `subject ID`：按被试配对统计，防止某些被试被重复计权；
> - `recording ID`：避免跨录音拼接，支持条件和失败案例分析；
> - `sampling rate`：把采样点映射为实际时间，并检查不同数据是否对齐；
> - `valid time index`：不同模型可能因 lag、卷积或窗口边缘而不能预测完全相同的时间点；
> - `checkpoint/model ID`：保证分数能追溯到具体训练结果。
>
> 最理想的流程是：模型各自训练并生成连续预测，统一 scorer 在相同有效时间范围上重新计算每被试、每 recording 的正式指标。这样以后改变统计方法时不需要重训模型。
（这里的意思就是说 每个网络的输入都不一样？那这样的话前期分的所谓的split data又有什么意义呢？还是说这是另外一层？我有点不太理解）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 这是两个层次，split 仍然非常重要：
>
> 1. **数据划分层**：先决定哪些完整 recording/story 属于 train、val、test。所有模型必须遵守同一划分。
> 2. **模型取窗层**：在各自 split 内，Ridge 取约 0.8 秒 lag，CNN 取 50 点，ADT/VLAAI 取 320 点，HappyQuokka 取 640 点。窗口长度可以不同，但任何模型都不能从 test recording 取训练窗口。
> 3. **统一评分层**：不同模型在同一 test recording 上生成预测，然后在共同有效时间段用同一个 scorer 计算相关。
>
> 可以把 split 理解为“哪几本书用于训练、哪几本书用于考试”，窗口长度则是“每次读一页还是十页”。读法不同不影响考试材料必须隔离。
---

## 三、公平评价与统计

### DRAFT-A11：当前 Pearson 指标定义不统一

- 严重程度：`Blocker`
- 置信度：高
- 类别：指标一致性

**问题**

当前结果混合了：

- 拼接标量预测后的全局 Pearson；
- 每窗口 sequence Pearson；
- 非重叠分段 Pearson；
- CCA reconstruction correlation；
- canonical correlation。

**影响**

这些数值不能直接组成统一排行榜。即使都称为 correlation，计算对象和聚合顺序也不同。

**建议动作**

建立模型无关的统一测试 scorer。主指标固定为同一种 per-recording/per-subject 定义，其他指标作为辅助结果分表报告。

**用户决定：**

> 

**用户注释或问题：**

> 为什么会有这么多person，你能否告诉我都是哪些方法用了哪些person，以及按照最正规的比较方法来看，她们应该统一使用哪个person形式？

**审阅端回复（2026-06-25）：**

> 这里是 `Pearson` 相关系数，不是 person。当前出现多种形式，是因为各实现先产生不同形状的预测，再以不同顺序聚合：
>
> - Ridge、FCNN、CNN、EEGNet：目前倾向于将标量预测拼接后计算 `test_pearson`；
> - CCA：主表使用重建相关 `test_recon_corr`，另有 canonical correlation 和 match/mismatch accuracy；
> - ADT、VLAAI、HappyQuokka、NULL：通常先在每个窗口或分段上算 Pearson，再平均到 recording/subject；
> - canonical correlation 表示 CCA 隐变量间的相关，不是最终 envelope reconstruction Pearson，不能混入同一列。
>
> 最正规的主评价建议是：
>
> 1. 每个模型先对每段 test recording 生成对齐后的连续预测；
> 2. 不跨 recording 构造特征；
> 3. 在共同有效时间点计算 Pearson；
> 4. 得到每个 subject 的主分数，所有 subject 等权进入组统计；
> 5. 同时保留 per-recording 分数作为补充分析。
>
> 对重叠窗口的 sequence 模型，应先通过 overlap-add/平均重叠预测恢复连续时间序列，再用同一 scorer。主表只放 reconstruction Pearson；canonical correlation、窗口级相关和 match/mismatch accuracy 分别进入辅助表。聚合前可以对 `r` 做 Fisher `z` 变换，以改善统计性质。
（你的意思就是说这里目前ADT VLAAI这些架构使用的方法是对的是吧？？其他的都不太对？？ 另外也就是说这个部分 大家都应该固定，不然没法评估的对吧？？，如果是这样的话 那前面提到的什么拼接什么的，完全可以不用大家保持一样的输入，因为最后评估只要看恢复的总值就行。这里只考虑重建任务，先不考虑match任务）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 不是“ADT/VLAAI 正确、其他都错误”。目前各自内部的分数都有意义，但它们不是同一种正式主指标：
>
> - ADT/VLAAI 当前是窗口级 sequence Pearson 再平均；
> - Ridge/FCNN/CNN/EEGNet 当前更接近把标量预测拼起来算全局 Pearson；
> - CCA 主表是 reconstruction correlation，但还有 canonical correlation 等其他值。
>
> 因此所有模型都需要经过统一 scorer 重新计算，ADT/VLAAI 也不例外。你的后半句是对的：**模型训练输入不必完全相同，正式测试的任务定义、test recording 和评分公式必须相同。**
>
> 第一阶段只处理重建任务时，建议固定为：
>
> 1. 每个模型输出每段 test recording 上的时间对齐预测；
> 2. 重叠窗口通过平均恢复连续预测；
> 3. lag 或卷积边缘标记为无效；
> 4. 对所有模型取该 recording 的共同有效区间；
> 5. 计算 per-recording Pearson，再按预先固定规则汇总为 per-subject 分数；
> 6. 主表只使用这一列。
>
> 拼接问题仍需修复，因为虚假的 A-B 边界会污染模型输入；“最终统一评分”不能抵消错误特征。
---

### DRAFT-A12：训练制度混合导致架构比较不公平

- 严重程度：`Blocker`
- 置信度：高
- 类别：比较公平性

**问题**

当前主图混合了：

- subject-specific；
- pooled multi-subject；
- pooled + subject conditioning。

**影响**

结果差异无法归因于模型架构。subject conditioning 还引入了其他模型没有的 subject ID 信息。

**建议动作**

至少拆分为：

1. subject-specific；
2. pooled seen-subject；
3. pooled subject-conditioned；
4. leave-one-subject-out。

跨制度比较只能作为补充分析，不能作为直接架构排名。

**用户决定：**

> 

**用户注释或问题：**

> 我不太理解你说的这个跨制度比较，这是什么意思？？另外我确实疑惑的是 这里的评估到底应该包含哪些评估，首先我们只针对单数据集的情况，假设受试者分为ABCD，那么训练A测试A是一种，训练ABC测试D是一种，还有你说的这里的2 3分别代表什么我不理解，能否进一步解释？

**审阅端回复（2026-06-25）：**

> “制度”指模型在训练时能看到哪些被试、是否获得被试身份，而不是网络结构本身。用 A、B、C、D 四名被试说明：
>
> 1. **Subject-specific**：用 A 的训练 recording 训练，用 A 的验证 recording 选模型，用 A 的测试 recording 测试。B、C、D 分别各训练一个模型。不是拿 A 的全部数据训练后再测 A。
> 2. **Pooled seen-subject**：把 A、B、C、D 各自的训练 recording 合在一起训练一个共享模型，不输入 subject ID；测试仍是这四人的未见 recording。模型见过这些人的训练数据。
> 3. **Pooled subject-conditioned**：数据划分和 2 相同，但模型额外知道当前样本来自 A、B、C 还是 D。`g_con=True` 属于这一类。
> 4. **LOSO / unseen-subject**：训练 A、B、C，完全不使用 D 的训练数据，最后测试 D。然后轮换留出每一名被试。
> 5. **New-subject adaptation**：先训练 A、B、C，再给模型少量 D 的校准数据，测试 D 剩余数据。
>
> 你说的“训练 ABC、测试 D”就是 LOSO，前提是 D 没参与超参数选择和 normalization 拟合。
>
> 所谓跨制度直接比较，是把 subject-specific Ridge 与 pooled-conditioned deep model 放在一个柱状图里，然后把差异归因于架构。这不公平，因为后者使用更多人的数据和额外 subject ID。
>
> 建议论文至少分两个主 track：
>
> - **个性化/seen-subject track**：subject-specific 与 subject-conditioned 分开报告，回答“有个体数据时谁最好”；
> - **通用/unseen-subject track**：pooled unconditioned 与 LOSO，回答“新被试上谁泛化最好”。
>
> 同一 track 内比较模型，跨 track 只讨论应用场景，不做直接排名。

（我这里觉得你提到的5个场景很好，那我想知道 我这个五个场景都需要测试吗？？虽然这个评估是越全面越好的）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 不需要让每个模型、每个数据集都完整跑五种场景，那会形成非常庞大且重复的实验矩阵。建议分为三档：
>
> **正文核心，必须完成：**
>
> 1. `Subject-specific`：有该被试训练数据时，模型能达到什么水平。
> 2. `Pooled seen-subject, unconditioned`：一个共享模型服务已见被试，但不输入 ID。
> 3. `LOSO / unseen-subject`：完全未见的新被试泛化。
>
> **机制消融，只对支持该功能的模型完成：**
>
> 4. `Pooled subject-conditioned`：例如 HappyQuokka 和本地模型的 `g_con=True/False` 对照。
>
> **应用扩展，只选 3 至 4 个代表模型和 1 至 2 个数据集：**
>
> 5. `New-subject adaptation`：0、1、5、10 分钟校准数据。
>
> 在一张 5060 GPU 上，先完成 1、2、3 就已经能构成强有力的跨被试评价；4 用于解释被试 ID 带来的收益；5 可以在主线稳定后添加，不应阻塞第一批完整结果。
---

### DRAFT-A13：训练预算和看到的数据量不统一

- 严重程度：`Major`
- 置信度：高
- 类别：训练公平性

**问题**

当前存在 10、20、100 epoch、early stopping 和不同 `windows_per_sample`。单纯比较 epoch 也不能代表相同训练预算。

**影响**

某模型可能因为每 epoch 看到更多窗口、训练更久或参数更多而占优。

**建议动作**

统一报告：

- optimizer steps；
-训练窗口总数；
- epoch 数；
-参数量；
-运行时间；
-显存；
-checkpoint 选择规则。

正式比较可以采用统一数据预算或统一调参预算。

**用户决定：**

> 

**用户注释或问题：**

> 这里那张曲线图好像验证集的结果吧？？你的意思是说 我这里应该统一给 训练集的训练曲线，然后训练轮数应该统一为一个？？

**审阅端回复（2026-06-25）：**

> 当前主要曲线确实是验证相关；部分图也包含训练 loss。建议所有深度模型至少保存：
>
> - train loss；
> - validation loss；
> - validation 的统一 reconstruction metric；
> - learning rate；
> - optimizer step；
> - best checkpoint 位置。
>
> 但不建议机械地把所有模型训练轮数强制为同一个数。一个 epoch 对不同模型不等价：NULL 每个 recording 每 epoch 可随机采 20 个窗口，另一个模型可能遍历全部窗口；因此“100 epoch”未必是相同数据量。
>
> 更合理的是两层实验：
>
> 1. **Reference budget**：遵循原论文推荐 optimizer、调度器和 early stopping，检验忠实复现。
> 2. **Matched budget**：统一最大 optimizer steps、可见训练窗口数和调参次数，比较在相同资源下的能力。
>
> 所有模型使用验证集选 checkpoint，测试集只在最终 checkpoint 上评分。曲线不是最终公平性的依据，而是检查收敛和过拟合的审计材料。

（你帮我看看实现是否现实，我肯定不推荐你用原来的论文进行，因为数据都不一样了，没必要。而且我都进行架构对比了，本来就是刁难她们的，就需要考虑跨场景使用呀。）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 实现是现实的，而且我同意：最终主排名不应照搬每篇原论文在各自数据上的训练轮数。你的目标是统一压力测试，而不是复刻论文排行榜。
>
> 因此 A13 调整为：
>
> - 官方训练设置只用于确认模型没有被移植错，不作为最终主排名预算；
> - 主实验使用统一 benchmark budget；
> - 不强制相同 epoch，而是固定最大 optimizer updates、验证频率、early-stopping 规则和超参数搜索次数；
> - 各模型可以保留其必要 loss 和 optimizer，但每个模型获得相同数量的调参试验；
> - 对核心模型额外做 25%、50%、100% budget 的学习曲线，判断排名是否只是训练不充分造成。
>
> 一个可执行的初版可以是：先用 pilot 测出在主数据集上可承受的更新步数上限，例如每次训练最多 `20,000` optimizer steps，每 `500` steps 验证，连续若干次无改善早停；具体数值由执行端根据单卡耗时试跑后固定到 protocol，之后不得为某个模型单独增加预算。
>
> “跨场景刁难模型”在论文里可表述为 systematic stress testing：同一架构在被试内、共享模型、未见被试和跨数据集条件下接受一致评估。
---

### DRAFT-A14：深度模型缺少多随机种子结果

- 严重程度：`Major`
- 置信度：高
- 类别：稳定性

**问题**

多数 pooled deep model 当前只有一次训练历史，部分实现也没有完整保存 random seed。

**影响**

模型间的小差异可能只是初始化、batch 顺序或随机裁窗波动。

**建议动作**

正式结果至少运行多个 seed；如果算力受限，先对核心模型和主数据集做多 seed，并将单次结果标记为 preliminary。

**用户决定：**

> 

**用户注释或问题：**

> 我想知道这个多seed结果是不是必要的？如果是必要的那么使用多少seed才是规范的？我这里只有一个5060的GPU，你帮我看看如何实现是最合适的？

**审阅端回复（2026-06-25）：**

> 对深度模型，正式比较最好有多 seed；线性模型若求解确定，则通常不需要。没有统一法规规定必须几个，但常见最低标准是 3 个，5 个更稳健。1 个 seed 只能作为开发结果。
>
> 单张 GPU 建议采用分阶段策略：
>
> 1. 所有模型、所有数据集先跑 1 seed，用于查 bug 和筛选协议；
> 2. 在 2 个主 reconstruction 数据集上，对进入论文主表的核心深度模型跑 3 seeds；
> 3. 对排名接近、方差较大或拟作为关键结论的模型扩展到 5 seeds；
> 4. 外部泛化或超大模型可先 3 seeds，并把资源限制写入 limitations。
>
> seed 必须预先固定，例如 `[42, 2024, 2025]`，不能根据结果挑选。保存 seed、最佳 checkpoint、每 seed 的每被试结果。可以通过 mixed precision、早停、减少非核心模型的调参次数来控制成本。
>
> 最终统计需要区分 subject variability 和 seed variability；不能将多个 seed 简单当成更多独立被试。A14 建议保留为 `Major`，但执行端可以对非核心模型申请资源豁免。

（我觉得你说的有道理，确实应该补上随机数）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 该条可以与训练预算合并为“训练稳定性与资源公平性”。执行端应固定开发 seed 和正式 seed 列表，结果文件必须包含 seed；第一轮全矩阵允许 1 seed，进入正文的深度模型至少补 3 seeds，关键结论再补至 5 seeds。
---

### DRAFT-A15：缺少论文级统计检验

- 严重程度：`Major`
- 置信度：高
- 类别：统计分析

**问题**

当前主要报告 mean 和 standard deviation，尚未形成配对统计分析。

**影响**

无法判断模型差异是否稳定、是否具有实际效应，以及结论能否跨数据集保持。

**建议动作**

考虑报告：

- paired subject-level comparison；
- bootstrap 95% confidence interval；
- effect size；
-多重比较校正；
-排名稳定性；
- individual-subject distribution；
-失败案例。

**用户决定：**

> 

**用户注释或问题：**

> 我觉得你这里说的很有道理，但是具体选择哪些方法以及要说明什么问题这一点很关键。因为我这里是评估，而不是为了说明哪个方法相比于benchmark更好，因此这些检验可能需要相互做，而不是只需要1对n就行的，请你谈谈你的理解。

**审阅端回复（2026-06-25）：**

> 同意。评估论文不应默认选一个新模型作为中心做 `1 对 N`。更合适的是“整体差异 + 预先定义的成对比较”。
>
> 建议统计框架：
>
> 1. 每个数据集、每个协议先做 omnibus test，判断模型整体是否存在差异。若分布和样本量不适合参数检验，可使用 Friedman test；更完整时可用 mixed-effects model。
> 2. 之后做模型间 paired comparison，因为同一批 subject 上的结果是配对的。
> 3. 多模型两两比较会很多，使用 Holm 校正控制多重比较。
> 4. 同时报告效应量，而不只报告 `p` 值。非参数配对可以报告 rank-biserial correlation，参数条件满足时可报告 paired Cohen's `d_z`。
> 5. 使用 subject-level bootstrap 给每个模型及模型差值提供 95% CI。
>
> 主文不一定塞入所有两两比较。可以预先定义有科学意义的 contrast：
>
> - 线性 vs 轻量深度；
> - subject-specific vs pooled；
> - conditioned vs unconditioned；
> - transformer/conformer family vs CNN family；
> - 同一模型在不同数据集和条件下的稳定性。
>
> 完整 pairwise matrix 放补充材料。这样既不是围绕某个冠军，也避免十个模型产生大量缺乏解释意义的比较。
（我也觉得确实需要补充统计检验，但是如何补充以及补充到什么程度，确实是需要斟酌的地方）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 建议把统计方案固定为“主分析简洁、完整矩阵放补充材料”：
>
> **单数据集、单协议、所有模型都覆盖同一批被试时：**
>
> 1. 每个深度模型先在 seed 内得到每被试分数，再对 seed 求均值；seed 不作为额外被试。
> 2. 对 Pearson `r` 做 Fisher-z。
> 3. 先做 Friedman omnibus test。
> 4. 若整体显著，再做配对 Wilcoxon signed-rank 两两比较。
> 5. 使用 Holm 校正，并报告 rank-biserial effect size 和模型差值的 subject bootstrap 95% CI。
>
> **跨多个数据集或存在不完整模型覆盖时：**
>
> - 使用线性混合模型，例如 `Fisher-z ~ model * regime + dataset + modality + (1 | dataset:subject)`；
> - 重点检验 `model × regime` 和 `model × dataset`，因为论文真正关心的是排名是否随场景变化；
> - 数据集间不是同一批被试，不能假装做配对检验。
>
> **正文只放预先定义的五类比较：**
>
> - linear vs CNN；
> - CNN vs long-context deep model；
> - subject-specific vs pooled；
> - seen-subject vs LOSO；
> - in-dataset vs cross-dataset retention。
>
> 全部模型两两比较矩阵、校正 p 值和效应量放补充材料。这样足以达到论文级统计要求，又不会让正文变成 p 值清单。
---

### DRAFT-A16：标准差定义不统一

- 严重程度：`Minor`
- 置信度：高
- 类别：统计报告

**问题**

部分 baseline 使用 pandas 默认样本标准差 `ddof=1`，部分 deep model 使用 NumPy population standard deviation `ddof=0`。

**影响**

同一图中的误差线含义不同。

**建议动作**

统一使用预先定义的 SD、SE 或 95% CI，并在图注中明确说明。

**用户决定：**

> 

**用户注释或问题：**

> 这里应该使用哪个最规范？这里确实不应该不统一，这是最基本的要求。需要改正。

**审阅端回复（2026-06-25）：**

> 建议统一如下：
>
> - 表格描述个体差异：`mean ± sample SD`，即跨 subject 使用 `ddof=1`；
> - 图中的不确定性：优先使用 subject-level bootstrap 95% CI；
> - 图上同时显示每个 subject 的散点或配对线，避免只看柱形误差线；
> - 不建议只使用 SE，因为它容易让个体差异看起来很小；
> - seed 波动单独报告，不与 subject SD 混在一个误差条中。
>
> 因此 A16 应正式接受，并将所有旧汇总重新由统一统计脚本生成。

---

## 四、评估的全面性与多样性

### DRAFT-A17：当前正式比较仍主要局限于 envelope reconstruction

- 严重程度：`Blocker`
- 置信度：高
- 类别：任务覆盖

**问题**

Match/mismatch 目前只有局部 CCA 输出，真正 AAD 尚未接入统一模型和指标层。

**影响**

如果论文声称是广义 auditory decoding benchmark，仅一个重建任务不足以支撑全面性。

**建议动作**

可以选择以下论文定位之一：

1. 明确限定为深入的 envelope reconstruction benchmark；
2. reconstruction 为主任务，match/mismatch 为派生任务，AAD 为外部验证。

若采用第一种，必须通过更多数据集、泛化制度和条件分析补足深度。

**用户决定：**

> 

**用户注释或问题：**

> 我觉得这里可以将match和mismatch作为派生任务，即不仅要重建，是否还能在重建高的时候具有明确的区分能力？但是正如你之前所说的，现在这里的输入输出好像还没有进行很好的规范和统一，这样的就很难进行match评估，不然数据都不一样长。另外就是目前这些数据都是单人说话的声音，如果要做这个任务的话，需要人为构建多个对比数据，比如同时选10个片段，然后看他能否选对？？这样的话就更需要统一的数据了。

**审阅端回复（2026-06-25）：**

> 你的定位合理：match/mismatch 可以作为 reconstruction 的派生决策任务，检验“相关系数较高是否真的能支持候选语音选择”。
>
> 单人语音数据也可以构造 match/mismatch。标准二分类方式是：对一个 EEG test window，提供一个时间对齐的真实 envelope 和一个不匹配 envelope，比较模型重建结果与两者的相似度，判断真实候选是否更高。
>
> 需要统一：
>
> - EEG window 和两个 candidate envelope 的有效长度；
> - mismatch 的抽样规则；
> - 与真实窗口的最小时间间隔；
> - 是否来自同一故事、同一说话人或不同 recording；
> - window duration；
> - 每个 test window 使用多少次负样本抽样；
> - chance level 和统计单位。
>
> “同时选 10 个片段”属于 10-way retrieval，chance level 为 10%，可以作为更困难的辅助任务，但不建议代替标准二元 match/mismatch。推荐：
>
> 1. 主结果：binary match/mismatch，多个固定 window duration；
> 2. 稳健性：重复多个预先生成的 mismatch 列表；
> 3. 补充实验：5-way 或 10-way retrieval。
>
> 它不是真正的 AAD，因为没有同时存在且被试选择性注意的两个说话人。统一连续预测输出后，不同模型输入窗口可以不同，但 scorer 会在共同有效测试片段上构造相同候选。
（如果是这样的话，我目前唯一的疑问就是 那CCA这类方法是不是就需要和普通的网络存在修正？因为我看到前文说CCA好像需要拼接什么东西才可以继续？）

**审阅端进一步回复（2026-06-25，第二轮）：**

> CCA 不需要为了 match/mismatch 改成神经网络，也不需要改变它的数学模型。需要修正的是数据边界和统一 scorer：
>
> 1. CCA 在每段 recording 内构造 lag 和连续重建，不能先把 A、B 原始时间序列连接。
> 2. CCA 输出的重建包络与神经网络输出一样，交给同一个 match/mismatch scorer。
> 3. scorer 对所有模型使用同一批 EEG 窗口、同一 matched candidate 和同一预生成 mismatch candidate 列表。
> 4. CCA 的 canonical correlation 可以保留为辅助指标，但不能代替主表中的 reconstruction Pearson。
>
> 因此 CCA 与普通网络不需要拥有相同训练窗口，只需要在相同 test recording、相同有效区间和相同候选列表上作决策。当前分支缺少 CCA 核心依赖代码，执行端需先补齐后才能核验其 lag、PCA、CCA 和 ridge readout 的具体实现。
---

### DRAFT-A18：缺少完整的跨被试评价

- 严重程度：`Major`
- 置信度：高
- 类别：泛化评价

**问题**

当前没有统一完成：

- subject-specific；
- pooled seen-subject；
- leave-one-subject-out；
- new-subject adaptation。

**影响**

无法回答模型是记住个体特征，还是能泛化到新受试者。

**建议动作**

为核心模型建立至少 subject-specific、pooled seen-subject 和 LOSO 三层协议。

**用户决定：**

> 

**用户注释或问题：**

> 你这里的LOSO是什么意思？？这又和前文的12问题我的注释是一样的。

**审阅端回复（2026-06-25）：**

> LOSO 是 `Leave-One-Subject-Out`，即留一被试法。若有 A、B、C、D：
>
> - 第 1 折：训练 BCD，测试 A；
> - 第 2 折：训练 ACD，测试 B；
> - 第 3 折：训练 ABD，测试 C；
> - 第 4 折：训练 ABC，测试 D。
>
> 被留出的 subject 不能参与训练、normalization 拟合或 checkpoint 选择。验证集可从训练 subjects 中再留出 recording 或 subject，严格版本采用 nested split。
>
> LOSO 回答“模型能否直接泛化到从未见过的新被试”。它与 pooled seen-subject 的区别是，后者测试的人在训练阶段已经出现过。A18 与 A12 可以在正式版中合并成一个“subject evaluation regimes”大问题，避免重复。

---

### DRAFT-A19：尚无真正的跨数据集迁移实验

- 严重程度：`Major`
- 置信度：高
- 类别：跨数据集泛化

**问题**

目前 Weissbart、Etard 等数据集主要是分别训练、分别测试。这不是 train A → test B 的跨数据集迁移。

**影响**

不能据此得出模型跨数据集泛化能力的结论。

**建议动作**

增加明确协议：

- train Weissbart → test Etard；
- train Etard → test Weissbart；
- pooled train → held-out dataset；
-是否允许目标域 normalization/adaptation。

**用户决定：**

> 

**用户注释或问题：**

> 后面确实需要做跨数据集的转化，帮我看看怎么设计，以及作为一个评估文章，我现在觉得数据集的个数是不够的。你觉得需要多少最合适，以及实验怎么分布？

**审阅端回复（2026-06-25）：**

> 数据集不是越多越好，关键是每个数据集承担明确问题。对于目标为中等影响力期刊的评估论文，我建议争取 4 个正式 EEG 数据集，5 个会更有说服力；Hugo 若主要是 tutorial/dev 数据，可不计入正式核心数量。
>
> 推荐角色分配：
>
> 1. Weissbart：主 reconstruction 数据集；
> 2. Etard：多条件、语言分析；
> 3. SparrKULee/KUL：大规模或 AAD/challenge 生态；
> 4. DTU/Fuglsang：外部泛化；
> 5. 可选额外 AAD 数据或中文数据；
> 6. MEG 建议作为扩展，不要在第一版中承担核心公平排名。
>
> 实验不需要穷举所有数据集两两迁移，可按研究问题组织：
>
> - **表 1：in-dataset reconstruction**，所有核心模型在每个适用数据集内训练测试；
> - **表 2：subject regimes**，在 1 至 2 个主数据集做 subject-specific、pooled、LOSO；
> - **表 3：cross-dataset zero-shot**，只选预处理和任务兼容的方向；
> - **表 4：pooled-source → held-out dataset**；
> - **表 5：match/mismatch 或 AAD**；
> - **表 6：语言、噪声、说话人条件分析**。
>
> 跨数据集时应关闭 subject conditioning，并预先规定通道交集/映射、目标域是否允许 normalization、刺激 envelope 是否用相同方法提取。建议至少有 3 个文章级数据集完成 in-dataset，其中至少 1 个做外部 zero-shot；达到 4 至 5 个数据集后，不应再为了数字增加低质量数据。

---

### DRAFT-A20：语言和实验条件尚未形成实际分析

- 严重程度：`Major`
- 置信度：高
- 类别：条件分析

**问题**

Etard 包含语言和多种听觉条件，但当前主要报告整体数据集均值，语言、噪声和 competing-speaker 影响仍停留在计划层面。

**影响**

无法回答模型对语言变化和实验难度是否稳健。

**建议动作**

按照 language、noise、speaker condition 分层报告，并使用配对或混合效应分析。

**用户决定：**

> 

**用户注释或问题：**

> 这里你的建议是什么？因为很多数据集，如果不统一各分析各的，我觉得有点难弄，你帮我看看如何更好的组织。

**审阅端回复（2026-06-25）：**

> 建议采用“统一核心层 + 数据集特有层”，而不是要求每个数据集支持完全相同的条件。
>
> **统一核心层**对所有数据集都要求：
>
> - reconstruction；
> -同一主 Pearson scorer；
> -同一 subject 聚合；
> -同一统计输出格式；
> -统一 model/regime/seed 标签。
>
> **数据集特有层**只分析该数据真正拥有的因素：
>
> - Etard：language、noise、competing speaker；
> - KUL/SparrKULee：attended/unattended、AAD；
> - DTU/Fuglsang：外部泛化、听力或实验条件；
> - MEG：modality。
>
> 结果表采用 long-format schema，例如每行包含 `dataset, model, task, regime, subject, recording, condition, language, seed, metric`。不支持的字段留空，而不是强行伪造统一条件。
>
> 论文结构则先报告跨数据集共同结论，再用专门小节回答语言、噪声和 AAD。这样既有统一主线，也不浪费各数据集独有价值。

---

### DRAFT-A21：核心公开数据集覆盖仍不足

- 严重程度：`Major`
- 置信度：高
- 类别：数据集覆盖

**问题**

SparrKULee/KUL、DTU/Fuglsang 和真正 AAD 数据尚未进入同一 adapter 和评价层。

**影响**

当前结果更适合作为多数据集 reconstruction prototype，还不足以声称全面覆盖 auditory EEG decoding 领域。

**建议动作**

建立分层接入顺序：

1. 完成当前三套数据的统一 reconstruction；
2. 接入 DTU/Fuglsang 做外部泛化；
3. 接入 SparrKULee/KUL 做大规模 benchmark；
4. 接入真实 attended/unattended 数据做 AAD；
5. MEG 作为扩展或补充实验。

**用户决定：**

> 

**用户注释或问题：**

> 同问题19

**审阅端回复（2026-06-25）：**

> 与 A19 联动处理。A19 解决“如何做跨数据集”，A21 解决“选哪些数据集、每个承担什么角色”。正式审阅时两条可以保留为不同 issue，也可以合并为一个 dataset coverage and transfer plan。
>
> 当前倾向是：先完成 3 个数据集的统一地基，再接入第 4 个外部泛化数据和第 5 个真实 AAD/大规模数据。只有当 adapter、scorer 和 manifest 固定后才批量扩展，否则新增数据会放大返工。

---

## 五、候选优先级

### 建议第一批正式立项

- `DRAFT-A01`
- `DRAFT-A02`
- `DRAFT-A07`
- `DRAFT-A08`
- `DRAFT-A09`
- `DRAFT-A10`
- `DRAFT-A11`
- `DRAFT-A12`

这些问题决定当前结果是否具有可复现性和可比较性。

### 建议第二批立项

- `DRAFT-A03`
- `DRAFT-A04`
- `DRAFT-A05`
- `DRAFT-A06`
- `DRAFT-A13`
- `DRAFT-A14`
- `DRAFT-A15`

这些问题决定模型排名和论文结论是否可信。

### 建议实验扩展阶段立项

- `DRAFT-A17`
- `DRAFT-A18`
- `DRAFT-A19`
- `DRAFT-A20`
- `DRAFT-A21`

这些问题决定文章的完整度、广度和投稿上限。

### 建议随统一工作同步修正

- `DRAFT-A16`

---

## 六、用户总体意见

**你认为这篇文章应采用的主要定位：**

> 

**你希望优先解决的问题：**

> 

**你认为可以因算力、数据或时间限制暂缓的内容：**

> 

**你希望进一步向审阅端询问的问题：**

> 

**是否允许将确认后的意见上传 GitHub：**

> 暂不允许。等待逐条确认。

---

## 七、阶段性共识与下一轮需要确认的决策

### 7.1 当前较合理的论文定位

> 这篇文章不应只是“把十个模型在若干数据上跑一遍”，而应定位为：**面向听觉神经解码的多数据集、分协议基准评估研究**。核心任务以 EEG 语音包络重建为主，在统一输出与评分协议下比较模型复现质量、被试内表现、跨被试泛化和跨数据集迁移；由包络重建派生的 match-mismatch 作为第二任务，真正的多说话人 AAD 则在具备相应刺激和注意标签的数据集上单独评估。
>
> 这种定位允许不同模型在不同数据集上各有优势。文章的价值不依赖某个新模型必须全面胜出，而在于解释：**什么模型在什么数据条件、训练协议和泛化场景下有效，以及为什么。**

（这里不应该只局限在EEG 因为我后期可能要找更多的MEG数据进来，我觉得现在的数据太少了）

**审阅端进一步回复（2026-06-25，第二轮）：**

> 同意，论文总范围应改为 **auditory neural signal decoding**，而不是标题和主张都锁死在 EEG。但 MEG 应作为同一研究中的独立模态层，而不是直接与 EEG 混成一张总排行榜。
>
> 推荐结构：
>
> - EEG reconstruction 主层：多个公开 EEG 数据集；
> - MEG reconstruction 扩展层：MEG-SCANS、MEG-MASC 或后续确认的数据；
> - 跨模态分析：比较模型排名、相对增益和 transfer retention，而不是直接比较 EEG 与 MEG 的绝对相关系数；
> - AAD 层仍主要依赖具有 competing-speaker 和注意标签的 EEG 数据。
>
> 如果最终只有一个 MEG 数据集，它只能证明“可扩展到 MEG”；若希望声称 MEG 内部的跨数据集泛化，至少需要两个任务兼容的 MEG 数据集。统一 adapter、输出 schema 和 scorer 从现在就保留 `modality` 字段，后续加入 MEG 不需要推翻 EEG 流程。

### 7.2 推荐的工作顺序

1. **仓库与模型身份清理**  
   补齐实际执行代码、配置、依赖和入口；将每个方法标记为 official reference、faithful port、architecture baseline 或 local exploratory model。

2. **统一数据层**  
   建立 canonical EEG/envelope 表示、数据集 adapter、划分 manifest、训练集归一化记录、通道与采样率映射，以及防止数据泄漏的自动测试。

3. **统一输出和评分层**  
   要求模型输出按 recording 对齐的连续预测或可无歧义恢复的窗口预测；由同一个 scorer 计算 reconstruction Pearson、match-mismatch 和后续任务指标。

4. **复现正确性验证**  
   对线性模型检查 lag 和边界，对 ADT 检查 mask，对 VLAAI 等移植模型检查结构、感受野、数值或行为一致性，并保存证据。

5. **核心比较实验**  
   先完成 1 至 2 个核心数据集上的被试内、pooled seen-subject、subject-conditioned 和 LOSO 协议，再扩展至更多模型和数据集。

6. **泛化、任务扩展与统计分析**  
   开展预先选择的跨数据集迁移、二分类 match-mismatch、真实 AAD、语言或噪声条件分析；最后统一完成多重比较、效应量、置信区间和稳定性分析。

### 7.3 当前仍需用户决定

**论文类型：**

> 纯 benchmark，还是 benchmark 加一个新模型。当前更稳妥的起点是纯 benchmark；NULL 可先作为探索模型保留，不让它承担论文成立与否。

**核心数据集：**

> 建议先确定 3 个能够稳定获得并完成统一预处理的数据集，再选择第 4 个作为外部泛化数据集、第 5 个作为真实 AAD 或语言扩展数据集。

**核心模型集合：**

> 不必一开始机械凑满十个。优先覆盖线性、CCA、浅层神经网络、CNN、VLAAI、ADT/Transformer、HappyQuokka 等有代表性的模型族；只有代码来源和复现身份明确后才进入正式主表。

**算力与随机种子预算：**

> 建议确定“全部模型最低 1 个种子、核心深度模型至少 3 个种子、关键结论必要时 5 个种子”的分层预算。

**MEG 是否进入主研究：**

> 当前建议把 MEG 视为模态外推扩展，而不是与 EEG 混在同一主排名中。只有建立独立的传感器映射和模态协议后，才讨论共享模型或跨模态迁移。

### 7.4 审阅状态

> 本文件仍是本地候选审阅稿。以上回复用于和用户逐条讨论，不代表意见已经接受，也没有上传 GitHub。用户确认后的条目才会转换为执行端 issue；执行端可以对每条意见提交“已完成、部分完成、无法完成或提出反驳”的证据，审阅端再根据代码、配置和结果决定关闭、保留或修订。

---

## 八、将原 21 条意见合并为 5 个可执行工作包

原有 `DRAFT-A01` 至 `DRAFT-A21` 保留，便于追踪讨论来源；真正交给执行端时，不建议继续创建 21 个相互重叠的任务，而应合并为以下工作包。

### WORKPACKAGE-01：审阅包完整性与模型身份

**合并来源：** A01、A02、A05、A06。

**目标：** 审阅者能够从提交内容中运行并追踪每一个进入结果表的模型。

**执行端必须提交：**

- 所有被报告模型的实际实现代码；
- 可运行入口、环境文件和最小配置；
- 官方论文、官方仓库、上游 commit 和许可证；
- 本地修改清单；
- 模型身份：官方实现、忠实移植、通用架构 baseline 或本地探索模型；
- 每条结果对应的 config、seed、checkpoint 和原始 per-subject 指标。

**验收条件：**

- 报告里引用的文件全部真实存在；
- 从一个小型 fixture 可以跑通数据到结果导出的完整流程；
- NULL 不再与 NeuroConformer 或任何官方模型混淆；
- 未通过复现核验的模型不使用 `exact`。

### WORKPACKAGE-02：核心模型保真度核验

**合并来源：** A03、A04，以及 A02/A05 中与具体模型复现有关的部分。

**目标：** 排除“代码移植错误被误解释为迁移失败”。

**执行端必须提交：**

- ADT mask、参数量、shape、时间方向和固定输入测试；
- VLAAI 激活斜率、LayerNorm epsilon、共享模块、padding 和官方预训练输出测试；
- HappyQuokka 官方 loss、optimizer、conditioning 和数据取窗对照；
- FCNN/CNN/EEGNet 的具体架构定义或明确声明为通用 baseline；
- `model_fidelity_table.csv` 和每个核心模型的 parity JSON。

**验收条件：**

- ADT 官方 mask 行为得到测试确认；
- VLAAI 能复现官方 DTU 固定测试，或明确降级为 structural port；
- HappyQuokka 实现文件可审阅；
- 关键差异均能追踪，且不会用高低分替代正确性判断。

### WORKPACKAGE-03：统一数据、边界、输出与评分

**合并来源：** A07、A08、A09、A10、A11、A16。

**目标：** 所有重建模型在同一 test recording、共同有效时间点和同一公式下评分。

**执行端必须提交：**

- dataset adapter、dataset config、split JSON 和 manifest；
- train-only normalization 证据；
- recording 内部构造 lag/window 的实现；
- 每个模型的 recording-level 连续预测导出；
- overlap averaging、valid mask 和统一 scorer；
- 统一的 subject-level 结果 schema；
- 一份修复前后结果差异报告。

**验收条件：**

- 不跨 recording 构造 lag 或窗口；
- split、模型窗口和 scorer 三层职责分离；
- 所有模型的 reconstruction 主指标由同一 scorer 生成；
- SD、CI 和聚合单位统一；
- 当前混合的窗口 Pearson、全局 Pearson 和 CCA canonical correlation 不再进入同一主列。

### WORKPACKAGE-04：训练制度、预算、随机性与统计

**合并来源：** A12、A13、A14、A15、A18。

**目标：** 模型差异能够归因于架构和场景，而不是训练制度、额外被试信息或随机波动。

**正文核心协议：**

- subject-specific；
- pooled seen-subject without ID；
- LOSO unseen-subject。

**机制与应用扩展：**

- subject-conditioned 仅用于支持该功能的模型消融；
- few-shot adaptation 只选择代表模型和数据集。

**执行端必须提交：**

- 固定 optimizer-step budget、验证频率、early stopping 和调参次数；
- 训练/验证 loss、统一验证指标、学习率、optimizer steps；
- seed 列表与每 seed、每 subject 结果；
- 预注册统计脚本；
- 运行时、显存、参数量等资源指标。

**验收条件：**

- 不再把 subject-specific、pooled 和 conditioned 分数直接作为架构总排名；
- 主表深度模型至少 3 seeds；
- 单数据集使用 Friedman + 配对 Wilcoxon + Holm + effect size；
- 跨数据集和不完整覆盖使用 mixed-effects analysis；
- seed 不被当作额外被试扩大样本量。

### WORKPACKAGE-05：任务、数据集、模态与论文叙事

**合并来源：** A17、A19、A20、A21，并吸收 A12/A18 的泛化问题。

**目标：** 让论文由研究问题驱动，而不是无边界增加模型和数据。

**建议任务层级：**

1. **主任务：Envelope reconstruction。**
2. **派生任务：Binary match-mismatch；5/10-way retrieval 作为补充。**
3. **独立注意力层：真实 competing-speaker AAD，只在兼容数据和模型上完成。**
4. **可选任务：Mel reconstruction，不阻塞第一篇文章。**

**建议模态层级：**

- EEG 和 MEG 均进入 auditory neural signal 总框架；
- 绝对指标按模态分别报告；
- 跨模态比较模型排名、相对增益和迁移保持率；
- MEG 不与 EEG-AAD 混为一个排行榜。

**验收条件：**

- 每个数据集在论文中只有明确角色；
- 每个模型只进入兼容任务；
- 不暴力遍历所有 train-test 数据集组合；
- 图、表、统计检验和正文结论一一对应。

---

## 九、对执行端当前报告的审阅结论

### 9.1 报告身份

远端 `audit/reproduction-note` 最新 commit 仍为 `a43831b83598c82520be320c21b56e92b73b7dcd`。仓库没有提交编译后的 PDF，执行端实际提供的是：

- `docs/reproduction_audit_note.tex`；
- `docs/CURRENT_EVALUATION_STATUS.md`；
- 汇总 CSV 和 PNG；
- 部分模型代码。

本地 `paper/main.pdf` 是较早的论文框架草稿，不是执行端复现报告。

### 9.2 报告已经做对的部分

- 明确承认当前结果不是最终 apples-to-apples leaderboard；
- 区分了 subject-specific、pooled 和 subject-conditioned；
- 记录了 requested epochs、completed epochs 和 best epoch；
- 保存了若干训练曲线与 per-dataset 汇总；
- 正确指出 Ridge 仍可能是强基线，不能只追求深度模型高分；
- 已开始讨论 HappyQuokka `g_con=True/False` 的比较边界。

### 9.3 当前不能接受为论文基石的原因

1. **报告引用了不存在的文件。**  
   `happyquokka_reference.py`、baseline/exact runner 和 `src/repro/mldecoders` 依赖没有提交，导致关键结果无法复核。

2. **报告内容落后于当前分支。**  
   当前分支已经加入 NULL/NeuroConformer 相关结果和路线文档，但 `reproduction_audit_note.tex` 没有同步纳入，报告与分支实际内容不一致。

3. **主表混合训练制度与指标。**  
   subject-specific baseline、pooled exact port 和 conditioned model 放在同一图中，且 Pearson 聚合方式不同，当前排名不能归因于模型架构。

4. **Ridge/CCA 存在跨 recording lag 风险。**  
   当前外围代码先拼接 recording，再构造 lag；核心依赖未提交，无法排除虚假边界。

5. **`VLAAI-exact` 尚不满足 exact。**  
   激活斜率和 LayerNorm epsilon 与官方默认实现不同，尚未运行官方预训练 parity test。

6. **训练曲线聚合具有误导性。**  
   subject-specific 模型在不同 epoch 存活的被试数不同，但曲线脚本对“仍有该 epoch 的 history”动态求平均；曲线后段可能只剩少数被试，标准差甚至变成 0，却仍标记原始总 history 数。该图不能用于比较收敛速度。

7. **结果图还不是论文级图。**  
   当前主要是 mean ± SD 柱状图，缺少被试点、配对线、置信区间、协议分面和统计标记；长模型标签也不适合最终版式。

8. **缺少原始 per-subject × seed × recording 结果。**  
   仅凭均值、SD 和 N 无法重做统一评分、配对统计或错误分析。

### 9.4 当前报告的正式状态

> **结论：可作为开发进度说明和问题清单，不可作为论文结果章节的基石。**

现有数值不需要删除，但应统一标记为 `preliminary / non-comparable`。在 WORKPACKAGE-01 至 WORKPACKAGE-03 通过前，不建议继续大规模增加模型或数据集，否则会在错误地基上积累更多需重跑的结果。

---

## 十、下一阶段执行顺序

### Gate 0：补齐审阅包

- 补交报告中引用的全部代码、入口和依赖；
- 导出 per-subject、per-recording、per-seed 原始结果；
- 重新生成与当前 commit 一致的报告。

### Gate 1：修复模型保真度和数据边界

- 完成 ADT、VLAAI、HappyQuokka parity；
- 修复 Ridge/CCA 跨 recording lag；
- 明确 NULL 的本地模型身份。

### Gate 2：完成统一 reconstruction scorer

- 所有模型导出时间对齐预测；
- 同一 test recording 共同有效区间评分；
- 生成统一长表结果。

### Gate 3：小规模冻结协议

- 选择 2 个核心重建数据集；
- 选择 6 至 8 个核心模型；
- 每模型先跑 1 seed；
- 同时生成论文计划中的全部主图草图；
- 审阅端确认结果 schema、图形和统计可以闭环。

### Gate 4：正式核心实验

- 扩展到 subject-specific、pooled seen-subject 和 LOSO；
- 核心深度模型运行 3 seeds；
- 完成统计分析和效应量。

### Gate 5：数据、任务和模态扩展

- 增加 SparrKULee、外部 EEG 数据和 MEG；
- 增加 match-mismatch；
- 最后加入真实 AAD、few-shot、Mel 或 gaze 等扩展。

> 只有 Gate 3 通过后才适合让执行端“持续跑满当前架构所需结果”。这能避免现在零散运行、之后因评分或 split 改变而全部返工。

---

## 十一、本轮生成的配套材料

### 执行端报告审阅版

- Markdown：`reviews/execution_report_a43831b_review_zh_2026-06-25.md`
- PDF：`reviews/execution_report_a43831b_review_zh_2026-06-25.pdf`

说明：执行端没有提交原始 PDF。本文件根据其 `reproduction_audit_note.tex`、CSV、PNG 和代码生成，明确区分原报告陈述与审阅端判断。

### 论文执行蓝图

- Markdown：`paper/benchmark_execution_blueprint_zh_2026-06-25.md`
- PDF：`paper/benchmark_execution_blueprint_zh_2026-06-25.pdf`

蓝图已经固定：

- 论文控制性问题；
- 引言逐段功能；
- 数据集和模型介绍方式；
- 主任务、派生任务和扩展任务；
- subject、dataset 和 modality 协议；
- 模型-任务-协议实验矩阵；
- 8 张正文图与 5 张正文表；
- 统计分析；
- 结果文件 schema；
- 执行门禁。

这些材料仍只保存在本地，没有上传 GitHub。
