# AR-20260625-161300-a43831b Review Package

本轮审阅对象是 `DDDingJin/eeg-lky-workspace` 的 `audit/reproduction-note` 分支，锁定 commit：

`a43831b83598c82520be320c21b56e92b73b7dcd`

本目录不是单纯的细节问题清单，而是一个“主线蓝图 + 具体审稿意见”的组合包。执行端后续修改时，需要先确认整体 benchmark 论文路线是否服从蓝图，再逐项处理具体可复现性、模型保真度、数据边界、指标统计和论文叙事问题。

## 建议阅读顺序

1. `benchmark_execution_blueprint_zh_2026-06-25.pdf`
2. `execution_report_a43831b_review_zh_2026-06-25.pdf`
3. `candidate_full_benchmark_review_a43831b_2026-06-25.md`
4. `issues.json`

Markdown 版本与 PDF 版本内容对应。PDF 适合人工快速阅读，Markdown 适合执行端引用、拆任务和写 response。

## 文件含义

`benchmark_execution_blueprint_zh_2026-06-25.md/pdf` 是长期执行蓝图。它定义了这篇评估类论文的核心问题、任务层级、数据集角色、模型集合、训练公平性、统计原则、图表规划、实验矩阵和阶段 gate。执行端不应只围绕局部报错修补，而应把后续实现对齐到这个主线。

`execution_report_a43831b_review_zh_2026-06-25.md/pdf` 是针对目标 commit 的具体审稿意见。核心判断是：当前报告更接近开发进展记录，还不能直接作为论文结果段落。主要阻塞点包括审阅包缺失、模型身份不完整、VLAAI/HappyQuokka/ADT 等关键复现点仍需核验、Ridge/CCA 边界与评分方式未统一、当前图表不可作为跨模型结论。

`candidate_full_benchmark_review_a43831b_2026-06-25.md` 是更长的候选意见和工作包整理。它保留了从草稿问题到 5 个 workpackage 的归并过程，便于执行端理解为什么这些问题不是孤立细节，而是共同决定论文是否能成立。

`issues.json` 是结构化工作包登记表。执行端可以把它作为任务拆分入口，并在后续 `fix_manifest.json` 或 `review_response.md` 中逐项回应。

## 执行端处理原则

1. 不要直接把这个 review 分支合并进执行分支；先阅读、拆任务，然后从目标 commit 或当前执行基线创建 `fix/` 分支。
2. 每个关闭的问题都需要证据，包括代码路径、输出文件、运行命令、关键指标或复现实验说明。
3. 如果某条意见被认为不适合执行，需要在 response 中解释原因，而不是静默跳过。
4. 先处理 Gate 0 到 Gate 2 的可复现性、模型身份和统一评分问题，再推进完整 benchmark 结果。
5. 蓝图中的长期问题可以分阶段完成，但任何阶段性结果都要说明它覆盖了蓝图中的哪些任务、数据集、模型和指标。

## 本轮最低验收线

执行端下一轮至少应提交：

1. 一个 `review_response.md`，逐项回应 5 个 workpackage。
2. 一个 `fix_manifest.json`，说明基于哪个 review round、哪个目标 commit、改了哪些文件、生成了哪些验证产物。
3. 能够证明模型、数据切分、输出 schema 和 scorer 已经统一的最小运行证据。
4. 如果暂时不能完成完整实验，至少给出 Gate 0 到 Gate 2 的可执行闭环。
