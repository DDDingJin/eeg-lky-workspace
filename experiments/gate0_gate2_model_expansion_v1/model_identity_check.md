# Model Identity Check

本文件只回答本轮 `gate0_gate2_model_expansion_v1` 中模型实现身份问题，不对论文最终命名做定稿。

## 1. `dnn` 与 `fcnn` 是否是不同实现？

是，不应视为同一个实现身份。

- `fcnn` 使用本仓库本地实现：`src/repro/simple_models.py::FCNNBaseline`
- `dnn` 使用共享上游实现：`external/upstream/mldecoders/pipeline/dnn.py::FCNN`

两者的高层结构都属于全连接 MLP / FCNN family，但当前分支中的实现来源、类名、模块路径、命名身份都不同，因此不应在审计层面把它们写成“同一个实现”。

## 2. `dnn` 的 implementation path、class name、config、runner

本轮 `dnn` 的实际运行身份如下。

- implementation path: `external/upstream/mldecoders/pipeline/dnn.py`
- class name: `FCNN`
- unified runner entry: `scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_dnn`
- shared training helper: `src/repro/reference_baselines.py::train_dnn_reference_logged`
- model-expansion aggregate runner: `scripts/run_gate0_gate2_model_expansion_v1.py`
- config block: `configs/benchmark/gate0_gate2_model_expansion_v1.json` 中的 `dnn`

本轮 `dnn` config 要点：

- `architecture_id`: `upstream_fcnn_subject_specific`
- `implementation_file`: `external/upstream/mldecoders/pipeline/dnn.py::FCNN`
- `window_size`: `50`
- `hidden_layers`: `3`
- `dropout_rate`: `0.45`
- `batch_size`: `256`
- `max_epochs`: `100`
- `early_stopping_patience`: `10`
- `learning_rate`: `1e-4`
- `weight_decay`: `1e-4`

## 3. `dnn` 和 `fcnn` 的架构差异

从当前代码看，二者在宏观架构上非常接近，都是：

- 输入窗口展平
- 多层全连接
- `Tanh`
- `Dropout`
- 单标量回归输出

但它们仍然存在以下差异层级：

- 实现身份不同：本地 `FCNNBaseline` vs 上游 `FCNN`
- 命名与审计归属不同：`fcnn` 在当前统一流程里被当作本地 simple baseline；`dnn` 明确映射到上游 `pipeline.dnn.FCNN`
- 历史可比性不同：已有 `fcnn_protocol_audit.md` 明确指出，本地 `fcnn` 不应直接当作旧 summary 中 FCNN 的严格同一实现

因此，`dnn` 与 `fcnn` 的主要区别目前更偏“实现身份与可追溯来源”，而不是“已经证实的重大结构创新差异”。

## 4. 论文中应把 `dnn` 写成独立模型，还是写成 MLP / FCNN family 的一个变体？

建议：

- 在执行与审计材料中，`dnn` 保持独立条目，因为它在本轮确实是独立 registry / config / result row
- 在论文叙述层面，不建议把 `dnn` 描述成与 `fcnn` 完全独立的一类新架构
- 更稳妥的写法是：`dnn` 属于 `MLP/FCNN family` 的一个上游实现变体；`fcnn` 是当前 unified benchmark 中的本地 FCNN baseline

也就是说：

- 结果表可以分开列 `fcnn` 和 `dnn`
- 论文正文解释时应注明二者属于同一大类的不同实现身份，避免让读者误解为两个架构上强烈异质的模型家族

## 5. `cnn / eegnet` 与旧 reference summary 的实现身份关系

### `cnn`

`cnn` 与旧 reference summary 的实现身份基本一致，至少在当前仓库可恢复证据内是同一上游类族：

- 本轮使用：`external/upstream/mldecoders/pipeline/dnn.py::CNN`
- 历史 reference baseline runner `scripts/run_reference_baselines.py` 也直接调用同一上游 `CNN`

但本轮仍然不是“原始旧实验目录的原位重跑”，而是：

- 在 unified scorer / unified schema / full-subject single-seed runner 下复用该上游实现

因此更准确的表述是：

- `cnn` 具有较强的实现身份连续性
- 但结果产物本身属于 `unified-port under current benchmark flow`，不是旧 summary 文件的二进制同一运行

### `eegnet`

`eegnet` 不是旧 summary 的同实现身份恢复。

- 本轮使用：`src/repro/mldecoders/models.py::EEGNetRegressor`
- 历史 reference baseline runner `scripts/run_reference_baselines.py` 也调用 `EEGNetRegressor`

这说明当前仓库中的 reference baseline 流程与本轮 unified 流程在类名上是一致的，但该类本身是本仓库本地实现，不是外部上游 reference package 中的独立官方实现标识。

因此更准确的表述是：

- `eegnet` 与当前仓库内历史 reference baseline 记录具有实现连续性
- 但它更像 `local reference-compatible baseline`，而不是可无条件宣称为“旧 reference summary 原始实现的完整身份恢复”

## 审计结论

建议在后续审阅与写作中使用以下保守表述：

- `fcnn`: local unified FCNN baseline
- `dnn`: upstream FCNN-family variant under unified flow
- `cnn`: upstream CNN with stronger identity continuity, but rerun under unified scorer/schema
- `eegnet`: local reference-compatible EEGNet baseline under unified flow

这样可以最大限度避免把“同一家族模型”误写成“完全独立新架构”，也避免把“统一流程下的新聚合结果”误写成“旧 reference summary 的严格同一实验身份”。
