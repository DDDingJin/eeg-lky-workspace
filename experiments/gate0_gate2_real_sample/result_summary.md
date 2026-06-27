# Result Summary

## 1. 本轮用了什么数据？
- 使用数据：`hugo_sample_tf64_p00`
- 运行对象：单被试最小公开样例 `P00`，采用已存在的 `train/val/test` 参考切分。
- 当前定位：真实 sample 的 pipeline validation，不是完整 benchmark 结果。

## 2. 跑了哪些模型？
- classical baseline: `ridge`
- additional classical boundary check: `cca`
- simple neural baseline: `fcnn`
- target model: `adt`

## 3. 每个模型是否真实训练/推理，还是只跑了接口？
- `ridge`: 真实拟合并在测试集上生成预测。
- `cca`: 真实拟合并在测试集上生成预测，但本轮不把其 canonical/match-mismatch 指标当作主比较列。
- `fcnn`: 真实训练并在测试集上推理。
- `adt`: 真实训练并在测试集上按滑窗重建、再由统一 scorer 聚合。

## 4. 指标是什么？
- 主指标：`Pearson correlation`。
- 统一输出：`recording_metrics.csv` 与 `subject_metrics.csv`。
- `cca` 的 canonical correlation 与 match-mismatch accuracy 仅作为附加元数据保留。

## 5. 当前结果能说明什么？
- `adt` 在该最小 sample 上通过统一 scorer 生成了可追溯结果，当前 subject-level Pearson 为 `0.243442`。
- `cca` 在该最小 sample 上通过统一 scorer 生成了可追溯结果，当前 subject-level Pearson 为 `0.153673`。
- `fcnn` 在该最小 sample 上通过统一 scorer 生成了可追溯结果，当前 subject-level Pearson 为 `0.197046`。
- `ridge` 在该最小 sample 上通过统一 scorer 生成了可追溯结果，当前 subject-level Pearson 为 `0.186105`。

## 6. 当前结果不能说明什么？
- 不能说明模型总体优劣。
- 不能替代多被试、多数据集、多 seed 的正式 benchmark。
- 不能作为论文主结果图直接使用。

## 7. 哪些结果只能作为 smoke evidence，不能写成论文结论？
- 本目录下全部结果都只能作为 pipeline validation 或 minimal real-sample evidence。
- `cca` 的附加 canonical / match-mismatch 输出在当前轮次只用于边界与 schema 说明。
