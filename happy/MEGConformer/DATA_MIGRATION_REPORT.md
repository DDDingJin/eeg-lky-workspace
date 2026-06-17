# 数据迁移完成报告

## 执行摘要

已成功将17个受试者的MEG数据从 `data/processed/` 迁移到 `data/split_data/`，按时间段8:1:1划分为训练/验证/测试集。

## 数据统计

### 文件数量
- **训练集**: 34个文件（17个受试者 × 2种特征）
- **验证集**: 34个文件（17个受试者 × 2种特征）
- **测试集**: 34个文件（17个受试者 × 2种特征）
- **总计**: 102个文件

### 数据形状
- **EEG数据**: `(time, 64)` - 时间步 × 64通道
  - 训练集: `(72952, 64)` - 80%数据
  - 验证集: `(9119, 64)` - 10%数据
  - 测试集: `(9120, 64)` - 10%数据

- **Envelope数据**: `(time,)` - 1D数组
  - 训练集: `(72952,)`
  - 验证集: `(9119,)`
  - 测试集: `(9120,)`

### 存储空间
- **原始数据** (`data/processed/`): 758 MB
- **切分数据** (`data/split_data/`): 769 MB
- **总计**: 1.5 GB

## 数据划分策略

### 时间段划分（每个受试者）
- **训练集**: 时间索引 0-72951（前80%）
- **验证集**: 时间索引 72952-82070（中间10%）
- **测试集**: 时间索引 82071-91190（最后10%）

### 受试者ID映射
```
sub-01: 20260317_184929_extracted.npy
sub-02: 20260317_192241_extracted.npy
sub-03: 20260317_195233_extracted.npy
sub-04: 20260317_203620_extracted.npy
sub-05: 20260318_104305_extracted.npy
sub-06: 20260318_112136_extracted.npy
sub-07: 20260318_151010_extracted.npy
sub-08: 20260318_201301_extracted.npy
sub-09: 20260318_204420_extracted.npy
sub-10: 20260318_213626_extracted.npy
sub-11: 20260320_141142_extracted.npy
sub-12: 20260320_145005_extracted.npy
sub-13: 20260320_151854_extracted.npy
sub-14: 20260320_154927_extracted.npy
sub-15: 20260325_203840_extracted.npy
sub-16: 20260325_211910_extracted.npy
sub-17: 20260326_103308_extracted.npy
```

## 数据加载验证

### 训练集
- **样本数**: 340（17个受试者 × 20个窗口）
- **预加载数据组**: 17组
- **样本形状**: EEG `[640, 64]`, Envelope `[640]`

### 验证集
- **样本数**: 17（每个受试者1个样本）
- **数据长度**: 9119时间步

### 测试集
- **样本数**: 17（每个受试者1个样本）
- **数据长度**: 9120时间步

## 关键改进

1. **数据格式转换**: 将原始的 `(channels, time)` 格式转换为模型期望的 `(time, channels)` 格式
2. **时间段划分**: 每个受试者的数据按时间顺序切分，保持时序连续性
3. **零代码改动**: 现有训练代码无需修改，直接兼容新数据格式
4. **完整覆盖**: 每个受试者都在训练/验证/测试集中出现

## 创建的脚本

1. **`scripts/prepare_split_data.py`**: 数据切分脚本
   - 读取 `data/processed/` 中的原始数据
   - 按8:1:1时间段划分
   - 转换为正确的数据格式
   - 保存到 `data/split_data/`

2. **`scripts/test_data_loading.py`**: 数据加载测试脚本
   - 验证数据集加载功能
   - 检查数据形状和范围
   - 确认样本数量正确

3. **`scripts/verify_data_format.py`**: 数据格式验证脚本
   - 验证数据形状
   - 检查数据连续性
   - 确认划分比例

## 下一步操作

现在可以直接运行训练命令：

### 单GPU训练
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --gpu 0 \
  --n_layers 4 --d_model 256 --n_head 4 \
  --batch_size 64 --epoch 500 --learning_rate 0.0001
```

### 分布式训练
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --use_ddp --batch_size 64 --n_layers 4
```

## 验证命令

```bash
# 验证数据格式
python scripts/verify_data_format.py

# 测试数据加载
source ~/miniconda3/etc/profile.d/conda.sh
conda activate happy
python scripts/test_data_loading.py

# 检查文件数量
ls data/split_data/*.npy | wc -l  # 应该是102
```

## 注意事项

- 数据已经转换为 `(time, channels)` 格式，符合模型输入要求
- 每个受试者的数据按时间顺序切分，保持了时序特性
- 训练集使用多窗口采样（20个窗口/受试者），验证和测试集使用单窗口
- 所有数据预加载到内存，减少训练时的IO开销
