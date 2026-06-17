# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

MEGConformer 是一个基于 Conformer 的深度学习模型，用于 EEG 脑电信号处理，专门设计用于听觉注意力解码。项目实现了 Conformer 架构（卷积增强的 Transformer）来从 EEG 信号预测语音包络。

## 环境设置

**Conda 环境**: `happy`
```bash
conda env create -f happy.yml
conda activate happy
```

环境使用 PyTorch 2.5.1 和 CUDA 11.8。

## 核心架构

模型架构流程：
1. **输入**: 64通道 EEG 信号（10秒窗口，64Hz采样率 = 640时间步）
2. **可选的 CNN 特征提取**（可通过 `--skip_cnn` 跳过）：
   - 三层 1D 卷积（卷积核大小：7, 5, 3）
   - SE（Squeeze-and-Excitation）通道注意力
3. **Conformer 层**（`models/ConformerLayers.py`）：
   - 带相对位置编码的多头自注意力
   - 卷积模块（深度可分离卷积）
   - Macaron 风格的前馈网络
4. **门控残差连接**（v2 改进）
5. **输出头**: MLP 或线性层 → 语音包络预测

**关键模型文件**：
- `models/FFT_block_conformer_v2.py`: 当前生产模型（Decoder 类）
- `models/ConformerLayers.py`: ConformerBlock 实现
- `models/SubLayers.py`: 注意力和 FFN 子层

## 训练命令

**单 GPU 训练**：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --gpu 0 \
  --n_layers 4 --d_model 256 --n_head 4 \
  --batch_size 64 --epoch 500 --learning_rate 0.0001
```

**分布式训练**（推荐）：
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --use_ddp --batch_size 64 --n_layers 4
```

**关键训练参数**：
- `--skip_cnn`: 跳过 CNN 特征提取，直接使用原始 EEG（默认：True）
- `--use_se`: 启用 SE 通道注意力（默认：True）
- `--use_gated_residual`: 使用门控残差连接（默认：True）
- `--use_mlp_head`: 使用 MLP 输出头而非单层线性（默认：True）
- `--use_llrd`: 启用分层学习率衰减（默认：True）
- `--gradient_scale`: Conformer 层的梯度缩放因子（默认：1.0）
- `--windows_per_sample`: 每个样本每轮采样的窗口数（默认：20）

**消融实验**：
```bash
# 禁用 CNN
python train.py --no_skip_cnn --experiment_folder Exp-01-无CNN

# 禁用 SE 注意力
python train.py --no_se --experiment_folder Exp-02-无SE

# 改变层数
python train.py --n_layers 6 --experiment_folder Exp-03-6层
```

## 测试和评估

**评估检查点**：
```bash
python test_model.py --checkpoint test_results/<实验文件夹>/best_model.pt --gpu 0
```

生成内容：
- 每个受试者的 Pearson 相关系数指标
- `test_results_eval/<实验文件夹>/` 中的箱线图
- JSON 格式的指标文件

**消融分析**：
```bash
# 对多个模型运行推理
python ablation_inference.py --models Exp-00 Exp-01-无CNN Exp-02-无SE --gpu 0

# 生成图表
python ablation_plot.py  # 柱状图
python ablation_plot_violin.py  # 小提琴图
```

**跨受试者分析**：
```bash
python plot_cross_subject_analysis.py --all_models
python plot_cross_subject_analysis.py --ablation --grouped
```

**TensorBoard 可视化**：
```bash
python plot_tensorboard.py --logdir test_results/<实验文件夹>/tb_logs
```

## 数据组织

**数据集结构**：
```
data/
├── split_data/          # 训练/验证/测试划分
│   ├── train_-_sub-01_-_task-*_-_eeg.npy
│   ├── train_-_sub-01_-_task-*_-_envelope.npy
│   └── ...
└── test_data/           # 额外测试数据
```

**数据格式**：
- EEG: `[时间步, 64通道]` numpy 数组
- Envelope: `[时间步, 1]` numpy 数组
- 采样率: 64 Hz
- 窗口长度: 10 秒（640 时间步）

## 损失函数

训练脚本（`train.py`）使用多尺度 Pearson 损失配合 Huber 正则化（V8 策略）：
```python
# 高频尺度 (1, 2) 权重 70%
# 低频尺度 (4, 8, 16) 权重 30%
l_pearson = 0.7 * l_pearson_high + 0.3 * l_pearson_low
l_huber = F.smooth_l1_loss(outputs, labels, beta=0.1)
loss = l_pearson + 0.3 * l_huber
```

之前的损失策略（V1-V7）在 `train.py:483-517` 中有注释可供参考。

## 关键工具模块

- `util/dataset.py`: 带预加载和多窗口采样的 RegressionDataset
- `util/cal_pearson.py`: Pearson 相关系数指标和损失函数
- `util/logger.py`: 用于 TensorBoard 日志的 TrainingLogger
- `util/utils.py`: 检查点保存和 writer 工具

## 输出目录

- `test_results/<实验文件夹>/`: 训练输出（检查点、TensorBoard 日志）
- `test_results_eval/<实验文件夹>/`: 评估结果（图表、JSON 指标）
- `ablation_plots/`: 消融研究可视化
- `comparison_results/`: 跨模型对比图表
- `prediction_analysis/`: 预测质量分析

## 实验命名规范

遵循模式：`Exp-##-描述符`
- `Exp-00`: 基线模型
- `Exp-01-无CNN`: 不使用 CNN 特征提取
- `Exp-02-无SE`: 不使用 SE 注意力
- `Exp-03-6层`: 6 层 Conformer

这个命名被绘图脚本引用（`ablation_plot.py`、`plot_cross_subject_analysis.py`）。

## 分布式训练说明

- 使用 PyTorch DDP（DistributedDataParallel）
- 用 `torchrun --standalone --nproc_per_node=N` 启动
- 只有 rank 0 进程写日志和保存检查点
- 训练使用 DistributedSampler 进行数据并行
- 验证/测试在所有进程上运行，但只有 rank 0 记录结果

## 模型检查点

训练脚本保存：
- `best_model.pt`: 基于验证损失的最佳模型
- `conformer_v2_model_step_<步数>.pt`: 每 `--saving_interval` 轮的定期检查点

检查点内容：
- `model_state_dict` 或 `state_dict`: 模型权重
- `optimizer_state_dict` 或 `optimizer`: 优化器状态
- `epoch`、`step`: 训练进度
- `val_loss`、`val_pearson`: 验证指标
- `args`: 所有训练参数，用于可重现性
