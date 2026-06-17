"""
Conformer-based Training Script for EEG Signal Processing (v2 - Improved)
使用改进版 Conformer 模型，解决特征提取不足问题

主要改进：
1. 使用 FFT_block_conformer_v2.py 中的改进模型
2. 添加全局残差连接和门控机制
3. 使用 MLP 输出头替代单层线性
4. 支持梯度缩放以增强前层学习

Based on train_v10_conformer.py
"""

import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.backends.cudnn as cudnn
from util.utils import get_writer, save_checkpoint
from util.logger import TrainingLogger
from torch.optim.lr_scheduler import StepLR
from models.FFT_block_conformer_v2 import Decoder  # 使用改进版模型
from util.cal_pearson import l1_loss, mse_loss, pearson_loss, pearson_metric, multi_scale_pearson_loss, variance_ratio_loss, si_sdr, si_sdr_loss
from util.dataset import RegressionDataset
import time
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--win_len', type=int, default=10)
parser.add_argument('--sample_rate', type=int, default=64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--g_con', default=True, help="experiment for within subject")

parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal")
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--d_inner', type=int, default=1024)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--fft_conv1d_kernel', type=tuple, default=(9, 1))
parser.add_argument('--fft_conv1d_padding', type=tuple, default=(4, 0))
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--lamda', type=float, default=1)
parser.add_argument('--writing_interval', type=int, default=10)
parser.add_argument('--saving_interval', type=int, default=5)
parser.add_argument('--eval_interval', type=int, default=5, help='evaluation interval (epochs)')
parser.add_argument('--viz_sample_idx', type=int, default=0, help='fixed test sample index for visualization')
parser.add_argument('--grad_log_interval', type=int, default=100, help='gradient logging interval (steps)')

# Conformer-specific parameters
parser.add_argument('--conv_kernel_size', type=int, default=31, help='kernel size for Conformer convolution module')
# Use action with default=True means the value is True unless the negative flag is used
parser.add_argument('--use_relative_pos', action='store_true', dest='use_relative_pos', help='use relative positional encoding in attention (default: True)')
parser.add_argument('--no_relative_pos', action='store_false', dest='use_relative_pos', help='disable relative positional encoding')
parser.set_defaults(use_relative_pos=True)

parser.add_argument('--use_macaron_ffn', action='store_true', dest='use_macaron_ffn', help='use Macaron-style FFN in Conformer (default: True)')
parser.add_argument('--no_macaron_ffn', action='store_false', dest='use_macaron_ffn', help='disable Macaron-style FFN')
parser.set_defaults(use_macaron_ffn=True)

parser.add_argument('--use_sinusoidal_pos', action='store_true', default=False, help='use additional sinusoidal positional encoding')

# ============ v2 改进参数 ============
parser.add_argument('--use_gated_residual', action='store_true', dest='use_gated_residual', help='use gated residual connection (default: True)')
parser.add_argument('--no_gated_residual', action='store_false', dest='use_gated_residual', help='disable gated residual connection')
parser.set_defaults(use_gated_residual=True)

parser.add_argument('--use_mlp_head', action='store_true', dest='use_mlp_head', help='use MLP output head instead of single linear (default: True)')
parser.add_argument('--no_mlp_head', action='store_false', dest='use_mlp_head', help='disable MLP output head')
parser.set_defaults(use_mlp_head=True)

parser.add_argument('--gradient_scale', type=float, default=1.0, help='gradient scaling factor for Conformer layers')

parser.add_argument('--skip_cnn', action='store_true', dest='skip_cnn', help='skip CNN feature extraction, use raw EEG directly (default: True)')
parser.add_argument('--no_skip_cnn', action='store_false', dest='skip_cnn', help='enable CNN feature extraction')
parser.set_defaults(skip_cnn=True)

parser.add_argument('--use_se', action='store_true', dest='use_se', help='use SE channel attention module (default: True)')
parser.add_argument('--no_se', action='store_false', dest='use_se', help='disable SE channel attention')
parser.set_defaults(use_se=True)

# LLRD (Layer-wise Learning Rate Decay) 参数
parser.add_argument('--use_llrd', action='store_true', dest='use_llrd', help='use layer-wise learning rate decay (default: True)')
parser.add_argument('--no_llrd', action='store_false', dest='use_llrd', help='disable layer-wise learning rate decay')
parser.set_defaults(use_llrd=True)

parser.add_argument('--use_conv_module', action='store_true', dest='use_conv_module', help='use convolution module in Conformer block (default: True)')
parser.add_argument('--no_conv_module', action='store_false', dest='use_conv_module', help='disable convolution module in Conformer block')
parser.set_defaults(use_conv_module=True)

parser.add_argument('--llrd_front_scale', type=float, default=1.0, help='LR scale for front layers (CNN, SE, early Conformer)')
parser.add_argument('--llrd_back_scale', type=float, default=3.0, help='LR scale for back layers (late Conformer, gated_residual)')
parser.add_argument('--llrd_output_scale', type=float, default=0.5, help='LR scale for output head')
# 输出层梯度缩放
parser.add_argument('--output_grad_scale', type=float, default=1, help='scale factor for output head gradients after backward')
# ===================================

# Spatial GAT parameters
parser.add_argument('--use_spatial_gat', action='store_true', default=False,
                    help='在最前端加入空间图注意力')
parser.add_argument('--gat_n_heads', type=int, default=4, help='GAT注意力头数')
parser.add_argument('--gat_d_head', type=int, default=32, help='GAT每头维度')
parser.add_argument('--sensor_pos_file', type=str,
                    default='/RAID5/projects/likeyang/MEGdenoise/vision/sensors_mecg64.mat',
                    help='传感器坐标.mat文件路径')

parser.add_argument('--dataset_folder', type=str, default="/RAID5/projects/likeyang/happy/MEGConformer/data", help='write down your absolute path of dataset folder')
parser.add_argument('--split_folder', type=str, default="split_data")
parser.add_argument('--experiment_folder', default=None, help='write down experiment name')

# Distributed training parameters
parser.add_argument('--use_ddp', action='store_true', help='use distributed training')
parser.add_argument('--local-rank', default=-1, type=int, help='local process rank for distributed training')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')

# Data augmentation
parser.add_argument('--windows_per_sample', type=int, default=20, help='number of windows sampled per sample in one epoch')

args = parser.parse_args()

# Load sensor positions for spatial GAT (if enabled)
import scipy.io
sensor_pos = None
if args.use_spatial_gat:
    mat = scipy.io.loadmat(args.sensor_pos_file)
    sensor_pos = mat['pos']  # [64, 3]

# Set random seed for reproducibility
torch.manual_seed(args.seed)
cudnn.benchmark = True

# Determine whether to enable DDP based on command line arguments
use_ddp = args.use_ddp
local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", -1))

# Initialize distributed environment (if enabled)
if use_ddp and local_rank != -1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    world_size = 1
    rank = 0
    local_rank = -1

# Set input length = sample_rate * window_length (seconds)
input_length = args.sample_rate * args.win_len
device = torch.device(f"cuda:{local_rank}" if use_ddp and local_rank != -1 else f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Provide the path of the dataset
data_folder = os.path.join(args.dataset_folder, args.split_folder)
features = ["eeg"] + ["envelope"]

# Create a directory to store results
result_folder = 'test_results'
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.experiment_folder is None:
    if use_ddp:
        experiment_folder = "conformer_v2_nlayer{}_dmodel{}_nhead{}_gscale{}_dist_{}".format(
            args.n_layers, args.d_model, args.n_head, args.gradient_scale, current_time)
    else:
        experiment_folder = "conformer_v2_nlayer{}_dmodel{}_nhead{}_gscale{}_single_{}".format(
            args.n_layers, args.d_model, args.n_head, args.gradient_scale, current_time)
else:
    experiment_folder = args.experiment_folder

save_path = os.path.join(result_folder, experiment_folder)
# Only create writer and logger in main process
is_main_process = (not use_ddp) or rank == 0
writer = get_writer(result_folder, experiment_folder) if is_main_process else None
logger = TrainingLogger(writer, save_path, is_main_process, enable_grad_histogram=True)


def create_dataloader(split_name, data_folder, features, input_length, args, use_ddp, local_rank):
    is_train = split_name == 'train'
    files = [x for x in glob.glob(os.path.join(data_folder, f"{split_name}_-_*"))
             if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]

    # Add windows_per_sample parameter for training set
    windows_per_sample = args.windows_per_sample if hasattr(args, 'windows_per_sample') else 10

    dataset = RegressionDataset(
        files,
        input_length,
        args.in_channel,
        split_name,
        args.g_con,
        windows_per_sample=windows_per_sample if is_train else 1  # Only use multi-window for training
    )

    # Only use DistributedSampler for training set in distributed mode
    sampler = DistributedSampler(dataset) if is_train and use_ddp and local_rank != -1 else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        num_workers=args.workers,
        sampler=sampler,
        drop_last=True,
        shuffle=(sampler is None and is_train)  # Only shuffle for training without sampler
    )


def get_llrd_param_groups(model, base_lr, front_scale, back_scale, output_scale, n_layers):
    """
    获取Layer-wise Learning Rate Decay参数组

    分层策略：
    - 前层（CNN, SE, 位置编码, Conformer前半）：base_lr * front_scale
    - 后层（Conformer后半, gated_residual）：base_lr * back_scale
    - 输出层（output_head/fc）：base_lr * output_scale

    Args:
        model: 模型
        base_lr: 基础学习率
        front_scale: 前层学习率倍率
        back_scale: 后层学习率倍率
        output_scale: 输出层学习率倍率
        n_layers: Conformer层数

    Returns:
        param_groups: 参数组列表
    """
    # 定义层的划分
    front_layers = ['conv1', 'conv2', 'conv3', 'norm1', 'norm2', 'norm3',
                    'act1', 'act2', 'act3', 'drop1', 'drop2', 'drop3',
                    'se', 'sub_proj', 'pos_encoder']
    output_layers = ['output_head', 'fc']

    # Conformer层的划分：前半为front，后半为back
    mid_layer = n_layers // 2  # 例如8层时，0-3为前半，4-7为后半

    front_params = []
    back_params = []
    output_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 判断参数属于哪一层
        is_front = any(layer in name for layer in front_layers)
        is_output = any(layer in name for layer in output_layers)

        # 检查是否是Conformer层
        is_conformer_layer = 'layer_stack' in name
        if is_conformer_layer:
            # 提取层号：layer_stack.0.xxx -> 0
            try:
                layer_idx = int(name.split('layer_stack.')[1].split('.')[0])
                if layer_idx < mid_layer:
                    is_front = True
                else:
                    is_front = False
            except (ValueError, IndexError):
                is_front = False

        # 检查是否是gated_residual
        is_gated_residual = 'gated_residual' in name

        if is_output:
            output_params.append(param)
        elif is_front:
            front_params.append(param)
        elif is_gated_residual or (is_conformer_layer and not is_front):
            back_params.append(param)
        else:
            # 默认归为back层
            back_params.append(param)

    param_groups = [
        {'params': front_params, 'lr': base_lr * front_scale, 'name': 'front_layers'},
        {'params': back_params, 'lr': base_lr * back_scale, 'name': 'back_layers'},
        {'params': output_params, 'lr': base_lr * output_scale, 'name': 'output_layers'}
    ]

    return param_groups


def scale_output_gradients(model, scale, use_ddp, local_rank):
    """
    缩放输出层的梯度

    在 loss.backward() 之后调用，用于降低输出层梯度的幅度，
    避免输出层梯度过大导致前层学习不足。

    Args:
        model: 模型
        scale: 缩放因子 (小于1表示缩小梯度)
        use_ddp: 是否使用分布式训练
        local_rank: 本地rank
    """
    base_model = model.module if use_ddp and local_rank != -1 else model

    # 定义输出层
    output_layers = ['output_head', 'fc']

    for name, param in base_model.named_parameters():
        if param.grad is not None and any(layer in name for layer in output_layers):
            param.grad.data.mul_(scale)


def multi_scale_pearson_metric(y_pred, y_true, scales=[4, 8, 16, 32], axis=1):
    """
    计算多尺度 Pearson 相关系数（用于评估）

    参数:
    y_pred: 预测值 [B, T, C]
    y_true: 真实值 [B, T, C]
    scales: 下采样尺度列表
    axis: 计算 Pearson 的维度（默认为时间维度 axis=1）

    返回:
    metrics_dict: 包含每个尺度的 Pearson 系数的字典
    """
    import torch.nn.functional as F

    metrics = {}

    # 原始尺度的 Pearson 系数
    metrics['pearson_scale_1'] = pearson_metric(y_true, y_pred, axis=axis).mean()

    # 对于每个尺度
    for scale in scales:
        if y_pred.shape[axis] >= scale:
            # 需要转置以使用 avg_pool1d (需要 [B, C, T] 格式)
            pred_transposed = y_pred.transpose(1, 2)  # [B, 1, T]
            true_transposed = y_true.transpose(1, 2)  # [B, 1, T]

            # 平均池化降采样
            pred_pooled = F.avg_pool1d(pred_transposed, kernel_size=scale, stride=scale)
            true_pooled = F.avg_pool1d(true_transposed, kernel_size=scale, stride=scale)

            # 转回 [B, T', 1] 格式
            pred_pooled = pred_pooled.transpose(1, 2)
            true_pooled = true_pooled.transpose(1, 2)

            # 计算这个尺度的 Pearson 系数
            scale_metric = pearson_metric(true_pooled, pred_pooled, axis=axis).mean()
            metrics[f'pearson_scale_{scale}'] = scale_metric

    return metrics


def main():
    # ============ 使用改进版 Conformer 模型 ============
    model = Decoder(
        in_channel=args.in_channel,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_head=args.n_head,
        n_layers=args.n_layers,
        fft_conv1d_kernel=args.fft_conv1d_kernel,
        fft_conv1d_padding=args.fft_conv1d_padding,
        dropout=args.dropout,
        g_con=args.g_con,
        within_sub_num=85,
        # Conformer-specific parameters
        conv_kernel_size=args.conv_kernel_size,
        use_relative_pos=args.use_relative_pos,
        use_macaron_ffn=args.use_macaron_ffn,
        use_sinusoidal_pos=args.use_sinusoidal_pos,
        # v2 改进参数
        use_gated_residual=args.use_gated_residual,
        use_mlp_head=args.use_mlp_head,
        gradient_scale=args.gradient_scale,
        skip_cnn=args.skip_cnn,  # 是否跳过CNN特征提取
        use_se=args.use_se,       # 是否使用SE通道注意力(独立控制)
        use_conv_module=args.use_conv_module,  # 是否使用Conformer内部卷积模块
        use_spatial_gat=args.use_spatial_gat,
        gat_n_heads=args.gat_n_heads,
        gat_d_head=args.gat_d_head,
        sensor_pos=sensor_pos,
    ).to(device)
    # ==============================================

    # Print model info in main process
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'=' * 60}")
        print(f"Conformer Model Configuration (v2 - Improved)")
        print(f"{'=' * 60}")
        print(f"基础配置:")
        print(f"  - Model dimension: {args.d_model}")
        print(f"  - FFN inner dimension: {args.d_inner}")
        print(f"  - Number of heads: {args.n_head}")
        print(f"  - Number of layers: {args.n_layers}")
        print(f"  - Conv kernel size: {args.conv_kernel_size}")
        print(f"  - Dropout: {args.dropout}")
        print(f"\nConformer特性:")
        print(f"  - Use relative pos: {args.use_relative_pos}")
        print(f"  - Use Macaron FFN: {args.use_macaron_ffn}")
        print(f"  - Use sinusoidal pos: {args.use_sinusoidal_pos}")
        print(f"\n【v2 改进】:")
        print(f"  - Use gated residual: {args.use_gated_residual}")
        print(f"  - Use MLP head: {args.use_mlp_head}")
        print(f"  - Gradient scale: {args.gradient_scale}x")
        print(f"  - Skip CNN: {args.skip_cnn} {'⚠️  直接使用原始EEG数据' if args.skip_cnn else ''}")
        print(f"  - Use SE attention: {args.use_se} {'(独立控制)' if args.use_se else ''}")
        print(f"\n参数统计:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
        print(f"{'=' * 60}\n")

    # Wrap model with DDP if using distributed training
    if use_ddp and local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ============ 优化器设置 (支持LLRD) ============
    if args.use_llrd:
        # 使用Layer-wise Learning Rate Decay
        # 注意：DDP模型需要使用model.module来获取参数
        base_model = model.module if use_ddp and local_rank != -1 else model
        param_groups = get_llrd_param_groups(
            base_model,
            args.learning_rate,
            args.llrd_front_scale,
            args.llrd_back_scale,
            args.llrd_output_scale,
            args.n_layers
        )
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.98), eps=1e-09)

        if is_main_process:
            print(f"\n【LLRD 配置】:")
            for group in param_groups:
                print(f"  - {group['name']}: lr={group['lr']:.6f} ({len(group['params'])} params)")
    else:
        # 使用统一学习率
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.98),
                                     eps=1e-09)
    # ==============================================

    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # Create data loaders
    train_dataloader = create_dataloader('train', data_folder, features, input_length, args, use_ddp, local_rank)
    val_dataloader = create_dataloader('val', data_folder, features, input_length, args, use_ddp, local_rank)
    test_dataloader = create_dataloader('test', data_folder, features, input_length, args, use_ddp, local_rank)

    # Get fixed test sample for visualization
    viz_sample = None
    if is_main_process:
        for i, (test_inputs, test_labels, test_sub_id) in enumerate(test_dataloader):
            if i == args.viz_sample_idx:
                viz_sample = (test_inputs.squeeze(0).to(device),
                              test_labels.squeeze(0).to(device),
                              test_sub_id.to(device))
                break

    # Calculate total steps per epoch
    iter_per_epoch = len(train_dataloader)
    global_step = 0

    # ============ 跟踪最佳模型 ============
    best_val_loss = float('inf')  # 最佳validation loss (越小越好)
    best_epoch = 0
    best_step = 0
    # ====================================

    # Train the model
    for epoch in range(args.epoch):
        model.train()
        start_time = time.time()

        # Set epoch for distributed sampler
        if use_ddp and train_dataloader.sampler is not None:
            train_dataloader.sampler.set_epoch(epoch)

        # 创建进度条（只在主进程显示）
        if is_main_process:
            pbar = tqdm(enumerate(train_dataloader),
                       total=len(train_dataloader),
                       desc=f'Epoch {epoch+1}/{args.epoch}',
                       ncols=120,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            pbar = enumerate(train_dataloader)

        for step, (inputs, labels, sub_id) in pbar:
            global_step += 1
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            sub_id = sub_id.to(device)
            outputs = model(inputs, sub_id)

            # ========== 损失函数选择 ==========
            # V1: 原始损失函数（MSE + Pearson²）
            # l_p = pearson_loss(outputs, labels)
            # l_mse = mse_loss(outputs, labels)
            # loss = l_mse + args.lamda * (l_p ** 2)
            # loss = loss.mean()

            # V2: 多尺度 Pearson Loss（移除 MSE，避免预测均值）
            # 问题：相关性提升到 0.206，但预测幅度过大（方差比 4.09）
            # loss = multi_scale_pearson_loss(outputs, labels, scales=[32, 64, 128])
            # loss = loss.mean()

            # V3: 多尺度 Pearson + L1 约束（控制幅度）
            # 问题：L1 权重 0.1 过强，验证 Pearson 从 0.218 降到 0.203
            #       L1 梯度恒定，与 Pearson 优化目标冲突
            # l_pearson = multi_scale_pearson_loss(outputs, labels, scales=[32, 64, 128])
            # l_l1 = l1_loss(outputs, labels)
            # loss = l_pearson.mean() + 0.1 * l_l1.mean()

            # V4: 多尺度 Pearson + 方差比约束
            # 优势：直接约束 var(pred)/var(true) ≈ 1，精准控制幅度
            #       不惩罚均值偏移，只关心波动幅度
            #       梯度平滑，易于优化，与 Pearson 目标兼容
            # l_pearson = multi_scale_pearson_loss(outputs, labels, scales=[32, 64, 128])
            # l_var_ratio = variance_ratio_loss(labels, outputs)
            # loss = l_pearson.mean() + 0.5 * l_var_ratio.mean()

            # V6: 动态权重MSE策略
            # 训练初期用MSE稳定训练，后期降低MSE权重避免梯度冲突
            # 优势：初期快速收敛减少数值偏差，后期主要优化Pearson相关性
            # epoch_ratio = epoch / args.epoch  # 当前epoch占总epoch的比例
            # mse_weight = max(0.05, 1.0 - epoch_ratio)  # 从1.0逐渐降到0.05
            # l_pearson = multi_scale_pearson_loss(outputs, labels, scales=[4, 8, 16, 32])
            # l_mse = mse_loss(outputs, labels)
            # loss = l_pearson.mean() + mse_weight * l_mse.mean()

            # V8: Pearson + Huber Loss（当前版本，调整尺度权重）
            # Huber Loss优势：
            # 1. 小误差用L2（平滑梯度），大误差用L1（对异常值鲁棒）
            # 2. 比MSE更适合不规则曲线（不会被峰值主导）
            # 3. 与Pearson梯度兼容性更好
            #
            # 尺度权重调整策略：
            # - 高频组 (尺度1,2): 加权60%，强化高频细节学习
            # - 低频组 (尺度4,8,16): 加权40%，保持低频趋势捕捉
            # - 原来权重：高频20%(1/5)，低频80%(4/5)
            # - 现在权重：高频60%，低频40%，提升高频权重3倍
            l_pearson_high = multi_scale_pearson_loss(outputs, labels, scales=[1, 2])      # 尺度1+2
            l_pearson_low = multi_scale_pearson_loss(outputs, labels, scales=[4, 8, 16]) # 尺度4+8+16
            l_pearson = 0.7 * l_pearson_high + 0.3 * l_pearson_low
            l_huber = F.smooth_l1_loss(outputs, labels, reduction='none', beta=0.1).mean()
            # loss = l_pearson.mean() + 0.3 * l_huber

            loss = l_pearson.mean()
            # 用于日志记录和监控
            with torch.no_grad():
                l_p = pearson_loss(outputs, labels)
                # 计算实际的方差比，用于监控幅度控制效果
                var_true = torch.var(labels, dim=1, keepdim=False, unbiased=False)
                var_pred = torch.var(outputs, dim=1, keepdim=False, unbiased=False)
                actual_var_ratio = (var_pred / (var_true + 1e-6)).mean()
            # ==================================
            loss.backward()

            # 策略4：缩放输出层梯度
            if args.output_grad_scale != 1.0:
                scale_output_gradients(model, args.output_grad_scale, use_ddp, local_rank)

            # Log gradient information
            if global_step % args.grad_log_interval == 0:
                logger.log_gradients(model, global_step, key_layers_only=True)

            optimizer.step()

            # 更新进度条显示（每步都更新）
            if is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pearson': f'{l_p.mean().item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'step': global_step
                })

            # Log training progress (减少日志频率)
            if step % args.writing_interval == 0:
                spend_time = time.time() - start_time
                learning_rate = optimizer.param_groups[0]["lr"]

                # Calculate time per step and speed
                time_per_step = spend_time / (step + 1)
                speed = 1.0 / time_per_step if time_per_step > 0 else 0

                # Calculate remaining time
                steps_per_epoch = iter_per_epoch
                total_steps = args.epoch * steps_per_epoch
                current_total_steps = epoch * steps_per_epoch + step
                remaining_steps = total_steps - current_total_steps
                total_remaining = time_per_step * remaining_steps

                # Use unified logging interface (只记录到TensorBoard，不打印到终端)
                logger.log_training(
                    epoch=epoch + 1,
                    total_epochs=args.epoch,
                    step=step,
                    total_steps=iter_per_epoch,
                    loss_dict={
                        'total': loss.item(),
                        'pearson_loss': l_pearson.mean().item(),
                        'huber_loss': l_huber.item(),
                        'var_ratio': actual_var_ratio.item(),
                        'pearson_metric': l_p.mean().item()
                    },
                    lr=learning_rate,
                    speed=speed,
                    time_remaining=total_remaining,
                    global_step=global_step
                )

        # 关闭进度条
        if is_main_process:
            pbar.close()

        # Epoch-based evaluation
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            val_loss = 0
            val_metric = 0

            with torch.no_grad():
                for val_inputs, val_labels, val_sub_id in val_dataloader:
                    val_inputs = val_inputs.squeeze(0).to(device)
                    val_labels = val_labels.squeeze(0).to(device)
                    val_sub_id = val_sub_id.to(device)

                    val_outputs = model(val_inputs, val_sub_id)
                    val_loss += pearson_loss(val_outputs, val_labels).mean()
                    val_metric += pearson_metric(val_outputs, val_labels).mean()

                val_loss /= len(val_dataloader)
                val_metric /= len(val_dataloader)

                # Log validation results
                logger.log_validation(global_step, val_loss.item(), val_metric.item())

                # ============ 保存最佳模型（基于validation loss） ============
                if is_main_process and val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_epoch = epoch + 1
                    best_step = global_step

                    print(f"\n{'🎉'*30}")
                    print(f"🏆 发现更好的模型！")
                    print(f"  Epoch: {best_epoch}, Step: {best_step}")
                    print(f"  Val Loss: {best_val_loss:.4f}")
                    print(f"  Val Pearson: {val_metric.item():.4f}")
                    print(f"{'🎉'*30}\n")

                    # 保存最佳模型权重
                    best_model_path = os.path.join(save_path, 'best_model.pt')
                    if use_ddp and local_rank != -1:
                        checkpoint = {
                            'epoch': best_epoch,
                            'step': best_step,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'learning_rate': optimizer.param_groups[0]["lr"],
                            'val_loss': best_val_loss,
                            'val_pearson': val_metric.item(),
                            'args': vars(args)
                        }
                    else:
                        checkpoint = {
                            'epoch': best_epoch,
                            'step': best_step,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'learning_rate': optimizer.param_groups[0]["lr"],
                            'val_loss': best_val_loss,
                            'val_pearson': val_metric.item(),
                            'args': vars(args)
                        }

                    torch.save(checkpoint, best_model_path)
                    print(f"✓ 最佳模型已保存到: {best_model_path}\n")
                # ============================================================

                # Visualization on val sample (for monitoring)
                if is_main_process and viz_sample is not None:
                    viz_inputs, viz_labels, viz_sub_id = viz_sample
                    viz_outputs = model(viz_inputs, viz_sub_id)

                    # Convert to numpy for plotting
                    viz_outputs_np = viz_outputs[0].cpu().numpy()
                    viz_labels_np = viz_labels[0].cpu().numpy()

                    # Log visualization
                    logger.log_visualization(
                        predictions=viz_outputs_np,
                        targets=viz_labels_np,
                        epoch=epoch + 1,
                        global_step=global_step,
                        save_png=True
                    )

            model.train()

        # Save model (only in main process)
        if (epoch + 1) % args.saving_interval == 0 and is_main_process:
            learning_rate = optimizer.param_groups[0]["lr"]
            # Special handling for DDP model state_dict
            if use_ddp and local_rank != -1:
                checkpoint = {
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': learning_rate,
                    'args': vars(args)  # Save all args including v2 improvements
                }
                torch.save(checkpoint, os.path.join(save_path, f'conformer_v2_model_step_{global_step}.pt'))
            else:
                save_checkpoint(model, optimizer, learning_rate, epoch, save_path, step=global_step)

        scheduler.step()


if __name__ == '__main__':
    # Ensure directory exists (only in main process)
    if is_main_process:
        os.makedirs(save_path, exist_ok=True)
        print(f"\n💾 Results will be saved to: {save_path}\n")

    # Synchronize all processes if using distributed training
    if use_ddp and local_rank != -1:
        dist.barrier()

    main()
