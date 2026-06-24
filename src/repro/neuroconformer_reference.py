from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from .reference_baselines import list_reference_subjects, load_reference_recordings


def pearson_metric(y_true: torch.Tensor, y_pred: torch.Tensor, axis: int = 1, eps: float = 1e-6) -> torch.Tensor:
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)
    numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=axis)
    std_true = torch.sum((y_true - y_true_mean) ** 2, dim=axis)
    std_pred = torch.sum((y_pred - y_pred_mean) ** 2, dim=axis)
    denominator = torch.sqrt(std_true * std_pred + eps)
    return numerator / (denominator + eps)


def pearson_loss(y_true: torch.Tensor, y_pred: torch.Tensor, axis: int = 1) -> torch.Tensor:
    return 1 - pearson_metric(y_true, y_pred, axis=axis)


def multi_scale_pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor, scales: list[int], axis: int = 1) -> torch.Tensor:
    total_loss = torch.zeros(y_pred.shape[0], device=y_pred.device, dtype=y_pred.dtype)
    used = 0
    for scale in scales:
        if y_pred.shape[axis] >= scale:
            pred_transposed = y_pred.transpose(1, 2)
            true_transposed = y_true.transpose(1, 2)
            pred_pooled = F.avg_pool1d(pred_transposed, kernel_size=scale, stride=scale).transpose(1, 2)
            true_pooled = F.avg_pool1d(true_transposed, kernel_size=scale, stride=scale).transpose(1, 2)
            total_loss = total_loss + pearson_loss(true_pooled, pred_pooled, axis=axis)
            used += 1
    return total_loss / max(1, used)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        self.max_len = max_len
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_len - 1, d_model) / np.sqrt(d_model))

    def forward(self, seq_len: int) -> torch.Tensor:
        center = self.max_len - 1
        start = center - (seq_len - 1)
        end = center + seq_len
        rel_pos = self.rel_pos_emb[start:end]
        positions = torch.arange(seq_len, device=rel_pos.device)
        rel_indices = positions.unsqueeze(1) - positions.unsqueeze(0) + (seq_len - 1)
        return rel_pos[rel_indices]


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, use_relative_pos: bool = True) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.use_relative_pos = use_relative_pos
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        if use_relative_pos:
            self.rel_pos_enc = RelativePositionalEncoding(self.d_k)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = q.size(0), q.size(1)
        residual = q
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)
        q = self.w_q(q).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if self.use_relative_pos:
            rel_pos = self.rel_pos_enc(seq_len)
            rel_pos_scores = torch.einsum("bhik,ijk->bhij", q, rel_pos)
            scores = scores + rel_pos_scores / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.dropout(self.w_o(output))
        return output + residual


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_inner: int, dropout: float = 0.1, use_macaron: bool = False) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = Swish()
        self.use_macaron = use_macaron
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.dropout(self.activation(self.w_1(x)))
        x = self.dropout(self.w_2(x))
        if self.use_macaron:
            x = 0.5 * x
        return x + residual


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, expansion_factor: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = d_model * expansion_factor
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = (kernel_size - 1) // 2
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim * 2, kernel_size=1)
        self.glu = GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding, groups=inner_dim)
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(inner_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x).transpose(1, 2)
        x = self.glu(self.pointwise_conv1(x))
        x = self.activation(self.batch_norm(self.depthwise_conv(x)))
        x = self.dropout(self.pointwise_conv2(x))
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_relative_pos: bool = True,
        use_macaron_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.use_macaron_ffn = use_macaron_ffn
        if use_macaron_ffn:
            self.ffn1 = FeedForwardModule(d_model, d_inner, dropout, use_macaron=True)
        self.self_attn = RelativeMultiHeadAttention(d_model, n_head, dropout, use_relative_pos=use_relative_pos)
        self.conv_module = ConvolutionModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, d_inner, dropout, use_macaron=use_macaron_ffn)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_macaron_ffn:
            x = self.ffn1(x)
        x = self.self_attn(x, x, x, mask)
        x = x + self.conv_module(x)
        x = self.ffn2(x)
        return self.layer_norm(x)


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class GatedResidual(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_layer1 = nn.Linear(d_model, d_model // 4)
        self.gate_layer2 = nn.Linear(d_model // 4, d_model)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        global_feat = x.mean(dim=1)
        gate = torch.sigmoid(self.gate_layer2(self.activation(self.gate_layer1(global_feat)))).unsqueeze(1)
        return gate * x + (1 - gate) * residual


class ImprovedOutputHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output * ctx.scale, None


class NeuroConformerDecoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        d_model: int,
        d_inner: int,
        n_head: int,
        n_layers: int,
        dropout: float,
        g_con: bool,
        within_sub_num: int,
        conv_kernel_size: int = 31,
        use_relative_pos: bool = True,
        use_macaron_ffn: bool = True,
        use_sinusoidal_pos: bool = False,
        use_gated_residual: bool = True,
        use_mlp_head: bool = True,
        gradient_scale: float = 1.0,
        skip_cnn: bool = True,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        self.use_sinusoidal_pos = use_sinusoidal_pos
        self.use_gated_residual = use_gated_residual
        self.use_mlp_head = use_mlp_head
        self.gradient_scale = gradient_scale
        self.skip_cnn = skip_cnn
        self.use_se = use_se
        if use_mlp_head:
            self.output_head = ImprovedOutputHead(d_model, dropout)
        else:
            self.fc = nn.Linear(d_model, 1)
        if skip_cnn:
            self.input_proj = nn.Linear(in_channel, d_model)
        else:
            self.conv1 = nn.Conv1d(in_channel, d_model, kernel_size=7, padding=3)
            self.norm1 = nn.LayerNorm(d_model)
            self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            self.drop1 = nn.Dropout(dropout)
            self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
            self.norm2 = nn.LayerNorm(d_model)
            self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            self.drop2 = nn.Dropout(dropout)
            self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.norm3 = nn.LayerNorm(d_model)
            self.act3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            self.drop3 = nn.Dropout(dropout)
        if use_se:
            self.se = SEBlock(d_model, reduction=16)
        if use_sinusoidal_pos:
            pe = torch.zeros(640, d_model)
            position = torch.arange(0, 640, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_head=n_head,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    use_relative_pos=use_relative_pos,
                    use_macaron_ffn=use_macaron_ffn,
                )
                for _ in range(n_layers)
            ]
        )
        if use_gated_residual:
            self.gated_residual = GatedResidual(d_model)
        self.sub_proj = nn.Linear(self.within_sub_num, d_model) if g_con else None

    def forward(self, dec_input: torch.Tensor, sub_id: torch.Tensor) -> torch.Tensor:
        if self.skip_cnn:
            dec_output = self.input_proj(dec_input)
            if self.use_se:
                dec_output = self.se(dec_output.transpose(1, 2)).transpose(1, 2)
        else:
            x = dec_input.transpose(1, 2)
            x = self.drop1(self.act1(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2)))
            x = self.drop2(self.act2(self.norm2(self.conv2(x).transpose(1, 2)).transpose(1, 2)))
            x = self.drop3(self.act3(self.norm3(self.conv3(x).transpose(1, 2)).transpose(1, 2)))
            if self.use_se:
                x = self.se(x)
            dec_output = x.transpose(1, 2)
        if self.g_con:
            sub_emb = self.sub_proj(F.one_hot(sub_id, self.within_sub_num).float())
            output = dec_output + sub_emb.unsqueeze(1)
        else:
            output = dec_output
        if self.use_sinusoidal_pos:
            output = output + self.pe[:, : output.size(1), :]
        residual = output.clone()
        for layer in self.layer_stack:
            output = layer(output)
        if self.use_gated_residual:
            output = self.gated_residual(output, residual)
        else:
            output = output + residual
        if self.gradient_scale != 1.0 and self.training:
            output = GradientScaleFunction.apply(output, self.gradient_scale)
        return self.output_head(output) if self.use_mlp_head else self.fc(output)


class NeuroConformerReferenceDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        split: str,
        *,
        input_length: int,
        channels: Iterable[int] | None,
        subject_to_id: dict[str, int],
        g_con: bool,
        windows_per_sample: int = 20,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.input_length = int(input_length)
        self.channels = channels
        self.subject_to_id = subject_to_id
        self.g_con = g_con
        self.windows_per_sample = int(windows_per_sample)
        self.items: list[tuple[str, np.ndarray, np.ndarray]] = []
        for subject in list_reference_subjects(self.input_dir, split=split):
            self.items.extend(load_reference_recordings(self.input_dir, split, subject, channels=channels))

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.items) * self.windows_per_sample
        return len(self.items)

    def __getitem__(self, index: int):
        if self.split == "train":
            recording_index = index // self.windows_per_sample
            window_index = index % self.windows_per_sample
            recording_id, eeg, env = self.items[recording_index]
            subject = recording_id.split("_-_")[1]
            subject_id = self.subject_to_id[subject] if self.g_con else 0
            max_start = eeg.shape[0] - self.input_length
            if max_start < 0:
                raise ValueError(f"recording too short for input_length={self.input_length}: {recording_id}")
            if self.windows_per_sample > 1:
                segment_size = max(1, max_start // self.windows_per_sample)
                segment_start = min(window_index * segment_size, max_start)
                segment_end = max_start if window_index == self.windows_per_sample - 1 else min(max_start, segment_start + segment_size)
                start_idx = np.random.randint(segment_start, segment_end + 1) if segment_end > segment_start else segment_start
            else:
                start_idx = np.random.randint(0, max_start + 1)
            eeg_chunk = eeg[start_idx : start_idx + self.input_length]
            env_chunk = env[start_idx : start_idx + self.input_length]
            return (
                torch.from_numpy(eeg_chunk.astype(np.float32)),
                torch.from_numpy(env_chunk.astype(np.float32)).unsqueeze(-1),
                torch.tensor(subject_id, dtype=torch.long),
            )

        recording_id, eeg, env = self.items[index]
        subject = recording_id.split("_-_")[1]
        subject_id = self.subject_to_id[subject] if self.g_con else 0
        n_segment = eeg.shape[0] // self.input_length
        eeg = eeg[: n_segment * self.input_length]
        env = env[: n_segment * self.input_length]
        eeg_segments = torch.stack(
            [torch.from_numpy(eeg[i : i + self.input_length].astype(np.float32)) for i in range(0, eeg.shape[0], self.input_length)],
            dim=0,
        )
        env_segments = torch.stack(
            [torch.from_numpy(env[i : i + self.input_length].astype(np.float32)).unsqueeze(-1) for i in range(0, env.shape[0], self.input_length)],
            dim=0,
        )
        return eeg_segments, env_segments, torch.tensor(subject_id, dtype=torch.long), subject


@dataclass
class NeuroConformerTrainSummary:
    best_val_metric: float
    best_epoch: int
    epochs_completed: int
    history: dict[str, list[float]]
    state_dict: dict
    subject_to_id: dict[str, int]
    config: dict


def _neuroconformer_loss(outputs: torch.Tensor, targets: torch.Tensor, huber_weight: float = 0.3) -> tuple[torch.Tensor, dict[str, float]]:
    l_pearson_high = multi_scale_pearson_loss(outputs, targets, scales=[1, 2])
    l_pearson_low = multi_scale_pearson_loss(outputs, targets, scales=[4, 8, 16])
    l_pearson = 0.7 * l_pearson_high + 0.3 * l_pearson_low
    l_huber = F.smooth_l1_loss(outputs, targets, reduction="none", beta=0.1).mean(dim=(1, 2))
    loss = l_pearson + huber_weight * l_huber
    stats = {
        "pearson_loss": float(l_pearson.mean().item()),
        "huber_loss": float(l_huber.mean().item()),
    }
    return loss, stats


def _evaluate_recording_batch(model: nn.Module, batch, device: str) -> tuple[float, float]:
    eeg_segments, env_segments, subject_id, *_ = batch
    eeg_segments = eeg_segments.squeeze(0).to(device)
    env_segments = env_segments.squeeze(0).to(device)
    subject_id = subject_id.to(device)
    if subject_id.ndim == 0:
        subject_ids = subject_id.repeat(eeg_segments.shape[0])
    else:
        subject_ids = subject_id.view(-1).repeat(eeg_segments.shape[0])
    outputs = model(eeg_segments, subject_ids)
    loss, _ = _neuroconformer_loss(outputs, env_segments)
    metric = pearson_metric(env_segments, outputs).mean()
    return float(loss.mean().item()), float(metric.item())


def train_neuroconformer_reference(
    input_dir: str | Path,
    *,
    win_len_seconds: int = 10,
    sample_rate: int = 64,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    dropout: float = 0.4,
    g_con: bool = True,
    device: str = "cuda",
    channels: Iterable[int] | None = None,
    windows_per_sample: int = 20,
    d_model: int = 256,
    d_inner: int = 1024,
    n_head: int = 4,
    n_layers: int = 4,
    conv_kernel_size: int = 31,
    use_relative_pos: bool = True,
    use_macaron_ffn: bool = True,
    use_sinusoidal_pos: bool = False,
    use_gated_residual: bool = True,
    use_mlp_head: bool = True,
    gradient_scale: float = 1.0,
    skip_cnn: bool = True,
    use_se: bool = True,
) -> tuple[nn.Module, NeuroConformerTrainSummary]:
    input_dir = Path(input_dir)
    subjects = list_reference_subjects(input_dir, split="train")
    subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
    input_length = int(win_len_seconds * sample_rate)
    train_set = NeuroConformerReferenceDataset(
        input_dir,
        "train",
        input_length=input_length,
        channels=channels,
        subject_to_id=subject_to_id,
        g_con=g_con,
        windows_per_sample=windows_per_sample,
    )
    val_set = NeuroConformerReferenceDataset(
        input_dir,
        "val",
        input_length=input_length,
        channels=channels,
        subject_to_id=subject_to_id,
        g_con=g_con,
        windows_per_sample=1,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    config = {
        "d_model": d_model,
        "d_inner": d_inner,
        "n_head": n_head,
        "n_layers": n_layers,
        "conv_kernel_size": conv_kernel_size,
        "use_relative_pos": use_relative_pos,
        "use_macaron_ffn": use_macaron_ffn,
        "use_sinusoidal_pos": use_sinusoidal_pos,
        "use_gated_residual": use_gated_residual,
        "use_mlp_head": use_mlp_head,
        "gradient_scale": gradient_scale,
        "skip_cnn": skip_cnn,
        "use_se": use_se,
    }
    model = NeuroConformerDecoder(
        in_channel=len(list(channels)) if channels is not None else 64,
        dropout=dropout,
        g_con=g_con,
        within_sub_num=max(1, len(subject_to_id)),
        **config,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    history = {"train_loss": [], "val_loss": [], "val_metric": [], "lr": []}
    best_state = None
    best_metric = -float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for eeg_batch, env_batch, subject_ids in train_loader:
            eeg_batch = eeg_batch.to(device)
            env_batch = env_batch.to(device)
            subject_ids = subject_ids.to(device)
            optimizer.zero_grad()
            outputs = model(eeg_batch, subject_ids)
            loss, _ = _neuroconformer_loss(outputs, env_batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                loss, metric = _evaluate_recording_batch(model, batch, device)
                val_losses.append(loss)
                val_metrics.append(metric)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_metric = float(np.mean(val_metrics))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
    if best_state is None:
        raise RuntimeError("NeuroConformer training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    summary = NeuroConformerTrainSummary(
        best_val_metric=best_metric,
        best_epoch=best_epoch,
        epochs_completed=epochs,
        history=history,
        state_dict=best_state,
        subject_to_id=subject_to_id,
        config=config,
    )
    return model, summary


def evaluate_neuroconformer_reference(
    input_dir: str | Path,
    *,
    state_dict: dict,
    subject_to_id: dict[str, int],
    config: dict,
    win_len_seconds: int = 10,
    sample_rate: int = 64,
    dropout: float = 0.4,
    g_con: bool = True,
    device: str = "cuda",
    channels: Iterable[int] | None = None,
) -> dict:
    input_dir = Path(input_dir)
    input_length = int(win_len_seconds * sample_rate)
    test_set = NeuroConformerReferenceDataset(
        input_dir,
        "test",
        input_length=input_length,
        channels=channels,
        subject_to_id=subject_to_id,
        g_con=g_con,
        windows_per_sample=1,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model = NeuroConformerDecoder(
        in_channel=len(list(channels)) if channels is not None else 64,
        dropout=dropout,
        g_con=g_con,
        within_sub_num=max(1, len(subject_to_id)),
        **config,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    per_subject: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in test_loader:
            _, metric = _evaluate_recording_batch(model, batch, device)
            subject = batch[3][0]
            per_subject.setdefault(subject, []).append(metric)
    subject_scores = {subject: float(np.mean(scores)) for subject, scores in per_subject.items()}
    values = np.asarray(list(subject_scores.values()), dtype=np.float32)
    return {
        "metric_name": "pearson_metric",
        "mean_test_metric": float(values.mean()),
        "std_test_metric": float(values.std(ddof=0)),
        "subjects": subject_scores,
    }
