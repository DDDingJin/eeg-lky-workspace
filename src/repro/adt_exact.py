from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


def positional_encoding(position: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    angle_rates = 1.0 / torch.pow(10000.0, (2.0 * torch.floor(idx / 2.0)) / float(d_model))
    angle_rads = pos * angle_rates

    even_mask = (torch.arange(d_model) % 2 == 0).to(torch.float32).unsqueeze(0)
    odd_mask = (torch.arange(d_model) % 2 == 1).to(torch.float32).unsqueeze(0)

    sines = torch.sin(angle_rads) * even_mask
    cosines = torch.cos(angle_rads) * odd_mask
    return (sines + cosines).unsqueeze(0)


def anti_causal_attention_mask(length: int) -> torch.Tensor:
    lower = torch.tril(torch.ones(length, length, dtype=torch.float32))
    return 1.0 - lower


def pearson_tf_torch(y_true: torch.Tensor, y_pred: torch.Tensor, axis: int = 1, eps: float = 1e-8) -> torch.Tensor:
    y_true_mean = y_true.mean(dim=axis, keepdim=True)
    y_pred_mean = y_pred.mean(dim=axis, keepdim=True)
    numerator = ((y_true - y_true_mean) * (y_pred - y_pred_mean)).sum(dim=axis, keepdim=True)
    std_true = ((y_true - y_true_mean) ** 2).sum(dim=axis, keepdim=True)
    std_pred = ((y_pred - y_pred_mean) ** 2).sum(dim=axis, keepdim=True)
    denominator = torch.sqrt(std_true * std_pred).clamp_min(eps)
    return (numerator / denominator).mean(dim=-1)


def pearson_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return -pearson_tf_torch(y_true, y_pred).mean()


def pearson_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return pearson_tf_torch(y_true, y_pred).mean()


class RecordingWindowDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        split: str,
        *,
        participant: str | None = None,
        window_length: int = 320,
        hop_length: int = 64,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.participant = participant
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        self.records: list[tuple[str, int]] = []
        self.recordings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        eeg_files = sorted(self.input_dir.glob(f"{self.split}_-_*_-_eeg.npy"))
        for eeg_path in eeg_files:
            parts = eeg_path.name.split("_-_")
            if len(parts) != 4:
                continue
            subject = parts[1]
            if self.participant is not None and subject != self.participant:
                continue
            recording_id = "_-_".join(parts[:3])
            env_path = self.input_dir / f"{recording_id}_-_envelope.npy"
            if not env_path.exists():
                continue
            eeg = np.load(eeg_path).astype(np.float32)
            env = np.load(env_path).astype(np.float32)
            if eeg.ndim != 2 or env.ndim != 2:
                raise ValueError(f"unexpected shape for {recording_id}")
            if eeg.shape[0] != env.shape[0]:
                raise ValueError(f"length mismatch for {recording_id}: eeg={eeg.shape[0]} env={env.shape[0]}")
            self.recordings[recording_id] = (eeg, env)
            max_start = eeg.shape[0] - self.window_length
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.hop_length):
                self.records.append((recording_id, start))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        recording_id, start = self.records[idx]
        eeg, env = self.recordings[recording_id]
        eeg_window = eeg[start : start + self.window_length]
        env_window = env[start : start + self.window_length]
        return torch.from_numpy(eeg_window), torch.from_numpy(env_window)


class FeatureExtraction(nn.Module):
    def __init__(self, chans: int, filters: int, temporal_kernel: int, depth_multiplier: int, use_bias: bool = False) -> None:
        super().__init__()
        total_pad = temporal_kernel - 1
        self.pad_left = total_pad // 2
        self.pad_right = total_pad - self.pad_left
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=filters,
            kernel_size=(temporal_kernel, 1),
            padding=(0, 0),
            bias=use_bias,
        )
        self.norm1 = nn.LayerNorm(filters)
        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters * depth_multiplier,
            kernel_size=(1, chans),
            groups=filters,
            bias=use_bias,
        )
        self.norm2 = nn.LayerNorm(filters * depth_multiplier)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (0, 0, self.pad_left, self.pad_right))
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class ExactMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, key_dim: int, dropout: float, use_bias: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.inner_dim = num_heads * key_dim
        self.query = nn.Linear(embed_dim, self.inner_dim, bias=use_bias)
        self.key = nn.Linear(embed_dim, self.inner_dim, bias=use_bias)
        self.value = nn.Linear(embed_dim, self.inner_dim, bias=use_bias)
        self.output = nn.Linear(self.inner_dim, embed_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, _ = x.shape
        q = self.query(x).view(batch, steps, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.key(x).view(batch, steps, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.value(x).view(batch, steps, self.num_heads, self.key_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(float(self.key_dim))
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~valid, float("-inf"))
            all_masked = (~valid).all(dim=-1, keepdim=True)
        else:
            valid = None
            all_masked = None

        weights = torch.softmax(scores, dim=-1)
        if attention_mask is not None:
            weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        weights = self.dropout(weights)

        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, self.inner_dim)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, rate: float = 0.5, ff_dim: int = 128, use_bias: bool = True, mask: bool = True, seq_len: int = 320) -> None:
        super().__init__()
        self.register_buffer("mask", anti_causal_attention_mask(seq_len) if mask else None, persistent=False)
        self.att = ExactMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, key_dim=embed_dim, dropout=rate, use_bias=use_bias)
        self.ffnsub = nn.Conv1d(embed_dim, ff_dim, kernel_size=3, padding=1, bias=False)
        self.ffn = nn.Conv1d(ff_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(rate)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.att(x, attention_mask=self.mask)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffnsub(out1.transpose(1, 2))
        ffn_output = self.activation(ffn_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.ffn(ffn_output)
        ffn_output = self.activation(ffn_output).transpose(1, 2)
        return self.layernorm2(out1 + ffn_output)


class ADTExactRegressor(nn.Module):
    def __init__(
        self,
        chans: int = 64,
        output_dims: int = 1,
        filters: int = 8,
        temporal_kernel: int = 16,
        depth_multiplier: int = 4,
        heads: int = 4,
        ff_dim: int = 128,
        blocks: int = 4,
        mask: bool = True,
        use_bias: bool = False,
        dropout_rate: float = 0.5,
        seq_len: int = 320,
    ) -> None:
        super().__init__()
        embed_dim = filters * depth_multiplier
        self.seq_len = seq_len
        self.chans = chans
        self.feature_extraction = FeatureExtraction(
            chans=chans,
            filters=filters,
            temporal_kernel=temporal_kernel,
            depth_multiplier=depth_multiplier,
            use_bias=use_bias,
        )
        self.register_buffer("positional_encoding", positional_encoding(seq_len, embed_dim), persistent=False)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=heads,
                    rate=dropout_rate,
                    ff_dim=ff_dim,
                    use_bias=use_bias,
                    mask=mask,
                    seq_len=seq_len,
                )
                for _ in range(blocks)
            ]
        )
        self.linear_projection = nn.Linear(embed_dim, output_dims, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input shape (batch, time, channels), got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {x.shape[1]}")
        if x.shape[2] != self.chans:
            raise ValueError(f"expected channels {self.chans}, got {x.shape[2]}")

        x = x.unsqueeze(-1).permute(0, 3, 1, 2)
        x = self.feature_extraction(x)
        x = x.squeeze(2)
        x = x + self.positional_encoding[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)
        for block in self.transformer:
            x = block(x)
        return self.linear_projection(x)


@dataclass
class SubjectMetrics:
    loss: float
    pearson_metric: float


@dataclass
class TrainSummary:
    best_epoch: int
    best_val_loss: float
    best_val_metric: float
    history: dict[str, list[float]]
    subject_metrics: dict[str, SubjectMetrics]


def _run_epoch(model: nn.Module, loader: DataLoader, *, optimizer: Adam | None, device: str) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_metric = 0.0
    total_count = 0

    for eeg, env in loader:
        eeg = eeg.to(device=device, dtype=torch.float32)
        env = env.to(device=device, dtype=torch.float32)

        if training:
            optimizer.zero_grad()

        pred = model(eeg)
        loss = pearson_loss(env, pred)
        metric = pearson_metric(env, pred)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = eeg.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_metric += float(metric.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1), total_metric / max(total_count, 1)


def train_adt_exact_reference(
    input_dir: str | Path,
    *,
    participants: Iterable[str] | None = None,
    train_participant: str | None = None,
    output_dir: str | Path | None = None,
    seq_len: int = 320,
    hop_length: int = 64,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3,
    min_lr: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[ADTExactRegressor, TrainSummary]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    train_dataset = RecordingWindowDataset(
        input_dir,
        "train",
        participant=train_participant,
        window_length=seq_len,
        hop_length=hop_length,
    )
    val_dataset = RecordingWindowDataset(
        input_dir,
        "val",
        participant=train_participant,
        window_length=seq_len,
        hop_length=hop_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    model = ADTExactRegressor(seq_len=seq_len).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_metric = float("-inf")
    stale_epochs = 0
    history = {"train_loss": [], "train_pearson_metric": [], "val_loss": [], "val_pearson_metric": [], "lr": []}

    for epoch in range(epochs):
        train_loss, train_metric = _run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss, val_metric = _run_epoch(model, val_loader, optimizer=None, device=device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_pearson_metric"].append(train_metric)
        history["val_loss"].append(val_loss)
        history["val_pearson_metric"].append(val_metric)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_loss < best_val_loss:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_metric = val_metric
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)

    subject_metrics: dict[str, SubjectMetrics] = {}
    subjects = sorted({path.name.split("_-_")[1] for path in input_dir.glob("test_-_*_-_eeg.npy")})
    if participants is not None:
        wanted = set(participants)
        subjects = [subject for subject in subjects if subject in wanted]

    for subject in subjects:
        test_dataset = RecordingWindowDataset(input_dir, "test", participant=subject, window_length=seq_len, hop_length=hop_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
        test_loss, test_metric = _run_epoch(model, test_loader, optimizer=None, device=device)
        subject_metrics[subject] = SubjectMetrics(loss=test_loss, pearson_metric=test_metric)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "state_dict.pt")

    return model, TrainSummary(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_metric=best_val_metric,
        history=history,
        subject_metrics=subject_metrics,
    )
