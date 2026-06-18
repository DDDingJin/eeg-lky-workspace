from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .adt_exact import RecordingWindowDataset, pearson_loss, pearson_metric


class VLAAIExtractorExact(nn.Module):
    def __init__(
        self,
        input_channels: int = 64,
        filters: tuple[int, ...] = (256, 256, 256, 128, 128),
        kernels: tuple[int, ...] = (8, 8, 8, 8, 8),
    ) -> None:
        super().__init__()
        if len(filters) != len(kernels):
            raise ValueError("filters and kernels must have the same length")

        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels, kernel in zip(filters, kernels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=0))
            layers.append(nn.ConstantPad1d((0, kernel - 1), 0.0))
            layers.append(nn.LeakyReLU())
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(0, len(self.layers), 3):
            conv = self.layers[idx]
            pad = self.layers[idx + 1]
            act = self.layers[idx + 2]
            x = conv(x)
            x = x.transpose(1, 2)
            x = nn.functional.layer_norm(x, x.shape[-1:])
            x = x.transpose(1, 2)
            x = act(x)
            x = pad(x)
        return x


class VLAAIOutputContextExact(nn.Module):
    def __init__(self, input_channels: int = 64, filter_: int = 64, kernel: int = 32) -> None:
        super().__init__()
        self.pad = nn.ConstantPad1d((kernel - 1, 0), 0.0)
        self.conv = nn.Conv1d(input_channels, filter_, kernel_size=kernel, padding=0)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = nn.functional.layer_norm(x, x.shape[-1:])
        x = x.transpose(1, 2)
        return self.activation(x)


class VLAAIExactOfficial(nn.Module):
    def __init__(
        self,
        nb_blocks: int = 4,
        input_channels: int = 64,
        output_dim: int = 1,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.nb_blocks = nb_blocks
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.use_skip = use_skip

        self.extractor = VLAAIExtractorExact(input_channels=input_channels)
        self.output_context = VLAAIOutputContextExact(input_channels=input_channels)
        self.block_projections = nn.ModuleList([nn.Linear(128, input_channels, bias=True) for _ in range(nb_blocks)])
        self.final_projection = nn.Linear(input_channels, output_dim, bias=True)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"expected input shape (batch, time, channels), got {tuple(eeg.shape)}")
        x = torch.zeros_like(eeg) if self.use_skip else eeg

        for block_idx in range(self.nb_blocks):
            source = eeg + x if self.use_skip else x
            current = self.extractor(source.transpose(1, 2)).transpose(1, 2)
            current = self.block_projections[block_idx](current)
            x = self.output_context(current.transpose(1, 2)).transpose(1, 2)

        return self.final_projection(x)


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


def train_vlaai_exact_reference(
    input_dir: str | Path,
    *,
    participants: Iterable[str] | None = None,
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
) -> tuple[VLAAIExactOfficial, TrainSummary]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    train_dataset = RecordingWindowDataset(input_dir, "train", window_length=seq_len, hop_length=hop_length)
    val_dataset = RecordingWindowDataset(input_dir, "val", window_length=seq_len, hop_length=hop_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    model = VLAAIExactOfficial().to(device)
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
