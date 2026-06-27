from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from .reference_baselines import list_reference_subjects, load_reference_recordings


UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "external" / "upstream" / "HappyQuokka_system_for_EEG_Challenge"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from models.FFT_block import Decoder  # type: ignore


def l1_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_hat - y_true), dim=1)


def pearson_metric(y_hat: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_hat = y_hat.squeeze(-1)
    y_true = y_true.squeeze(-1)
    y_hat = y_hat - y_hat.mean(dim=1, keepdim=True)
    y_true = y_true - y_true.mean(dim=1, keepdim=True)
    num = torch.sum(y_hat * y_true, dim=1)
    den = torch.sqrt(torch.sum(y_hat**2, dim=1) * torch.sum(y_true**2, dim=1) + eps)
    return num / den


def pearson_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return -pearson_metric(y_hat, y_true)


class HappyQuokkaReferenceDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        split: str,
        *,
        input_length: int,
        channels: Iterable[int] | None = None,
        subject_to_id: dict[str, int],
        g_con: bool,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.input_length = int(input_length)
        self.channels = channels
        self.subject_to_id = subject_to_id
        self.g_con = g_con
        self.items: list[tuple[str, np.ndarray, np.ndarray]] = []
        for subject in list_reference_subjects(self.input_dir, split=split):
            self.items.extend(load_reference_recordings(self.input_dir, split, subject, channels=channels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        recording_id, eeg, env = self.items[index]
        subject = recording_id.split("_-_")[1]
        subject_id = self.subject_to_id[subject] if self.g_con else 0
        if self.split == "train":
            start = np.random.randint(0, eeg.shape[0] - self.input_length + 1)
            eeg_chunk = eeg[start : start + self.input_length]
            env_chunk = env[start : start + self.input_length]
            return (
                torch.from_numpy(eeg_chunk.astype(np.float32)),
                torch.from_numpy(env_chunk.astype(np.float32)).unsqueeze(-1),
                torch.tensor(subject_id, dtype=torch.long),
            )

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
class HappyQuokkaTrainSummary:
    best_val_metric: float
    best_epoch: int
    epochs_completed: int
    history: dict[str, list[float]]
    state_dict: dict
    subject_to_id: dict[str, int]


def _evaluate_recording_batch(model: torch.nn.Module, batch, device: str, g_con: bool, lamda: float) -> tuple[float, float]:
    eeg_segments, env_segments, subject_id, *_ = batch
    eeg_segments = eeg_segments.squeeze(0).to(device)
    env_segments = env_segments.squeeze(0).to(device)
    subject_id = subject_id.to(device)
    if subject_id.ndim == 0:
        subject_ids = subject_id.repeat(eeg_segments.shape[0])
    else:
        subject_ids = subject_id.view(-1).repeat(eeg_segments.shape[0])
    outputs = model(eeg_segments, subject_ids)
    loss = (pearson_loss(outputs, env_segments) + lamda * l1_loss(outputs, env_segments)).mean()
    metric = pearson_metric(outputs, env_segments).mean()
    return float(loss.item()), float(metric.item())


def train_happyquokka_reference(
    input_dir: str | Path,
    *,
    win_len_seconds: int = 10,
    sample_rate: int = 64,
    batch_size: int = 64,
    epochs: int = 1000,
    learning_rate: float = 5e-4,
    dropout: float = 0.3,
    lamda: float = 0.2,
    g_con: bool = True,
    device: str = "cuda",
    channels: Iterable[int] | None = None,
) -> tuple[torch.nn.Module, HappyQuokkaTrainSummary]:
    input_dir = Path(input_dir)
    subjects = list_reference_subjects(input_dir, split="train")
    subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
    input_length = int(win_len_seconds * sample_rate)

    train_set = HappyQuokkaReferenceDataset(
        input_dir,
        "train",
        input_length=input_length,
        channels=channels,
        subject_to_id=subject_to_id,
        g_con=g_con,
    )
    val_set = HappyQuokkaReferenceDataset(
        input_dir,
        "val",
        input_length=input_length,
        channels=channels,
        subject_to_id=subject_to_id,
        g_con=g_con,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = Decoder(
        in_channel=len(list(channels)) if channels is not None else 64,
        d_model=128,
        d_inner=1024,
        n_head=2,
        n_layers=8,
        fft_conv1d_kernel=(9, 1),
        fft_conv1d_padding=(4, 0),
        dropout=dropout,
        g_con=g_con,
        within_sub_num=max(1, len(subject_to_id)),
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
            loss = (pearson_loss(outputs, env_batch) + lamda * l1_loss(outputs, env_batch)).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                loss, metric = _evaluate_recording_batch(model, batch, device, g_con, lamda)
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
        raise RuntimeError("HappyQuokka training failed to produce a checkpoint")

    model.load_state_dict(best_state)
    summary = HappyQuokkaTrainSummary(
        best_val_metric=best_metric,
        best_epoch=best_epoch,
        epochs_completed=epochs,
        history=history,
        state_dict=best_state,
        subject_to_id=subject_to_id,
    )
    return model, summary


def evaluate_happyquokka_reference(
    input_dir: str | Path,
    *,
    state_dict: dict,
    subject_to_id: dict[str, int],
    win_len_seconds: int = 10,
    sample_rate: int = 64,
    dropout: float = 0.3,
    g_con: bool = True,
    device: str = "cuda",
    channels: Iterable[int] | None = None,
) -> dict:
    input_dir = Path(input_dir)
    input_length = int(win_len_seconds * sample_rate)
    test_set = HappyQuokkaReferenceDataset(
        input_dir,
        "test",
        input_length=input_length,
        channels=channels,
        subject_to_id=subject_to_id,
        g_con=g_con,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model = Decoder(
        in_channel=len(list(channels)) if channels is not None else 64,
        d_model=128,
        d_inner=1024,
        n_head=2,
        n_layers=8,
        fft_conv1d_kernel=(9, 1),
        fft_conv1d_padding=(4, 0),
        dropout=dropout,
        g_con=g_con,
        within_sub_num=max(1, len(subject_to_id)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    per_subject: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in test_loader:
            _, metric, subject = None, None, None
            loss, metric = _evaluate_recording_batch(model, batch, device, g_con, lamda=0.2)
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
