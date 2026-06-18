from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from torch.optim import NAdam
from torch.utils.data import DataLoader, Dataset

from .mldecoders.cca import build_lag_matrix, trim_valid_range
from .mldecoders.cca import fit_cca_reconstruction, score_match_mismatch, score_reconstruction
from .mldecoders.linear_baselines import (
    fit_avgcorr_lasso_model,
    fit_avgcorr_ridge_model,
    fit_avgdec_lasso_model_fast,
    fit_avgdec_model,
    predict_linear_model,
    predict_standardized_linear_model,
    score_linear_model,
    score_standardized_linear_model,
)


def correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)


def list_reference_subjects(input_dir: str | Path, split: str = "test") -> list[str]:
    input_dir = Path(input_dir)
    subjects = sorted({path.name.split("_-_")[1] for path in input_dir.glob(f"{split}_-_*_-_eeg.npy")})
    return subjects


def load_reference_recordings(
    input_dir: str | Path,
    split: str,
    participant: str,
    *,
    channels: Iterable[int] | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    input_dir = Path(input_dir)
    channel_idx = None if channels is None else np.asarray(list(channels), dtype=int)
    recordings: list[tuple[str, np.ndarray, np.ndarray]] = []
    for eeg_path in sorted(input_dir.glob(f"{split}_-_{participant}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
        if not env_path.exists():
            continue
        eeg = np.load(eeg_path).astype(np.float32)
        env = np.load(env_path).astype(np.float32)
        if channel_idx is not None:
            eeg = eeg[:, channel_idx]
        if eeg.shape[0] != env.shape[0]:
            raise ValueError(f"length mismatch for {recording_id}: eeg={eeg.shape[0]} env={env.shape[0]}")
        recordings.append((recording_id, eeg, env[:, 0]))
    if not recordings:
        raise ValueError(f"no recordings found for split={split} participant={participant}")
    return recordings


def concatenate_reference_split(
    input_dir: str | Path,
    split: str,
    participant: str,
    *,
    channels: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    recordings = load_reference_recordings(input_dir, split, participant, channels=channels)
    x = np.concatenate([eeg for _, eeg, _ in recordings], axis=0)
    y = np.concatenate([env for _, _, env in recordings], axis=0)
    return x.astype(np.float32), y.astype(np.float32)


def split_reference_parts(
    input_dir: str | Path,
    split: str,
    participant: str,
    *,
    channels: Iterable[int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    recordings = load_reference_recordings(input_dir, split, participant, channels=channels)
    x_parts = [eeg for _, eeg, _ in recordings]
    y_parts = [env for _, _, env in recordings]
    return x_parts, y_parts


class ReferenceWindowDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        split: str,
        participant: str,
        *,
        window_size: int,
        channels: Iterable[int] | None = None,
        target_index: str = "last",
    ) -> None:
        self.window_size = int(window_size)
        self.target_index = target_index
        self.recordings = load_reference_recordings(input_dir, split, participant, channels=channels)
        self.records: list[tuple[int, int]] = []
        for rec_idx, (_, eeg, _) in enumerate(self.recordings):
            max_start = eeg.shape[0] - self.window_size
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                self.records.append((rec_idx, start))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self.records[idx]
        _, eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)


@dataclass
class DNNReferenceTrainResult:
    best_val_score: float
    best_epoch: int
    epochs_completed: int
    val_history: list[float]
    state_dict: dict


def train_dnn_reference_logged(
    input_dir: str | Path,
    participant: str,
    model_handle,
    model_kwargs: dict,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    early_stopping_patience: int,
    device: str,
    seed: int = 0,
    channels: Iterable[int] | None = None,
) -> DNNReferenceTrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_handle(**model_kwargs).to(device)
    optimizer = NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = ReferenceWindowDataset(input_dir, "train", participant, window_size=model.input_length, channels=channels)
    val_dataset = ReferenceWindowDataset(input_dir, "val", participant, window_size=model.input_length, channels=channels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    best_val = -np.inf
    best_epoch = -1
    best_state = deepcopy(model.state_dict())
    val_history: list[float] = []

    for epoch in range(epochs):
        if best_epoch >= 0 and epoch > best_epoch + early_stopping_patience:
            break

        model.train()
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            loss = -correlation(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                y_hat = model(x)
                scores.append(correlation(y, y_hat).item())
        val_score = float(np.mean(scores))
        val_history.append(val_score)

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    return DNNReferenceTrainResult(
        best_val_score=float(best_val),
        best_epoch=int(best_epoch),
        epochs_completed=len(val_history),
        val_history=val_history,
        state_dict=best_state,
    )


def evaluate_dnn_reference_on_test(
    input_dir: str | Path,
    participant: str,
    model_handle,
    model_kwargs: dict,
    state_dict: dict,
    *,
    device: str,
    channels: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    test_dataset = ReferenceWindowDataset(input_dir, "test", participant, window_size=model.input_length, channels=channels)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device, dtype=torch.float32)
            preds.append(model(x).detach().cpu().numpy())
            targets.append(y.numpy())
    predictions = np.concatenate(preds).astype(np.float32)
    y_test = np.concatenate(targets).astype(np.float32)
    score = float(pearsonr(predictions, y_test)[0])
    return predictions, y_test, score


def fit_reference_linear_variant(
    input_dir: str | Path,
    participant: str,
    variant: str,
    *,
    start_lag: int = 0,
    end_lag: int = 50,
    alphas: list[float] | None = None,
    channels: Iterable[int] | None = None,
) -> tuple[object, dict]:
    alphas = alphas or [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    x_train_parts, y_train_parts = split_reference_parts(input_dir, "train", participant, channels=channels)
    x_val, y_val = concatenate_reference_split(input_dir, "val", participant, channels=channels)

    best_model = None
    best_summary = None
    for alpha in alphas:
        if variant == "avgdec_ridge":
            model = fit_avgdec_model(x_train_parts, y_train_parts, start_lag=start_lag, end_lag=end_lag, alpha=alpha, method="ridge")
            val = score_linear_model(model, x_val, y_val)
        elif variant == "avgdec_lasso":
            model = fit_avgdec_lasso_model_fast(x_train_parts, y_train_parts, start_lag=start_lag, end_lag=end_lag, alpha=alpha)
            val = score_standardized_linear_model(model, x_val, y_val)
        elif variant == "avgcorr_ridge":
            model = fit_avgcorr_ridge_model(x_train_parts, y_train_parts, start_lag=start_lag, end_lag=end_lag, alpha=alpha)
            val = score_linear_model(model, x_val, y_val)
        elif variant == "avgcorr_lasso":
            model = fit_avgcorr_lasso_model(x_train_parts, y_train_parts, start_lag=start_lag, end_lag=end_lag, alpha=alpha)
            val = score_linear_model(model, x_val, y_val)
        else:
            raise ValueError(variant)
        if best_summary is None or val["pearson"] > best_summary["val_pearson"]:
            best_model = model
            best_summary = {"variant": variant, "best_alpha": float(alpha), "val_pearson": float(val["pearson"])}
    if best_model is None or best_summary is None:
        raise RuntimeError(f"failed to fit {variant}")
    return best_model, best_summary


def fit_reference_ridge(
    input_dir: str | Path,
    participant: str,
    *,
    start_lag: int = 0,
    end_lag: int = 50,
    alphas: list[float] | None = None,
    channels: Iterable[int] | None = None,
) -> tuple[object, dict]:
    alphas = alphas or [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    x_train, y_train = concatenate_reference_split(input_dir, "train", participant, channels=channels)
    x_val, y_val = concatenate_reference_split(input_dir, "val", participant, channels=channels)
    x_train_lag, _, y_train_center = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag, _, y_val_center = trim_valid_range(x_val, y_val, start_lag, end_lag)

    best_model = None
    best_summary = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train_lag, y_train_center)
        val_pred = model.predict(x_val_lag)
        val_pearson = float(pearsonr(val_pred, y_val_center)[0])
        if best_summary is None or val_pearson > best_summary["val_pearson"]:
            best_model = model
            best_summary = {"variant": "ridge", "best_alpha": float(alpha), "val_pearson": val_pearson, "start_lag": start_lag, "end_lag": end_lag}
    if best_model is None or best_summary is None:
        raise RuntimeError("failed to fit ridge")
    return best_model, best_summary


def evaluate_reference_ridge(
    input_dir: str | Path,
    participant: str,
    model,
    *,
    start_lag: int = 0,
    end_lag: int = 50,
    channels: Iterable[int] | None = None,
) -> dict:
    x_test, y_test = concatenate_reference_split(input_dir, "test", participant, channels=channels)
    x_test_lag, _, y_test_center = trim_valid_range(x_test, y_test, start_lag, end_lag)
    pred = model.predict(x_test_lag).astype(np.float32)
    target = y_test_center.astype(np.float32)
    return {"prediction": pred, "target": target, "pearson": float(pearsonr(pred, target)[0])}


def evaluate_reference_linear_variant(
    input_dir: str | Path,
    participant: str,
    variant: str,
    model,
    *,
    channels: Iterable[int] | None = None,
) -> dict:
    x_test, y_test = concatenate_reference_split(input_dir, "test", participant, channels=channels)
    if variant == "avgdec_lasso":
        return score_standardized_linear_model(model, x_test, y_test)
    return score_linear_model(model, x_test, y_test)


def fit_reference_cca(
    input_dir: str | Path,
    participant: str,
    *,
    start_lag: int = 0,
    end_lag: int = 50,
    channels: Iterable[int] | None = None,
) -> tuple[object, dict]:
    x_train, y_train = concatenate_reference_split(input_dir, "train", participant, channels=channels)
    x_val, y_val = concatenate_reference_split(input_dir, "val", participant, channels=channels)
    model, fit_metrics = fit_cca_reconstruction(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        start_lag=start_lag,
        end_lag=end_lag,
        n_components_grid=[1, 2, 4],
        x_pca_grid=[32, 64, 128],
        y_pca_grid=[8, 16, 32],
        alpha_grid=[0.1, 1.0, 10.0, 100.0],
    )
    return model, fit_metrics


def evaluate_reference_cca(
    input_dir: str | Path,
    participant: str,
    model,
    *,
    channels: Iterable[int] | None = None,
) -> dict:
    x_test, y_test = concatenate_reference_split(input_dir, "test", participant, channels=channels)
    recon = score_reconstruction(model, x_test, y_test)
    mm = score_match_mismatch(model, x_test, y_test, window=320, stride=64, mismatch_shift=320)
    return {
        "prediction": recon["prediction"],
        "target": recon["target"],
        "canonical_corr": recon["canonical_corr"],
        "recon_corr": recon["recon_corr"],
        "match_mismatch_accuracy": mm["accuracy"],
        "match_mismatch_margin_mean": mm["margin_mean"],
    }
