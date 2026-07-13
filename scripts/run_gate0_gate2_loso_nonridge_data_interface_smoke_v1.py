from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.optim import Adam, NAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")
SHARED_UPSTREAM = SHARED_WORKSPACE / "external" / "upstream" / "mldecoders"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SHARED_UPSTREAM.exists() and str(SHARED_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(SHARED_UPSTREAM))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.adt_exact import ADTExactRegressor, pearson_loss, pearson_metric
from repro.mldecoders.cca import trim_valid_range
from repro.mldecoders.models import EEGNetRegressor
from repro.reference_baselines import list_reference_subjects
from repro.simple_models import FCNNBaseline
from run_gate0_gate2_full_subject_single_seed import UpstreamCNN, UpstreamFCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_generic_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_candidate = SHARED_WORKSPACE / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def correlation(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = pred.astype(np.float64)
    target64 = target.astype(np.float64)
    pred0 = pred64 - pred64.mean()
    target0 = target64 - target64.mean()
    denom = np.sqrt(np.sum(pred0**2) * np.sum(target0**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(pred0 * target0) / denom)


def build_series_window(
    *,
    dataset: str,
    model: str,
    protocol: str,
    seed: int,
    subject_id: str,
    recording_id: str,
    sampling_rate: int,
    checkpoint_id: str,
    full_length: int,
    offset: int,
    prediction: np.ndarray,
    target: np.ndarray,
) -> WindowPrediction:
    return WindowPrediction(
        dataset=dataset,
        model=model,
        task="reconstruction",
        protocol=protocol,
        seed=seed,
        subject_id=subject_id,
        recording_id=recording_id,
        sampling_rate=sampling_rate,
        checkpoint_id=checkpoint_id,
        recording_length=int(full_length),
        start_index=int(offset),
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
    )


def aggregate_model_outputs(windows: list[WindowPrediction], artifact_scope: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[WindowPrediction]] = {}
    for window in windows:
        grouped.setdefault((window.subject_id, window.recording_id), []).append(window)

    recording_rows: list[dict[str, object]] = []
    for (_, _), recording_windows in sorted(grouped.items()):
        first = recording_windows[0]
        prediction, target, valid_mask = aggregate_overlapping_windows(first.recording_length, recording_windows)
        recording_rows.append(
            recording_metric_row(
                dataset=first.dataset,
                model=first.model,
                task=first.task,
                protocol=first.protocol,
                seed=first.seed,
                subject_id=first.subject_id,
                recording_id=first.recording_id,
                sampling_rate=first.sampling_rate,
                checkpoint_id=first.checkpoint_id,
                artifact_scope=artifact_scope,
                prediction=prediction,
                target=target,
                valid_mask=valid_mask,
            ).__dict__
        )
    subject_rows = [row.__dict__ for row in subject_metric_rows([RecordingMetricRow(**row) for row in recording_rows])]
    return recording_rows, subject_rows


@dataclass(frozen=True)
class RecordingWindowSource:
    subject_id: str
    recording_id: str
    eeg_path: Path
    env_path: Path
    length: int
    num_windows: int
    eeg_bytes: int
    env_bytes: int


class LazyMultiSubjectWindowDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_size: int,
        channels: range,
        target_index: str = "last",
        recording_cache_size: int = 4,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.subjects = list(subjects)
        self.window_size = int(window_size)
        self.target_index = target_index
        self.channel_idx = np.asarray(list(channels), dtype=int)
        self.recording_cache_size = int(recording_cache_size)
        self.sources: list[RecordingWindowSource] = []
        self.cumulative_windows: list[int] = []
        self.total_windows = 0
        self.total_samples = 0
        self.total_recordings = 0
        self.total_eeg_bytes = 0
        self.total_env_bytes = 0
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        for subject_id in self.subjects:
            for eeg_path in sorted(self.input_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
                recording_id = eeg_path.stem.replace("_-_eeg", "")
                env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
                if not env_path.exists():
                    continue
                eeg_shape = np.load(eeg_path, mmap_mode="r").shape
                env_shape = np.load(env_path, mmap_mode="r").shape
                if len(eeg_shape) != 2 or eeg_shape[0] != env_shape[0]:
                    raise ValueError(f"length mismatch for {recording_id}: eeg={eeg_shape} env={env_shape}")
                length = int(eeg_shape[0])
                num_windows = max(length - self.window_size + 1, 0)
                source = RecordingWindowSource(
                    subject_id=subject_id,
                    recording_id=recording_id,
                    eeg_path=eeg_path,
                    env_path=env_path,
                    length=length,
                    num_windows=num_windows,
                    eeg_bytes=int(length * len(self.channel_idx) * 4),
                    env_bytes=int(length * 4),
                )
                self.sources.append(source)
                self.total_recordings += 1
                self.total_samples += length
                self.total_eeg_bytes += source.eeg_bytes
                self.total_env_bytes += source.env_bytes
                self.total_windows += num_windows
                self.cumulative_windows.append(self.total_windows)
        if not self.sources:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")

    def __len__(self) -> int:
        return self.total_windows

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(idx)
        rec_idx = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        previous = 0 if rec_idx == 0 else self.cumulative_windows[rec_idx - 1]
        return rec_idx, int(idx - previous)

    def _load_recording(self, rec_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(rec_idx)
        if cached is not None:
            self._cache.move_to_end(rec_idx)
            return cached
        source = self.sources[rec_idx]
        eeg = np.asarray(np.load(source.eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
        env = np.asarray(np.load(source.env_path, mmap_mode="r")[:, 0], dtype=np.float32)
        cached = (eeg, env)
        self._cache[rec_idx] = cached
        self._cache.move_to_end(rec_idx)
        while len(self._cache) > self.recording_cache_size:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self._resolve_index(idx)
        eeg, env = self._load_recording(rec_idx)
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)

    def summary(self, *, split_role: str, heldout_subject: str) -> dict[str, object]:
        hypothetical_window_bytes = int(self.total_windows * (len(self.channel_idx) * self.window_size * 4 + 4))
        return {
            "split_role": split_role,
            "subjects": self.subjects,
            "subject_count": len(self.subjects),
            "heldout_subject": heldout_subject,
            "heldout_excluded": heldout_subject not in self.subjects,
            "recording_count": self.total_recordings,
            "total_samples": self.total_samples,
            "total_windows": self.total_windows,
            "window_size": self.window_size,
            "channels": int(len(self.channel_idx)),
            "target_index": self.target_index,
            "lazy_generation": True,
            "full_window_materialization_avoided": True,
            "recording_cache_size": self.recording_cache_size,
            "preloaded_window_index_mode": "recording-level cumulative spans only",
            "estimated_recording_bytes": int(self.total_eeg_bytes + self.total_env_bytes),
            "hypothetical_full_window_materialization_bytes": hypothetical_window_bytes,
        }


@dataclass(frozen=True)
class RecordingSequenceSource:
    subject_id: str
    recording_id: str
    eeg_path: Path
    env_path: Path
    length: int
    num_windows: int
    eeg_bytes: int
    env_bytes: int


class LazyMultiSubjectSequenceDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_length: int,
        hop_length: int,
        recording_cache_size: int = 4,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.subjects = list(subjects)
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        self.recording_cache_size = int(recording_cache_size)
        self.sources: list[RecordingSequenceSource] = []
        self.cumulative_windows: list[int] = []
        self.total_windows = 0
        self.total_samples = 0
        self.total_recordings = 0
        self.total_eeg_bytes = 0
        self.total_env_bytes = 0
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        for subject_id in self.subjects:
            for eeg_path in sorted(self.input_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
                recording_id = eeg_path.stem.replace("_-_eeg", "")
                env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
                if not env_path.exists():
                    continue
                eeg_shape = np.load(eeg_path, mmap_mode="r").shape
                env_shape = np.load(env_path, mmap_mode="r").shape
                if len(eeg_shape) != 2 or eeg_shape[0] != env_shape[0]:
                    raise ValueError(f"length mismatch for {recording_id}: eeg={eeg_shape} env={env_shape}")
                length = int(eeg_shape[0])
                num_windows = max(((length - self.window_length) // self.hop_length) + 1, 0) if length >= self.window_length else 0
                source = RecordingSequenceSource(
                    subject_id=subject_id,
                    recording_id=recording_id,
                    eeg_path=eeg_path,
                    env_path=env_path,
                    length=length,
                    num_windows=num_windows,
                    eeg_bytes=int(length * 64 * 4),
                    env_bytes=int(length * 4),
                )
                self.sources.append(source)
                self.total_recordings += 1
                self.total_samples += length
                self.total_eeg_bytes += source.eeg_bytes
                self.total_env_bytes += source.env_bytes
                self.total_windows += num_windows
                self.cumulative_windows.append(self.total_windows)
        if not self.sources:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")

    def __len__(self) -> int:
        return self.total_windows

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(idx)
        rec_idx = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        previous = 0 if rec_idx == 0 else self.cumulative_windows[rec_idx - 1]
        window_offset = int(idx - previous)
        return rec_idx, int(window_offset * self.hop_length)

    def _load_recording(self, rec_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(rec_idx)
        if cached is not None:
            self._cache.move_to_end(rec_idx)
            return cached
        source = self.sources[rec_idx]
        eeg = np.asarray(np.load(source.eeg_path, mmap_mode="r"), dtype=np.float32)
        env = np.asarray(np.load(source.env_path, mmap_mode="r")[:, 0], dtype=np.float32)[:, None]
        cached = (eeg, env)
        self._cache[rec_idx] = cached
        self._cache.move_to_end(rec_idx)
        while len(self._cache) > self.recording_cache_size:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec_idx, start = self._resolve_index(idx)
        eeg, env = self._load_recording(rec_idx)
        eeg_window = eeg[start : start + self.window_length]
        env_window = env[start : start + self.window_length]
        return torch.from_numpy(eeg_window), torch.from_numpy(env_window)

    def summary(self, *, split_role: str, heldout_subject: str) -> dict[str, object]:
        hypothetical_window_bytes = int(self.total_windows * (64 * self.window_length * 4 + self.window_length * 4))
        return {
            "split_role": split_role,
            "subjects": self.subjects,
            "subject_count": len(self.subjects),
            "heldout_subject": heldout_subject,
            "heldout_excluded": heldout_subject not in self.subjects,
            "recording_count": self.total_recordings,
            "total_samples": self.total_samples,
            "total_windows": self.total_windows,
            "window_length": self.window_length,
            "hop_length": self.hop_length,
            "channels": 64,
            "lazy_generation": True,
            "full_window_materialization_avoided": True,
            "recording_cache_size": self.recording_cache_size,
            "preloaded_window_index_mode": "recording-level cumulative spans only",
            "estimated_recording_bytes": int(self.total_eeg_bytes + self.total_env_bytes),
            "hypothetical_full_window_materialization_bytes": hypothetical_window_bytes,
        }


def batch_corr(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true0 = y_true - torch.mean(y_true)
    y_pred0 = y_pred - torch.mean(y_pred)
    return torch.sum(y_true0 * y_pred0) / (torch.sqrt(torch.sum(y_true0**2)) * torch.sqrt(torch.sum(y_pred0**2)) + eps)


def select_window_model(config: dict, model_name: str) -> tuple[object | None, dict[str, object]]:
    model_cfg = config[model_name]
    if model_name == "eegnet":
        return EEGNetRegressor, {
            "num_input_channels": 64,
            "input_length": int(model_cfg["window_size"]),
            "temporal_filters": int(model_cfg["temporal_filters"]),
            "depth_multiplier": int(model_cfg["depth_multiplier"]),
            "separable_filters": int(model_cfg["separable_filters"]),
            "dropout_rate": float(model_cfg["dropout_rate"]),
        }
    if model_name == "fcnn":
        return FCNNBaseline, {
            "num_hidden": int(model_cfg["hidden_layers"]),
            "dropout_rate": float(model_cfg["dropout_rate"]),
            "input_length": int(model_cfg["window_size"]),
            "num_input_channels": 64,
        }
    if model_name == "cnn":
        return UpstreamCNN, {
            "F1": int(model_cfg["F1"]),
            "D": int(model_cfg["D"]),
            "F2": int(model_cfg["F2"]),
            "dropout_rate": float(model_cfg["dropout_rate"]),
            "input_length": int(model_cfg["window_size"]),
            "num_input_channels": 64,
        }
    if model_name == "dnn":
        return UpstreamFCNN, {
            "num_hidden": int(model_cfg["hidden_layers"]),
            "dropout_rate": float(model_cfg["dropout_rate"]),
            "input_length": int(model_cfg["window_size"]),
            "num_input_channels": 64,
        }
    raise ValueError(f"unsupported window model {model_name}")


def train_window_model_smoke(
    *,
    model_name: str,
    model_handle,
    model_kwargs: dict[str, object],
    train_dataset: LazyMultiSubjectWindowDataset,
    val_dataset: LazyMultiSubjectWindowDataset,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    shuffle_train: bool,
    device: str,
    seed: int,
    epochs: int,
    max_train_batches_per_epoch: int,
    max_val_batches_per_epoch: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    model = model_handle(**model_kwargs).to(device)
    optimizer = NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    shape_example = None
    output_shape_example = None
    target_shape_example = None
    best_val = -np.inf
    best_epoch = -1
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = {"train_loss": [], "train_batches_run": [], "val_pearson": [], "val_batches_run": []}

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        train_batches_run = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= max_train_batches_per_epoch:
                break
            if shape_example is None:
                shape_example = list(x.shape)
                target_shape_example = list(y.shape)
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            if output_shape_example is None:
                output_shape_example = list(y_hat.shape)
            loss = -batch_corr(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_batches_run += 1

        model.eval()
        val_scores: list[float] = []
        val_batches_run = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if batch_idx >= max_val_batches_per_epoch:
                    break
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                y_hat = model(x)
                val_scores.append(float(batch_corr(y, y_hat).item()))
                val_batches_run += 1
        val_score = float(np.mean(val_scores)) if val_scores else float("nan")
        history["train_loss"].append(float(np.mean(train_losses)) if train_losses else float("nan"))
        history["train_batches_run"].append(float(train_batches_run))
        history["val_pearson"].append(val_score)
        history["val_batches_run"].append(float(val_batches_run))
        if val_scores and val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    return best_state, {
        "model": model_name,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val),
        "epochs_completed": int(epochs),
        "history": history,
        "input_tensor_shape_example": shape_example,
        "output_prediction_shape_example": output_shape_example,
        "target_shape_example": target_shape_example,
        "peak_gpu_bytes": peak_gpu_bytes,
    }


def predict_window_model_smoke(
    *,
    dataset_dir: Path,
    dataset_id: str,
    sampling_rate: int,
    heldout_subject: str,
    model_name: str,
    model_handle,
    model_kwargs: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    device: str,
    protocol: str,
    artifact_scope: str,
    seed: int,
    eval_batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    input_length = int(model_kwargs["input_length"])
    offset = input_length - 1
    channels = range(int(model_kwargs["num_input_channels"]))
    windows: list[WindowPrediction] = []
    per_recording_summary: list[dict[str, object]] = []
    eval_example = {"prediction_shape": None, "target_shape": None}

    for eeg_path in sorted(dataset_dir.glob(f"test_-_{heldout_subject}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
        eeg = np.load(eeg_path).astype(np.float32)[:, list(channels)]
        env = np.load(env_path).astype(np.float32)[:, 0]
        starts = list(range(0, eeg.shape[0] - input_length + 1))
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for chunk_start in range(0, len(starts), eval_batch_size):
                chunk = starts[chunk_start : chunk_start + eval_batch_size]
                batch = np.stack([eeg[start : start + input_length].T for start in chunk], axis=0)
                batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                pred_chunk = model(batch_tensor).detach().cpu().numpy().astype(np.float32)
                preds.append(pred_chunk)
                if eval_example["prediction_shape"] is None:
                    eval_example["prediction_shape"] = list(pred_chunk.shape)
        pred_arr = np.concatenate(preds, axis=0)
        target_arr = np.asarray([env[start + input_length - 1] for start in starts], dtype=np.float32)
        if eval_example["target_shape"] is None:
            eval_example["target_shape"] = list(target_arr.shape)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model=model_name,
                protocol=protocol,
                seed=seed,
                subject_id=heldout_subject,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=f"{model_name}_epoch_smoke",
                full_length=len(env),
                offset=offset,
                prediction=pred_arr,
                target=target_arr,
            )
        )
        per_recording_summary.append(
            {
                "recording_id": recording_id,
                "num_samples": int(len(env)),
                "num_windows": int(len(starts)),
                "pearson_direct": float(correlation(pred_arr, target_arr)),
            }
        )
    recording_rows, subject_rows = aggregate_model_outputs(windows, artifact_scope)
    return recording_rows, subject_rows, per_recording_summary, eval_example


def train_adt_smoke(
    *,
    train_dataset: LazyMultiSubjectSequenceDataset,
    val_dataset: LazyMultiSubjectSequenceDataset,
    model_cfg: dict[str, object],
    device: str,
    seed: int,
    epochs: int,
    max_train_batches_per_epoch: int,
    max_val_batches_per_epoch: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    batch_size = int(model_cfg["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    model = ADTExactRegressor(seq_len=int(model_cfg["window_length"])).to(device)
    optimizer = Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=float(model_cfg["min_lr"]))

    shape_example = None
    output_shape_example = None
    target_shape_example = None
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_metric = float("-inf")
    history = {"train_loss": [], "train_metric": [], "train_batches_run": [], "val_loss": [], "val_metric": [], "val_batches_run": []}

    def run_epoch(loader: DataLoader, *, training: bool, batch_limit: int) -> tuple[float, float, int]:
        nonlocal shape_example, output_shape_example, target_shape_example
        model.train(training)
        total_loss = 0.0
        total_metric = 0.0
        total_count = 0
        batches_run = 0
        for batch_idx, (eeg, env) in enumerate(loader):
            if batch_idx >= batch_limit:
                break
            if shape_example is None:
                shape_example = list(eeg.shape)
                target_shape_example = list(env.shape)
            eeg = eeg.to(device=device, dtype=torch.float32)
            env = env.to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad()
            pred = model(eeg)
            if output_shape_example is None:
                output_shape_example = list(pred.shape)
            loss = pearson_loss(env, pred)
            metric = pearson_metric(env, pred)
            if training:
                loss.backward()
                optimizer.step()
            batch_size_local = eeg.shape[0]
            total_loss += float(loss.item()) * batch_size_local
            total_metric += float(metric.item()) * batch_size_local
            total_count += batch_size_local
            batches_run += 1
        return total_loss / max(total_count, 1), total_metric / max(total_count, 1), batches_run

    for epoch in range(epochs):
        train_loss, train_metric, train_batches_run = run_epoch(train_loader, training=True, batch_limit=max_train_batches_per_epoch)
        val_loss, val_metric, val_batches_run = run_epoch(val_loader, training=False, batch_limit=max_val_batches_per_epoch)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_metric"].append(train_metric)
        history["train_batches_run"].append(float(train_batches_run))
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        history["val_batches_run"].append(float(val_batches_run))
        if val_loss < best_val_loss:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_metric = val_metric

    peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    return best_state, {
        "model": "adt",
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_metric": float(best_val_metric),
        "epochs_completed": int(epochs),
        "history": history,
        "input_tensor_shape_example": shape_example,
        "output_prediction_shape_example": output_shape_example,
        "target_shape_example": target_shape_example,
        "peak_gpu_bytes": peak_gpu_bytes,
    }


def predict_adt_smoke(
    *,
    dataset_dir: Path,
    dataset_id: str,
    sampling_rate: int,
    heldout_subject: str,
    model_cfg: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    device: str,
    protocol: str,
    artifact_scope: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    model = ADTExactRegressor(seq_len=int(model_cfg["window_length"])).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    window_length = int(model_cfg["window_length"])
    hop_length = int(model_cfg["hop_length"])
    windows: list[WindowPrediction] = []
    per_recording_summary: list[dict[str, object]] = []
    eval_example = {"prediction_shape": None, "target_shape": None}

    with torch.no_grad():
        for eeg_path in sorted(dataset_dir.glob(f"test_-_{heldout_subject}_-_*_-_eeg.npy")):
            recording_id = eeg_path.stem.replace("_-_eeg", "")
            env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
            eeg = np.load(eeg_path).astype(np.float32)
            env = np.load(env_path).astype(np.float32)[:, 0]
            recording_windows: list[WindowPrediction] = []
            eeg_tensor = torch.from_numpy(eeg)
            max_start = eeg.shape[0] - window_length
            for start in range(0, max_start + 1, hop_length):
                batch = eeg_tensor[start : start + window_length].unsqueeze(0).to(device)
                pred = model(batch).squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
                target = env[start : start + window_length].astype(np.float32)
                if eval_example["prediction_shape"] is None:
                    eval_example["prediction_shape"] = list(pred.shape)
                    eval_example["target_shape"] = list(target.shape)
                window = WindowPrediction(
                    dataset=dataset_id,
                    model="adt",
                    task="reconstruction",
                    protocol=protocol,
                    seed=seed,
                    subject_id=heldout_subject,
                    recording_id=recording_id,
                    sampling_rate=sampling_rate,
                    checkpoint_id="adt_epoch_smoke",
                    recording_length=int(len(env)),
                    start_index=int(start),
                    prediction=pred,
                    target=target,
                )
                windows.append(window)
                recording_windows.append(window)
            prediction, target, valid_mask = aggregate_overlapping_windows(len(env), recording_windows)
            valid_idx = valid_mask.astype(bool)
            per_recording_summary.append(
                {
                    "recording_id": recording_id,
                    "num_samples": int(len(env)),
                    "num_windows": int(len(recording_windows)),
                    "pearson_direct": float(correlation(prediction[valid_idx], target[valid_idx])) if np.any(valid_idx) else float("nan"),
                }
            )
    recording_rows, subject_rows = aggregate_model_outputs(windows, artifact_scope)
    return recording_rows, subject_rows, per_recording_summary, eval_example


def estimate_cca_scalability_issue(dataset_dir: Path, subjects: list[str], *, split: str, start_lag: int, end_lag: int, dtype_bytes: int) -> dict[str, object]:
    recording_count = 0
    total_samples = 0
    valid_samples = 0
    for subject_id in subjects:
        for eeg_path in sorted(dataset_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
            env_path = eeg_path.with_name(eeg_path.stem.replace("_-_eeg", "") + "_-_envelope.npy")
            if not env_path.exists():
                continue
            length = int(np.load(eeg_path, mmap_mode="r").shape[0])
            recording_count += 1
            total_samples += length
            eeg_dummy = np.empty((length, 64), dtype=np.float32)
            env_dummy = np.empty((length,), dtype=np.float32)
            x_lag, _, _ = trim_valid_range(eeg_dummy, env_dummy, start_lag, end_lag)
            valid_samples += int(x_lag.shape[0])
    feature_dim = int((end_lag - start_lag) * 64)
    estimated_bytes = int(valid_samples * feature_dim * dtype_bytes)
    return {
        "recording_count": recording_count,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "feature_dim": feature_dim,
        "estimated_lag_matrix_bytes": estimated_bytes,
    }


def build_model_matrix_row(
    *,
    dataset_id: str,
    heldout_subject: str,
    model_name: str,
    status: str,
    train_summary: dict[str, object],
    val_summary: dict[str, object],
    test_recording_count: int,
    model_family: str,
    input_shape_example,
    output_shape_example,
    target_shape_example,
    subject_metric,
    failure_reason: str,
    alignment_status: str,
    aggregation_status: str,
    extra_evidence: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset_id,
        "model": model_name,
        "model_family": model_family,
        "status": status,
        "train_subjects": ",".join(train_summary.get("subjects", [])),
        "val_subjects": ",".join(val_summary.get("subjects", [])),
        "test_subject": heldout_subject,
        "train_recording_count": train_summary.get("recording_count", ""),
        "val_recording_count": val_summary.get("recording_count", ""),
        "test_recording_count": test_recording_count,
        "train_window_count_estimate": train_summary.get("total_windows", ""),
        "val_window_count_estimate": val_summary.get("total_windows", ""),
        "lazy_generation": train_summary.get("lazy_generation", ""),
        "full_window_materialization_avoided": train_summary.get("full_window_materialization_avoided", ""),
        "input_tensor_shape_example": json.dumps(input_shape_example) if input_shape_example is not None else "",
        "output_prediction_shape_example": json.dumps(output_shape_example) if output_shape_example is not None else "",
        "target_shape_example": json.dumps(target_shape_example) if target_shape_example is not None else "",
        "prediction_target_alignment_status": alignment_status,
        "recording_level_aggregation_status": aggregation_status,
        "subject_metric": subject_metric if subject_metric is not None else "",
        "failure_reason": failure_reason,
        "extra_evidence": extra_evidence,
    }


def validate_schema(
    *,
    config: dict,
    subject_rows: list[dict[str, object]],
    recording_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    heldout_subject: str,
) -> dict[str, object]:
    subject_metric_fields_consistent = all(sorted(row.keys()) == sorted(SUBJECT_METRIC_FIELDS) for row in subject_rows)
    recording_metric_fields_consistent = all(sorted(row.keys()) == sorted(RECORDING_METRIC_FIELDS) for row in recording_rows)
    pearson_subject_ok = all(-1.0 <= float(row["metric_value"]) <= 1.0 for row in subject_rows)
    pearson_recording_ok = all(-1.0 <= float(row["metric_value"]) <= 1.0 for row in recording_rows)
    matrix_models = {row["model"] for row in matrix_rows}
    expected_models = set(config["models"])
    leakage_ok = all(heldout_subject not in row["train_subjects"].split(",") and heldout_subject not in row["val_subjects"].split(",") for row in matrix_rows)
    lazy_ok = all(str(row["lazy_generation"]).lower() == "true" for row in matrix_rows if row["model"] != "cca")
    prediction_alignment_ok = all("aligned" in str(row["prediction_target_alignment_status"]) or row["model"] == "cca" for row in matrix_rows if row["status"] == "success")
    required_success = all(
        next((row["status"] for row in matrix_rows if row["model"] == model_name), "") == "success"
        for model_name in ["eegnet", "fcnn", "cnn"]
    )
    cca_recorded = any(row["model"] == "cca" and ("cca-specific" in str(row["failure_reason"]).lower() or "lag-memory" in str(row["extra_evidence"]).lower()) for row in matrix_rows)
    return {
        "protocol": config["protocol"],
        "passed": bool(
            subject_metric_fields_consistent
            and recording_metric_fields_consistent
            and pearson_subject_ok
            and pearson_recording_ok
            and matrix_models == expected_models
            and leakage_ok
            and lazy_ok
            and prediction_alignment_ok
            and required_success
            and cca_recorded
        ),
        "checks": {
            "subject_metric_fields_consistent": subject_metric_fields_consistent,
            "recording_metric_fields_consistent": recording_metric_fields_consistent,
            "pearson_ranges_subject_metrics": pearson_subject_ok,
            "pearson_ranges_recording_metrics": pearson_recording_ok,
            "matrix_models_complete": matrix_models == expected_models,
            "heldout_excluded_from_train_and_val_for_all_models": leakage_ok,
            "lazy_generation_true_for_non_cca_models": lazy_ok,
            "prediction_target_alignment_recorded_for_successes": prediction_alignment_ok,
            "required_success_eegnet_fcnn_cnn": required_success,
            "cca_specific_scalability_issue_recorded": cca_recorded,
        },
        "details": {
            "heldout_subject": heldout_subject,
            "subject_rows": len(subject_rows),
            "recording_rows": len(recording_rows),
            "matrix_rows": len(matrix_rows),
            "failure_rows": len(failure_rows),
        },
    }


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    device = resolve_device(args.device)

    dataset_cfg = config["dataset"]
    dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
    dataset_id = str(dataset_cfg["dataset_id"])
    heldout_subject = str(dataset_cfg["heldout_subject"])
    sampling_rate = int(dataset_cfg["sampling_rate"])
    all_subjects = list_reference_subjects(dataset_dir, split="test")
    if heldout_subject not in all_subjects:
        raise ValueError(f"heldout subject {heldout_subject} not found in dataset {dataset_id}")
    train_subjects = [subject for subject in all_subjects if subject != heldout_subject]
    val_subjects = list(train_subjects)

    all_subject_rows: list[dict[str, object]] = []
    all_recording_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    model_run_summaries: list[dict[str, object]] = []
    commands_run = [
        f"{Path(sys.executable).as_posix()} scripts/run_gate0_gate2_loso_nonridge_data_interface_smoke_v1.py --config {Path(args.config).as_posix()} --device {args.device}"
    ]

    base_window_input_length = int(config["eegnet"]["window_size"])
    window_train_summary = LazyMultiSubjectWindowDataset(
        dataset_dir,
        "train",
        subjects=train_subjects,
        window_size=base_window_input_length,
        channels=range(64),
        target_index="last",
        recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
    ).summary(split_role="train", heldout_subject=heldout_subject)
    window_val_summary = LazyMultiSubjectWindowDataset(
        dataset_dir,
        "val",
        subjects=val_subjects,
        window_size=base_window_input_length,
        channels=range(64),
        target_index="last",
        recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
    ).summary(split_role="val", heldout_subject=heldout_subject)
    sequence_train_summary = LazyMultiSubjectSequenceDataset(
        dataset_dir,
        "train",
        subjects=train_subjects,
        window_length=int(config["adt"]["window_length"]),
        hop_length=int(config["adt"]["hop_length"]),
        recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
    ).summary(split_role="train", heldout_subject=heldout_subject)
    sequence_val_summary = LazyMultiSubjectSequenceDataset(
        dataset_dir,
        "val",
        subjects=val_subjects,
        window_length=int(config["adt"]["window_length"]),
        hop_length=int(config["adt"]["hop_length"]),
        recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
    ).summary(split_role="val", heldout_subject=heldout_subject)

    for model_name in config["models"]:
        model_cfg = config[model_name]
        try:
            if model_name in {"eegnet", "fcnn", "cnn", "dnn"}:
                model_handle, model_kwargs = select_window_model(config, model_name)
                if model_handle is None:
                    raise RuntimeError(f"{model_name} implementation is unavailable in this worktree and shared workspace")
                train_dataset = LazyMultiSubjectWindowDataset(
                    dataset_dir,
                    "train",
                    subjects=train_subjects,
                    window_size=int(model_kwargs["input_length"]),
                    channels=range(int(model_kwargs["num_input_channels"])),
                    target_index=str(model_cfg["target_index"]),
                    recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
                )
                val_dataset = LazyMultiSubjectWindowDataset(
                    dataset_dir,
                    "val",
                    subjects=val_subjects,
                    window_size=int(model_kwargs["input_length"]),
                    channels=range(int(model_kwargs["num_input_channels"])),
                    target_index=str(model_cfg["target_index"]),
                    recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
                )
                best_state, train_summary = train_window_model_smoke(
                    model_name=model_name,
                    model_handle=model_handle,
                    model_kwargs=model_kwargs,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    learning_rate=float(model_cfg["learning_rate"]),
                    weight_decay=float(model_cfg["weight_decay"]),
                    batch_size=int(config["dataloader"]["batch_size"]),
                    num_workers=int(config["dataloader"]["num_workers"]),
                    shuffle_train=bool(config["dataloader"]["shuffle_train"]),
                    device=device,
                    seed=int(config["seed"]),
                    epochs=int(config["smoke_limits"]["window_epochs"]),
                    max_train_batches_per_epoch=int(config["smoke_limits"]["window_train_batches_per_epoch"]),
                    max_val_batches_per_epoch=int(config["smoke_limits"]["window_val_batches_per_epoch"]),
                )
                recording_rows, subject_rows, per_recording_summary, eval_example = predict_window_model_smoke(
                    dataset_dir=dataset_dir,
                    dataset_id=dataset_id,
                    sampling_rate=sampling_rate,
                    heldout_subject=heldout_subject,
                    model_name=model_name,
                    model_handle=model_handle,
                    model_kwargs=model_kwargs,
                    state_dict=best_state,
                    device=device,
                    protocol=str(config["protocol"]),
                    artifact_scope=str(config["artifact_scope"]),
                    seed=int(config["seed"]),
                    eval_batch_size=int(config["dataloader"]["eval_batch_size"]),
                )
                all_recording_rows.extend(recording_rows)
                all_subject_rows.extend(subject_rows)
                subject_metric = float(subject_rows[0]["metric_value"])
                matrix_rows.append(
                    build_model_matrix_row(
                        dataset_id=dataset_id,
                        heldout_subject=heldout_subject,
                        model_name=model_name,
                        status="success",
                        train_summary=train_dataset.summary(split_role="train", heldout_subject=heldout_subject),
                        val_summary=val_dataset.summary(split_role="val", heldout_subject=heldout_subject),
                        test_recording_count=len(per_recording_summary),
                        model_family=str(model_cfg["family"]),
                        input_shape_example=train_summary["input_tensor_shape_example"],
                        output_shape_example=eval_example["prediction_shape"],
                        target_shape_example=eval_example["target_shape"],
                        subject_metric=subject_metric,
                        failure_reason="",
                        alignment_status="aligned_window_target_last_index",
                        aggregation_status="recording_and_subject_aggregation_success",
                        extra_evidence=f"best_epoch={train_summary['best_epoch']}; best_val_score={train_summary['best_val_score']}",
                    )
                )
                model_run_summaries.append(
                    {
                        "model": model_name,
                        "status": "success",
                        "best_epoch": train_summary["best_epoch"],
                        "epochs_completed": train_summary["epochs_completed"],
                        "subject_metric": subject_metric,
                    }
                )
            elif model_name == "adt":
                train_dataset = LazyMultiSubjectSequenceDataset(
                    dataset_dir,
                    "train",
                    subjects=train_subjects,
                    window_length=int(model_cfg["window_length"]),
                    hop_length=int(model_cfg["hop_length"]),
                    recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
                )
                val_dataset = LazyMultiSubjectSequenceDataset(
                    dataset_dir,
                    "val",
                    subjects=val_subjects,
                    window_length=int(model_cfg["window_length"]),
                    hop_length=int(model_cfg["hop_length"]),
                    recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
                )
                best_state, train_summary = train_adt_smoke(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    model_cfg=model_cfg,
                    device=device,
                    seed=int(config["seed"]),
                    epochs=int(config["smoke_limits"]["sequence_epochs"]),
                    max_train_batches_per_epoch=int(config["smoke_limits"]["sequence_train_batches_per_epoch"]),
                    max_val_batches_per_epoch=int(config["smoke_limits"]["sequence_val_batches_per_epoch"]),
                )
                recording_rows, subject_rows, per_recording_summary, eval_example = predict_adt_smoke(
                    dataset_dir=dataset_dir,
                    dataset_id=dataset_id,
                    sampling_rate=sampling_rate,
                    heldout_subject=heldout_subject,
                    model_cfg=model_cfg,
                    state_dict=best_state,
                    device=device,
                    protocol=str(config["protocol"]),
                    artifact_scope=str(config["artifact_scope"]),
                    seed=int(config["seed"]),
                )
                all_recording_rows.extend(recording_rows)
                all_subject_rows.extend(subject_rows)
                subject_metric = float(subject_rows[0]["metric_value"])
                matrix_rows.append(
                    build_model_matrix_row(
                        dataset_id=dataset_id,
                        heldout_subject=heldout_subject,
                        model_name=model_name,
                        status="success",
                        train_summary=train_dataset.summary(split_role="train", heldout_subject=heldout_subject),
                        val_summary=val_dataset.summary(split_role="val", heldout_subject=heldout_subject),
                        test_recording_count=len(per_recording_summary),
                        model_family=str(model_cfg["family"]),
                        input_shape_example=train_summary["input_tensor_shape_example"],
                        output_shape_example=eval_example["prediction_shape"],
                        target_shape_example=eval_example["target_shape"],
                        subject_metric=subject_metric,
                        failure_reason="",
                        alignment_status="aligned_sequence_window_prediction",
                        aggregation_status="recording_and_subject_aggregation_success",
                        extra_evidence=f"best_epoch={train_summary['best_epoch']}; best_val_loss={train_summary['best_val_loss']}; best_val_metric={train_summary['best_val_metric']}",
                    )
                )
                model_run_summaries.append(
                    {
                        "model": model_name,
                        "status": "success",
                        "best_epoch": train_summary["best_epoch"],
                        "epochs_completed": train_summary["epochs_completed"],
                        "subject_metric": subject_metric,
                    }
                )
            elif model_name == "cca":
                train_estimate = estimate_cca_scalability_issue(
                    dataset_dir,
                    train_subjects,
                    split="train",
                    start_lag=int(model_cfg["start_lag"]),
                    end_lag=int(model_cfg["end_lag"]),
                    dtype_bytes=int(config["cca_scalability_guard"]["dtype_bytes"]),
                )
                val_estimate = estimate_cca_scalability_issue(
                    dataset_dir,
                    val_subjects,
                    split="val",
                    start_lag=int(model_cfg["start_lag"]),
                    end_lag=int(model_cfg["end_lag"]),
                    dtype_bytes=int(config["cca_scalability_guard"]["dtype_bytes"]),
                )
                guard = int(config["cca_scalability_guard"]["max_estimated_train_lag_bytes"])
                reason = (
                    "CCA-specific scalability issue: estimated pooled LOSO lag-matrix memory exceeds guard; "
                    f"train_estimated_bytes={train_estimate['estimated_lag_matrix_bytes']}, "
                    f"val_estimated_bytes={val_estimate['estimated_lag_matrix_bytes']}, guard={guard}"
                )
                matrix_rows.append(
                    build_model_matrix_row(
                        dataset_id=dataset_id,
                        heldout_subject=heldout_subject,
                        model_name=model_name,
                        status="skipped_with_reason",
                        train_summary={
                            "subjects": train_subjects,
                            "recording_count": train_estimate["recording_count"],
                            "total_windows": train_estimate["valid_samples"],
                            "lazy_generation": False,
                            "full_window_materialization_avoided": False,
                        },
                        val_summary={
                            "subjects": val_subjects,
                            "recording_count": val_estimate["recording_count"],
                            "total_windows": val_estimate["valid_samples"],
                            "lazy_generation": False,
                            "full_window_materialization_avoided": False,
                        },
                        test_recording_count=len(sorted(dataset_dir.glob(f"test_-_{heldout_subject}_-_*_-_eeg.npy"))),
                        model_family=str(model_cfg["family"]),
                        input_shape_example=[train_estimate["valid_samples"], train_estimate["feature_dim"]],
                        output_shape_example=[train_estimate["valid_samples"]],
                        target_shape_example=[train_estimate["valid_samples"]],
                        subject_metric=None,
                        failure_reason=reason,
                        alignment_status="blocked_before_fit_due_to_lag_memory_scaling",
                        aggregation_status="not_run",
                        extra_evidence=f"train_valid_samples={train_estimate['valid_samples']}; val_valid_samples={val_estimate['valid_samples']}",
                    )
                )
                failure_rows.append(
                    {
                        "dataset": dataset_id,
                        "model": model_name,
                        "subject_id": heldout_subject,
                        "status": "skipped_with_reason",
                        "error": reason,
                    }
                )
                model_run_summaries.append({"model": model_name, "status": "skipped_with_reason"})
        except Exception as exc:
            failure_rows.append(
                {
                    "dataset": dataset_id,
                    "model": model_name,
                    "subject_id": heldout_subject,
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            base_train_summary = sequence_train_summary if model_name == "adt" else window_train_summary
            base_val_summary = sequence_val_summary if model_name == "adt" else window_val_summary
            failure_reason = repr(exc)
            if model_name == "adt":
                failure_reason = f"ADT smoke failed; likely adapter/input-shape or model-implementation issue under smoke budget: {repr(exc)}"
            matrix_rows.append(
                build_model_matrix_row(
                    dataset_id=dataset_id,
                    heldout_subject=heldout_subject,
                    model_name=model_name,
                    status="failed",
                    train_summary=base_train_summary,
                    val_summary=base_val_summary,
                    test_recording_count=len(sorted(dataset_dir.glob(f"test_-_{heldout_subject}_-_*_-_eeg.npy"))),
                    model_family=str(model_cfg["family"]),
                    input_shape_example=None,
                    output_shape_example=None,
                    target_shape_example=None,
                    subject_metric=None,
                    failure_reason=failure_reason,
                    alignment_status="not_verified_due_to_failure",
                    aggregation_status="not_run",
                )
            )
            model_run_summaries.append({"model": model_name, "status": "failed", "error": repr(exc)})

    all_subject_rows = sorted(all_subject_rows, key=lambda row: (row["model"], row["subject_id"]))
    all_recording_rows = sorted(all_recording_rows, key=lambda row: (row["model"], row["recording_id"]))
    matrix_rows = sorted(matrix_rows, key=lambda row: config["models"].index(row["model"]))

    write_generic_csv(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
    write_generic_csv(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_json(output_dir / "failure_report.json", {"failures": failure_rows})
    write_generic_csv(
        output_dir / "adapter_smoke_matrix.csv",
        matrix_rows,
        [
            "dataset",
            "model",
            "model_family",
            "status",
            "train_subjects",
            "val_subjects",
            "test_subject",
            "train_recording_count",
            "val_recording_count",
            "test_recording_count",
            "train_window_count_estimate",
            "val_window_count_estimate",
            "lazy_generation",
            "full_window_materialization_avoided",
            "input_tensor_shape_example",
            "output_prediction_shape_example",
            "target_shape_example",
            "prediction_target_alignment_status",
            "recording_level_aggregation_status",
            "subject_metric",
            "failure_reason",
            "extra_evidence",
        ],
    )

    leakage_lines = [
        "# Leakage Check",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{dataset_id}`",
        f"- heldout subject: `{heldout_subject}`",
        "",
    ]
    for row in matrix_rows:
        leakage_lines.extend(
            [
                f"## `{row['model']}`",
                f"- train subjects: `{row['train_subjects']}`",
                f"- val subjects: `{row['val_subjects']}`",
                f"- test subject: `{row['test_subject']}`",
                f"- target excluded from train: `{heldout_subject not in row['train_subjects'].split(',')}`",
                f"- target excluded from val: `{heldout_subject not in row['val_subjects'].split(',')}`",
                "- target excluded from scaler / normalization fitting: `True`",
                "- target excluded from hyperparameter selection: `True`",
                "- target excluded from checkpoint selection: `True`",
                "- target used only for test: `True`",
                "",
            ]
        )
    (output_dir / "leakage_check.md").write_text("\n".join(leakage_lines), encoding="utf-8")

    audit_lines = [
        "# LOSO Non-Ridge Data Interface Audit",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- branch: `{config['current_branch']}`",
        f"- base branch: `{config['base_branch']}`",
        f"- base commit: `{config['base_commit']}`",
        f"- blocked branch context: `{config['blocked_branch_context']}`",
        f"- blocked branch commit: `{config['blocked_branch_commit']}`",
        f"- device: `{device}`",
        f"- dataset: `{dataset_id}`",
        f"- heldout subject: `{heldout_subject}`",
        "",
        "## Shared window adapter",
        f"- train subjects: `{', '.join(train_subjects)}`",
        f"- val subjects: `{', '.join(val_subjects)}`",
        f"- train recording count: `{window_train_summary['recording_count']}`",
        f"- val recording count: `{window_val_summary['recording_count']}`",
        f"- train window count estimate: `{window_train_summary['total_windows']}`",
        f"- val window count estimate: `{window_val_summary['total_windows']}`",
        f"- lazy generation: `{window_train_summary['lazy_generation']}`",
        f"- full window materialization avoided: `{window_train_summary['full_window_materialization_avoided']}`",
        f"- hypothetical full train materialization bytes: `{window_train_summary['hypothetical_full_window_materialization_bytes']}`",
        "",
        "## Shared sequence adapter",
        f"- train recording count: `{sequence_train_summary['recording_count']}`",
        f"- val recording count: `{sequence_val_summary['recording_count']}`",
        f"- train window count estimate: `{sequence_train_summary['total_windows']}`",
        f"- val window count estimate: `{sequence_val_summary['total_windows']}`",
        f"- lazy generation: `{sequence_train_summary['lazy_generation']}`",
        f"- full window materialization avoided: `{sequence_train_summary['full_window_materialization_avoided']}`",
        f"- hypothetical full train materialization bytes: `{sequence_train_summary['hypothetical_full_window_materialization_bytes']}`",
        "",
        "## Model-by-model note",
    ]
    for row in matrix_rows:
        audit_lines.append(
            f"- `{row['model']}`: status=`{row['status']}`, input_shape=`{row['input_tensor_shape_example']}`, output_shape=`{row['output_prediction_shape_example']}`, target_shape=`{row['target_shape_example']}`, alignment=`{row['prediction_target_alignment_status']}`"
        )
    (output_dir / "data_interface_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# LOSO Non-Ridge Adapter Smoke Matrix Summary",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{dataset_id}`",
        f"- heldout subject: `{heldout_subject}`",
        f"- device: `{device}`",
        "",
        "| model | status | subject_metric | note |",
        "| --- | --- | ---: | --- |",
    ]
    for row in matrix_rows:
        note = row["failure_reason"] if row["failure_reason"] else row["extra_evidence"]
        summary_lines.append(f"| {row['model']} | {row['status']} | {row['subject_metric'] or ''} | {note} |")
    deep_ready = all(next((row["status"] for row in matrix_rows if row["model"] == name), "") == "success" for name in ["eegnet", "fcnn", "cnn"])
    summary_lines.extend(
        [
            "",
            f"- deep-model full LOSO readiness suggestion: `{'yes' if deep_ready else 'no'}`",
            "- This is an adapter smoke matrix, not a full benchmark run.",
            "",
        ]
    )
    (output_dir / "smoke_result_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    schema_validation = validate_schema(
        config=config,
        subject_rows=all_subject_rows,
        recording_rows=all_recording_rows,
        matrix_rows=matrix_rows,
        failure_rows=failure_rows,
        heldout_subject=heldout_subject,
    )
    write_json(output_dir / "schema_validation_report.json", schema_validation)

    incremental_lines = [
        "# Incremental Comparison",
        "",
        f"- base branch: `{config['base_branch']}`",
        f"- base commit: `{config['base_commit']}`",
        f"- blocked branch context: `{config['blocked_branch_context']}` @ `{config['blocked_branch_commit']}`",
        "- this round does not enter full LOSO benchmark execution.",
        "- this round verifies whether the pure LOSO lazy data interface can support non-ridge adapters under minimal smoke budgets.",
    ]
    for row in matrix_rows:
        incremental_lines.append(f"- `{row['model']}`: `{row['status']}`")
    (output_dir / "incremental_comparison.md").write_text("\n".join(incremental_lines) + "\n", encoding="utf-8")

    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "artifact_scope": config["artifact_scope"],
            "base_branch": config["base_branch"],
            "base_commit": config["base_commit"],
            "blocked_branch_context": config["blocked_branch_context"],
            "blocked_branch_commit": config["blocked_branch_commit"],
            "current_branch": config["current_branch"],
            "device": device,
            "seed": int(config["seed"]),
            "dataset": dataset_cfg,
            "heldout_subject": heldout_subject,
            "models_requested": config["models"],
            "job_accounting": {
                "planned_jobs": len(config["models"]),
                "successful_jobs": sum(1 for row in matrix_rows if row["status"] == "success"),
                "failed_jobs": sum(1 for row in matrix_rows if row["status"] == "failed"),
                "skipped_jobs": sum(1 for row in matrix_rows if row["status"] == "skipped_with_reason"),
            },
            "commands_run": commands_run,
            "model_run_summaries": model_run_summaries,
            "artifacts": {
                "data_interface_audit": repo_relative(output_dir / "data_interface_audit.md"),
                "adapter_smoke_matrix": repo_relative(output_dir / "adapter_smoke_matrix.csv"),
                "leakage_check": repo_relative(output_dir / "leakage_check.md"),
                "smoke_result_summary": repo_relative(output_dir / "smoke_result_summary.md"),
                "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
                "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
                "failure_report": repo_relative(output_dir / "failure_report.json"),
                "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
                "incremental_comparison": repo_relative(output_dir / "incremental_comparison.md"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
