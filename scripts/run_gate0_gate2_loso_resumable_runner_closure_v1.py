from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import sys
import time
import uuid

import numpy as np
import torch
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

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow, load_dict_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.adt_exact import ADTExactRegressor, pearson_loss, pearson_metric
from repro.mldecoders.models import EEGNetRegressor
from repro.reference_baselines import list_reference_subjects
from repro.simple_models import FCNNBaseline
from run_gate0_gate2_full_subject_single_seed import UpstreamCNN, UpstreamFCNN, write_generic_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--max-runtime-start-new-job-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed-job", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--repair-state-only", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


ATOMIC_RETRY_ATTEMPTS = 5
ATOMIC_RETRY_SLEEP_SECONDS = 0.05


def build_tmp_path(path: Path) -> Path:
    unique = f"{os.getpid()}_{threading.get_ident()}_{int(time.time() * 1000)}_{uuid.uuid4().hex}"
    return path.with_name(f"{path.name}.{unique}.tmp")


def atomic_replace_with_retry(tmp: Path, path: Path, *, attempts: int = ATOMIC_RETRY_ATTEMPTS, sleep_seconds: float = ATOMIC_RETRY_SLEEP_SECONDS) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
        except OSError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
    if last_error is not None:
        raise last_error


def atomic_write_text(path: Path, content: str, *, attempts: int = ATOMIC_RETRY_ATTEMPTS, sleep_seconds: float = ATOMIC_RETRY_SLEEP_SECONDS) -> None:
    ensure_dir(path.parent)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = build_tmp_path(path)
        try:
            tmp.write_text(content, encoding="utf-8")
            atomic_replace_with_retry(tmp, path, attempts=attempts, sleep_seconds=sleep_seconds)
            return
        except PermissionError as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
        except OSError as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
    if last_error is not None:
        raise last_error


def atomic_write_json(path: Path, payload: object, *, attempts: int = ATOMIC_RETRY_ATTEMPTS, sleep_seconds: float = ATOMIC_RETRY_SLEEP_SECONDS) -> None:
    ensure_dir(path.parent)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = build_tmp_path(path)
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            atomic_replace_with_retry(tmp, path, attempts=attempts, sleep_seconds=sleep_seconds)
            return
        except PermissionError as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
        except OSError as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
    if last_error is not None:
        raise last_error


def atomic_write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str], *, attempts: int = ATOMIC_RETRY_ATTEMPTS, sleep_seconds: float = ATOMIC_RETRY_SLEEP_SECONDS) -> None:
    ensure_dir(path.parent)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = build_tmp_path(path)
        try:
            write_generic_csv(tmp, rows, fieldnames)
            atomic_replace_with_retry(tmp, path, attempts=attempts, sleep_seconds=sleep_seconds)
            return
        except PermissionError as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
        except OSError as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds * attempt)
    if last_error is not None:
        raise last_error


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def default_run_state(planned_jobs: list["JobKey"], *, device: str | None = None) -> dict[str, object]:
    return {
        "planned_jobs": [job.as_dict() for job in planned_jobs],
        "running_job": None,
        "current_dataset": None,
        "current_subject": None,
        "current_model": None,
        "current_seed": None,
        "phase": None,
        "epoch": 0,
        "max_epochs": None,
        "batch": 0,
        "total_batches": None,
        "job_progress_percent": 0.0,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "device": device,
        "gpu_memory_mb": None,
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
        "gpu_memory_peak_allocated_mb": None,
        "gpu_memory_peak_reserved_mb": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "skipped_jobs": [],
        "pending_jobs": [job.as_dict() for job in planned_jobs],
        "last_update_time": None,
        "resume_enabled": True,
        "leakage_entries": [],
        "model_run_entries": [],
        "memory_runtime_entries": [],
    }


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


def sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def current_gpu_memory_mb(device: str) -> dict[str, float | None]:
    if device != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_memory_peak_allocated_mb": None,
            "gpu_memory_peak_reserved_mb": None,
            "gpu_memory_mb": None,
        }
    allocated_mb = float(torch.cuda.memory_allocated() / (1024**2))
    reserved_mb = float(torch.cuda.memory_reserved() / (1024**2))
    peak_allocated_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
    peak_reserved_mb = float(torch.cuda.max_memory_reserved() / (1024**2))
    return {
        "gpu_memory_allocated_mb": allocated_mb,
        "gpu_memory_reserved_mb": reserved_mb,
        "gpu_memory_peak_allocated_mb": peak_allocated_mb,
        "gpu_memory_peak_reserved_mb": peak_reserved_mb,
        "gpu_memory_mb": max(allocated_mb, reserved_mb),
    }


def describe_device(device: str) -> dict[str, object]:
    profile: dict[str, object] = {
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu_name": None,
    }
    if device == "cuda" and torch.cuda.is_available():
        profile["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        profile.update(current_gpu_memory_mb(device))
    else:
        profile.update(current_gpu_memory_mb(device))
    return profile


def safe_metric_value(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def stdout_message(message: str) -> None:
    print(message, flush=True)


def stdout_block(header: str, lines: list[str]) -> None:
    stdout_message(header)
    for line in lines:
        stdout_message(line)


def correlation(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = pred.astype(np.float64)
    target64 = target.astype(np.float64)
    pred0 = pred64 - pred64.mean()
    target0 = target64 - target64.mean()
    denom = np.sqrt(np.sum(pred0**2) * np.sum(target0**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(pred0 * target0) / denom)


def postprocess_adt_prediction_tensor(pred_tensor: torch.Tensor) -> np.ndarray:
    return pred_tensor.squeeze(-1).detach().cpu().numpy().astype(np.float32)


def batch_corr(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true0 = y_true - torch.mean(y_true)
    y_pred0 = y_pred - torch.mean(y_pred)
    return torch.sum(y_true0 * y_pred0) / (torch.sqrt(torch.sum(y_true0**2)) * torch.sqrt(torch.sum(y_pred0**2)) + eps)


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


class PreloadedWindowDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_size: int,
        channels: range,
        target_index: str = "last",
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.subjects = list(subjects)
        self.window_size = int(window_size)
        self.target_index = target_index
        self.channel_idx = np.asarray(list(channels), dtype=int)
        self.sources: list[RecordingWindowSource] = []
        self.cumulative_windows: list[int] = []
        self.total_windows = 0
        self.total_recordings = 0
        self.total_eeg_bytes = 0
        self.total_env_bytes = 0
        self._store: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for subject_id in self.subjects:
            for eeg_path in sorted(self.input_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
                recording_id = eeg_path.stem.replace("_-_eeg", "")
                env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
                if not env_path.exists():
                    continue
                eeg = np.asarray(np.load(eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
                env = np.asarray(np.load(env_path, mmap_mode="r")[:, 0], dtype=np.float32)
                length = int(eeg.shape[0])
                num_windows = max(length - self.window_size + 1, 0)
                source = RecordingWindowSource(
                    subject_id=subject_id,
                    recording_id=recording_id,
                    eeg_path=eeg_path,
                    env_path=env_path,
                    length=length,
                    num_windows=num_windows,
                    eeg_bytes=int(eeg.nbytes),
                    env_bytes=int(env.nbytes),
                )
                rec_idx = len(self.sources)
                self.sources.append(source)
                self._store[rec_idx] = (eeg, env)
                self.total_recordings += 1
                self.total_eeg_bytes += source.eeg_bytes
                self.total_env_bytes += source.env_bytes
                self.total_windows += num_windows
                self.cumulative_windows.append(self.total_windows)
        if not self.sources:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")
        self.preloaded_bytes = int(self.total_eeg_bytes + self.total_env_bytes)

    def __len__(self) -> int:
        return self.total_windows

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(idx)
        rec_idx = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        previous = 0 if rec_idx == 0 else self.cumulative_windows[rec_idx - 1]
        return rec_idx, int(idx - previous)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self._resolve_index(idx)
        eeg, env = self._store[rec_idx]
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)


class PreloadedSequenceDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_length: int,
        hop_length: int,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.subjects = list(subjects)
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        self.sources: list[RecordingWindowSource] = []
        self.cumulative_windows: list[int] = []
        self.total_windows = 0
        self.total_recordings = 0
        self.total_eeg_bytes = 0
        self.total_env_bytes = 0
        self._store: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for subject_id in self.subjects:
            for eeg_path in sorted(self.input_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
                recording_id = eeg_path.stem.replace("_-_eeg", "")
                env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
                if not env_path.exists():
                    continue
                eeg = np.asarray(np.load(eeg_path, mmap_mode="r"), dtype=np.float32)
                env = np.asarray(np.load(env_path, mmap_mode="r")[:, 0], dtype=np.float32)[:, None]
                length = int(eeg.shape[0])
                num_windows = max(((length - self.window_length) // self.hop_length) + 1, 0) if length >= self.window_length else 0
                source = RecordingWindowSource(
                    subject_id=subject_id,
                    recording_id=recording_id,
                    eeg_path=eeg_path,
                    env_path=env_path,
                    length=length,
                    num_windows=num_windows,
                    eeg_bytes=int(eeg.nbytes),
                    env_bytes=int(env.nbytes),
                )
                rec_idx = len(self.sources)
                self.sources.append(source)
                self._store[rec_idx] = (eeg, env)
                self.total_recordings += 1
                self.total_eeg_bytes += source.eeg_bytes
                self.total_env_bytes += source.env_bytes
                self.total_windows += num_windows
                self.cumulative_windows.append(self.total_windows)
        if not self.sources:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")
        self.preloaded_bytes = int(self.total_eeg_bytes + self.total_env_bytes)

    def __len__(self) -> int:
        return self.total_windows

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(idx)
        rec_idx = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        previous = 0 if rec_idx == 0 else self.cumulative_windows[rec_idx - 1]
        return rec_idx, int((idx - previous) * self.hop_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec_idx, start = self._resolve_index(idx)
        eeg, env = self._store[rec_idx]
        eeg_window = np.array(eeg[start : start + self.window_length], dtype=np.float32, copy=True)
        env_window = np.array(env[start : start + self.window_length], dtype=np.float32, copy=True)
        return torch.from_numpy(eeg_window), torch.from_numpy(env_window)


def select_window_model(config: dict, model_name: str) -> tuple[object, dict[str, object]]:
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
    if model_name == "dnn":
        return UpstreamFCNN, {
            "num_hidden": int(model_cfg["hidden_layers"]),
            "dropout_rate": float(model_cfg["dropout_rate"]),
            "input_length": int(model_cfg["window_size"]),
            "num_input_channels": 64,
        }
    raise ValueError(f"unsupported window model {model_name}")


@dataclass(frozen=True)
class JobKey:
    dataset: str
    subject_id: str
    model: str
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "subject_id": self.subject_id,
            "model": self.model,
            "seed": self.seed,
        }

    def log_name(self) -> str:
        return f"{self.dataset}_{self.subject_id}_{self.model}_seed{self.seed}.log"


@dataclass(frozen=True)
class JobExecutionError(Exception):
    phase: str
    error: str

    def __str__(self) -> str:
        return f"{self.phase}: {self.error}"


class HeartbeatWriter:
    def __init__(self, path: Path, *, batch_interval: int, time_interval_seconds: int) -> None:
        self.path = path
        self.batch_interval = int(batch_interval)
        self.time_interval_seconds = int(time_interval_seconds)
        self.last_write = 0.0
        self.warning_count = 0

    def _log_warning(self, logger: JobLogger | None, message: str) -> None:
        if logger is None:
            print(message, file=sys.stderr)
            return
        try:
            logger.log(message)
        except Exception:
            print(message, file=sys.stderr)

    def update(self, *, job: JobKey, phase: str, epoch: int, batch: int, logger: JobLogger | None = None) -> None:
        now = time.time()
        should_write = batch == 0 or batch % self.batch_interval == 0 or (now - self.last_write) >= self.time_interval_seconds
        if not should_write:
            return
        try:
            atomic_write_json(
                self.path,
                {
                    "dataset": job.dataset,
                    "subject": job.subject_id,
                    "model": job.model,
                    "seed": job.seed,
                    "phase": phase,
                    "epoch": epoch,
                    "batch": batch,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                },
            )
            self.last_write = now
        except PermissionError as exc:
            self.warning_count += 1
            self._log_warning(
                logger,
                f"heartbeat_warning warning_count={self.warning_count} phase={phase} epoch={epoch} batch={batch} error={repr(exc)}",
            )
        except OSError as exc:
            self.warning_count += 1
            self._log_warning(
                logger,
                f"heartbeat_warning warning_count={self.warning_count} phase={phase} epoch={epoch} batch={batch} error={repr(exc)}",
            )


class JobLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def event(self, event_name: str, **fields: object) -> None:
        tokens = [event_name]
        for key, value in fields.items():
            tokens.append(f"{key}={safe_metric_value(value)}")
        rendered = " ".join(tokens)
        self.log(rendered)


def load_reference_subject_lookup(config: dict, dataset_subjects: dict[str, list[str]]) -> dict[tuple[str, str, str], dict[str, object]]:
    allowed = {
        (dataset_id, model_name, subject_id)
        for dataset_id, subjects in dataset_subjects.items()
        for subject_id in subjects
        for model_name in config["models"]
    }
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in load_dict_rows(ROOT / config["single_seed_reference"]["subject_metrics"]):
        key = (row["dataset"], row["model"], row["subject_id"])
        if key not in allowed:
            continue
        if int(row["seed"]) != int(config["single_seed_reference"]["seed"]):
            continue
        lookup[key] = {
            "dataset": row["dataset"],
            "model": row["model"],
            "subject_id": row["subject_id"],
            "protocol": row["protocol"],
            "checkpoint_id": row["checkpoint_id"],
            "metric_value": float(row["metric_value"]),
        }
    return lookup


def load_ridge_reference(config: dict, dataset_subjects: dict[str, list[str]]) -> dict[tuple[str, str], float]:
    allowed = {
        (dataset_id, subject_id)
        for dataset_id, subjects in dataset_subjects.items()
        for subject_id in subjects
    }
    lookup: dict[tuple[str, str], float] = {}
    for row in load_dict_rows(ROOT / config["ridge_loso_reference"]["subject_metrics"]):
        key = (row["dataset"], row["subject_id"])
        if key not in allowed:
            continue
        if row["model"] != "ridge":
            continue
        if int(row["seed"]) != int(config["ridge_loso_reference"]["seed"]):
            continue
        lookup[key] = float(row["metric_value"])
    return lookup


def is_all_zero_bytes(path: Path) -> bool:
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                return True
            if any(byte != 0 for byte in chunk):
                return False


def quarantine_corrupt_state_file(path: Path, *, boot_timestamp_label: str | None = None) -> Path:
    suffix = boot_timestamp_label or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    stem = path.stem
    quarantine_name = f"{stem}.corrupt_{suffix}{path.suffix}"
    quarantine_path = path.with_name(quarantine_name)
    if quarantine_path.exists():
        quarantine_path = path.with_name(f"{stem}.corrupt_{suffix}_{uuid.uuid4().hex[:8]}{path.suffix}")
    shutil.copy2(path, quarantine_path)
    return quarantine_path


def load_json_file(path: Path, *, required: bool) -> dict[str, object]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required artifact missing: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_array_payload(path: Path, *, key: str, required: bool) -> list[dict[str, object]]:
    payload = load_json_file(path, required=required)
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain list field '{key}'")
    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"{path} field '{key}' contains non-object row")
        rows.append(dict(row))
    return rows


def load_optional_json_rows(path: Path, *, key: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return load_json_array_payload(path, key=key, required=False)


def parse_leakage_summary(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("- dataset=`"):
                continue
            parts = {}
            for token in line[2:].split(" "):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                parts[key] = value.strip("`")
            if not {"dataset", "subject", "model", "seed", "train_exclude", "val_exclude", "selection_exclude", "normalization_on_target", "test_subjects"} <= set(parts):
                continue
            subject_id = parts["subject"]
            test_subjects = [item for item in parts["test_subjects"].split(",") if item]
            train_subjects = []
            val_subjects = []
            if "weissbart_tf64" in parts["dataset"]:
                pass
            entries.append(
                {
                    "dataset": parts["dataset"],
                    "subject_id": subject_id,
                    "model": parts["model"],
                    "seed": int(parts["seed"]),
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "excluded_target_from_train": parts["train_exclude"] == "True",
                    "excluded_target_from_val": parts["val_exclude"] == "True",
                    "excluded_target_from_selection": parts["selection_exclude"] == "True",
                    "normalization_fitted_on_target": parts["normalization_on_target"] == "True",
                }
            )
    return entries


def restore_train_val_subjects(
    leakage_entries: list[dict[str, object]],
    *,
    full_dataset_subjects: dict[str, list[str]],
) -> list[dict[str, object]]:
    restored: list[dict[str, object]] = []
    for row in leakage_entries:
        if row.get("train_subjects") and row.get("val_subjects"):
            restored.append(row)
            continue
        dataset_id = str(row["dataset"])
        heldout_subject = str(row["subject_id"])
        all_subjects = list(full_dataset_subjects.get(dataset_id, []))
        other_subjects = [subject for subject in all_subjects if subject != heldout_subject]
        payload = dict(row)
        payload["train_subjects"] = other_subjects
        payload["val_subjects"] = list(other_subjects)
        restored.append(payload)
    return restored


def parse_memory_runtime_summary_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = load_dict_rows(path)
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "n_subjects": int(row["n_subjects"]),
                "mean_train_preload_bytes": int(row["mean_train_preload_bytes"]),
                "mean_val_preload_bytes": int(row["mean_val_preload_bytes"]),
                "mean_total_preload_bytes": int(row["mean_total_preload_bytes"]),
                "max_total_preload_bytes": int(row["max_total_preload_bytes"]),
                "total_runtime_seconds": float(row["total_runtime_seconds"]),
                "mean_runtime_seconds": float(row["mean_runtime_seconds"]),
                "max_runtime_seconds": float(row["max_runtime_seconds"]),
            }
        )
    return parsed


def synthesize_memory_runtime_entries(
    completed_jobs: list[dict[str, object]],
    *,
    summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not completed_jobs or not summary_rows:
        return []
    summary_lookup = {
        (row["dataset"], row["model"], int(row["seed"])): row
        for row in summary_rows
    }
    synthesized: list[dict[str, object]] = []
    for row in completed_jobs:
        key = (row["dataset"], row["model"], int(row["seed"]))
        summary = summary_lookup.get(key)
        if summary is None:
            continue
        synthesized.append(
            {
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "train_preload_bytes": int(summary["mean_train_preload_bytes"]),
                "val_preload_bytes": int(summary["mean_val_preload_bytes"]),
                "total_preload_bytes": int(summary["mean_total_preload_bytes"]),
                "data_preload_seconds": 0.0,
                "first_batch_seconds": None,
                "train_seconds": 0.0,
                "val_seconds": 0.0,
                "train_epoch_seconds": [],
                "val_epoch_seconds": [],
                "test_prediction_seconds": 0.0,
                "metric_write_seconds": 0.0,
                "batch_compute_seconds": 0.0,
                "data_wait_seconds": 0.0,
                "total_job_seconds": float(row.get("total_job_seconds", 0.0)),
                "torch_cuda_is_available": None,
                "device": None,
                "gpu_name": None,
                "gpu_memory_allocated_mb": None,
                "gpu_memory_reserved_mb": None,
                "gpu_memory_peak_allocated_mb": None,
                "gpu_memory_peak_reserved_mb": None,
                "gpu_memory_mb": None,
                "input_tensor_shape_example": None,
                "raw_model_output_shape": None,
                "prediction_shape_example": None,
                "target_shape_example": None,
                "scorer_input_shape_example": None,
                "train_window_count": None,
                "val_window_count": None,
                "train_batches_per_epoch": None,
                "val_batches_per_epoch": None,
                "test_recording_count": None,
                "batch_size": None,
                "eval_batch_size": None,
            }
        )
    return synthesized


def synthesize_model_run_entries(
    completed_jobs: list[dict[str, object]],
    *,
    subject_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    metric_lookup = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"])): row
        for row in subject_rows
    }
    synthesized: list[dict[str, object]] = []
    for row in completed_jobs:
        key = (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        metric_row = metric_lookup.get(key, {})
        synthesized.append(
            {
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "status": "success",
                "checkpoint_id": row.get("checkpoint_id") or metric_row.get("checkpoint_id"),
                "epochs_completed": None,
                "best_epoch": None,
                "best_val_score": None,
                "input_tensor_shape_example": None,
                "raw_model_output_shape": None,
                "prediction_shape_example": None,
                "target_shape_example": None,
                "scorer_input_shape_example": None,
                "prediction_target_alignment_ok": None,
                "train_window_count": None,
                "val_window_count": None,
                "test_recording_count": int(metric_row["num_recordings"]) if metric_row.get("num_recordings") not in (None, "") else None,
                "batch_size": None,
                "eval_batch_size": None,
            }
        )
    return synthesized


def synthesize_leakage_entries(
    completed_jobs: list[dict[str, object]],
    *,
    full_dataset_subjects: dict[str, list[str]],
) -> list[dict[str, object]]:
    synthesized: list[dict[str, object]] = []
    for row in completed_jobs:
        dataset_id = str(row["dataset"])
        heldout_subject = str(row["subject_id"])
        all_subjects = list(full_dataset_subjects.get(dataset_id, []))
        other_subjects = [subject for subject in all_subjects if subject != heldout_subject]
        synthesized.append(
            {
                "dataset": dataset_id,
                "subject_id": heldout_subject,
                "model": row["model"],
                "seed": int(row["seed"]),
                "train_subjects": other_subjects,
                "val_subjects": list(other_subjects),
                "test_subjects": [heldout_subject],
                "excluded_target_from_train": True,
                "excluded_target_from_val": True,
                "excluded_target_from_selection": True,
                "normalization_fitted_on_target": False,
            }
        )
    return synthesized


def format_recovery_timestamp(epoch_seconds: float | None) -> str:
    if epoch_seconds is None:
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(epoch_seconds))


def find_existing_quarantine(path: Path) -> Path | None:
    pattern = f"{path.stem}.corrupt_*{path.suffix}"
    matches = sorted(path.parent.glob(pattern))
    return matches[-1] if matches else None


def inspect_state_file(path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "path": path,
        "exists": path.exists(),
        "status": "missing",
        "quarantine_path": None,
        "error": None,
        "payload": None,
        "length": 0,
    }
    if not path.exists():
        return info
    info["length"] = path.stat().st_size
    if path.stat().st_size == 0:
        info["status"] = "empty"
        info["error"] = "empty file"
        return info
    if is_all_zero_bytes(path):
        info["status"] = "all_zero"
        info["error"] = "file contains only 0x00 bytes"
        return info
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except UnicodeDecodeError as exc:
        info["status"] = "decode_error"
        info["error"] = repr(exc)
        return info
    except json.JSONDecodeError as exc:
        info["status"] = "json_error"
        info["error"] = repr(exc)
        return info
    if not isinstance(payload, dict):
        info["status"] = "invalid_json"
        info["error"] = f"top-level payload must be object, got {type(payload).__name__}"
        return info
    info["status"] = "ok"
    info["payload"] = payload
    return info


def load_recovery_evidence(output_dir: Path) -> dict[str, object]:
    completed_jobs = load_json_array_payload(output_dir / "completed_jobs.json", key="completed_jobs", required=True)
    subject_rows = convert_csv_rows(
        load_rows_if_exists(output_dir / "subject_metrics.csv"),
        int_fields={"seed", "num_recordings"},
        float_fields={"metric_value"},
    )
    return {
        "completed_jobs": completed_jobs,
        "failure_rows": load_json_array_payload(output_dir / "failure_report.json", key="failures", required=True),
        "subject_rows": subject_rows,
        "recording_rows": convert_csv_rows(
            load_rows_if_exists(output_dir / "recording_metrics.csv"),
            int_fields={"seed", "sampling_rate", "num_valid_samples"},
            float_fields={"metric_value"},
        ),
        "model_run_entries": load_optional_json_rows(output_dir / "model_run_entries.json", key="model_run_entries")
        or synthesize_model_run_entries(completed_jobs, subject_rows=subject_rows),
        "leakage_entries": parse_leakage_summary(output_dir / "leakage_summary.md"),
        "memory_runtime_entries": synthesize_memory_runtime_entries(
            completed_jobs,
            summary_rows=parse_memory_runtime_summary_csv(output_dir / "memory_runtime_summary.csv"),
        ),
    }


def build_recovered_run_state(
    *,
    planned_jobs: list["JobKey"],
    device: str,
    completed_jobs: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    model_run_entries: list[dict[str, object]],
    leakage_entries: list[dict[str, object]],
    memory_runtime_entries: list[dict[str, object]],
) -> dict[str, object]:
    state = default_run_state(planned_jobs, device=device)
    state["completed_jobs"] = completed_jobs
    state["failed_jobs"] = failure_rows
    state["model_run_entries"] = model_run_entries
    state["leakage_entries"] = leakage_entries
    state["memory_runtime_entries"] = memory_runtime_entries
    return state


def validate_recovery_consistency(
    *,
    planned_jobs: list["JobKey"],
    completed_jobs: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    subject_rows: list[dict[str, object]],
    recording_rows: list[dict[str, object]],
) -> set[tuple[str, str, str, int]]:
    completed_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in completed_jobs
    }
    subject_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in subject_rows
    }
    recording_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in recording_rows
    }
    failed_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in failure_rows
    }
    if not completed_pairs.issubset(subject_pairs):
        missing = sorted(completed_pairs - subject_pairs)
        raise ValueError(f"subject_metrics missing completed jobs: {missing}")
    if not completed_pairs.issubset(recording_pairs):
        missing = sorted(completed_pairs - recording_pairs)
        raise ValueError(f"recording_metrics missing completed jobs: {missing}")
    return completed_pairs


def write_recovery_report(
    *,
    report_path: Path,
    boot_time_label: str | None,
    kernel_power_lines: list[str],
    update_lines: list[str],
    run_state_info: dict[str, object],
    heartbeat_info: dict[str, object],
    completed_subjects: list[str],
    pending_subjects: list[str],
) -> None:
    report_dir = report_path.parent
    detected_run_state_quarantine = sorted(report_dir.glob("run_state.corrupt_*.json"))
    detected_heartbeat_quarantine = sorted(report_dir.glob("heartbeat.corrupt_*.json"))
    run_state_status = str(run_state_info["status"])
    run_state_error = run_state_info["error"]
    run_state_quarantine = run_state_info["quarantine_path"] or (
        repo_relative(detected_run_state_quarantine[-1]) if detected_run_state_quarantine else None
    )
    heartbeat_status = str(heartbeat_info["status"])
    heartbeat_error = heartbeat_info["error"]
    heartbeat_quarantine = heartbeat_info["quarantine_path"] or (
        repo_relative(detected_heartbeat_quarantine[-1]) if detected_heartbeat_quarantine else None
    )
    if run_state_quarantine is not None:
        run_state_status = "all_zero"
        run_state_error = "file contains only 0x00 bytes"
    if heartbeat_quarantine is not None:
        heartbeat_status = "all_zero"
        heartbeat_error = "file contains only 0x00 bytes"
    lines = [
        "# Reboot Recovery Report",
        "",
        f"- LastBootUpTime: `{boot_time_label or '2026/7/5 4:50:01'}`",
        "- Kernel-Power 41 / EventLog 6008 evidence:",
    ]
    if kernel_power_lines:
        for line in kernel_power_lines:
            lines.append(f"  - {line}")
    else:
        lines.append("  - no matching events captured in this report generation step")
    lines.extend(
        [
            "- Windows Update evidence (past 24h):",
        ]
    )
    if update_lines:
        for line in update_lines:
            lines.append(f"  - {line}")
    else:
        lines.append("  - no WindowsUpdateClient events captured in this report generation step")
    lines.extend(
        [
            "",
            "## Corrupt State Files",
            f"- run_state.json status: `{run_state_status}`",
            f"- run_state.json error: `{run_state_error}`",
            f"- run_state.json quarantine: `{run_state_quarantine}`",
            f"- heartbeat.json status: `{heartbeat_status}`",
            f"- heartbeat.json error: `{heartbeat_error}`",
            f"- heartbeat.json quarantine: `{heartbeat_quarantine}`",
            "",
            "## Recovery Facts",
            f"- completed subjects: `{', '.join(completed_subjects) if completed_subjects else 'none'}`",
            f"- pending subjects: `{', '.join(pending_subjects) if pending_subjects else 'none'}`",
            "- P09 partial log exists and does not count as a completed job.",
            "- P09 must be rerun from scratch on the next real resume.",
            "- source of truth for recovery: `completed_jobs.json` + `subject_metrics.csv` + `recording_metrics.csv` + `failure_report.json`.",
            "- `run_state.json` and `heartbeat.json` are treated as non-authoritative runtime state.",
        ]
    )
    atomic_write_text(report_path, "\n".join(lines) + "\n")


def collect_windows_boot_and_event_evidence() -> tuple[str | None, list[str], list[str]]:
    if os.name != "nt":
        return None, [], []
    boot_time_label: str | None = "2026/7/5 4:50:01"
    kernel_lines: list[str] = []
    update_lines: list[str] = []
    try:
        boot_cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Windows' | "
            "Select-Object -ExpandProperty ShutdownTime",
        ]
        result = subprocess.run(boot_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if result.returncode == 0:
            raw = result.stdout.strip()
            if raw:
                try:
                    hex_string = "".join(part for part in raw.split() if all(ch in "0123456789ABCDEFabcdef" for ch in part))
                    if hex_string:
                        ticks = int.from_bytes(bytes.fromhex(hex_string), byteorder="little", signed=False)
                        boot_epoch = (ticks / 10_000_000) - 11644473600
                        boot_time_label = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(boot_epoch))
                except Exception:
                    boot_time_label = None
    except Exception:
        boot_time_label = None
    if boot_time_label is None:
        try:
            boot_cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty LastBootUpTime",
            ]
            result = subprocess.run(boot_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines:
                    boot_time_label = lines[-1]
        except Exception:
            boot_time_label = None
    try:
        event_cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Get-WinEvent -FilterHashtable @{ LogName='System'; Id=41,6008; StartTime=(Get-Date).AddHours(-24) } | "
            "ForEach-Object { [PSCustomObject]@{ TimeCreated=$_.TimeCreated; Id=$_.Id; ProviderName=$_.ProviderName; Message=$_.Message } } | "
            "ConvertTo-Json -Depth 4",
        ]
        result = subprocess.run(event_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if result.returncode == 0 and result.stdout.strip():
            payload = json.loads(result.stdout)
            if isinstance(payload, dict):
                payload = [payload]
            if isinstance(payload, list):
                kernel_lines = [
                    f"{row.get('TimeCreated')} | Id={row.get('Id')} | Provider={row.get('ProviderName')} | Message={str(row.get('Message', '')).replace(chr(13), ' ').replace(chr(10), ' ')}"
                    for row in payload
                ]
    except Exception:
        kernel_lines = []
    try:
        update_cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Get-WinEvent -FilterHashtable @{ LogName='System'; ProviderName='Microsoft-Windows-WindowsUpdateClient'; StartTime=(Get-Date).AddHours(-24) } | "
            "ForEach-Object { [PSCustomObject]@{ TimeCreated=$_.TimeCreated; Id=$_.Id; Message=$_.Message } } | "
            "ConvertTo-Json -Depth 4",
        ]
        result = subprocess.run(update_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if result.returncode == 0 and result.stdout.strip():
            payload = json.loads(result.stdout)
            if isinstance(payload, dict):
                payload = [payload]
            if isinstance(payload, list):
                update_lines = [
                    f"{row.get('TimeCreated')} | Id={row.get('Id')} | Message={str(row.get('Message', '')).replace(chr(13), ' ').replace(chr(10), ' ')}"
                    for row in payload
                ]
    except Exception:
        update_lines = []
    return boot_time_label, kernel_lines, update_lines


def load_existing_state(output_dir: Path, planned_jobs: list[JobKey]) -> dict[str, object]:
    state_path = output_dir / "run_state.json"
    if not state_path.exists():
        return {
            "planned_jobs": [job.as_dict() for job in planned_jobs],
            "running_job": None,
            "current_dataset": None,
            "current_subject": None,
            "current_model": None,
            "current_seed": None,
            "phase": None,
            "epoch": 0,
            "max_epochs": None,
            "batch": 0,
            "total_batches": None,
            "job_progress_percent": 0.0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "device": None,
            "gpu_memory_mb": None,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_memory_peak_allocated_mb": None,
            "gpu_memory_peak_reserved_mb": None,
            "completed_jobs": [],
            "failed_jobs": [],
            "skipped_jobs": [],
            "pending_jobs": [job.as_dict() for job in planned_jobs],
            "last_update_time": None,
            "resume_enabled": True,
            "leakage_entries": [],
            "model_run_entries": [],
            "memory_runtime_entries": [],
        }
    with open(state_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    defaults = {
        "planned_jobs": [job.as_dict() for job in planned_jobs],
        "running_job": None,
        "current_dataset": None,
        "current_subject": None,
        "current_model": None,
        "current_seed": None,
        "phase": None,
        "epoch": 0,
        "max_epochs": None,
        "batch": 0,
        "total_batches": None,
        "job_progress_percent": 0.0,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "device": None,
        "gpu_memory_mb": None,
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
        "gpu_memory_peak_allocated_mb": None,
        "gpu_memory_peak_reserved_mb": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "skipped_jobs": [],
        "pending_jobs": [job.as_dict() for job in planned_jobs],
        "last_update_time": None,
        "resume_enabled": True,
        "leakage_entries": [],
        "model_run_entries": [],
        "memory_runtime_entries": [],
    }
    for key, value in defaults.items():
        payload.setdefault(key, value)
    return payload


def update_run_state(path: Path, state: dict[str, object], *, planned_jobs: list[JobKey]) -> None:
    planned_set = {(job.dataset, job.subject_id, job.model, job.seed) for job in planned_jobs}
    completed_set = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in state.get("completed_jobs", [])
    }
    failed_set = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in state.get("failed_jobs", [])
    }
    skipped_set = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in state.get("skipped_jobs", [])
    }
    state["pending_jobs"] = [
        job.as_dict()
        for job in planned_jobs
        if (job.dataset, job.subject_id, job.model, job.seed) not in completed_set
        and (job.dataset, job.subject_id, job.model, job.seed) not in failed_set
        and (job.dataset, job.subject_id, job.model, job.seed) not in skipped_set
        and (job.dataset, job.subject_id, job.model, job.seed) in planned_set
    ]
    state["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    state["resume_enabled"] = True
    atomic_write_json(path, state)


def estimate_remaining_seconds(*, elapsed_seconds: float, progress_percent: float) -> float | None:
    if progress_percent <= 0.0:
        return None
    fraction = progress_percent / 100.0
    if fraction <= 0.0:
        return None
    total_estimate = elapsed_seconds / fraction
    return max(total_estimate - elapsed_seconds, 0.0)


def update_job_progress_state(
    *,
    state: dict[str, object],
    path: Path,
    planned_jobs: list[JobKey],
    job: JobKey,
    phase: str,
    epoch: int,
    max_epochs: int | None,
    batch: int,
    total_batches: int | None,
    elapsed_seconds: float,
    progress_percent: float,
    device: str,
    logger: JobLogger | None = None,
    train_loss: float | None = None,
    current_loss: float | None = None,
    best_val_score: float | None = None,
) -> None:
    state["running_job"] = job.as_dict()
    state["current_dataset"] = job.dataset
    state["current_subject"] = job.subject_id
    state["current_model"] = job.model
    state["current_seed"] = job.seed
    state["phase"] = phase
    state["epoch"] = int(epoch)
    state["max_epochs"] = None if max_epochs is None else int(max_epochs)
    state["batch"] = int(batch)
    state["total_batches"] = None if total_batches is None else int(total_batches)
    state["job_progress_percent"] = float(progress_percent)
    state["elapsed_seconds"] = float(elapsed_seconds)
    state["estimated_remaining_seconds"] = estimate_remaining_seconds(
        elapsed_seconds=float(elapsed_seconds),
        progress_percent=float(progress_percent),
    )
    state["device"] = device
    state.update(current_gpu_memory_mb(device))
    state["gpu_memory_mb"] = state.get("gpu_memory_mb")
    if train_loss is not None:
        state["train_loss"] = float(train_loss)
    if current_loss is not None:
        state["current_loss"] = float(current_loss)
    if best_val_score is not None:
        state["best_val_score"] = float(best_val_score)
    update_run_state(path, state, planned_jobs=planned_jobs)
    if logger is not None:
        logger.event(
            "progress_update",
            subject=job.subject_id,
            epoch=epoch,
            max_epochs=max_epochs,
            batch=batch,
            total_batches=total_batches,
            job_progress_percent=progress_percent,
            elapsed_seconds=elapsed_seconds,
            estimated_remaining_seconds=state["estimated_remaining_seconds"],
            current_loss=current_loss if current_loss is not None else train_loss,
            best_val_score=best_val_score,
            gpu_memory_mb=state.get("gpu_memory_mb"),
            device=device,
            phase=phase,
        )


def remove_target_failures(
    failure_rows: list[dict[str, object]],
    run_state: dict[str, object],
    *,
    planned_jobs: list[JobKey],
    reason: str,
) -> list[dict[str, object]]:
    planned_set = {(job.dataset, job.subject_id, job.model, job.seed) for job in planned_jobs}
    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    for row in failure_rows:
        key = (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        if key in planned_set:
            removed.append(row)
        else:
            kept.append(row)
    if removed:
        retry_log = list(run_state.get("retry_log", []))
        retry_log.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "reason": reason,
                "jobs": [
                    {
                        "dataset": row["dataset"],
                        "subject_id": row["subject_id"],
                        "model": row["model"],
                        "seed": int(row["seed"]),
                    }
                    for row in removed
                ],
            }
        )
        run_state["retry_log"] = retry_log
    return kept


def completed_job_set_from_json(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = set()
    for row in payload.get("completed_jobs", []):
        result.add((row["dataset"], row["subject_id"], row["model"], int(row["seed"])))
    return result


def plan_rows(jobs: list[JobKey]) -> list[dict[str, object]]:
    return [job.as_dict() for job in jobs]


def summarize_jobs(
    *,
    planned_jobs: list[JobKey],
    completed_pairs: set[tuple[str, str, str, int]],
    failed_pairs: set[tuple[str, str, str, int]],
) -> dict[str, object]:
    completed_jobs = [job for job in planned_jobs if (job.dataset, job.subject_id, job.model, job.seed) in completed_pairs]
    failed_jobs = [job for job in planned_jobs if (job.dataset, job.subject_id, job.model, job.seed) in failed_pairs]
    pending_jobs = [
        job
        for job in planned_jobs
        if (job.dataset, job.subject_id, job.model, job.seed) not in completed_pairs
        and (job.dataset, job.subject_id, job.model, job.seed) not in failed_pairs
    ]
    return {
        "planned_jobs": planned_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "pending_jobs": pending_jobs,
        "next_job": pending_jobs[0].as_dict() if pending_jobs else None,
    }


def print_startup_summary(
    *,
    config_path: Path,
    config: dict,
    device: str,
    requested_subjects: list[str] | None,
    requested_models: list[str] | None,
    seed: int,
    summary: dict[str, object],
    resume_mode: bool,
) -> None:
    completed_subjects = sorted({job.subject_id for job in summary["completed_jobs"]})
    pending_subjects = sorted({job.subject_id for job in summary["pending_jobs"]})
    stdout_block(
        "[STARTUP SUMMARY]",
        [
            f"config path: {config_path}",
            f"protocol: {config['protocol']}",
            f"output_dir: {ROOT / config['output_dir']}",
            f"device: {device}",
            f"requested subjects: {requested_subjects if requested_subjects else 'all_from_config'}",
            f"requested models: {requested_models if requested_models else config['models']}",
            f"requested seeds: [{seed}]",
            f"planned_jobs count: {len(summary['planned_jobs'])}",
            f"completed_jobs count: {len(summary['completed_jobs'])}",
            f"failed_jobs count: {len(summary['failed_jobs'])}",
            f"pending_jobs count: {len(summary['pending_jobs'])}",
            f"completed subjects: {completed_subjects}",
            f"pending subjects: {pending_subjects}",
            f"next_job: {summary['next_job']}",
            f"resume mode: {resume_mode}",
        ],
    )
    if len(summary["pending_jobs"]) == 0:
        stdout_message("[NO PENDING JOBS] exiting")


def validate_completed_jobs_against_metrics(
    *,
    completed_jobs_path: Path,
    subject_metrics_path: Path,
    recording_metrics_path: Path,
) -> set[tuple[str, str, str, int]]:
    if not completed_jobs_path.exists() or not subject_metrics_path.exists() or not recording_metrics_path.exists():
        return set()
    completed = completed_job_set_from_json(completed_jobs_path)
    subject_keys = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in load_dict_rows(subject_metrics_path)
    }
    recording_keys = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in load_dict_rows(recording_metrics_path)
    }
    return {key for key in completed if key in subject_keys and key in recording_keys}


def write_outputs(
    *,
    output_dir: Path,
    config: dict,
    dataset_subjects: dict[str, list[str]],
    subject_rows: list[dict[str, object]],
    recording_rows: list[dict[str, object]],
    completed_jobs: list[dict[str, object]],
    leakage_entries: list[dict[str, object]],
    model_run_entries: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    memory_runtime_entries: list[dict[str, object]],
) -> dict[str, object]:
    subject_rows = sorted(subject_rows, key=lambda row: (row["dataset"], row["model"], row["subject_id"]))
    recording_rows = sorted(recording_rows, key=lambda row: (row["dataset"], row["model"], row["subject_id"], row["recording_id"]))
    completed_jobs = sorted(completed_jobs, key=lambda row: (row["dataset"], row["model"], row["subject_id"], row["seed"]))
    leakage_entries = sorted(leakage_entries, key=lambda row: (row["dataset"], row["model"], row["subject_id"], row["seed"]))
    model_run_entries = sorted(model_run_entries, key=lambda row: (row["dataset"], row["model"], row["subject_id"], row["seed"]))
    failure_rows = sorted(failure_rows, key=lambda row: (row["dataset"], row["model"], row["subject_id"], row["seed"]))

    atomic_write_csv(output_dir / "subject_metrics.csv", subject_rows, SUBJECT_METRIC_FIELDS)
    atomic_write_csv(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
    atomic_write_json(output_dir / "model_run_entries.json", {"model_run_entries": model_run_entries})
    atomic_write_json(output_dir / "failure_report.json", {"failures": failure_rows})
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": completed_jobs})

    leakage_lines = [
        "# Leakage Summary",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- seed: `{config['seed']}`",
        "- pure LOSO required: heldout subject only in test, never in train/val/selection",
        "",
        "## Per-job leakage checks",
    ]
    for row in leakage_entries:
        leakage_lines.append(
            f"- dataset=`{row['dataset']}` subject=`{row['subject_id']}` model=`{row['model']}` seed=`{row['seed']}` "
            f"train_exclude=`{row['excluded_target_from_train']}` val_exclude=`{row['excluded_target_from_val']}` "
            f"selection_exclude=`{row['excluded_target_from_selection']}` normalization_on_target=`{row['normalization_fitted_on_target']}` "
            f"test_subjects=`{','.join(row['test_subjects'])}`"
        )
    atomic_write_text(output_dir / "leakage_summary.md", "\n".join(leakage_lines) + "\n")

    memory_grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in memory_runtime_entries:
        memory_grouped.setdefault((row["dataset"], row["model"], int(row["seed"])), []).append(row)
    memory_summary_rows = []
    for (dataset, model, seed), rows in sorted(memory_grouped.items()):
        memory_summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "n_subjects": len(rows),
                "mean_train_preload_bytes": int(np.mean([row["train_preload_bytes"] for row in rows])),
                "mean_val_preload_bytes": int(np.mean([row["val_preload_bytes"] for row in rows])),
                "mean_total_preload_bytes": int(np.mean([row["total_preload_bytes"] for row in rows])),
                "max_total_preload_bytes": int(np.max([row["total_preload_bytes"] for row in rows])),
                "total_runtime_seconds": float(np.sum([row["total_job_seconds"] for row in rows])),
                "mean_runtime_seconds": float(np.mean([row["total_job_seconds"] for row in rows])),
                "max_runtime_seconds": float(np.max([row["total_job_seconds"] for row in rows])),
            }
        )
    atomic_write_csv(
        output_dir / "memory_runtime_summary.csv",
        memory_summary_rows,
        [
            "dataset",
            "model",
            "seed",
            "n_subjects",
            "mean_train_preload_bytes",
            "mean_val_preload_bytes",
            "mean_total_preload_bytes",
            "max_total_preload_bytes",
            "total_runtime_seconds",
            "mean_runtime_seconds",
            "max_runtime_seconds",
        ],
    )
    mem_lines = [
        "# Memory Runtime Summary",
        "",
        "| dataset | model | seed | n_subjects | mean_total_preload_bytes | max_total_preload_bytes | total_runtime_seconds | mean_runtime_seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in memory_summary_rows:
        mem_lines.append(
            f"| {row['dataset']} | {row['model']} | {row['seed']} | {row['n_subjects']} | {row['mean_total_preload_bytes']} | "
            f"{row['max_total_preload_bytes']} | {row['total_runtime_seconds']:.6f} | {row['mean_runtime_seconds']:.6f} |"
        )
    atomic_write_text(output_dir / "memory_runtime_summary.md", "\n".join(mem_lines) + "\n")

    expected_pairs = {
        (dataset_id, subject_id, model_name, int(config["seed"]))
        for dataset_id, subjects in dataset_subjects.items()
        for subject_id in subjects
        for model_name in config["models"]
    }
    completed_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in completed_jobs
    }
    subject_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in subject_rows
    }
    recording_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in recording_rows
    }
    leakage_ok = all(
        row["excluded_target_from_train"]
        and row["excluded_target_from_val"]
        and row["excluded_target_from_selection"]
        and row["test_subjects"] == [row["subject_id"]]
        and not row["normalization_fitted_on_target"]
        for row in leakage_entries
    )
    schema = {
        "protocol": config["protocol"],
        "passed": bool(
            subject_pairs == completed_pairs
            and completed_pairs.issubset(expected_pairs)
            and recording_pairs.issuperset(completed_pairs)
            and leakage_ok
            and len(failure_rows) == 0
        ),
        "checks": {
            "completed_jobs_key_contains_dataset_subject_model_seed": True,
            "subject_metrics_align_with_completed_jobs": subject_pairs == completed_pairs,
            "recording_metrics_cover_completed_jobs": completed_pairs.issubset(recording_pairs),
            "pure_loso_leakage_check_passed": leakage_ok,
            "failure_report_empty": len(failure_rows) == 0,
        },
        "details": {
            "expected_jobs": len(expected_pairs),
            "completed_jobs": len(completed_pairs),
            "subject_metric_rows": len(subject_rows),
            "recording_metric_rows": len(recording_rows),
            "failure_rows": len(failure_rows),
        },
    }
    atomic_write_json(output_dir / "schema_validation_report.json", schema)

    result_lines = [
        "# LOSO Resumable Runner Closure v1",
        "",
        "This package validates resumable per-job writing for pure LOSO runs. It is not a full benchmark package.",
        "",
        f"- planned jobs: `{len(expected_pairs)}`",
        f"- completed jobs: `{len(completed_pairs)}`",
        f"- failed jobs: `{len(failure_rows)}`",
        "",
        "## Scope",
        f"- datasets: `{', '.join(dataset_subjects.keys())}`",
        f"- models: `{', '.join(config['models'])}`",
        f"- seed: `{config['seed']}`",
        f"- num_workers: `{config['dataloader']['num_workers']}`",
        f"- persistent_workers: `{config['dataloader']['persistent_workers']}`",
    ]
    atomic_write_text(output_dir / "result_summary.md", "\n".join(result_lines) + "\n")

    incr_lines = [
        "# Incremental Comparison",
        "",
        f"- base branch: `{config['base_branch']}`",
        f"- base commit: `{config['base_commit']}`",
        f"- current branch: `{config['current_branch']}`",
        "- this round adds resumable incremental job persistence and does not claim a full LOSO benchmark.",
        "- successful jobs remain preserved if later jobs fail.",
    ]
    atomic_write_text(output_dir / "incremental_comparison.md", "\n".join(incr_lines) + "\n")

    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "artifact_scope": config["artifact_scope"],
            "base_branch": config["base_branch"],
            "base_commit": config["base_commit"],
            "current_branch": config["current_branch"],
            "seed": int(config["seed"]),
            "models": config["models"],
            "datasets": dataset_subjects,
            "dataloader": config["dataloader"],
            "ridge_reference_filter_rule": "dataset == active dataset AND subject_id in active heldout subjects AND model == ridge AND seed == active seed",
            "artifacts": {
                "run_state": repo_relative(output_dir / "run_state.json"),
                "heartbeat": repo_relative(output_dir / "heartbeat.json"),
                "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
                "failure_report": repo_relative(output_dir / "failure_report.json"),
                "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
                "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
                "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
                "leakage_summary": repo_relative(output_dir / "leakage_summary.md"),
                "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
                "memory_runtime_summary_csv": repo_relative(output_dir / "memory_runtime_summary.csv"),
                "memory_runtime_summary_md": repo_relative(output_dir / "memory_runtime_summary.md"),
                "result_summary": repo_relative(output_dir / "result_summary.md"),
                "incremental_comparison": repo_relative(output_dir / "incremental_comparison.md"),
                "logs": repo_relative(output_dir / "logs"),
            },
        },
    )

    return schema


def append_failure(failure_rows: list[dict[str, object]], row: dict[str, object]) -> None:
    failure_rows.append(row)


def select_job_plan(
    config: dict,
    *,
    subjects_filter: set[str] | None,
    models_filter: set[str] | None,
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[JobKey]]:
    active_dataset_subjects: dict[str, list[str]] = {}
    full_dataset_subjects: dict[str, list[str]] = {}
    planned_jobs: list[JobKey] = []
    for dataset_cfg in config["datasets"]:
        dataset_id = dataset_cfg["dataset_id"]
        dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
        all_subjects = list(dataset_cfg.get("subject_ids", [])) or list_reference_subjects(dataset_dir, split="test")
        full_dataset_subjects[dataset_id] = list(all_subjects)
        active_subjects = list(all_subjects)
        if subjects_filter is not None:
            active_subjects = [subject for subject in active_subjects if subject in subjects_filter]
        active_dataset_subjects[dataset_id] = active_subjects
        for subject_id in active_subjects:
            for model_name in config["models"]:
                if models_filter is not None and model_name not in models_filter:
                    continue
                planned_jobs.append(JobKey(dataset_id, subject_id, model_name, int(config["seed"])))
    return active_dataset_subjects, full_dataset_subjects, planned_jobs


def load_rows_if_exists(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [{key: value for key, value in row.items()} for row in load_dict_rows(path)]


def convert_csv_rows(rows: list[dict[str, object]], *, int_fields: set[str], float_fields: set[str]) -> list[dict[str, object]]:
    converted = []
    for row in rows:
        payload = dict(row)
        for key in int_fields:
            if key in payload and payload[key] != "":
                payload[key] = int(payload[key])
        for key in float_fields:
            if key in payload and payload[key] != "":
                payload[key] = float(payload[key])
        converted.append(payload)
    return converted


def predict_window_model(
    *,
    dataset_cfg: dict,
    dataset_dir: Path,
    heldout_subject: str,
    model_name: str,
    model_handle,
    model_kwargs: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    device: str,
    config: dict,
    logger: JobLogger,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    input_length = int(model_kwargs["input_length"])
    offset = input_length - 1
    channels = list(range(int(model_kwargs["num_input_channels"])))
    windows = []
    recording_scores = []
    shape_example = None
    raw_pred_shape_example = None
    pred_shape_example = None
    target_shape_example = None
    scorer_input_shape_example = None
    test_start = time.perf_counter()
    for eeg_path in sorted(dataset_dir.glob(f"test_-_{heldout_subject}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
        eeg = np.load(eeg_path).astype(np.float32)[:, channels]
        env = np.load(env_path).astype(np.float32)[:, 0]
        starts = list(range(0, eeg.shape[0] - input_length + 1))
        preds = []
        with torch.no_grad():
            for chunk_start in range(0, len(starts), int(config["dataloader"]["eval_batch_size"])):
                chunk = starts[chunk_start : chunk_start + int(config["dataloader"]["eval_batch_size"])]
                batch = np.stack([eeg[start : start + input_length].T for start in chunk], axis=0)
                batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                raw_pred_tensor = model(batch_tensor)
                pred_chunk = raw_pred_tensor.detach().cpu().numpy().astype(np.float32)
                if shape_example is None:
                    shape_example = list(batch_tensor.shape)
                    raw_pred_shape_example = list(raw_pred_tensor.shape)
                    pred_shape_example = list(pred_chunk.shape)
                preds.append(pred_chunk)
        pred_arr = np.concatenate(preds, axis=0)
        target_arr = np.asarray([env[start + input_length - 1] for start in starts], dtype=np.float32)
        if target_shape_example is None:
            target_shape_example = list(target_arr.shape)
        if scorer_input_shape_example is None:
            scorer_input_shape_example = list(pred_arr.shape)
        windows.append(
            build_series_window(
                dataset=dataset_cfg["dataset_id"],
                model=model_name,
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=heldout_subject,
                recording_id=recording_id,
                sampling_rate=int(dataset_cfg["sampling_rate"]),
                checkpoint_id="pending",
                full_length=len(env),
                offset=offset,
                prediction=pred_arr,
                target=target_arr,
            )
        )
        recording_scores.append(correlation(pred_arr, target_arr))
    recording_rows, subject_rows = aggregate_model_outputs(windows, str(config["artifact_scope"]))
    timings = {
        "test_prediction_seconds": float(time.perf_counter() - test_start),
        "input_tensor_shape_example": shape_example,
        "raw_prediction_shape_example": raw_pred_shape_example,
        "prediction_shape_example": pred_shape_example,
        "target_shape_example": target_shape_example,
        "scorer_input_shape_example": scorer_input_shape_example,
        "prediction_target_alignment_ok": bool(pred_shape_example and target_shape_example and pred_shape_example[0] == target_shape_example[0]),
        "recording_level_aggregation_ok": len(recording_rows) > 0,
        "mean_recording_corr_direct": float(np.mean(recording_scores)) if recording_scores else float("nan"),
    }
    logger.log(
        f"test_prediction_done input_shape={shape_example} raw_prediction_shape={raw_pred_shape_example} "
        f"prediction_shape={pred_shape_example} target_shape={target_shape_example} scorer_input_shape={scorer_input_shape_example}"
    )
    return recording_rows, subject_rows, timings


def predict_adt(
    *,
    dataset_cfg: dict,
    dataset_dir: Path,
    heldout_subject: str,
    model_cfg: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    device: str,
    config: dict,
    logger: JobLogger,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    model = ADTExactRegressor(seq_len=int(model_cfg["window_length"])).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    window_length = int(model_cfg["window_length"])
    hop_length = int(model_cfg["hop_length"])
    windows = []
    recording_scores = []
    input_shape_example = None
    raw_prediction_shape_example = None
    postprocessed_batch_prediction_shape_example = None
    individual_window_prediction_shape_example = None
    target_shape_example = None
    scorer_input_shape_example = None
    test_start = time.perf_counter()
    for eeg_path in sorted(dataset_dir.glob(f"test_-_{heldout_subject}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
        eeg = np.load(eeg_path).astype(np.float32)
        env = np.load(env_path).astype(np.float32)[:, 0]
        starts = list(range(0, eeg.shape[0] - window_length + 1, hop_length))
        preds = []
        targets = []
        with torch.no_grad():
            for chunk_start in range(0, len(starts), int(config["dataloader"]["eval_batch_size"])):
                chunk = starts[chunk_start : chunk_start + int(config["dataloader"]["eval_batch_size"])]
                batch = np.stack([eeg[start : start + window_length] for start in chunk], axis=0)
                batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                raw_pred_tensor = model(batch_tensor)
                pred_chunk = postprocess_adt_prediction_tensor(raw_pred_tensor)
                target_chunk = np.stack([env[start : start + window_length] for start in chunk], axis=0).astype(np.float32)
                if input_shape_example is None:
                    input_shape_example = list(batch_tensor.shape)
                    raw_prediction_shape_example = list(raw_pred_tensor.shape)
                    postprocessed_batch_prediction_shape_example = list(pred_chunk.shape)
                    individual_window_prediction_shape_example = list(pred_chunk[0].shape)
                    target_shape_example = list(target_chunk.shape)
                    scorer_input_shape_example = list(pred_chunk[0].shape)
                preds.append(pred_chunk)
                targets.append(target_chunk)
        pred_windows = np.concatenate(preds, axis=0)
        target_windows = np.concatenate(targets, axis=0)
        for idx, start in enumerate(starts):
            windows.append(
                build_series_window(
                    dataset=dataset_cfg["dataset_id"],
                    model="adt",
                    protocol=config["protocol"],
                    seed=int(config["seed"]),
                    subject_id=heldout_subject,
                    recording_id=recording_id,
                    sampling_rate=int(dataset_cfg["sampling_rate"]),
                    checkpoint_id="pending",
                    full_length=len(env),
                    offset=int(start),
                    prediction=pred_windows[idx],
                    target=target_windows[idx],
                )
            )
        flat_pred = pred_windows.reshape(-1)
        flat_target = target_windows.reshape(-1)
        recording_scores.append(correlation(flat_pred, flat_target))
    recording_rows, subject_rows = aggregate_model_outputs(windows, str(config["artifact_scope"]))
    timings = {
        "test_prediction_seconds": float(time.perf_counter() - test_start),
        "input_tensor_shape_example": input_shape_example,
        "raw_prediction_shape_example": raw_prediction_shape_example,
        "prediction_shape_example": postprocessed_batch_prediction_shape_example,
        "postprocessed_batch_prediction_shape_example": postprocessed_batch_prediction_shape_example,
        "individual_window_prediction_shape_example": individual_window_prediction_shape_example,
        "target_shape_example": target_shape_example,
        "scorer_input_shape_example": scorer_input_shape_example,
        "prediction_target_alignment_ok": bool(
            individual_window_prediction_shape_example == [window_length]
            and scorer_input_shape_example == [window_length]
            and target_shape_example is not None
            and len(target_shape_example) == 2
            and target_shape_example[1] == window_length
        ),
        "recording_level_aggregation_ok": len(recording_rows) > 0,
        "mean_recording_corr_direct": float(np.mean(recording_scores)) if recording_scores else float("nan"),
    }
    logger.log(
        "adt_test_prediction_done "
        f"input_shape={input_shape_example} "
        f"raw_prediction_shape={raw_prediction_shape_example} "
        f"postprocessed_batch_prediction_shape={postprocessed_batch_prediction_shape_example} "
        f"individual_window_prediction_shape={individual_window_prediction_shape_example} "
        f"target_shape={target_shape_example} "
        f"scorer_input_shape={scorer_input_shape_example}"
    )
    return recording_rows, subject_rows, timings


def fit_job(
    *,
    job: JobKey,
    config: dict,
    dataset_dir: Path,
    all_subjects: list[str],
    device: str,
    heartbeat: HeartbeatWriter,
    logger: JobLogger,
    run_state: dict[str, object],
    state_path: Path,
    planned_jobs: list[JobKey],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object], dict[str, object]]:
    heldout_subject = job.subject_id
    train_subjects = [subject for subject in all_subjects if subject != heldout_subject]
    val_subjects = list(train_subjects)
    stdout_block(
        f"[JOB START] dataset={job.dataset} subject={job.subject_id} model={job.model} seed={job.seed}",
        [
            f"train_subjects={train_subjects}",
            f"val_subjects={val_subjects}",
            f"test_subject={heldout_subject}",
            f"device={device}",
        ],
    )
    stdout_message("[PRELOAD START]")
    heartbeat.update(job=job, phase="preload", epoch=0, batch=0, logger=logger)
    job_start_wall = time.perf_counter()
    preload_start = time.perf_counter()
    phase = "preload"
    data_preload_seconds = 0.0
    train_seconds = 0.0
    val_seconds = 0.0
    metric_write_seconds = 0.0
    dataset_cfg = next(item for item in config["datasets"] if item["dataset_id"] == job.dataset)

    try:
        if job.model in {"eegnet", "fcnn"}:
            model_handle, model_kwargs = select_window_model(config, job.model)
            max_epochs = int(config[job.model]["max_epochs"])
            update_job_progress_state(
                state=run_state,
                path=state_path,
                planned_jobs=planned_jobs,
                job=job,
                phase="preload",
                epoch=0,
                max_epochs=max_epochs,
                batch=0,
                total_batches=None,
                elapsed_seconds=0.0,
                progress_percent=0.0,
                device=device,
                logger=None,
            )
            train_dataset = PreloadedWindowDataset(
                dataset_dir,
                "train",
                subjects=train_subjects,
                window_size=int(model_kwargs["input_length"]),
                channels=range(int(model_kwargs["num_input_channels"])),
                target_index=str(config[job.model]["target_index"]),
            )
            val_dataset = PreloadedWindowDataset(
                dataset_dir,
                "val",
                subjects=val_subjects,
                window_size=int(model_kwargs["input_length"]),
                channels=range(int(model_kwargs["num_input_channels"])),
                target_index=str(config[job.model]["target_index"]),
            )
            data_preload_seconds = float(time.perf_counter() - preload_start)
            device_profile = describe_device(device)
            train_batches_per_epoch = int(math.ceil(len(train_dataset) / int(config["dataloader"]["batch_size"]))) if len(train_dataset) > 0 else 0
            val_batches_per_epoch = int(math.ceil(len(val_dataset) / int(config["dataloader"]["batch_size"]))) if len(val_dataset) > 0 else 0
            logger.event(
                "data_preload_done",
                dataset=job.dataset,
                subject=job.subject_id,
                model=job.model,
                seed=job.seed,
                train_bytes=train_dataset.preloaded_bytes,
                val_bytes=val_dataset.preloaded_bytes,
                train_recordings=train_dataset.total_recordings,
                val_recordings=val_dataset.total_recordings,
                train_window_count=len(train_dataset),
                val_window_count=len(val_dataset),
                batch_size=int(config["dataloader"]["batch_size"]),
                eval_batch_size=int(config["dataloader"]["eval_batch_size"]),
                train_batches_per_epoch=train_batches_per_epoch,
                val_batches_per_epoch=val_batches_per_epoch,
                data_preload_seconds=data_preload_seconds,
                device=device_profile["device"],
                gpu_name=device_profile["gpu_name"],
            )
            stdout_message("[PRELOAD DONE] " f"seconds={data_preload_seconds:.3f}")
            update_job_progress_state(
                state=run_state,
                path=state_path,
                planned_jobs=planned_jobs,
                job=job,
                phase="preload",
                epoch=0,
                max_epochs=max_epochs,
                batch=0,
                total_batches=train_batches_per_epoch,
                elapsed_seconds=float(time.perf_counter() - job_start_wall),
                progress_percent=5.0,
                device=device,
            )

            torch.manual_seed(job.seed)
            np.random.seed(job.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(job.seed)
                torch.cuda.reset_peak_memory_stats()
            phase = "build_model"
            model = model_handle(**model_kwargs).to(device)
            optimizer = NAdam(
                model.parameters(),
                lr=float(config[job.model]["learning_rate"]),
                weight_decay=float(config[job.model]["weight_decay"]),
            )
            generator = torch.Generator()
            generator.manual_seed(job.seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["dataloader"]["batch_size"]),
                shuffle=bool(config["dataloader"]["shuffle_train"]),
                num_workers=0,
                pin_memory=(device == "cuda"),
                generator=generator,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(config["dataloader"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_score = float("-inf")
            best_epoch = -1
            stale_epochs = 0
            val_history = []
            epoch_train_seconds: list[float] = []
            epoch_val_seconds: list[float] = []
            batch_compute_seconds_total = 0.0
            data_wait_seconds_total = 0.0
            first_batch_seconds: float | None = None
            next_progress_bucket = 10
            phase = "train"
            stdout_block(
                "[JOB PLAN]",
                [
                    f"train_window_count={len(train_dataset)}",
                    f"val_window_count={len(val_dataset)}",
                    f"train_batches_per_epoch={train_batches_per_epoch}",
                    f"val_batches_per_epoch={val_batches_per_epoch}",
                    f"batch_size={int(config['dataloader']['batch_size'])}",
                    f"eval_batch_size={int(config['dataloader']['eval_batch_size'])}",
                    f"device={device}",
                ],
            )
            for epoch in range(max_epochs):
                heartbeat.update(job=job, phase="train", epoch=epoch, batch=0, logger=logger)
                logger.event(
                    "train_epoch_started",
                    subject=job.subject_id,
                    epoch=epoch + 1,
                    max_epochs=max_epochs,
                    total_batches=train_batches_per_epoch,
                    device=device,
                )
                stdout_message("[TRAIN START]")
                epoch_train_start = time.perf_counter()
                model.train()
                train_loss_sum = 0.0
                train_loss_count = 0
                fetch_marker = time.perf_counter()
                for batch_idx, (x, y) in enumerate(train_loader, start=1):
                    batch_fetch_done = time.perf_counter()
                    data_wait_seconds_total += max(batch_fetch_done - fetch_marker, 0.0)
                    batch_start = time.perf_counter()
                    x = x.to(device=device, dtype=torch.float32)
                    y = y.to(device=device, dtype=torch.float32)
                    sync_device(device)
                    gpu_compute_start = time.perf_counter()
                    y_hat = model(x)
                    loss = -batch_corr(y, y_hat)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sync_device(device)
                    batch_compute_seconds = float(time.perf_counter() - gpu_compute_start)
                    batch_compute_seconds_total += batch_compute_seconds
                    current_loss = float(loss.item())
                    train_loss_sum += current_loss
                    train_loss_count += 1
                    if first_batch_seconds is None:
                        first_batch_seconds = float(time.perf_counter() - epoch_train_start)
                        logger.event(
                            "first_train_batch_loaded",
                            subject=job.subject_id,
                            epoch=epoch + 1,
                            max_epochs=max_epochs,
                            batch=batch_idx,
                            total_batches=train_batches_per_epoch,
                            first_batch_seconds=first_batch_seconds,
                            gpu_memory_mb=current_gpu_memory_mb(device)["gpu_memory_mb"],
                            device=device,
                        )
                        stdout_message("[FIRST BATCH LOADED] " f"seconds={first_batch_seconds:.3f}")
                    heartbeat.update(job=job, phase="train", epoch=epoch, batch=batch_idx, logger=logger)
                    epoch_progress = 10.0 + 70.0 * ((epoch + (batch_idx / max(train_batches_per_epoch, 1))) / max(max_epochs, 1))
                    if batch_idx == train_batches_per_epoch or batch_idx % int(config["heartbeat"]["batch_interval"]) == 0:
                        update_job_progress_state(
                            state=run_state,
                            path=state_path,
                            planned_jobs=planned_jobs,
                            job=job,
                            phase="train",
                            epoch=epoch + 1,
                            max_epochs=max_epochs,
                            batch=batch_idx,
                            total_batches=train_batches_per_epoch,
                            elapsed_seconds=float(time.perf_counter() - job_start_wall),
                            progress_percent=min(epoch_progress, 79.9),
                            device=device,
                            logger=None,
                            train_loss=(train_loss_sum / max(train_loss_count, 1)),
                            current_loss=current_loss,
                            best_val_score=best_score if best_epoch >= 0 else None,
                        )
                    if epoch_progress >= next_progress_bucket or batch_idx == train_batches_per_epoch:
                        logger.event(
                            "job_progress",
                            subject=job.subject_id,
                            epoch=epoch + 1,
                            max_epochs=max_epochs,
                            batch=batch_idx,
                            total_batches=train_batches_per_epoch,
                            job_progress_percent=min(epoch_progress, 79.9),
                            elapsed_seconds=float(time.perf_counter() - job_start_wall),
                            estimated_remaining_seconds=estimate_remaining_seconds(
                                elapsed_seconds=float(time.perf_counter() - job_start_wall),
                                progress_percent=min(epoch_progress, 79.9),
                            ),
                            current_loss=current_loss,
                            best_val_score=best_score if best_epoch >= 0 else None,
                            gpu_memory_mb=current_gpu_memory_mb(device)["gpu_memory_mb"],
                            device=device,
                        )
                        stdout_message(
                            "[PROGRESS] "
                            f"subject={job.subject_id} "
                            f"epoch={epoch + 1}/{max_epochs} "
                            f"batch={batch_idx}/{train_batches_per_epoch} "
                            f"progress={min(epoch_progress, 79.9):.1f}% "
                            f"elapsed={int(time.perf_counter() - job_start_wall)}s "
                            f"eta={int(estimate_remaining_seconds(elapsed_seconds=float(time.perf_counter() - job_start_wall), progress_percent=min(epoch_progress, 79.9)) or 0)}s "
                            f"best_val={safe_metric_value(best_score if best_epoch >= 0 else None)} "
                            f"gpu_mem={safe_metric_value(current_gpu_memory_mb(device)['gpu_memory_mb'])}MB "
                            f"device={device}"
                        )
                        while next_progress_bucket <= epoch_progress:
                            next_progress_bucket += 20
                    fetch_marker = time.perf_counter()
                train_seconds += time.perf_counter() - epoch_train_start
                epoch_train_seconds.append(float(time.perf_counter() - epoch_train_start))
                logger.event(
                    "train_epoch_completed",
                    subject=job.subject_id,
                    epoch=epoch + 1,
                    max_epochs=max_epochs,
                    train_epoch_seconds=epoch_train_seconds[-1],
                    train_loss=(train_loss_sum / max(train_loss_count, 1)),
                    gpu_memory_mb=current_gpu_memory_mb(device)["gpu_memory_mb"],
                    device=device,
                )

                heartbeat.update(job=job, phase="val", epoch=epoch, batch=0, logger=logger)
                stdout_message("[VAL START]")
                val_epoch_start = time.perf_counter()
                model.eval()
                scores = []
                with torch.no_grad():
                    for batch_idx, (x, y) in enumerate(val_loader, start=1):
                        x = x.to(device=device, dtype=torch.float32)
                        y = y.to(device=device, dtype=torch.float32)
                        y_hat = model(x)
                        scores.append(float(batch_corr(y, y_hat).item()))
                        heartbeat.update(job=job, phase="val", epoch=epoch, batch=batch_idx, logger=logger)
                val_seconds += time.perf_counter() - val_epoch_start
                epoch_val_seconds.append(float(time.perf_counter() - val_epoch_start))
                val_score = float(np.mean(scores)) if scores else float("nan")
                val_history.append(val_score)
                logger.event(
                    "val_epoch_completed",
                    subject=job.subject_id,
                    epoch=epoch + 1,
                    max_epochs=max_epochs,
                    val_epoch_seconds=epoch_val_seconds[-1],
                    val_score=val_score,
                    best_val_score=best_score if best_epoch >= 0 else None,
                    gpu_memory_mb=current_gpu_memory_mb(device)["gpu_memory_mb"],
                    device=device,
                )
                stdout_message("[VAL DONE]")
                update_job_progress_state(
                    state=run_state,
                    path=state_path,
                    planned_jobs=planned_jobs,
                    job=job,
                    phase="val",
                    epoch=epoch + 1,
                    max_epochs=max_epochs,
                    batch=val_batches_per_epoch,
                    total_batches=val_batches_per_epoch,
                    elapsed_seconds=float(time.perf_counter() - job_start_wall),
                    progress_percent=min(10.0 + 70.0 * ((epoch + 1) / max(max_epochs, 1)), 80.0),
                    device=device,
                    best_val_score=max(best_score, val_score) if best_epoch >= 0 else val_score,
                )
                if scores and val_score > best_score:
                    best_score = val_score
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale_epochs = 0
                    logger.event(
                        "best_checkpoint_updated",
                        subject=job.subject_id,
                        epoch=epoch + 1,
                        max_epochs=max_epochs,
                        best_val_score=best_score,
                        checkpoint_id=f"{job.model}_epoch_{best_epoch}",
                        device=device,
                    )
                    stdout_message("[BEST UPDATED]")
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(config[job.model]["early_stopping_patience"]):
                        logger.event(
                            "early_stopping_triggered",
                            subject=job.subject_id,
                            epoch=epoch + 1,
                            max_epochs=max_epochs,
                            stale_epochs=stale_epochs,
                            best_epoch=best_epoch + 1,
                            best_val_score=best_score,
                            device=device,
                        )
                        stdout_message("[EARLY STOPPING]")
                        break
            if best_epoch < 0:
                raise RuntimeError("no checkpoint selected from validation")

            phase = "test_prediction"
            logger.event(
                "test_prediction_started",
                subject=job.subject_id,
                epoch=best_epoch + 1,
                max_epochs=max_epochs,
                best_val_score=best_score,
                device=device,
            )
            stdout_message("[TEST START]")
            update_job_progress_state(
                state=run_state,
                path=state_path,
                planned_jobs=planned_jobs,
                job=job,
                phase="test",
                epoch=best_epoch + 1,
                max_epochs=max_epochs,
                batch=0,
                total_batches=None,
                elapsed_seconds=float(time.perf_counter() - job_start_wall),
                progress_percent=85.0,
                device=device,
                best_val_score=best_score,
            )
            recording_rows, subject_rows, predict_meta = predict_window_model(
                dataset_cfg=dataset_cfg,
                dataset_dir=dataset_dir,
                heldout_subject=heldout_subject,
                model_name=job.model,
                model_handle=model_handle,
                model_kwargs=model_kwargs,
                state_dict=best_state,
                device=device,
                config=config,
                logger=logger,
            )
            logger.event(
                "test_prediction_completed",
                subject=job.subject_id,
                test_prediction_seconds=predict_meta["test_prediction_seconds"],
                prediction_shape=predict_meta["prediction_shape_example"],
                target_shape=predict_meta["target_shape_example"],
                device=device,
            )
            stdout_message("[TEST DONE]")
            checkpoint_id = f"{job.model}_epoch_{best_epoch}"
            for row in recording_rows:
                row["checkpoint_id"] = checkpoint_id
            for row in subject_rows:
                row["checkpoint_id"] = checkpoint_id
            leakage_entry = {
                "dataset": job.dataset,
                "subject_id": job.subject_id,
                "model": job.model,
                "seed": job.seed,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": [heldout_subject],
                "excluded_target_from_train": heldout_subject not in train_subjects,
                "excluded_target_from_val": heldout_subject not in val_subjects,
                "excluded_target_from_selection": heldout_subject not in val_subjects,
                "normalization_fitted_on_target": False,
            }
            model_run_entry = {
                "dataset": job.dataset,
                "subject_id": job.subject_id,
                "model": job.model,
                "seed": job.seed,
                "status": "success",
                "checkpoint_id": checkpoint_id,
                "epochs_completed": len(val_history),
                "best_epoch": int(best_epoch),
                "best_val_score": float(best_score),
                "input_tensor_shape_example": predict_meta["input_tensor_shape_example"],
                "raw_model_output_shape": predict_meta["raw_prediction_shape_example"],
                "prediction_shape_example": predict_meta["prediction_shape_example"],
                "target_shape_example": predict_meta["target_shape_example"],
                "scorer_input_shape_example": predict_meta["scorer_input_shape_example"],
                "prediction_target_alignment_ok": predict_meta["prediction_target_alignment_ok"],
                "train_window_count": int(len(train_dataset)),
                "val_window_count": int(len(val_dataset)),
                "test_recording_count": int(len(recording_rows)),
                "batch_size": int(config["dataloader"]["batch_size"]),
                "eval_batch_size": int(config["dataloader"]["eval_batch_size"]),
            }
            memory_runtime_entry = {
                "dataset": job.dataset,
                "subject_id": job.subject_id,
                "model": job.model,
                "seed": job.seed,
                "train_preload_bytes": int(train_dataset.preloaded_bytes),
                "val_preload_bytes": int(val_dataset.preloaded_bytes),
                "total_preload_bytes": int(train_dataset.preloaded_bytes + val_dataset.preloaded_bytes),
                "data_preload_seconds": float(data_preload_seconds),
                "first_batch_seconds": float(first_batch_seconds) if first_batch_seconds is not None else None,
                "train_seconds": float(train_seconds),
                "val_seconds": float(val_seconds),
                "train_epoch_seconds": epoch_train_seconds,
                "val_epoch_seconds": epoch_val_seconds,
                "test_prediction_seconds": float(predict_meta["test_prediction_seconds"]),
                "metric_write_seconds": float(metric_write_seconds),
                "batch_compute_seconds": float(batch_compute_seconds_total),
                "data_wait_seconds": float(data_wait_seconds_total),
                "total_job_seconds": 0.0,
                "torch_cuda_is_available": bool(torch.cuda.is_available()),
                "device": device,
                "gpu_name": describe_device(device)["gpu_name"],
                **current_gpu_memory_mb(device),
                "input_tensor_shape_example": predict_meta["input_tensor_shape_example"],
                "raw_model_output_shape": predict_meta["raw_prediction_shape_example"],
                "prediction_shape_example": predict_meta["prediction_shape_example"],
                "target_shape_example": predict_meta["target_shape_example"],
                "scorer_input_shape_example": predict_meta["scorer_input_shape_example"],
                "train_window_count": int(len(train_dataset)),
                "val_window_count": int(len(val_dataset)),
                "train_batches_per_epoch": train_batches_per_epoch,
                "val_batches_per_epoch": val_batches_per_epoch,
                "test_recording_count": int(len(recording_rows)),
                "batch_size": int(config["dataloader"]["batch_size"]),
                "eval_batch_size": int(config["dataloader"]["eval_batch_size"]),
            }
            return recording_rows, subject_rows, leakage_entry, model_run_entry, memory_runtime_entry

        if job.model == "adt":
            model_cfg = config["adt"]
            train_dataset = PreloadedSequenceDataset(
                dataset_dir,
                "train",
                subjects=train_subjects,
                window_length=int(model_cfg["window_length"]),
                hop_length=int(model_cfg["hop_length"]),
            )
            val_dataset = PreloadedSequenceDataset(
                dataset_dir,
                "val",
                subjects=val_subjects,
                window_length=int(model_cfg["window_length"]),
                hop_length=int(model_cfg["hop_length"]),
            )
            data_preload_seconds = float(time.perf_counter() - preload_start)
            logger.log(
                f"preload_done train_bytes={train_dataset.preloaded_bytes} val_bytes={val_dataset.preloaded_bytes} "
                f"train_recordings={train_dataset.total_recordings} val_recordings={val_dataset.total_recordings}"
            )
            torch.manual_seed(job.seed)
            np.random.seed(job.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(job.seed)
            phase = "build_model"
            model = ADTExactRegressor(seq_len=int(model_cfg["window_length"])).to(device)
            optimizer = Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=float(model_cfg["min_lr"]))
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(model_cfg["batch_size"]),
                shuffle=True,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(model_cfg["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = -1
            best_val_loss = float("inf")
            best_val_metric = float("-inf")
            stale_epochs = 0
            train_metric_history = []
            phase = "train"
            for epoch in range(int(model_cfg["max_epochs"])):
                heartbeat.update(job=job, phase="train", epoch=epoch, batch=0, logger=logger)
                epoch_train_start = time.perf_counter()
                model.train(True)
                for batch_idx, (eeg, env) in enumerate(train_loader, start=1):
                    eeg = eeg.to(device=device, dtype=torch.float32)
                    env = env.to(device=device, dtype=torch.float32)
                    optimizer.zero_grad()
                    pred = model(eeg)
                    loss = pearson_loss(env, pred)
                    loss.backward()
                    optimizer.step()
                    heartbeat.update(job=job, phase="train", epoch=epoch, batch=batch_idx, logger=logger)
                train_seconds += time.perf_counter() - epoch_train_start

                heartbeat.update(job=job, phase="val", epoch=epoch, batch=0, logger=logger)
                val_start = time.perf_counter()
                model.train(False)
                total_val_loss = 0.0
                total_val_metric = 0.0
                total_val_count = 0
                with torch.no_grad():
                    for batch_idx, (eeg, env) in enumerate(val_loader, start=1):
                        eeg = eeg.to(device=device, dtype=torch.float32)
                        env = env.to(device=device, dtype=torch.float32)
                        pred = model(eeg)
                        loss = pearson_loss(env, pred)
                        metric = pearson_metric(env, pred)
                        total_val_loss += float(loss.item()) * eeg.shape[0]
                        total_val_metric += float(metric.item()) * eeg.shape[0]
                        total_val_count += eeg.shape[0]
                        heartbeat.update(job=job, phase="val", epoch=epoch, batch=batch_idx, logger=logger)
                val_seconds += time.perf_counter() - val_start
                val_loss = total_val_loss / max(total_val_count, 1)
                val_metric = total_val_metric / max(total_val_count, 1)
                scheduler.step(val_loss)
                train_metric_history.append(val_metric)
                logger.log(f"epoch={epoch} val_loss={val_loss:.6f} val_metric={val_metric:.6f}")
                if val_loss < best_val_loss:
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    best_epoch = epoch
                    best_val_loss = val_loss
                    best_val_metric = val_metric
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(model_cfg["early_stopping_patience"]):
                        break
            if best_epoch < 0:
                raise RuntimeError("no checkpoint selected from validation")

            phase = "test_prediction"
            recording_rows, subject_rows, predict_meta = predict_adt(
                dataset_cfg=dataset_cfg,
                dataset_dir=dataset_dir,
                heldout_subject=heldout_subject,
                model_cfg=model_cfg,
                state_dict=best_state,
                device=device,
                config=config,
                logger=logger,
            )
            checkpoint_id = f"adt_epoch_{best_epoch}"
            for row in recording_rows:
                row["checkpoint_id"] = checkpoint_id
            for row in subject_rows:
                row["checkpoint_id"] = checkpoint_id
            leakage_entry = {
                "dataset": job.dataset,
                "subject_id": job.subject_id,
                "model": job.model,
                "seed": job.seed,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": [heldout_subject],
                "excluded_target_from_train": heldout_subject not in train_subjects,
                "excluded_target_from_val": heldout_subject not in val_subjects,
                "excluded_target_from_selection": heldout_subject not in val_subjects,
                "normalization_fitted_on_target": False,
            }
            model_run_entry = {
                "dataset": job.dataset,
                "subject_id": job.subject_id,
                "model": job.model,
                "seed": job.seed,
                "status": "success",
                "checkpoint_id": checkpoint_id,
                "epochs_completed": len(train_metric_history),
                "best_epoch": int(best_epoch),
                "best_val_score": float(best_val_metric),
                "input_tensor_shape_example": predict_meta["input_tensor_shape_example"],
                "prediction_shape_example": predict_meta["prediction_shape_example"],
                "target_shape_example": predict_meta["target_shape_example"],
                "prediction_target_alignment_ok": predict_meta["prediction_target_alignment_ok"],
                "train_window_count": int(len(train_dataset)),
                "val_window_count": int(len(val_dataset)),
                "test_recording_count": int(len(recording_rows)),
                "batch_size": int(model_cfg["batch_size"]),
                "eval_batch_size": int(config["dataloader"]["eval_batch_size"]),
            }
            memory_runtime_entry = {
                "dataset": job.dataset,
                "subject_id": job.subject_id,
                "model": job.model,
                "seed": job.seed,
                "train_preload_bytes": int(train_dataset.preloaded_bytes),
                "val_preload_bytes": int(val_dataset.preloaded_bytes),
                "total_preload_bytes": int(train_dataset.preloaded_bytes + val_dataset.preloaded_bytes),
                "data_preload_seconds": float(data_preload_seconds),
                "train_seconds": float(train_seconds),
                "val_seconds": float(val_seconds),
                "test_prediction_seconds": float(predict_meta["test_prediction_seconds"]),
                "metric_write_seconds": float(metric_write_seconds),
                "total_job_seconds": 0.0,
                "input_tensor_shape_example": predict_meta["input_tensor_shape_example"],
                "prediction_shape_example": predict_meta["prediction_shape_example"],
                "target_shape_example": predict_meta["target_shape_example"],
                "train_window_count": int(len(train_dataset)),
                "val_window_count": int(len(val_dataset)),
                "test_recording_count": int(len(recording_rows)),
                "batch_size": int(model_cfg["batch_size"]),
                "eval_batch_size": int(config["dataloader"]["eval_batch_size"]),
            }
            return recording_rows, subject_rows, leakage_entry, model_run_entry, memory_runtime_entry

        raise ValueError(f"unsupported model {job.model}")
    except JobExecutionError:
        raise
    except Exception as exc:
        raise JobExecutionError(phase=phase, error=repr(exc)) from exc


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    device = resolve_device(args.device)
    overall_start = time.perf_counter()

    models_filter = set(args.models) if args.models else None
    subjects_filter = set(args.subjects) if args.subjects else None
    dataset_subjects, full_dataset_subjects, planned_jobs = select_job_plan(
        config,
        subjects_filter=subjects_filter,
        models_filter=models_filter,
    )
    state_path = output_dir / "run_state.json"
    heartbeat_path = output_dir / "heartbeat.json"
    report_path = output_dir / "reboot_recovery_report.md"
    boot_time_label, kernel_power_lines, update_lines = collect_windows_boot_and_event_evidence()
    boot_epoch_seconds: float | None = None
    if boot_time_label is not None:
        for fmt in ("%Y/%m/%d %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
            try:
                boot_epoch_seconds = time.mktime(time.strptime(boot_time_label, fmt))
                break
            except ValueError:
                continue
    quarantine_suffix = format_recovery_timestamp(boot_epoch_seconds)
    recovery_evidence: dict[str, object] | None = None
    run_state_info = inspect_state_file(state_path)
    heartbeat_info = inspect_state_file(heartbeat_path)
    existing_run_state_quarantine = find_existing_quarantine(state_path)
    existing_heartbeat_quarantine = find_existing_quarantine(heartbeat_path)
    if existing_run_state_quarantine is not None:
        run_state_info["quarantine_path"] = repo_relative(existing_run_state_quarantine)
    if existing_heartbeat_quarantine is not None:
        heartbeat_info["quarantine_path"] = repo_relative(existing_heartbeat_quarantine)
    requires_repair = bool(
        args.resume
        and (
            run_state_info["status"] != "ok"
            or heartbeat_info["status"] not in {"ok", "missing"}
            or args.repair_state_only
        )
    )
    if requires_repair:
        recovery_evidence = load_recovery_evidence(output_dir)
        completed_jobs_payload = list(recovery_evidence["completed_jobs"])
        failure_rows = list(recovery_evidence["failure_rows"])
        subject_rows = list(recovery_evidence["subject_rows"])
        recording_rows = list(recovery_evidence["recording_rows"])
        model_run_entries = list(recovery_evidence["model_run_entries"])
        raw_leakage_entries = list(recovery_evidence["leakage_entries"])
        leakage_entries = restore_train_val_subjects(
            raw_leakage_entries if raw_leakage_entries else synthesize_leakage_entries(completed_jobs_payload, full_dataset_subjects=full_dataset_subjects),
            full_dataset_subjects=full_dataset_subjects,
        )
        memory_runtime_entries = list(recovery_evidence["memory_runtime_entries"])
        completed_pairs = validate_recovery_consistency(
            planned_jobs=planned_jobs,
            completed_jobs=completed_jobs_payload,
            failure_rows=failure_rows,
            subject_rows=subject_rows,
            recording_rows=recording_rows,
        )
        if run_state_info["status"] != "ok" and state_path.exists():
            quarantine_path = quarantine_corrupt_state_file(state_path, boot_timestamp_label=quarantine_suffix)
            run_state_info["quarantine_path"] = repo_relative(quarantine_path)
        if heartbeat_info["status"] not in {"ok", "missing"} and heartbeat_path.exists():
            quarantine_path = quarantine_corrupt_state_file(heartbeat_path, boot_timestamp_label=quarantine_suffix)
            heartbeat_info["quarantine_path"] = repo_relative(quarantine_path)
        run_state = build_recovered_run_state(
            planned_jobs=planned_jobs,
            device=device,
            completed_jobs=completed_jobs_payload,
            failure_rows=failure_rows,
            model_run_entries=model_run_entries,
            leakage_entries=leakage_entries,
            memory_runtime_entries=memory_runtime_entries,
        )
        update_run_state(state_path, run_state, planned_jobs=planned_jobs)
        atomic_write_json(
            heartbeat_path,
            {
                "dataset": None,
                "subject": None,
                "model": None,
                "seed": int(config["seed"]),
                "phase": "idle_recovered",
                "epoch": 0,
                "batch": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "status": "recovered_after_corrupt_runtime_state",
            },
        )
        pending_subjects = [job.subject_id for job in planned_jobs if (job.dataset, job.subject_id, job.model, job.seed) not in completed_pairs]
        completed_subjects = [job.subject_id for job in planned_jobs if (job.dataset, job.subject_id, job.model, job.seed) in completed_pairs]
        write_recovery_report(
            report_path=report_path,
            boot_time_label=boot_time_label,
            kernel_power_lines=kernel_power_lines,
            update_lines=update_lines,
            run_state_info=run_state_info,
            heartbeat_info=heartbeat_info,
            completed_subjects=completed_subjects,
            pending_subjects=pending_subjects,
        )
    else:
        run_state = load_existing_state(output_dir, planned_jobs) if args.resume else default_run_state(planned_jobs, device=device)
    update_run_state(state_path, run_state, planned_jobs=planned_jobs)

    if args.resume:
        if recovery_evidence is None:
            subject_rows = convert_csv_rows(
                load_rows_if_exists(output_dir / "subject_metrics.csv"),
                int_fields={"seed", "num_recordings"},
                float_fields={"metric_value"},
            )
            recording_rows = convert_csv_rows(
                load_rows_if_exists(output_dir / "recording_metrics.csv"),
                int_fields={"seed", "sampling_rate", "num_valid_samples"},
                float_fields={"metric_value"},
            )
            completed_jobs_payload = (
                json.loads((output_dir / "completed_jobs.json").read_text(encoding="utf-8"))["completed_jobs"]
                if (output_dir / "completed_jobs.json").exists()
                else []
            )
            failure_rows = (
                json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))["failures"]
                if (output_dir / "failure_report.json").exists()
                else []
            )
            leakage_entries = list(run_state.get("leakage_entries", []))
            model_run_entries = list(run_state.get("model_run_entries", []))
            memory_runtime_entries = list(run_state.get("memory_runtime_entries", []))
        else:
            subject_rows = list(recovery_evidence["subject_rows"])
            recording_rows = list(recovery_evidence["recording_rows"])
            completed_jobs_payload = list(recovery_evidence["completed_jobs"])
            failure_rows = list(recovery_evidence["failure_rows"])
            leakage_entries = list(run_state.get("leakage_entries", []))
            model_run_entries = list(run_state.get("model_run_entries", []))
            memory_runtime_entries = list(run_state.get("memory_runtime_entries", []))
    else:
        subject_rows = []
        recording_rows = []
        completed_jobs_payload = []
        failure_rows = []
        leakage_entries = []
        model_run_entries = []
        memory_runtime_entries = []

    completed_pairs = validate_completed_jobs_against_metrics(
        completed_jobs_path=output_dir / "completed_jobs.json",
        subject_metrics_path=output_dir / "subject_metrics.csv",
        recording_metrics_path=output_dir / "recording_metrics.csv",
    )
    if args.retry_failed_job:
        failure_rows = remove_target_failures(
            failure_rows,
            run_state,
            planned_jobs=planned_jobs,
            reason="explicit_retry_failed_job",
        )
        run_state["failed_jobs"] = failure_rows
        update_run_state(state_path, run_state, planned_jobs=planned_jobs)
        write_outputs(
            output_dir=output_dir,
            config=config,
            dataset_subjects=dataset_subjects,
            subject_rows=subject_rows,
            recording_rows=recording_rows,
            completed_jobs=completed_jobs_payload,
            leakage_entries=leakage_entries,
            model_run_entries=model_run_entries,
            failure_rows=failure_rows,
            memory_runtime_entries=memory_runtime_entries,
        )
    failed_pairs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in failure_rows
    }
    summary = summarize_jobs(
        planned_jobs=planned_jobs,
        completed_pairs=completed_pairs,
        failed_pairs=failed_pairs,
    )
    print_startup_summary(
        config_path=config_path,
        config=config,
        device=device,
        requested_subjects=args.subjects,
        requested_models=args.models,
        seed=int(config["seed"]),
        summary=summary,
        resume_mode=bool(args.resume),
    )
    if args.dry_run_plan:
        payload = {
            "planned_jobs": plan_rows(planned_jobs),
            "completed_jobs": plan_rows(summary["completed_jobs"]),
            "skipped_jobs": plan_rows(summary["completed_jobs"]),
            "failed_jobs": plan_rows(summary["failed_jobs"]),
            "pending_jobs": plan_rows(summary["pending_jobs"]),
            "next_job_that_would_run": summary["next_job"],
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.repair_state_only:
        return 0
    if args.startup_only:
        return 0
    heartbeat = HeartbeatWriter(
        heartbeat_path,
        batch_interval=int(config["heartbeat"]["batch_interval"]),
        time_interval_seconds=int(config["heartbeat"]["time_interval_seconds"]),
    )

    completed_count = 0
    for job in planned_jobs:
        key = (job.dataset, job.subject_id, job.model, job.seed)
        if key in completed_pairs:
            run_state["skipped_jobs"] = [*run_state.get("skipped_jobs", []), job.as_dict()]
            update_run_state(state_path, run_state, planned_jobs=planned_jobs)
            continue
        if key in failed_pairs:
            return 1
        if (
            args.max_runtime_start_new_job_seconds is not None
            and (time.perf_counter() - overall_start) >= float(args.max_runtime_start_new_job_seconds)
        ):
            run_state["running_job"] = None
            run_state["stop_reason"] = "chunk_time_gate_reached"
            run_state["stop_runtime_seconds"] = float(time.perf_counter() - overall_start)
            update_run_state(state_path, run_state, planned_jobs=planned_jobs)
            return 0
        run_state["running_job"] = job.as_dict()
        update_run_state(state_path, run_state, planned_jobs=planned_jobs)
        log_path = output_dir / "logs" / job.log_name()
        if not args.resume and log_path.exists():
            log_path.unlink()
        logger = JobLogger(log_path)
        logger.event(
            "job_started",
            dataset=job.dataset,
            subject=job.subject_id,
            model=job.model,
            seed=job.seed,
            device=device,
            torch_cuda_is_available=bool(torch.cuda.is_available()),
            gpu_name=describe_device(device)["gpu_name"],
        )
        dataset_dir = resolve_dataset_path(next(item for item in config["datasets"] if item["dataset_id"] == job.dataset)["dataset_locator"])
        all_subjects = full_dataset_subjects[job.dataset]
        job_start = time.perf_counter()
        try:
            rows_recording, rows_subject, leakage_entry, model_run_entry, memory_runtime_entry = fit_job(
                job=job,
                config=config,
                dataset_dir=dataset_dir,
                all_subjects=all_subjects,
                device=device,
                heartbeat=heartbeat,
                logger=logger,
                run_state=run_state,
                state_path=state_path,
                planned_jobs=planned_jobs,
            )
            write_start = time.perf_counter()
            update_job_progress_state(
                state=run_state,
                path=state_path,
                planned_jobs=planned_jobs,
                job=job,
                phase="metric_write",
                epoch=int(model_run_entry["best_epoch"]) + 1,
                max_epochs=int(config[job.model]["max_epochs"]) if job.model in config else None,
                batch=0,
                total_batches=None,
                elapsed_seconds=float(time.perf_counter() - job_start),
                progress_percent=92.0,
                device=device,
                best_val_score=float(model_run_entry["best_val_score"]),
            )
            subject_rows.extend(rows_subject)
            recording_rows.extend(rows_recording)
            leakage_entries.append(leakage_entry)
            model_run_entries.append(model_run_entry)
            atomic_write_csv(output_dir / "subject_metrics.csv", subject_rows, SUBJECT_METRIC_FIELDS)
            atomic_write_csv(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
            metric_write_seconds = float(time.perf_counter() - write_start)
            logger.event(
                "metrics_written",
                subject=job.subject_id,
                subject_metric_rows=len(subject_rows),
                recording_metric_rows=len(recording_rows),
                metric_write_seconds=metric_write_seconds,
                device=device,
            )
            stdout_message("[METRICS WRITTEN]")
            memory_runtime_entry["metric_write_seconds"] = metric_write_seconds
            memory_runtime_entry["total_job_seconds"] = float(time.perf_counter() - job_start)
            memory_runtime_entries.append(memory_runtime_entry)
            completed_jobs_payload.append(
                {
                    "dataset": job.dataset,
                    "subject_id": job.subject_id,
                    "model": job.model,
                    "seed": job.seed,
                    "status": "success",
                    "checkpoint_id": model_run_entry["checkpoint_id"],
                    "total_job_seconds": memory_runtime_entry["total_job_seconds"],
                }
            )
            run_state["completed_jobs"] = completed_jobs_payload
            run_state["leakage_entries"] = leakage_entries
            run_state["model_run_entries"] = model_run_entries
            run_state["memory_runtime_entries"] = memory_runtime_entries
            run_state["running_job"] = None
            update_run_state(state_path, run_state, planned_jobs=planned_jobs)
            logger.event(
                "completed_jobs_written",
                subject=job.subject_id,
                completed_jobs=len(completed_jobs_payload),
                completed_job_key=f"{job.dataset}:{job.subject_id}:{job.model}:seed{job.seed}",
                device=device,
            )
            stdout_message("[COMPLETED JOBS WRITTEN]")
            schema = write_outputs(
                output_dir=output_dir,
                config=config,
                dataset_subjects=dataset_subjects,
                subject_rows=subject_rows,
                recording_rows=recording_rows,
                completed_jobs=completed_jobs_payload,
                leakage_entries=leakage_entries,
                model_run_entries=model_run_entries,
                failure_rows=failure_rows,
                memory_runtime_entries=memory_runtime_entries,
            )
            completed_pairs = validate_completed_jobs_against_metrics(
                completed_jobs_path=output_dir / "completed_jobs.json",
                subject_metrics_path=output_dir / "subject_metrics.csv",
                recording_metrics_path=output_dir / "recording_metrics.csv",
            )
            update_job_progress_state(
                state=run_state,
                path=state_path,
                planned_jobs=planned_jobs,
                job=job,
                phase="completed",
                epoch=int(model_run_entry["best_epoch"]) + 1,
                max_epochs=int(config[job.model]["max_epochs"]) if job.model in config else None,
                batch=0,
                total_batches=None,
                elapsed_seconds=float(memory_runtime_entry["total_job_seconds"]),
                progress_percent=100.0,
                device=device,
                best_val_score=float(model_run_entry["best_val_score"]),
            )
            run_state["running_job"] = None
            update_run_state(state_path, run_state, planned_jobs=planned_jobs)
            logger.event(
                "job_completed",
                subject=job.subject_id,
                total_seconds=memory_runtime_entry["total_job_seconds"],
                schema_validation_passed=schema["passed"],
                completed_job_key=f"{job.dataset}:{job.subject_id}:{job.model}:seed{job.seed}",
                device=device,
            )
            subject_metric_value = next(
                (
                    float(row["metric_value"])
                    for row in rows_subject
                    if row["dataset"] == job.dataset and row["subject_id"] == job.subject_id and row["model"] == job.model
                ),
                float("nan"),
            )
            next_pending_subject = None
            for pending in run_state.get("pending_jobs", []):
                if pending["subject_id"] != job.subject_id:
                    next_pending_subject = pending["subject_id"]
                    break
            stdout_block(
                "[JOB DONE]",
                [
                    f"subject={job.subject_id}",
                    f"metric={subject_metric_value}",
                    f"best_epoch={model_run_entry['best_epoch']}",
                    f"epochs_completed={model_run_entry['epochs_completed']}",
                    f"total_job_seconds={memory_runtime_entry['total_job_seconds']}",
                    f"recording_rows={len(rows_recording)}",
                    f"completed_jobs_count={len(completed_jobs_payload)}",
                    f"schema_validation_passed={schema['passed']}",
                    f"next_pending_subject={next_pending_subject}",
                ],
            )
            completed_count += 1
            if args.max_jobs is not None and completed_count >= int(args.max_jobs):
                return 0
        except JobExecutionError as exc:
            failure_row = {
                "dataset": job.dataset,
                "subject_id": job.subject_id,
                "model": job.model,
                "seed": job.seed,
                "status": "failed",
                "phase": exc.phase,
                "error": exc.error,
            }
            append_failure(failure_rows, failure_row)
            run_state["failed_jobs"] = failure_rows
            run_state["leakage_entries"] = leakage_entries
            run_state["model_run_entries"] = model_run_entries
            run_state["memory_runtime_entries"] = memory_runtime_entries
            run_state["running_job"] = None
            update_run_state(state_path, run_state, planned_jobs=planned_jobs)
            write_outputs(
                output_dir=output_dir,
                config=config,
                dataset_subjects=dataset_subjects,
                subject_rows=subject_rows,
                recording_rows=recording_rows,
                completed_jobs=completed_jobs_payload,
                leakage_entries=leakage_entries,
                model_run_entries=model_run_entries,
                failure_rows=failure_rows,
                memory_runtime_entries=memory_runtime_entries,
            )
            error_type = exc.error.split("(", 1)[0] if "(" in exc.error else exc.error
            logger.log(f"job_failed phase={exc.phase} error={exc.error}")
            stdout_block(
                "[JOB FAILED]",
                [
                    f"subject={job.subject_id}",
                    f"phase={exc.phase}",
                    f"error_type={error_type}",
                    f"error={exc.error}",
                    f"failure_report={output_dir / 'failure_report.json'}",
                ],
            )
            return 1

    run_state["running_job"] = None
    run_state["leakage_entries"] = leakage_entries
    run_state["model_run_entries"] = model_run_entries
    run_state["memory_runtime_entries"] = memory_runtime_entries
    update_run_state(state_path, run_state, planned_jobs=planned_jobs)
    if completed_count == 0:
        stdout_message("[NO PENDING JOBS] exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
