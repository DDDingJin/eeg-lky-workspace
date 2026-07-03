from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed-job", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
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


def load_existing_state(output_dir: Path, planned_jobs: list[JobKey]) -> dict[str, object]:
    state_path = output_dir / "run_state.json"
    if not state_path.exists():
        return {
            "planned_jobs": [job.as_dict() for job in planned_jobs],
            "running_job": None,
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
        return json.load(handle)


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
    pred_shape_example = None
    target_shape_example = None
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
                pred_chunk = model(batch_tensor).detach().cpu().numpy().astype(np.float32)
                if shape_example is None:
                    shape_example = list(batch_tensor.shape)
                    pred_shape_example = list(pred_chunk.shape)
                preds.append(pred_chunk)
        pred_arr = np.concatenate(preds, axis=0)
        target_arr = np.asarray([env[start + input_length - 1] for start in starts], dtype=np.float32)
        if target_shape_example is None:
            target_shape_example = list(target_arr.shape)
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
        "prediction_shape_example": pred_shape_example,
        "target_shape_example": target_shape_example,
        "prediction_target_alignment_ok": bool(pred_shape_example and target_shape_example and pred_shape_example[0] == target_shape_example[0]),
        "recording_level_aggregation_ok": len(recording_rows) > 0,
        "mean_recording_corr_direct": float(np.mean(recording_scores)) if recording_scores else float("nan"),
    }
    logger.log(
        f"test_prediction_done input_shape={shape_example} prediction_shape={pred_shape_example} target_shape={target_shape_example}"
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
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object], dict[str, object]]:
    heldout_subject = job.subject_id
    train_subjects = [subject for subject in all_subjects if subject != heldout_subject]
    val_subjects = list(train_subjects)
    heartbeat.update(job=job, phase="preload", epoch=0, batch=0, logger=logger)
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
            logger.log(
                f"preload_done train_bytes={train_dataset.preloaded_bytes} val_bytes={val_dataset.preloaded_bytes} "
                f"train_recordings={train_dataset.total_recordings} val_recordings={val_dataset.total_recordings}"
            )

            torch.manual_seed(job.seed)
            np.random.seed(job.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(job.seed)
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
            phase = "train"
            for epoch in range(int(config[job.model]["max_epochs"])):
                heartbeat.update(job=job, phase="train", epoch=epoch, batch=0, logger=logger)
                epoch_train_start = time.perf_counter()
                model.train()
                for batch_idx, (x, y) in enumerate(train_loader, start=1):
                    x = x.to(device=device, dtype=torch.float32)
                    y = y.to(device=device, dtype=torch.float32)
                    y_hat = model(x)
                    loss = -batch_corr(y, y_hat)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    heartbeat.update(job=job, phase="train", epoch=epoch, batch=batch_idx, logger=logger)
                train_seconds += time.perf_counter() - epoch_train_start

                heartbeat.update(job=job, phase="val", epoch=epoch, batch=0, logger=logger)
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
                val_score = float(np.mean(scores)) if scores else float("nan")
                val_history.append(val_score)
                logger.log(f"epoch={epoch} val_score={val_score:.6f}")
                if scores and val_score > best_score:
                    best_score = val_score
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(config[job.model]["early_stopping_patience"]):
                        break
            if best_epoch < 0:
                raise RuntimeError("no checkpoint selected from validation")

            phase = "test_prediction"
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
                "prediction_shape_example": predict_meta["prediction_shape_example"],
                "target_shape_example": predict_meta["target_shape_example"],
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
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    device = resolve_device(args.device)

    models_filter = set(args.models) if args.models else None
    subjects_filter = set(args.subjects) if args.subjects else None
    dataset_subjects, full_dataset_subjects, planned_jobs = select_job_plan(
        config,
        subjects_filter=subjects_filter,
        models_filter=models_filter,
    )
    state_path = output_dir / "run_state.json"
    heartbeat_path = output_dir / "heartbeat.json"
    run_state = load_existing_state(output_dir, planned_jobs) if args.resume else {
        "planned_jobs": [job.as_dict() for job in planned_jobs],
        "running_job": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "skipped_jobs": [],
        "last_update_time": None,
        "resume_enabled": True,
    }
    update_run_state(state_path, run_state, planned_jobs=planned_jobs)

    if args.resume:
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
        leakage_entries: list[dict[str, object]] = list(run_state.get("leakage_entries", []))
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
    if args.dry_run_plan:
        completed_jobs = [job for job in planned_jobs if (job.dataset, job.subject_id, job.model, job.seed) in completed_pairs]
        failed_jobs = [job for job in planned_jobs if (job.dataset, job.subject_id, job.model, job.seed) in failed_pairs]
        skipped_jobs = list(completed_jobs)
        pending_jobs = [
            job
            for job in planned_jobs
            if (job.dataset, job.subject_id, job.model, job.seed) not in completed_pairs
            and (job.dataset, job.subject_id, job.model, job.seed) not in failed_pairs
        ]
        payload = {
            "planned_jobs": plan_rows(planned_jobs),
            "completed_jobs": plan_rows(completed_jobs),
            "skipped_jobs": plan_rows(skipped_jobs),
            "failed_jobs": plan_rows(failed_jobs),
            "pending_jobs": plan_rows(pending_jobs),
            "next_job_that_would_run": pending_jobs[0].as_dict() if pending_jobs else None,
        }
        print(json.dumps(payload, indent=2))
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
        run_state["running_job"] = job.as_dict()
        update_run_state(state_path, run_state, planned_jobs=planned_jobs)
        log_path = output_dir / "logs" / job.log_name()
        if not args.resume and log_path.exists():
            log_path.unlink()
        logger = JobLogger(log_path)
        logger.log("job_started")
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
            )
            write_start = time.perf_counter()
            subject_rows.extend(rows_subject)
            recording_rows.extend(rows_recording)
            leakage_entries.append(leakage_entry)
            model_run_entries.append(model_run_entry)
            atomic_write_csv(output_dir / "subject_metrics.csv", subject_rows, SUBJECT_METRIC_FIELDS)
            atomic_write_csv(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
            metric_write_seconds = float(time.perf_counter() - write_start)
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
            logger.log(f"job_completed total_seconds={memory_runtime_entry['total_job_seconds']:.6f}")
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
            logger.log(f"job_failed phase={exc.phase} error={exc.error}")
            return 1

    run_state["running_job"] = None
    run_state["leakage_entries"] = leakage_entries
    run_state["model_run_entries"] = model_run_entries
    run_state["memory_runtime_entries"] = memory_runtime_entries
    update_run_state(state_path, run_state, planned_jobs=planned_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
