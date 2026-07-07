from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import torch
from torch.optim import NAdam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")
ZERO_SHOT_ROOT = SHARED_WORKSPACE / "_fix_loso_resumable_runner_closure_v1"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.adt_exact import ADTExactRegressor, pearson_loss as adt_pearson_loss, pearson_metric as adt_pearson_metric
from repro.mldecoders.models import EEGNetRegressor
from repro.simple_models import FCNNBaseline


ATOMIC_RETRY_ATTEMPTS = 5
ATOMIC_RETRY_SLEEP_SECONDS = 0.05
DATASET_METRIC_FIELDS = [
    "dataset",
    "model",
    "seed",
    "split_id",
    "stage",
    "n_subjects",
    "mean_pearson",
    "std_pearson",
    "min_pearson",
    "max_pearson",
]
COMPARISON_FIELDS = [
    "dataset",
    "model",
    "seed",
    "split_id",
    "subject_id",
    "source_checkpoint_local_id",
    "source_checkpoint_relative_path",
    "source_checkpoint_best_epoch",
    "source_checkpoint_best_val_score",
    "zero_shot_metric",
    "pooled10_fine_tuned_metric",
    "delta",
    "actual_calibration_seconds",
    "actual_train_seconds",
    "actual_val_seconds",
    "final_test_seconds",
    "calibration_train_recording_ids",
    "calibration_val_recording_ids",
    "test_recording_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--stage", choices=["zero_shot", "pooled_finetune", "all"], default="all")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--duration-audit-only", action="store_true")
    parser.add_argument("--shape-check-only", action="store_true")
    parser.add_argument("--checkpoint-path-check-only", action="store_true")
    parser.add_argument("--rebuild-preflight-artifacts-only", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_tmp_path(path: Path) -> Path:
    unique = f"{os.getpid()}_{threading.get_ident()}_{int(time.time() * 1000)}_{uuid.uuid4().hex}"
    return path.with_name(f"{path.name}.{unique}.tmp")


def atomic_replace_with_retry(tmp: Path, path: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, ATOMIC_RETRY_ATTEMPTS + 1):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == ATOMIC_RETRY_ATTEMPTS:
                raise
            time.sleep(ATOMIC_RETRY_SLEEP_SECONDS * attempt)
        except OSError as exc:
            last_error = exc
            if attempt == ATOMIC_RETRY_ATTEMPTS:
                raise
            time.sleep(ATOMIC_RETRY_SLEEP_SECONDS * attempt)
    if last_error is not None:
        raise last_error


def atomic_write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    atomic_replace_with_retry(tmp, path)


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    tmp.write_text(content, encoding="utf-8")
    atomic_replace_with_retry(tmp, path)


def atomic_write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    atomic_replace_with_retry(tmp, path)


def read_json(path: Path, default: object) -> object:
    if not path.exists() or path.stat().st_size == 0:
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def repo_relative(path: Path) -> str:
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_candidate = SHARED_WORKSPACE / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def current_git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        return None
    return None


def current_git_branch_name() -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        return None
    return None


def stdout_message(message: str) -> None:
    print(message, flush=True)


def stdout_block(title: str, lines: list[str]) -> None:
    print(title, flush=True)
    for line in lines:
        print(line, flush=True)


class JobLogger:
    def __init__(self, path: Path) -> None:
        ensure_dir(path.parent)
        self.path = path

    def event(self, name: str, **payload: object) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join([name] + [f"{key}={payload[key]}" for key in payload]) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)


class SubjectSplitWindowDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_size: int,
        channels: range,
        target_index: str,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.subjects = subjects
        self.window_size = int(window_size)
        self.target_index = target_index
        self.channel_idx = np.asarray(list(channels), dtype=int)
        self.recordings: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self.index_rows: list[tuple[int, int]] = []
        self.total_recordings = 0
        self.preloaded_bytes = 0
        for subject_id in subjects:
            for eeg_path in sorted(dataset_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
                recording_id = eeg_path.stem.replace("_-_eeg", "")
                env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
                if not env_path.exists():
                    continue
                eeg = np.asarray(np.load(eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
                env = np.asarray(np.load(env_path, mmap_mode="r")[:, 0], dtype=np.float32)
                rec_idx = len(self.recordings)
                self.recordings.append((subject_id, recording_id, eeg, env))
                self.total_recordings += 1
                self.preloaded_bytes += int(eeg.nbytes + env.nbytes)
                max_start = eeg.shape[0] - self.window_size
                if max_start >= 0:
                    for start in range(max_start + 1):
                        self.index_rows.append((rec_idx, start))
        if not self.recordings:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")

    def __len__(self) -> int:
        return len(self.index_rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self.index_rows[idx]
        _, _, eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)


class SelectedRecordingWindowDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        recording_ids: list[str],
        window_size: int,
        channels: range,
        target_index: str,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.recording_ids = recording_ids
        self.window_size = int(window_size)
        self.target_index = target_index
        self.channel_idx = np.asarray(list(channels), dtype=int)
        self.recordings: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self.index_rows: list[tuple[int, int]] = []
        self.total_recordings = 0
        self.preloaded_bytes = 0
        for recording_id in recording_ids:
            eeg_path = dataset_dir / f"{recording_id}_-_eeg.npy"
            env_path = dataset_dir / f"{recording_id}_-_envelope.npy"
            if not eeg_path.exists() or not env_path.exists():
                raise FileNotFoundError(f"missing recording pair for {recording_id}")
            eeg = np.asarray(np.load(eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
            env = np.asarray(np.load(env_path, mmap_mode="r")[:, 0], dtype=np.float32)
            rec_idx = len(self.recordings)
            self.recordings.append((parse_subject_id(recording_id), recording_id, eeg, env))
            self.total_recordings += 1
            self.preloaded_bytes += int(eeg.nbytes + env.nbytes)
            max_start = eeg.shape[0] - self.window_size
            if max_start >= 0:
                for start in range(max_start + 1):
                    self.index_rows.append((rec_idx, start))
        if not self.recordings:
            raise ValueError("no recordings selected")

    def __len__(self) -> int:
        return len(self.index_rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self.index_rows[idx]
        _, _, eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)


class SubjectSplitSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_length: int,
        hop_length: int,
        channels: range,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.subjects = subjects
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        self.channel_idx = np.asarray(list(channels), dtype=int)
        self.recordings: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self.index_rows: list[tuple[int, int]] = []
        self.total_recordings = 0
        self.preloaded_bytes = 0
        for subject_id in subjects:
            for eeg_path in sorted(dataset_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
                recording_id = eeg_path.stem.replace("_-_eeg", "")
                env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
                if not env_path.exists():
                    continue
                eeg = np.asarray(np.load(eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
                env = np.asarray(np.load(env_path, mmap_mode="r")[:, 0], dtype=np.float32)
                rec_idx = len(self.recordings)
                self.recordings.append((subject_id, recording_id, eeg, env))
                self.total_recordings += 1
                self.preloaded_bytes += int(eeg.nbytes + env.nbytes)
                max_start = eeg.shape[0] - self.window_length
                if max_start >= 0:
                    for start in range(0, max_start + 1, self.hop_length):
                        self.index_rows.append((rec_idx, start))
        if not self.recordings:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")

    def __len__(self) -> int:
        return len(self.index_rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec_idx, start = self.index_rows[idx]
        _, _, eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_length]
        y = env[start : start + self.window_length]
        return x.astype(np.float32), y.astype(np.float32)


class SelectedRecordingSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        recording_ids: list[str],
        window_length: int,
        hop_length: int,
        channels: range,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.recording_ids = recording_ids
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        self.channel_idx = np.asarray(list(channels), dtype=int)
        self.recordings: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self.index_rows: list[tuple[int, int]] = []
        self.total_recordings = 0
        self.preloaded_bytes = 0
        for recording_id in recording_ids:
            eeg_path = dataset_dir / f"{recording_id}_-_eeg.npy"
            env_path = dataset_dir / f"{recording_id}_-_envelope.npy"
            if not eeg_path.exists() or not env_path.exists():
                raise FileNotFoundError(f"missing recording pair for {recording_id}")
            eeg = np.asarray(np.load(eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
            env = np.asarray(np.load(env_path, mmap_mode="r")[:, 0], dtype=np.float32)
            rec_idx = len(self.recordings)
            self.recordings.append((parse_subject_id(recording_id), recording_id, eeg, env))
            self.total_recordings += 1
            self.preloaded_bytes += int(eeg.nbytes + env.nbytes)
            max_start = eeg.shape[0] - self.window_length
            if max_start >= 0:
                for start in range(0, max_start + 1, self.hop_length):
                    self.index_rows.append((rec_idx, start))
        if not self.recordings:
            raise ValueError("no recordings selected")

    def __len__(self) -> int:
        return len(self.index_rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec_idx, start = self.index_rows[idx]
        _, _, eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_length]
        y = env[start : start + self.window_length]
        return x.astype(np.float32), y.astype(np.float32)


def batch_corr(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true0 = y_true - torch.mean(y_true)
    y_pred0 = y_pred - torch.mean(y_pred)
    return torch.sum(y_true0 * y_pred0) / (torch.sqrt(torch.sum(y_true0**2)) * torch.sqrt(torch.sum(y_pred0**2)) + eps)


def parse_subject_id(recording_id: str) -> str:
    parts = recording_id.split("_-_")
    if len(parts) < 3:
        raise ValueError(f"unexpected recording id: {recording_id}")
    return parts[1]


def stage_protocol(config: dict, stage: str) -> str:
    return f"{config['protocol']}::{stage}"


def stage_artifact_scope(config: dict, stage: str) -> str:
    return f"{config['artifact_scope']}::{stage}"


def model_contract(config: dict, model_name: str) -> dict[str, object]:
    if model_name == "eegnet":
        return {
            "family": "window_scalar",
            "input_tensor_shape": [1, 64, int(config["eegnet"]["window_size"])],
            "raw_output_shape": [1],
            "postprocessed_prediction_shape": [1],
            "scorer_input_shape": [1],
            "target_index": str(config["eegnet"]["target_index"]),
            "target_contract": "scalar target per window",
        }
    if model_name == "fcnn":
        return {
            "family": "window_scalar",
            "input_tensor_shape": [1, 64, int(config["fcnn"]["window_size"])],
            "raw_output_shape": [1],
            "postprocessed_prediction_shape": [1],
            "scorer_input_shape": [1],
            "target_index": str(config["fcnn"]["target_index"]),
            "target_contract": "scalar target per window",
        }
    if model_name == "adt":
        return {
            "family": "sequence",
            "input_tensor_shape": [1, int(config["adt"]["window_length"]), 64],
            "raw_output_shape": [1, int(config["adt"]["window_length"]), 1],
            "postprocessed_prediction_shape": [1, int(config["adt"]["window_length"])],
            "scorer_input_shape": [int(config["adt"]["window_length"])],
            "target_index": "full_window_sequence",
            "target_contract": "sequence target per window",
        }
    raise ValueError(f"unsupported model {model_name}")


def instantiate_model(config: dict, model_name: str) -> tuple[object, dict[str, object]]:
    if model_name == "eegnet":
        cfg = config["eegnet"]
        return EEGNetRegressor, {
            "num_input_channels": 64,
            "input_length": int(cfg["window_size"]),
            "temporal_filters": int(cfg["temporal_filters"]),
            "depth_multiplier": int(cfg["depth_multiplier"]),
            "separable_filters": int(cfg["separable_filters"]),
            "dropout_rate": float(cfg["dropout_rate"]),
        }
    if model_name == "fcnn":
        cfg = config["fcnn"]
        return FCNNBaseline, {
            "num_hidden": int(cfg["hidden_layers"]),
            "dropout_rate": float(cfg["dropout_rate"]),
            "input_length": int(cfg["window_size"]),
            "num_input_channels": 64,
        }
    if model_name == "adt":
        cfg = config["adt"]
        return ADTExactRegressor, {
            "seq_len": int(cfg["window_length"]),
        }
    raise ValueError(f"unsupported model {model_name}")


def postprocess_model_output(model_name: str, raw_output: torch.Tensor) -> torch.Tensor:
    if model_name in {"eegnet", "fcnn"}:
        return raw_output.reshape(-1)
    if model_name == "adt":
        if raw_output.ndim == 3 and raw_output.shape[-1] == 1:
            return raw_output.squeeze(-1)
        return raw_output
    raise ValueError(f"unsupported model {model_name}")


def model_window_size(config: dict, model_name: str) -> int:
    if model_name == "adt":
        return int(config["adt"]["window_length"])
    return int(config[model_name]["window_size"])


def model_hop_length(config: dict, model_name: str) -> int | None:
    if model_name == "adt":
        return int(config["adt"]["hop_length"])
    return None


def duration_prefix(rows: list[dict[str, object]], budget_seconds: float) -> tuple[list[dict[str, object]], float]:
    selected: list[dict[str, object]] = []
    total = 0.0
    for row in rows:
        selected.append(row)
        total += float(row["seconds"])
        if total >= budget_seconds:
            break
    return selected, total


def discover_recordings(dataset_dir: Path, *, split: str, subject_id: str, sampling_rate: int, window_size: int, hop_length: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eeg_path in sorted(dataset_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        num_samples = int(np.load(eeg_path, mmap_mode="r").shape[0])
        if hop_length is None:
            window_count = max(0, num_samples - window_size + 1)
        else:
            window_count = max(((num_samples - window_size) // hop_length) + 1, 0) if num_samples >= window_size else 0
        rows.append(
            {
                "recording_id": recording_id,
                "num_samples": num_samples,
                "seconds": float(num_samples / sampling_rate),
                "window_count": int(window_count),
            }
        )
    return rows


def load_split_manifest(path: Path) -> dict[str, object]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        raise ValueError(f"invalid split manifest: {path}")
    required = {"split_id", "dataset_id", "train_subjects", "val_subjects", "test_subjects", "subject_ids_all", "split_seed", "split_rule", "created_at", "no_overlap"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"split manifest missing fields: {sorted(missing)}")
    return payload


def resolve_selected_datasets(config: dict, requested: list[str] | None) -> list[dict[str, object]]:
    datasets = list(config["datasets"])
    if not requested:
        return datasets
    requested_set = set(requested)
    selected = [item for item in datasets if str(item["dataset_id"]) in requested_set]
    missing = sorted(requested_set - {str(item["dataset_id"]) for item in selected})
    if missing:
        raise ValueError(f"unknown datasets requested: {missing}")
    return selected


def resolve_selected_models(config: dict, requested: list[str] | None) -> list[str]:
    models = list(config["models"])
    if not requested:
        return models
    missing = sorted(set(requested) - set(models))
    if missing:
        raise ValueError(f"unknown models requested: {missing}")
    return list(requested)


def build_duration_audit(config: dict, dataset_cfg: dict[str, object]) -> dict[str, object]:
    dataset_dir = resolve_dataset_path(str(dataset_cfg["dataset_locator"]))
    split_manifest = load_split_manifest(ROOT / str(dataset_cfg["split_manifest_path"]))
    sampling_rate = int(dataset_cfg["sampling_rate"])
    audit_window = int(config["pooled_fine_tune"]["calibration_window_size_for_audit"])
    subjects_report: list[dict[str, object]] = []
    pooled = {
        "train_seconds": 0.0,
        "val_seconds": 0.0,
        "calibration_total_seconds": 0.0,
        "final_test_seconds": 0.0,
        "train_recording_count": 0,
        "val_recording_count": 0,
        "final_test_recording_count": 0,
        "train_window_count": 0,
        "val_window_count": 0,
        "final_test_window_count": 0,
    }
    for subject_id in split_manifest["test_subjects"]:
        train_rows = discover_recordings(dataset_dir, split="train", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=audit_window)
        val_rows = discover_recordings(dataset_dir, split="val", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=audit_window)
        test_rows = discover_recordings(dataset_dir, split="test", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=audit_window)
        train_selected, train_seconds = duration_prefix(train_rows, float(config["pooled_fine_tune"]["train_budget_seconds"]))
        val_selected, val_seconds = duration_prefix(val_rows, float(config["pooled_fine_tune"]["val_budget_seconds"]))
        test_seconds = float(sum(row["seconds"] for row in test_rows))
        test_windows = int(sum(row["window_count"] for row in test_rows))
        subject_row = {
            "subject_id": str(subject_id),
            "train_total_seconds": float(sum(row["seconds"] for row in train_rows)),
            "val_total_seconds": float(sum(row["seconds"] for row in val_rows)),
            "test_total_seconds": test_seconds,
            "calibration": {
                "required_seconds": float(config["pooled_fine_tune"]["calibration_budget_seconds"]),
                "train_budget_seconds": float(config["pooled_fine_tune"]["train_budget_seconds"]),
                "val_budget_seconds": float(config["pooled_fine_tune"]["val_budget_seconds"]),
                "actual_train_seconds": train_seconds,
                "actual_val_seconds": val_seconds,
                "actual_total_seconds": train_seconds + val_seconds,
                "train_recording_count": len(train_selected),
                "val_recording_count": len(val_selected),
                "train_window_count": int(sum(row["window_count"] for row in train_selected)),
                "val_window_count": int(sum(row["window_count"] for row in val_selected)),
                "train_recording_ids": [str(row["recording_id"]) for row in train_selected],
                "val_recording_ids": [str(row["recording_id"]) for row in val_selected],
                "supports_10min": train_seconds >= float(config["pooled_fine_tune"]["train_budget_seconds"]) and val_seconds >= float(config["pooled_fine_tune"]["val_budget_seconds"]),
            },
            "final_test": {
                "recording_count": len(test_rows),
                "seconds": test_seconds,
                "window_count": test_windows,
                "recording_ids": [str(row["recording_id"]) for row in test_rows],
                "non_empty": len(test_rows) > 0 and test_windows > 0,
            },
        }
        subjects_report.append(subject_row)
        pooled["train_seconds"] += train_seconds
        pooled["val_seconds"] += val_seconds
        pooled["calibration_total_seconds"] += train_seconds + val_seconds
        pooled["final_test_seconds"] += test_seconds
        pooled["train_recording_count"] += len(train_selected)
        pooled["val_recording_count"] += len(val_selected)
        pooled["final_test_recording_count"] += len(test_rows)
        pooled["train_window_count"] += int(sum(row["window_count"] for row in train_selected))
        pooled["val_window_count"] += int(sum(row["window_count"] for row in val_selected))
        pooled["final_test_window_count"] += test_windows
    return {
        "dataset_id": str(dataset_cfg["dataset_id"]),
        "split_id": str(split_manifest["split_id"]),
        "train_subjects": list(split_manifest["train_subjects"]),
        "val_subjects": list(split_manifest["val_subjects"]),
        "test_subjects": list(split_manifest["test_subjects"]),
        "subjects": subjects_report,
        "pooled": pooled,
        "all_test_subjects_support_10min": all(
            bool(subject["calibration"]["supports_10min"]) and bool(subject["final_test"]["non_empty"]) for subject in subjects_report
        ),
        "final_test_uses_original_test_split": True,
    }


def shape_check_for_model(config: dict, model_name: str, device: str) -> dict[str, object]:
    model_handle, model_kwargs = instantiate_model(config, model_name)
    contract = model_contract(config, model_name)
    model = model_handle(**model_kwargs).to(device)
    model.eval()
    x = torch.zeros(*contract["input_tensor_shape"], device=device, dtype=torch.float32)
    with torch.no_grad():
        raw = model(x)
        post = postprocess_model_output(model_name, raw)
    return {
        "model": model_name,
        "input_shape": list(contract["input_tensor_shape"]),
        "raw_output_shape": list(raw.shape),
        "postprocessed_prediction_shape": list(post.shape),
        "scorer_input_shape": contract["scorer_input_shape"],
        "target_index": contract["target_index"],
        "target_contract": contract["target_contract"],
        "passed": True,
    }


def zero_shot_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "subject_holdout"


def pooled10_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "finetune" / "subject_holdout_pooled10"


def zero_shot_checkpoint_local_id(*, dataset: str, model: str, seed: int, split_id: str, best_epoch: int) -> str:
    return f"subject_holdout:{model}:{dataset}:{split_id}:seed{seed}:best_epoch_{best_epoch}"


def pooled10_checkpoint_local_id(*, dataset: str, model: str, seed: int, split_id: str, best_epoch: int) -> str:
    return f"subject_holdout_pooled10_finetune:{model}:{dataset}:{split_id}:seed{seed}:best_epoch_{best_epoch}"


def zero_shot_checkpoint_relative_path(*, dataset: str, model: str, seed: int, split_id: str, best_epoch: int) -> Path:
    return Path(model) / dataset / split_id / f"seed{seed}" / f"best_epoch_{best_epoch}.pt"


def pooled10_checkpoint_relative_path(*, dataset: str, model: str, seed: int, split_id: str, best_epoch: int) -> Path:
    return Path(model) / dataset / split_id / f"seed{seed}" / f"best_epoch_{best_epoch}.pt"


def shared_zero_shot_checkpoint_path(*, dataset: str, model: str, seed: int, split_id: str) -> Path | None:
    if dataset == "weissbart_tf64" and model == "eegnet" and seed == 0 and split_id == "subject_holdout_fixed_split_v1":
        return ZERO_SHOT_ROOT / "local_checkpoints" / "subject_holdout" / "eegnet" / "weissbart_tf64" / split_id / "seed0" / "best_epoch_5.pt"
    return None


def checkpoint_epoch_from_name(path: Path) -> int:
    match = re.search(r"best_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint_in_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = sorted(path.glob("best_epoch_*.pt"), key=checkpoint_epoch_from_name)
    return candidates[-1] if candidates else None


def zero_shot_checkpoint_path_pattern(*, dataset: str, model: str, seed: int, split_id: str) -> Path:
    return zero_shot_checkpoint_root() / Path(model) / dataset / split_id / f"seed{seed}" / "best_epoch_PENDING.pt"


def pooled10_checkpoint_path_pattern(*, dataset: str, model: str, seed: int, split_id: str) -> Path:
    return pooled10_checkpoint_root() / Path(model) / dataset / split_id / f"seed{seed}" / "best_epoch_PENDING.pt"


def zero_shot_checkpoint_dir(*, dataset: str, model: str, seed: int, split_id: str) -> Path:
    return zero_shot_checkpoint_root() / Path(model) / dataset / split_id / f"seed{seed}"


def pooled10_checkpoint_dir(*, dataset: str, model: str, seed: int, split_id: str) -> Path:
    return pooled10_checkpoint_root() / Path(model) / dataset / split_id / f"seed{seed}"


def discover_reusable_zero_shot_checkpoint(*, dataset: str, model: str, seed: int, split_id: str) -> Path | None:
    shared = shared_zero_shot_checkpoint_path(dataset=dataset, model=model, seed=seed, split_id=split_id)
    if shared is not None and shared.exists():
        return shared
    local = latest_checkpoint_in_dir(zero_shot_checkpoint_dir(dataset=dataset, model=model, seed=seed, split_id=split_id))
    return local


def load_checkpoint_metadata(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    return payload


def checkpoint_path_check(config: dict, selected_datasets: list[dict[str, object]], selected_models: list[str], device: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_cfg in selected_datasets:
        dataset_id = str(dataset_cfg["dataset_id"])
        split_id = str(load_split_manifest(ROOT / str(dataset_cfg["split_manifest_path"]))["split_id"])
        for model_name in selected_models:
            reuse_path = discover_reusable_zero_shot_checkpoint(dataset=dataset_id, model=model_name, seed=int(config["seed"]), split_id=split_id)
            load_ok = False
            strict_load_ok = False
            if reuse_path is not None and reuse_path.exists():
                model_handle, model_kwargs = instantiate_model(config, model_name)
                payload = torch.load(reuse_path, map_location="cpu")
                model = model_handle(**model_kwargs)
                model.load_state_dict(payload["model_state_dict"], strict=True)
                load_ok = True
                strict_load_ok = True
            rows.append(
                {
                    "dataset": dataset_id,
                    "model": model_name,
                    "seed": int(config["seed"]),
                    "split_id": split_id,
                    "shared_reuse_checkpoint_path": repo_relative(reuse_path) if reuse_path is not None else None,
                    "shared_reuse_checkpoint_exists": bool(reuse_path is not None and reuse_path.exists()),
                    "zero_shot_checkpoint_path_pattern": repo_relative(zero_shot_checkpoint_path_pattern(dataset=dataset_id, model=model_name, seed=int(config["seed"]), split_id=split_id)),
                    "pooled10_checkpoint_path_pattern": repo_relative(pooled10_checkpoint_path_pattern(dataset=dataset_id, model=model_name, seed=int(config["seed"]), split_id=split_id)),
                    "torch_load_smoke": load_ok,
                    "strict_load_state_dict_smoke": strict_load_ok,
                }
            )
    return rows


def build_job_key(dataset: str, model: str, seed: int, split_id: str, stage: str) -> str:
    return f"{dataset}:{model}:seed{seed}:{split_id}:{stage}"


def build_job_plan(config: dict, selected_datasets: list[dict[str, object]], selected_models: list[str], stage: str) -> dict[str, object]:
    jobs: list[dict[str, object]] = []
    for dataset_cfg in selected_datasets:
        dataset_id = str(dataset_cfg["dataset_id"])
        split_id = str(load_split_manifest(ROOT / str(dataset_cfg["split_manifest_path"]))["split_id"])
        for model_name in selected_models:
            if stage in {"zero_shot", "all"}:
                reuse_path = discover_reusable_zero_shot_checkpoint(dataset=dataset_id, model=model_name, seed=int(config["seed"]), split_id=split_id)
                jobs.append(
                    {
                        "dataset": dataset_id,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "split_id": split_id,
                        "stage": "zero_shot",
                        "job_key": build_job_key(dataset_id, model_name, int(config["seed"]), split_id, "zero_shot"),
                        "reuse_allowed": bool(reuse_path is not None and reuse_path.exists()),
                        "reusable_checkpoint_path": repo_relative(reuse_path) if reuse_path is not None else None,
                    }
                )
            if stage in {"pooled_finetune", "all"}:
                jobs.append(
                    {
                        "dataset": dataset_id,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "split_id": split_id,
                        "stage": "pooled10_finetune",
                        "job_key": build_job_key(dataset_id, model_name, int(config["seed"]), split_id, "pooled10_finetune"),
                        "depends_on_zero_shot": True,
                    }
                )
    return {
        "planned_jobs": jobs,
        "zero_shot_job_count": sum(1 for job in jobs if job["stage"] == "zero_shot"),
        "pooled10_finetune_job_count": sum(1 for job in jobs if job["stage"] == "pooled10_finetune"),
    }


def mean_dataset_metric(subject_rows: list[dict[str, object]], *, stage: str, split_id: str) -> dict[str, object]:
    values = [float(row["metric_value"]) for row in subject_rows]
    return {
        "dataset": subject_rows[0]["dataset"] if subject_rows else "",
        "model": subject_rows[0]["model"] if subject_rows else "",
        "seed": int(subject_rows[0]["seed"]) if subject_rows else 0,
        "split_id": split_id,
        "stage": stage,
        "n_subjects": len(subject_rows),
        "mean_pearson": float(np.mean(values)) if values else float("nan"),
        "std_pearson": float(np.std(values, ddof=0)) if values else float("nan"),
        "min_pearson": float(np.min(values)) if values else float("nan"),
        "max_pearson": float(np.max(values)) if values else float("nan"),
    }


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


def collect_test_recording_ids(dataset_dir: Path, test_subjects: list[str]) -> list[str]:
    rows: list[str] = []
    for subject_id in test_subjects:
        for eeg_path in sorted(dataset_dir.glob(f"test_-_{subject_id}_-_*_-_eeg.npy")):
            rows.append(eeg_path.stem.replace("_-_eeg", ""))
    return rows


def predict_recordings_window(
    *,
    config: dict,
    dataset_cfg: dict[str, object],
    model_name: str,
    model: torch.nn.Module,
    checkpoint_id: str,
    recording_ids: list[str],
    device: str,
    stage: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[WindowPrediction] = []
    dataset_dir = resolve_dataset_path(str(dataset_cfg["dataset_locator"]))
    dataset_id = str(dataset_cfg["dataset_id"])
    sampling_rate = int(dataset_cfg["sampling_rate"])
    window_size = int(config[model_name]["window_size"])
    eval_batch_size = int(config["dataloader"]["eval_batch_size"])
    model.eval()
    for recording_id in recording_ids:
        eeg_path = dataset_dir / f"{recording_id}_-_eeg.npy"
        env_path = dataset_dir / f"{recording_id}_-_envelope.npy"
        eeg = np.asarray(np.load(eeg_path)[:, :64], dtype=np.float32)
        env = np.asarray(np.load(env_path)[:, 0], dtype=np.float32)
        total_windows = eeg.shape[0] - window_size + 1
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, total_windows, eval_batch_size):
                stop = min(total_windows, start + eval_batch_size)
                batch = np.stack([eeg[offset : offset + window_size].T for offset in range(start, stop)], axis=0)
                batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                preds.append(model(batch_tensor).detach().cpu().numpy().astype(np.float32).reshape(-1))
        pred_arr = np.concatenate(preds, axis=0) if preds else np.asarray([], dtype=np.float32)
        target_arr = np.asarray([env[offset + window_size - 1] for offset in range(total_windows)], dtype=np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model=model_name,
                protocol=stage_protocol(config, stage),
                seed=int(config["seed"]),
                subject_id=parse_subject_id(recording_id),
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=checkpoint_id,
                full_length=len(env),
                offset=window_size - 1,
                prediction=pred_arr,
                target=target_arr,
            )
        )
    return aggregate_model_outputs(windows, stage_artifact_scope(config, stage))


def predict_recordings_sequence(
    *,
    config: dict,
    dataset_cfg: dict[str, object],
    model_name: str,
    model: torch.nn.Module,
    checkpoint_id: str,
    recording_ids: list[str],
    device: str,
    stage: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[WindowPrediction] = []
    dataset_dir = resolve_dataset_path(str(dataset_cfg["dataset_locator"]))
    dataset_id = str(dataset_cfg["dataset_id"])
    sampling_rate = int(dataset_cfg["sampling_rate"])
    window_length = int(config["adt"]["window_length"])
    hop_length = int(config["adt"]["hop_length"])
    model.eval()
    with torch.no_grad():
        for recording_id in recording_ids:
            eeg_path = dataset_dir / f"{recording_id}_-_eeg.npy"
            env_path = dataset_dir / f"{recording_id}_-_envelope.npy"
            eeg = np.asarray(np.load(eeg_path)[:, :64], dtype=np.float32)
            env = np.asarray(np.load(env_path)[:, 0], dtype=np.float32)
            max_start = eeg.shape[0] - window_length
            for start in range(0, max_start + 1, hop_length):
                batch = torch.from_numpy(eeg[start : start + window_length]).unsqueeze(0).to(device=device, dtype=torch.float32)
                pred = model(batch).squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
                target = env[start : start + window_length].astype(np.float32)
                windows.append(
                    WindowPrediction(
                        dataset=dataset_id,
                        model=model_name,
                        task="reconstruction",
                        protocol=stage_protocol(config, stage),
                        seed=int(config["seed"]),
                        subject_id=parse_subject_id(recording_id),
                        recording_id=recording_id,
                        sampling_rate=sampling_rate,
                        checkpoint_id=checkpoint_id,
                        recording_length=int(len(env)),
                        start_index=int(start),
                        prediction=pred,
                        target=target,
                    )
                )
    return aggregate_model_outputs(windows, stage_artifact_scope(config, stage))


def evaluate_window_val(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            scores.append(float(batch_corr(y, y_hat).item()))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_sequence_val(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x).squeeze(-1)
            scores.append(float(adt_pearson_metric(y, y_hat).item()))
    return float(np.mean(scores)) if scores else float("nan")


def train_window_model(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    logger: JobLogger,
) -> tuple[dict[str, torch.Tensor], int, float, int]:
    optimizer = NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    epochs_completed = 0
    stdout_message("[TRAIN START]")
    for epoch in range(max_epochs):
        logger.event("train_epoch_started", epoch=epoch + 1, max_epochs=max_epochs)
        stdout_message(f"[TRAIN EPOCH] epoch={epoch + 1}/{max_epochs}")
        model.train()
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            loss = -batch_corr(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_score = evaluate_window_val(model, val_loader, device)
        epochs_completed = epoch + 1
        logger.event("val_epoch_completed", epoch=epoch + 1, val_score=val_score)
        stdout_message(f"[VAL DONE] epoch={epoch + 1} val_score={val_score} best_val={best_score if best_epoch >= 0 else 'None'}")
        if math.isfinite(val_score) and val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
            logger.event("best_checkpoint_updated", epoch=epoch + 1, best_val_score=best_score)
            stdout_message(f"[BEST UPDATED] epoch={epoch + 1} best_val={best_score}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.event("early_stopping_triggered", epoch=epoch + 1, best_epoch=best_epoch, best_val_score=best_score)
                stdout_message(f"[EARLY STOPPING] epoch={epoch + 1} best_epoch={best_epoch} best_val={best_score}")
                break
    if best_epoch < 0:
        raise RuntimeError("no checkpoint selected from validation set")
    return best_state, best_epoch, best_score, epochs_completed


def train_sequence_model(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float,
    max_epochs: int,
    patience: int,
    logger: JobLogger,
) -> tuple[dict[str, torch.Tensor], int, float, int]:
    optimizer = NAdam(model.parameters(), lr=lr, weight_decay=0.0)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    epochs_completed = 0
    stdout_message("[TRAIN START]")
    for epoch in range(max_epochs):
        logger.event("train_epoch_started", epoch=epoch + 1, max_epochs=max_epochs)
        stdout_message(f"[TRAIN EPOCH] epoch={epoch + 1}/{max_epochs}")
        model.train()
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x).squeeze(-1)
            loss = adt_pearson_loss(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_score = evaluate_sequence_val(model, val_loader, device)
        epochs_completed = epoch + 1
        logger.event("val_epoch_completed", epoch=epoch + 1, val_score=val_score)
        stdout_message(f"[VAL DONE] epoch={epoch + 1} val_score={val_score} best_val={best_score if best_epoch >= 0 else 'None'}")
        if math.isfinite(val_score) and val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
            logger.event("best_checkpoint_updated", epoch=epoch + 1, best_val_score=best_score)
            stdout_message(f"[BEST UPDATED] epoch={epoch + 1} best_val={best_score}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.event("early_stopping_triggered", epoch=epoch + 1, best_epoch=best_epoch, best_val_score=best_score)
                stdout_message(f"[EARLY STOPPING] epoch={epoch + 1} best_epoch={best_epoch} best_val={best_score}")
                break
    if best_epoch < 0:
        raise RuntimeError("no checkpoint selected from validation set")
    return best_state, best_epoch, best_score, epochs_completed


def save_zero_shot_checkpoint(
    *,
    config: dict,
    dataset_cfg: dict[str, object],
    split_manifest: dict[str, object],
    model_name: str,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_score: float,
    config_path: Path,
) -> tuple[str, Path]:
    dataset = str(dataset_cfg["dataset_id"])
    split_id = str(split_manifest["split_id"])
    seed = int(config["seed"])
    checkpoint_id = zero_shot_checkpoint_local_id(dataset=dataset, model=model_name, seed=seed, split_id=split_id, best_epoch=best_epoch)
    path = zero_shot_checkpoint_root() / zero_shot_checkpoint_relative_path(dataset=dataset, model=model_name, seed=seed, split_id=split_id, best_epoch=best_epoch)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in best_state.items()},
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "train_subjects": list(split_manifest["train_subjects"]),
        "val_subjects": list(split_manifest["val_subjects"]),
        "test_subjects": list(split_manifest["test_subjects"]),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "protocol": stage_protocol(config, "zero_shot"),
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / str(dataset_cfg["split_manifest_path"])),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    torch.save(payload, path)
    return checkpoint_id, path


def save_pooled10_checkpoint(
    *,
    config: dict,
    dataset_cfg: dict[str, object],
    split_manifest: dict[str, object],
    model_name: str,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_score: float,
    config_path: Path,
    source_payload: dict[str, object],
    duration_audit: dict[str, object],
) -> tuple[str, Path]:
    dataset = str(dataset_cfg["dataset_id"])
    split_id = str(split_manifest["split_id"])
    seed = int(config["seed"])
    checkpoint_id = pooled10_checkpoint_local_id(dataset=dataset, model=model_name, seed=seed, split_id=split_id, best_epoch=best_epoch)
    path = pooled10_checkpoint_root() / pooled10_checkpoint_relative_path(dataset=dataset, model=model_name, seed=seed, split_id=split_id, best_epoch=best_epoch)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in best_state.items()},
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "protocol": stage_protocol(config, "pooled10_finetune"),
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / str(dataset_cfg["split_manifest_path"])),
        "source_checkpoint_local_id": source_payload.get("checkpoint_local_id") or source_payload.get("source_checkpoint_local_id"),
        "source_checkpoint_relative_path": source_payload.get("checkpoint_relative_path") or source_payload.get("source_checkpoint_relative_path"),
        "source_checkpoint_best_epoch": int(source_payload["best_epoch"]),
        "source_checkpoint_best_val_score": float(source_payload["best_val_score"]),
        "pooled_test_subjects": list(split_manifest["test_subjects"]),
        "calibration_train_recording_ids": [
            rec_id
            for subject in duration_audit["subjects"]
            for rec_id in subject["calibration"]["train_recording_ids"]
        ],
        "calibration_val_recording_ids": [
            rec_id
            for subject in duration_audit["subjects"]
            for rec_id in subject["calibration"]["val_recording_ids"]
        ],
        "test_recording_ids": [
            rec_id
            for subject in duration_audit["subjects"]
            for rec_id in subject["final_test"]["recording_ids"]
        ],
        "actual_train_seconds": float(duration_audit["pooled"]["train_seconds"]),
        "actual_val_seconds": float(duration_audit["pooled"]["val_seconds"]),
        "actual_calibration_seconds": float(duration_audit["pooled"]["calibration_total_seconds"]),
        "final_test_seconds": float(duration_audit["pooled"]["final_test_seconds"]),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    torch.save(payload, path)
    return checkpoint_id, path


def build_manual_commands(config_path: Path, selected_datasets: list[dict[str, object]], selected_models: list[str]) -> list[str]:
    datasets_arg = " ".join(str(item["dataset_id"]) for item in selected_datasets)
    models_arg = " ".join(selected_models)
    config_path_ps = str(config_path).replace("/", "\\")
    return [
        f"cd {ROOT}",
        f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_fixed_split_pooled10_modelset_v1.py --config {config_path_ps} --device auto --resume --datasets {datasets_arg} --models {models_arg} --stage zero_shot --max-jobs 1",
        f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_fixed_split_pooled10_modelset_v1.py --config {config_path_ps} --device auto --resume --datasets {datasets_arg} --models {models_arg} --stage pooled_finetune --max-jobs 1",
        f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_fixed_split_pooled10_modelset_v1.py --config {config_path_ps} --device auto --resume --datasets {datasets_arg} --models {models_arg} --stage all --max-jobs 2",
    ]


def write_preflight_artifacts(
    *,
    config: dict,
    config_path: Path,
    output_dir: Path,
    duration_audits: list[dict[str, object]],
    shape_checks: list[dict[str, object]],
    checkpoint_checks: list[dict[str, object]],
    job_plan: dict[str, object],
) -> None:
    run_manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "base_branch": config["base_branch"],
        "base_commit": config["base_commit"],
        "current_branch": current_git_branch_name(),
        "config_path": repo_relative(config_path),
        "commit_sha": current_git_commit_sha(),
        "stage_scope": "preflight_only_no_training_started",
        "artifacts": {
            "duration_audit": repo_relative(output_dir / "duration_audit.json"),
            "calibration_plan": repo_relative(output_dir / "calibration_plan.json"),
            "checkpoint_manifest": repo_relative(output_dir / "checkpoint_manifest.json"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
        },
    }
    calibration_plan = {
        "protocol": config["protocol"],
        "selection_rule": config["pooled_fine_tune"]["selection_rule"],
        "datasets": duration_audits,
    }
    checkpoint_manifest = {"checkpoints": checkpoint_checks}
    model_run_entries = {"model_run_entries": shape_checks}
    failures = {"failures": []}
    schema_validation = {
        "protocol": config["protocol"],
        "passed": all(item["all_test_subjects_support_10min"] for item in duration_audits) and all(item["passed"] for item in shape_checks),
        "checks": {
            "split_manifests_loaded": True,
            "duration_audit_passed": all(item["all_test_subjects_support_10min"] for item in duration_audits),
            "shape_checks_passed": all(item["passed"] for item in shape_checks),
            "failure_report_empty": True,
        },
        "notes": ["preflight only; no zero-shot or pooled10 fine-tuning jobs executed"],
    }
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    atomic_write_json(output_dir / "duration_audit.json", {"datasets": duration_audits})
    atomic_write_json(output_dir / "calibration_plan.json", calibration_plan)
    atomic_write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    atomic_write_json(output_dir / "model_run_entries.json", model_run_entries)
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": []})
    atomic_write_json(output_dir / "run_state.json", {"phase": "preflight_only", "pending_jobs": job_plan["planned_jobs"], "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
    atomic_write_json(output_dir / "failure_report.json", failures)
    atomic_write_json(output_dir / "schema_validation_report.json", schema_validation)
    atomic_write_text(output_dir / "adaptation_or_leakage_summary.md", "# Preflight Only\n\nNo training started.\n")


def load_existing_runtime_state(output_dir: Path) -> dict[str, object]:
    return {
        "recording_rows": read_csv_rows(output_dir / "recording_metrics.csv"),
        "subject_rows": read_csv_rows(output_dir / "subject_metrics.csv"),
        "dataset_rows": read_csv_rows(output_dir / "dataset_metrics.csv"),
        "comparison_rows": read_csv_rows(output_dir / "zero_shot_vs_pooled10_finetune_comparison.csv"),
        "checkpoint_entries": list(read_json(output_dir / "checkpoint_manifest.json", {"checkpoints": []}).get("checkpoints", [])),
        "model_run_entries": list(read_json(output_dir / "model_run_entries.json", {"model_run_entries": []}).get("model_run_entries", [])),
        "completed_jobs": list(read_json(output_dir / "completed_jobs.json", {"completed_jobs": []}).get("completed_jobs", [])),
        "failures": list(read_json(output_dir / "failure_report.json", {"failures": []}).get("failures", [])),
    }


def write_runtime_state(
    *,
    config: dict,
    config_path: Path,
    output_dir: Path,
    duration_audits: list[dict[str, object]],
    job_plan: dict[str, object],
    state: dict[str, object],
) -> None:
    completed_job_keys = {str(item["job_key"]) for item in state["completed_jobs"]}
    pending_jobs = [job for job in job_plan["planned_jobs"] if str(job["job_key"]) not in completed_job_keys]
    dataset_ids = [audit["dataset_id"] for audit in duration_audits]
    leakage_lines = [
        "# Adaptation / Leakage Summary",
        "",
        f"- protocol: `{config['protocol']}`",
        "- pooled calibration uses only original train/val split recordings of fixed test subjects",
        "- final test always remains the original fixed test split",
        "",
    ]
    for audit in duration_audits:
        leakage_lines.append(f"## {audit['dataset_id']}")
        leakage_lines.append(f"- test_subjects: `{', '.join(audit['test_subjects'])}`")
        leakage_lines.append(f"- pooled_train_seconds: `{audit['pooled']['train_seconds']}`")
        leakage_lines.append(f"- pooled_val_seconds: `{audit['pooled']['val_seconds']}`")
        leakage_lines.append(f"- final_test_seconds: `{audit['pooled']['final_test_seconds']}`")
        leakage_lines.append("")
    schema_validation = {
        "protocol": config["protocol"],
        "passed": all(audit["all_test_subjects_support_10min"] for audit in duration_audits) and len(state["failures"]) == 0,
        "checks": {
            "duration_audit_passed": all(audit["all_test_subjects_support_10min"] for audit in duration_audits),
            "failure_report_empty": len(state["failures"]) == 0,
            "completed_jobs_written_incrementally": True,
            "checkpoint_manifest_written_incrementally": True,
        },
        "datasets": dataset_ids,
    }
    run_manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "base_branch": config["base_branch"],
        "base_commit": config["base_commit"],
        "current_branch": current_git_branch_name(),
        "config_path": repo_relative(config_path),
        "commit_sha": current_git_commit_sha(),
        "artifacts": {
            "duration_audit": repo_relative(output_dir / "duration_audit.json"),
            "calibration_plan": repo_relative(output_dir / "calibration_plan.json"),
            "checkpoint_manifest": repo_relative(output_dir / "checkpoint_manifest.json"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
            "zero_shot_vs_pooled10_finetune_comparison": repo_relative(output_dir / "zero_shot_vs_pooled10_finetune_comparison.csv"),
            "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "adaptation_or_leakage_summary": repo_relative(output_dir / "adaptation_or_leakage_summary.md"),
            "logs": repo_relative(output_dir / "logs"),
        },
    }
    calibration_plan = {
        "protocol": config["protocol"],
        "selection_rule": config["pooled_fine_tune"]["selection_rule"],
        "datasets": duration_audits,
    }
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    atomic_write_json(output_dir / "duration_audit.json", {"datasets": duration_audits})
    atomic_write_json(output_dir / "calibration_plan.json", calibration_plan)
    atomic_write_json(output_dir / "checkpoint_manifest.json", {"checkpoints": state["checkpoint_entries"]})
    atomic_write_json(output_dir / "model_run_entries.json", {"model_run_entries": state["model_run_entries"]})
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": state["completed_jobs"]})
    atomic_write_json(output_dir / "failure_report.json", {"failures": state["failures"]})
    atomic_write_json(output_dir / "schema_validation_report.json", schema_validation)
    atomic_write_json(
        output_dir / "run_state.json",
        {
            "phase": "partial" if pending_jobs else "completed",
            "completed_job_keys": sorted(completed_job_keys),
            "pending_jobs": pending_jobs,
            "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )
    atomic_write_text(output_dir / "adaptation_or_leakage_summary.md", "\n".join(leakage_lines) + "\n")
    atomic_write_csv(output_dir / "recording_metrics.csv", state["recording_rows"], RECORDING_METRIC_FIELDS)
    atomic_write_csv(output_dir / "subject_metrics.csv", state["subject_rows"], SUBJECT_METRIC_FIELDS)
    atomic_write_csv(output_dir / "dataset_metrics.csv", state["dataset_rows"], DATASET_METRIC_FIELDS)
    atomic_write_csv(output_dir / "zero_shot_vs_pooled10_finetune_comparison.csv", state["comparison_rows"], COMPARISON_FIELDS)


def read_duration_audit_for_dataset(output_dir: Path, dataset_id: str) -> dict[str, object]:
    payload = read_json(output_dir / "duration_audit.json", {"datasets": []})
    datasets = payload.get("datasets", [])
    for item in datasets:
        if str(item["dataset_id"]) == dataset_id:
            return item
    raise KeyError(f"duration audit not found for {dataset_id}")


def source_checkpoint_info_from_state(state: dict[str, object], dataset: str, model: str, seed: int, split_id: str) -> tuple[Path, dict[str, object]]:
    for entry in state["checkpoint_entries"]:
        if str(entry.get("dataset")) == dataset and str(entry.get("model")) == model and int(entry.get("seed")) == seed and str(entry.get("split_id")) == split_id and str(entry.get("stage")) == "zero_shot":
            path = ROOT / str(entry["checkpoint_relative_path"])
            payload = load_checkpoint_metadata(path)
            payload["checkpoint_local_id"] = entry["checkpoint_local_id"]
            payload["checkpoint_relative_path"] = entry["checkpoint_relative_path"]
            return path, payload
    reusable = discover_reusable_zero_shot_checkpoint(dataset=dataset, model=model, seed=seed, split_id=split_id)
    if reusable is None or not reusable.exists():
        raise FileNotFoundError(f"zero-shot checkpoint not found for pooled10 fine-tune: {dataset}/{model}/seed{seed}/{split_id}")
    payload = load_checkpoint_metadata(reusable)
    payload["checkpoint_local_id"] = payload.get("checkpoint_local_id") or zero_shot_checkpoint_local_id(dataset=dataset, model=model, seed=seed, split_id=split_id, best_epoch=int(payload["best_epoch"]))
    payload["checkpoint_relative_path"] = repo_relative(reusable)
    return reusable, payload


def run_zero_shot_job(
    *,
    config: dict,
    config_path: Path,
    dataset_cfg: dict[str, object],
    split_manifest: dict[str, object],
    model_name: str,
    output_dir: Path,
    device: str,
    job: dict[str, object],
) -> dict[str, object]:
    dataset_id = str(dataset_cfg["dataset_id"])
    split_id = str(split_manifest["split_id"])
    seed = int(config["seed"])
    stage = "zero_shot"
    logger = JobLogger(output_dir / "logs" / f"{dataset_id}_{split_id}_{model_name}_seed{seed}_zero_shot.log")
    dataset_dir = resolve_dataset_path(str(dataset_cfg["dataset_locator"]))
    contract = model_contract(config, model_name)
    model_handle, model_kwargs = instantiate_model(config, model_name)
    if bool(job.get("reuse_allowed")):
        source_path = discover_reusable_zero_shot_checkpoint(dataset=dataset_id, model=model_name, seed=seed, split_id=split_id)
        if source_path is None:
            raise FileNotFoundError(f"reusable checkpoint disappeared for {dataset_id}/{model_name}")
        payload = load_checkpoint_metadata(source_path)
        state_dict = payload["model_state_dict"]
        best_epoch = int(payload["best_epoch"])
        best_val_score = float(payload["best_val_score"])
        epochs_completed = int(payload.get("epochs_completed", best_epoch + 1))
        logger.event("zero_shot_reused", checkpoint_path=repo_relative(source_path))
    else:
        if contract["family"] == "window_scalar":
            train_dataset = SubjectSplitWindowDataset(
                dataset_dir,
                "train",
                subjects=list(split_manifest["train_subjects"]),
                window_size=model_window_size(config, model_name),
                channels=range(64),
                target_index=str(config[model_name]["target_index"]),
            )
            val_dataset = SubjectSplitWindowDataset(
                dataset_dir,
                "val",
                subjects=list(split_manifest["val_subjects"]),
                window_size=model_window_size(config, model_name),
                channels=range(64),
                target_index=str(config[model_name]["target_index"]),
            )
            train_loader = DataLoader(train_dataset, batch_size=int(config[model_name]["batch_size"]), shuffle=bool(config["dataloader"]["shuffle_train"]), num_workers=0, pin_memory=(device == "cuda"))
            val_loader = DataLoader(val_dataset, batch_size=int(config["dataloader"]["eval_batch_size"]), shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
            model = model_handle(**model_kwargs).to(device)
            state_dict, best_epoch, best_val_score, epochs_completed = train_window_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=float(config[model_name]["learning_rate"]),
                weight_decay=float(config[model_name].get("weight_decay", 0.0)),
                max_epochs=int(config[model_name]["max_epochs"]),
                patience=int(config[model_name]["early_stopping_patience"]),
                logger=logger,
            )
        else:
            train_dataset = SubjectSplitSequenceDataset(
                dataset_dir,
                "train",
                subjects=list(split_manifest["train_subjects"]),
                window_length=int(config["adt"]["window_length"]),
                hop_length=int(config["adt"]["hop_length"]),
                channels=range(64),
            )
            val_dataset = SubjectSplitSequenceDataset(
                dataset_dir,
                "val",
                subjects=list(split_manifest["val_subjects"]),
                window_length=int(config["adt"]["window_length"]),
                hop_length=int(config["adt"]["hop_length"]),
                channels=range(64),
            )
            train_loader = DataLoader(train_dataset, batch_size=int(config["adt"]["batch_size"]), shuffle=bool(config["dataloader"]["shuffle_train"]), num_workers=0, pin_memory=(device == "cuda"))
            val_loader = DataLoader(val_dataset, batch_size=int(config["dataloader"]["eval_batch_size"]), shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
            model = model_handle(**model_kwargs).to(device)
            state_dict, best_epoch, best_val_score, epochs_completed = train_sequence_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=float(config["adt"]["learning_rate"]),
                max_epochs=int(config["adt"]["max_epochs"]),
                patience=int(config["adt"]["early_stopping_patience"]),
                logger=logger,
            )
        checkpoint_local_id, checkpoint_path = save_zero_shot_checkpoint(
            config=config,
            dataset_cfg=dataset_cfg,
            split_manifest=split_manifest,
            model_name=model_name,
            best_state=state_dict,
            best_epoch=best_epoch,
            best_val_score=best_val_score,
            config_path=config_path,
        )
        logger.event("checkpoint_saved", checkpoint_local_id=checkpoint_local_id, checkpoint_path=repo_relative(checkpoint_path))
        stdout_message(f"[CHECKPOINT SAVED] checkpoint_local_id={checkpoint_local_id} path={repo_relative(checkpoint_path)}")
        payload = load_checkpoint_metadata(checkpoint_path)
        payload["checkpoint_local_id"] = checkpoint_local_id
        payload["checkpoint_relative_path"] = repo_relative(checkpoint_path)
        source_path = checkpoint_path
    checkpoint_local_id = str(payload.get("checkpoint_local_id") or zero_shot_checkpoint_local_id(dataset=dataset_id, model=model_name, seed=seed, split_id=split_id, best_epoch=best_epoch))
    checkpoint_rel_path = str(payload.get("checkpoint_relative_path") or repo_relative(source_path))
    checkpoint_id = f"{model_name}_epoch_{best_epoch}"
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    stdout_message("[TEST START]")
    test_recording_ids = collect_test_recording_ids(dataset_dir, list(split_manifest["test_subjects"]))
    if contract["family"] == "window_scalar":
        recording_rows, subject_rows = predict_recordings_window(
            config=config,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model=model,
            checkpoint_id=checkpoint_id,
            recording_ids=test_recording_ids,
            device=device,
            stage=stage,
        )
    else:
        recording_rows, subject_rows = predict_recordings_sequence(
            config=config,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model=model,
            checkpoint_id=checkpoint_id,
            recording_ids=test_recording_ids,
            device=device,
            stage=stage,
        )
    stdout_message("[TEST DONE]")
    dataset_row = mean_dataset_metric(subject_rows, stage=stage, split_id=split_id)
    logger.event("job_completed", mean_pearson=dataset_row["mean_pearson"], n_test_subjects=dataset_row["n_subjects"])
    stdout_message(f"[JOB DONE] mean_pearson={dataset_row['mean_pearson']} n_test_subjects={dataset_row['n_subjects']}")
    checkpoint_entry = {
        "dataset": dataset_id,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "stage": stage,
        "source_job_key": build_job_key(dataset_id, model_name, seed, split_id, stage),
        "checkpoint_local_id": checkpoint_local_id,
        "checkpoint_relative_path": checkpoint_rel_path,
        "checkpoint_artifact_status": "local_only_not_committed",
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "train_subjects": list(split_manifest["train_subjects"]),
        "val_subjects": list(split_manifest["val_subjects"]),
        "test_subjects": list(split_manifest["test_subjects"]),
        "protocol": stage_protocol(config, stage),
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / str(dataset_cfg["split_manifest_path"])),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
        "reused_existing_checkpoint": bool(job.get("reuse_allowed")),
    }
    model_run_entry = {
        "job_key": build_job_key(dataset_id, model_name, seed, split_id, stage),
        "dataset": dataset_id,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "stage": stage,
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id,
        "epochs_completed": int(epochs_completed),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "prediction_target_alignment_ok": True,
    }
    completed_job = {
        "job_key": build_job_key(dataset_id, model_name, seed, split_id, stage),
        "dataset": dataset_id,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "stage": stage,
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id,
    }
    return {
        "recording_rows": recording_rows,
        "subject_rows": subject_rows,
        "dataset_row": dataset_row,
        "comparison_rows": [],
        "checkpoint_entry": checkpoint_entry,
        "model_run_entry": model_run_entry,
        "completed_job": completed_job,
    }


def group_subject_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    return {str(row["subject_id"]): float(row["metric_value"]) for row in rows}


def run_pooled10_finetune_job(
    *,
    config: dict,
    config_path: Path,
    dataset_cfg: dict[str, object],
    split_manifest: dict[str, object],
    model_name: str,
    output_dir: Path,
    device: str,
    duration_audit: dict[str, object],
    state: dict[str, object],
) -> dict[str, object]:
    if not duration_audit["all_test_subjects_support_10min"]:
        raise RuntimeError(f"duration audit failed for pooled10 on dataset={dataset_cfg['dataset_id']}")
    dataset_id = str(dataset_cfg["dataset_id"])
    split_id = str(split_manifest["split_id"])
    seed = int(config["seed"])
    stage = "pooled10_finetune"
    logger = JobLogger(output_dir / "logs" / f"{dataset_id}_{split_id}_{model_name}_seed{seed}_pooled10_finetune.log")
    dataset_dir = resolve_dataset_path(str(dataset_cfg["dataset_locator"]))
    source_path, source_payload = source_checkpoint_info_from_state(state, dataset_id, model_name, seed, split_id)
    model_handle, model_kwargs = instantiate_model(config, model_name)
    contract = model_contract(config, model_name)
    pooled_train_ids = [rec_id for subject in duration_audit["subjects"] for rec_id in subject["calibration"]["train_recording_ids"]]
    pooled_val_ids = [rec_id for subject in duration_audit["subjects"] for rec_id in subject["calibration"]["val_recording_ids"]]
    pooled_test_ids = [rec_id for subject in duration_audit["subjects"] for rec_id in subject["final_test"]["recording_ids"]]
    if contract["family"] == "window_scalar":
        train_dataset = SelectedRecordingWindowDataset(
            dataset_dir,
            recording_ids=pooled_train_ids,
            window_size=model_window_size(config, model_name),
            channels=range(64),
            target_index=str(config[model_name]["target_index"]),
        )
        val_dataset = SelectedRecordingWindowDataset(
            dataset_dir,
            recording_ids=pooled_val_ids,
            window_size=model_window_size(config, model_name),
            channels=range(64),
            target_index=str(config[model_name]["target_index"]),
        )
        train_loader = DataLoader(train_dataset, batch_size=int(config[model_name]["batch_size"]), shuffle=bool(config["dataloader"]["shuffle_train"]), num_workers=0, pin_memory=(device == "cuda"))
        val_loader = DataLoader(val_dataset, batch_size=int(config["dataloader"]["eval_batch_size"]), shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
        model = model_handle(**model_kwargs).to(device)
        model.load_state_dict(source_payload["model_state_dict"], strict=True)
        state_dict, best_epoch, best_val_score, epochs_completed = train_window_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=float(config[model_name]["fine_tune_learning_rate"]),
            weight_decay=float(config[model_name].get("weight_decay", 0.0)),
            max_epochs=int(config[model_name]["max_epochs"]),
            patience=int(config[model_name]["early_stopping_patience"]),
            logger=logger,
        )
    else:
        train_dataset = SelectedRecordingSequenceDataset(
            dataset_dir,
            recording_ids=pooled_train_ids,
            window_length=int(config["adt"]["window_length"]),
            hop_length=int(config["adt"]["hop_length"]),
            channels=range(64),
        )
        val_dataset = SelectedRecordingSequenceDataset(
            dataset_dir,
            recording_ids=pooled_val_ids,
            window_length=int(config["adt"]["window_length"]),
            hop_length=int(config["adt"]["hop_length"]),
            channels=range(64),
        )
        train_loader = DataLoader(train_dataset, batch_size=int(config["adt"]["batch_size"]), shuffle=bool(config["dataloader"]["shuffle_train"]), num_workers=0, pin_memory=(device == "cuda"))
        val_loader = DataLoader(val_dataset, batch_size=int(config["dataloader"]["eval_batch_size"]), shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
        model = model_handle(**model_kwargs).to(device)
        model.load_state_dict(source_payload["model_state_dict"], strict=True)
        state_dict, best_epoch, best_val_score, epochs_completed = train_sequence_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=float(config["adt"]["fine_tune_learning_rate"]),
            max_epochs=int(config["adt"]["max_epochs"]),
            patience=int(config["adt"]["early_stopping_patience"]),
            logger=logger,
        )
    checkpoint_local_id, checkpoint_path = save_pooled10_checkpoint(
        config=config,
        dataset_cfg=dataset_cfg,
        split_manifest=split_manifest,
        model_name=model_name,
        best_state=state_dict,
        best_epoch=best_epoch,
        best_val_score=best_val_score,
        config_path=config_path,
        source_payload=source_payload,
        duration_audit=duration_audit,
    )
    logger.event("checkpoint_saved", checkpoint_local_id=checkpoint_local_id, checkpoint_path=repo_relative(checkpoint_path))
    stdout_message(f"[CHECKPOINT SAVED] checkpoint_local_id={checkpoint_local_id} path={repo_relative(checkpoint_path)}")
    fine_model = model_handle(**model_kwargs).to(device)
    fine_model.load_state_dict(state_dict, strict=True)
    zero_model = model_handle(**model_kwargs).to(device)
    zero_model.load_state_dict(source_payload["model_state_dict"], strict=True)
    checkpoint_id = f"{model_name}_pooled10_epoch_{best_epoch}"
    zero_checkpoint_id = f"{model_name}_epoch_{int(source_payload['best_epoch'])}"
    stdout_message("[TEST START]")
    if contract["family"] == "window_scalar":
        recording_rows, subject_rows = predict_recordings_window(
            config=config,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model=fine_model,
            checkpoint_id=checkpoint_id,
            recording_ids=pooled_test_ids,
            device=device,
            stage=stage,
        )
        zero_recording_rows, zero_subject_rows = predict_recordings_window(
            config=config,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model=zero_model,
            checkpoint_id=zero_checkpoint_id,
            recording_ids=pooled_test_ids,
            device=device,
            stage="zero_shot",
        )
    else:
        recording_rows, subject_rows = predict_recordings_sequence(
            config=config,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model=fine_model,
            checkpoint_id=checkpoint_id,
            recording_ids=pooled_test_ids,
            device=device,
            stage=stage,
        )
        zero_recording_rows, zero_subject_rows = predict_recordings_sequence(
            config=config,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model=zero_model,
            checkpoint_id=zero_checkpoint_id,
            recording_ids=pooled_test_ids,
            device=device,
            stage="zero_shot",
        )
    stdout_message("[TEST DONE]")
    zero_metrics = group_subject_metrics(zero_subject_rows)
    fine_metrics = group_subject_metrics(subject_rows)
    comparison_rows: list[dict[str, object]] = []
    for subject in duration_audit["subjects"]:
        subject_id = str(subject["subject_id"])
        comparison_rows.append(
            {
                "dataset": dataset_id,
                "model": model_name,
                "seed": seed,
                "split_id": split_id,
                "subject_id": subject_id,
                "source_checkpoint_local_id": str(source_payload["checkpoint_local_id"]),
                "source_checkpoint_relative_path": str(source_payload["checkpoint_relative_path"]),
                "source_checkpoint_best_epoch": int(source_payload["best_epoch"]),
                "source_checkpoint_best_val_score": float(source_payload["best_val_score"]),
                "zero_shot_metric": float(zero_metrics[subject_id]),
                "pooled10_fine_tuned_metric": float(fine_metrics[subject_id]),
                "delta": float(fine_metrics[subject_id] - zero_metrics[subject_id]),
                "actual_calibration_seconds": float(subject["calibration"]["actual_total_seconds"]),
                "actual_train_seconds": float(subject["calibration"]["actual_train_seconds"]),
                "actual_val_seconds": float(subject["calibration"]["actual_val_seconds"]),
                "final_test_seconds": float(subject["final_test"]["seconds"]),
                "calibration_train_recording_ids": "|".join(subject["calibration"]["train_recording_ids"]),
                "calibration_val_recording_ids": "|".join(subject["calibration"]["val_recording_ids"]),
                "test_recording_ids": "|".join(subject["final_test"]["recording_ids"]),
            }
        )
    dataset_row = mean_dataset_metric(subject_rows, stage=stage, split_id=split_id)
    logger.event("job_completed", mean_pearson=dataset_row["mean_pearson"], n_test_subjects=dataset_row["n_subjects"])
    stdout_message(f"[JOB DONE] mean_pearson={dataset_row['mean_pearson']} n_test_subjects={dataset_row['n_subjects']}")
    checkpoint_entry = {
        "dataset": dataset_id,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "stage": stage,
        "source_job_key": build_job_key(dataset_id, model_name, seed, split_id, "zero_shot"),
        "source_checkpoint_local_id": str(source_payload["checkpoint_local_id"]),
        "source_checkpoint_relative_path": str(source_payload["checkpoint_relative_path"]),
        "source_checkpoint_load_verified": True,
        "source_checkpoint_best_epoch": int(source_payload["best_epoch"]),
        "source_checkpoint_best_val_score": float(source_payload["best_val_score"]),
        "checkpoint_local_id": checkpoint_local_id,
        "checkpoint_relative_path": repo_relative(checkpoint_path),
        "checkpoint_artifact_status": "local_only_not_committed",
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "protocol": stage_protocol(config, stage),
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / str(dataset_cfg["split_manifest_path"])),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
    }
    model_run_entry = {
        "job_key": build_job_key(dataset_id, model_name, seed, split_id, stage),
        "dataset": dataset_id,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "stage": stage,
        "status": "success",
        "source_checkpoint_local_id": str(source_payload["checkpoint_local_id"]),
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id,
        "epochs_completed": int(epochs_completed),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "pooled_train_seconds": float(duration_audit["pooled"]["train_seconds"]),
        "pooled_val_seconds": float(duration_audit["pooled"]["val_seconds"]),
        "pooled_calibration_total_seconds": float(duration_audit["pooled"]["calibration_total_seconds"]),
        "final_test_seconds": float(duration_audit["pooled"]["final_test_seconds"]),
        "prediction_target_alignment_ok": True,
    }
    completed_job = {
        "job_key": build_job_key(dataset_id, model_name, seed, split_id, stage),
        "dataset": dataset_id,
        "model": model_name,
        "seed": seed,
        "split_id": split_id,
        "stage": stage,
        "status": "success",
        "source_checkpoint_local_id": str(source_payload["checkpoint_local_id"]),
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id,
    }
    return {
        "recording_rows": recording_rows,
        "subject_rows": subject_rows,
        "dataset_row": dataset_row,
        "comparison_rows": comparison_rows,
        "checkpoint_entry": checkpoint_entry,
        "model_run_entry": model_run_entry,
        "completed_job": completed_job,
    }


def append_job_result(state: dict[str, object], result: dict[str, object]) -> None:
    state["recording_rows"].extend(result["recording_rows"])
    state["subject_rows"].extend(result["subject_rows"])
    state["dataset_rows"].append(result["dataset_row"])
    state["comparison_rows"].extend(result["comparison_rows"])
    state["checkpoint_entries"].append(result["checkpoint_entry"])
    state["model_run_entries"].append(result["model_run_entry"])
    state["completed_jobs"].append(result["completed_job"])


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    device = resolve_device(args.device)
    selected_datasets = resolve_selected_datasets(config, args.datasets)
    selected_models = resolve_selected_models(config, args.models)
    duration_audits = [build_duration_audit(config, dataset_cfg) for dataset_cfg in selected_datasets]
    shape_checks = [shape_check_for_model(config, model_name, device) for model_name in selected_models]
    checkpoint_checks = checkpoint_path_check(config, selected_datasets, selected_models, device)
    job_plan = build_job_plan(config, selected_datasets, selected_models, args.stage)
    write_preflight_artifacts(
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        duration_audits=duration_audits,
        shape_checks=shape_checks,
        checkpoint_checks=checkpoint_checks,
        job_plan=job_plan,
    )
    manual_commands = build_manual_commands(config_path, selected_datasets, selected_models)
    if args.duration_audit_only:
        stdout_block("[DURATION AUDIT]", [json.dumps({"datasets": duration_audits}, indent=2)])
        return 0
    if args.shape_check_only:
        stdout_block("[SHAPE CHECK]", [json.dumps({"models": shape_checks}, indent=2)])
        return 0
    if args.checkpoint_path_check_only:
        stdout_block("[CHECKPOINT PATH CHECK]", [json.dumps({"checkpoints": checkpoint_checks}, indent=2)])
        return 0
    if args.startup_only or args.dry_run_plan:
        startup_lines = [
            f"config path: {config_path}",
            f"output_dir: {output_dir}",
            f"device: {device}",
            f"datasets: {[item['dataset_id'] for item in selected_datasets]}",
            f"models: {selected_models}",
            f"stage: {args.stage}",
            f"planned_jobs: {len(job_plan['planned_jobs'])}",
            f"zero_shot_jobs: {job_plan['zero_shot_job_count']}",
            f"pooled10_finetune_jobs: {job_plan['pooled10_finetune_job_count']}",
            "no training started: True",
        ]
        for audit in duration_audits:
            startup_lines.append(f"{audit['dataset_id']} test_subjects: {audit['test_subjects']}")
        stdout_block("[STARTUP SUMMARY]", startup_lines)
        if args.dry_run_plan:
            stdout_block("[DRY RUN PLAN]", [json.dumps(job_plan, indent=2), "", "Recommended manual commands:"] + manual_commands)
        return 0
    if args.rebuild_preflight_artifacts_only:
        stdout_block("[REBUILD DONE]", ["preflight artifacts rewritten", "no training started"])
        return 0

    state = load_existing_runtime_state(output_dir) if args.resume else {
        "recording_rows": [],
        "subject_rows": [],
        "dataset_rows": [],
        "comparison_rows": [],
        "checkpoint_entries": [],
        "model_run_entries": [],
        "completed_jobs": [],
        "failures": [],
    }
    completed_keys = {str(item["job_key"]) for item in state["completed_jobs"]}
    pending_jobs = [job for job in job_plan["planned_jobs"] if str(job["job_key"]) not in completed_keys]
    if args.max_jobs is not None:
        pending_jobs = pending_jobs[: int(args.max_jobs)]

    dataset_cfg_by_id = {str(item["dataset_id"]): item for item in selected_datasets}
    duration_by_id = {str(item["dataset_id"]): item for item in duration_audits}

    if not pending_jobs:
        stdout_block("[NO PENDING JOBS]", ["All requested jobs are already completed for the selected stage/dataset/model set."])
        write_runtime_state(
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            duration_audits=duration_audits,
            job_plan=job_plan,
            state=state,
        )
        return 0

    for job in pending_jobs:
        dataset_id = str(job["dataset"])
        model_name = str(job["model"])
        stage = str(job["stage"])
        dataset_cfg = dataset_cfg_by_id[dataset_id]
        split_manifest = load_split_manifest(ROOT / str(dataset_cfg["split_manifest_path"]))
        try:
            if stage == "zero_shot":
                result = run_zero_shot_job(
                    config=config,
                    config_path=config_path,
                    dataset_cfg=dataset_cfg,
                    split_manifest=split_manifest,
                    model_name=model_name,
                    output_dir=output_dir,
                    device=device,
                    job=job,
                )
            else:
                result = run_pooled10_finetune_job(
                    config=config,
                    config_path=config_path,
                    dataset_cfg=dataset_cfg,
                    split_manifest=split_manifest,
                    model_name=model_name,
                    output_dir=output_dir,
                    device=device,
                    duration_audit=duration_by_id[dataset_id],
                    state=state,
                )
            append_job_result(state, result)
        except Exception as exc:
            state["failures"].append(
                {
                    "job_key": job["job_key"],
                    "dataset": dataset_id,
                    "model": model_name,
                    "seed": int(config["seed"]),
                    "split_id": str(job["split_id"]),
                    "stage": stage,
                    "error": repr(exc),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
            )
            stdout_message(f"[JOB FAILED] {job['job_key']} error={exc!r}")
        write_runtime_state(
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            duration_audits=duration_audits,
            job_plan=job_plan,
            state=state,
        )
        stdout_message("[METRICS WRITTEN]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
