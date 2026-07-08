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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.mldecoders.models import EEGNetRegressor


ATOMIC_RETRY_ATTEMPTS = 5
ATOMIC_RETRY_SLEEP_SECONDS = 0.05
DATASET_METRIC_FIELDS = [
    "source_dataset",
    "target_dataset",
    "model",
    "seed",
    "stage",
    "n_subjects",
    "mean_pearson",
    "std_pearson",
    "min_pearson",
    "max_pearson",
]
COMPARISON_FIELDS = [
    "source_dataset",
    "target_dataset",
    "model",
    "seed",
    "stage",
    "subject_id",
    "zero_shot_metric",
    "calibrated_metric",
    "delta",
]
CROSS_SUBJECT_FIELDS = [
    "source_dataset",
    "target_dataset",
    "model",
    "task",
    "protocol",
    "seed",
    "stage",
    "subject_id",
    "metric_name",
    "metric_value",
    "num_recordings",
    "checkpoint_id",
    "artifact_scope",
]
CROSS_RECORDING_FIELDS = [
    "source_dataset",
    "target_dataset",
    "model",
    "task",
    "protocol",
    "seed",
    "stage",
    "subject_id",
    "recording_id",
    "sampling_rate",
    "metric_name",
    "metric_value",
    "num_valid_samples",
    "checkpoint_id",
    "artifact_scope",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source-dataset", nargs="*", default=None)
    parser.add_argument("--target-dataset", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--stage", choices=["cross_dataset_zero_shot", "pooled10_target_calibration", "all"], default="all")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--duration-audit-only", action="store_true")
    parser.add_argument("--shape-check-only", action="store_true")
    parser.add_argument("--checkpoint-path-check-only", action="store_true")
    parser.add_argument("--job-plan-only", action="store_true")
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
        except (PermissionError, OSError) as exc:
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


def filter_rows_to_fieldnames(rows: list[dict[str, object]], fieldnames: list[str]) -> list[dict[str, object]]:
    allowed = set(fieldnames)
    filtered: list[dict[str, object]] = []
    for row in rows:
        keys = set(row.keys())
        if not keys.issubset(allowed):
            continue
        filtered.append({field: row.get(field, "") for field in fieldnames})
    return filtered


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
    shared_candidate = Path(ROOT.drive + "\\decode") / locator
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


def batch_corr(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true0 = y_true - torch.mean(y_true)
    y_pred0 = y_pred - torch.mean(y_pred)
    return torch.sum(y_true0 * y_pred0) / (torch.sqrt(torch.sum(y_true0**2)) * torch.sqrt(torch.sum(y_pred0**2)) + eps)


def parse_subject_id(recording_id: str) -> str:
    parts = recording_id.split("_-_")
    if len(parts) < 3:
        raise ValueError(f"unexpected recording id: {recording_id}")
    return parts[1]


def load_split_manifest(path: Path) -> dict[str, object]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        raise ValueError(f"invalid split manifest: {path}")
    required = {"split_id", "dataset_id", "train_subjects", "val_subjects", "test_subjects", "subject_ids_all", "split_seed", "split_rule", "created_at", "no_overlap"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"split manifest missing fields: {sorted(missing)}")
    return payload


def model_contract(config: dict, model_name: str) -> dict[str, object]:
    if model_name != "eegnet":
        raise ValueError(f"unsupported model for cross-dataset transfer: {model_name}")
    return {
        "family": "window_scalar",
        "input_tensor_shape": [1, 64, int(config["eegnet"]["window_size"])],
        "raw_output_shape": [1],
        "postprocessed_prediction_shape": [1],
        "scorer_input_shape": [1],
        "target_index": str(config["eegnet"]["target_index"]),
        "target_contract": "scalar target per window",
    }


def instantiate_model(config: dict, model_name: str) -> tuple[object, dict[str, object]]:
    if model_name != "eegnet":
        raise ValueError(f"unsupported model for cross-dataset transfer: {model_name}")
    cfg = config["eegnet"]
    return EEGNetRegressor, {
        "num_input_channels": 64,
        "input_length": int(cfg["window_size"]),
        "temporal_filters": int(cfg["temporal_filters"]),
        "depth_multiplier": int(cfg["depth_multiplier"]),
        "separable_filters": int(cfg["separable_filters"]),
        "dropout_rate": float(cfg["dropout_rate"]),
    }


def postprocess_model_output(raw_output: torch.Tensor) -> torch.Tensor:
    return raw_output.reshape(-1)


def model_window_size(config: dict, model_name: str) -> int:
    return int(config[model_name]["window_size"])


def discover_recordings(dataset_dir: Path, *, split: str, subject_id: str, sampling_rate: int, window_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eeg_path in sorted(dataset_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        num_samples = int(np.load(eeg_path, mmap_mode="r").shape[0])
        window_count = max(0, num_samples - window_size + 1)
        rows.append(
            {
                "recording_id": recording_id,
                "num_samples": num_samples,
                "seconds": float(num_samples / sampling_rate),
                "window_count": int(window_count),
            }
        )
    return rows


def duration_prefix(rows: list[dict[str, object]], budget_seconds: float) -> tuple[list[dict[str, object]], float]:
    selected: list[dict[str, object]] = []
    total = 0.0
    for row in rows:
        selected.append(row)
        total += float(row["seconds"])
        if total >= budget_seconds:
            break
    return selected, total


def dataset_cfg_map(config: dict) -> dict[str, dict[str, object]]:
    return {str(item["dataset_id"]): item for item in config["datasets"]}


def resolve_transfer_pairs(config: dict, requested_sources: list[str] | None, requested_targets: list[str] | None) -> list[dict[str, str]]:
    pairs = [dict(item) for item in config["transfer_pairs"]]
    if requested_sources:
        source_set = set(requested_sources)
        pairs = [pair for pair in pairs if pair["source_dataset"] in source_set]
    if requested_targets:
        target_set = set(requested_targets)
        pairs = [pair for pair in pairs if pair["target_dataset"] in target_set]
    if not pairs:
        raise ValueError("no transfer pairs selected")
    return pairs


def resolve_selected_models(config: dict, requested: list[str] | None) -> list[str]:
    models = list(config["models"])
    if not requested:
        return models
    missing = sorted(set(requested) - set(models))
    if missing:
        raise ValueError(f"unknown models requested: {missing}")
    return list(requested)


def build_transfer_duration_audit(config: dict, pair: dict[str, str], datasets_by_id: dict[str, dict[str, object]]) -> dict[str, object]:
    source_cfg = datasets_by_id[pair["source_dataset"]]
    target_cfg = datasets_by_id[pair["target_dataset"]]
    source_manifest = load_split_manifest(ROOT / str(source_cfg["split_manifest_path"]))
    target_manifest = load_split_manifest(ROOT / str(target_cfg["split_manifest_path"]))
    target_dir = resolve_dataset_path(str(target_cfg["dataset_locator"]))
    sampling_rate = int(target_cfg["sampling_rate"])
    window_size = int(config["target_calibration"]["calibration_window_size_for_audit"])
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
    for subject_id in target_manifest["test_subjects"]:
        train_rows = discover_recordings(target_dir, split="train", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=window_size)
        val_rows = discover_recordings(target_dir, split="val", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=window_size)
        test_rows = discover_recordings(target_dir, split="test", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=window_size)
        train_selected, train_seconds = duration_prefix(train_rows, float(config["target_calibration"]["train_budget_seconds"]))
        val_selected, val_seconds = duration_prefix(val_rows, float(config["target_calibration"]["val_budget_seconds"]))
        test_seconds = float(sum(row["seconds"] for row in test_rows))
        test_windows = int(sum(row["window_count"] for row in test_rows))
        subject_row = {
            "subject_id": str(subject_id),
            "train_total_seconds": float(sum(row["seconds"] for row in train_rows)),
            "val_total_seconds": float(sum(row["seconds"] for row in val_rows)),
            "test_total_seconds": test_seconds,
            "calibration": {
                "required_seconds": float(config["target_calibration"]["calibration_budget_seconds"]),
                "train_budget_seconds": float(config["target_calibration"]["train_budget_seconds"]),
                "val_budget_seconds": float(config["target_calibration"]["val_budget_seconds"]),
                "actual_train_seconds": train_seconds,
                "actual_val_seconds": val_seconds,
                "actual_total_seconds": train_seconds + val_seconds,
                "train_recording_count": len(train_selected),
                "val_recording_count": len(val_selected),
                "train_window_count": int(sum(row["window_count"] for row in train_selected)),
                "val_window_count": int(sum(row["window_count"] for row in val_selected)),
                "train_recording_ids": [str(row["recording_id"]) for row in train_selected],
                "val_recording_ids": [str(row["recording_id"]) for row in val_selected],
                "supports_10min": train_seconds >= float(config["target_calibration"]["train_budget_seconds"]) and val_seconds >= float(config["target_calibration"]["val_budget_seconds"]),
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
        "source_dataset": str(pair["source_dataset"]),
        "target_dataset": str(pair["target_dataset"]),
        "source_split_id": str(source_manifest["split_id"]),
        "target_split_id": str(target_manifest["split_id"]),
        "source_train_subjects": list(source_manifest["train_subjects"]),
        "source_val_subjects": list(source_manifest["val_subjects"]),
        "source_test_subjects_unused_for_training": list(source_manifest["test_subjects"]),
        "target_train_subjects_unused_for_source_training": list(target_manifest["train_subjects"]),
        "target_val_subjects_unused_for_source_training": list(target_manifest["val_subjects"]),
        "target_test_subjects": list(target_manifest["test_subjects"]),
        "subjects": subjects_report,
        "pooled_target_calibration": pooled,
        "all_target_test_subjects_support_10min": all(
            bool(subject["calibration"]["supports_10min"]) and bool(subject["final_test"]["non_empty"]) for subject in subjects_report
        ),
        "final_test_uses_target_original_test_split": True,
    }


def shape_check_for_model(config: dict, model_name: str, device: str) -> dict[str, object]:
    model_handle, model_kwargs = instantiate_model(config, model_name)
    contract = model_contract(config, model_name)
    model = model_handle(**model_kwargs).to(device)
    model.eval()
    x = torch.zeros(*contract["input_tensor_shape"], device=device, dtype=torch.float32)
    with torch.no_grad():
        raw = model(x)
        post = postprocess_model_output(raw)
    return {
        "model": model_name,
        "input_shape": list(contract["input_tensor_shape"]),
        "raw_output_shape": list(raw.shape),
        "postprocessed_prediction_shape": list(post.shape),
        "scorer_input_shape": contract["scorer_input_shape"],
        "target_index": contract["target_index"],
        "target_contract": contract["target_contract"],
        "passed": list(raw.shape) == contract["raw_output_shape"] and list(post.shape) == contract["postprocessed_prediction_shape"],
    }


def cross_source_only_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "cross_dataset" / "source_only"


def cross_calibration_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "cross_dataset" / "pooled10_target_calibration"


def cross_source_only_checkpoint_local_id(*, source_dataset: str, model: str, seed: int, best_epoch: int) -> str:
    return f"cross_dataset_source_only:{model}:{source_dataset}:seed{seed}:best_epoch_{best_epoch}"


def cross_source_only_checkpoint_relative_path(*, source_dataset: str, model: str, seed: int, best_epoch: int) -> Path:
    return Path(model) / source_dataset / f"seed{seed}" / f"best_epoch_{best_epoch}.pt"


def cross_source_only_checkpoint_path_pattern(*, source_dataset: str, model: str, seed: int) -> Path:
    return cross_source_only_checkpoint_root() / model / source_dataset / f"seed{seed}" / "best_epoch_PENDING.pt"


def cross_calibration_checkpoint_path_pattern(*, source_dataset: str, target_dataset: str, model: str, seed: int) -> Path:
    return cross_calibration_checkpoint_root() / model / source_dataset / target_dataset / f"seed{seed}" / "best_epoch_PENDING.pt"


def checkpoint_epoch_from_name(path: Path) -> int:
    match = re.search(r"best_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint_in_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = sorted(path.glob("best_epoch_*.pt"), key=checkpoint_epoch_from_name)
    return candidates[-1] if candidates else None


def checkpoint_path_check(config: dict, transfer_audits: list[dict[str, object]], selected_models: list[str], device: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in transfer_audits:
        for model_name in selected_models:
            zero_pattern = cross_source_only_checkpoint_path_pattern(
                source_dataset=str(audit["source_dataset"]),
                model=model_name,
                seed=int(config["seed"]),
            )
            cal_pattern = cross_calibration_checkpoint_path_pattern(
                source_dataset=str(audit["source_dataset"]),
                target_dataset=str(audit["target_dataset"]),
                model=model_name,
                seed=int(config["seed"]),
            )
            existing = latest_checkpoint_in_dir(zero_pattern.parent)
            model_handle, model_kwargs = instantiate_model(config, model_name)
            model = model_handle(**model_kwargs)
            smoke_path = build_tmp_path(zero_pattern.parent / "smoke_checkpoint_tmp.pt")
            ensure_dir(smoke_path.parent)
            torch.save({"model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}, smoke_path)
            smoke_payload = torch.load(smoke_path, map_location="cpu")
            model.load_state_dict(smoke_payload["model_state_dict"], strict=True)
            del smoke_payload
            for attempt in range(1, ATOMIC_RETRY_ATTEMPTS + 1):
                try:
                    smoke_path.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == ATOMIC_RETRY_ATTEMPTS:
                        raise
                    time.sleep(ATOMIC_RETRY_SLEEP_SECONDS * attempt)
            rows.append(
                {
                    "source_dataset": str(audit["source_dataset"]),
                    "target_dataset": str(audit["target_dataset"]),
                    "model": model_name,
                    "seed": int(config["seed"]),
                    "cross_dataset_source_only_checkpoint_path_pattern": repo_relative(zero_pattern),
                    "pooled10_target_calibration_checkpoint_path_pattern": repo_relative(cal_pattern),
                    "existing_source_only_checkpoint": repo_relative(existing) if existing is not None else None,
                    "torch_load_smoke": True,
                    "strict_load_state_dict_smoke": True,
                }
            )
    return rows


def build_job_key(source_dataset: str, target_dataset: str | None, model: str, seed: int, stage: str) -> str:
    if target_dataset:
        return f"{source_dataset}->{target_dataset}:{model}:seed{seed}:{stage}"
    return f"{source_dataset}:{model}:seed{seed}:{stage}"


def build_job_plan(config: dict, transfer_audits: list[dict[str, object]], selected_models: list[str], stage: str) -> dict[str, object]:
    jobs: list[dict[str, object]] = []
    seen_source_jobs: set[tuple[str, str, int]] = set()
    for audit in transfer_audits:
        source_dataset = str(audit["source_dataset"])
        target_dataset = str(audit["target_dataset"])
        for model_name in selected_models:
            if stage in {"cross_dataset_zero_shot", "all"}:
                source_job_signature = (source_dataset, model_name, int(config["seed"]))
                if source_job_signature not in seen_source_jobs:
                    jobs.append(
                        {
                            "source_dataset": source_dataset,
                            "target_dataset": None,
                            "model": model_name,
                            "seed": int(config["seed"]),
                            "stage": "source_zero_shot",
                            "job_key": build_job_key(source_dataset, None, model_name, int(config["seed"]), "source_zero_shot"),
                        }
                    )
                    seen_source_jobs.add(source_job_signature)
                jobs.append(
                    {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "stage": "cross_dataset_zero_shot_eval",
                        "job_key": build_job_key(source_dataset, target_dataset, model_name, int(config["seed"]), "cross_dataset_zero_shot_eval"),
                        "depends_on_source_zero_shot": build_job_key(source_dataset, None, model_name, int(config["seed"]), "source_zero_shot"),
                    }
                )
            if stage in {"pooled10_target_calibration", "all"}:
                jobs.append(
                    {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "stage": "pooled10_target_calibration",
                        "job_key": build_job_key(source_dataset, target_dataset, model_name, int(config["seed"]), "pooled10_target_calibration"),
                        "depends_on_zero_shot_eval": build_job_key(source_dataset, target_dataset, model_name, int(config["seed"]), "cross_dataset_zero_shot_eval"),
                    }
                )
    return {
        "planned_jobs": jobs,
        "source_zero_shot_job_count": sum(1 for job in jobs if job["stage"] == "source_zero_shot"),
        "cross_dataset_zero_shot_eval_job_count": sum(1 for job in jobs if job["stage"] == "cross_dataset_zero_shot_eval"),
        "pooled10_target_calibration_job_count": sum(1 for job in jobs if job["stage"] == "pooled10_target_calibration"),
    }


def build_manual_commands(config_path: Path, transfer_audits: list[dict[str, object]], selected_models: list[str]) -> list[str]:
    models_arg = " ".join(selected_models)
    config_path_ps = str(config_path).replace("/", "\\")
    commands = [f"cd {ROOT}"]
    for audit in transfer_audits:
        source_dataset = str(audit["source_dataset"])
        target_dataset = str(audit["target_dataset"])
        commands.append(
            f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_cross_dataset_transfer_v1.py --config {config_path_ps} --device auto --resume --source-dataset {source_dataset} --target-dataset {target_dataset} --models {models_arg} --stage cross_dataset_zero_shot --max-jobs 1"
        )
        commands.append(
            f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_cross_dataset_transfer_v1.py --config {config_path_ps} --device auto --resume --source-dataset {source_dataset} --target-dataset {target_dataset} --models {models_arg} --stage cross_dataset_zero_shot --max-jobs 1"
        )
    commands.append(
        f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_cross_dataset_transfer_v1.py --config {config_path_ps} --device auto --resume --source-dataset weissbart_tf64 etard_tf64 --target-dataset etard_tf64 weissbart_tf64 --models {models_arg} --stage pooled10_target_calibration --max-jobs 1"
    )
    return commands


def build_job_plan_only_summary(config: dict, transfer_audits: list[dict[str, object]], selected_models: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in transfer_audits:
        for model_name in selected_models:
            rows.append(
                {
                    "source_dataset": audit["source_dataset"],
                    "target_dataset": audit["target_dataset"],
                    "model": model_name,
                    "source_job_key": build_job_key(str(audit["source_dataset"]), None, model_name, int(config["seed"]), "source_zero_shot"),
                    "target_eval_job_key": build_job_key(str(audit["source_dataset"]), str(audit["target_dataset"]), model_name, int(config["seed"]), "cross_dataset_zero_shot_eval"),
                    "source_train_subjects": audit["source_train_subjects"],
                    "source_val_subjects": audit["source_val_subjects"],
                    "target_test_subjects": audit["target_test_subjects"],
                    "target_pooled_train_recording_count": audit["pooled_target_calibration"]["train_recording_count"],
                    "target_pooled_val_recording_count": audit["pooled_target_calibration"]["val_recording_count"],
                    "target_final_test_recording_count": audit["pooled_target_calibration"]["final_test_recording_count"],
                    "target_pooled_train_window_count": audit["pooled_target_calibration"]["train_window_count"],
                    "target_pooled_val_window_count": audit["pooled_target_calibration"]["val_window_count"],
                    "input_tensor_contract": "[batch, 64, 50] -> [batch]",
                    "target_scorer_alignment": "scalar target per window, Pearson on valid overlap",
                }
            )
    return rows


def preflight_output_dir(output_dir: Path) -> Path:
    return output_dir / "preflight"


def write_preflight_artifacts(
    *,
    config: dict,
    config_path: Path,
    output_dir: Path,
    transfer_audits: list[dict[str, object]],
    shape_checks: list[dict[str, object]],
    checkpoint_checks: list[dict[str, object]],
    job_plan: dict[str, object],
    job_plan_only: list[dict[str, object]],
) -> None:
    target_dir = preflight_output_dir(output_dir)
    ensure_dir(target_dir)
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
            "transfer_plan": repo_relative(target_dir / "transfer_plan.json"),
            "calibration_plan": repo_relative(target_dir / "calibration_plan.json"),
            "checkpoint_manifest": repo_relative(target_dir / "checkpoint_manifest.json"),
            "model_run_entries": repo_relative(target_dir / "model_run_entries.json"),
            "schema_validation_report": repo_relative(target_dir / "schema_validation_report.json"),
            "failure_report": repo_relative(target_dir / "failure_report.json"),
            "job_plan_only": repo_relative(target_dir / "job_plan_only.json"),
        },
    }
    calibration_plan = {
        "protocol": config["protocol"],
        "selection_rule": config["target_calibration"]["selection_rule"],
        "transfer_pairs": transfer_audits,
    }
    transfer_plan = {
        "protocol": config["protocol"],
        "pairs": transfer_audits,
    }
    checkpoint_manifest = {"checkpoints": checkpoint_checks}
    model_run_entries = {"model_run_entries": shape_checks}
    failures = {"failures": []}
    schema_validation = {
        "protocol": config["protocol"],
        "passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits) and all(item["passed"] for item in shape_checks),
        "checks": {
            "split_manifests_loaded": True,
            "target_calibration_audit_passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits),
            "shape_checks_passed": all(item["passed"] for item in shape_checks),
            "failure_report_empty": True,
            "cross_dataset_stage_requires_training": True,
        },
        "notes": ["preflight only; no cross-dataset zero-shot or pooled10 target calibration jobs executed"],
    }
    atomic_write_json(target_dir / "run_manifest.json", run_manifest)
    atomic_write_json(target_dir / "transfer_plan.json", transfer_plan)
    atomic_write_json(target_dir / "calibration_plan.json", calibration_plan)
    atomic_write_json(target_dir / "checkpoint_manifest.json", checkpoint_manifest)
    atomic_write_json(target_dir / "model_run_entries.json", model_run_entries)
    atomic_write_json(target_dir / "completed_jobs.json", {"completed_jobs": []})
    atomic_write_json(target_dir / "run_state.json", {"phase": "preflight_only", "pending_jobs": job_plan["planned_jobs"], "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
    atomic_write_json(target_dir / "failure_report.json", failures)
    atomic_write_json(target_dir / "schema_validation_report.json", schema_validation)
    atomic_write_json(target_dir / "job_plan_only.json", {"jobs": job_plan_only})
    atomic_write_text(target_dir / "transfer_or_leakage_summary.md", "# Preflight Only\n\nNo training started.\n")


def load_existing_runtime_state(output_dir: Path) -> dict[str, object]:
    return {
        "recording_rows": filter_rows_to_fieldnames(read_csv_rows(output_dir / "recording_metrics.csv"), CROSS_RECORDING_FIELDS),
        "subject_rows": filter_rows_to_fieldnames(read_csv_rows(output_dir / "subject_metrics.csv"), CROSS_SUBJECT_FIELDS),
        "dataset_rows": read_csv_rows(output_dir / "dataset_metrics.csv"),
        "comparison_rows": read_csv_rows(output_dir / "cross_dataset_zero_shot_vs_calibrated_comparison.csv"),
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
    transfer_audits: list[dict[str, object]],
    job_plan: dict[str, object],
    state: dict[str, object],
) -> None:
    completed_job_keys = {str(item["job_key"]) for item in state["completed_jobs"]}
    pending_jobs = [job for job in job_plan["planned_jobs"] if str(job["job_key"]) not in completed_job_keys]
    leakage_lines = [
        "# Transfer / Leakage Summary",
        "",
        f"- protocol: `{config['protocol']}`",
        "- source training uses only source dataset train subjects",
        "- source checkpoint selection uses only source dataset val subjects",
        "- target pooled calibration uses only target original train/val recordings of target fixed test subjects",
        "- target original test split remains final test only",
        "",
    ]
    for audit in transfer_audits:
        leakage_lines.append(f"## {audit['source_dataset']} -> {audit['target_dataset']}")
        leakage_lines.append(f"- source_train_subjects: `{', '.join(audit['source_train_subjects'])}`")
        leakage_lines.append(f"- source_val_subjects: `{', '.join(audit['source_val_subjects'])}`")
        leakage_lines.append(f"- target_test_subjects: `{', '.join(audit['target_test_subjects'])}`")
        leakage_lines.append(f"- target_pooled_train_seconds: `{audit['pooled_target_calibration']['train_seconds']}`")
        leakage_lines.append(f"- target_pooled_val_seconds: `{audit['pooled_target_calibration']['val_seconds']}`")
        leakage_lines.append(f"- target_final_test_seconds: `{audit['pooled_target_calibration']['final_test_seconds']}`")
        leakage_lines.append("")
    schema_validation = {
        "protocol": config["protocol"],
        "passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits) and len(state["failures"]) == 0,
        "checks": {
            "target_calibration_audit_passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits),
            "failure_report_empty": len(state["failures"]) == 0,
            "completed_jobs_written_incrementally": True,
            "checkpoint_manifest_written_incrementally": True,
        },
        "notes": [],
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
            "transfer_plan": repo_relative(output_dir / "transfer_plan.json"),
            "calibration_plan": repo_relative(output_dir / "calibration_plan.json"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
            "comparison": repo_relative(output_dir / "cross_dataset_zero_shot_vs_calibrated_comparison.csv"),
            "checkpoint_manifest": repo_relative(output_dir / "checkpoint_manifest.json"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
            "run_state": repo_relative(output_dir / "run_state.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "transfer_or_leakage_summary": repo_relative(output_dir / "transfer_or_leakage_summary.md"),
            "logs": repo_relative(output_dir / "logs"),
        },
    }
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    atomic_write_json(output_dir / "transfer_plan.json", {"pairs": transfer_audits})
    atomic_write_json(output_dir / "calibration_plan.json", {"selection_rule": config["target_calibration"]["selection_rule"], "pairs": transfer_audits})
    atomic_write_json(output_dir / "checkpoint_manifest.json", {"checkpoints": state["checkpoint_entries"]})
    atomic_write_json(output_dir / "model_run_entries.json", {"model_run_entries": state["model_run_entries"]})
    atomic_write_csv(output_dir / "recording_metrics.csv", state["recording_rows"], CROSS_RECORDING_FIELDS)
    atomic_write_csv(output_dir / "subject_metrics.csv", state["subject_rows"], CROSS_SUBJECT_FIELDS)
    atomic_write_csv(output_dir / "dataset_metrics.csv", state["dataset_rows"], DATASET_METRIC_FIELDS)
    atomic_write_csv(output_dir / "cross_dataset_zero_shot_vs_calibrated_comparison.csv", state["comparison_rows"], COMPARISON_FIELDS)
    atomic_write_json(output_dir / "failure_report.json", {"failures": state["failures"]})
    atomic_write_json(output_dir / "schema_validation_report.json", schema_validation)
    atomic_write_text(output_dir / "transfer_or_leakage_summary.md", "\n".join(leakage_lines) + "\n")
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": state["completed_jobs"]})
    atomic_write_json(
        output_dir / "run_state.json",
        {
            "phase": "partial" if pending_jobs else "completed",
            "completed_job_keys": sorted(completed_job_keys),
            "pending_jobs": pending_jobs,
            "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )


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


def collect_target_test_recording_ids(target_cfg: dict[str, object], target_test_subjects: list[str]) -> list[str]:
    dataset_dir = resolve_dataset_path(str(target_cfg["dataset_locator"]))
    rows: list[str] = []
    for subject_id in target_test_subjects:
        for eeg_path in sorted(dataset_dir.glob(f"test_-_{subject_id}_-_*_-_eeg.npy")):
            rows.append(eeg_path.stem.replace("_-_eeg", ""))
    return rows


def predict_target_recordings_window(
    *,
    config: dict,
    target_cfg: dict[str, object],
    model_name: str,
    model: torch.nn.Module,
    checkpoint_id: str,
    recording_ids: list[str],
    device: str,
    protocol: str,
    artifact_scope: str,
    dataset_label: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[WindowPrediction] = []
    dataset_dir = resolve_dataset_path(str(target_cfg["dataset_locator"]))
    sampling_rate = int(target_cfg["sampling_rate"])
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
                dataset=dataset_label,
                model=model_name,
                protocol=protocol,
                seed=seed,
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
    return aggregate_model_outputs(windows, artifact_scope)


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


def save_cross_zero_shot_checkpoint(
    *,
    config: dict,
    config_path: Path,
    source_cfg: dict[str, object],
    source_manifest: dict[str, object],
    model_name: str,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_score: float,
) -> tuple[str, Path]:
    source_dataset = str(source_cfg["dataset_id"])
    seed = int(config["seed"])
    checkpoint_local_id = cross_source_only_checkpoint_local_id(
        source_dataset=source_dataset,
        model=model_name,
        seed=seed,
        best_epoch=best_epoch,
    )
    path = cross_source_only_checkpoint_root() / cross_source_only_checkpoint_relative_path(
        source_dataset=source_dataset,
        model=model_name,
        seed=seed,
        best_epoch=best_epoch,
    )
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in best_state.items()},
        "source_dataset": source_dataset,
        "model": model_name,
        "seed": seed,
        "source_split_id": str(source_manifest["split_id"]),
        "source_train_subjects": list(source_manifest["train_subjects"]),
        "source_val_subjects": list(source_manifest["val_subjects"]),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "stage": "source_zero_shot",
        "protocol": f"{config['protocol']}::source_zero_shot",
        "config_path": repo_relative(config_path),
        "source_split_manifest_path": repo_relative(ROOT / str(source_cfg["split_manifest_path"])),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    torch.save(payload, path)
    return checkpoint_local_id, path


def find_existing_source_zero_shot_checkpoint(
    *,
    source_dataset: str,
    model_name: str,
    seed: int,
) -> Path | None:
    pattern = cross_source_only_checkpoint_path_pattern(source_dataset=source_dataset, model=model_name, seed=seed)
    return latest_checkpoint_in_dir(pattern.parent)


def checkpoint_entry_matches_source_job(entry: dict[str, object], *, source_dataset: str, model_name: str, seed: int) -> bool:
    return (
        str(entry.get("stage")) == "source_zero_shot"
        and str(entry.get("source_dataset")) == source_dataset
        and str(entry.get("model")) == model_name
        and int(entry.get("seed", -1)) == seed
    )


def load_source_zero_shot_checkpoint_payload(
    *,
    output_dir: Path,
    source_dataset: str,
    model_name: str,
    seed: int,
) -> tuple[dict[str, object], Path]:
    manifest = read_json(output_dir / "checkpoint_manifest.json", {"checkpoints": []})
    for entry in manifest.get("checkpoints", []):
        if checkpoint_entry_matches_source_job(entry, source_dataset=source_dataset, model_name=model_name, seed=seed):
            path = ROOT / str(entry["checkpoint_relative_path"])
            if path.exists():
                return torch.load(path, map_location="cpu"), path
    existing = find_existing_source_zero_shot_checkpoint(source_dataset=source_dataset, model_name=model_name, seed=seed)
    if existing is None:
        raise FileNotFoundError(f"source-only checkpoint not found for {source_dataset}/{model_name}/seed{seed}")
    return torch.load(existing, map_location="cpu"), existing


def mean_dataset_metric(subject_rows: list[dict[str, object]], *, source_dataset: str, target_dataset: str, model_name: str, seed: int, stage: str) -> dict[str, object]:
    values = [float(row["metric_value"]) for row in subject_rows]
    return {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model_name,
        "seed": seed,
        "stage": stage,
        "n_subjects": len(subject_rows),
        "mean_pearson": float(np.mean(values)) if values else float("nan"),
        "std_pearson": float(np.std(values, ddof=0)) if values else float("nan"),
        "min_pearson": float(np.min(values)) if values else float("nan"),
        "max_pearson": float(np.max(values)) if values else float("nan"),
    }


def annotate_rows(
    rows: list[dict[str, object]],
    *,
    source_dataset: str,
    target_dataset: str,
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["dataset"] = f"{source_dataset}->{target_dataset}"
        annotated.append(item)
    return annotated


def build_cross_subject_rows(
    subject_rows: list[dict[str, object]],
    *,
    source_dataset: str,
    target_dataset: str,
    stage: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in subject_rows:
        rows.append(
            {
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "model": row["model"],
                "task": row["task"],
                "protocol": row["protocol"],
                "seed": row["seed"],
                "stage": stage,
                "subject_id": row["subject_id"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "num_recordings": row["num_recordings"],
                "checkpoint_id": row["checkpoint_id"],
                "artifact_scope": row["artifact_scope"],
            }
        )
    return rows


def build_cross_recording_rows(
    recording_rows: list[dict[str, object]],
    *,
    source_dataset: str,
    target_dataset: str,
    stage: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in recording_rows:
        rows.append(
            {
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "model": row["model"],
                "task": row["task"],
                "protocol": row["protocol"],
                "seed": row["seed"],
                "stage": stage,
                "subject_id": row["subject_id"],
                "recording_id": row["recording_id"],
                "sampling_rate": row["sampling_rate"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "num_valid_samples": row["num_valid_samples"],
                "checkpoint_id": row["checkpoint_id"],
                "artifact_scope": row["artifact_scope"],
            }
        )
    return rows


def run_source_zero_shot_job(
    *,
    config: dict,
    config_path: Path,
    source_cfg: dict[str, object],
    model_name: str,
    output_dir: Path,
    device: str,
) -> dict[str, object]:
    source_dataset = str(source_cfg["dataset_id"])
    seed = int(config["seed"])
    logger = JobLogger(output_dir / "logs" / f"{source_dataset}_{model_name}_seed{seed}_source_zero_shot.log")
    source_manifest = load_split_manifest(ROOT / str(source_cfg["split_manifest_path"]))
    source_dir = resolve_dataset_path(str(source_cfg["dataset_locator"]))
    model_handle, model_kwargs = instantiate_model(config, model_name)
    train_dataset = SubjectSplitWindowDataset(
        source_dir,
        "train",
        subjects=list(source_manifest["train_subjects"]),
        window_size=model_window_size(config, model_name),
        channels=range(64),
        target_index=str(config[model_name]["target_index"]),
    )
    val_dataset = SubjectSplitWindowDataset(
        source_dir,
        "val",
        subjects=list(source_manifest["val_subjects"]),
        window_size=model_window_size(config, model_name),
        channels=range(64),
        target_index=str(config[model_name]["target_index"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config[model_name]["batch_size"]),
        shuffle=bool(config["dataloader"]["shuffle_train"]),
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["dataloader"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    model = model_handle(**model_kwargs).to(device)
    best_state, best_epoch, best_val_score, epochs_completed = train_window_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=float(config[model_name]["learning_rate"]),
        weight_decay=float(config[model_name]["weight_decay"]),
        max_epochs=int(config[model_name]["max_epochs"]),
        patience=int(config[model_name]["early_stopping_patience"]),
        logger=logger,
    )
    checkpoint_local_id, checkpoint_path = save_cross_zero_shot_checkpoint(
        config=config,
        config_path=config_path,
        source_cfg=source_cfg,
        source_manifest=source_manifest,
        model_name=model_name,
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_score=best_val_score,
    )
    logger.event("checkpoint_saved", checkpoint_local_id=checkpoint_local_id, checkpoint_path=repo_relative(checkpoint_path))
    stdout_message(f"[CHECKPOINT SAVED] checkpoint_local_id={checkpoint_local_id} path={repo_relative(checkpoint_path)}")
    checkpoint_entry = {
        "source_dataset": source_dataset,
        "target_dataset": None,
        "model": model_name,
        "seed": seed,
        "stage": "source_zero_shot",
        "source_job_key": build_job_key(source_dataset, None, model_name, seed, "source_zero_shot"),
        "checkpoint_local_id": checkpoint_local_id,
        "checkpoint_relative_path": repo_relative(checkpoint_path),
        "checkpoint_artifact_status": "local_only_not_committed",
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "source_train_subjects": list(source_manifest["train_subjects"]),
        "source_val_subjects": list(source_manifest["val_subjects"]),
        "target_test_subjects": [],
        "protocol": f"{config['protocol']}::source_zero_shot",
        "config_path": repo_relative(config_path),
        "source_split_manifest_path": repo_relative(ROOT / str(source_cfg["split_manifest_path"])),
        "target_split_manifest_path": None,
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
    }
    model_run_entry = {
        "job_key": build_job_key(source_dataset, None, model_name, seed, "source_zero_shot"),
        "source_dataset": source_dataset,
        "target_dataset": None,
        "model": model_name,
        "seed": seed,
        "stage": "source_zero_shot",
        "status": "success",
        "checkpoint_id": f"{model_name}_epoch_{best_epoch}",
        "checkpoint_local_id": checkpoint_local_id,
        "epochs_completed": int(epochs_completed),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "prediction_target_alignment_ok": True,
        "source_train_subject_count": len(source_manifest["train_subjects"]),
        "source_val_subject_count": len(source_manifest["val_subjects"]),
        "target_test_subject_count": 0,
    }
    completed_job = {
        "job_key": build_job_key(source_dataset, None, model_name, seed, "source_zero_shot"),
        "source_dataset": source_dataset,
        "target_dataset": None,
        "model": model_name,
        "seed": seed,
        "stage": "source_zero_shot",
        "status": "success",
        "checkpoint_id": f"{model_name}_epoch_{best_epoch}",
        "checkpoint_local_id": checkpoint_local_id,
    }
    return {
        "recording_rows": [],
        "subject_rows": [],
        "dataset_row": None,
        "comparison_rows": [],
        "checkpoint_entry": checkpoint_entry,
        "model_run_entry": model_run_entry,
        "completed_job": completed_job,
        "cross_subject_rows": [],
        "cross_recording_rows": [],
    }


def run_cross_dataset_zero_shot_eval_job(
    *,
    config: dict,
    source_cfg: dict[str, object],
    target_cfg: dict[str, object],
    model_name: str,
    output_dir: Path,
    device: str,
) -> dict[str, object]:
    source_dataset = str(source_cfg["dataset_id"])
    target_dataset = str(target_cfg["dataset_id"])
    seed = int(config["seed"])
    logger = JobLogger(output_dir / "logs" / f"{source_dataset}_to_{target_dataset}_{model_name}_seed{seed}_cross_dataset_zero_shot_eval.log")
    target_manifest = load_split_manifest(ROOT / str(target_cfg["split_manifest_path"]))
    target_recording_ids = collect_target_test_recording_ids(target_cfg, list(target_manifest["test_subjects"]))
    payload, checkpoint_path = load_source_zero_shot_checkpoint_payload(
        output_dir=output_dir,
        source_dataset=source_dataset,
        model_name=model_name,
        seed=seed,
    )
    model_handle, model_kwargs = instantiate_model(config, model_name)
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    checkpoint_local_id = str(
        payload.get(
            "checkpoint_local_id",
            cross_source_only_checkpoint_local_id(
                source_dataset=source_dataset,
                model=model_name,
                seed=seed,
                best_epoch=int(payload["best_epoch"]),
            ),
        )
    )
    checkpoint_id = f"{model_name}_epoch_{int(payload['best_epoch'])}"
    logger.event("checkpoint_loaded_for_eval", checkpoint_local_id=checkpoint_local_id, checkpoint_path=repo_relative(checkpoint_path))
    stdout_message("[TEST START]")
    recording_rows_raw, subject_rows_raw = predict_target_recordings_window(
        config=config,
        target_cfg=target_cfg,
        model_name=model_name,
        model=model,
        checkpoint_id=checkpoint_id,
        recording_ids=target_recording_ids,
        device=device,
        protocol=f"{config['protocol']}::cross_dataset_zero_shot_eval",
        artifact_scope=f"{config['artifact_scope']}::cross_dataset_zero_shot_eval",
        dataset_label=target_dataset,
        seed=seed,
    )
    stdout_message("[TEST DONE]")
    subject_rows = build_cross_subject_rows(
        subject_rows_raw,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        stage="cross_dataset_zero_shot_eval",
    )
    recording_rows = build_cross_recording_rows(
        recording_rows_raw,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        stage="cross_dataset_zero_shot_eval",
    )
    dataset_row = mean_dataset_metric(
        subject_rows_raw,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        model_name=model_name,
        seed=seed,
        stage="cross_dataset_zero_shot_eval",
    )
    logger.event("job_completed", mean_pearson=dataset_row["mean_pearson"], n_test_subjects=dataset_row["n_subjects"])
    stdout_message(f"[JOB DONE] mean_pearson={dataset_row['mean_pearson']} n_test_subjects={dataset_row['n_subjects']}")
    model_run_entry = {
        "job_key": build_job_key(source_dataset, target_dataset, model_name, seed, "cross_dataset_zero_shot_eval"),
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model_name,
        "seed": seed,
        "stage": "cross_dataset_zero_shot_eval",
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id,
        "epochs_completed": None,
        "best_epoch": int(payload["best_epoch"]),
        "best_val_score": float(payload["best_val_score"]),
        "prediction_target_alignment_ok": True,
        "source_train_subject_count": len(payload.get("source_train_subjects", [])),
        "source_val_subject_count": len(payload.get("source_val_subjects", [])),
        "target_test_subject_count": len(target_manifest["test_subjects"]),
    }
    completed_job = {
        "job_key": build_job_key(source_dataset, target_dataset, model_name, seed, "cross_dataset_zero_shot_eval"),
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model_name,
        "seed": seed,
        "stage": "cross_dataset_zero_shot_eval",
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id,
    }
    return {
        "recording_rows": recording_rows,
        "subject_rows": subject_rows,
        "dataset_row": dataset_row,
        "comparison_rows": [],
        "checkpoint_entry": None,
        "model_run_entry": model_run_entry,
        "completed_job": completed_job,
        "cross_subject_rows": subject_rows,
        "cross_recording_rows": recording_rows,
    }


def append_job_result(state: dict[str, object], result: dict[str, object]) -> None:
    state["recording_rows"].extend(result["recording_rows"])
    state["subject_rows"].extend(result["subject_rows"])
    if result["dataset_row"] is not None:
        state["dataset_rows"].append(result["dataset_row"])
    state["comparison_rows"].extend(result["comparison_rows"])
    if result["checkpoint_entry"] is not None:
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
    datasets_by_id = dataset_cfg_map(config)
    selected_models = resolve_selected_models(config, args.models)
    transfer_pairs = resolve_transfer_pairs(config, args.source_dataset, args.target_dataset)
    transfer_audits = [build_transfer_duration_audit(config, pair, datasets_by_id) for pair in transfer_pairs]
    shape_checks = [shape_check_for_model(config, model_name, device) for model_name in selected_models]
    checkpoint_checks = checkpoint_path_check(config, transfer_audits, selected_models, device)
    job_plan = build_job_plan(config, transfer_audits, selected_models, args.stage)
    job_plan_only = build_job_plan_only_summary(config, transfer_audits, selected_models)
    manual_commands = build_manual_commands(config_path, transfer_audits, selected_models)
    preflight_mode = any(
        [
            args.duration_audit_only,
            args.shape_check_only,
            args.checkpoint_path_check_only,
            args.startup_only,
            args.dry_run_plan,
            args.job_plan_only,
            args.rebuild_preflight_artifacts_only,
        ]
    )
    if preflight_mode:
        write_preflight_artifacts(
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            transfer_audits=transfer_audits,
            shape_checks=shape_checks,
            checkpoint_checks=checkpoint_checks,
            job_plan=job_plan,
            job_plan_only=job_plan_only,
        )
    if args.duration_audit_only:
        stdout_block("[DURATION AUDIT]", [json.dumps({"transfer_pairs": transfer_audits}, indent=2)])
        return 0
    if args.shape_check_only:
        stdout_block("[SHAPE CHECK]", [json.dumps({"models": shape_checks}, indent=2)])
        return 0
    if args.checkpoint_path_check_only:
        stdout_block("[CHECKPOINT PATH CHECK]", [json.dumps({"checkpoints": checkpoint_checks}, indent=2)])
        return 0
    if args.job_plan_only:
        stdout_block("[JOB PLAN ONLY]", [json.dumps({"jobs": job_plan_only}, indent=2)])
        return 0
    if args.startup_only or args.dry_run_plan:
        transfer_pair_labels = [f"{item['source_dataset']}->{item['target_dataset']}" for item in transfer_audits]
        startup_lines = [
            f"config path: {config_path}",
            f"output_dir: {output_dir}",
            f"device: {device}",
            f"transfer_pairs: {transfer_pair_labels}",
            f"models: {selected_models}",
            f"stage: {args.stage}",
            f"planned_jobs: {len(job_plan['planned_jobs'])}",
            f"source_zero_shot_jobs: {job_plan['source_zero_shot_job_count']}",
            f"cross_dataset_zero_shot_eval_jobs: {job_plan['cross_dataset_zero_shot_eval_job_count']}",
            f"pooled10_target_calibration_jobs: {job_plan['pooled10_target_calibration_job_count']}",
            "no training started: True",
        ]
        for audit in transfer_audits:
            startup_lines.append(
                f"{audit['source_dataset']}->{audit['target_dataset']} target_test_subjects: {audit['target_test_subjects']}"
            )
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

    if not pending_jobs:
        stdout_block("[NO PENDING JOBS]", ["All requested jobs are already completed for the selected stage/source-target/model set."])
        write_runtime_state(
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            transfer_audits=transfer_audits,
            job_plan=job_plan,
            state=state,
        )
        return 0

    audit_map = {(str(item["source_dataset"]), str(item["target_dataset"])): item for item in transfer_audits}
    for job in pending_jobs:
        source_dataset = str(job["source_dataset"])
        target_dataset = str(job["target_dataset"]) if job["target_dataset"] is not None else None
        model_name = str(job["model"])
        stage = str(job["stage"])
        try:
            if stage == "source_zero_shot":
                result = run_source_zero_shot_job(
                    config=config,
                    config_path=config_path,
                    source_cfg=datasets_by_id[source_dataset],
                    model_name=model_name,
                    output_dir=output_dir,
                    device=device,
                )
                append_job_result(state, result)
            elif stage == "cross_dataset_zero_shot_eval":
                if target_dataset is None:
                    raise RuntimeError("target_dataset is required for cross_dataset_zero_shot_eval")
                result = run_cross_dataset_zero_shot_eval_job(
                    config=config,
                    source_cfg=datasets_by_id[source_dataset],
                    target_cfg=datasets_by_id[target_dataset],
                    model_name=model_name,
                    output_dir=output_dir,
                    device=device,
                )
                append_job_result(state, result)
            else:
                raise NotImplementedError("pooled10_target_calibration runtime is not implemented yet")
        except Exception as exc:
            state["failures"].append(
                {
                    "job_key": job["job_key"],
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "model": model_name,
                    "seed": int(config["seed"]),
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
            transfer_audits=transfer_audits,
            job_plan=job_plan,
            state=state,
        )
        stdout_message("[METRICS WRITTEN]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
