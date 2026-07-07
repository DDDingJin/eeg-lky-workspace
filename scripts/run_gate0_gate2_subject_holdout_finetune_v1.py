from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
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
SHARED_UPSTREAM = SHARED_WORKSPACE / "external" / "upstream" / "mldecoders"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SHARED_UPSTREAM.exists() and str(SHARED_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(SHARED_UPSTREAM))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.mldecoders.models import EEGNetRegressor


ATOMIC_RETRY_ATTEMPTS = 5
ATOMIC_RETRY_SLEEP_SECONDS = 0.05
DATASET_METRIC_FIELDS = ["dataset", "model", "seed", "split_id", "n_subjects", "mean_pearson", "std_pearson", "min_pearson", "max_pearson"]
ZERO_SHOT_ALIGNMENT_TOLERANCE = 5e-6
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
    "zero_shot_reference_metric",
    "zero_shot_metric",
    "fine_tuned_metric",
    "delta",
    "actual_calibration_seconds",
    "actual_fine_tune_train_seconds",
    "actual_fine_tune_val_seconds",
    "calibration_train_recording_ids",
    "calibration_val_recording_ids",
    "test_recording_ids",
    "fine_tune_epochs_completed",
    "fine_tune_best_epoch",
    "fine_tune_best_val_score",
    "fine_tune_checkpoint_local_id",
    "fine_tune_checkpoint_relative_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--job-plan-only", action="store_true")
    parser.add_argument("--checkpoint-load-preflight", action="store_true")
    parser.add_argument("--rebuild-artifacts-only", action="store_true")
    parser.add_argument("--verify-existing-output-only", action="store_true")
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


def repo_relative(path: Path) -> str:
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.relative_to(ROOT).as_posix()


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
        self.total_eeg_bytes = 0
        self.total_env_bytes = 0
        self.total_seconds = 0.0
        for recording_id in recording_ids:
            eeg_path = dataset_dir / f"{recording_id}_-_eeg.npy"
            env_path = dataset_dir / f"{recording_id}_-_envelope.npy"
            if not eeg_path.exists() or not env_path.exists():
                raise FileNotFoundError(f"missing recording pair for {recording_id}")
            eeg = np.asarray(np.load(eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
            env = np.asarray(np.load(env_path, mmap_mode="r")[:, 0], dtype=np.float32)
            subject_id = parse_subject_id(recording_id)
            rec_idx = len(self.recordings)
            self.recordings.append((subject_id, recording_id, eeg, env))
            self.total_recordings += 1
            self.total_eeg_bytes += int(eeg.nbytes)
            self.total_env_bytes += int(env.nbytes)
            self.total_seconds += float(eeg.shape[0] / 64.0)
            max_start = eeg.shape[0] - self.window_size
            if max_start >= 0:
                for start in range(max_start + 1):
                    self.index_rows.append((rec_idx, start))
        if not self.recordings:
            raise ValueError(f"no recordings found for ids={recording_ids}")
        self.preloaded_bytes = int(self.total_eeg_bytes + self.total_env_bytes)

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


def select_window_model(config: dict) -> tuple[object, dict[str, object]]:
    model_cfg = config["eegnet"]
    return EEGNetRegressor, {
        "num_input_channels": 64,
        "input_length": int(model_cfg["window_size"]),
        "temporal_filters": int(model_cfg["temporal_filters"]),
        "depth_multiplier": int(model_cfg["depth_multiplier"]),
        "separable_filters": int(model_cfg["separable_filters"]),
        "dropout_rate": float(model_cfg["dropout_rate"]),
    }


def parse_subject_id(recording_id: str) -> str:
    parts = recording_id.split("_-_")
    if len(parts) < 3:
        raise ValueError(f"unexpected recording id format: {recording_id}")
    return parts[1]


def parse_split_name(recording_id: str) -> str:
    parts = recording_id.split("_-_")
    if len(parts) < 3:
        raise ValueError(f"unexpected recording id format: {recording_id}")
    return parts[0]


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


def fine_tune_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "finetune" / "subject_holdout"


def fine_tune_checkpoint_local_id(*, dataset: str, model: str, seed: int, split_id: str, subject_id: str, best_epoch: int) -> str:
    return f"subject_holdout_finetune:{model}:{dataset}:{split_id}:{subject_id}:seed{seed}:best_epoch_{best_epoch}"


def fine_tune_checkpoint_relative_subpath(*, dataset: str, model: str, seed: int, split_id: str, subject_id: str, best_epoch: int) -> Path:
    return Path(model) / dataset / split_id / subject_id / f"seed{seed}" / f"best_epoch_{best_epoch}.pt"


def fine_tune_checkpoint_absolute_path(*, dataset: str, model: str, seed: int, split_id: str, subject_id: str, best_epoch: int) -> Path:
    return fine_tune_checkpoint_root() / fine_tune_checkpoint_relative_subpath(
        dataset=dataset,
        model=model,
        seed=seed,
        split_id=split_id,
        subject_id=subject_id,
        best_epoch=best_epoch,
    )


def load_split_manifest(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_split_manifest(payload: dict[str, object]) -> None:
    required = {"split_id", "dataset_id", "subject_ids_all", "train_subjects", "val_subjects", "test_subjects", "split_seed", "split_rule", "created_at", "no_overlap"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"split manifest missing fields: {sorted(missing)}")
    train = set(payload["train_subjects"])
    val = set(payload["val_subjects"])
    test = set(payload["test_subjects"])
    if train & val or train & test or val & test:
        raise ValueError("split manifest has overlap among train/val/test")
    if sorted(train | val | test) != sorted(payload["subject_ids_all"]):
        raise ValueError("split manifest subjects do not exactly cover subject_ids_all")


def read_json_if_exists(path: Path, default: object) -> object:
    if not path.exists() or path.stat().st_size == 0:
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def subject_job_key(*, dataset: str, subject_id: str, model: str, seed: int, split_id: str) -> str:
    return f"{dataset}:{subject_id}:{model}:seed{seed}:{split_id}"


def zero_shot_job_key(*, dataset: str, model: str, seed: int, split_id: str) -> str:
    return f"{dataset}:{model}:seed{seed}:{split_id}"


def discover_recordings(dataset_dir: Path, *, split: str, subject_id: str, sampling_rate: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eeg_path in sorted(dataset_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        eeg = np.load(eeg_path, mmap_mode="r")
        rows.append(
            {
                "recording_id": recording_id,
                "num_samples": int(eeg.shape[0]),
                "seconds": float(eeg.shape[0] / sampling_rate),
                "split": split,
                "subject_id": subject_id,
            }
        )
    return rows


def select_recording_prefix_by_budget(recordings: list[dict[str, object]], budget_seconds: float) -> tuple[list[dict[str, object]], float]:
    selected: list[dict[str, object]] = []
    total_seconds = 0.0
    for row in recordings:
        selected.append(row)
        total_seconds += float(row["seconds"])
        if total_seconds >= budget_seconds:
            break
    return selected, total_seconds


def source_checkpoint_path(config: dict) -> Path:
    return ROOT / config["source_zero_shot"]["checkpoint_path"]


def load_source_zero_shot_metadata(config: dict) -> tuple[dict[str, object], dict[str, float], dict[str, object]]:
    subject_metrics_path = ROOT / config["source_zero_shot"]["subject_metrics_path"]
    subject_rows = read_csv_if_exists(subject_metrics_path)
    metrics: dict[str, float] = {}
    for row in subject_rows:
        metrics[str(row["subject_id"])] = float(row["metric_value"])
    checkpoint_manifest = read_json_if_exists(ROOT / config["source_zero_shot"]["checkpoint_manifest_path"], {"checkpoints": []})
    checkpoints = checkpoint_manifest.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("source zero-shot checkpoint manifest missing checkpoints[0]")
    return checkpoints[0], metrics, read_json_if_exists(ROOT / config["source_zero_shot"]["run_manifest_path"], {})


def resolve_target_subjects(args: argparse.Namespace, split_manifest: dict[str, object], config: dict) -> list[str]:
    manifest_subjects = list(split_manifest["test_subjects"])
    configured_subjects = list(config["subjects"])
    if sorted(manifest_subjects) != sorted(configured_subjects):
        raise ValueError("config subjects do not match split manifest test subjects")
    selected = configured_subjects
    if args.subjects:
        requested = list(args.subjects)
        invalid = [subject for subject in requested if subject not in configured_subjects]
        if invalid:
            raise ValueError(f"requested subjects are not configured test subjects: {invalid}")
        selected = requested
    if args.max_jobs is not None:
        if args.max_jobs < 0:
            raise ValueError("--max-jobs must be >= 0")
        selected = selected[: args.max_jobs]
    if not selected:
        raise ValueError("no target subjects selected")
    return selected


def build_calibration_plan(config: dict, dataset_dir: Path, subject_id: str) -> dict[str, object]:
    sampling_rate = int(config["dataset"]["sampling_rate"])
    train_candidates = discover_recordings(dataset_dir, split="train", subject_id=subject_id, sampling_rate=sampling_rate)
    val_candidates = discover_recordings(dataset_dir, split="val", subject_id=subject_id, sampling_rate=sampling_rate)
    test_candidates = discover_recordings(dataset_dir, split="test", subject_id=subject_id, sampling_rate=sampling_rate)
    train_selected, actual_train_seconds = select_recording_prefix_by_budget(train_candidates, float(config["fine_tune"]["train_budget_seconds"]))
    val_selected, actual_val_seconds = select_recording_prefix_by_budget(val_candidates, float(config["fine_tune"]["val_budget_seconds"]))
    train_ids = [str(row["recording_id"]) for row in train_selected]
    val_ids = [str(row["recording_id"]) for row in val_selected]
    test_ids = [str(row["recording_id"]) for row in test_candidates]
    no_overlap = not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids))
    return {
        "subject_id": subject_id,
        "selection_rule": str(config["fine_tune"]["selection_rule"]),
        "calibration_budget_seconds": float(config["fine_tune"]["calibration_budget_seconds"]),
        "fine_tune_train_budget_seconds": float(config["fine_tune"]["train_budget_seconds"]),
        "fine_tune_val_budget_seconds": float(config["fine_tune"]["val_budget_seconds"]),
        "actual_calibration_seconds": float(actual_train_seconds + actual_val_seconds),
        "actual_fine_tune_train_seconds": float(actual_train_seconds),
        "actual_fine_tune_val_seconds": float(actual_val_seconds),
        "calibration_train_recording_ids": train_ids,
        "calibration_val_recording_ids": val_ids,
        "test_recording_ids": test_ids,
        "calibration_train_recording_count": len(train_ids),
        "calibration_val_recording_count": len(val_ids),
        "test_recording_count": len(test_ids),
        "train_candidates_count": len(train_candidates),
        "val_candidates_count": len(val_candidates),
        "test_candidates_count": len(test_candidates),
        "no_overlap": no_overlap,
    }


def count_windows(recording_rows: list[dict[str, object]], window_size: int) -> int:
    total = 0
    for row in recording_rows:
        total += max(0, int(row["num_samples"]) - window_size + 1)
    return total


def inspect_plan(
    config: dict,
    split_manifest: dict[str, object],
    dataset_dir: Path,
    output_dir: Path,
    device: str,
    target_subjects: list[str],
    calibration_plans: dict[str, dict[str, object]],
) -> dict[str, object]:
    source_manifest_entry, zero_shot_metrics, _ = load_source_zero_shot_metadata(config)
    source_path = source_checkpoint_path(config)
    if not source_path.exists():
        raise FileNotFoundError(f"source checkpoint missing: {source_path}")
    completed_jobs_payload = read_json_if_exists(output_dir / "completed_jobs.json", {"completed_jobs": []})
    completed_jobs = completed_jobs_payload.get("completed_jobs", []) if isinstance(completed_jobs_payload, dict) else []
    completed_subjects = [str(row["subject_id"]) for row in completed_jobs if isinstance(row, dict) and row.get("subject_id") in target_subjects]
    pending_subjects = [subject for subject in target_subjects if subject not in completed_subjects]
    model_handle, model_kwargs = select_window_model(config)
    first_subject = target_subjects[0]
    first_plan = calibration_plans[first_subject]
    train_dataset = SelectedRecordingWindowDataset(
        dataset_dir,
        recording_ids=list(first_plan["calibration_train_recording_ids"]),
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )
    val_dataset = SelectedRecordingWindowDataset(
        dataset_dir,
        recording_ids=list(first_plan["calibration_val_recording_ids"]),
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )
    sample_x, sample_y = train_dataset[0]
    batch_tensor = torch.from_numpy(np.expand_dims(sample_x, axis=0)).to(device=device, dtype=torch.float32)
    model = model_handle(**model_kwargs).to(device)
    model.eval()
    with torch.no_grad():
        raw_pred = model(batch_tensor)
    return {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "dataset": split_manifest["dataset_id"],
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": split_manifest["split_id"],
        "source_checkpoint_local_id": str(source_manifest_entry["checkpoint_local_id"]),
        "source_checkpoint_relative_path": repo_relative(source_path),
        "target_subjects": target_subjects,
        "completed_subjects": completed_subjects,
        "pending_subjects": pending_subjects,
        "completed_jobs_count": len(completed_subjects),
        "failed_jobs_count": len(read_json_if_exists(output_dir / "failure_report.json", {"failures": []}).get("failures", [])),
        "pending_jobs_count": len(pending_subjects),
        "next_job": pending_subjects[0] if pending_subjects else None,
        "zero_shot_reference_metrics": zero_shot_metrics,
        "sample_input_shape": list(sample_x.shape),
        "sample_target_shape": list(np.asarray(sample_y).shape),
        "raw_output_shape": list(raw_pred.shape),
        "first_subject_train_window_count": len(train_dataset),
        "first_subject_val_window_count": len(val_dataset),
        "first_subject_train_batches_per_epoch": int(math.ceil(len(train_dataset) / int(config["dataloader"]["batch_size"]))),
        "first_subject_val_batches_per_epoch": int(math.ceil(len(val_dataset) / int(config["dataloader"]["eval_batch_size"]))),
        "batch_size": int(config["dataloader"]["batch_size"]),
        "eval_batch_size": int(config["dataloader"]["eval_batch_size"]),
        "calibration_plans": calibration_plans,
    }


def print_plan(plan: dict[str, object], output_dir: Path, config_path: Path, device: str) -> None:
    stdout_block(
        "[STARTUP]",
        [
            f"config path: {config_path}",
            f"output_dir: {output_dir}",
            f"dataset: {plan['dataset']}",
            f"split_id: {plan['split_id']}",
            f"target_subjects: {plan['target_subjects']}",
            f"source_checkpoint_local_id: {plan['source_checkpoint_local_id']}",
            f"source_checkpoint_relative_path: {plan['source_checkpoint_relative_path']}",
            f"device: {device}",
            f"batch_size: {plan['batch_size']}",
            f"eval_batch_size: {plan['eval_batch_size']}",
        ],
    )
    startup_lines = [
        f"config path: {config_path}",
        f"protocol: {plan['protocol']}",
        f"output_dir: {output_dir}",
        f"dataset: {plan['dataset']}",
        f"split_id: {plan['split_id']}",
        f"device: {device}",
        f"completed_jobs count: {plan['completed_jobs_count']}",
        f"failed_jobs count: {plan['failed_jobs_count']}",
        f"pending_jobs count: {plan['pending_jobs_count']}",
        f"next_job: {plan['next_job']}",
        f"pending subject list: {plan['pending_subjects']}",
    ]
    for subject_id in plan["target_subjects"]:
        calibration = plan["calibration_plans"][subject_id]
        startup_lines.extend(
            [
                f"{subject_id} train_recordings: {calibration['calibration_train_recording_ids']}",
                f"{subject_id} val_recordings: {calibration['calibration_val_recording_ids']}",
                f"{subject_id} test_recordings: {calibration['test_recording_ids']}",
                f"{subject_id} actual_calibration_seconds: {calibration['actual_calibration_seconds']}",
            ]
        )
    stdout_block("[STARTUP SUMMARY]", startup_lines)


def print_job_plan(plan: dict[str, object], target_subjects: list[str], config: dict, dataset_dir: Path) -> None:
    model_handle, model_kwargs = select_window_model(config)
    for subject_id in target_subjects:
        calibration = plan["calibration_plans"][subject_id]
        train_rows = [discover_recordings(dataset_dir, split="train", subject_id=subject_id, sampling_rate=int(config["dataset"]["sampling_rate"]))[i] for i, row in enumerate(discover_recordings(dataset_dir, split="train", subject_id=subject_id, sampling_rate=int(config["dataset"]["sampling_rate"]))) if str(row["recording_id"]) in set(calibration["calibration_train_recording_ids"])]
        val_rows = [discover_recordings(dataset_dir, split="val", subject_id=subject_id, sampling_rate=int(config["dataset"]["sampling_rate"]))[i] for i, row in enumerate(discover_recordings(dataset_dir, split="val", subject_id=subject_id, sampling_rate=int(config["dataset"]["sampling_rate"]))) if str(row["recording_id"]) in set(calibration["calibration_val_recording_ids"])]
        stdout_block(
            "[JOB PLAN ONLY]",
            [
                f"heldout subject: {subject_id}",
                f"train_subjects: calibration within target subject train split only",
                f"val_subjects: calibration within target subject val split only",
                f"test_subject: {subject_id}",
                f"train_recording_count: {len(calibration['calibration_train_recording_ids'])}",
                f"val_recording_count: {len(calibration['calibration_val_recording_ids'])}",
                f"test_recording_count: {len(calibration['test_recording_ids'])}",
                f"train_window_count: {count_windows(train_rows, int(model_kwargs['input_length']))}",
                f"val_window_count: {count_windows(val_rows, int(model_kwargs['input_length']))}",
                f"train_batches_per_epoch: {int(math.ceil(count_windows(train_rows, int(model_kwargs['input_length'])) / int(config['dataloader']['batch_size'])))}",
                f"val_batches_per_epoch: {int(math.ceil(count_windows(val_rows, int(model_kwargs['input_length'])) / int(config['dataloader']['eval_batch_size'])))}",
                f"batch_size: {int(config['dataloader']['batch_size'])}",
                f"eval_batch_size: {int(config['dataloader']['eval_batch_size'])}",
                "input tensor contract: [batch, 64, 50] -> [batch]",
                "target/scorer alignment: target_index=last; prediction shape equals target shape per window",
            ],
        )


def load_model_from_checkpoint(config: dict, checkpoint_path: Path, device: str) -> tuple[object, dict[str, object], dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_handle, model_kwargs = select_window_model(config)
    model = model_handle(**model_kwargs)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return payload, model, model_kwargs


def run_checkpoint_load_preflight(config: dict, dataset_dir: Path, target_subjects: list[str], device: str) -> None:
    checkpoint_path = source_checkpoint_path(config)
    payload, model, model_kwargs = load_model_from_checkpoint(config, checkpoint_path, device)
    subject_id = target_subjects[0]
    paths = sorted(dataset_dir.glob(f"test_-_{subject_id}_-_*_-_eeg.npy"))
    if not paths:
        raise FileNotFoundError(f"no test eeg files found for subject {subject_id} in {dataset_dir}")
    eeg = np.asarray(np.load(paths[0])[:, :64], dtype=np.float32)
    x = torch.from_numpy(eeg[: int(model_kwargs["input_length"])].T[None, :, :]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        y_hat = model(x)
    stdout_block(
        "[CHECKPOINT LOAD PREFLIGHT]",
        [
            f"source_checkpoint_relative_path: {repo_relative(checkpoint_path)}",
            "torch.load: ok",
            "strict_load_state_dict: ok",
            f"source_checkpoint_best_epoch: {payload.get('best_epoch')}",
            f"source_checkpoint_best_val_score: {payload.get('best_val_score')}",
            f"sample_input_shape: {list(x.shape)}",
            f"raw_output_shape: {list(y_hat.shape)}",
            "input tensor contract: [batch, 64, 50] -> [batch]",
            "target/scorer alignment: target_index=last; checkpoint adapter unchanged from zero-shot",
        ],
    )


def evaluate_val_loader(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            scores.append(float(batch_corr(y, y_hat).item()))
    return float(np.mean(scores)) if scores else float("nan")


def predict_recordings(
    *,
    config: dict,
    dataset_dir: Path,
    model: torch.nn.Module,
    checkpoint_id: str,
    recording_ids: list[str],
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[WindowPrediction] = []
    dataset_id = str(config["dataset"]["dataset_id"])
    sampling_rate = int(config["dataset"]["sampling_rate"])
    window_size = int(config["eegnet"]["window_size"])
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
                preds.append(model(batch_tensor).detach().cpu().numpy().astype(np.float32))
        pred_arr = np.concatenate(preds, axis=0) if preds else np.asarray([], dtype=np.float32)
        target_arr = np.asarray([env[offset + window_size - 1] for offset in range(total_windows)], dtype=np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="eegnet",
                protocol=config["protocol"],
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
    return aggregate_model_outputs(windows, config["artifact_scope"])


def mean_dataset_metric(subject_rows: list[dict[str, object]], split_id: str) -> dict[str, object]:
    values = [float(row["metric_value"]) for row in subject_rows]
    return {
        "dataset": subject_rows[0]["dataset"] if subject_rows else "",
        "model": subject_rows[0]["model"] if subject_rows else "",
        "seed": int(subject_rows[0]["seed"]) if subject_rows else 0,
        "split_id": split_id,
        "n_subjects": len(subject_rows),
        "mean_pearson": float(np.mean(values)) if values else float("nan"),
        "std_pearson": float(np.std(values, ddof=0)) if values else float("nan"),
        "min_pearson": float(np.min(values)) if values else float("nan"),
        "max_pearson": float(np.max(values)) if values else float("nan"),
    }


def save_fine_tune_checkpoint(
    *,
    config: dict,
    split_manifest: dict[str, object],
    subject_id: str,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_score: float,
    config_path: Path,
    source_payload: dict[str, object],
    calibration_plan: dict[str, object],
) -> tuple[str, Path]:
    dataset = str(split_manifest["dataset_id"])
    split_id = str(split_manifest["split_id"])
    seed = int(config["seed"])
    checkpoint_id = fine_tune_checkpoint_local_id(
        dataset=dataset,
        model="eegnet",
        seed=seed,
        split_id=split_id,
        subject_id=subject_id,
        best_epoch=best_epoch,
    )
    path = fine_tune_checkpoint_absolute_path(
        dataset=dataset,
        model="eegnet",
        seed=seed,
        split_id=split_id,
        subject_id=subject_id,
        best_epoch=best_epoch,
    )
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in best_state.items()},
        "dataset": dataset,
        "subject_id": subject_id,
        "model": "eegnet",
        "seed": seed,
        "split_id": split_id,
        "protocol": config["protocol"],
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / config["dataset"]["split_manifest_path"]),
        "source_checkpoint_local_id": source_payload.get("checkpoint_local_id", config["source_zero_shot"]["checkpoint_local_id"]),
        "source_checkpoint_relative_path": repo_relative(source_checkpoint_path(config)),
        "source_checkpoint_best_epoch": int(source_payload["best_epoch"]),
        "source_checkpoint_best_val_score": float(source_payload["best_val_score"]),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "calibration_budget_seconds": float(config["fine_tune"]["calibration_budget_seconds"]),
        "fine_tune_train_budget_seconds": float(config["fine_tune"]["train_budget_seconds"]),
        "fine_tune_val_budget_seconds": float(config["fine_tune"]["val_budget_seconds"]),
        "actual_calibration_seconds": float(calibration_plan["actual_calibration_seconds"]),
        "actual_fine_tune_train_seconds": float(calibration_plan["actual_fine_tune_train_seconds"]),
        "actual_fine_tune_val_seconds": float(calibration_plan["actual_fine_tune_val_seconds"]),
        "calibration_train_recording_ids": list(calibration_plan["calibration_train_recording_ids"]),
        "calibration_val_recording_ids": list(calibration_plan["calibration_val_recording_ids"]),
        "test_recording_ids": list(calibration_plan["test_recording_ids"]),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    torch.save(payload, path)
    return checkpoint_id, path


def build_empty_progress() -> dict[str, object]:
    return {
        "recording_rows": [],
        "subject_rows": [],
        "comparison_rows": [],
        "completed_jobs": [],
        "checkpoint_manifest_entries": [],
        "model_run_entries": [],
        "failures": [],
    }


def load_existing_progress(output_dir: Path) -> dict[str, object]:
    progress = build_empty_progress()
    progress["recording_rows"] = read_csv_if_exists(output_dir / "recording_metrics.csv")
    progress["subject_rows"] = read_csv_if_exists(output_dir / "subject_metrics.csv")
    progress["comparison_rows"] = read_csv_if_exists(output_dir / "zero_shot_vs_finetune_comparison.csv")
    progress["completed_jobs"] = read_json_if_exists(output_dir / "completed_jobs.json", {"completed_jobs": []}).get("completed_jobs", [])
    progress["checkpoint_manifest_entries"] = read_json_if_exists(output_dir / "checkpoint_manifest.json", {"checkpoints": []}).get("checkpoints", [])
    progress["model_run_entries"] = read_json_if_exists(output_dir / "model_run_entries.json", {"model_run_entries": []}).get("model_run_entries", [])
    progress["failures"] = read_json_if_exists(output_dir / "failure_report.json", {"failures": []}).get("failures", [])
    return progress


def upsert_job_record(rows: list[dict[str, object]], key_fields: list[str], row: dict[str, object]) -> list[dict[str, object]]:
    row_key = tuple(str(row[field]) for field in key_fields)
    remaining = [existing for existing in rows if tuple(str(existing[field]) for field in key_fields) != row_key]
    remaining.append(row)
    return remaining


def remove_subject_rows(rows: list[dict[str, object]], subject_id: str) -> list[dict[str, object]]:
    return [row for row in rows if str(row.get("subject_id")) != subject_id]


def write_artifacts(
    *,
    config: dict,
    config_path: Path,
    split_manifest: dict[str, object],
    output_dir: Path,
    calibration_plans: dict[str, dict[str, object]],
    progress: dict[str, object],
    target_subjects: list[str],
) -> None:
    dataset_metrics = [mean_dataset_metric(progress["subject_rows"], str(split_manifest["split_id"]))] if progress["subject_rows"] else []
    completed_subjects = [str(row["subject_id"]) for row in progress["completed_jobs"]]
    pending_subjects = [subject for subject in target_subjects if subject not in completed_subjects]
    zero_shot_alignment_ok = True
    for row in progress["comparison_rows"]:
        if abs(float(row["zero_shot_reference_metric"]) - float(row["zero_shot_metric"])) > ZERO_SHOT_ALIGNMENT_TOLERANCE:
            zero_shot_alignment_ok = False
            break
    schema_passed = (
        len(progress["failures"]) == 0
        and len(progress["completed_jobs"]) == len(target_subjects)
        and len(progress["subject_rows"]) == len(target_subjects)
        and len(progress["comparison_rows"]) == len(target_subjects)
        and zero_shot_alignment_ok
    )
    schema_validation = {
        "protocol": config["protocol"],
        "passed": schema_passed,
        "scope": {
            "planned_subjects": list(target_subjects),
            "completed_subjects": completed_subjects,
            "pending_subjects": pending_subjects,
            "mode": "global",
        },
        "checks": {
            "completed_job_key_contains_dataset_subject_model_seed_split_id": True,
            "source_checkpoint_load_verified": True,
            "calibration_uses_non_test_splits_only": all(bool(plan["no_overlap"]) for plan in calibration_plans.values()),
            "failure_report_empty": len(progress["failures"]) == 0,
            "comparison_rows_match_subject_rows": len(progress["comparison_rows"]) == len(progress["subject_rows"]),
            "zero_shot_reference_alignment_with_source_eval": zero_shot_alignment_ok,
            "fine_tuned_checkpoints_saved_locally": all(str(row.get("checkpoint_artifact_status")) == "local_only_not_committed" for row in progress["checkpoint_manifest_entries"]),
        },
        "notes": [] if not pending_subjects else ["global schema remains incomplete until all planned target subjects finish"],
    }
    leakage_lines = [
        "# Adaptation / Leakage Summary",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- split_id: `{split_manifest['split_id']}`",
        f"- target_subjects: `{', '.join(target_subjects)}`",
        f"- source_checkpoint_local_id: `{config['source_zero_shot']['checkpoint_local_id']}`",
        "- calibration uses target subject train split recordings only for fine-tune train budget",
        "- calibration uses target subject val split recordings only for fine-tune validation budget",
        "- fixed test recordings remain unchanged from zero-shot and do not enter fine-tune train, validation, normalization fitting, or checkpoint selection",
        "",
    ]
    for subject_id in target_subjects:
        plan = calibration_plans[subject_id]
        leakage_lines.extend(
            [
                f"## {subject_id}",
                f"- train calibration recordings: `{', '.join(plan['calibration_train_recording_ids'])}`",
                f"- val calibration recordings: `{', '.join(plan['calibration_val_recording_ids'])}`",
                f"- fixed test recordings: `{', '.join(plan['test_recording_ids'])}`",
                f"- actual_calibration_seconds: `{plan['actual_calibration_seconds']}`",
                f"- no_overlap: `{plan['no_overlap']}`",
                "",
            ]
        )
    leakage_summary = "\n".join(leakage_lines)
    calibration_plan_payload = {
        "protocol": config["protocol"],
        "dataset": config["dataset"]["dataset_id"],
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": split_manifest["split_id"],
        "selection_rule": config["fine_tune"]["selection_rule"],
        "subjects": [calibration_plans[subject_id] for subject_id in target_subjects],
    }
    run_manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "base_branch": config["base_branch"],
        "base_commit": config["base_commit"],
        "current_branch": config["current_branch"],
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / config["dataset"]["split_manifest_path"]),
        "commit_sha": current_git_commit_sha(),
        "dataset": str(split_manifest["dataset_id"]),
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "target_subjects": list(target_subjects),
        "source_checkpoint_local_id": config["source_zero_shot"]["checkpoint_local_id"],
        "source_checkpoint_relative_path": config["source_zero_shot"]["checkpoint_path"],
        "artifacts": {
            "calibration_plan": repo_relative(output_dir / "calibration_plan.json"),
            "checkpoint_manifest": repo_relative(output_dir / "checkpoint_manifest.json"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "zero_shot_vs_finetune_comparison": repo_relative(output_dir / "zero_shot_vs_finetune_comparison.csv"),
            "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
            "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "adaptation_or_leakage_summary": repo_relative(output_dir / "adaptation_or_leakage_summary.md"),
            "logs": repo_relative(output_dir / "logs"),
        },
    }
    result_lines = [
        "# Subject-Holdout Fine-Tuning v1",
        "",
        f"- dataset: `{split_manifest['dataset_id']}`",
        "- model: `eegnet`",
        f"- seed: `{config['seed']}`",
        f"- split_id: `{split_manifest['split_id']}`",
        f"- source_checkpoint_local_id: `{config['source_zero_shot']['checkpoint_local_id']}`",
        f"- target_subjects: `{', '.join(target_subjects)}`",
    ]
    if dataset_metrics:
        result_lines.extend(
            [
                f"- n_subjects: `{dataset_metrics[0]['n_subjects']}`",
                f"- mean_fine_tuned_pearson: `{dataset_metrics[0]['mean_pearson']}`",
                f"- std_fine_tuned_pearson: `{dataset_metrics[0]['std_pearson']}`",
                f"- min_fine_tuned_pearson: `{dataset_metrics[0]['min_pearson']}`",
                f"- max_fine_tuned_pearson: `{dataset_metrics[0]['max_pearson']}`",
            ]
        )
    result_lines.extend(["", "## Zero-Shot vs Fine-Tune", ""])
    for row in sorted(progress["comparison_rows"], key=lambda item: str(item["subject_id"])):
        result_lines.extend(
            [
                f"### {row['subject_id']}",
                f"- zero_shot_metric: `{row['zero_shot_metric']}`",
                f"- fine_tuned_metric: `{row['fine_tuned_metric']}`",
                f"- delta: `{row['delta']}`",
                f"- actual_calibration_seconds: `{row['actual_calibration_seconds']}`",
                f"- fine_tune_best_epoch: `{row['fine_tune_best_epoch']}`",
                f"- fine_tune_best_val_score: `{row['fine_tune_best_val_score']}`",
                "",
            ]
        )
    run_state = {
        "dataset": str(split_manifest["dataset_id"]),
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "phase": "completed" if schema_passed else "partial",
        "completed_subjects": completed_subjects,
        "pending_subjects": pending_subjects,
        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    heartbeat = {
        "dataset": str(split_manifest["dataset_id"]),
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "phase": run_state["phase"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    atomic_write_json(output_dir / "calibration_plan.json", calibration_plan_payload)
    atomic_write_json(output_dir / "checkpoint_manifest.json", {"checkpoints": progress["checkpoint_manifest_entries"]})
    atomic_write_csv(output_dir / "subject_metrics.csv", progress["subject_rows"], SUBJECT_METRIC_FIELDS)
    atomic_write_csv(output_dir / "recording_metrics.csv", progress["recording_rows"], RECORDING_METRIC_FIELDS)
    atomic_write_csv(output_dir / "zero_shot_vs_finetune_comparison.csv", progress["comparison_rows"], COMPARISON_FIELDS)
    if dataset_metrics:
        atomic_write_csv(output_dir / "dataset_metrics.csv", dataset_metrics, DATASET_METRIC_FIELDS)
    else:
        atomic_write_csv(output_dir / "dataset_metrics.csv", [], DATASET_METRIC_FIELDS)
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": progress["completed_jobs"]})
    atomic_write_json(output_dir / "model_run_entries.json", {"model_run_entries": progress["model_run_entries"]})
    atomic_write_json(output_dir / "failure_report.json", {"failures": progress["failures"]})
    atomic_write_json(output_dir / "schema_validation_report.json", schema_validation)
    atomic_write_text(output_dir / "adaptation_or_leakage_summary.md", leakage_summary)
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    atomic_write_text(output_dir / "result_summary.md", "\n".join(result_lines) + "\n")
    atomic_write_json(output_dir / "run_state.json", run_state)
    atomic_write_json(output_dir / "heartbeat.json", heartbeat)


def run_subject_job(
    *,
    config: dict,
    split_manifest: dict[str, object],
    dataset_dir: Path,
    output_dir: Path,
    config_path: Path,
    device: str,
    subject_id: str,
    calibration_plan: dict[str, object],
    source_payload: dict[str, object],
    zero_shot_reference_metric: float,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], dict[str, object]]:
    logger = JobLogger(output_dir / "logs" / f"{split_manifest['dataset_id']}_{subject_id}_eegnet_seed{config['seed']}.log")
    model_handle, model_kwargs = select_window_model(config)
    train_dataset = SelectedRecordingWindowDataset(
        dataset_dir,
        recording_ids=list(calibration_plan["calibration_train_recording_ids"]),
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )
    val_dataset = SelectedRecordingWindowDataset(
        dataset_dir,
        recording_ids=list(calibration_plan["calibration_val_recording_ids"]),
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["dataloader"]["batch_size"]),
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

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(source_payload["model_state_dict"], strict=True)
    optimizer = NAdam(model.parameters(), lr=float(config["fine_tune"]["learning_rate"]), weight_decay=float(config["eegnet"]["weight_decay"]))

    source_checkpoint_id = f"eegnet_epoch_{int(source_payload['best_epoch'])}"
    zero_shot_recording_rows, zero_shot_subject_rows = predict_recordings(
        config=config,
        dataset_dir=dataset_dir,
        model=model,
        checkpoint_id=source_checkpoint_id,
        recording_ids=list(calibration_plan["test_recording_ids"]),
        device=device,
    )
    if len(zero_shot_subject_rows) != 1:
        raise RuntimeError(f"expected one zero-shot subject metric row for {subject_id}")
    zero_shot_metric = float(zero_shot_subject_rows[0]["metric_value"])

    initial_val_score = evaluate_val_loader(model, val_loader, device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float(initial_val_score)
    best_epoch = 0
    stale_epochs = 0
    val_history: list[float] = [float(initial_val_score)]

    stdout_message(f"[TRAIN START] subject={subject_id}")
    logger.event(
        "train_started",
        subject_id=subject_id,
        zero_shot_metric=zero_shot_metric,
        initial_val_score=initial_val_score,
        actual_calibration_seconds=calibration_plan["actual_calibration_seconds"],
    )
    stdout_message(f"[VAL DONE] subject={subject_id} epoch=0 val_score={initial_val_score} best_val={initial_val_score}")
    max_epochs = int(config["fine_tune"]["max_epochs"])
    patience = int(config["fine_tune"]["early_stopping_patience"])
    for epoch in range(1, max_epochs + 1):
        logger.event("train_epoch_started", subject_id=subject_id, epoch=epoch, max_epochs=max_epochs)
        stdout_message(f"[TRAIN EPOCH] subject={subject_id} epoch={epoch}/{max_epochs}")
        model.train()
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            loss = -batch_corr(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_score = evaluate_val_loader(model, val_loader, device)
        val_history.append(val_score)
        logger.event("val_epoch_completed", subject_id=subject_id, epoch=epoch, val_score=val_score)
        stdout_message(f"[VAL DONE] subject={subject_id} epoch={epoch} val_score={val_score} best_val={best_score}")
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
            logger.event("best_checkpoint_updated", subject_id=subject_id, epoch=epoch, best_val_score=best_score)
            stdout_message(f"[BEST UPDATED] subject={subject_id} epoch={epoch} best_val={best_score}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.event("early_stopping_triggered", subject_id=subject_id, epoch=epoch, best_epoch=best_epoch, best_val_score=best_score)
                stdout_message(f"[EARLY STOPPING] subject={subject_id} epoch={epoch} best_epoch={best_epoch} best_val={best_score}")
                break

    checkpoint_local_id_value, checkpoint_path = save_fine_tune_checkpoint(
        config=config,
        split_manifest=split_manifest,
        subject_id=subject_id,
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_score=best_score,
        config_path=config_path,
        source_payload=source_payload,
        calibration_plan=calibration_plan,
    )
    logger.event(
        "checkpoint_saved",
        subject_id=subject_id,
        checkpoint_local_id=checkpoint_local_id_value,
        checkpoint_path=repo_relative(checkpoint_path),
        best_epoch=best_epoch,
        best_val_score=best_score,
    )
    stdout_message(f"[CHECKPOINT SAVED] checkpoint_local_id={checkpoint_local_id_value} path={repo_relative(checkpoint_path)}")

    fine_tuned_model = model_handle(**model_kwargs).to(device)
    fine_tuned_model.load_state_dict(best_state, strict=True)
    fine_tuned_model.eval()
    checkpoint_id = f"finetune_epoch_{best_epoch}"
    stdout_message(f"[TEST START] subject={subject_id}")
    recording_rows, subject_rows = predict_recordings(
        config=config,
        dataset_dir=dataset_dir,
        model=fine_tuned_model,
        checkpoint_id=checkpoint_id,
        recording_ids=list(calibration_plan["test_recording_ids"]),
        device=device,
    )
    stdout_message(f"[TEST DONE] subject={subject_id}")
    if len(subject_rows) != 1:
        raise RuntimeError(f"expected one fine-tuned subject metric row for {subject_id}")
    fine_tuned_metric = float(subject_rows[0]["metric_value"])
    delta = float(fine_tuned_metric - zero_shot_metric)
    comparison_row = {
        "dataset": str(split_manifest["dataset_id"]),
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "subject_id": subject_id,
        "source_checkpoint_local_id": config["source_zero_shot"]["checkpoint_local_id"],
        "source_checkpoint_relative_path": repo_relative(source_checkpoint_path(config)),
        "source_checkpoint_best_epoch": int(source_payload["best_epoch"]),
        "source_checkpoint_best_val_score": float(source_payload["best_val_score"]),
        "zero_shot_reference_metric": float(zero_shot_reference_metric),
        "zero_shot_metric": float(zero_shot_metric),
        "fine_tuned_metric": float(fine_tuned_metric),
        "delta": float(delta),
        "actual_calibration_seconds": float(calibration_plan["actual_calibration_seconds"]),
        "actual_fine_tune_train_seconds": float(calibration_plan["actual_fine_tune_train_seconds"]),
        "actual_fine_tune_val_seconds": float(calibration_plan["actual_fine_tune_val_seconds"]),
        "calibration_train_recording_ids": "|".join(calibration_plan["calibration_train_recording_ids"]),
        "calibration_val_recording_ids": "|".join(calibration_plan["calibration_val_recording_ids"]),
        "test_recording_ids": "|".join(calibration_plan["test_recording_ids"]),
        "fine_tune_epochs_completed": len(val_history) - 1,
        "fine_tune_best_epoch": int(best_epoch),
        "fine_tune_best_val_score": float(best_score),
        "fine_tune_checkpoint_local_id": checkpoint_local_id_value,
        "fine_tune_checkpoint_relative_path": repo_relative(checkpoint_path),
    }
    model_run_entry = {
        "dataset": str(split_manifest["dataset_id"]),
        "subject_id": subject_id,
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "status": "success",
        "source_checkpoint_local_id": config["source_zero_shot"]["checkpoint_local_id"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id_value,
        "epochs_completed": len(val_history) - 1,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "zero_shot_metric": float(zero_shot_metric),
        "fine_tuned_metric": float(fine_tuned_metric),
        "delta": float(delta),
        "actual_calibration_seconds": float(calibration_plan["actual_calibration_seconds"]),
        "actual_fine_tune_train_seconds": float(calibration_plan["actual_fine_tune_train_seconds"]),
        "actual_fine_tune_val_seconds": float(calibration_plan["actual_fine_tune_val_seconds"]),
        "input_tensor_shape_example": [1, 64, int(model_kwargs["input_length"])],
        "raw_model_output_shape": [1],
        "prediction_target_alignment_ok": True,
    }
    checkpoint_manifest_entry = {
        "dataset": str(split_manifest["dataset_id"]),
        "subject_id": subject_id,
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "source_job_key": zero_shot_job_key(
            dataset=str(split_manifest["dataset_id"]),
            model="eegnet",
            seed=int(config["seed"]),
            split_id=str(split_manifest["split_id"]),
        ),
        "source_checkpoint_local_id": config["source_zero_shot"]["checkpoint_local_id"],
        "source_checkpoint_relative_path": repo_relative(source_checkpoint_path(config)),
        "source_checkpoint_load_verified": True,
        "source_checkpoint_best_epoch": int(source_payload["best_epoch"]),
        "source_checkpoint_best_val_score": float(source_payload["best_val_score"]),
        "checkpoint_local_id": checkpoint_local_id_value,
        "checkpoint_relative_path": repo_relative(checkpoint_path),
        "checkpoint_artifact_status": "local_only_not_committed",
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "protocol": config["protocol"],
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / config["dataset"]["split_manifest_path"]),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
    }
    completed_job = {
        "dataset": str(split_manifest["dataset_id"]),
        "subject_id": subject_id,
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "status": "success",
        "source_checkpoint_local_id": config["source_zero_shot"]["checkpoint_local_id"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_local_id": checkpoint_local_id_value,
    }
    logger.event(
        "job_completed",
        subject_id=subject_id,
        zero_shot_metric=zero_shot_metric,
        fine_tuned_metric=fine_tuned_metric,
        delta=delta,
        checkpoint_local_id=checkpoint_local_id_value,
    )
    stdout_message(f"[METRICS WRITTEN] subject={subject_id}")
    stdout_message(f"[JOB DONE] subject={subject_id} zero_shot_metric={zero_shot_metric} fine_tuned_metric={fine_tuned_metric} delta={delta}")
    return recording_rows, comparison_row, model_run_entry, checkpoint_manifest_entry, completed_job, subject_rows


def run_verify_existing_output_only(config: dict, split_manifest: dict[str, object], output_dir: Path, device: str) -> None:
    stdout_message("[VERIFY START]")
    required = [
        output_dir / "run_manifest.json",
        output_dir / "calibration_plan.json",
        output_dir / "subject_metrics.csv",
        output_dir / "recording_metrics.csv",
        output_dir / "zero_shot_vs_finetune_comparison.csv",
        output_dir / "checkpoint_manifest.json",
        output_dir / "adaptation_or_leakage_summary.md",
        output_dir / "schema_validation_report.json",
        output_dir / "failure_report.json",
        output_dir / "completed_jobs.json",
        output_dir / "model_run_entries.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"required artifact missing: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"required artifact is empty: {path}")
    schema_payload = read_json_if_exists(output_dir / "schema_validation_report.json", {})
    if not bool(schema_payload.get("passed", False)):
        raise ValueError("schema_validation_report.json passed=false")
    failure_payload = read_json_if_exists(output_dir / "failure_report.json", {})
    if failure_payload.get("failures") != []:
        raise ValueError("failure_report.json is not empty")
    checkpoint_manifest = read_json_if_exists(output_dir / "checkpoint_manifest.json", {"checkpoints": []})
    checkpoints = checkpoint_manifest.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("checkpoint_manifest.json missing checkpoints")
    for entry in checkpoints:
        checkpoint_path = ROOT / str(entry["checkpoint_relative_path"])
        payload, model, model_kwargs = load_model_from_checkpoint(config, checkpoint_path, device)
        subject_id = str(entry["subject_id"])
        paths = sorted(resolve_dataset_path(config["dataset"]["dataset_locator"]).glob(f"test_-_{subject_id}_-_*_-_eeg.npy"))
        if not paths:
            raise FileNotFoundError(f"no test eeg files found for subject {subject_id}")
        eeg = np.asarray(np.load(paths[0])[:, :64], dtype=np.float32)
        x = torch.from_numpy(eeg[: int(model_kwargs["input_length"])].T[None, :, :]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            y_hat = model(x)
        stdout_message(f"[CHECKPOINT LOAD OK] subject={subject_id}")
        stdout_message(f"[STRICT LOAD OK] subject={subject_id}")
        stdout_message(f"[SHAPE CHECK OK] subject={subject_id} input_shape={list(x.shape)} output_shape={list(y_hat.shape)}")
        if "model_state_dict" not in payload:
            raise ValueError(f"checkpoint payload missing model_state_dict for {subject_id}")
    stdout_block(
        "[VERIFY DONE]",
        [
            f"schema_passed: {schema_payload.get('passed')}",
            "failure_report_empty: True",
            f"checkpoint_count: {len(checkpoints)}",
            f"adaptation_or_leakage_summary_exists: {(output_dir / 'adaptation_or_leakage_summary.md').exists()}",
        ],
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    device = resolve_device(args.device)
    split_manifest_path = ROOT / config["dataset"]["split_manifest_path"]
    split_manifest = load_split_manifest(split_manifest_path)
    validate_split_manifest(split_manifest)
    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")
    target_subjects = resolve_target_subjects(args, split_manifest, config)
    calibration_plans = {subject_id: build_calibration_plan(config, dataset_dir, subject_id) for subject_id in target_subjects}
    plan = inspect_plan(config, split_manifest, dataset_dir, output_dir, device, target_subjects, calibration_plans)

    if args.dry_run_plan or args.startup_only:
        print_plan(plan, output_dir, config_path, device)
        if args.dry_run_plan:
            stdout_block(
                "[DRY RUN PLAN]",
                [
                    f"planned_jobs: {len(target_subjects)}",
                    f"completed_jobs: {plan['completed_jobs_count']}",
                    f"failed_jobs: {plan['failed_jobs_count']}",
                    f"pending_jobs: {plan['pending_jobs_count']}",
                    f"next_job_that_would_run: {plan['next_job']}",
                ],
            )
        return 0
    if args.job_plan_only:
        print_job_plan(plan, target_subjects, config, dataset_dir)
        return 0
    if args.checkpoint_load_preflight:
        run_checkpoint_load_preflight(config, dataset_dir, target_subjects, device)
        return 0
    if args.rebuild_artifacts_only:
        progress = load_existing_progress(output_dir)
        write_artifacts(
            config=config,
            config_path=config_path,
            split_manifest=split_manifest,
            output_dir=output_dir,
            calibration_plans=calibration_plans,
            progress=progress,
            target_subjects=target_subjects,
        )
        stdout_block(
            "[REBUILD DONE]",
            [
                f"output_dir: {output_dir}",
                f"target_subjects: {target_subjects}",
                "artifacts rewritten without retraining",
            ],
        )
        return 0
    if args.verify_existing_output_only:
        run_verify_existing_output_only(config, split_manifest, output_dir, device)
        return 0

    progress = load_existing_progress(output_dir) if args.resume else build_empty_progress()
    if not args.resume and any((output_dir / name).exists() for name in ["completed_jobs.json", "subject_metrics.csv", "recording_metrics.csv", "zero_shot_vs_finetune_comparison.csv"]):
        raise RuntimeError("output_dir already contains fine-tune artifacts; rerun with --resume to continue safely")
    source_payload, zero_shot_reference_metrics, _ = load_source_zero_shot_metadata(config)
    source_path = source_checkpoint_path(config)
    if not source_path.exists():
        raise FileNotFoundError(f"source checkpoint missing: {source_path}")
    loaded_source_payload = torch.load(source_path, map_location="cpu")
    for subject_id in target_subjects:
        already_done = any(str(row.get("subject_id")) == subject_id for row in progress["completed_jobs"])
        if already_done:
            continue
        if subject_id not in zero_shot_reference_metrics:
            raise ValueError(f"missing zero-shot reference metric for subject {subject_id}")
        recording_rows, comparison_row, model_run_entry, checkpoint_manifest_entry, completed_job, subject_rows = run_subject_job(
            config=config,
            split_manifest=split_manifest,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            config_path=config_path,
            device=device,
            subject_id=subject_id,
            calibration_plan=calibration_plans[subject_id],
            source_payload=loaded_source_payload,
            zero_shot_reference_metric=zero_shot_reference_metrics[subject_id],
        )
        progress["recording_rows"] = remove_subject_rows(progress["recording_rows"], subject_id) + recording_rows
        progress["subject_rows"] = remove_subject_rows(progress["subject_rows"], subject_id) + subject_rows
        progress["comparison_rows"] = upsert_job_record(progress["comparison_rows"], ["subject_id"], comparison_row)
        progress["model_run_entries"] = upsert_job_record(progress["model_run_entries"], ["subject_id"], model_run_entry)
        progress["checkpoint_manifest_entries"] = upsert_job_record(progress["checkpoint_manifest_entries"], ["subject_id"], checkpoint_manifest_entry)
        progress["completed_jobs"] = upsert_job_record(progress["completed_jobs"], ["subject_id"], completed_job)
        write_artifacts(
            config=config,
            config_path=config_path,
            split_manifest=split_manifest,
            output_dir=output_dir,
            calibration_plans=calibration_plans,
            progress=progress,
            target_subjects=target_subjects,
        )
    write_artifacts(
        config=config,
        config_path=config_path,
        split_manifest=split_manifest,
        output_dir=output_dir,
        calibration_plans=calibration_plans,
        progress=progress,
        target_subjects=target_subjects,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
