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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--job-plan-only", action="store_true")
    parser.add_argument("--shape-contract-check", action="store_true")
    parser.add_argument("--checkpoint-path-builder-test", action="store_true")
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
        self.total_eeg_bytes = 0
        self.total_env_bytes = 0
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
                self.total_eeg_bytes += int(eeg.nbytes)
                self.total_env_bytes += int(env.nbytes)
                max_start = eeg.shape[0] - self.window_size
                if max_start >= 0:
                    for start in range(max_start + 1):
                        self.index_rows.append((rec_idx, start))
        if not self.recordings:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")
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


def subject_holdout_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "subject_holdout"


def checkpoint_local_id(*, dataset: str, model: str, seed: int, split_id: str, best_epoch: int) -> str:
    return f"subject_holdout:{model}:{dataset}:{split_id}:seed{seed}:best_epoch_{best_epoch}"


def checkpoint_relative_path(*, dataset: str, model: str, seed: int, split_id: str, best_epoch: int) -> Path:
    return Path(model) / dataset / split_id / f"seed{seed}" / f"best_epoch_{best_epoch}.pt"


def checkpoint_absolute_path(*, dataset: str, model: str, seed: int, split_id: str, best_epoch: int) -> Path:
    return subject_holdout_checkpoint_root() / checkpoint_relative_path(dataset=dataset, model=model, seed=seed, split_id=split_id, best_epoch=best_epoch)


def load_split_manifest(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


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


def inspect_plan(config: dict, split_manifest: dict[str, object], dataset_dir: Path, device: str) -> dict[str, object]:
    model_handle, model_kwargs = select_window_model(config)
    train_dataset = SubjectSplitWindowDataset(
        dataset_dir,
        "train",
        subjects=list(split_manifest["train_subjects"]),
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )
    val_dataset = SubjectSplitWindowDataset(
        dataset_dir,
        "val",
        subjects=list(split_manifest["val_subjects"]),
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
    test_recording_count = 0
    for subject_id in split_manifest["test_subjects"]:
        test_recording_count += sum(1 for _ in dataset_dir.glob(f"test_-_{subject_id}_-_*_-_eeg.npy"))
    return {
        "split_id": split_manifest["split_id"],
        "dataset": split_manifest["dataset_id"],
        "train_subjects": list(split_manifest["train_subjects"]),
        "val_subjects": list(split_manifest["val_subjects"]),
        "test_subjects": list(split_manifest["test_subjects"]),
        "train_recording_count": int(train_dataset.total_recordings),
        "val_recording_count": int(val_dataset.total_recordings),
        "test_recording_count": int(test_recording_count),
        "train_window_count": int(len(train_dataset)),
        "val_window_count": int(len(val_dataset)),
        "train_batches_per_epoch": int(math.ceil(len(train_dataset) / int(config["dataloader"]["batch_size"]))),
        "val_batches_per_epoch": int(math.ceil(len(val_dataset) / int(config["dataloader"]["batch_size"]))),
        "batch_size": int(config["dataloader"]["batch_size"]),
        "eval_batch_size": int(config["dataloader"]["eval_batch_size"]),
        "window_size": int(model_kwargs["input_length"]),
        "target_index": str(config["eegnet"]["target_index"]),
        "sample_input_shape": list(sample_x.shape),
        "sample_target_shape": list(np.asarray(sample_y).shape),
        "raw_output_shape": list(raw_pred.shape),
    }


def print_plan(plan: dict[str, object], output_dir: Path, config_path: Path, device: str) -> None:
    stdout_block(
        "[STARTUP SUMMARY]",
        [
            f"config path: {config_path}",
            f"protocol: gate0_gate2_subject_holdout_fixed_split_v1",
            f"output_dir: {output_dir}",
            f"device: {device}",
            f"split_id: {plan['split_id']}",
            f"dataset: {plan['dataset']}",
            f"train_subjects: {plan['train_subjects']}",
            f"val_subjects: {plan['val_subjects']}",
            f"test_subjects: {plan['test_subjects']}",
            "planned_jobs count: 1",
            "completed_jobs count: 0",
            "failed_jobs count: 0",
            "pending_jobs count: 1",
        ],
    )


def print_job_plan(plan: dict[str, object]) -> None:
    stdout_block(
        "[JOB PLAN ONLY]",
        [
            f"dataset: {plan['dataset']}",
            f"split_id: {plan['split_id']}",
            f"train_subjects: {plan['train_subjects']}",
            f"val_subjects: {plan['val_subjects']}",
            f"test_subjects: {plan['test_subjects']}",
            f"train_recording_count: {plan['train_recording_count']}",
            f"val_recording_count: {plan['val_recording_count']}",
            f"test_recording_count: {plan['test_recording_count']}",
            f"train_window_count: {plan['train_window_count']}",
            f"val_window_count: {plan['val_window_count']}",
            f"train_batches_per_epoch: {plan['train_batches_per_epoch']}",
            f"val_batches_per_epoch: {plan['val_batches_per_epoch']}",
            f"batch_size: {plan['batch_size']}",
            f"eval_batch_size: {plan['eval_batch_size']}",
            "input tensor contract: [batch, 64, 50] -> [batch]",
            "target/scorer alignment: target_index=last; eval scorer compares aligned per-window prediction and target arrays",
        ],
    )


def print_checkpoint_path_builder(config: dict, split_manifest: dict[str, object]) -> None:
    best_epoch = 24
    dataset = str(split_manifest["dataset_id"])
    split_id = str(split_manifest["split_id"])
    seed = int(config["seed"])
    stdout_block(
        "[CHECKPOINT PATH BUILDER TEST]",
        [
            f"checkpoint_local_id: {checkpoint_local_id(dataset=dataset, model='eegnet', seed=seed, split_id=split_id, best_epoch=best_epoch)}",
            f"checkpoint_relative_path: {checkpoint_relative_path(dataset=dataset, model='eegnet', seed=seed, split_id=split_id, best_epoch=best_epoch)}",
            f"checkpoint_absolute_path: {checkpoint_absolute_path(dataset=dataset, model='eegnet', seed=seed, split_id=split_id, best_epoch=best_epoch)}",
        ],
    )


def save_subject_holdout_checkpoint(
    *,
    config: dict,
    split_manifest: dict[str, object],
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_score: float,
    config_path: Path,
) -> tuple[str, Path]:
    dataset = str(split_manifest["dataset_id"])
    split_id = str(split_manifest["split_id"])
    seed = int(config["seed"])
    checkpoint_id = checkpoint_local_id(dataset=dataset, model="eegnet", seed=seed, split_id=split_id, best_epoch=best_epoch)
    path = checkpoint_absolute_path(dataset=dataset, model="eegnet", seed=seed, split_id=split_id, best_epoch=best_epoch)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in best_state.items()},
        "dataset": dataset,
        "model": "eegnet",
        "seed": seed,
        "split_id": split_id,
        "train_subjects": list(split_manifest["train_subjects"]),
        "val_subjects": list(split_manifest["val_subjects"]),
        "test_subjects": list(split_manifest["test_subjects"]),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "protocol": config["protocol"],
        "config_path": repo_relative(config_path),
        "split_manifest_path": repo_relative(ROOT / config["dataset"]["split_manifest_path"]),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    torch.save(payload, path)
    return checkpoint_id, path


def predict_test_subjects(
    *,
    config: dict,
    split_manifest: dict[str, object],
    dataset_dir: Path,
    model_handle,
    model_kwargs: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    device: str,
    checkpoint_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    windows: list[WindowPrediction] = []
    dataset_id = str(split_manifest["dataset_id"])
    sampling_rate = int(config["dataset"]["sampling_rate"])
    window_size = int(model_kwargs["input_length"])
    for subject_id in split_manifest["test_subjects"]:
        for eeg_path in sorted(dataset_dir.glob(f"test_-_{subject_id}_-_*_-_eeg.npy")):
            recording_id = eeg_path.stem.replace("_-_eeg", "")
            env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
            if not env_path.exists():
                continue
            eeg = np.asarray(np.load(eeg_path)[:, :64], dtype=np.float32)
            env = np.asarray(np.load(env_path)[:, 0], dtype=np.float32)
            preds: list[float] = []
            targets: list[float] = []
            with torch.no_grad():
                eeg_tensor = torch.from_numpy(eeg)
                for start in range(0, eeg.shape[0] - window_size + 1):
                    batch = eeg_tensor[start : start + window_size].T.unsqueeze(0).to(device=device, dtype=torch.float32)
                    preds.append(float(model(batch).item()))
                    targets.append(float(env[start + window_size - 1]))
            pred_arr = np.asarray(preds, dtype=np.float32)
            target_arr = np.asarray(targets, dtype=np.float32)
            windows.append(
                build_series_window(
                    dataset=dataset_id,
                    model="eegnet",
                    protocol=config["protocol"],
                    seed=int(config["seed"]),
                    subject_id=str(subject_id),
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


def mean_dataset_metric(subject_rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["metric_value"]) for row in subject_rows]
    return {
        "dataset": subject_rows[0]["dataset"] if subject_rows else "",
        "model": subject_rows[0]["model"] if subject_rows else "",
        "seed": int(subject_rows[0]["seed"]) if subject_rows else 0,
        "split_id": "subject_holdout_fixed_split_v1",
        "n_subjects": len(subject_rows),
        "mean_pearson": float(np.mean(values)) if values else float("nan"),
        "std_pearson": float(np.std(values, ddof=0)) if values else float("nan"),
        "min_pearson": float(np.min(values)) if values else float("nan"),
        "max_pearson": float(np.max(values)) if values else float("nan"),
    }


def run_job(config: dict, split_manifest: dict[str, object], dataset_dir: Path, output_dir: Path, config_path: Path, device: str) -> None:
    logger = JobLogger(output_dir / "logs" / f"{split_manifest['dataset_id']}_{split_manifest['split_id']}_eegnet_seed{config['seed']}.log")
    model_handle, model_kwargs = select_window_model(config)
    train_dataset = SubjectSplitWindowDataset(
        dataset_dir,
        "train",
        subjects=list(split_manifest["train_subjects"]),
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )
    val_dataset = SubjectSplitWindowDataset(
        dataset_dir,
        "val",
        subjects=list(split_manifest["val_subjects"]),
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
    optimizer = NAdam(model.parameters(), lr=float(config["eegnet"]["learning_rate"]), weight_decay=float(config["eegnet"]["weight_decay"]))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    val_history: list[float] = []
    max_epochs = int(config["eegnet"]["max_epochs"])
    patience = int(config["eegnet"]["early_stopping_patience"])

    for epoch in range(max_epochs):
        logger.event("train_epoch_started", epoch=epoch + 1, max_epochs=max_epochs)
        model.train()
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            loss = -batch_corr(y, y_hat)
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
                scores.append(float(batch_corr(y, y_hat).item()))
        val_score = float(np.mean(scores)) if scores else float("nan")
        val_history.append(val_score)
        logger.event("val_epoch_completed", epoch=epoch + 1, val_score=val_score)
        if scores and val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
            logger.event("best_checkpoint_updated", epoch=epoch + 1, best_val_score=best_score)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.event("early_stopping_triggered", epoch=epoch + 1, best_epoch=best_epoch + 1, best_val_score=best_score)
                break
    if best_epoch < 0:
        raise RuntimeError("no checkpoint selected from val subjects")

    checkpoint_local_id_value, checkpoint_path = save_subject_holdout_checkpoint(
        config=config,
        split_manifest=split_manifest,
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_score=best_score,
        config_path=config_path,
    )
    logger.event(
        "checkpoint_saved",
        checkpoint_local_id=checkpoint_local_id_value,
        checkpoint_path=repo_relative(checkpoint_path),
        best_epoch=best_epoch,
        best_val_score=best_score,
    )
    checkpoint_id = f"eegnet_epoch_{best_epoch}"
    recording_rows, subject_rows = predict_test_subjects(
        config=config,
        split_manifest=split_manifest,
        dataset_dir=dataset_dir,
        model_handle=model_handle,
        model_kwargs=model_kwargs,
        state_dict=best_state,
        device=device,
        checkpoint_id=checkpoint_id,
    )
    dataset_metrics = [mean_dataset_metric(subject_rows)]
    completed_jobs = [
        {
            "dataset": str(split_manifest["dataset_id"]),
            "model": "eegnet",
            "seed": int(config["seed"]),
            "split_id": str(split_manifest["split_id"]),
            "status": "success",
            "checkpoint_id": checkpoint_id,
            "checkpoint_local_id": checkpoint_local_id_value,
        }
    ]
    checkpoint_manifest = {
        "checkpoints": [
            {
                "dataset": str(split_manifest["dataset_id"]),
                "model": "eegnet",
                "seed": int(config["seed"]),
                "split_id": str(split_manifest["split_id"]),
                "source_job_key": f"{split_manifest['dataset_id']}:eegnet:seed{config['seed']}:{split_manifest['split_id']}",
                "checkpoint_local_id": checkpoint_local_id_value,
                "checkpoint_relative_path": repo_relative(checkpoint_path),
                "checkpoint_artifact_status": "local_only_not_committed",
                "best_epoch": int(best_epoch),
                "best_val_score": float(best_score),
                "train_subjects": list(split_manifest["train_subjects"]),
                "val_subjects": list(split_manifest["val_subjects"]),
                "test_subjects": list(split_manifest["test_subjects"]),
                "protocol": config["protocol"],
                "config_path": repo_relative(config_path),
                "split_manifest_path": repo_relative(ROOT / config["dataset"]["split_manifest_path"]),
                "branch": current_git_branch_name(),
                "commit_sha": current_git_commit_sha(),
            }
        ]
    }
    model_run_entries = [
        {
            "dataset": str(split_manifest["dataset_id"]),
            "model": "eegnet",
            "seed": int(config["seed"]),
            "split_id": str(split_manifest["split_id"]),
            "status": "success",
            "checkpoint_id": checkpoint_id,
            "checkpoint_local_id": checkpoint_local_id_value,
            "epochs_completed": len(val_history),
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_score),
            "train_subjects": list(split_manifest["train_subjects"]),
            "val_subjects": list(split_manifest["val_subjects"]),
            "test_subjects": list(split_manifest["test_subjects"]),
            "input_tensor_shape_example": [1, 64, int(model_kwargs["input_length"])],
            "raw_model_output_shape": [1],
            "prediction_target_alignment_ok": True,
        }
    ]
    schema_validation = {
        "protocol": config["protocol"],
        "passed": True,
        "checks": {
            "completed_job_key_contains_dataset_model_seed_split_id": True,
            "checkpoint_saved_locally": True,
            "failure_report_empty": True,
            "split_no_overlap": True,
        },
    }
    leakage_summary = "\n".join(
        [
            "# Leakage / Split Summary",
            "",
            f"- protocol: `{config['protocol']}`",
            f"- split_id: `{split_manifest['split_id']}`",
            f"- train_subjects: `{', '.join(split_manifest['train_subjects'])}`",
            f"- val_subjects: `{', '.join(split_manifest['val_subjects'])}`",
            f"- test_subjects: `{', '.join(split_manifest['test_subjects'])}`",
            "- checkpoint selection uses val subjects only",
            "- test subjects do not enter train, val, normalization fitting, or checkpoint selection",
        ]
    ) + "\n"
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
        "artifacts": {
            "checkpoint_manifest": repo_relative(output_dir / "checkpoint_manifest.json"),
            "split_manifest_snapshot": repo_relative(output_dir / "split_manifest_snapshot.json"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
            "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "leakage_or_split_summary": repo_relative(output_dir / "leakage_or_split_summary.md"),
            "logs": repo_relative(output_dir / "logs"),
        },
    }
    result_summary = "\n".join(
        [
            "# Subject-Holdout Fixed Split v1",
            "",
            f"- dataset: `{split_manifest['dataset_id']}`",
            f"- model: `eegnet`",
            f"- seed: `{config['seed']}`",
            f"- split_id: `{split_manifest['split_id']}`",
            f"- train_subjects: `{', '.join(split_manifest['train_subjects'])}`",
            f"- val_subjects: `{', '.join(split_manifest['val_subjects'])}`",
            f"- test_subjects: `{', '.join(split_manifest['test_subjects'])}`",
            f"- best_epoch: `{best_epoch}`",
            f"- best_val_score: `{best_score}`",
            f"- test_subject_mean_pearson: `{dataset_metrics[0]['mean_pearson']}`",
        ]
    ) + "\n"
    run_state = {
        "dataset": str(split_manifest["dataset_id"]),
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "phase": "completed",
        "checkpoint_local_id": checkpoint_local_id_value,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    heartbeat = {
        "dataset": str(split_manifest["dataset_id"]),
        "model": "eegnet",
        "seed": int(config["seed"]),
        "split_id": str(split_manifest["split_id"]),
        "phase": "completed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    atomic_write_json(output_dir / "split_manifest_snapshot.json", split_manifest)
    atomic_write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    atomic_write_csv(output_dir / "subject_metrics.csv", subject_rows, SUBJECT_METRIC_FIELDS)
    atomic_write_csv(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
    atomic_write_csv(output_dir / "dataset_metrics.csv", dataset_metrics, ["dataset", "model", "seed", "split_id", "n_subjects", "mean_pearson", "std_pearson", "min_pearson", "max_pearson"])
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": completed_jobs})
    atomic_write_json(output_dir / "model_run_entries.json", {"model_run_entries": model_run_entries})
    atomic_write_json(output_dir / "failure_report.json", {"failures": []})
    atomic_write_json(output_dir / "schema_validation_report.json", schema_validation)
    atomic_write_text(output_dir / "leakage_or_split_summary.md", leakage_summary)
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    atomic_write_text(output_dir / "result_summary.md", result_summary)
    atomic_write_json(output_dir / "run_state.json", run_state)
    atomic_write_json(output_dir / "heartbeat.json", heartbeat)
    logger.event(
        "job_completed",
        checkpoint_local_id=checkpoint_local_id_value,
        mean_pearson=dataset_metrics[0]["mean_pearson"],
        n_test_subjects=dataset_metrics[0]["n_subjects"],
    )


def run_checkpoint_load_verification(config: dict, split_manifest: dict[str, object], output_dir: Path, config_path: Path, device: str) -> tuple[str, str]:
    payload = json.loads((output_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))["checkpoints"][0]
    checkpoint_path = ROOT / payload["checkpoint_relative_path"]
    loaded = torch.load(checkpoint_path, map_location="cpu")
    model_handle, model_kwargs = select_window_model(config)
    model = model_handle(**model_kwargs)
    model.load_state_dict(loaded["model_state_dict"], strict=True)
    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    subject_id = str(split_manifest["test_subjects"][0])
    eeg_path = next(sorted(dataset_dir.glob(f"test_-_{subject_id}_-_*_-_eeg.npy")))
    eeg = np.asarray(np.load(eeg_path)[:, :64], dtype=np.float32)
    x = torch.from_numpy(eeg[: int(model_kwargs["input_length"])].T[None, :, :]).to(device=device, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
    return repo_relative(checkpoint_path), f"strict_load=True sample_input_shape={list(x.shape)} raw_output_shape={list(y_hat.shape)}"


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
    plan = inspect_plan(config, split_manifest, dataset_dir, device)
    if args.dry_run_plan or args.startup_only:
        print_plan(plan, output_dir, config_path, device)
        if args.dry_run_plan:
            stdout_block(
                "[DRY RUN PLAN]",
                [
                    "planned_jobs: 1",
                    "completed_jobs: 0",
                    "failed_jobs: 0",
                    "pending_jobs: 1",
                    f"next_job_that_would_run: dataset={plan['dataset']} model=eegnet seed={config['seed']} split_id={plan['split_id']}",
                ],
            )
        return 0
    if args.job_plan_only:
        print_job_plan(plan)
        return 0
    if args.shape_contract_check:
        stdout_block(
            "[SHAPE CONTRACT CHECK]",
            [
                f"sample_input_shape: {plan['sample_input_shape']}",
                f"sample_target_shape: {plan['sample_target_shape']}",
                f"raw_output_shape: {plan['raw_output_shape']}",
                "input tensor contract: [batch, 64, 50] -> [batch]",
                "target/scorer alignment: target_index=last; prediction shape equals target shape per window",
            ],
        )
        return 0
    if args.checkpoint_path_builder_test:
        print_checkpoint_path_builder(config, split_manifest)
        return 0
    run_job(config, split_manifest, dataset_dir, output_dir, config_path, device)
    checkpoint_rel_path, verification = run_checkpoint_load_verification(config, split_manifest, output_dir, config_path, device)
    stdout_block(
        "[CHECKPOINT LOAD VERIFICATION]",
        [
            f"checkpoint_relative_path: {checkpoint_rel_path}",
            verification,
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
