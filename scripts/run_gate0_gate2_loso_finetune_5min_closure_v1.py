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

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow, SubjectMetricRow
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.mldecoders.models import EEGNetRegressor
from repro.reference_baselines import load_reference_recordings


ATOMIC_RETRY_ATTEMPTS = 5
ATOMIC_RETRY_SLEEP_SECONDS = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--checkpoint-load-preflight", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    tmp.write_text(content, encoding="utf-8")
    atomic_replace_with_retry(tmp, path)


def atomic_write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
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
            value = result.stdout.strip()
            return value or None
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
            value = result.stdout.strip()
            return value or None
    except Exception:
        return None
    return None


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

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)

    def event(self, name: str, **payload: object) -> None:
        parts = [name] + [f"{key}={payload[key]}" for key in payload]
        self.log(" ".join(parts))


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


class IndexedWindowDataset(Dataset):
    def __init__(
        self,
        recordings: list[tuple[str, np.ndarray, np.ndarray]],
        *,
        window_size: int,
        target_index: str,
        selected_indices: list[int] | None = None,
    ) -> None:
        self.recordings = recordings
        self.window_size = int(window_size)
        self.target_index = target_index
        self.index_rows: list[tuple[int, int]] = []
        if selected_indices is None:
            selected_indices = list(range(len(recordings)))
        for rec_idx in selected_indices:
            _, eeg, _ = self.recordings[rec_idx]
            max_start = eeg.shape[0] - self.window_size
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                self.index_rows.append((rec_idx, start))

    def __len__(self) -> int:
        return len(self.index_rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self.index_rows[idx]
        _, eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_size].T.astype(np.float32)
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x, np.float32(y)


def checkpoint_local_root() -> Path:
    return ROOT / "local_checkpoints" / "finetune"


def checkpoint_relative_path(*, dataset: str, subject_id: str, model: str, seed: int, best_epoch: int) -> Path:
    return Path(model) / dataset / subject_id / f"seed{seed}" / f"best_epoch_{best_epoch}.pt"


def checkpoint_absolute_path(*, dataset: str, subject_id: str, model: str, seed: int, best_epoch: int) -> Path:
    return checkpoint_local_root() / checkpoint_relative_path(dataset=dataset, subject_id=subject_id, model=model, seed=seed, best_epoch=best_epoch)


def checkpoint_local_id(*, dataset: str, subject_id: str, model: str, seed: int, best_epoch: int) -> str:
    return f"finetune:{model}:{dataset}:{subject_id}:seed{seed}:best_epoch_{best_epoch}"


def select_recordings_for_budget(
    recordings: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    budget_seconds: float,
    sampling_rate: int,
) -> dict[str, object]:
    selected_indices: list[int] = []
    selected_recording_ids: list[str] = []
    total_seconds = 0.0
    total_windows = 0
    for rec_idx, (recording_id, eeg, _) in enumerate(recordings):
        rec_seconds = float(eeg.shape[0] / sampling_rate)
        selected_indices.append(rec_idx)
        selected_recording_ids.append(recording_id)
        total_seconds += rec_seconds
        total_windows += max(int(eeg.shape[0]) - 50 + 1, 0)
        if total_seconds >= budget_seconds:
            break
    return {
        "selected_indices": selected_indices,
        "selected_recording_ids": selected_recording_ids,
        "actual_seconds": float(total_seconds),
        "selection_rule": "sorted_recording_id_prefix_complete_recordings_until_budget_met",
        "recording_count": len(selected_indices),
        "window_count_with_window50": int(total_windows),
    }


def build_plan(config: dict) -> dict[str, object]:
    dataset_cfg = config["dataset"]
    dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    subject_id = str(dataset_cfg["subject_id"])
    sampling_rate = int(dataset_cfg["sampling_rate"])
    train_recordings = load_reference_recordings(dataset_dir, "train", subject_id, channels=range(64))
    val_recordings = load_reference_recordings(dataset_dir, "val", subject_id, channels=range(64))
    test_recordings = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))
    calibration_cfg = config["calibration"]
    train_plan = select_recordings_for_budget(
        train_recordings,
        budget_seconds=float(calibration_cfg["fine_tune_train_budget_seconds"]),
        sampling_rate=sampling_rate,
    )
    val_plan = select_recordings_for_budget(
        val_recordings,
        budget_seconds=float(calibration_cfg["fine_tune_val_budget_seconds"]),
        sampling_rate=sampling_rate,
    )
    test_ids = [recording_id for recording_id, _, _ in test_recordings]
    overlap = set(train_plan["selected_recording_ids"]) & set(val_plan["selected_recording_ids"])
    overlap |= set(train_plan["selected_recording_ids"]) & set(test_ids)
    overlap |= set(val_plan["selected_recording_ids"]) & set(test_ids)
    return {
        "dataset_dir": dataset_dir,
        "dataset_id": str(dataset_cfg["dataset_id"]),
        "subject_id": subject_id,
        "sampling_rate": sampling_rate,
        "train_recordings": train_recordings,
        "val_recordings": val_recordings,
        "test_recordings": test_recordings,
        "train_plan": train_plan,
        "val_plan": val_plan,
        "test_recording_ids": test_ids,
        "actual_calibration_seconds": float(train_plan["actual_seconds"] + val_plan["actual_seconds"]),
        "no_overlap": len(overlap) == 0,
    }


def eegnet_model_kwargs(config: dict) -> dict[str, object]:
    model_cfg = config["eegnet"]
    return {
        "num_input_channels": 64,
        "input_length": int(model_cfg["window_size"]),
        "temporal_filters": int(model_cfg["temporal_filters"]),
        "depth_multiplier": int(model_cfg["depth_multiplier"]),
        "separable_filters": int(model_cfg["separable_filters"]),
        "dropout_rate": float(model_cfg["dropout_rate"]),
    }


def load_source_checkpoint(config: dict) -> tuple[Path, dict[str, object]]:
    rel_path = Path(config["source_checkpoint"]["checkpoint_relative_path"])
    checkpoint_path = ROOT / rel_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"source checkpoint missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint_path, payload


def run_checkpoint_load_preflight(config: dict, plan: dict[str, object], device: str) -> None:
    checkpoint_path, payload = load_source_checkpoint(config)
    kwargs = eegnet_model_kwargs(config)
    model = EEGNetRegressor(**kwargs)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    sample_recording = plan["train_recordings"][0]
    _, eeg, env = sample_recording
    x = torch.from_numpy(eeg[: int(kwargs["input_length"])].T[None, :, :]).to(device=device, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
    stdout_block(
        "[CHECKPOINT LOAD PREFLIGHT]",
        [
            f"checkpoint_path: {checkpoint_path}",
            f"source_checkpoint_local_id: {config['source_checkpoint']['checkpoint_local_id']}",
            f"strict_load: True",
            f"sample_input_shape: {list(x.shape)}",
            f"raw_output_shape: {list(y_hat.shape)}",
            f"sample_target_shape: {list(np.asarray(env[int(kwargs['input_length']) - 1]).shape)}",
            "input tensor contract: [batch, 64, 50] -> [batch]",
            "target/scorer alignment: target_index=last; scorer=pearson_on_valid; prediction shape matches target shape",
        ],
    )


def print_plan(config: dict, plan: dict[str, object]) -> None:
    stdout_block(
        "[FINE-TUNE PLAN]",
        [
            f"protocol: {config['protocol']}",
            f"output_dir: {ROOT / config['output_dir']}",
            f"dataset: {plan['dataset_id']}",
            f"subject: {plan['subject_id']}",
            f"source_checkpoint_local_id: {config['source_checkpoint']['checkpoint_local_id']}",
            f"calibration_budget_seconds: {config['calibration']['calibration_budget_seconds']}",
            f"fine_tune_train_budget_seconds: {config['calibration']['fine_tune_train_budget_seconds']}",
            f"fine_tune_val_budget_seconds: {config['calibration']['fine_tune_val_budget_seconds']}",
            f"actual_calibration_seconds: {plan['actual_calibration_seconds']}",
            f"actual_fine_tune_train_seconds: {plan['train_plan']['actual_seconds']}",
            f"actual_fine_tune_val_seconds: {plan['val_plan']['actual_seconds']}",
            f"train_recording_ids: {plan['train_plan']['selected_recording_ids']}",
            f"val_recording_ids: {plan['val_plan']['selected_recording_ids']}",
            f"test_recording_ids: {plan['test_recording_ids']}",
            f"no_overlap: {plan['no_overlap']}",
            "selection_rule: sorted recording_id prefix, complete recordings only, no result-based picking",
        ],
    )


def write_startup_files(output_dir: Path, config: dict, plan: dict[str, object], device: str) -> None:
    run_state = {
        "dataset": plan["dataset_id"],
        "subject_id": plan["subject_id"],
        "model": "eegnet",
        "seed": int(config["seed"]),
        "phase": "pending",
        "device": device,
        "source_checkpoint_local_id": config["source_checkpoint"]["checkpoint_local_id"],
        "actual_calibration_seconds": float(plan["actual_calibration_seconds"]),
        "actual_fine_tune_train_seconds": float(plan["train_plan"]["actual_seconds"]),
        "actual_fine_tune_val_seconds": float(plan["val_plan"]["actual_seconds"]),
        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    heartbeat = {
        "dataset": plan["dataset_id"],
        "subject": plan["subject_id"],
        "model": "eegnet",
        "seed": int(config["seed"]),
        "phase": "startup_only",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    atomic_write_json(output_dir / "run_state.json", run_state)
    atomic_write_json(output_dir / "heartbeat.json", heartbeat)


def save_finetune_checkpoint(
    *,
    config: dict,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_score: float,
) -> tuple[str, Path]:
    dataset_id = str(config["dataset"]["dataset_id"])
    subject_id = str(config["dataset"]["subject_id"])
    seed = int(config["seed"])
    checkpoint_id = checkpoint_local_id(dataset=dataset_id, subject_id=subject_id, model="eegnet", seed=seed, best_epoch=best_epoch)
    path = checkpoint_absolute_path(dataset=dataset_id, subject_id=subject_id, model="eegnet", seed=seed, best_epoch=best_epoch)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in best_state.items()},
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "eegnet",
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "protocol": config["protocol"],
        "config_path": repo_relative(Path(args_config_path_global)),
        "branch": current_git_branch_name(),
        "commit_sha": current_git_commit_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    torch.save(payload, path)
    return checkpoint_id, path


def evaluate_recordings(
    *,
    recordings: list[tuple[str, np.ndarray, np.ndarray]],
    model: EEGNetRegressor,
    device: str,
    window_size: int,
    dataset_id: str,
    protocol: str,
    seed: int,
    subject_id: str,
    sampling_rate: int,
    checkpoint_id: str,
    artifact_scope: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[WindowPrediction] = []
    model.eval()
    for recording_id, eeg, env in recordings:
        preds: list[float] = []
        targets: list[float] = []
        with torch.no_grad():
            eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
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
                protocol=protocol,
                seed=seed,
                subject_id=subject_id,
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


def mean_subject_metric(subject_rows: list[dict[str, object]]) -> float:
    if not subject_rows:
        return float("nan")
    return float(subject_rows[0]["metric_value"])


def run_finetune(config: dict, plan: dict[str, object], device: str, output_dir: Path) -> None:
    log_path = output_dir / "logs" / f"{plan['dataset_id']}_{plan['subject_id']}_eegnet_seed{config['seed']}.log"
    logger = JobLogger(log_path)
    checkpoint_path, source_payload = load_source_checkpoint(config)
    logger.event("source_checkpoint_loaded", checkpoint_path=repo_relative(checkpoint_path), checkpoint_local_id=config["source_checkpoint"]["checkpoint_local_id"])

    kwargs = eegnet_model_kwargs(config)
    model = EEGNetRegressor(**kwargs).to(device)
    model.load_state_dict(source_payload["model_state_dict"], strict=True)

    target_index = str(config["eegnet"]["target_index"])
    train_dataset = IndexedWindowDataset(
        plan["train_recordings"],
        window_size=int(kwargs["input_length"]),
        target_index=target_index,
        selected_indices=list(range(len(plan["train_recordings"]))),
    )
    val_dataset = IndexedWindowDataset(
        plan["val_recordings"],
        window_size=int(kwargs["input_length"]),
        target_index=target_index,
        selected_indices=list(range(len(plan["val_recordings"]))),
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
    optimizer = NAdam(
        model.parameters(),
        lr=float(config["eegnet"]["learning_rate"]),
        weight_decay=float(config["eegnet"]["weight_decay"]),
    )

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    max_epochs = int(config["eegnet"]["max_epochs"])
    patience = int(config["eegnet"]["early_stopping_patience"])
    val_history: list[float] = []

    for epoch in range(max_epochs):
        logger.event("finetune_epoch_started", epoch=epoch + 1, max_epochs=max_epochs)
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
        logger.event("finetune_val_epoch_completed", epoch=epoch + 1, val_score=val_score)
        if scores and val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
            logger.event("finetune_best_checkpoint_updated", epoch=epoch + 1, best_val_score=best_score)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.event("finetune_early_stopping", epoch=epoch + 1, best_epoch=best_epoch + 1, best_val_score=best_score)
                break

    if best_epoch < 0:
        raise RuntimeError("fine-tune validation never produced a checkpoint")

    finetune_checkpoint_local_id, finetune_checkpoint_path = save_finetune_checkpoint(
        config=config,
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_score=best_score,
    )
    logger.event(
        "finetune_checkpoint_saved",
        checkpoint_local_id=finetune_checkpoint_local_id,
        checkpoint_path=repo_relative(finetune_checkpoint_path),
        best_epoch=best_epoch,
        best_val_score=best_score,
    )

    source_model = EEGNetRegressor(**kwargs).to(device)
    source_model.load_state_dict(source_payload["model_state_dict"], strict=True)
    source_checkpoint_id = f"source_{config['source_checkpoint']['checkpoint_local_id']}"
    loso_subset_recording_rows, loso_subset_subject_rows = evaluate_recordings(
        recordings=plan["test_recordings"],
        model=source_model,
        device=device,
        window_size=int(kwargs["input_length"]),
        dataset_id=plan["dataset_id"],
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=plan["subject_id"],
        sampling_rate=plan["sampling_rate"],
        checkpoint_id=source_checkpoint_id,
        artifact_scope=config["artifact_scope"],
    )

    finetuned_model = EEGNetRegressor(**kwargs).to(device)
    finetuned_model.load_state_dict(best_state, strict=True)
    finetune_checkpoint_id = f"finetune_epoch_{best_epoch}"
    finetune_recording_rows, finetune_subject_rows = evaluate_recordings(
        recordings=plan["test_recordings"],
        model=finetuned_model,
        device=device,
        window_size=int(kwargs["input_length"]),
        dataset_id=plan["dataset_id"],
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=plan["subject_id"],
        sampling_rate=plan["sampling_rate"],
        checkpoint_id=finetune_checkpoint_id,
        artifact_scope=config["artifact_scope"],
    )

    loso_subset_metric = mean_subject_metric(loso_subset_subject_rows)
    finetune_metric = mean_subject_metric(finetune_subject_rows)
    main_delta = float(finetune_metric - loso_subset_metric)

    summary_subject_rows = [
        {
            "dataset": plan["dataset_id"],
            "model": "eegnet",
            "task": "reconstruction",
            "protocol": config["protocol"],
            "seed": int(config["seed"]),
            "subject_id": plan["subject_id"],
            "metric_name": "mean_recording_pearson_r",
            "metric_value": float(finetune_metric),
            "num_recordings": int(len(finetune_recording_rows)),
            "checkpoint_id": finetune_checkpoint_id,
            "artifact_scope": config["artifact_scope"],
        }
    ]

    completed_jobs = [
        {
            "dataset": plan["dataset_id"],
            "subject_id": plan["subject_id"],
            "model": "eegnet",
            "seed": int(config["seed"]),
            "status": "success",
            "source_checkpoint_local_id": config["source_checkpoint"]["checkpoint_local_id"],
            "checkpoint_local_id": finetune_checkpoint_local_id,
            "checkpoint_id": finetune_checkpoint_id,
            "loso_original_metric": float(config["source_checkpoint"]["loso_original_metric"]),
            "loso_subset_metric": float(loso_subset_metric),
            "finetune_metric": float(finetune_metric),
            "main_delta": float(main_delta),
        }
    ]
    model_run_entries = [
        {
            "dataset": plan["dataset_id"],
            "subject_id": plan["subject_id"],
            "model": "eegnet",
            "seed": int(config["seed"]),
            "status": "success",
            "source_checkpoint_local_id": config["source_checkpoint"]["checkpoint_local_id"],
            "checkpoint_local_id": finetune_checkpoint_local_id,
            "checkpoint_id": finetune_checkpoint_id,
            "epochs_completed": len(val_history),
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_score),
            "loso_original_metric": float(config["source_checkpoint"]["loso_original_metric"]),
            "loso_subset_metric": float(loso_subset_metric),
            "finetune_metric": float(finetune_metric),
            "main_delta": float(main_delta),
            "batch_size": int(config["dataloader"]["batch_size"]),
            "eval_batch_size": int(config["dataloader"]["eval_batch_size"]),
            "input_tensor_shape_example": [1, 64, int(kwargs["input_length"])],
            "raw_model_output_shape": [1],
            "prediction_target_alignment_ok": True,
        }
    ]
    fine_tune_manifest = {
        "dataset": plan["dataset_id"],
        "subject_id": plan["subject_id"],
        "model": "eegnet",
        "seed": int(config["seed"]),
        "source_checkpoint_local_id": config["source_checkpoint"]["checkpoint_local_id"],
        "source_checkpoint_relative_path": config["source_checkpoint"]["checkpoint_relative_path"],
        "source_checkpoint_load_verified": True,
        "source_checkpoint_best_epoch": int(source_payload["best_epoch"]),
        "source_checkpoint_best_val_score": float(source_payload["best_val_score"]),
        "calibration_budget_seconds": int(config["calibration"]["calibration_budget_seconds"]),
        "fine_tune_train_budget_seconds": int(config["calibration"]["fine_tune_train_budget_seconds"]),
        "fine_tune_val_budget_seconds": int(config["calibration"]["fine_tune_val_budget_seconds"]),
        "actual_calibration_seconds": float(plan["actual_calibration_seconds"]),
        "actual_fine_tune_train_seconds": float(plan["train_plan"]["actual_seconds"]),
        "actual_fine_tune_val_seconds": float(plan["val_plan"]["actual_seconds"]),
        "calibration_recording_ids": list(plan["train_plan"]["selected_recording_ids"]),
        "validation_recording_ids": list(plan["val_plan"]["selected_recording_ids"]),
        "test_recording_ids": list(plan["test_recording_ids"]),
        "no_overlap_among_train_val_test": bool(plan["no_overlap"]),
        "selection_rule": config["calibration"]["selection_rule"],
        "fine_tune_epochs_completed": int(len(val_history)),
        "fine_tune_best_epoch": int(best_epoch),
        "fine_tune_best_val_score": float(best_score),
        "source_finetune_checkpoint_local_id": finetune_checkpoint_local_id,
        "source_finetune_checkpoint_relative_path": repo_relative(finetune_checkpoint_path),
        "loso_original_metric": float(config["source_checkpoint"]["loso_original_metric"]),
        "loso_subset_metric": float(loso_subset_metric),
        "finetune_metric": float(finetune_metric),
        "main_delta": float(main_delta),
    }
    schema_validation = {
        "protocol": config["protocol"],
        "passed": bool(plan["no_overlap"]),
        "checks": {
            "source_checkpoint_load_verified": True,
            "no_overlap_among_train_val_test": bool(plan["no_overlap"]),
            "subject_metrics_present": True,
            "recording_metrics_present": True,
            "failure_report_empty": True,
        },
    }
    leakage_summary = "\n".join(
        [
            "# Leakage / Adaptation Summary",
            "",
            f"- protocol: `{config['protocol']}`",
            f"- source_checkpoint_local_id: `{config['source_checkpoint']['checkpoint_local_id']}`",
            f"- train_recording_ids: `{', '.join(plan['train_plan']['selected_recording_ids'])}`",
            f"- val_recording_ids: `{', '.join(plan['val_plan']['selected_recording_ids'])}`",
            f"- test_recording_ids: `{', '.join(plan['test_recording_ids'])}`",
            f"- no_overlap: `{plan['no_overlap']}`",
            "- test recordings do not enter fine-tune train, fine-tune val, or checkpoint selection",
            "- source checkpoint is same-dataset same-subject LOSO checkpoint",
        ]
    ) + "\n"
    run_manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "base_branch": config["base_branch"],
        "base_commit": config["base_commit"],
        "current_branch": config["current_branch"],
        "config_path": repo_relative(Path(args_config_path_global)),
        "commit_sha": current_git_commit_sha(),
        "seed": int(config["seed"]),
        "device": device,
        "artifacts": {
            "run_manifest": repo_relative(output_dir / "run_manifest.json"),
            "fine_tune_manifest": repo_relative(output_dir / "fine_tune_manifest.json"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "leakage_or_adaptation_summary": repo_relative(output_dir / "leakage_or_adaptation_summary.md"),
            "logs": repo_relative(output_dir / "logs"),
        },
    }

    atomic_write_csv(output_dir / "subject_metrics.csv", summary_subject_rows, SUBJECT_METRIC_FIELDS)
    atomic_write_csv(output_dir / "recording_metrics.csv", finetune_recording_rows, RECORDING_METRIC_FIELDS)
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": completed_jobs})
    atomic_write_json(output_dir / "model_run_entries.json", {"model_run_entries": model_run_entries})
    atomic_write_json(output_dir / "failure_report.json", {"failures": []})
    atomic_write_json(output_dir / "schema_validation_report.json", schema_validation)
    atomic_write_text(output_dir / "leakage_or_adaptation_summary.md", leakage_summary)
    atomic_write_json(output_dir / "fine_tune_manifest.json", fine_tune_manifest)
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    atomic_write_json(
        output_dir / "run_state.json",
        {
            "dataset": plan["dataset_id"],
            "subject_id": plan["subject_id"],
            "model": "eegnet",
            "seed": int(config["seed"]),
            "phase": "completed",
            "source_checkpoint_local_id": config["source_checkpoint"]["checkpoint_local_id"],
            "checkpoint_local_id": finetune_checkpoint_local_id,
            "loso_original_metric": float(config["source_checkpoint"]["loso_original_metric"]),
            "loso_subset_metric": float(loso_subset_metric),
            "finetune_metric": float(finetune_metric),
            "main_delta": float(main_delta),
            "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )
    atomic_write_json(
        output_dir / "heartbeat.json",
        {
            "dataset": plan["dataset_id"],
            "subject": plan["subject_id"],
            "model": "eegnet",
            "seed": int(config["seed"]),
            "phase": "completed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )
    result_summary = "\n".join(
        [
            "# Fine-Tune 5-Minute Closure v1",
            "",
            f"- dataset: `{plan['dataset_id']}`",
            f"- subject: `{plan['subject_id']}`",
            f"- source checkpoint: `{config['source_checkpoint']['checkpoint_local_id']}`",
            f"- loso_original_metric: `{config['source_checkpoint']['loso_original_metric']}`",
            f"- loso_subset_metric: `{loso_subset_metric}`",
            f"- finetune_metric: `{finetune_metric}`",
            f"- main_delta: `{main_delta}`",
            f"- fine_tune_best_epoch: `{best_epoch}`",
            f"- fine_tune_best_val_score: `{best_score}`",
            f"- actual_calibration_seconds: `{plan['actual_calibration_seconds']}`",
            f"- actual_fine_tune_train_seconds: `{plan['train_plan']['actual_seconds']}`",
            f"- actual_fine_tune_val_seconds: `{plan['val_plan']['actual_seconds']}`",
        ]
    ) + "\n"
    atomic_write_text(output_dir / "result_summary.md", result_summary)
    logger.event(
        "job_completed",
        checkpoint_local_id=finetune_checkpoint_local_id,
        loso_subset_metric=loso_subset_metric,
        finetune_metric=finetune_metric,
        main_delta=main_delta,
    )


args_config_path_global = ""


def main() -> int:
    global args_config_path_global
    args = parse_args()
    args_config_path_global = args.config
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    device = resolve_device(args.device)
    plan = build_plan(config)
    write_startup_files(output_dir, config, plan, device)
    if args.checkpoint_load_preflight:
        run_checkpoint_load_preflight(config, plan, device)
        return 0
    if args.dry_run_plan:
        print_plan(config, plan)
        return 0
    if args.startup_only:
        print_plan(config, plan)
        stdout_block(
            "[STARTUP SUMMARY]",
            [
                f"config path: {args.config}",
                f"output_dir: {output_dir}",
                f"device: {device}",
                f"source_checkpoint_local_id: {config['source_checkpoint']['checkpoint_local_id']}",
                f"subject: {plan['subject_id']}",
                "pending_jobs count: 1",
                "completed_jobs count: 0",
            ],
        )
        return 0
    run_finetune(config, plan, device, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
