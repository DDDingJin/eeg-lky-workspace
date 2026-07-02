from __future__ import annotations

import argparse
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

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow, load_dict_rows, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.adt_exact import ADTExactRegressor, pearson_loss, pearson_metric
from repro.mldecoders.cca import fit_cca_reconstruction, score_reconstruction, trim_valid_range
from repro.mldecoders.models import EEGNetRegressor
from repro.reference_baselines import DNNReferenceTrainResult, list_reference_subjects, load_reference_recordings
from repro.simple_models import FCNNBaseline
from run_gate0_gate2_full_subject_single_seed import (
    UpstreamCNN,
    UpstreamFCNN,
    condition_from_recording_id,
    summarize_dataset_metrics,
    summarize_etard_condition_metrics,
    write_generic_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="*")
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


def read_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
    denom = np.sqrt(np.sum(pred0 ** 2) * np.sum(target0 ** 2))
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

    recording_rows = []
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


class PooledReferenceWindowDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        split: str,
        *,
        subjects: list[str],
        window_size: int,
        channels: range,
        target_index: str = "last",
    ) -> None:
        self.window_size = int(window_size)
        self.target_index = target_index
        self.records: list[tuple[int, int]] = []
        self.recordings: list[tuple[np.ndarray, np.ndarray]] = []
        for subject_id in subjects:
            for _, eeg, env in load_reference_recordings(input_dir, split, subject_id, channels=channels):
                rec_idx = len(self.recordings)
                self.recordings.append((eeg.astype(np.float32), env.astype(np.float32)))
                max_start = eeg.shape[0] - self.window_size
                if max_start < 0:
                    continue
                for start in range(max_start + 1):
                    self.records.append((rec_idx, start))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self.records[idx]
        eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)


class PooledSequenceWindowDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        split: str,
        *,
        subjects: list[str],
        window_length: int,
        hop_length: int,
    ) -> None:
        self.records: list[tuple[int, int]] = []
        self.recordings: list[tuple[np.ndarray, np.ndarray]] = []
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        for subject_id in subjects:
            for _, eeg, env in load_reference_recordings(input_dir, split, subject_id, channels=range(64)):
                rec_idx = len(self.recordings)
                self.recordings.append((eeg.astype(np.float32), env[:, None].astype(np.float32)))
                max_start = eeg.shape[0] - self.window_length
                if max_start < 0:
                    continue
                for start in range(0, max_start + 1, self.hop_length):
                    self.records.append((rec_idx, start))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec_idx, start = self.records[idx]
        eeg, env = self.recordings[rec_idx]
        eeg_window = eeg[start : start + self.window_length]
        env_window = env[start : start + self.window_length]
        return torch.from_numpy(eeg_window), torch.from_numpy(env_window)


def train_dnn_loso_pooled(
    dataset_dir: Path,
    *,
    train_subjects: list[str],
    val_subjects: list[str],
    model_handle,
    model_kwargs: dict[str, object],
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    early_stopping_patience: int,
    device: str,
    seed: int,
    target_index: str,
) -> DNNReferenceTrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_handle(**model_kwargs).to(device)
    optimizer = NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = PooledReferenceWindowDataset(
        dataset_dir,
        "train",
        subjects=train_subjects,
        window_size=model.input_length,
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=target_index,
    )
    val_dataset = PooledReferenceWindowDataset(
        dataset_dir,
        "val",
        subjects=val_subjects,
        window_size=model.input_length,
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=target_index,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    def batch_corr(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        y_true0 = y_true - torch.mean(y_true)
        y_pred0 = y_pred - torch.mean(y_pred)
        return torch.sum(y_true0 * y_pred0) / (torch.sqrt(torch.sum(y_true0 ** 2)) * torch.sqrt(torch.sum(y_pred0 ** 2)) + eps)

    best_val = -np.inf
    best_epoch = -1
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    val_history: list[float] = []

    for epoch in range(epochs):
        if best_epoch >= 0 and epoch > best_epoch + early_stopping_patience:
            break
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
        val_score = float(np.mean(scores))
        val_history.append(val_score)
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return DNNReferenceTrainResult(
        best_val_score=float(best_val),
        best_epoch=int(best_epoch),
        epochs_completed=len(val_history),
        val_history=val_history,
        state_dict=best_state,
    )


def train_adt_loso_pooled(
    dataset_dir: Path,
    *,
    train_subjects: list[str],
    val_subjects: list[str],
    seq_len: int,
    hop_length: int,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    min_lr: float,
    device: str,
    seed: int,
) -> tuple[ADTExactRegressor, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = PooledSequenceWindowDataset(
        dataset_dir,
        "train",
        subjects=train_subjects,
        window_length=seq_len,
        hop_length=hop_length,
    )
    val_dataset = PooledSequenceWindowDataset(
        dataset_dir,
        "val",
        subjects=val_subjects,
        window_length=seq_len,
        hop_length=hop_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    model = ADTExactRegressor(seq_len=seq_len).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_metric = float("-inf")
    stale_epochs = 0
    history = {"train_loss": [], "train_pearson_metric": [], "val_loss": [], "val_pearson_metric": [], "lr": []}

    def run_epoch(loader: DataLoader, *, training: bool) -> tuple[float, float]:
        model.train(training)
        total_loss = 0.0
        total_metric = 0.0
        total_count = 0
        for eeg, env in loader:
            eeg = eeg.to(device=device, dtype=torch.float32)
            env = env.to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad()
            pred = model(eeg)
            loss = pearson_loss(env, pred)
            metric = pearson_metric(env, pred)
            if training:
                loss.backward()
                optimizer.step()
            batch_size_local = eeg.shape[0]
            total_loss += float(loss.item()) * batch_size_local
            total_metric += float(metric.item()) * batch_size_local
            total_count += batch_size_local
        return total_loss / max(total_count, 1), total_metric / max(total_count, 1)

    for epoch in range(epochs):
        train_loss, train_metric = run_epoch(train_loader, training=True)
        val_loss, val_metric = run_epoch(val_loader, training=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_pearson_metric"].append(train_metric)
        history["val_loss"].append(val_loss)
        history["val_pearson_metric"].append(val_metric)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_loss < best_val_loss:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_metric = val_metric
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_metric": float(best_val_metric),
        "history": history,
    }


def concatenate_split_for_subjects(
    dataset_dir: Path,
    split: str,
    subjects: list[str],
    *,
    channels: range,
) -> tuple[np.ndarray, np.ndarray]:
    eeg_parts: list[np.ndarray] = []
    env_parts: list[np.ndarray] = []
    for subject_id in subjects:
        for _, eeg, env in load_reference_recordings(dataset_dir, split, subject_id, channels=channels):
            eeg_parts.append(eeg.astype(np.float32))
            env_parts.append(env.astype(np.float32))
    if not eeg_parts:
        raise ValueError(f"no recordings found for split={split} subjects={subjects}")
    return np.concatenate(eeg_parts, axis=0), np.concatenate(env_parts, axis=0)


def accumulate_gram_for_split(
    dataset_dir: Path,
    split: str,
    subjects: list[str],
    *,
    channels: range,
    start_lag: int,
    end_lag: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    total_xtx = None
    total_xty = None
    subject_xtx: dict[str, np.ndarray] = {}
    subject_xty: dict[str, np.ndarray] = {}
    sample_counts: dict[str, int] = {}
    for subject_id in subjects:
        subj_xtx = None
        subj_xty = None
        subj_count = 0
        for _, eeg, env in load_reference_recordings(dataset_dir, split, subject_id, channels=channels):
            x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
            x_block = x_lag.astype(np.float64, copy=False)
            y_block = y_center.astype(np.float64, copy=False)
            if subj_xtx is None:
                subj_xtx = np.zeros((x_block.shape[1], x_block.shape[1]), dtype=np.float64)
                subj_xty = np.zeros(x_block.shape[1], dtype=np.float64)
            subj_xtx += x_block.T @ x_block
            subj_xty += x_block.T @ y_block
            subj_count += int(x_block.shape[0])
        if subj_xtx is None or subj_xty is None:
            raise ValueError(f"no {split} data for subject {subject_id}")
        subject_xtx[subject_id] = subj_xtx
        subject_xty[subject_id] = subj_xty
        sample_counts[subject_id] = subj_count
        if total_xtx is None:
            total_xtx = np.zeros_like(subj_xtx)
            total_xty = np.zeros_like(subj_xty)
        total_xtx += subj_xtx
        total_xty += subj_xty
    if total_xtx is None or total_xty is None:
        raise ValueError(f"no {split} data found for subjects={subjects}")
    return total_xtx, total_xty, subject_xtx, subject_xty, sample_counts


def load_validation_parts(
    dataset_dir: Path,
    subjects: list[str],
    *,
    channels: range,
    start_lag: int,
    end_lag: int,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], dict[str, int]]:
    parts_by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    sample_counts: dict[str, int] = {}
    for subject_id in subjects:
        parts: list[tuple[np.ndarray, np.ndarray]] = []
        subj_count = 0
        for _, eeg, env in load_reference_recordings(dataset_dir, "val", subject_id, channels=channels):
            x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
            x_block = x_lag.astype(np.float64, copy=False)
            y_block = y_center.astype(np.float64, copy=False)
            parts.append((x_block, y_block))
            subj_count += int(x_block.shape[0])
        if not parts:
            raise ValueError(f"no val data for subject {subject_id}")
        parts_by_subject[subject_id] = parts
        sample_counts[subject_id] = subj_count
    return parts_by_subject, sample_counts


def partial_dir_for_model(output_dir: Path, model_name: str) -> Path:
    return output_dir / "_partial" / model_name


def load_partial_state(output_dir: Path, model_name: str) -> dict[str, object]:
    model_dir = partial_dir_for_model(output_dir, model_name)
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return {
            "recording_rows": [],
            "subject_rows": [],
            "model_runs": [],
            "leakage_entries": [],
            "failures": [],
            "completed_pairs": [],
        }
    meta = read_json(meta_path)
    recording_rows = load_dict_rows(model_dir / "recording_metrics.csv") if (model_dir / "recording_metrics.csv").exists() else []
    subject_rows = load_dict_rows(model_dir / "subject_metrics.csv") if (model_dir / "subject_metrics.csv").exists() else []
    return {
        "recording_rows": recording_rows,
        "subject_rows": subject_rows,
        "model_runs": list(meta.get("model_runs", [])),
        "leakage_entries": list(meta.get("leakage_entries", [])),
        "failures": list(meta.get("failures", [])),
        "completed_pairs": [tuple(item) for item in meta.get("completed_pairs", [])],
    }


def save_partial_state(output_dir: Path, model_name: str, state: dict[str, object]) -> None:
    model_dir = partial_dir_for_model(output_dir, model_name)
    ensure_dir(model_dir)
    write_generic_csv(model_dir / "recording_metrics.csv", list(state["recording_rows"]), RECORDING_METRIC_FIELDS)
    write_generic_csv(model_dir / "subject_metrics.csv", list(state["subject_rows"]), SUBJECT_METRIC_FIELDS)
    write_json(
        model_dir / "meta.json",
        {
            "model_runs": state["model_runs"],
            "leakage_entries": state["leakage_entries"],
            "failures": state["failures"],
            "completed_pairs": [list(item) for item in state["completed_pairs"]],
        },
    )


def predeclared_skip_state(
    *,
    config: dict,
    dataset_subjects: dict[str, list[str]],
    model_name: str,
    reason: str,
    status: str,
) -> dict[str, object]:
    model_cfg = config[model_name]
    failures: list[dict[str, object]] = []
    model_runs: list[dict[str, object]] = []
    for dataset_id, subjects in dataset_subjects.items():
        for heldout_subject in subjects:
            failures.append(
                {
                    "dataset": dataset_id,
                    "model": model_name,
                    "subject_id": heldout_subject,
                    "status": status,
                    "error": reason,
                }
            )
            model_runs.append(
                {
                    "dataset": dataset_id,
                    "heldout_subject": heldout_subject,
                    "model": model_name,
                    "status": status,
                    "reason": reason,
                    "implementation_file": model_cfg["implementation_file"],
                    "selection_rule": model_cfg["selection_rule"],
                    "training_protocol": model_cfg["training_protocol"],
                    "model_family": model_cfg["family"],
                }
            )
    return {
        "recording_rows": [],
        "subject_rows": [],
        "model_runs": model_runs,
        "leakage_entries": [],
        "failures": failures,
        "completed_pairs": [],
    }


def load_subject_specific_reference(config: dict) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = load_dict_rows(ROOT / config["single_seed_reference"]["subject_metrics"])
    target_protocol = config["single_seed_reference"]["protocol"]
    target_seed = str(config["single_seed_reference"]["seed"])
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row["protocol"] != target_protocol:
            continue
        if row["seed"] != target_seed:
            continue
        if row["metric_name"] != "mean_recording_pearson_r":
            continue
        lookup[(row["dataset"], row["model"], row["subject_id"])] = row
    return lookup


def load_ridge_loso_reference(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    target_protocol = config["ridge_loso_reference"]["protocol"]
    target_seed = str(config["ridge_loso_reference"]["seed"])
    subject_rows = [
        row
        for row in load_dict_rows(ROOT / config["ridge_loso_reference"]["subject_metrics"])
        if row["protocol"] == target_protocol and row["seed"] == target_seed and row["metric_name"] == "mean_recording_pearson_r"
    ]
    recording_rows = [
        row
        for row in load_dict_rows(ROOT / config["ridge_loso_reference"]["recording_metrics"])
        if row["protocol"] == target_protocol and row["seed"] == target_seed and row["metric_name"] == "pearson_r"
    ]
    return subject_rows, recording_rows


def remap_rows_to_current_protocol(rows: list[dict[str, str]], *, protocol: str, artifact_scope: str) -> list[dict[str, object]]:
    output = []
    for row in rows:
        cloned = dict(row)
        cloned["protocol"] = protocol
        cloned["artifact_scope"] = artifact_scope
        cloned["seed"] = int(cloned["seed"])
        if "sampling_rate" in cloned:
            cloned["sampling_rate"] = int(cloned["sampling_rate"])
        if "metric_value" in cloned:
            cloned["metric_value"] = float(cloned["metric_value"])
        if "num_recordings" in cloned:
            cloned["num_recordings"] = int(cloned["num_recordings"])
        if "num_valid_samples" in cloned:
            cloned["num_valid_samples"] = int(cloned["num_valid_samples"])
        output.append(cloned)
    return output


def write_loso_vs_subject_specific_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "model_family",
        "subject_id",
        "loso_protocol",
        "loso_metric",
        "loso_checkpoint_id",
        "subject_specific_protocol",
        "subject_specific_metric",
        "subject_specific_checkpoint_id",
        "delta_loso_minus_subject_specific",
    ]
    write_generic_csv(path, rows, fieldnames)


def write_loso_vs_subject_specific_comparison_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# LOSO vs Subject-Specific Comparison",
        "",
        "This file compares the current LOSO result against the existing subject-specific seed-0 baseline from `gate0_gate2_full_subject_single_seed_v1`.",
        "Note: `dnn` and `fcnn` remain two implementations within the same `MLP/FCNN family`; they are not described here as separate architecture families.",
        "",
        "| dataset | model | subject | loso_metric | subject_specific_metric | delta_loso_minus_subject_specific |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['subject_id']} | {float(row['loso_metric']):.6f} | "
            f"{float(row['subject_specific_metric']):.6f} | {float(row['delta_loso_minus_subject_specific']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_dataset_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["model"]), []).append(row)
    output = []
    for (dataset, model), group_rows in sorted(grouped.items()):
        loso_vals = np.asarray([float(item["loso_metric"]) for item in group_rows], dtype=np.float64)
        subj_vals = np.asarray([float(item["subject_specific_metric"]) for item in group_rows], dtype=np.float64)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "model_family": group_rows[0]["model_family"],
                "n_subjects": len(group_rows),
                "loso_mean_pearson": float(np.mean(loso_vals)),
                "subject_specific_mean_pearson": float(np.mean(subj_vals)),
                "delta_loso_minus_subject_specific": float(np.mean(loso_vals - subj_vals)),
            }
        )
    write_generic_csv(
        path,
        output,
        ["dataset", "model", "model_family", "n_subjects", "loso_mean_pearson", "subject_specific_mean_pearson", "delta_loso_minus_subject_specific"],
    )


def write_loso_vs_ridge_loso_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    write_generic_csv(
        path,
        rows,
        ["dataset", "subject_id", "current_protocol", "current_metric", "reference_protocol", "reference_metric", "delta_current_minus_reference"],
    )


def fit_loso_ridge_rows_from_reference(config: dict) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    subject_rows_ref, recording_rows_ref = load_ridge_loso_reference(config)
    recording_rows = remap_rows_to_current_protocol(recording_rows_ref, protocol=config["protocol"], artifact_scope=config["artifact_scope"])
    subject_rows = remap_rows_to_current_protocol(subject_rows_ref, protocol=config["protocol"], artifact_scope=config["artifact_scope"])
    model_runs = []
    leakage_entries = []
    for row in subject_rows:
        model_runs.append(
            {
                "dataset": row["dataset"],
                "heldout_subject": row["subject_id"],
                "model": "ridge",
                "status": "reused",
                "checkpoint_id": row["checkpoint_id"],
                "best_alpha": float(str(row["checkpoint_id"]).split("_")[-1]),
                "val_pearson": None,
                "test_pearson_direct": float(row["metric_value"]),
                "subject_metric": float(row["metric_value"]),
                "num_recordings": int(row["num_recordings"]),
                "implementation_file": config["ridge"]["implementation_file"],
                "selection_rule": config["ridge"]["selection_rule"],
                "training_protocol": config["ridge"]["training_protocol"],
                "model_family": config["ridge"]["family"],
            }
        )
        dataset_subjects = list_reference_subjects(resolve_dataset_path(next(item["dataset_locator"] for item in config["datasets"] if item["dataset_id"] == row["dataset"])), split="test")
        leakage_entries.append(
            {
                "dataset": row["dataset"],
                "model": "ridge",
                "model_family": config["ridge"]["family"],
                "implementation_file": config["ridge"]["implementation_file"],
                "selection_rule": config["ridge"]["selection_rule"],
                "training_protocol": config["ridge"]["training_protocol"],
                "heldout_subject": row["subject_id"],
                "train_subjects": [subject for subject in dataset_subjects if subject != row["subject_id"]],
                "val_subjects": [subject for subject in dataset_subjects if subject != row["subject_id"]],
                "test_subjects": [row["subject_id"]],
                "excluded_target_from_train": True,
                "excluded_target_from_val": True,
                "excluded_target_from_selection": True,
                "selected_checkpoint": row["checkpoint_id"],
            }
        )
    return recording_rows, subject_rows, model_runs, leakage_entries


def fit_loso_cca_for_subject(
    *,
    config: dict,
    model_cfg: dict,
    dataset_cfg: dict,
    dataset_dir: Path,
    heldout_subject: str,
    train_subjects: list[str],
    val_subjects: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    x_train, y_train = concatenate_split_for_subjects(dataset_dir, "train", train_subjects, channels=range(64))
    x_val, y_val = concatenate_split_for_subjects(dataset_dir, "val", val_subjects, channels=range(64))
    model, summary = fit_cca_reconstruction(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        start_lag=int(model_cfg["start_lag"]),
        end_lag=int(model_cfg["end_lag"]),
        n_components_grid=[1, 2, 4],
        x_pca_grid=[32, 64, 128],
        y_pca_grid=[8, 16, 32],
        alpha_grid=[0.1, 1.0, 10.0, 100.0],
    )
    windows: list[WindowPrediction] = []
    scores = []
    offset = int(model_cfg["end_lag"]) - 1
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", heldout_subject, channels=range(64)):
        recon = score_reconstruction(model, eeg, env)
        windows.append(
            build_series_window(
                dataset=dataset_cfg["dataset_id"],
                model="cca",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=heldout_subject,
                recording_id=recording_id,
                sampling_rate=int(dataset_cfg["sampling_rate"]),
                checkpoint_id=f"cca_nc_{summary['n_components']}",
                full_length=len(env),
                offset=offset,
                prediction=recon["prediction"],
                target=recon["target"],
            )
        )
        scores.append(float(recon["recon_corr"]))
    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    meta = {
        "dataset": dataset_cfg["dataset_id"],
        "heldout_subject": heldout_subject,
        "model": "cca",
        "status": "success",
        "checkpoint_id": f"cca_nc_{summary['n_components']}",
        "n_components": int(summary["n_components"]),
        "val_recon_corr": float(summary["val_recon_corr"]),
        "test_pearson_direct": float(np.mean(scores)),
        "subject_metric": float(subject_rows[0]["metric_value"]),
        "num_recordings": int(subject_rows[0]["num_recordings"]),
        "implementation_file": model_cfg["implementation_file"],
        "selection_rule": model_cfg["selection_rule"],
        "training_protocol": model_cfg["training_protocol"],
        "model_family": model_cfg["family"],
    }
    leakage_entry = {
        "dataset": dataset_cfg["dataset_id"],
        "model": "cca",
        "model_family": model_cfg["family"],
        "implementation_file": model_cfg["implementation_file"],
        "selection_rule": model_cfg["selection_rule"],
        "training_protocol": model_cfg["training_protocol"],
        "heldout_subject": heldout_subject,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": [heldout_subject],
        "excluded_target_from_train": heldout_subject not in train_subjects,
        "excluded_target_from_val": heldout_subject not in val_subjects,
        "excluded_target_from_selection": heldout_subject not in val_subjects,
        "selected_checkpoint": meta["checkpoint_id"],
    }
    return recording_rows, subject_rows, meta, leakage_entry


def fit_loso_window_model_for_subject(
    *,
    config: dict,
    model_name: str,
    model_cfg: dict,
    dataset_cfg: dict,
    dataset_dir: Path,
    heldout_subject: str,
    train_subjects: list[str],
    val_subjects: list[str],
    model_handle,
    model_kwargs: dict[str, object],
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    if model_handle is None:
        raise RuntimeError(f"{model_name} implementation is unavailable in this worktree and shared workspace")
    train_result = train_dnn_loso_pooled(
        dataset_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        model_handle=model_handle,
        model_kwargs=model_kwargs,
        epochs=int(model_cfg["max_epochs"]),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
        batch_size=int(model_cfg["batch_size"]),
        early_stopping_patience=int(model_cfg["early_stopping_patience"]),
        device=device,
        seed=int(config["seed"]),
        target_index=str(model_cfg["target_index"]),
    )
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(train_result.state_dict)
    model.eval()

    input_length = int(model_kwargs["input_length"])
    offset = input_length - 1
    windows: list[WindowPrediction] = []
    scores = []
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", heldout_subject, channels=range(int(model_kwargs["num_input_channels"]))):
        preds = []
        targets = []
        with torch.no_grad():
            eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
            for start in range(0, eeg.shape[0] - input_length + 1):
                batch = eeg_tensor[start : start + input_length].T.unsqueeze(0).to(device)
                preds.append(float(model(batch).item()))
                targets.append(float(env[start + input_length - 1]))
        pred_arr = np.asarray(preds, dtype=np.float32)
        target_arr = np.asarray(targets, dtype=np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_cfg["dataset_id"],
                model=model_name,
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=heldout_subject,
                recording_id=recording_id,
                sampling_rate=int(dataset_cfg["sampling_rate"]),
                checkpoint_id=f"{model_name}_epoch_{train_result.best_epoch}",
                full_length=len(env),
                offset=offset,
                prediction=pred_arr,
                target=target_arr,
            )
        )
        scores.append(correlation(pred_arr, target_arr))

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    meta = {
        "dataset": dataset_cfg["dataset_id"],
        "heldout_subject": heldout_subject,
        "model": model_name,
        "status": "success",
        "checkpoint_id": f"{model_name}_epoch_{train_result.best_epoch}",
        "best_epoch": int(train_result.best_epoch),
        "best_val_score": float(train_result.best_val_score),
        "epochs_completed": int(train_result.epochs_completed),
        "test_pearson_direct": float(np.mean(scores)),
        "subject_metric": float(subject_rows[0]["metric_value"]),
        "num_recordings": int(subject_rows[0]["num_recordings"]),
        "implementation_file": model_cfg["implementation_file"],
        "selection_rule": model_cfg["selection_rule"],
        "training_protocol": model_cfg["training_protocol"],
        "model_family": model_cfg["family"],
    }
    leakage_entry = {
        "dataset": dataset_cfg["dataset_id"],
        "model": model_name,
        "model_family": model_cfg["family"],
        "implementation_file": model_cfg["implementation_file"],
        "selection_rule": model_cfg["selection_rule"],
        "training_protocol": model_cfg["training_protocol"],
        "heldout_subject": heldout_subject,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": [heldout_subject],
        "excluded_target_from_train": heldout_subject not in train_subjects,
        "excluded_target_from_val": heldout_subject not in val_subjects,
        "excluded_target_from_selection": heldout_subject not in val_subjects,
        "selected_checkpoint": meta["checkpoint_id"],
    }
    return recording_rows, subject_rows, meta, leakage_entry


def fit_loso_adt_for_subject(
    *,
    config: dict,
    model_cfg: dict,
    dataset_cfg: dict,
    dataset_dir: Path,
    heldout_subject: str,
    train_subjects: list[str],
    val_subjects: list[str],
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    model, summary = train_adt_loso_pooled(
        dataset_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        seq_len=int(model_cfg["window_length"]),
        hop_length=int(model_cfg["hop_length"]),
        batch_size=int(model_cfg["batch_size"]),
        epochs=int(model_cfg["max_epochs"]),
        patience=int(model_cfg["early_stopping_patience"]),
        learning_rate=float(model_cfg["learning_rate"]),
        min_lr=float(model_cfg["min_lr"]),
        device=device,
        seed=int(config["seed"]),
    )
    checkpoint_id = f"adt_epoch_{summary['best_epoch']}"
    windows: list[WindowPrediction] = []
    scores = []
    with torch.no_grad():
        model.eval()
        for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", heldout_subject, channels=range(64)):
            recording_windows: list[WindowPrediction] = []
            eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
            max_start = eeg.shape[0] - int(model_cfg["window_length"])
            for start in range(0, max_start + 1, int(model_cfg["hop_length"])):
                batch = eeg_tensor[start : start + int(model_cfg["window_length"])].unsqueeze(0).to(device)
                pred = model(batch).squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
                target = env[start : start + int(model_cfg["window_length"])].astype(np.float32)
                window = WindowPrediction(
                    dataset=dataset_cfg["dataset_id"],
                    model="adt",
                    task="reconstruction",
                    protocol=config["protocol"],
                    seed=int(config["seed"]),
                    subject_id=heldout_subject,
                    recording_id=recording_id,
                    sampling_rate=int(dataset_cfg["sampling_rate"]),
                    checkpoint_id=checkpoint_id,
                    recording_length=int(len(env)),
                    start_index=int(start),
                    prediction=pred,
                    target=target,
                )
                windows.append(window)
                recording_windows.append(window)
            prediction, target, valid_mask = aggregate_overlapping_windows(len(env), recording_windows)
            valid_idx = valid_mask.astype(bool)
            if np.any(valid_idx):
                scores.append(correlation(prediction[valid_idx], target[valid_idx]))

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    meta = {
        "dataset": dataset_cfg["dataset_id"],
        "heldout_subject": heldout_subject,
        "model": "adt",
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "best_epoch": int(summary["best_epoch"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "best_val_metric": float(summary["best_val_metric"]),
        "epochs_completed": len(summary["history"]["val_pearson_metric"]),
        "test_pearson_direct": float(np.mean(scores)),
        "subject_metric": float(subject_rows[0]["metric_value"]),
        "num_recordings": int(subject_rows[0]["num_recordings"]),
        "implementation_file": model_cfg["implementation_file"],
        "selection_rule": model_cfg["selection_rule"],
        "training_protocol": model_cfg["training_protocol"],
        "model_family": model_cfg["family"],
    }
    leakage_entry = {
        "dataset": dataset_cfg["dataset_id"],
        "model": "adt",
        "model_family": model_cfg["family"],
        "implementation_file": model_cfg["implementation_file"],
        "selection_rule": model_cfg["selection_rule"],
        "training_protocol": model_cfg["training_protocol"],
        "heldout_subject": heldout_subject,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": [heldout_subject],
        "excluded_target_from_train": heldout_subject not in train_subjects,
        "excluded_target_from_val": heldout_subject not in val_subjects,
        "excluded_target_from_selection": heldout_subject not in val_subjects,
        "selected_checkpoint": checkpoint_id,
    }
    return recording_rows, subject_rows, meta, leakage_entry


def fit_subject_model(
    *,
    config: dict,
    model_name: str,
    dataset_cfg: dict,
    dataset_dir: Path,
    heldout_subject: str,
    all_subjects: list[str],
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    model_cfg = config[model_name]
    train_subjects = [subject for subject in all_subjects if subject != heldout_subject]
    val_subjects = train_subjects
    if model_name == "cca":
        return fit_loso_cca_for_subject(
            config=config,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            dataset_dir=dataset_dir,
            heldout_subject=heldout_subject,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
        )
    if model_name == "fcnn":
        return fit_loso_window_model_for_subject(
            config=config,
            model_name="fcnn",
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            dataset_dir=dataset_dir,
            heldout_subject=heldout_subject,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            model_handle=FCNNBaseline,
            model_kwargs={
                "num_hidden": int(model_cfg["hidden_layers"]),
                "dropout_rate": float(model_cfg["dropout_rate"]),
                "input_length": int(model_cfg["window_size"]),
                "num_input_channels": 64,
            },
            device=device,
        )
    if model_name == "dnn":
        return fit_loso_window_model_for_subject(
            config=config,
            model_name="dnn",
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            dataset_dir=dataset_dir,
            heldout_subject=heldout_subject,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            model_handle=UpstreamFCNN,
            model_kwargs={
                "num_hidden": int(model_cfg["hidden_layers"]),
                "dropout_rate": float(model_cfg["dropout_rate"]),
                "input_length": int(model_cfg["window_size"]),
                "num_input_channels": 64,
            },
            device=device,
        )
    if model_name == "cnn":
        return fit_loso_window_model_for_subject(
            config=config,
            model_name="cnn",
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            dataset_dir=dataset_dir,
            heldout_subject=heldout_subject,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            model_handle=UpstreamCNN,
            model_kwargs={
                "F1": int(model_cfg["F1"]),
                "D": int(model_cfg["D"]),
                "F2": int(model_cfg["F2"]),
                "dropout_rate": float(model_cfg["dropout_rate"]),
                "input_length": int(model_cfg["window_size"]),
                "num_input_channels": 64,
            },
            device=device,
        )
    if model_name == "eegnet":
        return fit_loso_window_model_for_subject(
            config=config,
            model_name="eegnet",
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            dataset_dir=dataset_dir,
            heldout_subject=heldout_subject,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            model_handle=EEGNetRegressor,
            model_kwargs={
                "num_input_channels": 64,
                "input_length": int(model_cfg["window_size"]),
                "temporal_filters": int(model_cfg["temporal_filters"]),
                "depth_multiplier": int(model_cfg["depth_multiplier"]),
                "separable_filters": int(model_cfg["separable_filters"]),
                "dropout_rate": float(model_cfg["dropout_rate"]),
            },
            device=device,
        )
    if model_name == "adt":
        return fit_loso_adt_for_subject(
            config=config,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            dataset_dir=dataset_dir,
            heldout_subject=heldout_subject,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            device=device,
        )
    raise ValueError(f"unsupported model {model_name}")


def aggregate_all_partials(config: dict, output_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    all_recording_rows: list[dict[str, object]] = []
    all_subject_rows: list[dict[str, object]] = []
    all_model_runs: list[dict[str, object]] = []
    all_leakage_entries: list[dict[str, object]] = []
    all_failures: list[dict[str, object]] = []
    for model_name in config["models"]:
        state = load_partial_state(output_dir, model_name)
        all_recording_rows.extend(state["recording_rows"])
        all_subject_rows.extend(state["subject_rows"])
        all_model_runs.extend(state["model_runs"])
        all_leakage_entries.extend(state["leakage_entries"])
        all_failures.extend(state["failures"])
    return all_recording_rows, all_subject_rows, all_model_runs, all_leakage_entries, all_failures


def write_leakage_summary(path: Path, *, config: dict, leakage_entries: list[dict[str, object]]) -> None:
    lines = [
        "# Leakage Summary",
        "",
        f"- `protocol`: `{config['protocol']}`",
        f"- `seed`: `{config['seed']}`",
        f"- `models_requested`: `{', '.join(config['models'])}`",
        "",
        "Model identity and training/selection protocol:",
        "",
    ]
    for model_name in config["models"]:
        if model_name not in config:
            continue
        model_cfg = config[model_name]
        lines.extend(
            [
                f"## `{model_name}`",
                f"- family: `{model_cfg['family']}`",
                f"- implementation: `{model_cfg['implementation_file']}`",
                f"- selection rule: `{model_cfg['selection_rule']}`",
                f"- training protocol: `{model_cfg['training_protocol']}`",
                "",
            ]
        )
    lines.extend(
        [
            "Pure LOSO guarantees for successful jobs:",
            "- held-out target subject excluded from train",
            "- held-out target subject excluded from val",
            "- held-out target subject excluded from scaler / normalization / alpha or checkpoint selection",
            "- held-out target subject used only for test evaluation",
            "",
        ]
    )
    for entry in leakage_entries:
        lines.append(
            f"- `{entry['dataset']}` / `{entry['model']}` / `{entry['heldout_subject']}`: "
            f"train_exclude=`{entry['excluded_target_from_train']}`, "
            f"val_exclude=`{entry['excluded_target_from_val']}`, "
            f"selection_exclude=`{entry['excluded_target_from_selection']}`, "
            f"test_only=`{','.join(entry['test_subjects'])}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_schema(
    *,
    config: dict,
    dataset_subjects: dict[str, list[str]],
    all_recording_rows: list[dict[str, object]],
    all_subject_rows: list[dict[str, object]],
    dataset_metrics: list[dict[str, object]],
    etard_condition_metrics: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    model_dataset_rows: list[dict[str, object]],
    leakage_entries: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    expected_pairs = {(dataset_id, model_name, subject_id) for dataset_id, subjects in dataset_subjects.items() for model_name in config["models"] for subject_id in subjects}
    successful_pairs = {(row["dataset"], row["model"], row["subject_id"]) for row in all_subject_rows}
    failed_pairs = {(item["dataset"], item["model"], item["subject_id"]) for item in failures if "subject_id" in item}
    missing_pairs = sorted(expected_pairs - successful_pairs - failed_pairs)

    subject_metric_fields_consistent = all(sorted(row.keys()) == sorted(SUBJECT_METRIC_FIELDS) for row in all_subject_rows)
    recording_metric_fields_consistent = all(sorted(row.keys()) == sorted(RECORDING_METRIC_FIELDS) for row in all_recording_rows)
    pearson_subject_ok = all(-1.0 <= float(row["metric_value"]) <= 1.0 for row in all_subject_rows)
    pearson_recording_ok = all(-1.0 <= float(row["metric_value"]) <= 1.0 for row in all_recording_rows)
    pearson_dataset_ok = all(
        all(-1.0 <= float(row[field]) <= 1.0 for field in ["mean_pearson", "std_pearson", "median_pearson", "min_pearson", "max_pearson", "backtransformed_mean_r"])
        for row in dataset_metrics
    )
    pearson_condition_ok = all(
        all(-1.0 <= float(row[field]) <= 1.0 for field in ["mean_pearson", "median_pearson", "std_pearson"])
        for row in etard_condition_metrics
    )
    leakage_ok = all(
        entry["excluded_target_from_train"] and entry["excluded_target_from_val"] and entry["excluded_target_from_selection"] and entry["test_subjects"] == [entry["heldout_subject"]]
        for entry in leakage_entries
    )
    comparison_rows_complete = len(comparison_rows) == len(all_subject_rows)
    model_dataset_rows_complete = len(model_dataset_rows) == len({(row["dataset"], row["model"]) for row in comparison_rows})
    return {
        "protocol": config["protocol"],
        "passed": bool(
            subject_metric_fields_consistent
            and recording_metric_fields_consistent
            and pearson_subject_ok
            and pearson_recording_ok
            and pearson_dataset_ok
            and pearson_condition_ok
            and leakage_ok
            and comparison_rows_complete
            and model_dataset_rows_complete
            and len(missing_pairs) == 0
        ),
        "checks": {
            "subject_metric_fields_consistent": subject_metric_fields_consistent,
            "recording_metric_fields_consistent": recording_metric_fields_consistent,
            "pearson_ranges_subject_metrics": pearson_subject_ok,
            "pearson_ranges_recording_metrics": pearson_recording_ok,
            "pearson_ranges_dataset_metrics": pearson_dataset_ok,
            "pearson_ranges_etard_condition_metrics": pearson_condition_ok,
            "target_subject_excluded_from_train_val_selection": leakage_ok,
            "comparison_rows_complete": comparison_rows_complete,
            "model_dataset_summary_complete": model_dataset_rows_complete,
            "dataset_model_subject_coverage_complete": len(missing_pairs) == 0,
        },
        "details": {
            "expected_jobs": len(expected_pairs),
            "successful_jobs": len(successful_pairs),
            "failed_jobs": len(failed_pairs),
            "missing_jobs": [
                {"dataset": dataset, "model": model, "subject_id": subject}
                for dataset, model, subject in missing_pairs
            ],
            "dataset_subjects": dataset_subjects,
            "recording_rows": len(all_recording_rows),
            "subject_rows": len(all_subject_rows),
            "dataset_metric_rows": len(dataset_metrics),
            "etard_condition_rows": len(etard_condition_metrics),
            "comparison_rows": len(comparison_rows),
            "model_dataset_rows": len(model_dataset_rows),
        },
    }


def write_result_summary(
    path: Path,
    *,
    config: dict,
    dataset_subjects: dict[str, list[str]],
    dataset_metrics: list[dict[str, object]],
    etard_condition_metrics: list[dict[str, object]],
    model_dataset_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    failure_groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for item in failures:
        failure_groups.setdefault((item["dataset"], item["model"], str(item.get("error", ""))), []).append(item)

    lines = [
        "# Gate0-Gate2 LOSO All-Models Single-Seed v1",
        "",
        "This directory extends the validated LOSO interface from ridge to the current unified subject-specific benchmark model set.",
        "It is a subject-independent LOSO single-seed round under the same scorer/schema style and does not add raw dumps or final paper prose.",
        "",
        "## Protocol",
        f"- `protocol`: `{config['protocol']}`",
        f"- `seed`: `{config['seed']}`",
        f"- `models_requested`: `{', '.join(config['models'])}`",
        "",
        "## Subject inventory",
    ]
    for dataset_id, subjects in dataset_subjects.items():
        lines.append(f"- `{dataset_id}` ({len(subjects)} subjects): `{', '.join(subjects)}`")
    lines.extend(
        [
            "",
            "## Dataset metrics",
            "| dataset | model | n_subjects | mean_pearson | std_pearson | min_pearson | max_pearson |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_metrics:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_subjects']} | {row['mean_pearson']:.6f} | {row['std_pearson']:.6f} | "
            f"{row['min_pearson']:.6f} | {row['max_pearson']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## LOSO vs subject-specific dataset summary",
            "| dataset | model | family | loso_mean_pearson | subject_specific_mean_pearson | delta_loso_minus_subject_specific |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in model_dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['model_family']} | {float(row['loso_mean_pearson']):.6f} | "
            f"{float(row['subject_specific_mean_pearson']):.6f} | {float(row['delta_loso_minus_subject_specific']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Etard condition metrics",
            "| model | condition | n_subjects | n_recordings | mean_pearson | std_pearson | negative_recording_count |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in etard_condition_metrics:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['n_subjects']} | {row['n_recordings']} | {row['mean_pearson']:.6f} | {row['std_pearson']:.6f} | {row['negative_recording_count']} |"
        )
    if failures:
        lines.extend(["", "## Failures and skips"])
        for (dataset_id, model_name, reason), items in sorted(failure_groups.items()):
            statuses = sorted({str(item.get("status", "failed")) for item in items})
            example_subjects = ", ".join(sorted(str(item["subject_id"]) for item in items[:5]))
            lines.append(
                f"- `{dataset_id}` / `{model_name}`: `{len(items)}` jobs, statuses=`{', '.join(statuses)}`, example_subjects=`{example_subjects}`, reason=`{reason}`"
            )
    lines.append("")
    lines.append("Note: `dnn` and `fcnn` remain two implementations within the same `MLP/FCNN family` and are not described as separate architecture families in this report.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_incremental_comparison(path: Path, *, config: dict, executed_models: list[str], failures: list[dict[str, object]]) -> None:
    failed_models = sorted({str(item["model"]) for item in failures})
    lines = [
        "# Incremental Comparison",
        "",
        f"- base branch: `{config['previous_fix_branch']}`",
        f"- base commit: `{config['previous_fix_commit']}`",
        f"- current branch: `{config['current_branch']}`",
        "- this round expands LOSO from ridge-only to the current unified subject-specific model set.",
        "- ridge is reused from the verified `loso-ridge-full-v1` branch instead of being rerun.",
        f"- executed models in this branch: `{', '.join(executed_models)}`",
    ]
    if failures:
        lines.append(f"- subject-level failures or skips recorded: `{len(failures)}`")
        lines.append(f"- models without successful LOSO result rows in this round: `{', '.join(failed_models)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    device = resolve_device(args.device)

    selected_models = list(dict.fromkeys(args.models if args.models else config["priority_order"]))
    selected_models = [model for model in selected_models if model in config["models"]]

    dataset_subjects: dict[str, list[str]] = {}
    dataset_dirs: dict[str, Path] = {}
    for dataset_cfg in config["datasets"]:
        dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
        if not dataset_dir.exists():
            raise FileNotFoundError(f"missing dataset locator: {dataset_dir}")
        dataset_dirs[dataset_cfg["dataset_id"]] = dataset_dir
        dataset_subjects[dataset_cfg["dataset_id"]] = list_reference_subjects(dataset_dir, split="test")

    for model_name in selected_models:
        state = load_partial_state(output_dir, model_name) if args.resume else {
            "recording_rows": [],
            "subject_rows": [],
            "model_runs": [],
            "leakage_entries": [],
            "failures": [],
            "completed_pairs": [],
        }
        skip_cfg = config.get("predeclared_skips", {}).get(model_name)
        if skip_cfg is not None:
            if not state["subject_rows"] and not state["failures"] and not state["completed_pairs"]:
                state = predeclared_skip_state(
                    config=config,
                    dataset_subjects=dataset_subjects,
                    model_name=model_name,
                    reason=str(skip_cfg["reason"]),
                    status=str(skip_cfg.get("status", "skipped_with_reason")),
                )
                save_partial_state(output_dir, model_name, state)
            continue
        if model_name == "ridge":
            if not state["completed_pairs"]:
                recording_rows, subject_rows, model_runs, leakage_entries = fit_loso_ridge_rows_from_reference(config)
                state["recording_rows"] = recording_rows
                state["subject_rows"] = subject_rows
                state["model_runs"] = model_runs
                state["leakage_entries"] = leakage_entries
                state["completed_pairs"] = [(row["dataset"], row["subject_id"]) for row in subject_rows]
                save_partial_state(output_dir, model_name, state)
            continue

        for dataset_cfg in config["datasets"]:
            dataset_id = dataset_cfg["dataset_id"]
            dataset_dir = dataset_dirs[dataset_id]
            all_subjects = dataset_subjects[dataset_id]
            for heldout_subject in all_subjects:
                pair = (dataset_id, heldout_subject)
                if pair in state["completed_pairs"]:
                    continue
                try:
                    recording_rows, subject_rows, meta, leakage_entry = fit_subject_model(
                        config=config,
                        model_name=model_name,
                        dataset_cfg=dataset_cfg,
                        dataset_dir=dataset_dir,
                        heldout_subject=heldout_subject,
                        all_subjects=all_subjects,
                        device=device,
                    )
                    state["recording_rows"].extend(recording_rows)
                    state["subject_rows"].extend(subject_rows)
                    state["model_runs"].append(meta)
                    state["leakage_entries"].append(leakage_entry)
                    state["completed_pairs"].append(pair)
                except Exception as exc:
                    state["failures"].append(
                        {
                            "dataset": dataset_id,
                            "model": model_name,
                            "subject_id": heldout_subject,
                            "status": "failed",
                            "error": repr(exc),
                        }
                    )
                save_partial_state(output_dir, model_name, state)

    all_recording_rows, all_subject_rows, all_model_runs, all_leakage_entries, all_failures = aggregate_all_partials(config, output_dir)
    write_generic_csv(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_generic_csv(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)

    dataset_metrics = summarize_dataset_metrics(all_subject_rows)
    etard_condition_metrics = summarize_etard_condition_metrics(all_recording_rows)
    write_generic_csv(
        output_dir / "dataset_metrics.csv",
        dataset_metrics,
        ["dataset", "model", "n_subjects", "mean_pearson", "std_pearson", "median_pearson", "min_pearson", "max_pearson", "fisher_z_mean", "backtransformed_mean_r"],
    )
    write_generic_csv(
        output_dir / "etard_condition_metrics.csv",
        etard_condition_metrics,
        ["dataset", "model", "condition", "n_subjects", "n_recordings", "mean_pearson", "median_pearson", "std_pearson", "negative_recording_count"],
    )

    subject_specific_lookup = load_subject_specific_reference(config)
    comparison_rows: list[dict[str, object]] = []
    ridge_reference_rows, _ = load_ridge_loso_reference(config)
    ridge_reference_lookup = {(row["dataset"], row["subject_id"]): row for row in ridge_reference_rows}
    ridge_comparison_rows: list[dict[str, object]] = []
    for row in sorted(all_subject_rows, key=lambda item: (item["dataset"], item["model"], item["subject_id"])):
        model_cfg = config[row["model"]]
        reference_row = subject_specific_lookup[(row["dataset"], row["model"], row["subject_id"])]
        loso_metric = float(row["metric_value"])
        subject_specific_metric = float(reference_row["metric_value"])
        comparison_rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "model_family": model_cfg["family"],
                "subject_id": row["subject_id"],
                "loso_protocol": row["protocol"],
                "loso_metric": loso_metric,
                "loso_checkpoint_id": row["checkpoint_id"],
                "subject_specific_protocol": reference_row["protocol"],
                "subject_specific_metric": subject_specific_metric,
                "subject_specific_checkpoint_id": reference_row["checkpoint_id"],
                "delta_loso_minus_subject_specific": loso_metric - subject_specific_metric,
            }
        )
        if row["model"] == "ridge":
            ridge_reference_row = ridge_reference_lookup[(row["dataset"], row["subject_id"])]
            ridge_comparison_rows.append(
                {
                    "dataset": row["dataset"],
                    "subject_id": row["subject_id"],
                    "current_protocol": row["protocol"],
                    "current_metric": loso_metric,
                    "reference_protocol": ridge_reference_row["protocol"],
                    "reference_metric": float(ridge_reference_row["metric_value"]),
                    "delta_current_minus_reference": loso_metric - float(ridge_reference_row["metric_value"]),
                }
            )
    write_loso_vs_subject_specific_comparison_csv(output_dir / "loso_vs_subject_specific_comparison.csv", comparison_rows)
    write_loso_vs_subject_specific_comparison_md(output_dir / "loso_vs_subject_specific_comparison.md", comparison_rows)
    write_model_dataset_summary_csv(output_dir / "model_dataset_summary.csv", comparison_rows)
    write_loso_vs_ridge_loso_comparison_csv(output_dir / "loso_vs_ridge_loso_comparison.csv", ridge_comparison_rows)
    model_dataset_rows = load_dict_rows(output_dir / "model_dataset_summary.csv")

    write_leakage_summary(output_dir / "leakage_summary.md", config=config, leakage_entries=all_leakage_entries)
    write_json(output_dir / "failure_report.json", {"failures": all_failures})
    schema_validation = validate_schema(
        config=config,
        dataset_subjects=dataset_subjects,
        all_recording_rows=all_recording_rows,
        all_subject_rows=all_subject_rows,
        dataset_metrics=dataset_metrics,
        etard_condition_metrics=etard_condition_metrics,
        comparison_rows=comparison_rows,
        model_dataset_rows=model_dataset_rows,
        leakage_entries=all_leakage_entries,
        failures=all_failures,
    )
    write_json(output_dir / "schema_validation_report.json", schema_validation)
    write_result_summary(
        output_dir / "result_summary.md",
        config=config,
        dataset_subjects=dataset_subjects,
        dataset_metrics=dataset_metrics,
        etard_condition_metrics=etard_condition_metrics,
        model_dataset_rows=model_dataset_rows,
        failures=all_failures,
    )
    write_incremental_comparison(
        output_dir / "incremental_comparison.md",
        config=config,
        executed_models=sorted({row["model"] for row in all_model_runs}),
        failures=all_failures,
    )

    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "artifact_scope": config["artifact_scope"],
            "previous_fix_branch": config["previous_fix_branch"],
            "previous_fix_commit": config["previous_fix_commit"],
            "current_branch": config["current_branch"],
            "device": device,
            "seed": int(config["seed"]),
            "models_requested": config["models"],
            "models_selected_this_run": selected_models,
            "datasets": [
                {
                    "dataset_id": item["dataset_id"],
                    "dataset_locator": item["dataset_locator"],
                    "sampling_rate": int(item["sampling_rate"]),
                    "heldout_subjects": dataset_subjects[item["dataset_id"]],
                }
                for item in config["datasets"]
            ],
            "job_accounting": {
                "planned_jobs": sum(len(subjects) for subjects in dataset_subjects.values()) * len(config["models"]),
                "successful_jobs": len(all_subject_rows),
                "failed_jobs": len(all_failures),
            },
            "model_specs": {
                model_name: {
                    key: value
                    for key, value in config[model_name].items()
                    if key
                    in {
                        "family",
                        "architecture_id",
                        "implementation_file",
                        "selection_rule",
                        "training_protocol",
                    }
                }
                for model_name in config["models"]
            },
            "model_runs": all_model_runs,
            "failures": all_failures,
            "artifacts": {
                "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
                "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
                "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
                "etard_condition_metrics": repo_relative(output_dir / "etard_condition_metrics.csv"),
                "model_dataset_summary": repo_relative(output_dir / "model_dataset_summary.csv"),
                "loso_vs_subject_specific_comparison_csv": repo_relative(output_dir / "loso_vs_subject_specific_comparison.csv"),
                "loso_vs_subject_specific_comparison_md": repo_relative(output_dir / "loso_vs_subject_specific_comparison.md"),
                "loso_vs_ridge_loso_comparison_csv": repo_relative(output_dir / "loso_vs_ridge_loso_comparison.csv"),
                "leakage_summary": repo_relative(output_dir / "leakage_summary.md"),
                "failure_report": repo_relative(output_dir / "failure_report.json"),
                "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
                "result_summary": repo_relative(output_dir / "result_summary.md"),
                "incremental_comparison": repo_relative(output_dir / "incremental_comparison.md"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
