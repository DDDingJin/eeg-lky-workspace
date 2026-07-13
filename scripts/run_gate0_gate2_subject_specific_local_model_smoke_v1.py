from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")
SHARED_MLDECODERS = SHARED_WORKSPACE / "external" / "upstream" / "mldecoders"
SHARED_HAPPYQUOKKA = SHARED_WORKSPACE / "external" / "upstream" / "HappyQuokka_system_for_EEG_Challenge"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SHARED_MLDECODERS.exists() and str(SHARED_MLDECODERS) not in sys.path:
    sys.path.insert(0, str(SHARED_MLDECODERS))
if SHARED_HAPPYQUOKKA.exists() and str(SHARED_HAPPYQUOKKA) not in sys.path:
    sys.path.insert(0, str(SHARED_HAPPYQUOKKA))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.happyquokka_reference import l1_loss, pearson_loss, pearson_metric
from repro.mldecoders.cca import trim_valid_range
from repro.mldecoders.models import VLAAIExactOfficialRegressor
from repro.reference_baselines import (
    ReferenceWindowDataset,
    concatenate_reference_split,
    load_reference_recordings,
    train_dnn_reference_logged,
)

try:
    from models.FFT_block import Decoder  # type: ignore
except Exception:
    Decoder = None


ADAPTER_MATRIX_FIELDS = [
    "dataset",
    "subject_id",
    "model",
    "seed",
    "status",
    "input_shape",
    "raw_output_shape",
    "postprocessed_prediction_shape",
    "target_shape",
    "scorer_input_shape",
    "selected_hyperparameters",
    "subject_metric",
    "recording_count",
    "failure_reason",
    "notes",
]


@dataclass
class ModelSmokeResult:
    matrix_row: dict[str, object]
    model_run_entry: dict[str, object]
    recording_rows: list[object]
    subject_rows: list[object]
    failure_entry: dict[str, object] | None
    shape_markdown: str


class SubjectSpecificHappyQuokkaTrainDataset(Dataset):
    def __init__(self, input_dir: Path, split: str, participant: str, *, input_length: int, channels: Iterable[int] | None = None) -> None:
        self.input_length = int(input_length)
        self.recordings = load_reference_recordings(input_dir, split, participant, channels=channels)

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int):
        _, eeg, env = self.recordings[index]
        if eeg.shape[0] < self.input_length:
            raise ValueError(f"recording shorter than HappyQuokka input_length={self.input_length}")
        start = np.random.randint(0, eeg.shape[0] - self.input_length + 1)
        eeg_chunk = eeg[start : start + self.input_length]
        env_chunk = env[start : start + self.input_length]
        return (
            torch.from_numpy(eeg_chunk.astype(np.float32)),
            torch.from_numpy(env_chunk.astype(np.float32)).unsqueeze(-1),
        )


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


def aggregate_model_outputs(windows: list[WindowPrediction], artifact_scope: str) -> tuple[list[object], list[object]]:
    grouped: dict[tuple[str, str], list[WindowPrediction]] = {}
    for window in windows:
        grouped.setdefault((window.subject_id, window.recording_id), []).append(window)

    recording_rows: list[object] = []
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
            )
        )
    subject_rows = subject_metric_rows(recording_rows)
    return recording_rows, subject_rows


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def shape_text(shape: Iterable[int] | tuple[int, ...] | None) -> str:
    if shape is None:
        return ""
    return "[" + ", ".join(str(int(item)) for item in shape) + "]"


def print_event(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def cap_fit_samples(x: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] <= max_samples:
        return x, y
    indices = np.linspace(0, x.shape[0] - 1, num=max_samples, dtype=np.int64)
    return x[indices], y[indices]


def capped_window_starts(signal_length: int, input_length: int, max_windows: int) -> list[int]:
    total = signal_length - input_length + 1
    if total <= 0:
        return []
    if total <= max_windows:
        return list(range(total))
    return np.linspace(0, total - 1, num=max_windows, dtype=np.int64).tolist()


def make_failure_result(
    *,
    dataset_id: str,
    subject_id: str,
    model_name: str,
    seed: int,
    status: str,
    failure_reason: str,
    notes: str,
    input_shape: str = "",
    raw_output_shape: str = "",
    postprocessed_prediction_shape: str = "",
    target_shape: str = "",
    scorer_input_shape: str = "",
    selected_hyperparameters: dict[str, object] | None = None,
) -> ModelSmokeResult:
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": model_name,
        "seed": seed,
        "status": status,
        "input_shape": input_shape,
        "raw_output_shape": raw_output_shape,
        "postprocessed_prediction_shape": postprocessed_prediction_shape,
        "target_shape": target_shape,
        "scorer_input_shape": scorer_input_shape,
        "selected_hyperparameters": json.dumps(selected_hyperparameters or {}, sort_keys=True),
        "subject_metric": "",
        "recording_count": 0,
        "failure_reason": failure_reason,
        "notes": notes,
    }
    failure_entry = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": model_name,
        "seed": seed,
        "status": status,
        "failure_reason": failure_reason,
        "notes": notes,
    }
    return ModelSmokeResult(
        matrix_row=matrix_row,
        model_run_entry=dict(matrix_row),
        recording_rows=[],
        subject_rows=[],
        failure_entry=failure_entry,
        shape_markdown="\n".join(
            [
                f"## `{model_name}`",
                "",
                f"- status: `{status}`",
                f"- input_shape: `{input_shape or 'n/a'}`",
                f"- raw_output_shape: `{raw_output_shape or 'n/a'}`",
                f"- postprocessed_prediction_shape: `{postprocessed_prediction_shape or 'n/a'}`",
                f"- target_shape: `{target_shape or 'n/a'}`",
                f"- scorer_input_shape: `{scorer_input_shape or 'n/a'}`",
                f"- failure_reason: `{failure_reason}`",
                f"- notes: {notes}",
            ]
        )
        + "\n",
    )


def run_linear_smoke(config: dict, dataset_dir: Path, dataset_id: str, subject_id: str, sampling_rate: int) -> ModelSmokeResult:
    linear_cfg = config["linear"]
    budget_cfg = config["smoke_runtime_budget"]
    start_lag = int(linear_cfg["start_lag"])
    end_lag = int(linear_cfg["end_lag"])
    x_train, y_train = concatenate_reference_split(dataset_dir, "train", subject_id, channels=range(64))
    x_val, y_val = concatenate_reference_split(dataset_dir, "val", subject_id, channels=range(64))
    x_train_lag, _, y_train_center = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag, _, y_val_center = trim_valid_range(x_val, y_val, start_lag, end_lag)
    x_train_lag, y_train_center = cap_fit_samples(x_train_lag, y_train_center, int(budget_cfg["max_fit_samples_per_split"]))
    x_val_lag, y_val_center = cap_fit_samples(x_val_lag, y_val_center, int(budget_cfg["max_fit_samples_per_split"]))

    model = LinearRegression()
    model.fit(x_train_lag, y_train_center)
    val_pred = model.predict(x_val_lag).astype(np.float32)
    val_pearson = correlation(val_pred, y_val_center.astype(np.float32))
    offset = end_lag - 1
    checkpoint_id = "linear_ols"
    windows: list[WindowPrediction] = []
    max_eval_windows = int(budget_cfg["max_eval_windows_per_recording"])

    first_recording = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    first_x_lag, _, first_y_center = trim_valid_range(first_recording[1], first_recording[2], start_lag, end_lag)
    if first_x_lag.shape[0] > max_eval_windows:
        first_indices = np.linspace(0, first_x_lag.shape[0] - 1, num=max_eval_windows, dtype=np.int64)
        first_x_lag = first_x_lag[first_indices]
        first_y_center = first_y_center[first_indices]
    first_pred = model.predict(first_x_lag).astype(np.float32)

    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
        if x_lag.shape[0] > max_eval_windows:
            indices = np.linspace(0, x_lag.shape[0] - 1, num=max_eval_windows, dtype=np.int64)
            x_lag = x_lag[indices]
            y_center = y_center[indices]
        pred = model.predict(x_lag).astype(np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="linear",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=checkpoint_id,
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=y_center.astype(np.float32),
            )
        )

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    subject_metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    selected_hyperparameters = {
        "start_lag": start_lag,
        "end_lag": end_lag,
        "max_fit_samples_per_split": int(budget_cfg["max_fit_samples_per_split"]),
        "max_eval_windows_per_recording": max_eval_windows,
    }
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "linear",
        "seed": int(config["seed"]),
        "status": "success",
        "input_shape": shape_text(first_x_lag.shape),
        "raw_output_shape": shape_text(first_pred.shape),
        "postprocessed_prediction_shape": shape_text(first_pred.shape),
        "target_shape": shape_text(first_y_center.shape),
        "scorer_input_shape": shape_text(first_pred.shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "subject_metric": subject_metric,
        "recording_count": len(recording_rows),
        "failure_reason": "",
        "notes": f"validation Pearson recorded as auxiliary metadata: {val_pearson:.6f}",
    }
    model_run_entry = dict(matrix_row)
    model_run_entry.update(
        {
            "checkpoint_id": checkpoint_id,
            "best_val_score": val_pearson,
        }
    )
    shape_markdown = "\n".join(
        [
            "## `linear`",
            "",
            "- status: `success`",
            f"- input_shape: `{shape_text(first_x_lag.shape)}`",
            f"- raw_output_shape: `{shape_text(first_pred.shape)}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_pred.shape)}`",
            f"- target_shape: `{shape_text(first_y_center.shape)}`",
            f"- scorer_input_shape: `{shape_text(first_pred.shape)}`",
            f"- selected_hyperparameters: `{json.dumps(selected_hyperparameters, sort_keys=True)}`",
            f"- validation_score: `{val_pearson:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, recording_rows, subject_rows, None, shape_markdown)


def run_lasso_smoke(config: dict, dataset_dir: Path, dataset_id: str, subject_id: str, sampling_rate: int) -> ModelSmokeResult:
    lasso_cfg = config["lasso"]
    budget_cfg = config["smoke_runtime_budget"]
    start_lag = int(lasso_cfg["start_lag"])
    end_lag = int(lasso_cfg["end_lag"])
    x_train, y_train = concatenate_reference_split(dataset_dir, "train", subject_id, channels=range(64))
    x_val, y_val = concatenate_reference_split(dataset_dir, "val", subject_id, channels=range(64))
    x_train_lag, _, y_train_center = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag, _, y_val_center = trim_valid_range(x_val, y_val, start_lag, end_lag)
    x_train_lag, y_train_center = cap_fit_samples(x_train_lag, y_train_center, int(budget_cfg["max_fit_samples_per_split"]))
    x_val_lag, y_val_center = cap_fit_samples(x_val_lag, y_val_center, int(budget_cfg["max_fit_samples_per_split"]))

    best_model = None
    best_summary = None
    for alpha in lasso_cfg["alphas"]:
        model = Lasso(alpha=float(alpha), max_iter=int(lasso_cfg["max_iter"]))
        model.fit(x_train_lag, y_train_center)
        val_pred = model.predict(x_val_lag).astype(np.float32)
        val_pearson = correlation(val_pred, y_val_center.astype(np.float32))
        if best_summary is None or val_pearson > float(best_summary["best_val_score"]):
            best_model = model
            best_summary = {"best_alpha": float(alpha), "best_val_score": float(val_pearson)}

    if best_model is None or best_summary is None:
        raise RuntimeError("lasso smoke failed to select a valid model")

    offset = end_lag - 1
    checkpoint_id = f"lasso_alpha_{best_summary['best_alpha']}"
    windows: list[WindowPrediction] = []
    max_eval_windows = int(budget_cfg["max_eval_windows_per_recording"])
    first_recording = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    first_x_lag, _, first_y_center = trim_valid_range(first_recording[1], first_recording[2], start_lag, end_lag)
    if first_x_lag.shape[0] > max_eval_windows:
        first_indices = np.linspace(0, first_x_lag.shape[0] - 1, num=max_eval_windows, dtype=np.int64)
        first_x_lag = first_x_lag[first_indices]
        first_y_center = first_y_center[first_indices]
    first_pred = best_model.predict(first_x_lag).astype(np.float32)

    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
        if x_lag.shape[0] > max_eval_windows:
            indices = np.linspace(0, x_lag.shape[0] - 1, num=max_eval_windows, dtype=np.int64)
            x_lag = x_lag[indices]
            y_center = y_center[indices]
        pred = best_model.predict(x_lag).astype(np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="lasso",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=checkpoint_id,
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=y_center.astype(np.float32),
            )
        )

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    subject_metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    selected_hyperparameters = {
        "start_lag": start_lag,
        "end_lag": end_lag,
        "alpha": best_summary["best_alpha"],
        "max_fit_samples_per_split": int(budget_cfg["max_fit_samples_per_split"]),
        "max_eval_windows_per_recording": max_eval_windows,
    }
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "lasso",
        "seed": int(config["seed"]),
        "status": "success",
        "input_shape": shape_text(first_x_lag.shape),
        "raw_output_shape": shape_text(first_pred.shape),
        "postprocessed_prediction_shape": shape_text(first_pred.shape),
        "target_shape": shape_text(first_y_center.shape),
        "scorer_input_shape": shape_text(first_pred.shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "subject_metric": subject_metric,
        "recording_count": len(recording_rows),
        "failure_reason": "",
        "notes": f"validation Pearson selected alpha={best_summary['best_alpha']}",
    }
    model_run_entry = dict(matrix_row)
    model_run_entry["best_val_score"] = best_summary["best_val_score"]
    shape_markdown = "\n".join(
        [
            "## `lasso`",
            "",
            "- status: `success`",
            f"- input_shape: `{shape_text(first_x_lag.shape)}`",
            f"- raw_output_shape: `{shape_text(first_pred.shape)}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_pred.shape)}`",
            f"- target_shape: `{shape_text(first_y_center.shape)}`",
            f"- scorer_input_shape: `{shape_text(first_pred.shape)}`",
            f"- selected_hyperparameters: `{json.dumps(selected_hyperparameters, sort_keys=True)}`",
            f"- validation_score: `{best_summary['best_val_score']:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, recording_rows, subject_rows, None, shape_markdown)


def run_elasticnet_smoke(config: dict, dataset_dir: Path, dataset_id: str, subject_id: str, sampling_rate: int) -> ModelSmokeResult:
    elastic_cfg = config["elasticnet"]
    budget_cfg = config["smoke_runtime_budget"]
    start_lag = int(elastic_cfg["start_lag"])
    end_lag = int(elastic_cfg["end_lag"])
    x_train, y_train = concatenate_reference_split(dataset_dir, "train", subject_id, channels=range(64))
    x_val, y_val = concatenate_reference_split(dataset_dir, "val", subject_id, channels=range(64))
    x_train_lag, _, y_train_center = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag, _, y_val_center = trim_valid_range(x_val, y_val, start_lag, end_lag)
    x_train_lag, y_train_center = cap_fit_samples(x_train_lag, y_train_center, int(budget_cfg["max_fit_samples_per_split"]))
    x_val_lag, y_val_center = cap_fit_samples(x_val_lag, y_val_center, int(budget_cfg["max_fit_samples_per_split"]))

    best_model = None
    best_summary = None
    for alpha in elastic_cfg["alphas"]:
        for l1_ratio in elastic_cfg["l1_ratios"]:
            model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=int(elastic_cfg["max_iter"]))
            model.fit(x_train_lag, y_train_center)
            val_pred = model.predict(x_val_lag).astype(np.float32)
            val_pearson = correlation(val_pred, y_val_center.astype(np.float32))
            if best_summary is None or val_pearson > float(best_summary["best_val_score"]):
                best_model = model
                best_summary = {
                    "best_alpha": float(alpha),
                    "best_l1_ratio": float(l1_ratio),
                    "best_val_score": float(val_pearson),
                }

    if best_model is None or best_summary is None:
        raise RuntimeError("elasticnet smoke failed to select a valid model")

    offset = end_lag - 1
    checkpoint_id = f"elasticnet_alpha_{best_summary['best_alpha']}_l1_{best_summary['best_l1_ratio']}"
    windows: list[WindowPrediction] = []
    max_eval_windows = int(budget_cfg["max_eval_windows_per_recording"])
    first_recording = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    first_x_lag, _, first_y_center = trim_valid_range(first_recording[1], first_recording[2], start_lag, end_lag)
    if first_x_lag.shape[0] > max_eval_windows:
        first_indices = np.linspace(0, first_x_lag.shape[0] - 1, num=max_eval_windows, dtype=np.int64)
        first_x_lag = first_x_lag[first_indices]
        first_y_center = first_y_center[first_indices]
    first_pred = best_model.predict(first_x_lag).astype(np.float32)

    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
        if x_lag.shape[0] > max_eval_windows:
            indices = np.linspace(0, x_lag.shape[0] - 1, num=max_eval_windows, dtype=np.int64)
            x_lag = x_lag[indices]
            y_center = y_center[indices]
        pred = best_model.predict(x_lag).astype(np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="elasticnet",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=checkpoint_id,
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=y_center.astype(np.float32),
            )
        )

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    subject_metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    selected_hyperparameters = {
        "start_lag": start_lag,
        "end_lag": end_lag,
        "alpha": best_summary["best_alpha"],
        "l1_ratio": best_summary["best_l1_ratio"],
        "max_fit_samples_per_split": int(budget_cfg["max_fit_samples_per_split"]),
        "max_eval_windows_per_recording": max_eval_windows,
    }
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "elasticnet",
        "seed": int(config["seed"]),
        "status": "success",
        "input_shape": shape_text(first_x_lag.shape),
        "raw_output_shape": shape_text(first_pred.shape),
        "postprocessed_prediction_shape": shape_text(first_pred.shape),
        "target_shape": shape_text(first_y_center.shape),
        "scorer_input_shape": shape_text(first_pred.shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "subject_metric": subject_metric,
        "recording_count": len(recording_rows),
        "failure_reason": "",
        "notes": "validation Pearson selected alpha and l1_ratio",
    }
    model_run_entry = dict(matrix_row)
    model_run_entry["best_val_score"] = best_summary["best_val_score"]
    shape_markdown = "\n".join(
        [
            "## `elasticnet`",
            "",
            "- status: `success`",
            f"- input_shape: `{shape_text(first_x_lag.shape)}`",
            f"- raw_output_shape: `{shape_text(first_pred.shape)}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_pred.shape)}`",
            f"- target_shape: `{shape_text(first_y_center.shape)}`",
            f"- scorer_input_shape: `{shape_text(first_pred.shape)}`",
            f"- selected_hyperparameters: `{json.dumps(selected_hyperparameters, sort_keys=True)}`",
            f"- validation_score: `{best_summary['best_val_score']:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, recording_rows, subject_rows, None, shape_markdown)


def fit_and_predict_window_regressor(
    *,
    config: dict,
    model_name: str,
    dataset_dir: Path,
    dataset_id: str,
    subject_id: str,
    sampling_rate: int,
    device: str,
    model_handle,
    model_kwargs: dict[str, object],
    train_kwargs: dict[str, object],
) -> ModelSmokeResult:
    max_eval_windows = int(config["smoke_runtime_budget"]["max_eval_windows_per_recording"])
    train_result = train_dnn_reference_logged(
        dataset_dir,
        subject_id,
        model_handle,
        model_kwargs,
        epochs=int(train_kwargs["max_epochs"]),
        lr=float(train_kwargs["learning_rate"]),
        weight_decay=float(train_kwargs["weight_decay"]),
        batch_size=int(train_kwargs["batch_size"]),
        early_stopping_patience=int(train_kwargs["early_stopping_patience"]),
        device=device,
        seed=int(config["seed"]),
        channels=range(int(model_kwargs["num_input_channels"])),
    )
    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(train_result.state_dict)
    model.eval()

    input_length = int(model_kwargs["input_length"])
    offset = input_length - 1
    first_recording = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(int(model_kwargs["num_input_channels"])))[0]
    first_eeg = torch.from_numpy(first_recording[1][:input_length].astype(np.float32)).T.unsqueeze(0).to(device)
    with torch.no_grad():
        first_raw = model(first_eeg).detach().cpu().numpy()

    windows: list[WindowPrediction] = []
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(int(model_kwargs["num_input_channels"]))):
        preds: list[float] = []
        targets: list[float] = []
        eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
        with torch.no_grad():
            for start in capped_window_starts(eeg.shape[0], input_length, max_eval_windows):
                batch = eeg_tensor[start : start + input_length].T.unsqueeze(0).to(device)
                preds.append(float(model(batch).item()))
                targets.append(float(env[start + input_length - 1]))
        pred_arr = np.asarray(preds, dtype=np.float32)
        target_arr = np.asarray(targets, dtype=np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model=model_name,
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=f"{model_name}_epoch_{train_result.best_epoch}",
                full_length=len(env),
                offset=offset,
                prediction=pred_arr,
                target=target_arr,
            )
        )

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    subject_metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    selected_hyperparameters = {
        "window_size": input_length,
        "batch_size": int(train_kwargs["batch_size"]),
        "max_epochs": int(train_kwargs["max_epochs"]),
        "learning_rate": float(train_kwargs["learning_rate"]),
        "weight_decay": float(train_kwargs["weight_decay"]),
        "best_epoch": int(train_result.best_epoch),
        "max_eval_windows_per_recording": max_eval_windows,
    }
    first_postprocessed_shape = (first_recording[1].shape[0] - input_length + 1,)
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": model_name,
        "seed": int(config["seed"]),
        "status": "success",
        "input_shape": shape_text(tuple(first_eeg.shape)),
        "raw_output_shape": shape_text(tuple(first_raw.shape)),
        "postprocessed_prediction_shape": shape_text(first_postprocessed_shape),
        "target_shape": shape_text(first_postprocessed_shape),
        "scorer_input_shape": shape_text(first_postprocessed_shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "subject_metric": subject_metric,
        "recording_count": len(recording_rows),
        "failure_reason": "",
        "notes": f"best_val_score={train_result.best_val_score:.6f}; max_eval_windows_per_recording={max_eval_windows}",
    }
    model_run_entry = dict(matrix_row)
    model_run_entry["best_val_score"] = float(train_result.best_val_score)
    shape_markdown = "\n".join(
        [
            f"## `{model_name}`",
            "",
            "- status: `success`",
            f"- input_shape: `{shape_text(tuple(first_eeg.shape))}`",
            f"- raw_output_shape: `{shape_text(tuple(first_raw.shape))}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_postprocessed_shape)}`",
            f"- target_shape: `{shape_text(first_postprocessed_shape)}`",
            f"- scorer_input_shape: `{shape_text(first_postprocessed_shape)}`",
            f"- selected_hyperparameters: `{json.dumps(selected_hyperparameters, sort_keys=True)}`",
            f"- best_val_score: `{train_result.best_val_score:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, recording_rows, subject_rows, None, shape_markdown)


def run_vlaai_smoke(config: dict, dataset_dir: Path, dataset_id: str, subject_id: str, sampling_rate: int, device: str) -> ModelSmokeResult:
    vlaai_cfg = config["vlaai"]
    return fit_and_predict_window_regressor(
        config=config,
        model_name="vlaai",
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        subject_id=subject_id,
        sampling_rate=sampling_rate,
        device=device,
        model_handle=VLAAIExactOfficialRegressor,
        model_kwargs={
            "num_input_channels": 64,
            "input_length": int(vlaai_cfg["window_size"]),
        },
        train_kwargs=vlaai_cfg,
    )


def train_happyquokka_subject_specific(
    *,
    input_dir: Path,
    participant: str,
    device: str,
    input_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    dropout: float,
    lamda: float,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if Decoder is None:
        raise RuntimeError("HappyQuokka upstream decoder import failed")

    train_set = SubjectSpecificHappyQuokkaTrainDataset(input_dir, "train", participant, input_length=input_length, channels=range(64))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    model = Decoder(
        in_channel=64,
        d_model=128,
        d_inner=1024,
        n_head=2,
        n_layers=8,
        fft_conv1d_kernel=(9, 1),
        fft_conv1d_padding=(4, 0),
        dropout=dropout,
        g_con=False,
        within_sub_num=1,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val_metric = -float("inf")
    history = {"train_loss": [], "val_metric": []}

    val_recordings = load_reference_recordings(input_dir, "val", participant, channels=range(64))
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for eeg_batch, env_batch in train_loader:
            eeg_batch = eeg_batch.to(device)
            env_batch = env_batch.to(device)
            sub_ids = torch.zeros(eeg_batch.shape[0], dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(eeg_batch, sub_ids)
            loss = (pearson_loss(outputs, env_batch) + lamda * l1_loss(outputs, env_batch)).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_metrics = []
        with torch.no_grad():
            for _, eeg, env in val_recordings:
                if eeg.shape[0] < input_length:
                    continue
                eeg_chunks = []
                env_chunks = []
                for start in range(0, eeg.shape[0] - input_length + 1, input_length):
                    eeg_chunks.append(torch.from_numpy(eeg[start : start + input_length].astype(np.float32)))
                    env_chunks.append(torch.from_numpy(env[start : start + input_length].astype(np.float32)).unsqueeze(-1))
                if not eeg_chunks:
                    continue
                eeg_tensor = torch.stack(eeg_chunks, dim=0).to(device)
                env_tensor = torch.stack(env_chunks, dim=0).to(device)
                sub_ids = torch.zeros(eeg_tensor.shape[0], dtype=torch.long, device=device)
                outputs = model(eeg_tensor, sub_ids)
                val_metrics.append(float(pearson_metric(outputs, env_tensor).mean().item()))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val_metric = float(np.mean(val_metrics)) if val_metrics else float("-inf")
        history["train_loss"].append(mean_train_loss)
        history["val_metric"].append(mean_val_metric)
        if mean_val_metric > best_val_metric:
            best_val_metric = mean_val_metric
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "epochs_completed": epochs,
        "history": history,
    }


def run_happyquokka_smoke(config: dict, dataset_dir: Path, dataset_id: str, subject_id: str, sampling_rate: int, device: str) -> ModelSmokeResult:
    hq_cfg = config["happyquokka"]
    if Decoder is None:
        return make_failure_result(
            dataset_id=dataset_id,
            subject_id=subject_id,
            model_name="happyquokka",
            seed=int(config["seed"]),
            status="skipped_with_reason",
            failure_reason="HappyQuokka upstream dependency could not be imported",
            notes="`models.FFT_block.Decoder` was unavailable in the current environment.",
            selected_hyperparameters={"win_len_seconds": int(hq_cfg["win_len_seconds"])},
        )

    input_length = int(hq_cfg["win_len_seconds"]) * int(hq_cfg["sample_rate"])
    first_test = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    if first_test[1].shape[0] < input_length:
        return make_failure_result(
            dataset_id=dataset_id,
            subject_id=subject_id,
            model_name="happyquokka",
            seed=int(config["seed"]),
            status="skipped_with_reason",
            failure_reason="P00 recordings are shorter than HappyQuokka 10-second chunk length",
            notes="Current chunk contract cannot produce any valid evaluation window.",
            selected_hyperparameters={"input_length": input_length},
        )

    model, summary = train_happyquokka_subject_specific(
        input_dir=dataset_dir,
        participant=subject_id,
        device=device,
        input_length=input_length,
        batch_size=int(hq_cfg["batch_size"]),
        epochs=int(hq_cfg["max_epochs"]),
        learning_rate=float(hq_cfg["learning_rate"]),
        dropout=float(hq_cfg["dropout"]),
        lamda=float(hq_cfg["lamda"]),
    )

    first_eeg = torch.from_numpy(first_test[1][:input_length].astype(np.float32)).unsqueeze(0).to(device)
    first_sub_id = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        first_raw = model(first_eeg, first_sub_id).detach().cpu().numpy()

    windows: list[WindowPrediction] = []
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        recording_windows: list[WindowPrediction] = []
        for start in range(0, eeg.shape[0] - input_length + 1, input_length):
            eeg_chunk = torch.from_numpy(eeg[start : start + input_length].astype(np.float32)).unsqueeze(0).to(device)
            sub_id = torch.zeros(1, dtype=torch.long, device=device)
            with torch.no_grad():
                pred = model(eeg_chunk, sub_id).squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            target = env[start : start + input_length].astype(np.float32)
            recording_windows.append(
                WindowPrediction(
                    dataset=dataset_id,
                    model="happyquokka",
                    task="reconstruction",
                    protocol=config["protocol"],
                    seed=int(config["seed"]),
                    subject_id=subject_id,
                    recording_id=recording_id,
                    sampling_rate=sampling_rate,
                    checkpoint_id=f"happyquokka_epoch_{summary['best_epoch']}",
                    recording_length=int(len(env)),
                    start_index=int(start),
                    prediction=pred,
                    target=target,
                )
            )
        windows.extend(recording_windows)

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    subject_metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    selected_hyperparameters = {
        "input_length": input_length,
        "batch_size": int(hq_cfg["batch_size"]),
        "max_epochs": int(hq_cfg["max_epochs"]),
        "learning_rate": float(hq_cfg["learning_rate"]),
        "dropout": float(hq_cfg["dropout"]),
        "lamda": float(hq_cfg["lamda"]),
        "g_con": False,
        "best_epoch": int(summary["best_epoch"]),
    }
    first_postprocessed_shape = (first_test[1].shape[0],)
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "happyquokka",
        "seed": int(config["seed"]),
        "status": "success",
        "input_shape": shape_text(tuple(first_eeg.shape)),
        "raw_output_shape": shape_text(tuple(first_raw.shape)),
        "postprocessed_prediction_shape": shape_text(first_postprocessed_shape),
        "target_shape": shape_text(first_postprocessed_shape),
        "scorer_input_shape": shape_text(first_postprocessed_shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "subject_metric": subject_metric,
        "recording_count": len(recording_rows),
        "failure_reason": "",
        "notes": "HappyQuokka smoke uses non-overlapping 10-second chunks and aggregates them back to recording-level scorer input.",
    }
    model_run_entry = dict(matrix_row)
    model_run_entry["best_val_score"] = float(summary["best_val_metric"])
    shape_markdown = "\n".join(
        [
            "## `happyquokka`",
            "",
            "- status: `success`",
            f"- input_shape: `{shape_text(tuple(first_eeg.shape))}`",
            f"- raw_output_shape: `{shape_text(tuple(first_raw.shape))}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_postprocessed_shape)}`",
            f"- target_shape: `{shape_text(first_postprocessed_shape)}`",
            f"- scorer_input_shape: `{shape_text(first_postprocessed_shape)}`",
            f"- selected_hyperparameters: `{json.dumps(selected_hyperparameters, sort_keys=True)}`",
            f"- best_val_score: `{summary['best_val_metric']:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, recording_rows, subject_rows, None, shape_markdown)


def run_model(
    *,
    model_name: str,
    config: dict,
    dataset_dir: Path,
    dataset_id: str,
    subject_id: str,
    sampling_rate: int,
    device: str,
) -> ModelSmokeResult:
    try:
        if model_name == "linear":
            return run_linear_smoke(config, dataset_dir, dataset_id, subject_id, sampling_rate)
        if model_name == "lasso":
            return run_lasso_smoke(config, dataset_dir, dataset_id, subject_id, sampling_rate)
        if model_name == "elasticnet":
            return run_elasticnet_smoke(config, dataset_dir, dataset_id, subject_id, sampling_rate)
        if model_name == "vlaai":
            return run_vlaai_smoke(config, dataset_dir, dataset_id, subject_id, sampling_rate, device)
        if model_name == "happyquokka":
            return run_happyquokka_smoke(config, dataset_dir, dataset_id, subject_id, sampling_rate, device)
        return make_failure_result(
            dataset_id=dataset_id,
            subject_id=subject_id,
            model_name=model_name,
            seed=int(config["seed"]),
            status="skipped_with_reason",
            failure_reason=f"unknown smoke model {model_name}",
            notes="Model is not part of this smoke runner.",
        )
    except Exception as exc:
        return make_failure_result(
            dataset_id=dataset_id,
            subject_id=subject_id,
            model_name=model_name,
            seed=int(config["seed"]),
            status="failed",
            failure_reason=repr(exc),
            notes="Runtime exception during local model smoke.",
        )


def build_schema_validation(
    *,
    expected_models: list[str],
    matrix_rows: list[dict[str, object]],
    subject_rows: list[object],
    recording_rows: list[object],
    model_run_entries: list[dict[str, object]],
) -> dict[str, object]:
    status_by_model = {str(row["model"]): str(row["status"]) for row in matrix_rows}
    success_models = sorted(model for model, status in status_by_model.items() if status == "success")
    subject_models = sorted({row.model for row in subject_rows})
    recording_models = sorted({row.model for row in recording_rows})
    run_entry_models = sorted({str(row["model"]) for row in model_run_entries if str(row["status"]) == "success"})

    checks = {
        "all_requested_models_recorded": sorted(status_by_model.keys()) == sorted(expected_models),
        "success_models_have_subject_metrics": sorted(success_models) == subject_models,
        "success_models_have_recording_metrics": sorted(success_models) == recording_models,
        "success_models_have_run_entries": sorted(success_models) == run_entry_models,
        "all_non_success_models_have_reason": all(
            str(row["status"]) == "success" or str(row["failure_reason"]).strip() != "" for row in matrix_rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "status_by_model": status_by_model,
        "success_models": success_models,
        "subject_metric_rows": len(subject_rows),
        "recording_metric_rows": len(recording_rows),
    }


def write_shape_audit(path: Path, config: dict, matrix_rows: list[dict[str, object]], sections: list[str]) -> None:
    lines = [
        "# Adapter Shape Audit",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- subject: `{config['dataset']['subject_id']}`",
        f"- seed: `{config['seed']}`",
        "",
        "## Status Overview",
    ]
    for row in matrix_rows:
        lines.append(f"- `{row['model']}`: `{row['status']}`")
    lines.append("")
    lines.extend(sections)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_summary(path: Path, config: dict, matrix_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Subject-Specific Local Model Smoke v1",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- subject: `{config['dataset']['subject_id']}`",
        f"- seed: `{config['seed']}`",
        "",
        "| model | status | subject_metric | notes |",
        "| --- | --- | ---: | --- |",
    ]
    for row in matrix_rows:
        metric = row["subject_metric"] if row["subject_metric"] != "" else "n/a"
        lines.append(f"| {row['model']} | {row['status']} | {metric} | {row['notes']} |")
    lines.extend(
        [
            "",
            "## Scope Notes",
            "- This is a local candidate-model adapter smoke matrix, not a full benchmark.",
            "- Accepted subject-specific models (`ridge / cca / fcnn / dnn / cnn / eegnet / adt`) were intentionally not rerun here.",
            "- `decaf` remains external-only and was not integrated in this round.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_next_steps(path: Path, matrix_rows: list[dict[str, object]]) -> None:
    success_models = [str(row["model"]) for row in matrix_rows if str(row["status"]) == "success"]
    failed_models = [str(row["model"]) for row in matrix_rows if str(row["status"]) == "failed"]
    skipped_models = [str(row["model"]) for row in matrix_rows if str(row["status"]) == "skipped_with_reason"]
    lines = [
        "# Recommended Next Steps",
        "",
        f"- successful local smoke models: `{', '.join(success_models) if success_models else 'none'}`",
        f"- failed local smoke models: `{', '.join(failed_models) if failed_models else 'none'}`",
        f"- skipped local smoke models: `{', '.join(skipped_models) if skipped_models else 'none'}`",
        "",
        "## Suggested follow-up order",
        "1. Promote successful smoke adapters to a small multi-subject closure before any wider benchmark run.",
        "2. If `elasticnet` failed, isolate whether the issue is solver stability or interface mismatch before expanding the alpha/l1 grid.",
        "3. If `happyquokka` succeeded, decide whether the 10-second chunk contract is scientifically acceptable as a separate model family rather than forcing 50-sample parity.",
        "4. Keep `decaf` on a separate external-integration branch.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    dataset_cfg = config["dataset"]
    dataset_id = str(dataset_cfg["dataset_id"])
    dataset_dir = resolve_dataset_path(str(dataset_cfg["dataset_locator"]))
    subject_id = str(dataset_cfg["subject_id"])
    sampling_rate = int(dataset_cfg["sampling_rate"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    print_event("STARTUP", f"config={args.config}")
    print_event("STARTUP", f"output_dir={repo_relative(output_dir)} dataset={dataset_id} subject={subject_id} seed={config['seed']} device={device}")
    print_event("STARTUP", f"models={', '.join(config['models'])}")

    matrix_rows: list[dict[str, object]] = []
    model_run_entries: list[dict[str, object]] = []
    failure_entries: list[dict[str, object]] = []
    all_recording_rows: list[object] = []
    all_subject_rows: list[object] = []
    shape_sections: list[str] = []

    for model_name in config["models"]:
        print_event("MODEL START", str(model_name))
        result = run_model(
            model_name=str(model_name),
            config=config,
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
            subject_id=subject_id,
            sampling_rate=sampling_rate,
            device=device,
        )
        matrix_rows.append(result.matrix_row)
        model_run_entries.append(result.model_run_entry)
        all_recording_rows.extend(result.recording_rows)
        all_subject_rows.extend(result.subject_rows)
        shape_sections.append(result.shape_markdown)
        if result.failure_entry is not None:
            failure_entries.append(result.failure_entry)
            print_event("MODEL DONE", f"{model_name} status={result.failure_entry['status']} reason={result.failure_entry['failure_reason']}")
        else:
            print_event("MODEL DONE", f"{model_name} status=success metric={result.matrix_row['subject_metric']}")

    write_csv_rows(output_dir / "adapter_smoke_matrix.csv", matrix_rows, ADAPTER_MATRIX_FIELDS)
    write_rows(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
    write_json(output_dir / "model_run_entries.json", model_run_entries)
    write_json(output_dir / "failure_report.json", {"failures": failure_entries})

    schema_validation = build_schema_validation(
        expected_models=[str(model_name) for model_name in config["models"]],
        matrix_rows=matrix_rows,
        subject_rows=all_subject_rows,
        recording_rows=all_recording_rows,
        model_run_entries=model_run_entries,
    )
    write_json(output_dir / "schema_validation_report.json", schema_validation)
    write_shape_audit(output_dir / "adapter_shape_audit.md", config, matrix_rows, shape_sections)
    write_result_summary(output_dir / "result_summary.md", config, matrix_rows)
    write_recommended_next_steps(output_dir / "recommended_next_steps.md", matrix_rows)

    run_manifest = {
        "protocol": config["protocol"],
        "dataset": dataset_id,
        "subject_id": subject_id,
        "seed": int(config["seed"]),
        "device": device,
        "models_requested": list(config["models"]),
        "artifacts": {
            "adapter_smoke_matrix": repo_relative(output_dir / "adapter_smoke_matrix.csv"),
            "adapter_shape_audit": repo_relative(output_dir / "adapter_shape_audit.md"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "result_summary": repo_relative(output_dir / "result_summary.md"),
            "recommended_next_steps": repo_relative(output_dir / "recommended_next_steps.md"),
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    print_event("DONE", f"schema_passed={schema_validation['passed']} successes={len([row for row in matrix_rows if row['status'] == 'success'])} failures={len(failure_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
