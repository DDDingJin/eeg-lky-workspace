from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from torch.optim import Adam
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")
SHARED_HAPPYQUOKKA = SHARED_WORKSPACE / "external" / "upstream" / "HappyQuokka_system_for_EEG_Challenge"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SHARED_HAPPYQUOKKA.exists() and str(SHARED_HAPPYQUOKKA) not in sys.path:
    sys.path.insert(0, str(SHARED_HAPPYQUOKKA))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.happyquokka_reference import (
    l1_loss,
    pearson_loss,
    pearson_metric,
)
from repro.mldecoders.cca import trim_valid_range
from repro.mldecoders.models import VLAAIExactOfficialRegressor
from repro.reference_baselines import (
    ReferenceWindowDataset,
    concatenate_reference_split,
    load_reference_recordings,
    train_dnn_reference_logged,
)
from run_gate0_gate2_subject_specific_local_model_smoke_v1 import (
    Decoder,
    ModelSmokeResult,
    SubjectSpecificHappyQuokkaTrainDataset,
    cap_fit_samples,
    correlation,
    ensure_dir,
    load_config,
    print_event,
    repo_relative,
    resolve_dataset_path,
    shape_text,
    write_csv_rows,
    write_json,
)


COMPARISON_FIELDS = [
    "subject_id",
    "model",
    "family",
    "metric_value",
    "reference_source",
    "checkpoint_id",
]


DIAGNOSTIC_MATRIX_FIELDS = [
    "dataset",
    "subject_id",
    "model",
    "seed",
    "status",
    "model_family_contract",
    "full_eval_not_capped",
    "input_shape",
    "raw_output_shape",
    "postprocessed_prediction_shape",
    "target_shape",
    "scorer_input_shape",
    "selected_hyperparameters",
    "train_fit_samples",
    "val_fit_samples",
    "subject_metric",
    "recording_count",
    "failure_reason",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


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


def make_failure_result(
    *,
    dataset_id: str,
    subject_id: str,
    model_name: str,
    seed: int,
    status: str,
    failure_reason: str,
    notes: str,
    model_family_contract: str,
    selected_hyperparameters: dict[str, object] | None = None,
) -> ModelSmokeResult:
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": model_name,
        "seed": seed,
        "status": status,
        "model_family_contract": model_family_contract,
        "full_eval_not_capped": False,
        "input_shape": "",
        "raw_output_shape": "",
        "postprocessed_prediction_shape": "",
        "target_shape": "",
        "scorer_input_shape": "",
        "selected_hyperparameters": json.dumps(selected_hyperparameters or {}, sort_keys=True),
        "train_fit_samples": "",
        "val_fit_samples": "",
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
        model_run_entry=deepcopy(matrix_row),
        recording_rows=[],
        subject_rows=[],
        failure_entry=failure_entry,
        shape_markdown="\n".join(
            [
                f"## `{model_name}`",
                "",
                f"- status: `{status}`",
                f"- model_family_contract: `{model_family_contract}`",
                f"- failure_reason: `{failure_reason}`",
                f"- notes: {notes}",
            ]
        )
        + "\n",
    )


def build_diag_recording_rows(
    *,
    dataset_id: str,
    model_name: str,
    protocol: str,
    seed: int,
    subject_id: str,
    checkpoint_id: str,
    model_family_contract: str,
    rows: list[object],
    diagnostics_by_recording: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        diag = diagnostics_by_recording[row.recording_id]
        output.append(
            {
                "dataset": dataset_id,
                "model": model_name,
                "task": "reconstruction",
                "protocol": protocol,
                "seed": seed,
                "subject_id": subject_id,
                "recording_id": row.recording_id,
                "sampling_rate": row.sampling_rate,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "num_valid_samples": row.num_valid_samples,
                "checkpoint_id": checkpoint_id,
                "artifact_scope": diag["artifact_scope"],
                "recording_length": diag["recording_length"],
                "coverage_ratio": diag["coverage_ratio"],
                "dropped_samples_reason": diag["dropped_samples_reason"],
                "full_eval_not_capped": diag["full_eval_not_capped"],
                "model_family_contract": model_family_contract,
            }
        )
    return output


def run_linear_family_full_eval(
    *,
    model_name: str,
    config: dict,
    dataset_dir: Path,
    dataset_id: str,
    subject_id: str,
    sampling_rate: int,
) -> ModelSmokeResult:
    model_cfg = config[model_name]
    fit_budget = config["fit_budget"]
    start_lag = int(model_cfg["start_lag"])
    end_lag = int(model_cfg["end_lag"])
    x_train, y_train = concatenate_reference_split(dataset_dir, "train", subject_id, channels=range(64))
    x_val, y_val = concatenate_reference_split(dataset_dir, "val", subject_id, channels=range(64))
    x_train_lag_full, _, y_train_center_full = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag_full, _, y_val_center_full = trim_valid_range(x_val, y_val, start_lag, end_lag)
    x_train_lag, y_train_center = cap_fit_samples(x_train_lag_full, y_train_center_full, int(fit_budget["max_fit_samples_per_split"]))
    x_val_lag, y_val_center = cap_fit_samples(x_val_lag_full, y_val_center_full, int(fit_budget["max_fit_samples_per_split"]))

    best_model = None
    best_summary = None
    if model_name == "linear":
        model = LinearRegression()
        model.fit(x_train_lag, y_train_center)
        val_pred = model.predict(x_val_lag).astype(np.float32)
        best_model = model
        best_summary = {"best_val_score": correlation(val_pred, y_val_center.astype(np.float32))}
    elif model_name == "lasso":
        for alpha in model_cfg["alphas"]:
            model = Lasso(alpha=float(alpha), max_iter=int(model_cfg["max_iter"]))
            model.fit(x_train_lag, y_train_center)
            val_pred = model.predict(x_val_lag).astype(np.float32)
            val_score = correlation(val_pred, y_val_center.astype(np.float32))
            if best_summary is None or val_score > float(best_summary["best_val_score"]):
                best_model = model
                best_summary = {"best_alpha": float(alpha), "best_val_score": float(val_score)}
    elif model_name == "elasticnet":
        for alpha in model_cfg["alphas"]:
            for l1_ratio in model_cfg["l1_ratios"]:
                model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=int(model_cfg["max_iter"]))
                model.fit(x_train_lag, y_train_center)
                val_pred = model.predict(x_val_lag).astype(np.float32)
                val_score = correlation(val_pred, y_val_center.astype(np.float32))
                if best_summary is None or val_score > float(best_summary["best_val_score"]):
                    best_model = model
                    best_summary = {
                        "best_alpha": float(alpha),
                        "best_l1_ratio": float(l1_ratio),
                        "best_val_score": float(val_score),
                    }
    else:
        raise ValueError(model_name)

    if best_model is None or best_summary is None:
        raise RuntimeError(f"{model_name} failed to select a valid model")

    offset = end_lag - 1
    checkpoint_id = (
        "linear_ols"
        if model_name == "linear"
        else f"{model_name}_alpha_{best_summary['best_alpha']}"
        if model_name == "lasso"
        else f"elasticnet_alpha_{best_summary['best_alpha']}_l1_{best_summary['best_l1_ratio']}"
    )
    windows: list[WindowPrediction] = []
    diagnostics_by_recording: dict[str, dict[str, object]] = {}
    first_recording = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    first_x_lag, _, first_y_center = trim_valid_range(first_recording[1], first_recording[2], start_lag, end_lag)
    first_pred = best_model.predict(first_x_lag).astype(np.float32)

    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
        pred = best_model.predict(x_lag).astype(np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model=model_name,
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
        diagnostics_by_recording[recording_id] = {
            "artifact_scope": config["artifact_scope"],
            "recording_length": int(len(env)),
            "coverage_ratio": float(len(y_center) / len(env)),
            "dropped_samples_reason": f"lag range [{start_lag}, {end_lag}) trims edges",
            "full_eval_not_capped": True,
        }

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    diag_recording_rows = build_diag_recording_rows(
        dataset_id=dataset_id,
        model_name=model_name,
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=subject_id,
        checkpoint_id=checkpoint_id,
        model_family_contract="lag_matrix_trf",
        rows=recording_rows,
        diagnostics_by_recording=diagnostics_by_recording,
    )
    subject_metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    selected_hyperparameters = {
        "start_lag": start_lag,
        "end_lag": end_lag,
        "train_fit_samples": int(x_train_lag.shape[0]),
        "val_fit_samples": int(x_val_lag.shape[0]),
    }
    if "best_alpha" in best_summary:
        selected_hyperparameters["alpha"] = best_summary["best_alpha"]
    if "best_l1_ratio" in best_summary:
        selected_hyperparameters["l1_ratio"] = best_summary["best_l1_ratio"]

    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": model_name,
        "seed": int(config["seed"]),
        "status": "success",
        "model_family_contract": "lag_matrix_trf",
        "full_eval_not_capped": True,
        "input_shape": shape_text(first_x_lag.shape),
        "raw_output_shape": shape_text(first_pred.shape),
        "postprocessed_prediction_shape": shape_text(first_pred.shape),
        "target_shape": shape_text(first_y_center.shape),
        "scorer_input_shape": shape_text(first_pred.shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "train_fit_samples": int(x_train_lag.shape[0]),
        "val_fit_samples": int(x_val_lag.shape[0]),
        "subject_metric": subject_metric,
        "recording_count": len(diag_recording_rows),
        "failure_reason": "",
        "notes": f"best_val_score={best_summary['best_val_score']:.6f}",
    }
    model_run_entry = deepcopy(matrix_row)
    model_run_entry["checkpoint_id"] = checkpoint_id
    model_run_entry["best_val_score"] = best_summary["best_val_score"]
    shape_markdown = "\n".join(
        [
            f"## `{model_name}`",
            "",
            "- status: `success`",
            "- model_family_contract: `lag_matrix_trf`",
            f"- input_shape: `{shape_text(first_x_lag.shape)}`",
            f"- raw_output_shape: `{shape_text(first_pred.shape)}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_pred.shape)}`",
            f"- target_shape: `{shape_text(first_y_center.shape)}`",
            f"- scorer_input_shape: `{shape_text(first_pred.shape)}`",
            f"- train_fit_samples: `{x_train_lag.shape[0]}`",
            f"- val_fit_samples: `{x_val_lag.shape[0]}`",
            f"- best_val_score: `{best_summary['best_val_score']:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, diag_recording_rows, subject_rows, None, shape_markdown)


def run_vlaai_full_eval(config: dict, dataset_dir: Path, dataset_id: str, subject_id: str, sampling_rate: int, device: str) -> ModelSmokeResult:
    model_cfg = config["vlaai"]
    model_kwargs = {"num_input_channels": 64, "input_length": int(model_cfg["window_size"])}
    train_result = train_dnn_reference_logged(
        dataset_dir,
        subject_id,
        VLAAIExactOfficialRegressor,
        model_kwargs,
        epochs=int(model_cfg["max_epochs"]),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
        batch_size=int(model_cfg["batch_size"]),
        early_stopping_patience=int(model_cfg["early_stopping_patience"]),
        device=device,
        seed=int(config["seed"]),
        channels=range(64),
    )
    model = VLAAIExactOfficialRegressor(**model_kwargs).to(device)
    model.load_state_dict(train_result.state_dict)
    model.eval()

    input_length = int(model_cfg["window_size"])
    offset = input_length - 1
    first_recording = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    first_eeg = torch.from_numpy(first_recording[1][:input_length].astype(np.float32)).T.unsqueeze(0).to(device)
    with torch.no_grad():
        first_raw = model(first_eeg).detach().cpu().numpy()

    windows: list[WindowPrediction] = []
    diagnostics_by_recording: dict[str, dict[str, object]] = {}
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        preds: list[float] = []
        targets: list[float] = []
        eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
        with torch.no_grad():
            for start in range(0, eeg.shape[0] - input_length + 1):
                batch = eeg_tensor[start : start + input_length].T.unsqueeze(0).to(device)
                preds.append(float(model(batch).item()))
                targets.append(float(env[start + input_length - 1]))
        pred_arr = np.asarray(preds, dtype=np.float32)
        target_arr = np.asarray(targets, dtype=np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="vlaai",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=f"vlaai_epoch_{train_result.best_epoch}",
                full_length=len(env),
                offset=offset,
                prediction=pred_arr,
                target=target_arr,
            )
        )
        diagnostics_by_recording[recording_id] = {
            "artifact_scope": config["artifact_scope"],
            "recording_length": int(len(env)),
            "coverage_ratio": float(len(target_arr) / len(env)),
            "dropped_samples_reason": f"window_size={input_length} drops first {offset} target positions",
            "full_eval_not_capped": True,
        }

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    diag_recording_rows = build_diag_recording_rows(
        dataset_id=dataset_id,
        model_name="vlaai",
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=subject_id,
        checkpoint_id=f"vlaai_epoch_{train_result.best_epoch}",
        model_family_contract="vlaai_local_adapter",
        rows=recording_rows,
        diagnostics_by_recording=diagnostics_by_recording,
    )
    subject_metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    selected_hyperparameters = {
        "window_size": input_length,
        "batch_size": int(model_cfg["batch_size"]),
        "max_epochs": int(model_cfg["max_epochs"]),
        "learning_rate": float(model_cfg["learning_rate"]),
        "weight_decay": float(model_cfg["weight_decay"]),
        "best_epoch": int(train_result.best_epoch),
    }
    first_post_shape = (first_recording[1].shape[0] - input_length + 1,)
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "vlaai",
        "seed": int(config["seed"]),
        "status": "success",
        "model_family_contract": "vlaai_local_adapter",
        "full_eval_not_capped": True,
        "input_shape": shape_text(tuple(first_eeg.shape)),
        "raw_output_shape": shape_text(tuple(first_raw.shape)),
        "postprocessed_prediction_shape": shape_text(first_post_shape),
        "target_shape": shape_text(first_post_shape),
        "scorer_input_shape": shape_text(first_post_shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "train_fit_samples": "",
        "val_fit_samples": "",
        "subject_metric": subject_metric,
        "recording_count": len(diag_recording_rows),
        "failure_reason": "",
        "notes": f"best_val_score={train_result.best_val_score:.6f}",
    }
    model_run_entry = deepcopy(matrix_row)
    model_run_entry["checkpoint_id"] = f"vlaai_epoch_{train_result.best_epoch}"
    model_run_entry["best_val_score"] = float(train_result.best_val_score)
    shape_markdown = "\n".join(
        [
            "## `vlaai`",
            "",
            "- status: `success`",
            "- model_family_contract: `vlaai_local_adapter`",
            f"- input_shape: `{shape_text(tuple(first_eeg.shape))}`",
            f"- raw_output_shape: `{shape_text(tuple(first_raw.shape))}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_post_shape)}`",
            f"- target_shape: `{shape_text(first_post_shape)}`",
            f"- scorer_input_shape: `{shape_text(first_post_shape)}`",
            f"- best_val_score: `{train_result.best_val_score:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, diag_recording_rows, subject_rows, None, shape_markdown)


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

    best_state = deepcopy(model.state_dict())
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
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "epochs_completed": epochs,
        "history": history,
    }


def run_happyquokka_full_eval(config: dict, dataset_dir: Path, dataset_id: str, subject_id: str, sampling_rate: int, device: str) -> ModelSmokeResult:
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
            model_family_contract="10s_chunk",
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
            model_family_contract="10s_chunk",
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
    diagnostics_by_recording: dict[str, dict[str, object]] = {}
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        for start in range(0, eeg.shape[0] - input_length + 1, input_length):
            eeg_chunk = torch.from_numpy(eeg[start : start + input_length].astype(np.float32)).unsqueeze(0).to(device)
            sub_id = torch.zeros(1, dtype=torch.long, device=device)
            with torch.no_grad():
                pred = model(eeg_chunk, sub_id).squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            target = env[start : start + input_length].astype(np.float32)
            windows.append(
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
        valid_chunks = (eeg.shape[0] // input_length)
        diagnostics_by_recording[recording_id] = {
            "artifact_scope": config["artifact_scope"],
            "recording_length": int(len(env)),
            "coverage_ratio": float((valid_chunks * input_length) / len(env)),
            "dropped_samples_reason": "tail shorter than 10s chunk",
            "full_eval_not_capped": True,
        }

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    diag_recording_rows = build_diag_recording_rows(
        dataset_id=dataset_id,
        model_name="happyquokka",
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=subject_id,
        checkpoint_id=f"happyquokka_epoch_{summary['best_epoch']}",
        model_family_contract="10s_chunk",
        rows=recording_rows,
        diagnostics_by_recording=diagnostics_by_recording,
    )
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
    first_post_shape = (first_test[1].shape[0] // input_length * input_length,)
    matrix_row = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "happyquokka",
        "seed": int(config["seed"]),
        "status": "success",
        "model_family_contract": "10s_chunk",
        "full_eval_not_capped": True,
        "input_shape": shape_text(tuple(first_eeg.shape)),
        "raw_output_shape": shape_text(tuple(first_raw.shape)),
        "postprocessed_prediction_shape": shape_text(first_post_shape),
        "target_shape": shape_text(first_post_shape),
        "scorer_input_shape": shape_text(first_post_shape),
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True),
        "train_fit_samples": "",
        "val_fit_samples": "",
        "subject_metric": subject_metric,
        "recording_count": len(diag_recording_rows),
        "failure_reason": "",
        "notes": "HappyQuokka evaluated on all available 10-second chunks from each P00 test recording.",
    }
    model_run_entry = deepcopy(matrix_row)
    model_run_entry["checkpoint_id"] = f"happyquokka_epoch_{summary['best_epoch']}"
    model_run_entry["best_val_score"] = float(summary["best_val_metric"])
    shape_markdown = "\n".join(
        [
            "## `happyquokka`",
            "",
            "- status: `success`",
            "- model_family_contract: `10s_chunk`",
            f"- input_shape: `{shape_text(tuple(first_eeg.shape))}`",
            f"- raw_output_shape: `{shape_text(tuple(first_raw.shape))}`",
            f"- postprocessed_prediction_shape: `{shape_text(first_post_shape)}`",
            f"- target_shape: `{shape_text(first_post_shape)}`",
            f"- scorer_input_shape: `{shape_text(first_post_shape)}`",
            f"- best_val_score: `{summary['best_val_metric']:.6f}`",
            f"- subject_metric: `{subject_metric:.6f}`",
        ]
    ) + "\n"
    return ModelSmokeResult(matrix_row, model_run_entry, diag_recording_rows, subject_rows, None, shape_markdown)


def load_accepted_p00_reference(config: dict) -> list[dict[str, object]]:
    source_cfg = config["reference_metrics_source"]
    csv_path = ROOT / source_cfg["subject_metrics_csv"]
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    accepted_models = set(source_cfg["accepted_models"])
    filtered = [
        row
        for row in rows
        if row["dataset"] == source_cfg["dataset_id"]
        and row["subject_id"] == source_cfg["subject_id"]
        and row["seed"] == str(source_cfg["seed"])
        and row["model"] in accepted_models
    ]
    filtered.sort(key=lambda row: row["model"])
    return [
        {
            "subject_id": row["subject_id"],
            "model": row["model"],
            "family": "accepted_reference",
            "metric_value": float(row["metric_value"]),
            "reference_source": source_cfg["subject_metrics_csv"],
            "checkpoint_id": row["checkpoint_id"],
        }
        for row in filtered
    ]


def build_schema_validation(
    *,
    matrix_rows: list[dict[str, object]],
    subject_rows: list[object],
    recording_rows: list[dict[str, object]],
    failure_entries: list[dict[str, object]],
) -> dict[str, object]:
    success_models = sorted(str(row["model"]) for row in matrix_rows if str(row["status"]) == "success")
    subject_models = sorted(row.model for row in subject_rows)
    recording_models = sorted({str(row["model"]) for row in recording_rows})
    checks = {
        "subject_metrics_match_success_models": success_models == subject_models,
        "recording_metrics_match_success_models": success_models == recording_models,
        "every_recording_has_num_valid_samples": all(int(row["num_valid_samples"]) >= 0 for row in recording_rows),
        "every_recording_has_coverage_ratio": all("coverage_ratio" in row and float(row["coverage_ratio"]) >= 0.0 for row in recording_rows),
        "every_model_full_eval_not_capped": all(bool(row["full_eval_not_capped"]) for row in matrix_rows if str(row["status"]) == "success"),
        "happyquokka_marked_10s_chunk": all(
            str(row["model"]) != "happyquokka" or str(row["model_family_contract"]) == "10s_chunk" for row in matrix_rows
        ),
        "failure_report_empty_or_reasoned": len(failure_entries) == 0 or all(str(item.get("failure_reason", "")).strip() != "" for item in failure_entries),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "success_models": success_models,
        "subject_metric_rows": len(subject_rows),
        "recording_metric_rows": len(recording_rows),
        "failure_count": len(failure_entries),
    }


def write_full_eval_diagnostic(path: Path, matrix_rows: list[dict[str, object]], recording_rows: list[dict[str, object]]) -> None:
    lines = ["# Full Evaluation Diagnostic", ""]
    for row in matrix_rows:
        model = row["model"]
        lines.append(f"## `{model}`")
        lines.append("")
        lines.append(f"- status: `{row['status']}`")
        lines.append(f"- contract: `{row['model_family_contract']}`")
        lines.append(f"- full_eval_not_capped: `{row['full_eval_not_capped']}`")
        model_recordings = [item for item in recording_rows if item["model"] == model]
        if model_recordings:
            coverages = [float(item["coverage_ratio"]) for item in model_recordings]
            valid_samples = [int(item["num_valid_samples"]) for item in model_recordings]
            lines.append(f"- num_valid_samples range: `{min(valid_samples)} .. {max(valid_samples)}`")
            lines.append(f"- coverage_ratio range: `{min(coverages):.6f} .. {max(coverages):.6f}`")
            reasons = sorted({str(item['dropped_samples_reason']) for item in model_recordings})
            lines.append(f"- dropped_samples_reason: `{'; '.join(reasons)}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def write_result_summary(path: Path, config: dict, matrix_rows: list[dict[str, object]], comparison_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Subject-Specific Local Model Full-Eval P00 v1",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- subject: `{config['dataset']['subject_id']}`",
        f"- seed: `{config['seed']}`",
        "",
        "| model | family | metric | note |",
        "| --- | --- | ---: | --- |",
    ]
    for row in sorted(comparison_rows, key=lambda item: (item["family"], item["model"])):
        note = "accepted reference" if row["family"] == "accepted_reference" else "local full-eval"
        lines.append(f"| {row['model']} | {row['family']} | {row['metric_value']} | {note} |")
    lines.extend(
        [
            "",
            "## Scope Notes",
            "- Test evaluation is no longer capped to 256 windows/points.",
            "- Train/validation fitting may still be subsampled for local diagnostic runtime, but test evaluation covers the full available P00 test recording contract for each model family.",
            "- HappyQuokka remains a `10s_chunk` family and is not forced into the 50-sample window contract.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_next_steps(path: Path, matrix_rows: list[dict[str, object]], accepted_rows: list[dict[str, object]]) -> None:
    accepted_lookup = {row["model"]: row["metric_value"] for row in accepted_rows}
    lines = ["# Recommended Next Steps", ""]
    for row in matrix_rows:
        model = str(row["model"])
        status = str(row["status"])
        metric = row["subject_metric"]
        lines.append(f"- `{model}`: status=`{status}`, metric=`{metric}`")
        if model in accepted_lookup:
            lines.append(f"  accepted reference does not apply directly; local candidate is compared against `{accepted_lookup[model]}` only when model names overlap.")
    lines.extend(
        [
            "",
            "## Suggested promotion criteria",
            "1. Promote models with stable full-eval coverage and non-pathological metrics to a small multi-subject local candidate closure.",
            "2. Keep HappyQuokka as a distinct 10-second-chunk family in any future expansion.",
            "3. Revisit train-fit subsampling only after P00 full-eval diagnostics are accepted.",
        ]
    )
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

    matrix_rows: list[dict[str, object]] = []
    recording_rows: list[dict[str, object]] = []
    subject_rows: list[object] = []
    model_run_entries: list[dict[str, object]] = []
    failure_entries: list[dict[str, object]] = []
    shape_sections: list[str] = []

    for model_name in config["models"]:
        print_event("MODEL START", str(model_name))
        try:
            if model_name in {"linear", "lasso", "elasticnet"}:
                result = run_linear_family_full_eval(
                    model_name=str(model_name),
                    config=config,
                    dataset_dir=dataset_dir,
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    sampling_rate=sampling_rate,
                )
            elif model_name == "vlaai":
                result = run_vlaai_full_eval(config, dataset_dir, dataset_id, subject_id, sampling_rate, device)
            elif model_name == "happyquokka":
                result = run_happyquokka_full_eval(config, dataset_dir, dataset_id, subject_id, sampling_rate, device)
            else:
                result = make_failure_result(
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    model_name=str(model_name),
                    seed=int(config["seed"]),
                    status="skipped_with_reason",
                    failure_reason=f"unknown model {model_name}",
                    notes="Not part of this diagnostic runner.",
                    model_family_contract="unknown",
                )
        except Exception as exc:
            result = make_failure_result(
                dataset_id=dataset_id,
                subject_id=subject_id,
                model_name=str(model_name),
                seed=int(config["seed"]),
                status="failed",
                failure_reason=repr(exc),
                notes="Runtime exception during full-eval diagnostic.",
                model_family_contract="unknown",
            )

        matrix_rows.append(result.matrix_row)
        recording_rows.extend(result.recording_rows)
        subject_rows.extend(result.subject_rows)
        model_run_entries.append(result.model_run_entry)
        shape_sections.append(result.shape_markdown)
        if result.failure_entry is not None:
            failure_entries.append(result.failure_entry)
            print_event("MODEL DONE", f"{model_name} status={result.failure_entry['status']} reason={result.failure_entry['failure_reason']}")
        else:
            print_event("MODEL DONE", f"{model_name} status=success metric={result.matrix_row['subject_metric']}")

    accepted_rows = load_accepted_p00_reference(config)
    comparison_rows = accepted_rows + [
        {
            "subject_id": subject_id,
            "model": row["model"],
            "family": "local_full_eval",
            "metric_value": row["subject_metric"],
            "reference_source": repo_relative(output_dir / "subject_metrics.csv"),
            "checkpoint_id": model_run_entry.get("checkpoint_id", ""),
        }
        for row, model_run_entry in zip(matrix_rows, model_run_entries)
        if row["status"] == "success"
    ]

    write_csv_rows(output_dir / "subject_metrics.csv", [row.__dict__ for row in subject_rows], SUBJECT_METRIC_FIELDS)
    write_csv_rows(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS + ["recording_length", "coverage_ratio", "dropped_samples_reason", "full_eval_not_capped", "model_family_contract"])
    write_csv_rows(output_dir / "accepted_p00_reference_metrics.csv", accepted_rows, COMPARISON_FIELDS)
    write_csv_rows(output_dir / "metric_comparison_p00.csv", comparison_rows, COMPARISON_FIELDS)
    write_csv_rows(output_dir / "full_eval_matrix.csv", matrix_rows, DIAGNOSTIC_MATRIX_FIELDS)
    write_json(output_dir / "model_run_entries.json", model_run_entries)
    write_json(output_dir / "failure_report.json", {"failures": failure_entries})

    schema_validation = build_schema_validation(
        matrix_rows=matrix_rows,
        subject_rows=subject_rows,
        recording_rows=recording_rows,
        failure_entries=failure_entries,
    )
    write_json(output_dir / "schema_validation_report.json", schema_validation)
    write_shape_audit(output_dir / "adapter_shape_audit.md", config, matrix_rows, shape_sections)
    write_full_eval_diagnostic(output_dir / "full_eval_diagnostic.md", matrix_rows, recording_rows)
    write_result_summary(output_dir / "result_summary.md", config, matrix_rows, comparison_rows)
    write_recommended_next_steps(output_dir / "recommended_next_steps.md", matrix_rows, accepted_rows)

    run_manifest = {
        "protocol": config["protocol"],
        "dataset": dataset_id,
        "subject_id": subject_id,
        "seed": int(config["seed"]),
        "device": device,
        "models_requested": list(config["models"]),
        "artifacts": {
            "full_eval_diagnostic": repo_relative(output_dir / "full_eval_diagnostic.md"),
            "adapter_shape_audit": repo_relative(output_dir / "adapter_shape_audit.md"),
            "accepted_p00_reference_metrics": repo_relative(output_dir / "accepted_p00_reference_metrics.csv"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "metric_comparison_p00": repo_relative(output_dir / "metric_comparison_p00.csv"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "result_summary": repo_relative(output_dir / "result_summary.md"),
            "recommended_next_steps": repo_relative(output_dir / "recommended_next_steps.md")
        }
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    print_event("DONE", f"schema_passed={schema_validation['passed']} successes={len([row for row in matrix_rows if row['status']=='success'])} failures={len(failure_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
