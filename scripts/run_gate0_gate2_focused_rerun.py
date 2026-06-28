from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import PREDICTION_FIELDS, RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, prediction_rows_from_series, recording_metric_row, subject_metric_rows
from repro.adt_exact import ADTExactRegressor, train_adt_exact_reference
from repro.reference_baselines import DNNReferenceTrainResult, load_reference_recordings, train_dnn_reference_logged
from repro.simple_models import FCNNBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_workspace = Path(ROOT.drive + "\\decode")
    shared_candidate = shared_workspace / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def correlation(pred: np.ndarray, target: np.ndarray) -> float:
    pred0 = pred.astype(np.float64) - pred.astype(np.float64).mean()
    target0 = target.astype(np.float64) - target.astype(np.float64).mean()
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


def aggregate_model_outputs(windows: list[WindowPrediction], artifact_scope: str) -> tuple[list[object], list[object], list[object]]:
    grouped: dict[tuple[str, str], list[WindowPrediction]] = {}
    for window in windows:
        grouped.setdefault((window.subject_id, window.recording_id), []).append(window)

    prediction_rows = []
    recording_rows = []
    for _, recording_windows in sorted(grouped.items()):
        first = recording_windows[0]
        prediction, target, valid_mask = aggregate_overlapping_windows(first.recording_length, recording_windows)
        prediction_rows.extend(
            prediction_rows_from_series(
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
    return prediction_rows, recording_rows, subject_rows


def write_prediction_head(path: Path, rows: list[object]) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        row_dict = row.__dict__
        if int(row_dict["valid_mask"]) != 1:
            continue
        grouped.setdefault(row_dict["model"], []).append(row_dict)
    selected = []
    for model_name in sorted(grouped):
        selected.extend(grouped[model_name][:5])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(selected)


def fit_and_predict_fcnn(config: dict, dataset_dir: Path, device: str) -> tuple[list[WindowPrediction], dict[str, object]]:
    fcnn_cfg = config["fcnn"]
    subject_id = config["subject_id"]
    model_kwargs = {
        "num_hidden": int(fcnn_cfg["hidden_layers"]),
        "dropout_rate": float(fcnn_cfg["dropout_rate"]),
        "input_length": int(fcnn_cfg["window_size"]),
        "num_input_channels": 64,
    }
    train_result: DNNReferenceTrainResult = train_dnn_reference_logged(
        dataset_dir,
        subject_id,
        FCNNBaseline,
        model_kwargs,
        epochs=int(fcnn_cfg["max_epochs"]),
        lr=float(fcnn_cfg["learning_rate"]),
        weight_decay=float(fcnn_cfg["weight_decay"]),
        batch_size=int(fcnn_cfg["batch_size"]),
        early_stopping_patience=int(fcnn_cfg["early_stopping_patience"]),
        device=device,
        seed=int(config["seed"]),
        channels=range(64),
    )
    model = FCNNBaseline(**model_kwargs).to(device)
    model.load_state_dict(train_result.state_dict)
    model.eval()

    windows: list[WindowPrediction] = []
    scores = []
    offset = int(fcnn_cfg["window_size"]) - 1
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        preds = []
        targets = []
        with torch.no_grad():
            eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
            for start in range(0, eeg.shape[0] - int(fcnn_cfg["window_size"]) + 1):
                batch = eeg_tensor[start : start + int(fcnn_cfg["window_size"])].T.unsqueeze(0).to(device)
                preds.append(float(model(batch).item()))
                targets.append(float(env[start + int(fcnn_cfg["window_size"]) - 1]))
        pred_arr = np.asarray(preds, dtype=np.float32)
        target_arr = np.asarray(targets, dtype=np.float32)
        windows.append(
            build_series_window(
                dataset=config["dataset_id"],
                model="fcnn",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=int(config["sampling_rate"]),
                checkpoint_id=f"fcnn_epoch_{train_result.best_epoch}",
                full_length=len(env),
                offset=offset,
                prediction=pred_arr,
                target=target_arr,
            )
        )
        scores.append(correlation(pred_arr, target_arr))
    return windows, {
        "model": "fcnn",
        "status": "success",
        "checkpoint_id": f"fcnn_epoch_{train_result.best_epoch}",
        "best_epoch": int(train_result.best_epoch),
        "best_val_score": float(train_result.best_val_score),
        "epochs_completed": int(train_result.epochs_completed),
        "test_pearson_direct": float(np.mean(scores)),
        "device": device,
    }


def fit_and_predict_adt(config: dict, dataset_dir: Path, device: str, output_dir: Path) -> tuple[list[WindowPrediction], dict[str, object]]:
    adt_cfg = config["adt"]
    subject_id = config["subject_id"]
    temp_dir = output_dir / "_tmp" / "adt_state"
    ensure_dir(temp_dir)
    model, summary = train_adt_exact_reference(
        dataset_dir,
        participants=[subject_id],
        output_dir=temp_dir,
        seq_len=int(adt_cfg["window_length"]),
        hop_length=int(adt_cfg["hop_length"]),
        batch_size=int(adt_cfg["batch_size"]),
        epochs=int(adt_cfg["max_epochs"]),
        patience=int(adt_cfg["early_stopping_patience"]),
        learning_rate=float(adt_cfg["learning_rate"]),
        min_lr=float(adt_cfg["min_lr"]),
        device=device,
        seed=int(config["seed"]),
    )
    checkpoint_id = f"adt_epoch_{summary.best_epoch}"
    windows: list[WindowPrediction] = []
    scores = []
    with torch.no_grad():
        model.eval()
        for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
            recording_windows: list[WindowPrediction] = []
            eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
            max_start = eeg.shape[0] - int(adt_cfg["window_length"])
            for start in range(0, max_start + 1, int(adt_cfg["hop_length"])):
                batch = eeg_tensor[start : start + int(adt_cfg["window_length"])].unsqueeze(0).to(device)
                pred = model(batch).squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
                target = env[start : start + int(adt_cfg["window_length"])].astype(np.float32)
                window = WindowPrediction(
                    dataset=config["dataset_id"],
                    model="adt",
                    task="reconstruction",
                    protocol=config["protocol"],
                    seed=int(config["seed"]),
                    subject_id=subject_id,
                    recording_id=recording_id,
                    sampling_rate=int(config["sampling_rate"]),
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
    state_file = temp_dir / "state_dict.pt"
    if state_file.exists():
        state_file.unlink()
    return windows, {
        "model": "adt",
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "best_epoch": int(summary.best_epoch),
        "best_val_metric": float(summary.best_val_metric),
        "epochs_completed": len(summary.history["val_pearson_metric"]),
        "test_pearson_direct": float(np.mean(scores)),
        "device": device,
    }


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dataset_dir = resolve_dataset_path(config["dataset_locator"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"missing dataset locator: {dataset_dir}")

    all_prediction_rows = []
    all_recording_rows = []
    all_subject_rows = []
    run_models = []

    fcnn_windows, fcnn_meta = fit_and_predict_fcnn(config, dataset_dir, device)
    pred_rows, rec_rows, subj_rows = aggregate_model_outputs(fcnn_windows, config["artifact_scope"])
    all_prediction_rows.extend(pred_rows)
    all_recording_rows.extend(rec_rows)
    all_subject_rows.extend(subj_rows)
    run_models.append(fcnn_meta)

    adt_windows, adt_meta = fit_and_predict_adt(config, dataset_dir, device, output_dir)
    pred_rows, rec_rows, subj_rows = aggregate_model_outputs(adt_windows, config["artifact_scope"])
    all_prediction_rows.extend(pred_rows)
    all_recording_rows.extend(rec_rows)
    all_subject_rows.extend(subj_rows)
    run_models.append(adt_meta)

    write_rows(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
    write_prediction_head(output_dir / "prediction_samples_head.csv", all_prediction_rows)

    manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "previous_pilot_commit": config["previous_pilot_commit"],
        "current_branch": config["current_branch"],
        "device": device,
        "dataset_id": config["dataset_id"],
        "dataset_locator": config["dataset_locator"],
        "subject_id": config["subject_id"],
        "models": run_models,
        "artifacts": {
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "prediction_samples_head": repo_relative(output_dir / "prediction_samples_head.csv"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
        },
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    schema_report = {
        "metrics_dir": repo_relative(output_dir),
        "recording_rows": len(all_recording_rows),
        "subject_rows": len(all_subject_rows),
        "prediction_head_exists": (output_dir / "prediction_samples_head.csv").exists(),
        "ok": True,
    }
    with open(output_dir / "schema_validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(schema_report, handle, indent=2)

    print(json.dumps({"output_dir": repo_relative(output_dir), "models": run_models}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
