from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.boundaries import safe_cross_boundary_rows, unsafe_cross_boundary_rows
from benchmark.result_schema import PREDICTION_FIELDS, RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, prediction_rows_from_series, recording_metric_row, subject_metric_rows
from repro.adt_exact import train_adt_exact_reference
from repro.reference_baselines import (
    DNNReferenceTrainResult,
    evaluate_dnn_reference_on_test,
    fit_reference_cca,
    fit_reference_ridge,
    list_reference_subjects,
    load_reference_recordings,
    train_dnn_reference_logged,
)
from repro.mldecoders.cca import score_reconstruction, trim_valid_range
from repro.simple_models import FCNNBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset_path(locator: str) -> Path:
    repo_path = ROOT / locator
    if repo_path.exists():
        return repo_path
    shared_workspace = Path(ROOT.drive + "\\decode")
    shared_candidate = shared_workspace / locator
    if shared_candidate.exists():
        return shared_candidate
    raise FileNotFoundError(f"dataset path not found for locator: {locator}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def correlation(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = pred.astype(np.float64)
    target64 = target.astype(np.float64)
    pred0 = pred64 - pred64.mean()
    target0 = target64 - target64.mean()
    denom = np.sqrt(np.sum(pred0 ** 2) * np.sum(target0 ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(pred0 * target0) / denom)


def build_recording_windows(
    *,
    dataset: str,
    model: str,
    task: str,
    protocol: str,
    seed: int,
    subject_id: str,
    recording_id: str,
    sampling_rate: int,
    checkpoint_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
    offset: int,
) -> WindowPrediction:
    return WindowPrediction(
        dataset=dataset,
        model=model,
        task=task,
        protocol=protocol,
        seed=seed,
        subject_id=subject_id,
        recording_id=recording_id,
        sampling_rate=sampling_rate,
        checkpoint_id=checkpoint_id,
        recording_length=int(offset + len(prediction)),
        start_index=int(offset),
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
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


def real_boundary_report(dataset_dir: Path, subject_id: str, start_lag: int, end_lag: int) -> dict[str, object]:
    recordings = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))
    lengths = [int(env.shape[0]) for _, _, env in recordings]
    return {
        "dataset_id": dataset_dir.name,
        "subject_id": subject_id,
        "recording_lengths": lengths,
        "start_lag": int(start_lag),
        "end_lag": int(end_lag),
        "unsafe_cross_boundary_rows": int(unsafe_cross_boundary_rows(lengths, start_lag, end_lag)),
        "safe_cross_boundary_rows": int(safe_cross_boundary_rows(lengths, start_lag, end_lag)),
    }


def fit_and_predict_ridge(config: dict, dataset_dir: Path, subject_id: str) -> tuple[list[WindowPrediction], dict[str, object]]:
    ridge_cfg = config["ridge"]
    model, summary = fit_reference_ridge(
        dataset_dir,
        subject_id,
        start_lag=int(ridge_cfg["start_lag"]),
        end_lag=int(ridge_cfg["end_lag"]),
        alphas=list(ridge_cfg["alphas"]),
        channels=range(64),
    )
    windows: list[WindowPrediction] = []
    direct_scores: list[float] = []
    offset = int(ridge_cfg["end_lag"]) - 1
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        x_lag, _, y_center = trim_valid_range(eeg, env, int(ridge_cfg["start_lag"]), int(ridge_cfg["end_lag"]))
        pred = model.predict(x_lag).astype(np.float32)
        windows.append(
            build_series_window(
                dataset=config["dataset_id"],
                model="ridge",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=int(config["sampling_rate"]),
                checkpoint_id=f"ridge_alpha_{summary['best_alpha']}",
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=y_center.astype(np.float32),
            )
        )
        direct_scores.append(correlation(pred, y_center.astype(np.float32)))
    meta = {
        "model": "ridge",
        "checkpoint_id": f"ridge_alpha_{summary['best_alpha']}",
        "train_mode": "fit_and_predict",
        "best_alpha": float(summary["best_alpha"]),
        "val_pearson": float(summary["val_pearson"]),
        "test_pearson_direct": float(np.mean(direct_scores)),
        "start_lag": int(summary["start_lag"]),
        "end_lag": int(summary["end_lag"]),
    }
    return windows, meta


def fit_and_predict_cca(config: dict, dataset_dir: Path, subject_id: str) -> tuple[list[WindowPrediction], dict[str, object]]:
    cca_cfg = config["ridge"]
    model, summary = fit_reference_cca(
        dataset_dir,
        subject_id,
        start_lag=int(cca_cfg["start_lag"]),
        end_lag=int(cca_cfg["end_lag"]),
        channels=range(64),
    )
    windows: list[WindowPrediction] = []
    recon_scores: list[float] = []
    offset = int(cca_cfg["end_lag"]) - 1
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        recon = score_reconstruction(model, eeg, env)
        windows.append(
            build_series_window(
                dataset=config["dataset_id"],
                model="cca",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=int(config["sampling_rate"]),
                checkpoint_id=f"cca_nc_{summary['n_components']}",
                full_length=len(env),
                offset=offset,
                prediction=recon["prediction"],
                target=recon["target"],
            )
        )
        recon_scores.append(float(recon["recon_corr"]))
    meta = {
        "model": "cca",
        "checkpoint_id": f"cca_nc_{summary['n_components']}",
        "train_mode": "fit_and_predict",
        "n_components": int(summary["n_components"]),
        "x_pca_components": int(summary["x_pca_components"]),
        "y_pca_components": int(summary["y_pca_components"]),
        "recon_alpha": float(summary["recon_alpha"]),
        "val_recon_corr": float(summary["val_recon_corr"]),
        "test_recon_corr_direct": float(np.mean(recon_scores)),
    }
    return windows, meta


def fit_and_predict_fcnn(config: dict, dataset_dir: Path, subject_id: str, device: str) -> tuple[list[WindowPrediction], dict[str, object]]:
    fcnn_cfg = config["fcnn"]
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
        epochs=int(fcnn_cfg["epochs"]),
        lr=float(fcnn_cfg["learning_rate"]),
        weight_decay=float(fcnn_cfg["weight_decay"]),
        batch_size=int(fcnn_cfg["batch_size"]),
        early_stopping_patience=int(fcnn_cfg["patience"]),
        device=device,
        seed=int(config["seed"]),
        channels=range(64),
    )
    _prediction, _target, test_score = evaluate_dnn_reference_on_test(
        dataset_dir,
        subject_id,
        FCNNBaseline,
        model_kwargs,
        train_result.state_dict,
        device=device,
        channels=range(64),
    )
    model = FCNNBaseline(**model_kwargs).to(device)
    model.load_state_dict(train_result.state_dict)
    model.eval()
    windows: list[WindowPrediction] = []
    per_recording_scores: list[float] = []
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
        per_recording_scores.append(correlation(pred_arr, target_arr))
    meta = {
        "model": "fcnn",
        "checkpoint_id": f"fcnn_epoch_{train_result.best_epoch}",
        "train_mode": "trained_and_predicted",
        "best_epoch": int(train_result.best_epoch),
        "epochs_completed": int(train_result.epochs_completed),
        "best_val_score": float(train_result.best_val_score),
        "test_pearson_direct": float(np.mean(per_recording_scores) if per_recording_scores else test_score),
        "window_size": int(fcnn_cfg["window_size"]),
        "device": device,
    }
    return windows, meta


def adt_prediction_windows(
    *,
    dataset_id: str,
    protocol: str,
    seed: int,
    subject_id: str,
    sampling_rate: int,
    checkpoint_id: str,
    dataset_dir: Path,
    window_length: int,
    hop_length: int,
    device: str,
) -> list[WindowPrediction]:
    from repro.adt_exact import ADTExactRegressor

    checkpoint_path = ROOT / "experiments" / "gate0_gate2_real_sample" / "model_outputs" / "adt" / "state_dict.pt"
    model = ADTExactRegressor(seq_len=window_length).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    windows: list[WindowPrediction] = []
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
        preds: list[np.ndarray] = []
        starts: list[int] = []
        with torch.no_grad():
            max_start = eeg.shape[0] - window_length
            for start in range(0, max_start + 1, hop_length):
                batch = eeg_tensor[start : start + window_length].unsqueeze(0).to(device)
                pred = model(batch).squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
                preds.append(pred)
                starts.append(start)
        for start, pred in zip(starts, preds):
            target = env[start : start + window_length].astype(np.float32)
            windows.append(
                WindowPrediction(
                    dataset=dataset_id,
                    model="adt",
                    task="reconstruction",
                    protocol=protocol,
                    seed=seed,
                    subject_id=subject_id,
                    recording_id=recording_id,
                    sampling_rate=sampling_rate,
                    checkpoint_id=checkpoint_id,
                    recording_length=int(len(env)),
                    start_index=int(start),
                    prediction=pred,
                    target=target,
                )
            )
    return windows


def fit_and_predict_adt(config: dict, dataset_dir: Path, subject_id: str, device: str, output_dir: Path) -> tuple[list[WindowPrediction], dict[str, object]]:
    adt_cfg = config["adt"]
    model_output_dir = output_dir / "model_outputs" / "adt"
    ensure_dir(model_output_dir)
    _, summary = train_adt_exact_reference(
        dataset_dir,
        participants=[subject_id],
        output_dir=model_output_dir,
        seq_len=int(adt_cfg["window_length"]),
        hop_length=int(adt_cfg["hop_length"]),
        batch_size=int(adt_cfg["batch_size"]),
        epochs=int(adt_cfg["epochs"]),
        patience=int(adt_cfg["patience"]),
        learning_rate=float(adt_cfg["learning_rate"]),
        min_lr=float(adt_cfg["min_lr"]),
        device=device,
        seed=int(config["seed"]),
    )
    checkpoint_id = f"adt_epoch_{summary.best_epoch}"
    windows = adt_prediction_windows(
        dataset_id=config["dataset_id"],
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=subject_id,
        sampling_rate=int(config["sampling_rate"]),
        checkpoint_id=checkpoint_id,
        dataset_dir=dataset_dir,
        window_length=int(adt_cfg["window_length"]),
        hop_length=int(adt_cfg["hop_length"]),
        device=device,
    )
    mean_subject_metric = None
    if summary.subject_metrics:
        mean_subject_metric = float(np.mean([item.pearson_metric for item in summary.subject_metrics.values()]))
    meta = {
        "model": "adt",
        "checkpoint_id": checkpoint_id,
        "train_mode": "trained_and_predicted",
        "best_epoch": int(summary.best_epoch),
        "best_val_loss": float(summary.best_val_loss),
        "best_val_metric": float(summary.best_val_metric),
        "test_subject_metric_direct": mean_subject_metric,
        "window_length": int(adt_cfg["window_length"]),
        "hop_length": int(adt_cfg["hop_length"]),
        "device": device,
    }
    return windows, meta


def aggregate_model_outputs(
    *,
    windows: Iterable[WindowPrediction],
    artifact_scope: str,
) -> tuple[list[object], list[object], list[object]]:
    grouped: dict[tuple[str, str], list[WindowPrediction]] = {}
    for window in windows:
        grouped.setdefault((window.subject_id, window.recording_id), []).append(window)

    prediction_rows = []
    recording_rows = []
    for (_, _), recording_windows in sorted(grouped.items()):
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


def write_model_prediction_dump(model_dir: Path, rows: list[object]) -> None:
    write_rows(model_dir / "prediction_samples.csv", rows, PREDICTION_FIELDS)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    dataset_dir = resolve_dataset_path(config["dataset_locator"])
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(output_dir / "model_outputs")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    subjects = list_reference_subjects(dataset_dir, split="test")
    subject_id = config.get("subject_id", subjects[0] if subjects else "P00")

    all_prediction_rows = []
    all_recording_rows = []
    all_subject_rows = []
    run_models: list[dict[str, object]] = []

    ridge_windows, ridge_meta = fit_and_predict_ridge(config, dataset_dir, subject_id)
    ridge_predictions, ridge_recording, ridge_subject = aggregate_model_outputs(
        windows=ridge_windows,
        artifact_scope=config["artifact_scope"],
    )
    run_models.append(ridge_meta)
    all_prediction_rows.extend(ridge_predictions)
    all_recording_rows.extend(ridge_recording)
    all_subject_rows.extend(ridge_subject)
    write_model_prediction_dump(output_dir / "model_outputs" / "ridge", ridge_predictions)

    cca_windows, cca_meta = fit_and_predict_cca(config, dataset_dir, subject_id)
    cca_predictions, cca_recording, cca_subject = aggregate_model_outputs(
        windows=cca_windows,
        artifact_scope=config["artifact_scope"],
    )
    run_models.append(cca_meta)
    all_prediction_rows.extend(cca_predictions)
    all_recording_rows.extend(cca_recording)
    all_subject_rows.extend(cca_subject)
    write_model_prediction_dump(output_dir / "model_outputs" / "cca", cca_predictions)

    fcnn_windows, fcnn_meta = fit_and_predict_fcnn(config, dataset_dir, subject_id, device)
    fcnn_predictions, fcnn_recording, fcnn_subject = aggregate_model_outputs(
        windows=fcnn_windows,
        artifact_scope=config["artifact_scope"],
    )
    run_models.append(fcnn_meta)
    all_prediction_rows.extend(fcnn_predictions)
    all_recording_rows.extend(fcnn_recording)
    all_subject_rows.extend(fcnn_subject)
    write_model_prediction_dump(output_dir / "model_outputs" / "fcnn", fcnn_predictions)

    adt_windows, adt_meta = fit_and_predict_adt(config, dataset_dir, subject_id, device, output_dir)
    adt_predictions, adt_recording, adt_subject = aggregate_model_outputs(
        windows=adt_windows,
        artifact_scope=config["artifact_scope"],
    )
    run_models.append(adt_meta)
    all_prediction_rows.extend(adt_predictions)
    all_recording_rows.extend(adt_recording)
    all_subject_rows.extend(adt_subject)
    write_model_prediction_dump(output_dir / "model_outputs" / "adt", adt_predictions)

    write_rows(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
    write_rows(output_dir / "prediction_samples.csv", all_prediction_rows, PREDICTION_FIELDS)

    boundary = real_boundary_report(
        dataset_dir,
        subject_id,
        int(config["ridge"]["start_lag"]),
        int(config["ridge"]["end_lag"]),
    )
    boundary["boundary_fix_passed"] = boundary["unsafe_cross_boundary_rows"] > 0 and boundary["safe_cross_boundary_rows"] == 0

    run_manifest = {
        "dataset_id": config["dataset_id"],
        "dataset_locator": config["dataset_locator"],
        "dataset_runtime_source": "shared_workspace_locator",
        "subject_id": subject_id,
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "seed": int(config["seed"]),
        "device": device,
        "models_requested": list(config["models"]),
        "models_executed": [item["model"] for item in run_models],
        "model_run_details": run_models,
        "artifacts": {
            "prediction_samples": repo_relative(output_dir / "prediction_samples.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "scorer_validation": repo_relative(output_dir / "scorer_validation.md"),
            "boundary_validation": repo_relative(output_dir / "boundary_validation.md"),
            "result_summary": repo_relative(output_dir / "result_summary.md"),
        },
    }

    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    scorer_lines = [
        "# Scorer Validation",
        "",
        f"- Dataset: `{config['dataset_id']}`",
        f"- Subject: `{subject_id}`",
        f"- Artifact scope: `{config['artifact_scope']}`",
        "- All benchmark rows in `recording_metrics.csv` and `subject_metrics.csv` were produced by the shared scorer in `src/benchmark/scoring.py`.",
        "- Model-specific direct metrics were recorded only as auxiliary metadata in `run_manifest.json` and were not used as the final reported schema output.",
        "- `prediction_samples.csv` files were written per model under `experiments/gate0_gate2_real_sample/model_outputs/`.",
        "",
        "## Direct-vs-unified checks",
        "",
    ]
    for item in run_models:
        if "test_pearson_direct" in item:
            scorer_lines.append(f"- `{item['model']}` direct Pearson: `{item['test_pearson_direct']:.6f}`")
        if "test_recon_corr_direct" in item:
            scorer_lines.append(f"- `{item['model']}` direct reconstruction correlation: `{item['test_recon_corr_direct']:.6f}`")
    write_markdown(output_dir / "scorer_validation.md", scorer_lines)

    boundary_lines = [
        "# Boundary Validation",
        "",
        f"- Dataset: `{config['dataset_id']}`",
        f"- Subject: `{subject_id}`",
        f"- Lag range: `{boundary['start_lag']}` to `{boundary['end_lag']}`",
        f"- Unsafe concatenation cross-boundary rows: `{boundary['unsafe_cross_boundary_rows']}`",
        f"- Safe per-recording lagging cross-boundary rows: `{boundary['safe_cross_boundary_rows']}`",
        f"- Boundary fix passed: `{boundary['boundary_fix_passed']}`",
        "",
        "This check uses real test-recording lengths from the selected sample dataset rather than a synthetic recording-length fixture.",
    ]
    write_markdown(output_dir / "boundary_validation.md", boundary_lines)

    mean_by_model: dict[str, float] = {}
    for row in all_subject_rows:
        mean_by_model[row.model] = float(row.metric_value)
    result_lines = [
        "# Result Summary",
        "",
        "## 1. 本轮用了什么数据？",
        f"- 使用数据：`{config['dataset_id']}`",
        "- 运行对象：单被试最小公开样例 `P00`，采用已存在的 `train/val/test` 参考切分。",
        "- 当前定位：真实 sample 的 pipeline validation，不是完整 benchmark 结果。",
        "",
        "## 2. 跑了哪些模型？",
        "- classical baseline: `ridge`",
        "- additional classical boundary check: `cca`",
        "- simple neural baseline: `fcnn`",
        "- target model: `adt`",
        "",
        "## 3. 每个模型是否真实训练/推理，还是只跑了接口？",
        "- `ridge`: 真实拟合并在测试集上生成预测。",
        "- `cca`: 真实拟合并在测试集上生成预测，但本轮不把其 canonical/match-mismatch 指标当作主比较列。",
        "- `fcnn`: 真实训练并在测试集上推理。",
        "- `adt`: 真实训练并在测试集上按滑窗重建、再由统一 scorer 聚合。",
        "",
        "## 4. 指标是什么？",
        "- 主指标：`Pearson correlation`。",
        "- 统一输出：`recording_metrics.csv` 与 `subject_metrics.csv`。",
        "- `cca` 的 canonical correlation 与 match-mismatch accuracy 仅作为附加元数据保留。",
        "",
        "## 5. 当前结果能说明什么？",
    ]
    for model_name, value in sorted(mean_by_model.items()):
        result_lines.append(f"- `{model_name}` 在该最小 sample 上通过统一 scorer 生成了可追溯结果，当前 subject-level Pearson 为 `{value:.6f}`。")
    result_lines.extend(
        [
            "",
            "## 6. 当前结果不能说明什么？",
            "- 不能说明模型总体优劣。",
            "- 不能替代多被试、多数据集、多 seed 的正式 benchmark。",
            "- 不能作为论文主结果图直接使用。",
            "",
            "## 7. 哪些结果只能作为 smoke evidence，不能写成论文结论？",
            "- 本目录下全部结果都只能作为 pipeline validation 或 minimal real-sample evidence。",
            "- `cca` 的附加 canonical / match-mismatch 输出在当前轮次只用于边界与 schema 说明。",
        ]
    )
    write_markdown(output_dir / "result_summary.md", result_lines)

    print(json.dumps({
        "output_dir": repo_relative(output_dir),
        "device": device,
        "models_executed": [item["model"] for item in run_models],
        "subject_metrics": mean_by_model,
        "boundary_fix_passed": boundary["boundary_fix_passed"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
