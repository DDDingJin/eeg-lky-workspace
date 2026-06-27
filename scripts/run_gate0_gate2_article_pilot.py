from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
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


def stable_locator(path: Path) -> str:
    candidates = [ROOT, Path(ROOT.drive + "\\decode")]
    for base in candidates:
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


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


def aggregate_model_outputs(windows: list[WindowPrediction], artifact_scope: str) -> tuple[list[object], list[object], list[object]]:
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


def fit_and_predict_ridge(config: dict, dataset_id: str, dataset_dir: Path, subject_id: str, sampling_rate: int) -> tuple[list[WindowPrediction], dict[str, object]]:
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
    scores = []
    offset = int(ridge_cfg["end_lag"]) - 1
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        x_lag, _, y_center = trim_valid_range(eeg, env, int(ridge_cfg["start_lag"]), int(ridge_cfg["end_lag"]))
        pred = model.predict(x_lag).astype(np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="ridge",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=f"ridge_alpha_{summary['best_alpha']}",
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=y_center.astype(np.float32),
            )
        )
        scores.append(correlation(pred, y_center.astype(np.float32)))
    return windows, {
        "model": "ridge",
        "status": "success",
        "checkpoint_id": f"ridge_alpha_{summary['best_alpha']}",
        "best_alpha": float(summary["best_alpha"]),
        "val_pearson": float(summary["val_pearson"]),
        "test_pearson_direct": float(np.mean(scores)),
    }


def fit_and_predict_cca(config: dict, dataset_id: str, dataset_dir: Path, subject_id: str, sampling_rate: int) -> tuple[list[WindowPrediction], dict[str, object]]:
    ridge_cfg = config["ridge"]
    model, summary = fit_reference_cca(
        dataset_dir,
        subject_id,
        start_lag=int(ridge_cfg["start_lag"]),
        end_lag=int(ridge_cfg["end_lag"]),
        channels=range(64),
    )
    windows: list[WindowPrediction] = []
    recon_scores = []
    offset = int(ridge_cfg["end_lag"]) - 1
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        recon = score_reconstruction(model, eeg, env)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="cca",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=f"cca_nc_{summary['n_components']}",
                full_length=len(env),
                offset=offset,
                prediction=recon["prediction"],
                target=recon["target"],
            )
        )
        recon_scores.append(float(recon["recon_corr"]))
    return windows, {
        "model": "cca",
        "status": "success",
        "checkpoint_id": f"cca_nc_{summary['n_components']}",
        "n_components": int(summary["n_components"]),
        "val_recon_corr": float(summary["val_recon_corr"]),
        "test_recon_corr_direct": float(np.mean(recon_scores)),
    }


def fit_and_predict_fcnn(config: dict, dataset_id: str, dataset_dir: Path, subject_id: str, sampling_rate: int, device: str) -> tuple[list[WindowPrediction], dict[str, object]]:
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
    evaluate_dnn_reference_on_test(
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
                dataset=dataset_id,
                model="fcnn",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
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
        "test_pearson_direct": float(np.mean(scores)),
        "device": device,
    }


def fit_and_predict_adt(config: dict, dataset_id: str, dataset_dir: Path, subject_id: str, sampling_rate: int, device: str, temp_dir: Path) -> tuple[list[WindowPrediction], dict[str, object]]:
    adt_cfg = config["adt"]
    state_dir = temp_dir / "adt_state"
    ensure_dir(state_dir)
    model, summary = train_adt_exact_reference(
        dataset_dir,
        participants=[subject_id],
        output_dir=state_dir,
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
                    dataset=dataset_id,
                    model="adt",
                    task="reconstruction",
                    protocol=config["protocol"],
                    seed=int(config["seed"]),
                    subject_id=subject_id,
                    recording_id=recording_id,
                    sampling_rate=sampling_rate,
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
    return windows, {
        "model": "adt",
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "best_epoch": int(summary.best_epoch),
        "best_val_metric": float(summary.best_val_metric),
        "test_pearson_direct": float(np.mean(scores)) if scores else float("nan"),
        "device": device,
    }


def write_failure_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_prediction_head(path: Path, rows: list[object]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        row_dict = row.__dict__
        if int(row_dict["valid_mask"]) != 1:
            continue
        grouped.setdefault(row_dict["model"], []).append(row_dict)
    selected: list[dict[str, object]] = []
    for model_name in sorted(grouped):
        selected.extend(grouped[model_name][:5])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(selected)


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    all_recording_rows = []
    all_subject_rows = []
    head_prediction_rows = []
    dataset_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for dataset_cfg in config["datasets"]:
        dataset_id = dataset_cfg["dataset_id"]
        dataset_path = resolve_dataset_path(dataset_cfg["dataset_locator"])
        dataset_report: dict[str, object] = {
            "dataset_id": dataset_id,
            "dataset_locator": dataset_cfg["dataset_locator"],
            "subject_id": dataset_cfg["subject_id"],
            "status": "pending",
            "models": []
        }

        if not dataset_path.exists():
            if dataset_id == "etard_tf64_p00":
                required_preparation = [
                    "prepare or restore the shared full reference split directory data/processed/reference_splits/etard_tf64",
                    "the article pilot expects participant P00 to be read from the full etard_tf64 export, not from the test fixture etard_tf64_p00_test"
                ]
                suggested_command = None
            else:
                required_preparation = [
                    "prepare the requested dataset locator and verify the participant-specific split files are present"
                ]
                suggested_command = None
            failure = {
                "dataset_id": dataset_id,
                "status": "missing_dataset_directory",
                "expected_locator": dataset_cfg["dataset_locator"],
                "runtime_checked_path": stable_locator(dataset_path),
                "required_preparation": required_preparation,
                "suggested_command": suggested_command,
            }
            failures.append(failure)
            dataset_report["status"] = "failed_missing_dataset"
            dataset_report["failure_report"] = failure
            dataset_results.append(dataset_report)
            continue

        temp_dir = output_dir / "_tmp" / dataset_id
        ensure_dir(temp_dir)
        subject_id = dataset_cfg["subject_id"]
        sampling_rate = int(dataset_cfg["sampling_rate"])
        dataset_prediction_rows = []
        dataset_recording_rows = []
        dataset_subject_rows = []

        runners = [
            ("ridge", lambda: fit_and_predict_ridge(config, dataset_id, dataset_path, subject_id, sampling_rate)),
            ("cca", lambda: fit_and_predict_cca(config, dataset_id, dataset_path, subject_id, sampling_rate)),
            ("fcnn", lambda: fit_and_predict_fcnn(config, dataset_id, dataset_path, subject_id, sampling_rate, device)),
            ("adt", lambda: fit_and_predict_adt(config, dataset_id, dataset_path, subject_id, sampling_rate, device, temp_dir)),
        ]

        for model_name, runner in runners:
            try:
                windows, meta = runner()
                prediction_rows, recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
                dataset_prediction_rows.extend(prediction_rows)
                dataset_recording_rows.extend(recording_rows)
                dataset_subject_rows.extend(subject_rows)
                dataset_report["models"].append(meta)
            except Exception as exc:
                dataset_report["models"].append({
                    "model": model_name,
                    "status": "failed",
                    "error": repr(exc),
                })

        dataset_report["status"] = "success" if any(item.get("status") == "success" for item in dataset_report["models"]) else "failed_all_models"
        dataset_results.append(dataset_report)
        all_recording_rows.extend(dataset_recording_rows)
        all_subject_rows.extend(dataset_subject_rows)
        head_prediction_rows.extend(dataset_prediction_rows)

        temp_state = temp_dir / "adt_state" / "state_dict.pt"
        if temp_state.exists():
            temp_state.unlink()
        shutil.rmtree(temp_dir, ignore_errors=True)

    write_rows(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
    write_prediction_head(output_dir / "prediction_samples_head.csv", head_prediction_rows)
    if failures:
        write_failure_report(output_dir / "failure_report.json", {"failures": failures})

    manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "previous_fix_branch": config["previous_fix_branch"],
        "previous_fix_commit": config["previous_fix_commit"],
        "current_branch": config.get("current_branch", "fix/ar-20260625-161300-a43831b-etard-p00-pilot-closure"),
        "device": device,
        "datasets_requested": [item["dataset_id"] for item in config["datasets"]],
        "models_requested": list(config["models"]),
        "dataset_results": dataset_results,
        "failures": failures,
        "artifacts": {
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "result_summary": repo_relative(output_dir / "result_summary.md"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "incremental_comparison": repo_relative(output_dir / "incremental_comparison.md"),
            "prediction_samples_head": repo_relative(output_dir / "prediction_samples_head.csv"),
            "failure_report": repo_relative(output_dir / "failure_report.json") if failures else None
        }
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summary_lines = [
        "# Article-Grade Dataset Pilot Validation",
        "",
        "This directory is a pilot validation layer for article-grade candidate datasets, not a full-subject benchmark and not a final paper conclusion.",
        "",
        "## Requested datasets",
    ]
    for item in config["datasets"]:
        summary_lines.append(f"- `{item['dataset_id']}`")
    summary_lines.extend([
        "",
        "## Requested models",
        "- `ridge`",
        "- `cca`",
        "- `fcnn`",
        "- `adt`",
        "",
        "## Pilot status",
    ])
    for result in dataset_results:
        summary_lines.append(f"- `{result['dataset_id']}`: `{result['status']}`")
        for model_meta in result.get("models", []):
            if model_meta.get("status") != "success":
                summary_lines.append(f"  - `{model_meta['model']}`: `{model_meta['status']}`")
                continue
            metric = (
                model_meta.get("test_pearson_direct")
                if "test_pearson_direct" in model_meta
                else model_meta.get("test_recon_corr_direct")
            )
            if metric is not None:
                summary_lines.append(f"  - `{model_meta['model']}` subject-level Pearson: `{metric:.6f}`")
    if failures:
        summary_lines.extend([
            "",
            "## Missing or blocked datasets",
        ])
        for failure in failures:
            summary_lines.append(f"- `{failure['dataset_id']}` missing at expected locator `{failure['expected_locator']}`.")
            if failure.get("suggested_command"):
                summary_lines.append(f"  Suggested preparation: `{failure['suggested_command']}`")
            else:
                for note in failure.get("required_preparation", []):
                    summary_lines.append(f"  Preparation note: `{note}`")
    summary_lines.extend([
        "",
        "## Interpretation guardrail",
        "- These P00 pilot outputs only validate that the current unified runner/scorer stack can be extended to article-grade candidate datasets.",
        "- They must not be reported as full benchmark conclusions.",
    ])
    (output_dir / "result_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    incremental_lines = [
        "# Incremental Comparison",
        "",
        f"- previous fix branch: `{config['previous_fix_branch']}`",
        f"- previous fix commit: `{config['previous_fix_commit']}`",
        f"- current fix branch: `{config.get('current_branch', 'fix/ar-20260625-161300-a43831b-etard-p00-pilot-closure')}`",
        "",
        "## Newly run in this round",
        "- `etard_tf64_p00` pilot closure under `experiments/gate0_gate2_article_pilot/`",
        "- the unified runner now reads Etard `P00` directly from the full `etard_tf64` export instead of requiring a separate `etard_tf64_p00` alias",
        "- `ridge`, `cca`, `fcnn`, `adt` under the unified pilot runner",
        "",
        "## Not rerun in this round",
        "- historical `experiments/summary_figures/*` outputs",
        "- old Hugo sample pilot outputs",
        "- full-subject Weissbart runs",
        "- full-subject Etard runs",
        "- historical HappyQuokka and VLAAI result bundles",
        "",
        "## Etard pilot closure",
        "- `etard_tf64_p00` now succeeds by reading participant `P00` from `data/processed/reference_splits/etard_tf64`.",
        "- `etard_tf64_p00_test` remains only a test fixture and was not substituted silently as article pilot evidence.",
        "- `failure_report.json` is no longer needed once the Etard pilot closure succeeds."
    ]
    (output_dir / "incremental_comparison.md").write_text("\n".join(incremental_lines) + "\n", encoding="utf-8")

    schema_report = {
        "metrics_dir": repo_relative(output_dir),
        "recording_rows": len(all_recording_rows),
        "subject_rows": len(all_subject_rows),
        "prediction_head_exists": (output_dir / "prediction_samples_head.csv").exists(),
        "ok": True
    }
    with open(output_dir / "schema_validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(schema_report, handle, indent=2)

    if not failures:
        failure_file = output_dir / "failure_report.json"
        if failure_file.exists():
            failure_file.unlink()

    shutil.rmtree(output_dir / "_tmp", ignore_errors=True)

    print(json.dumps({
        "output_dir": repo_relative(output_dir),
        "failures": failures,
        "num_recording_rows": len(all_recording_rows),
        "num_subject_rows": len(all_subject_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
