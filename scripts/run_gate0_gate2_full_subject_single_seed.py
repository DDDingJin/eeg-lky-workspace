from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.adt_exact import train_adt_exact_reference
from repro.reference_baselines import DNNReferenceTrainResult, fit_reference_cca, fit_reference_ridge, list_reference_subjects, load_reference_recordings, train_dnn_reference_logged
from repro.mldecoders.cca import score_reconstruction, trim_valid_range
from repro.simple_models import FCNNBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def stable_locator(path: Path) -> str:
    candidates = [ROOT, Path(ROOT.drive + "\\decode")]
    for base in candidates:
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


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
            )
        )
    subject_rows = subject_metric_rows(recording_rows)
    return recording_rows, subject_rows


def condition_from_recording_id(recording_id: str) -> str | None:
    marker = "_-_"
    parts = recording_id.split(marker)
    if len(parts) < 3:
        return None
    tail = parts[2]
    if "_part_" not in tail:
        return None
    return tail.split("_part_", 1)[0]


def canonical_fingerprint_payload(config: dict, dataset_cfg: dict, subject_id: str, model_name: str) -> dict:
    return {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "config_fingerprint_version": config["config_fingerprint_version"],
        "dataset_id": dataset_cfg["dataset_id"],
        "dataset_locator": dataset_cfg["dataset_locator"],
        "sampling_rate": int(dataset_cfg["sampling_rate"]),
        "subject_id": subject_id,
        "model": model_name,
        "seed": int(config["seed"]),
        "scorer_id": config["scorer_id"],
        "aggregation_rule": config["aggregation_rule"],
        "model_config": config[model_name],
    }


def fingerprint_for_job(config: dict, dataset_cfg: dict, subject_id: str, model_name: str) -> tuple[str, dict]:
    payload = canonical_fingerprint_payload(config, dataset_cfg, subject_id, model_name)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest(), payload


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
    cca_cfg = config["cca"]
    model, summary = fit_reference_cca(
        dataset_dir,
        subject_id,
        start_lag=int(cca_cfg["start_lag"]),
        end_lag=int(cca_cfg["end_lag"]),
        channels=range(64),
    )
    windows: list[WindowPrediction] = []
    recon_scores = []
    offset = int(cca_cfg["end_lag"]) - 1
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
        "epochs_completed": int(train_result.epochs_completed),
        "test_pearson_direct": float(np.mean(scores)),
        "device": device,
    }


def fit_and_predict_adt(
    config: dict,
    dataset_id: str,
    dataset_dir: Path,
    subject_id: str,
    sampling_rate: int,
    device: str,
    temp_dir: Path,
) -> tuple[list[WindowPrediction], dict[str, object]]:
    adt_cfg = config["adt"]
    state_dir = temp_dir / "adt_state"
    ensure_dir(state_dir)
    model, summary = train_adt_exact_reference(
        dataset_dir,
        participants=[subject_id],
        train_participant=subject_id,
        output_dir=state_dir,
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
    state_file = state_dir / "state_dict.pt"
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


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_job_ledger_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "subject_id",
        "model",
        "seed",
        "fingerprint",
        "action",
        "status",
        "job_dir",
        "checkpoint_id",
        "subject_metric",
        "num_recordings",
        "error",
    ]
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_dataset_metrics(subject_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in subject_rows:
        if row["metric_name"] != "mean_recording_pearson_r":
            continue
        grouped.setdefault((row["dataset"], row["model"]), []).append(float(row["metric_value"]))

    output = []
    for (dataset, model), values in sorted(grouped.items()):
        values_arr = np.asarray(values, dtype=np.float64)
        clipped = np.clip(values_arr, -0.999999, 0.999999)
        fisher_z = np.arctanh(clipped)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "n_subjects": int(values_arr.size),
                "mean_pearson": float(np.mean(values_arr)),
                "std_pearson": float(np.std(values_arr, ddof=0)),
                "median_pearson": float(np.median(values_arr)),
                "min_pearson": float(np.min(values_arr)),
                "max_pearson": float(np.max(values_arr)),
                "fisher_z_mean": float(np.mean(fisher_z)),
                "backtransformed_mean_r": float(np.tanh(np.mean(fisher_z))),
            }
        )
    return output


def summarize_etard_condition_metrics(recording_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in recording_rows:
        if row["dataset"] != "etard_tf64":
            continue
        condition = condition_from_recording_id(row["recording_id"])
        if condition not in {"clean", "fM", "fW", "hb", "lb", "mb"}:
            continue
        grouped.setdefault((row["model"], condition), []).append(row)

    output = []
    for (model, condition), rows in sorted(grouped.items()):
        values = np.asarray([float(item["metric_value"]) for item in rows], dtype=np.float64)
        output.append(
            {
                "dataset": "etard_tf64",
                "model": model,
                "condition": condition,
                "n_subjects": len({item["subject_id"] for item in rows}),
                "n_recordings": len(rows),
                "mean_pearson": float(np.mean(values)),
                "median_pearson": float(np.median(values)),
                "std_pearson": float(np.std(values, ddof=0)),
                "negative_recording_count": int(np.sum(values < 0)),
            }
        )
    return output


def write_generic_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_result_summary(
    path: Path,
    *,
    config: dict,
    device: str,
    dataset_subjects: dict[str, list[str]],
    expected_jobs: int,
    cumulative_successful_jobs: int,
    skipped_jobs: int,
    protocol_mismatch_jobs: int,
    run_jobs: int,
    failed_jobs: int,
    dataset_metrics: list[dict[str, object]],
    etard_condition_metrics: list[dict[str, object]],
) -> None:
    lines = [
        "# Full-Subject Single-Seed v1",
        "",
        "This directory contains the first article-grade full-subject single-seed run under the unified runner, schema, and scorer.",
        "It is not a multi-seed benchmark, not a cross-dataset transfer study, and not a final paper conclusion.",
        "",
        "## Protocol",
        f"- `protocol`: `{config['protocol']}`",
        f"- `device`: `{device}`",
        f"- `seed`: `{config['seed']}`",
        f"- `models`: `{', '.join(config['models'])}`",
        "",
        "## Subject inventory",
    ]
    for dataset_id, subjects in dataset_subjects.items():
        lines.append(f"- `{dataset_id}` ({len(subjects)} subjects): `{', '.join(subjects)}`")
    lines.extend(
        [
            "",
            "## Job accounting",
            f"- expected jobs: `{expected_jobs}`",
            f"- cumulative successful jobs available: `{cumulative_successful_jobs}`",
            f"- skipped existing jobs: `{skipped_jobs}`",
            f"- protocol mismatch jobs: `{protocol_mismatch_jobs}`",
            f"- actually run jobs: `{run_jobs}`",
            f"- failed jobs: `{failed_jobs}`",
            "",
            "## Dataset-level Pearson summary",
            "| dataset | model | n_subjects | mean_pearson | std_pearson | median_pearson | min_pearson | max_pearson | fisher_z_mean | backtransformed_mean_r |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_metrics:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_subjects']} | "
            f"{row['mean_pearson']:.6f} | {row['std_pearson']:.6f} | {row['median_pearson']:.6f} | "
            f"{row['min_pearson']:.6f} | {row['max_pearson']:.6f} | {row['fisher_z_mean']:.6f} | "
            f"{row['backtransformed_mean_r']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Etard condition summary",
            "| model | condition | n_subjects | n_recordings | mean_pearson | median_pearson | std_pearson | negative_recording_count |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in etard_condition_metrics:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['n_subjects']} | {row['n_recordings']} | "
            f"{row['mean_pearson']:.6f} | {row['median_pearson']:.6f} | {row['std_pearson']:.6f} | "
            f"{row['negative_recording_count']} |"
        )

    lines.extend(
        [
            "",
            "## Scope notes",
            "- Historical `experiments/summary_figures/*` were not rerun and remain reference-only.",
            "- This run keeps a single unified scorer and aggregation rule across all four models.",
            "- Per-job outputs are stored incrementally under `experiments/gate0_gate2_full_subject_single_seed/jobs/` to enable resume and protocol-mismatch detection without overwriting prior evidence.",
            "- Etard condition-level `n_subjects` is condition-specific rather than always 20 because the processed split itself is not perfectly balanced across condition families for every participant. The full dataset-level subject summary still covers all 20 Etard subjects.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_incremental_comparison(
    path: Path,
    *,
    previous_fix_commit: str,
    current_branch: str,
    dataset_subjects: dict[str, list[str]],
    cumulative_successful_jobs: int,
    run_jobs: int,
    skipped_jobs: int,
    failed_jobs: int,
) -> None:
    lines = [
        "# Incremental Comparison",
        "",
        f"- previous fix commit: `{previous_fix_commit}`",
        f"- current branch: `{current_branch}`",
        "- this round adds the first full-subject single-seed v1 run on `weissbart_tf64` and `etard_tf64`.",
        "- this round does not rerun historical summary figures, pilot-only artifacts, multi-seed experiments, or specialized models outside `ridge / cca / fcnn / adt`.",
        "",
        "## New in this round",
        "- `configs/benchmark/gate0_gate2_full_subject_single_seed.json`",
        "- `scripts/run_gate0_gate2_full_subject_single_seed.py`",
        "- full-subject per-job ledgers and aggregated dataset/condition metrics",
        "",
        "## Job scope",
    ]
    for dataset_id, subjects in dataset_subjects.items():
        lines.append(f"- `{dataset_id}`: {len(subjects)} subjects")
    lines.extend(
        [
            f"- cumulative successful jobs available after this round: `{cumulative_successful_jobs}`",
            f"- actually run jobs: `{run_jobs}`",
            f"- skipped existing jobs: `{skipped_jobs}`",
            f"- failed jobs: `{failed_jobs}`",
            "",
            "## Not rerun in this round",
            "- `experiments/summary_figures/*` historical summaries",
            "- `experiments/gate0_gate2_article_pilot/*`",
            "- `experiments/gate0_gate2_etard_p00_focused_rerun/*`",
            "- VLAAI / HappyQuokka / NULL / CNN / EEGNet bundles",
            "- any multi-seed or cross-dataset transfer experiment",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_paper_results(path: Path, dataset_metrics: list[dict[str, object]], etard_condition_metrics: list[dict[str, object]]) -> None:
    dataset_lookup = {(row["dataset"], row["model"]): row for row in dataset_metrics}
    text = path.read_text(encoding="utf-8")
    appendix = [
        "",
        "\\subsection{当前最小全被试结果：Full-Subject Single-Seed v1}",
        "",
        "在完成 P00 pilot、Etard 条件诊断与 focused rerun 之后，本轮进一步进入全被试 single-seed v1。这里仍然不是最终 benchmark 结论，而是统一 runner、schema 与 scorer 下的第一版 article-grade 全被试结果。当前仅覆盖两个主数据集与四个固定模型：Ridge、CCA、FCNN、ADT；不包含 multi-seed、cross-dataset transfer，也不包含 VLAAI、HappyQuokka、NULL、CNN、EEGNet 等扩展模型。",
        "",
        "在该设置下，Weissbart 共 13 名被试，Etard 共 20 名被试。按 subject-level `mean_recording_pearson_r` 聚合后，当前数据集级均值为：",
        "\\begin{itemize}",
        f"  \\item \\texttt{{weissbart\\_tf64}}: Ridge {dataset_lookup[('weissbart_tf64', 'ridge')]['mean_pearson']:.3f}, CCA {dataset_lookup[('weissbart_tf64', 'cca')]['mean_pearson']:.3f}, FCNN {dataset_lookup[('weissbart_tf64', 'fcnn')]['mean_pearson']:.3f}, ADT {dataset_lookup[('weissbart_tf64', 'adt')]['mean_pearson']:.3f}",
        f"  \\item \\texttt{{etard\\_tf64}}: Ridge {dataset_lookup[('etard_tf64', 'ridge')]['mean_pearson']:.3f}, CCA {dataset_lookup[('etard_tf64', 'cca')]['mean_pearson']:.3f}, FCNN {dataset_lookup[('etard_tf64', 'fcnn')]['mean_pearson']:.3f}, ADT {dataset_lookup[('etard_tf64', 'adt')]['mean_pearson']:.3f}",
        "\\end{itemize}",
        "",
        "对于 Etard，本轮还额外保留了按条件的聚合统计，以避免单一总体均值掩盖 clean、fM、fW、hb、lb、mb 之间的难度差异。这些结果用于说明 article-grade benchmark 已经进入 full-subject single-seed 阶段，但仍不足以支持最终论文中的模型优劣结论。下一步若继续推进，应优先考虑 multi-seed 复核或扩展模型，而不是将当前单次运行直接写成最终 finding。",
    ]
    if "\\subsection{当前最小全被试结果：Full-Subject Single-Seed v1}" not in text:
        text = text.rstrip() + "\n\n" + "\n".join(appendix) + "\n"
    path.write_text(text, encoding="utf-8")


def infer_dataset_subjects_from_subject_rows(subject_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for row in subject_rows:
        grouped.setdefault(row["dataset"], set()).add(row["subject_id"])
    return {dataset: sorted(subjects) for dataset, subjects in sorted(grouped.items())}


def run_job(
    *,
    config: dict,
    dataset_cfg: dict,
    dataset_dir: Path,
    subject_id: str,
    model_name: str,
    device: str,
    output_dir: Path,
) -> tuple[list[object], list[object], dict[str, object]]:
    dataset_id = dataset_cfg["dataset_id"]
    sampling_rate = int(dataset_cfg["sampling_rate"])
    temp_dir = output_dir / "_tmp" / dataset_id / subject_id / model_name
    ensure_dir(temp_dir)
    try:
        if model_name == "ridge":
            windows, meta = fit_and_predict_ridge(config, dataset_id, dataset_dir, subject_id, sampling_rate)
        elif model_name == "cca":
            windows, meta = fit_and_predict_cca(config, dataset_id, dataset_dir, subject_id, sampling_rate)
        elif model_name == "fcnn":
            windows, meta = fit_and_predict_fcnn(config, dataset_id, dataset_dir, subject_id, sampling_rate, device)
        elif model_name == "adt":
            windows, meta = fit_and_predict_adt(config, dataset_id, dataset_dir, subject_id, sampling_rate, device, temp_dir)
        else:
            raise ValueError(f"unknown model {model_name}")
        recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
        return recording_rows, subject_rows, meta
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    jobs_root = output_dir / "jobs"
    ensure_dir(jobs_root)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    resume_enabled = True
    skip_existing = True
    if args.force_rerun:
        resume_enabled = False
        skip_existing = False
    elif args.resume or args.skip_existing:
        resume_enabled = True
        skip_existing = True

    dataset_subjects: dict[str, list[str]] = {}
    dataset_cfg_lookup: dict[str, dict] = {}
    dataset_path_lookup: dict[str, Path] = {}
    failures: list[dict[str, object]] = []
    job_rows: list[dict[str, object]] = []

    for dataset_cfg in config["datasets"]:
        dataset_id = dataset_cfg["dataset_id"]
        dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
        dataset_cfg_lookup[dataset_id] = dataset_cfg
        dataset_path_lookup[dataset_id] = dataset_dir
        if not dataset_dir.exists():
            failures.append(
                {
                    "dataset": dataset_id,
                    "subject_id": "",
                    "model": "",
                    "seed": int(config["seed"]),
                    "status": "missing_dataset_directory",
                    "expected_locator": dataset_cfg["dataset_locator"],
                    "runtime_checked_path": stable_locator(dataset_dir),
                }
            )
            dataset_subjects[dataset_id] = []
            continue
        explicit_subjects = list(dataset_cfg.get("subject_ids", []))
        if explicit_subjects:
            subjects = explicit_subjects
        else:
            subjects = list_reference_subjects(dataset_dir, split="test")
        dataset_subjects[dataset_id] = subjects

    planned_jobs = sum(len(subjects) * len(config["models"]) for subjects in dataset_subjects.values())
    skipped_jobs = 0
    run_jobs = 0
    protocol_mismatch_jobs = 0

    for dataset_id, subjects in dataset_subjects.items():
        dataset_dir = dataset_path_lookup[dataset_id]
        dataset_cfg = dataset_cfg_lookup[dataset_id]
        if not dataset_dir.exists():
            continue
        for subject_id in subjects:
            for model_name in config["models"]:
                fingerprint, fingerprint_payload = fingerprint_for_job(config, dataset_cfg, subject_id, model_name)
                job_dir = jobs_root / dataset_id / subject_id / model_name / f"seed_{config['seed']}"
                manifest_path = job_dir / "job_manifest.json"
                recording_path = job_dir / "recording_metrics.csv"
                subject_path = job_dir / "subject_metrics.csv"
                ledger_entry = {
                    "dataset": dataset_id,
                    "subject_id": subject_id,
                    "model": model_name,
                    "seed": int(config["seed"]),
                    "fingerprint": fingerprint,
                    "job_dir": repo_relative(job_dir),
                }

                if manifest_path.exists() and not args.force_rerun:
                    existing_manifest = load_json(manifest_path)
                    existing_fingerprint = existing_manifest.get("fingerprint")
                    metrics_exist = recording_path.exists() and subject_path.exists()
                    if existing_fingerprint == fingerprint and metrics_exist and skip_existing:
                        ledger_entry.update(
                            {
                                "action": "skip_existing",
                                "status": "success",
                                "checkpoint_id": existing_manifest.get("checkpoint_id", ""),
                                "subject_metric": existing_manifest.get("subject_metric", ""),
                                "num_recordings": existing_manifest.get("num_recordings", ""),
                                "error": "",
                            }
                        )
                        job_rows.append(ledger_entry)
                        skipped_jobs += 1
                        continue
                    if existing_fingerprint != fingerprint:
                        ledger_entry.update(
                            {
                                "action": "protocol_mismatch",
                                "status": "protocol_mismatch",
                                "checkpoint_id": "",
                                "subject_metric": "",
                                "num_recordings": "",
                                "error": "existing job fingerprint differs from requested fingerprint",
                            }
                        )
                        failures.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model_name,
                                "seed": int(config["seed"]),
                                "status": "protocol_mismatch",
                                "job_dir": repo_relative(job_dir),
                                "existing_fingerprint": existing_fingerprint,
                                "requested_fingerprint": fingerprint,
                            }
                        )
                        job_rows.append(ledger_entry)
                        protocol_mismatch_jobs += 1
                        continue

                if args.force_rerun and job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                ensure_dir(job_dir)

                try:
                    recording_rows, subject_rows, meta = run_job(
                        config=config,
                        dataset_cfg=dataset_cfg,
                        dataset_dir=dataset_dir,
                        subject_id=subject_id,
                        model_name=model_name,
                        device=device,
                        output_dir=output_dir,
                    )
                    write_rows(recording_path, recording_rows, RECORDING_METRIC_FIELDS)
                    write_rows(subject_path, subject_rows, SUBJECT_METRIC_FIELDS)
                    subject_metric_value = float(subject_rows[0].metric_value) if subject_rows else float("nan")
                    num_recordings = int(subject_rows[0].num_recordings) if subject_rows else 0
                    job_manifest = {
                        "dataset": dataset_id,
                        "dataset_locator": dataset_cfg["dataset_locator"],
                        "subject_id": subject_id,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "protocol": config["protocol"],
                        "artifact_scope": config["artifact_scope"],
                        "fingerprint": fingerprint,
                        "fingerprint_payload": fingerprint_payload,
                        "scorer_id": config["scorer_id"],
                        "aggregation_rule": config["aggregation_rule"],
                        "checkpoint_id": meta.get("checkpoint_id"),
                        "subject_metric": subject_metric_value,
                        "num_recordings": num_recordings,
                        "status": "success",
                        "meta": meta,
                        "artifacts": {
                            "recording_metrics": repo_relative(recording_path),
                            "subject_metrics": repo_relative(subject_path),
                        },
                    }
                    write_json(manifest_path, job_manifest)
                    ledger_entry.update(
                        {
                            "action": "run",
                            "status": "success",
                            "checkpoint_id": meta.get("checkpoint_id", ""),
                            "subject_metric": subject_metric_value,
                            "num_recordings": num_recordings,
                            "error": "",
                        }
                    )
                    job_rows.append(ledger_entry)
                    run_jobs += 1
                except Exception as exc:
                    failure_payload = {
                        "dataset": dataset_id,
                        "subject_id": subject_id,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "status": "failed",
                        "error": repr(exc),
                        "job_dir": repo_relative(job_dir),
                    }
                    failures.append(failure_payload)
                    write_json(
                        manifest_path,
                        {
                            "dataset": dataset_id,
                            "subject_id": subject_id,
                            "model": model_name,
                            "seed": int(config["seed"]),
                            "protocol": config["protocol"],
                            "artifact_scope": config["artifact_scope"],
                            "fingerprint": fingerprint,
                            "fingerprint_payload": fingerprint_payload,
                            "status": "failed",
                            "error": repr(exc),
                        },
                    )
                    ledger_entry.update(
                        {
                            "action": "run",
                            "status": "failed",
                            "checkpoint_id": "",
                            "subject_metric": "",
                            "num_recordings": "",
                            "error": repr(exc),
                        }
                    )
                    job_rows.append(ledger_entry)

    aggregated_recording_rows: list[dict[str, str]] = []
    aggregated_subject_rows: list[dict[str, str]] = []
    for row in job_rows:
        if row["status"] != "success":
            continue
        job_dir = ROOT / row["job_dir"]
        recording_path = job_dir / "recording_metrics.csv"
        subject_path = job_dir / "subject_metrics.csv"
        if recording_path.exists():
            aggregated_recording_rows.extend(read_csv_rows(recording_path))
        if subject_path.exists():
            aggregated_subject_rows.extend(read_csv_rows(subject_path))

    write_generic_csv(output_dir / "recording_metrics.csv", aggregated_recording_rows, RECORDING_METRIC_FIELDS)
    write_generic_csv(output_dir / "subject_metrics.csv", aggregated_subject_rows, SUBJECT_METRIC_FIELDS)

    actual_dataset_subjects = infer_dataset_subjects_from_subject_rows(aggregated_subject_rows)
    cumulative_successful_jobs = len(
        [
            row
            for row in job_rows
            if row["status"] == "success"
        ]
    )
    dataset_metrics = summarize_dataset_metrics(aggregated_subject_rows)
    dataset_metric_fields = [
        "dataset",
        "model",
        "n_subjects",
        "mean_pearson",
        "std_pearson",
        "median_pearson",
        "min_pearson",
        "max_pearson",
        "fisher_z_mean",
        "backtransformed_mean_r",
    ]
    write_generic_csv(output_dir / "dataset_metrics.csv", dataset_metrics, dataset_metric_fields)

    etard_condition_metrics = summarize_etard_condition_metrics(aggregated_recording_rows)
    etard_condition_fields = [
        "dataset",
        "model",
        "condition",
        "n_subjects",
        "n_recordings",
        "mean_pearson",
        "median_pearson",
        "std_pearson",
        "negative_recording_count",
    ]
    write_generic_csv(output_dir / "etard_condition_metrics.csv", etard_condition_metrics, etard_condition_fields)

    job_ledger_json = {
        "protocol": config["protocol"],
        "device": device,
        "resume_enabled": resume_enabled,
        "skip_existing": skip_existing,
        "force_rerun": bool(args.force_rerun),
        "rows": job_rows,
    }
    write_json(output_dir / "job_ledger.json", job_ledger_json)
    write_job_ledger_csv(output_dir / "job_ledger.csv", job_rows)

    run_manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "previous_fix_branch": config["previous_fix_branch"],
        "previous_fix_commit": config["previous_fix_commit"],
        "current_branch": config["current_branch"],
        "device": device,
        "seed": int(config["seed"]),
        "resume_enabled": resume_enabled,
        "skip_existing": skip_existing,
        "force_rerun": bool(args.force_rerun),
        "datasets_requested": [item["dataset_id"] for item in config["datasets"]],
        "models_requested": list(config["models"]),
        "dataset_subjects": actual_dataset_subjects,
        "configured_dataset_subjects": dataset_subjects,
        "planned_jobs": sum(len(subjects) * len(config["models"]) for subjects in actual_dataset_subjects.values()),
        "cumulative_successful_jobs": cumulative_successful_jobs,
        "skipped_existing_jobs": skipped_jobs,
        "protocol_mismatch_jobs": protocol_mismatch_jobs,
        "run_jobs": run_jobs,
        "failed_jobs": len([item for item in failures if item["status"] == "failed"]),
        "failures": failures,
        "artifacts": {
            "job_ledger_csv": repo_relative(output_dir / "job_ledger.csv"),
            "job_ledger_json": repo_relative(output_dir / "job_ledger.json"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
            "etard_condition_metrics": repo_relative(output_dir / "etard_condition_metrics.csv"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "result_summary": repo_relative(output_dir / "result_summary.md"),
            "incremental_comparison": repo_relative(output_dir / "incremental_comparison.md"),
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "failure_report.json", {"failures": failures})

    write_result_summary(
        output_dir / "result_summary.md",
        config=config,
        device=device,
        dataset_subjects=actual_dataset_subjects,
        expected_jobs=sum(len(subjects) * len(config["models"]) for subjects in actual_dataset_subjects.values()),
        cumulative_successful_jobs=cumulative_successful_jobs,
        skipped_jobs=skipped_jobs,
        protocol_mismatch_jobs=protocol_mismatch_jobs,
        run_jobs=run_jobs,
        failed_jobs=len([item for item in failures if item["status"] == "failed"]),
        dataset_metrics=dataset_metrics,
        etard_condition_metrics=etard_condition_metrics,
    )
    write_incremental_comparison(
        output_dir / "incremental_comparison.md",
        previous_fix_commit=config["previous_fix_commit"],
        current_branch=config["current_branch"],
        dataset_subjects=actual_dataset_subjects,
        cumulative_successful_jobs=cumulative_successful_jobs,
        run_jobs=run_jobs,
        skipped_jobs=skipped_jobs,
        failed_jobs=len([item for item in failures if item["status"] == "failed"]),
    )
    update_paper_results(ROOT / "paper" / "draft_zh" / "sections" / "results_placeholder_zh.tex", dataset_metrics, etard_condition_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
