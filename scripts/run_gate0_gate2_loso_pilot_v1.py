from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.mldecoders.cca import trim_valid_range
from repro.reference_baselines import list_reference_subjects, load_reference_recordings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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


def concatenate_split_for_subjects(
    dataset_dir: Path,
    split: str,
    subjects: list[str],
    *,
    channels: range,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    eeg_parts: list[np.ndarray] = []
    env_parts: list[np.ndarray] = []
    sample_counts: dict[str, int] = {}
    for subject_id in subjects:
        subject_total = 0
        for _, eeg, env in load_reference_recordings(dataset_dir, split, subject_id, channels=channels):
            eeg_parts.append(eeg.astype(np.float32))
            env_parts.append(env.astype(np.float32))
            subject_total += int(eeg.shape[0])
        sample_counts[subject_id] = subject_total
    if not eeg_parts:
        raise ValueError(f"no recordings found for split={split} subjects={subjects}")
    return np.concatenate(eeg_parts, axis=0), np.concatenate(env_parts, axis=0), sample_counts


def accumulate_gram_for_subjects(
    dataset_dir: Path,
    split: str,
    subjects: list[str],
    *,
    channels: range,
    start_lag: int,
    end_lag: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    xtx = None
    xty = None
    sample_counts: dict[str, int] = {}
    for subject_id in subjects:
        subject_total = 0
        for _, eeg, env in load_reference_recordings(dataset_dir, split, subject_id, channels=channels):
            x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
            x_block = x_lag.astype(np.float64, copy=False)
            y_block = y_center.astype(np.float64, copy=False)
            if xtx is None:
                xtx = np.zeros((x_block.shape[1], x_block.shape[1]), dtype=np.float64)
                xty = np.zeros(x_block.shape[1], dtype=np.float64)
            xtx += x_block.T @ x_block
            xty += x_block.T @ y_block
            subject_total += int(x_block.shape[0])
        sample_counts[subject_id] = subject_total
    if xtx is None or xty is None:
        raise ValueError(f"no lagged training data found for split={split} subjects={subjects}")
    return xtx, xty, sample_counts


def lagged_validation_parts_for_subjects(
    dataset_dir: Path,
    split: str,
    subjects: list[str],
    *,
    channels: range,
    start_lag: int,
    end_lag: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    sample_counts: dict[str, int] = {}
    for subject_id in subjects:
        subject_total = 0
        for _, eeg, env in load_reference_recordings(dataset_dir, split, subject_id, channels=channels):
            x_lag, _, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
            x_block = x_lag.astype(np.float64, copy=False)
            y_block = y_center.astype(np.float64, copy=False)
            parts.append((x_block, y_block))
            subject_total += int(x_block.shape[0])
        sample_counts[subject_id] = subject_total
    if not parts:
        raise ValueError(f"no lagged validation data found for split={split} subjects={subjects}")
    return parts, sample_counts


def fit_loso_ridge_for_subject(
    config: dict,
    dataset_dir: Path,
    heldout_subject: str,
) -> tuple[list[WindowPrediction], dict[str, object], dict[str, object]]:
    dataset_cfg = config["dataset"]
    ridge_cfg = config["ridge"]
    dataset_id = dataset_cfg["dataset_id"]
    sampling_rate = int(dataset_cfg["sampling_rate"])
    all_subjects = list_reference_subjects(dataset_dir, split="test")
    train_subjects = sorted(subject for subject in all_subjects if subject != heldout_subject)
    if heldout_subject not in all_subjects:
        raise ValueError(f"heldout subject {heldout_subject} not found in dataset test split")

    start_lag = int(ridge_cfg["start_lag"])
    end_lag = int(ridge_cfg["end_lag"])
    channels = range(64)

    xtx, xty, train_sample_counts = accumulate_gram_for_subjects(
        dataset_dir,
        "train",
        train_subjects,
        channels=channels,
        start_lag=start_lag,
        end_lag=end_lag,
    )
    val_parts, val_sample_counts = lagged_validation_parts_for_subjects(
        dataset_dir,
        "val",
        train_subjects,
        channels=channels,
        start_lag=start_lag,
        end_lag=end_lag,
    )

    best_model = None
    best_alpha = None
    best_val_pearson = None
    alpha_rows: list[dict[str, float]] = []
    identity = np.eye(xtx.shape[0], dtype=np.float64)
    for alpha in list(ridge_cfg["alphas"]):
        coef = np.linalg.solve(xtx + float(alpha) * identity, xty)
        val_preds = []
        val_targets = []
        for x_val_part, y_val_part in val_parts:
            val_preds.append((x_val_part @ coef).astype(np.float64, copy=False))
            val_targets.append(y_val_part)
        val_pred = np.concatenate(val_preds, axis=0)
        y_val_center = np.concatenate(val_targets, axis=0)
        val_pearson = float(pearsonr(val_pred, y_val_center)[0])
        alpha_rows.append({"alpha": float(alpha), "val_pearson": val_pearson})
        if best_val_pearson is None or val_pearson > best_val_pearson:
            best_model = coef.copy()
            best_alpha = float(alpha)
            best_val_pearson = val_pearson
    if best_model is None or best_alpha is None or best_val_pearson is None:
        raise RuntimeError(f"failed to fit LOSO ridge for {heldout_subject}")

    checkpoint_id = f"ridge_alpha_{best_alpha}"
    offset = end_lag - 1
    windows: list[WindowPrediction] = []
    test_scores: list[float] = []
    test_recordings = load_reference_recordings(dataset_dir, "test", heldout_subject, channels=channels)
    for recording_id, eeg, env in test_recordings:
        x_test_lag, _, y_test_center = trim_valid_range(eeg, env, start_lag, end_lag)
        pred = (x_test_lag.astype(np.float64, copy=False) @ best_model).astype(np.float32)
        target = y_test_center.astype(np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_id,
                model="ridge",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=heldout_subject,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                checkpoint_id=checkpoint_id,
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=target,
            )
        )
        test_scores.append(correlation(pred, target))

    leakage_entry = {
        "heldout_subject": heldout_subject,
        "all_subjects_test_split": all_subjects,
        "train_subjects": train_subjects,
        "val_subjects": train_subjects,
        "test_subjects": [heldout_subject],
        "excluded_target_from_train": heldout_subject not in train_subjects,
        "excluded_target_from_val": heldout_subject not in train_subjects,
        "excluded_target_from_test_pool": True,
        "train_sample_counts": train_sample_counts,
        "val_sample_counts": val_sample_counts,
        "test_recording_count": len(test_recordings),
        "selection_rule": ridge_cfg["selection_rule"],
        "alpha_grid": list(ridge_cfg["alphas"]),
        "alpha_validation_scores": alpha_rows,
    }
    meta = {
        "model": "ridge",
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "best_alpha": best_alpha,
        "val_pearson": best_val_pearson,
        "test_pearson_direct": float(np.mean(test_scores)),
    }
    return windows, meta, leakage_entry


def validate_schema(
    *,
    config: dict,
    heldout_subjects: list[str],
    recording_rows: list[object],
    subject_rows: list[object],
    leakage_checks: list[dict[str, object]],
) -> dict[str, object]:
    subject_metric_fields_consistent = all(sorted(row.__dict__.keys()) == sorted(SUBJECT_METRIC_FIELDS) for row in subject_rows)
    recording_metric_fields_consistent = all(sorted(row.__dict__.keys()) == sorted(RECORDING_METRIC_FIELDS) for row in recording_rows)
    observed_subjects = sorted(row.subject_id for row in subject_rows)
    subject_complete = observed_subjects == sorted(heldout_subjects)
    pearson_subject_ok = all(-1.0 <= float(row.metric_value) <= 1.0 for row in subject_rows)
    pearson_recording_ok = all(-1.0 <= float(row.metric_value) <= 1.0 for row in recording_rows)
    leakage_ok = all(
        bool(entry["excluded_target_from_train"]) and bool(entry["excluded_target_from_val"]) and entry["test_subjects"] == [entry["heldout_subject"]]
        for entry in leakage_checks
    )
    return {
        "protocol": config["protocol"],
        "passed": bool(
            subject_metric_fields_consistent
            and recording_metric_fields_consistent
            and subject_complete
            and pearson_subject_ok
            and pearson_recording_ok
            and leakage_ok
        ),
        "checks": {
            "heldout_subjects_exact": subject_complete,
            "subject_metric_fields_consistent": subject_metric_fields_consistent,
            "recording_metric_fields_consistent": recording_metric_fields_consistent,
            "pearson_ranges_subject_metrics": pearson_subject_ok,
            "pearson_ranges_recording_metrics": pearson_recording_ok,
            "target_subject_excluded_from_train_val": leakage_ok,
        },
        "details": {
            "expected_heldout_subjects": heldout_subjects,
            "observed_subjects": observed_subjects,
            "recording_rows": len(recording_rows),
            "subject_rows": len(subject_rows),
        },
    }


def write_leakage_check(path: Path, *, config: dict, leakage_checks: list[dict[str, object]]) -> None:
    lines = [
        "# Leakage Check",
        "",
        f"- `protocol`: `{config['protocol']}`",
        f"- `dataset`: `{config['dataset']['dataset_id']}`",
        f"- `model`: `{config['model']}`",
        "",
        "This pilot uses pure LOSO subject-independent fitting:",
        "- train uses only non-heldout subjects' `train` split",
        "- validation uses only non-heldout subjects' `val` split",
        "- test uses only heldout subject `test` split",
        "- heldout subject train/val/test data are excluded from fit, scaler, and hyperparameter selection",
        "",
    ]
    for entry in leakage_checks:
        lines.extend(
            [
                f"## Held-out `{entry['heldout_subject']}`",
                f"- train subjects: `{', '.join(entry['train_subjects'])}`",
                f"- val subjects: `{', '.join(entry['val_subjects'])}`",
                f"- test subjects: `{', '.join(entry['test_subjects'])}`",
                f"- excluded target from train: `{entry['excluded_target_from_train']}`",
                f"- excluded target from val: `{entry['excluded_target_from_val']}`",
                f"- test recording count: `{entry['test_recording_count']}`",
                f"- selected alpha by pooled validation: `{max(entry['alpha_validation_scores'], key=lambda item: item['val_pearson'])['alpha']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_result_summary(path: Path, *, config: dict, recording_rows: list[object], subject_rows: list[object], leakage_checks: list[dict[str, object]]) -> None:
    lines = [
        "# Gate0-Gate2 LOSO Pilot v1",
        "",
        "This directory contains a minimal subject-independent LOSO pilot for `weissbart_tf64` using only `ridge` on held-out `P00`, `P01`, and `P02`.",
        "It is execution evidence for the LOSO interface and does not add new models or paper prose.",
        "",
        "## Protocol",
        f"- `protocol`: `{config['protocol']}`",
        f"- `dataset`: `{config['dataset']['dataset_id']}`",
        f"- `model`: `{config['model']}`",
        f"- `heldout_subjects`: `{', '.join(config['heldout_subjects'])}`",
        f"- `seed`: `{config['seed']}`",
        "",
        "## Subject metrics",
        "| subject | mean_recording_pearson_r | num_recordings | checkpoint_id |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in subject_rows:
        lines.append(
            f"| {row.subject_id} | {float(row.metric_value):.6f} | {int(row.num_recordings)} | {row.checkpoint_id} |"
        )
    lines.extend(
        [
            "",
            "## Recording rows",
            f"- total recording metric rows: `{len(recording_rows)}`",
            f"- total subject metric rows: `{len(subject_rows)}`",
            "",
            "## Leakage status",
        ]
    )
    for entry in leakage_checks:
        lines.append(
            f"- `{entry['heldout_subject']}`: train/val exclude target=`{entry['excluded_target_from_train'] and entry['excluded_target_from_val']}`, "
            f"test subjects=`{', '.join(entry['test_subjects'])}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)

    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"missing dataset locator: {dataset_dir}")

    all_windows: list[WindowPrediction] = []
    meta_rows: list[dict[str, object]] = []
    leakage_checks: list[dict[str, object]] = []
    for heldout_subject in config["heldout_subjects"]:
        windows, meta, leakage_entry = fit_loso_ridge_for_subject(config, dataset_dir, heldout_subject)
        all_windows.extend(windows)
        meta_rows.append({"heldout_subject": heldout_subject, **meta})
        leakage_checks.append(leakage_entry)

    recording_rows, subject_rows = aggregate_model_outputs(all_windows, config["artifact_scope"])
    write_rows(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", subject_rows, SUBJECT_METRIC_FIELDS)

    schema_validation = validate_schema(
        config=config,
        heldout_subjects=list(config["heldout_subjects"]),
        recording_rows=recording_rows,
        subject_rows=subject_rows,
        leakage_checks=leakage_checks,
    )
    with open(output_dir / "schema_validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(schema_validation, handle, indent=2)

    write_leakage_check(output_dir / "leakage_check.md", config=config, leakage_checks=leakage_checks)
    write_result_summary(
        output_dir / "result_summary.md",
        config=config,
        recording_rows=recording_rows,
        subject_rows=subject_rows,
        leakage_checks=leakage_checks,
    )

    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": config["protocol"],
                "artifact_scope": config["artifact_scope"],
                "previous_fix_branch": config["previous_fix_branch"],
                "previous_fix_commit": config["previous_fix_commit"],
                "current_branch": config["current_branch"],
                "dataset_id": config["dataset"]["dataset_id"],
                "dataset_locator": config["dataset"]["dataset_locator"],
                "sampling_rate": int(config["dataset"]["sampling_rate"]),
                "model": config["model"],
                "seed": int(config["seed"]),
                "heldout_subjects": list(config["heldout_subjects"]),
                "job_accounting": {
                    "planned_jobs": len(config["heldout_subjects"]),
                    "successful_jobs": len(meta_rows),
                    "failed_jobs": 0,
                },
                "model_runs": meta_rows,
                "artifacts": {
                    "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
                    "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
                    "leakage_check": repo_relative(output_dir / "leakage_check.md"),
                    "result_summary": repo_relative(output_dir / "result_summary.md"),
                    "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
                },
            },
            handle,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
