from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, load_dict_rows, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.mldecoders.cca import trim_valid_range
from repro.reference_baselines import list_reference_subjects, load_reference_recordings
from run_gate0_gate2_full_subject_single_seed import condition_from_recording_id, summarize_dataset_metrics, summarize_etard_condition_metrics, write_generic_csv


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


def accumulate_gram_for_split(dataset_dir: Path, split: str, subjects: list[str], *, channels: range, start_lag: int, end_lag: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
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


def load_validation_parts(dataset_dir: Path, subjects: list[str], *, channels: range, start_lag: int, end_lag: int) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], dict[str, int]]:
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


def subject_mean_pearson_from_recording_rows(recording_rows: list[object]) -> float:
    values = [float(row.metric_value) for row in recording_rows]
    return float(np.mean(values))


def fit_loso_ridge_for_dataset_subject(
    *,
    config: dict,
    dataset_cfg: dict,
    dataset_dir: Path,
    all_subjects: list[str],
    heldout_subject: str,
    train_total_xtx: np.ndarray,
    train_total_xty: np.ndarray,
    train_subject_xtx: dict[str, np.ndarray],
    train_subject_xty: dict[str, np.ndarray],
    val_parts_by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    train_sample_counts: dict[str, int],
    val_sample_counts: dict[str, int],
) -> tuple[list[WindowPrediction], list[object], list[object], dict[str, object], dict[str, object]]:
    ridge_cfg = config["ridge"]
    start_lag = int(ridge_cfg["start_lag"])
    end_lag = int(ridge_cfg["end_lag"])
    channels = range(64)

    loso_train_xtx = train_total_xtx - train_subject_xtx[heldout_subject]
    loso_train_xty = train_total_xty - train_subject_xty[heldout_subject]
    train_subjects = [subject for subject in all_subjects if subject != heldout_subject]
    val_subjects = train_subjects

    best_coef = None
    best_alpha = None
    best_val_pearson = None
    alpha_rows: list[dict[str, float]] = []
    identity = np.eye(loso_train_xtx.shape[0], dtype=np.float64)
    for alpha in list(ridge_cfg["alphas"]):
        coef = np.linalg.solve(loso_train_xtx + float(alpha) * identity, loso_train_xty)
        val_preds = []
        val_targets = []
        for subject_id in val_subjects:
            for x_val_part, y_val_part in val_parts_by_subject[subject_id]:
                val_preds.append((x_val_part @ coef).astype(np.float64, copy=False))
                val_targets.append(y_val_part)
        val_pred = np.concatenate(val_preds, axis=0)
        val_target = np.concatenate(val_targets, axis=0)
        val_pearson = float(pearsonr(val_pred, val_target)[0])
        alpha_rows.append({"alpha": float(alpha), "val_pearson": val_pearson})
        if best_val_pearson is None or val_pearson > best_val_pearson:
            best_coef = coef.copy()
            best_alpha = float(alpha)
            best_val_pearson = val_pearson
    if best_coef is None or best_alpha is None or best_val_pearson is None:
        raise RuntimeError(f"failed to fit LOSO ridge for {dataset_cfg['dataset_id']} {heldout_subject}")

    checkpoint_id = f"ridge_alpha_{best_alpha}"
    offset = end_lag - 1
    windows: list[WindowPrediction] = []
    test_scores: list[float] = []
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", heldout_subject, channels=channels):
        x_test_lag, _, y_test_center = trim_valid_range(eeg, env, start_lag, end_lag)
        pred = (x_test_lag.astype(np.float64, copy=False) @ best_coef).astype(np.float32)
        target = y_test_center.astype(np.float32)
        windows.append(
            build_series_window(
                dataset=dataset_cfg["dataset_id"],
                model="ridge",
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=heldout_subject,
                recording_id=recording_id,
                sampling_rate=int(dataset_cfg["sampling_rate"]),
                checkpoint_id=checkpoint_id,
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=target,
            )
        )
        test_scores.append(correlation(pred, target))

    recording_rows, subject_rows = aggregate_model_outputs(windows, config["artifact_scope"])
    leakage_entry = {
        "dataset": dataset_cfg["dataset_id"],
        "heldout_subject": heldout_subject,
        "all_subjects": all_subjects,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": [heldout_subject],
        "excluded_target_from_train": heldout_subject not in train_subjects,
        "excluded_target_from_val": heldout_subject not in val_subjects,
        "excluded_target_from_selection": heldout_subject not in val_subjects,
        "train_sample_count_without_target": int(sum(train_sample_counts[s] for s in train_subjects)),
        "val_sample_count_without_target": int(sum(val_sample_counts[s] for s in val_subjects)),
        "test_recording_count": len(recording_rows),
        "alpha_grid": list(ridge_cfg["alphas"]),
        "alpha_validation_scores": alpha_rows,
        "selected_alpha": best_alpha,
    }
    meta = {
        "dataset": dataset_cfg["dataset_id"],
        "heldout_subject": heldout_subject,
        "model": "ridge",
        "status": "success",
        "checkpoint_id": checkpoint_id,
        "best_alpha": best_alpha,
        "val_pearson": best_val_pearson,
        "test_pearson_direct": float(np.mean(test_scores)),
        "subject_metric": subject_mean_pearson_from_recording_rows(recording_rows),
        "num_recordings": len(recording_rows),
    }
    return windows, recording_rows, subject_rows, meta, leakage_entry


def load_subject_specific_reference(config: dict) -> dict[tuple[str, str], dict[str, str]]:
    rows = load_dict_rows(ROOT / config["subject_specific_reference"]["subject_metrics"])
    target_protocol = config["subject_specific_reference"]["protocol"]
    target_model = config["subject_specific_reference"]["model"]
    target_seed = str(config["subject_specific_reference"]["seed"])
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["protocol"] != target_protocol:
            continue
        if row["model"] != target_model:
            continue
        if row["seed"] != target_seed:
            continue
        if row["metric_name"] != "mean_recording_pearson_r":
            continue
        lookup[(row["dataset"], row["subject_id"])] = row
    return lookup


def write_loso_vs_subject_specific_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
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
        "This file compares the current LOSO ridge result against the existing subject-specific ridge baseline from `gate0_gate2_full_subject_single_seed_v1`.",
        "",
        "| dataset | subject | loso_metric | subject_specific_metric | delta_loso_minus_subject_specific |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['subject_id']} | {float(row['loso_metric']):.6f} | {float(row['subject_specific_metric']):.6f} | {float(row['delta_loso_minus_subject_specific']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leakage_summary(path: Path, *, config: dict, leakage_entries: list[dict[str, object]]) -> None:
    lines = [
        "# Leakage Summary",
        "",
        f"- `protocol`: `{config['protocol']}`",
        f"- `seed`: `{config['seed']}`",
        f"- `model`: `{config['model']}`",
        "",
        "Pure LOSO guarantees in this round:",
        "- held-out target subject is excluded from train",
        "- held-out target subject is excluded from val",
        "- held-out target subject is excluded from alpha selection",
        "- held-out target subject appears only in test",
        "",
    ]
    for entry in leakage_entries:
        lines.extend(
            [
                f"## `{entry['dataset']}` / `{entry['heldout_subject']}`",
                f"- train subjects: `{', '.join(entry['train_subjects'])}`",
                f"- val subjects: `{', '.join(entry['val_subjects'])}`",
                f"- test subjects: `{', '.join(entry['test_subjects'])}`",
                f"- excluded target from train: `{entry['excluded_target_from_train']}`",
                f"- excluded target from val: `{entry['excluded_target_from_val']}`",
                f"- excluded target from alpha selection: `{entry['excluded_target_from_selection']}`",
                f"- selected alpha: `{entry['selected_alpha']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_schema(
    *,
    config: dict,
    dataset_subjects: dict[str, list[str]],
    recording_rows: list[object],
    subject_rows: list[object],
    dataset_metrics: list[dict[str, object]],
    etard_condition_metrics: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    leakage_entries: list[dict[str, object]],
) -> dict[str, object]:
    subject_metric_fields_consistent = all(sorted(row.__dict__.keys()) == sorted(SUBJECT_METRIC_FIELDS) for row in subject_rows)
    recording_metric_fields_consistent = all(sorted(row.__dict__.keys()) == sorted(RECORDING_METRIC_FIELDS) for row in recording_rows)
    pearson_subject_ok = all(-1.0 <= float(row.metric_value) <= 1.0 for row in subject_rows)
    pearson_recording_ok = all(-1.0 <= float(row.metric_value) <= 1.0 for row in recording_rows)
    pearson_dataset_ok = all(
        all(-1.0 <= float(row[field]) <= 1.0 for field in ["mean_pearson", "std_pearson", "median_pearson", "min_pearson", "max_pearson", "backtransformed_mean_r"])
        for row in dataset_metrics
    )
    pearson_condition_ok = all(
        all(-1.0 <= float(row[field]) <= 1.0 for field in ["mean_pearson", "median_pearson", "std_pearson"])
        for row in etard_condition_metrics
    )
    heldout_exact = sorted((row.dataset, row.subject_id) for row in subject_rows) == sorted(
        (dataset_id, subject_id)
        for dataset_id, subjects in dataset_subjects.items()
        for subject_id in subjects
    )
    leakage_ok = all(
        entry["excluded_target_from_train"]
        and entry["excluded_target_from_val"]
        and entry["excluded_target_from_selection"]
        and entry["test_subjects"] == [entry["heldout_subject"]]
        for entry in leakage_entries
    )
    comparison_complete = len(comparison_rows) == len(subject_rows)
    return {
        "protocol": config["protocol"],
        "passed": bool(
            subject_metric_fields_consistent
            and recording_metric_fields_consistent
            and pearson_subject_ok
            and pearson_recording_ok
            and pearson_dataset_ok
            and pearson_condition_ok
            and heldout_exact
            and leakage_ok
            and comparison_complete
        ),
        "checks": {
            "heldout_subject_inventory_exact": heldout_exact,
            "subject_metric_fields_consistent": subject_metric_fields_consistent,
            "recording_metric_fields_consistent": recording_metric_fields_consistent,
            "pearson_ranges_subject_metrics": pearson_subject_ok,
            "pearson_ranges_recording_metrics": pearson_recording_ok,
            "pearson_ranges_dataset_metrics": pearson_dataset_ok,
            "pearson_ranges_etard_condition_metrics": pearson_condition_ok,
            "target_subject_excluded_from_train_val_selection": leakage_ok,
            "comparison_rows_complete": comparison_complete,
        },
        "details": {
            "dataset_subjects": dataset_subjects,
            "recording_rows": len(recording_rows),
            "subject_rows": len(subject_rows),
            "dataset_metric_rows": len(dataset_metrics),
            "etard_condition_rows": len(etard_condition_metrics),
            "comparison_rows": len(comparison_rows),
        },
    }


def write_result_summary(path: Path, *, config: dict, dataset_subjects: dict[str, list[str]], dataset_metrics: list[dict[str, object]], etard_condition_metrics: list[dict[str, object]], comparison_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Gate0-Gate2 LOSO Ridge Full v1",
        "",
        "This directory extends the validated LOSO ridge interface to all subjects on `weissbart_tf64` and `etard_tf64` with seed `0` only.",
        "It is a pure subject-independent LOSO ridge round and does not add deep models or paper prose.",
        "",
        "## Protocol",
        f"- `protocol`: `{config['protocol']}`",
        f"- `model`: `{config['model']}`",
        f"- `seed`: `{config['seed']}`",
        "",
        "## Subject inventory",
    ]
    for dataset_id, subjects in dataset_subjects.items():
        lines.append(f"- `{dataset_id}` ({len(subjects)} subjects): `{', '.join(subjects)}`")

    lines.extend(
        [
            "",
            "## Dataset metrics",
            "| dataset | model | n_subjects | mean_pearson | std_pearson | median_pearson | min_pearson | max_pearson |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_metrics:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_subjects']} | {row['mean_pearson']:.6f} | {row['std_pearson']:.6f} | "
            f"{row['median_pearson']:.6f} | {row['min_pearson']:.6f} | {row['max_pearson']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Etard condition metrics",
            "| model | condition | n_subjects | n_recordings | mean_pearson | median_pearson | std_pearson | negative_recording_count |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in etard_condition_metrics:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['n_subjects']} | {row['n_recordings']} | {row['mean_pearson']:.6f} | "
            f"{row['median_pearson']:.6f} | {row['std_pearson']:.6f} | {row['negative_recording_count']} |"
        )

    deltas = np.asarray([float(row["delta_loso_minus_subject_specific"]) for row in comparison_rows], dtype=np.float64)
    lines.extend(
        [
            "",
            "## LOSO vs subject-specific ridge",
            f"- compared rows: `{len(comparison_rows)}`",
            f"- mean delta (LOSO - subject-specific): `{float(np.mean(deltas)):.6f}`",
            f"- min delta: `{float(np.min(deltas)):.6f}`",
            f"- max delta: `{float(np.max(deltas)):.6f}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)

    all_recording_rows: list[object] = []
    all_subject_rows: list[object] = []
    model_runs: list[dict[str, object]] = []
    leakage_entries: list[dict[str, object]] = []
    dataset_subjects: dict[str, list[str]] = {}

    for dataset_cfg in config["datasets"]:
        dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
        if not dataset_dir.exists():
            raise FileNotFoundError(f"missing dataset locator: {dataset_dir}")
        if dataset_cfg.get("subject_selection") == "manual":
            all_subjects = list(dataset_cfg["subject_ids"])
        else:
            all_subjects = list_reference_subjects(dataset_dir, split="test")
        dataset_subjects[dataset_cfg["dataset_id"]] = all_subjects

        start_lag = int(config["ridge"]["start_lag"])
        end_lag = int(config["ridge"]["end_lag"])
        channels = range(64)
        train_total_xtx, train_total_xty, train_subject_xtx, train_subject_xty, train_sample_counts = accumulate_gram_for_split(
            dataset_dir,
            "train",
            all_subjects,
            channels=channels,
            start_lag=start_lag,
            end_lag=end_lag,
        )
        val_parts_by_subject, val_sample_counts = load_validation_parts(
            dataset_dir,
            all_subjects,
            channels=channels,
            start_lag=start_lag,
            end_lag=end_lag,
        )

        for heldout_subject in all_subjects:
            _windows, recording_rows, subject_rows, meta, leakage_entry = fit_loso_ridge_for_dataset_subject(
                config=config,
                dataset_cfg=dataset_cfg,
                dataset_dir=dataset_dir,
                all_subjects=all_subjects,
                heldout_subject=heldout_subject,
                train_total_xtx=train_total_xtx,
                train_total_xty=train_total_xty,
                train_subject_xtx=train_subject_xtx,
                train_subject_xty=train_subject_xty,
                val_parts_by_subject=val_parts_by_subject,
                train_sample_counts=train_sample_counts,
                val_sample_counts=val_sample_counts,
            )
            all_recording_rows.extend(recording_rows)
            all_subject_rows.extend(subject_rows)
            model_runs.append(meta)
            leakage_entries.append(leakage_entry)

    write_rows(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)

    recording_dict_rows = [row.__dict__ for row in all_recording_rows]
    subject_dict_rows = [row.__dict__ for row in all_subject_rows]
    dataset_metrics = summarize_dataset_metrics(subject_dict_rows)
    etard_condition_metrics = summarize_etard_condition_metrics(recording_dict_rows)
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

    reference_lookup = load_subject_specific_reference(config)
    comparison_rows: list[dict[str, object]] = []
    for row in subject_dict_rows:
        reference_row = reference_lookup[(row["dataset"], row["subject_id"])]
        loso_metric = float(row["metric_value"])
        subject_specific_metric = float(reference_row["metric_value"])
        comparison_rows.append(
            {
                "dataset": row["dataset"],
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
    comparison_rows.sort(key=lambda item: (item["dataset"], item["subject_id"]))
    write_loso_vs_subject_specific_comparison_csv(output_dir / "loso_vs_subject_specific_comparison.csv", comparison_rows)
    write_loso_vs_subject_specific_comparison_md(output_dir / "loso_vs_subject_specific_comparison.md", comparison_rows)

    write_leakage_summary(output_dir / "leakage_summary.md", config=config, leakage_entries=leakage_entries)
    schema_validation = validate_schema(
        config=config,
        dataset_subjects=dataset_subjects,
        recording_rows=all_recording_rows,
        subject_rows=all_subject_rows,
        dataset_metrics=dataset_metrics,
        etard_condition_metrics=etard_condition_metrics,
        comparison_rows=comparison_rows,
        leakage_entries=leakage_entries,
    )
    with open(output_dir / "schema_validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(schema_validation, handle, indent=2)
    write_result_summary(
        output_dir / "result_summary.md",
        config=config,
        dataset_subjects=dataset_subjects,
        dataset_metrics=dataset_metrics,
        etard_condition_metrics=etard_condition_metrics,
        comparison_rows=comparison_rows,
    )

    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": config["protocol"],
                "artifact_scope": config["artifact_scope"],
                "previous_fix_branch": config["previous_fix_branch"],
                "previous_fix_commit": config["previous_fix_commit"],
                "current_branch": config["current_branch"],
                "seed": int(config["seed"]),
                "model": config["model"],
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
                    "planned_jobs": sum(len(subjects) for subjects in dataset_subjects.values()),
                    "successful_jobs": len(model_runs),
                    "failed_jobs": 0,
                },
                "model_runs": model_runs,
                "artifacts": {
                    "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
                    "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
                    "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
                    "etard_condition_metrics": repo_relative(output_dir / "etard_condition_metrics.csv"),
                    "leakage_summary": repo_relative(output_dir / "leakage_summary.md"),
                    "loso_vs_subject_specific_comparison_csv": repo_relative(output_dir / "loso_vs_subject_specific_comparison.csv"),
                    "loso_vs_subject_specific_comparison_md": repo_relative(output_dir / "loso_vs_subject_specific_comparison.md"),
                    "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
                    "result_summary": repo_relative(output_dir / "result_summary.md"),
                },
            },
            handle,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
