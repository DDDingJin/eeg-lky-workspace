from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repro.reference_baselines import list_reference_subjects

from run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1 import (
    DIAGNOSTIC_MATRIX_FIELDS,
    build_training_curve_rows,
    ensure_dir,
    load_config,
    print_event,
    repo_relative,
    resolve_dataset_path,
    run_happyquokka_full_eval,
    run_linear_family_full_eval,
    run_shape_check,
    run_vlaai_full_eval,
    train_happyquokka_subject_specific,
    write_csv_rows,
    write_full_eval_diagnostic,
    write_json,
    write_shape_audit,
    write_training_summary,
)
from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS


COMPLETED_JOB_FIELDS = [
    "dataset",
    "subject_id",
    "model",
    "seed",
    "job_key",
    "status",
    "checkpoint_id",
    "subject_metric",
]

DATASET_METRIC_FIELDS = [
    "dataset",
    "model",
    "seed",
    "protocol",
    "n_subjects",
    "metric_mean",
    "metric_std",
    "metric_median",
    "metric_min",
    "metric_max",
    "artifact_scope",
]

AUDIT_FIELDS = [
    "model",
    "dataset",
    "subject_list",
    "seed",
    "split_source",
    "train_split_only_for_training",
    "val_split_only_for_selection",
    "test_split_only_for_final_metric",
    "target_alignment",
    "scorer",
    "aggregation",
    "full_test_eval_not_capped",
    "max_epochs",
    "early_stopping_patience",
    "batch_size",
    "eval_batch_size",
    "checkpoint_selection_rule",
    "model_family_contract",
    "known_caveats",
    "comparable",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--shape-check-only", action="store_true")
    parser.add_argument("--subjects", default="", help="Comma-separated subject filter, for example P00,P01.")
    parser.add_argument("--models", default="", help="Comma-separated model filter.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Cap the filtered planned job list.")
    parser.add_argument("--recompute-artifacts-only", action="store_true", help="Recompute compact aggregate artifacts from existing outputs; no training.")
    parser.add_argument("--happyquokka-seed-smoke", action="store_true", help="Run two one-batch HappyQuokka seed-control smoke trainings for P00; no artifact writes.")
    return parser.parse_args()


def load_existing_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_existing_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def job_key(dataset: str, subject_id: str, model: str, seed: int) -> str:
    return f"{dataset}:{subject_id}:{model}:seed{seed}"


def write_log(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_dataset_metrics(config: dict, subject_metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not subject_metric_rows:
        return []
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in subject_metric_rows:
        key = (str(row["dataset"]), str(row["model"]), int(row["seed"]))
        grouped.setdefault(key, []).append(float(row["metric_value"]))
    rows: list[dict[str, object]] = []
    for (dataset, model, seed), values in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "protocol": config["protocol"],
                "n_subjects": len(values),
                "metric_mean": float(statistics.fmean(values)),
                "metric_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "metric_median": float(statistics.median(values)),
                "metric_min": float(min(values)),
                "metric_max": float(max(values)),
                "artifact_scope": config["artifact_scope"],
            }
        )
    return rows


def parse_filter(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def filter_ordered(values: list[str], allowed: set[str] | None, label: str) -> list[str]:
    if allowed is None:
        return values
    unknown = sorted(allowed - set(values))
    if unknown:
        raise ValueError(f"unknown {label}: {','.join(unknown)}")
    return [value for value in values if value in allowed]


def dataset_metric_key(row: dict[str, object]) -> tuple[str, str, int]:
    return (str(row["dataset"]), str(row["model"]), int(row["seed"]))


def expected_dataset_metric_counts(subject_metric_rows: list[dict[str, object]]) -> dict[tuple[str, str, int], int]:
    counts: dict[tuple[str, str, int], set[str]] = {}
    for row in subject_metric_rows:
        counts.setdefault(dataset_metric_key(row), set()).add(str(row["subject_id"]))
    return {key: len(subjects) for key, subjects in counts.items()}


def normalize_model_run_entries(entries) -> list[dict[str, object]]:
    if isinstance(entries, list):
        return entries
    if isinstance(entries, dict):
        return [entries]
    raise TypeError(f"unexpected model_run_entries type: {type(entries).__name__}")


def update_happyquokka_seed_metadata(model_run_entries: list[dict[str, object]], seed: int, mark_original_missing: bool) -> None:
    if not mark_original_missing:
        for entry in model_run_entries:
            if str(entry.get("model")) != "happyquokka" or not bool(entry.get("seed_control")):
                continue
            entry["seed_control_status"] = "applied_in_this_run"
            entry.pop("seed_control_patch_available", None)
        return
    for entry in model_run_entries:
        if str(entry.get("model")) != "happyquokka":
            continue
        entry["seed_control"] = False
        entry["seed_control_status"] = "not_present_in_original_run"
        entry["seed_control_patch_available"] = True
        entry["seed_control_record"] = {
            "enabled": False,
            "seed": int(seed),
            "status": "not_present_in_original_run",
            "patch_available": True,
            "artifact_note": "Existing metrics were not rerun; seed-control code is available for future runs only.",
        }


def write_model_identity_audit(config: dict, output_dir: Path) -> None:
    lines = [
        "# Model Identity Audit",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- seed: `{config['seed']}`",
        "",
        "## VLAAI",
        "",
        "- current implementation: `src/repro/mldecoders/models.py::VLAAIExactOfficialRegressor`",
        "- current contract: within-subject local adaptation, `64 x 50` EEG window input, last-target scalar regression output, validation-selected checkpoint, full subject test aggregation.",
        "- reference path: repository reference VLAAI paths such as `src/repro/vlaai_exact.py` and `scripts/run_vlaai_exact_reference.py` represent reference/exact structural runs with their own dataset construction and benchmark protocol.",
        "- identity conclusion: this run is a local adaptation / not yet reference-protocol parity result. It should not be described as an official equivalent VLAAI reproduction.",
        "",
        "## HappyQuokka",
        "",
        "- current implementation: `src/repro/happyquokka_reference.py` helpers plus upstream `external/upstream/HappyQuokka_system_for_EEG_Challenge` decoder.",
        "- current contract: within-subject local adaptation, non-overlapping `10s_chunk` windows, `g_con=false`, one local subject id, validation-selected checkpoint, and full available test chunks per recording.",
        "- reference path: repository reference HappyQuokka paths such as `src/repro/happyquokka_reference.py`, `scripts/run_happyquokka_reference.py`, and archived `happyquokka_gcon` summaries use reference-style dataset packaging and commonly evaluate `g_con=true` subject-conditioned variants.",
        "- seed control after this fix: HappyQuokka training receives the config seed explicitly and sets Python, NumPy, Torch, CUDA-if-available, and DataLoader shuffle generator seeds.",
        "- identity conclusion: this run is a local adaptation / not yet reference-protocol parity result. It preserves the current 10s_chunk contract and must not be claimed as an official equivalent HappyQuokka reproduction.",
        "",
    ]
    (output_dir / "model_identity_audit.md").write_text("\n".join(lines), encoding="utf-8")


def rebuild_compact_artifacts(
    *,
    config: dict,
    output_dir: Path,
    dataset_id: str,
    subjects: list[str],
    planned_jobs: list[dict[str, object]],
    comparable_states: dict[str, str],
    subject_metric_rows: list[dict[str, object]],
    recording_rows: list[dict[str, object]],
    matrix_rows: list[dict[str, object]],
    model_run_entries: list[dict[str, object]],
    failure_entries: list[dict[str, object]],
    training_curve_rows: list[dict[str, object]],
) -> dict[str, object]:
    update_happyquokka_seed_metadata(
        model_run_entries,
        int(config["seed"]),
        mark_original_missing="seeded" not in str(config["protocol"]),
    )
    dataset_metric_rows = summarize_dataset_metrics(config, subject_metric_rows)
    subject_row_objs = []
    for row in subject_metric_rows:
        obj = type("SubjectRow", (), {})()
        for key, value in row.items():
            setattr(obj, key, value)
        subject_row_objs.append(obj)
    schema = build_schema_validation(
        matrix_rows=matrix_rows,
        subject_rows=subject_row_objs,
        recording_rows=recording_rows,
        dataset_metric_rows=dataset_metric_rows,
        failure_entries=failure_entries,
        model_run_entries=model_run_entries,
        training_curve_rows=training_curve_rows,
    )
    write_csv_rows(output_dir / "dataset_metrics.csv", dataset_metric_rows, DATASET_METRIC_FIELDS)
    write_json(output_dir / "model_run_entries.json", model_run_entries)
    write_json(output_dir / "schema_validation_report.json", schema)
    write_model_identity_audit(config, output_dir)
    run_manifest = {
        "protocol": config["protocol"],
        "dataset": dataset_id,
        "seed": int(config["seed"]),
        "subjects": subjects,
        "models": list(config["models"]),
        "planned_jobs": planned_jobs,
        "protocol_audit_summary": comparable_states,
        "notes": [
            "linear/lasso/elasticnet are budgeted linear-family baselines with max_fit_samples_per_split=12000",
            "happyquokka remains a distinct 10s_chunk contract",
            "P00 is intentionally rerun for directory consistency",
            "dataset_metrics.csv is grouped by dataset/model/seed and is not a cross-model aggregate",
            "model_identity_audit.md documents local adaptation / not yet reference-protocol parity status",
        ],
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    return schema


def print_job_plan(planned_jobs: list[dict[str, object]], completed_lookup: set[str], execution_jobs: list[dict[str, object]]) -> None:
    completed_jobs = [job for job in planned_jobs if str(job["job_key"]) in completed_lookup]
    pending_jobs = [job for job in planned_jobs if str(job["job_key"]) not in completed_lookup]
    print_event("DRY RUN", f"planned_jobs={len(planned_jobs)} completed_jobs={len(completed_jobs)} pending_jobs={len(pending_jobs)} execution_jobs={len(execution_jobs)}")
    for prefix, jobs in (("planned", planned_jobs), ("completed", completed_jobs), ("pending", pending_jobs), ("execution", execution_jobs)):
        keys = ",".join(str(job["job_key"]) for job in jobs) if jobs else "none"
        print_event("DRY RUN", f"{prefix}={keys}")


def run_happyquokka_seed_smoke(config: dict, dataset_dir: Path, subject_id: str, device: str) -> int:
    hq_cfg = config["happyquokka"]
    input_length = int(hq_cfg["win_len_seconds"]) * int(hq_cfg["sample_rate"])
    records = []
    for _ in range(2):
        _, summary = train_happyquokka_subject_specific(
            input_dir=dataset_dir,
            participant=subject_id,
            device=device,
            input_length=input_length,
            batch_size=int(hq_cfg["batch_size"]),
            epochs=1,
            learning_rate=float(hq_cfg["learning_rate"]),
            dropout=float(hq_cfg["dropout"]),
            lamda=float(hq_cfg["lamda"]),
            g_con=bool(hq_cfg.get("g_con", False)),
            early_stopping_patience=int(hq_cfg["early_stopping_patience"]),
            seed=int(config["seed"]),
            max_train_batches=1,
        )
        record = summary["seed_control"]
        records.append(
            {
                "initial_state_hash": record["initial_state_hash"],
                "first_train_batch_hash": record["first_train_batch_hash"],
                "best_state_hash": record["best_state_hash"],
                "fixed_prediction_hash": record["fixed_prediction_hash"],
            }
        )
    passed = records[0] == records[1]
    print_event("SEED SMOKE", f"subject={subject_id} model=happyquokka seed={config['seed']} passed={passed}")
    print_event("SEED SMOKE", f"run1={json.dumps(records[0], sort_keys=True)}")
    print_event("SEED SMOKE", f"run2={json.dumps(records[1], sort_keys=True)}")
    if not passed:
        print_event("SEED SMOKE", "failed because at least one deterministic hash differed")
    return 0 if passed else 1


def write_result_summary_all_subjects(
    path: Path,
    config: dict,
    comparison_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Subject-Specific Local Model Full-Eval Weissbart Full v1",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- subject: `{config['dataset']['subject_id']}`",
        f"- seed: `{config['seed']}`",
        "",
        "| model | family | metric | note |",
        "| --- | --- | ---: | --- |",
    ]
    for row in sorted(comparison_rows, key=lambda item: (item["family"], item["model"], item["subject_id"])):
        note = "accepted reference" if row["family"] == "accepted_reference" else "local full-eval"
        lines.append(f"| {row['model']} | {row['family']} | {row['metric_value']} | {note} |")
    lines.extend(
        [
            "",
            "## Scope Notes",
            "- Test evaluation is no longer capped to 256 windows/points.",
            "- Train/validation fitting may still be subsampled for local diagnostic runtime, but test evaluation covers the full available Weissbart test recording contract for each subject and model family.",
            "- HappyQuokka remains a `10s_chunk` family and is not forced into the 50-sample window contract.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_coverage_summary(recording_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in recording_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    for model, rows in sorted(by_model.items()):
        coverages = [float(item["coverage_ratio"]) for item in rows]
        valid_samples = [int(item["num_valid_samples"]) for item in rows]
        summary.append(
            {
                "model": model,
                "recording_count": len(rows),
                "coverage_ratio_min": min(coverages),
                "coverage_ratio_max": max(coverages),
                "num_valid_samples_min": min(valid_samples),
                "num_valid_samples_max": max(valid_samples),
                "model_family_contract": rows[0]["model_family_contract"],
            }
        )
    return summary


def audit_row_for_model(config: dict, model_name: str, subjects: list[str]) -> dict[str, object]:
    dataset = config["dataset"]["dataset_id"]
    subject_list = ",".join(subjects)
    scorer = "benchmark.scoring.pearson_on_valid"
    aggregation = "recording Pearson -> subject mean_recording_pearson_r -> dataset mean_subject_metric"
    split_source = f"{config['dataset']['dataset_locator']} with per-subject train/val/test reference split files"
    common = {
        "model": model_name,
        "dataset": dataset,
        "subject_list": subject_list,
        "seed": int(config["seed"]),
        "split_source": split_source,
        "train_split_only_for_training": True,
        "val_split_only_for_selection": True,
        "test_split_only_for_final_metric": True,
        "target_alignment": "recording-level Pearson on aligned valid samples; target contract depends on model family",
        "scorer": scorer,
        "aggregation": aggregation,
        "full_test_eval_not_capped": True,
        "batch_size": "",
        "eval_batch_size": "",
        "checkpoint_selection_rule": "",
        "model_family_contract": "",
        "known_caveats": "",
        "comparable": "",
        "reason": "",
        "max_epochs": "",
        "early_stopping_patience": "",
    }

    if model_name == "linear":
        common.update(
            {
                "checkpoint_selection_rule": "no checkpoint; OLS fit on train, validation score recorded only",
                "model_family_contract": "lag_matrix_trf",
                "known_caveats": "budgeted linear-family baseline with max_fit_samples_per_split=12000 on train/val",
                "comparable": "partial",
                "reason": "same split/scorer/aggregation/full-test eval, but train/val fit uses sample cap unlike accepted ridge full-fit baseline",
            }
        )
    elif model_name == "lasso":
        common.update(
            {
                "checkpoint_selection_rule": "choose alpha by validation Pearson",
                "model_family_contract": "lag_matrix_trf",
                "known_caveats": "budgeted linear-family baseline with max_fit_samples_per_split=12000 on train/val",
                "comparable": "partial",
                "reason": "same split/scorer/aggregation/full-test eval, but train/val fit uses sample cap and lasso regularization differs from accepted ridge baseline",
            }
        )
    elif model_name == "elasticnet":
        common.update(
            {
                "checkpoint_selection_rule": "choose alpha and l1_ratio by validation Pearson",
                "model_family_contract": "lag_matrix_trf",
                "known_caveats": "budgeted linear-family baseline with max_fit_samples_per_split=12000 on train/val",
                "comparable": "partial",
                "reason": "same split/scorer/aggregation/full-test eval, but train/val fit uses sample cap and elasticnet regularization differs from accepted ridge baseline",
            }
        )
    elif model_name == "vlaai":
        cfg = config["vlaai"]
        common.update(
            {
                "max_epochs": int(cfg["max_epochs"]),
                "early_stopping_patience": int(cfg["early_stopping_patience"]),
                "batch_size": int(cfg["batch_size"]),
                "eval_batch_size": "1024 via validation loader inside train_dnn_reference_logged",
                "checkpoint_selection_rule": "best validation Pearson",
                "model_family_contract": "vlaai_local_adapter",
                "known_caveats": "distinct VLAAI window contract with last-target regression output",
                "comparable": "true",
                "reason": "same split/scorer/aggregation/full-test eval as accepted deep subject-specific models; only model family contract differs",
            }
        )
    elif model_name == "happyquokka":
        cfg = config["happyquokka"]
        common.update(
            {
                "max_epochs": int(cfg["max_epochs"]),
                "early_stopping_patience": int(cfg["early_stopping_patience"]),
                "batch_size": int(cfg["batch_size"]),
                "eval_batch_size": "all available non-overlapping 10s chunks per validation recording",
                "checkpoint_selection_rule": "best validation Pearson",
                "model_family_contract": "10s_chunk",
                "known_caveats": "distinct 10s_chunk contract; per-recording coverage can be <1 due to tail shorter than 10s chunk",
                "comparable": "partial",
                "reason": "same split/scorer/aggregation/final-test usage, but model family contract and coverage differ from accepted 50-sample window families",
            }
        )
    else:
        raise ValueError(model_name)
    return common


def write_protocol_audit(config: dict, output_dir: Path, subjects: list[str]) -> list[dict[str, object]]:
    rows = [audit_row_for_model(config, model_name, subjects) for model_name in config["models"]]
    write_csv_rows(output_dir / "accepted_protocol_audit.csv", rows, AUDIT_FIELDS)
    lines = [
        "# Accepted Protocol Audit",
        "",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- subject count: `{len(subjects)}`",
        f"- seed: `{config['seed']}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## `{row['model']}`",
                "",
                f"- comparable: `{row['comparable']}`",
                f"- reason: {row['reason']}",
                f"- model_family_contract: `{row['model_family_contract']}`",
                f"- checkpoint_selection_rule: `{row['checkpoint_selection_rule']}`",
                f"- known_caveats: {row['known_caveats']}",
                "",
            ]
        )
    (output_dir / "accepted_protocol_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def append_log(lines: list[str], message: str) -> None:
    lines.append(message)
    print_event("LOG", message)


def summary_config(config: dict, dataset_id: str, subject_label: str) -> dict[str, object]:
    return {
        "protocol": config["protocol"],
        "dataset": {
            "dataset_id": dataset_id,
            "subject_id": subject_label,
        },
        "seed": int(config["seed"]),
    }


def build_schema_validation(
    *,
    matrix_rows: list[dict[str, object]],
    subject_rows: list[object],
    recording_rows: list[dict[str, object]],
    dataset_metric_rows: list[dict[str, object]],
    failure_entries: list[dict[str, object]],
    model_run_entries: list[dict[str, object]],
    training_curve_rows: list[dict[str, object]],
) -> dict[str, object]:
    success_pairs = sorted(
        (str(row["subject_id"]), str(row["model"]))
        for row in matrix_rows
        if str(row["status"]) == "success"
    )
    subject_pairs = sorted((str(row.subject_id), str(row.model)) for row in subject_rows)
    recording_pairs = sorted({(str(row["subject_id"]), str(row["model"])) for row in recording_rows})
    deep_success_pairs = sorted(
        (str(row["subject_id"]), str(row["model"]))
        for row in matrix_rows
        if str(row["status"]) == "success" and str(row["model"]) in {"vlaai", "happyquokka"}
    )
    training_curve_pairs = sorted({(str(row["subject_id"]), str(row["model"])) for row in training_curve_rows})
    training_metadata_pairs = sorted(
        (str(row["subject_id"]), str(row["model"]))
        for row in model_run_entries
        if str(row.get("status")) == "success"
        and row.get("best_epoch") is not None
        and row.get("epochs_completed") is not None
        and row.get("best_val_score") is not None
    )
    subject_metric_dict_rows = [row.__dict__ for row in subject_rows]
    expected_dataset_counts = expected_dataset_metric_counts(subject_metric_dict_rows)
    dataset_metric_counts = {
        dataset_metric_key(row): int(row["n_subjects"])
        for row in dataset_metric_rows
    }
    checks = {
        "subject_metrics_match_success_jobs": success_pairs == subject_pairs,
        "recording_metrics_cover_success_jobs": success_pairs == recording_pairs,
        "dataset_metrics_keys_match_subject_metrics": sorted(dataset_metric_counts) == sorted(expected_dataset_counts),
        "dataset_metrics_n_subjects_match_subject_metrics": dataset_metric_counts == expected_dataset_counts,
        "dataset_metrics_current_scope_subject_count_correct": dataset_metric_counts == expected_dataset_counts,
        "deep_models_have_training_curves": deep_success_pairs == training_curve_pairs,
        "deep_models_have_training_metadata": deep_success_pairs == training_metadata_pairs,
        "every_recording_has_num_valid_samples": all(int(row["num_valid_samples"]) >= 0 for row in recording_rows),
        "every_recording_has_coverage_ratio": all("coverage_ratio" in row and float(row["coverage_ratio"]) >= 0.0 for row in recording_rows),
        "every_success_model_full_eval_not_capped": all(bool(row["full_eval_not_capped"]) for row in matrix_rows if str(row["status"]) == "success"),
        "happyquokka_marked_10s_chunk": all(
            str(row["model"]) != "happyquokka" or str(row["model_family_contract"]) == "10s_chunk" for row in matrix_rows
        ),
        "failure_report_empty_or_reasoned": len(failure_entries) == 0 or all(str(item.get("failure_reason", "")).strip() != "" for item in failure_entries),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "success_job_count": len(success_pairs),
        "deep_success_job_count": len(deep_success_pairs),
        "subject_metric_rows": len(subject_rows),
        "dataset_metric_rows": len(dataset_metric_rows),
        "recording_metric_rows": len(recording_rows),
        "training_curve_rows": len(training_curve_rows),
        "failure_count": len(failure_entries),
    }


def write_training_summary_mixed(
    path: Path,
    *,
    config: dict,
    matrix_rows: list[dict[str, object]],
    model_run_entries: list[dict[str, object]],
    recording_rows: list[dict[str, object]],
) -> None:
    entry_lookup = {
        (str(item["subject_id"]), str(item["model"])): item
        for item in model_run_entries
    }
    lines = [
        "# Training Summary",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- subject: `{config['dataset']['subject_id']}`",
        f"- seed: `{config['seed']}`",
        "",
    ]
    for row in matrix_rows:
        if str(row["status"]) != "success":
            continue
        subject_id = str(row["subject_id"])
        model = str(row["model"])
        entry = entry_lookup[(subject_id, model)]
        model_recordings = [
            item
            for item in recording_rows
            if str(item["subject_id"]) == subject_id and str(item["model"]) == model
        ]
        valid_samples = [int(item["num_valid_samples"]) for item in model_recordings]
        coverages = [float(item["coverage_ratio"]) for item in model_recordings]
        epochs_completed = entry.get("epochs_completed", "n/a (non-iterative model)")
        best_epoch = entry.get("best_epoch", "n/a (non-iterative model)")
        best_val_score = entry.get("best_val_score", "n/a")
        if isinstance(best_val_score, float):
            best_val_text = f"{best_val_score:.6f}"
        else:
            best_val_text = str(best_val_score)
        lines.extend(
            [
                f"## `{subject_id}:{model}`",
                "",
                f"- epochs_completed: `{epochs_completed}`",
                f"- best_epoch: `{best_epoch}`",
                f"- best_val_score: `{best_val_text}`",
                f"- test_metric: `{row['subject_metric']}`",
                f"- num_valid_samples range: `{min(valid_samples)} .. {max(valid_samples)}`",
                f"- coverage_ratio range: `{min(coverages):.6f} .. {max(coverages):.6f}`",
                f"- full_eval_not_capped: `{row['full_eval_not_capped']}`",
                f"- model_family_contract: `{row['model_family_contract']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


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
    sampling_rate = int(dataset_cfg["sampling_rate"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")

    all_subjects = sorted(list_reference_subjects(dataset_dir, split="test"))
    subjects = filter_ordered(all_subjects, parse_filter(args.subjects), "subjects")
    models = filter_ordered(list(config["models"]), parse_filter(args.models), "models")
    planned_jobs = [
        {
            "dataset": dataset_id,
            "subject_id": subject_id,
            "model": model_name,
            "seed": int(config["seed"]),
            "job_key": job_key(dataset_id, subject_id, model_name, int(config["seed"])),
        }
        for subject_id in subjects
        for model_name in models
    ]
    if args.max_jobs is not None and args.max_jobs < 0:
        raise ValueError("--max-jobs must be non-negative")

    print_event("STARTUP", f"config={args.config}")
    print_event("STARTUP", f"output_dir={repo_relative(output_dir)} dataset={dataset_id} seed={config['seed']} device={device}")
    print_event("STARTUP", f"subjects={','.join(subjects)}")
    print_event("STARTUP", f"models={','.join(models)}")
    print_event("STARTUP", f"planned_jobs={len(planned_jobs)}")

    completed_path = output_dir / "completed_jobs.json"
    run_state_path = output_dir / "run_state.json"
    subject_metrics_path = output_dir / "subject_metrics.csv"
    recording_metrics_path = output_dir / "recording_metrics.csv"
    dataset_metrics_path = output_dir / "dataset_metrics.csv"
    matrix_path = output_dir / "full_eval_matrix.csv"
    model_entries_path = output_dir / "model_run_entries.json"
    failure_path = output_dir / "failure_report.json"
    training_curve_path = output_dir / "training_curve.csv"
    coverage_summary_path = output_dir / "coverage_summary.csv"
    schema_path = output_dir / "schema_validation_report.json"
    result_summary_path = output_dir / "result_summary.md"
    training_summary_path = output_dir / "training_summary.md"
    shape_audit_path = output_dir / "adapter_shape_audit.md"
    diagnostic_path = output_dir / "full_eval_diagnostic.md"
    log_path = output_dir / "logs" / "run.log"

    audit_config = dict(config)
    audit_config["models"] = models
    audit_rows = [audit_row_for_model(audit_config, model_name, subjects) for model_name in models]
    comparable_states = {row["model"]: row["comparable"] for row in audit_rows}
    if any(state == "false" for state in comparable_states.values()):
        print_event("AUDIT", "found comparable=false model; aborting before any training")
        return 1

    completed_jobs = load_existing_json(completed_path, {"completed_jobs": []})
    completed_lookup = {item["job_key"] for item in completed_jobs.get("completed_jobs", [])}
    pending_jobs = [item for item in planned_jobs if item["job_key"] not in completed_lookup]
    execution_jobs = pending_jobs[: args.max_jobs] if args.max_jobs is not None else pending_jobs

    if args.startup_only:
        print_event("STARTUP ONLY", "configuration and protocol audit validated; no training started")
        return 0

    if args.dry_run_plan:
        print_job_plan(planned_jobs, completed_lookup, execution_jobs)
        return 0

    if args.happyquokka_seed_smoke:
        smoke_subject = subjects[0] if subjects else "P00"
        return run_happyquokka_seed_smoke(config, dataset_dir, smoke_subject, device)

    if args.shape_check_only:
        shape_cfg = dict(config)
        shape_cfg["dataset"] = dict(config["dataset"])
        shape_cfg["dataset"]["subject_id"] = subjects[0]
        return run_shape_check(shape_cfg, dataset_dir, dataset_id, subjects[0], device)

    audit_rows = write_protocol_audit(audit_config, output_dir, subjects)
    comparable_states = {row["model"]: row["comparable"] for row in audit_rows}

    subject_metric_rows = load_existing_csv(subject_metrics_path)
    recording_rows = load_existing_csv(recording_metrics_path)
    dataset_metric_rows = load_existing_csv(dataset_metrics_path)
    matrix_rows = load_existing_csv(matrix_path)
    model_run_entries = normalize_model_run_entries(load_existing_json(model_entries_path, []))
    failure_entries = load_existing_json(failure_path, {"failures": []}).get("failures", [])
    training_curve_rows = load_existing_csv(training_curve_path)
    log_lines = []

    if args.recompute_artifacts_only:
        schema = rebuild_compact_artifacts(
            config=config,
            output_dir=output_dir,
            dataset_id=dataset_id,
            subjects=all_subjects,
            planned_jobs=[
                {
                    "dataset": dataset_id,
                    "subject_id": subject_id,
                    "model": model_name,
                    "seed": int(config["seed"]),
                    "job_key": job_key(dataset_id, subject_id, model_name, int(config["seed"])),
                }
                for subject_id in all_subjects
                for model_name in config["models"]
            ],
            comparable_states={row["model"]: row["comparable"] for row in write_protocol_audit(config, output_dir, all_subjects)},
            subject_metric_rows=subject_metric_rows,
            recording_rows=recording_rows,
            matrix_rows=matrix_rows,
            model_run_entries=model_run_entries,
            failure_entries=failure_entries,
            training_curve_rows=training_curve_rows,
        )
        print_event("RECOMPUTE", f"schema_passed={schema['passed']} dataset_metric_rows={schema['dataset_metric_rows']}")
        return 0 if schema["passed"] else 1

    if args.resume and not execution_jobs:
        append_log(log_lines, "no pending jobs")
        write_log(log_path, log_lines)
        print_event("NO PENDING JOBS", "all planned Weissbart jobs already completed")
        return 0

    for job in execution_jobs:
        subject_id = str(job["subject_id"])
        model_name = str(job["model"])
        append_log(log_lines, f"job_start {job['job_key']}")
        subject_cfg = dict(config)
        subject_cfg["dataset"] = dict(config["dataset"])
        subject_cfg["dataset"]["subject_id"] = subject_id
        try:
            if model_name in {"linear", "lasso", "elasticnet"}:
                result = run_linear_family_full_eval(
                    model_name=model_name,
                    config=subject_cfg,
                    dataset_dir=dataset_dir,
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    sampling_rate=sampling_rate,
                )
            elif model_name == "vlaai":
                result = run_vlaai_full_eval(subject_cfg, dataset_dir, dataset_id, subject_id, sampling_rate, device)
            elif model_name == "happyquokka":
                result = run_happyquokka_full_eval(subject_cfg, dataset_dir, dataset_id, subject_id, sampling_rate, device)
            else:
                raise ValueError(model_name)
        except Exception as exc:
            failure = {
                "dataset": dataset_id,
                "subject_id": subject_id,
                "model": model_name,
                "seed": int(config["seed"]),
                "status": "failed",
                "failure_reason": repr(exc),
                "notes": "runtime exception in Weissbart local-model full runner",
            }
            failure_entries.append(failure)
            append_log(log_lines, f"job_failed {job['job_key']} {repr(exc)}")
            write_json(failure_path, {"failures": failure_entries})
            write_log(log_path, log_lines)
            continue

        matrix_rows.append(result.matrix_row)
        recording_rows.extend(result.recording_rows)
        subject_metric_rows.extend([row.__dict__ for row in result.subject_rows])
        model_run_entries.append(result.model_run_entry)
        if result.failure_entry is not None:
            failure_entries.append(result.failure_entry)
            append_log(log_lines, f"job_failed {job['job_key']} {result.failure_entry['failure_reason']}")
            write_json(failure_path, {"failures": failure_entries})
            write_log(log_path, log_lines)
            continue

        completed_jobs.setdefault("completed_jobs", []).append(
            {
                "dataset": dataset_id,
                "subject_id": subject_id,
                "model": model_name,
                "seed": int(config["seed"]),
                "job_key": job["job_key"],
                "status": "success",
                "checkpoint_id": result.model_run_entry.get("checkpoint_id", ""),
                "subject_metric": result.matrix_row["subject_metric"],
            }
        )
        completed_lookup.add(job["job_key"])

        training_curve_rows = build_training_curve_rows(
            config=config,
            dataset_id=dataset_id,
            subject_id=subject_id,
            model_run_entries=[result.model_run_entry],
        ) + training_curve_rows
        dataset_metric_rows = summarize_dataset_metrics(config, subject_metric_rows)
        coverage_rows = build_coverage_summary(recording_rows)
        # Rebuild subject row dataclass-like objects for shared validators.
        subject_row_objs = []
        for row in subject_metric_rows:
            obj = type("SubjectRow", (), {})()
            for key, value in row.items():
                setattr(obj, key, value)
            subject_row_objs.append(obj)
        schema = build_schema_validation(
            matrix_rows=matrix_rows,
            subject_rows=subject_row_objs,
            recording_rows=recording_rows,
            dataset_metric_rows=dataset_metric_rows,
            failure_entries=failure_entries,
            model_run_entries=model_run_entries,
            training_curve_rows=training_curve_rows,
        )

        write_csv_rows(subject_metrics_path, subject_metric_rows, SUBJECT_METRIC_FIELDS)
        write_csv_rows(
            recording_metrics_path,
            recording_rows,
            RECORDING_METRIC_FIELDS + ["recording_length", "coverage_ratio", "dropped_samples_reason", "full_eval_not_capped", "model_family_contract"],
        )
        write_csv_rows(dataset_metrics_path, dataset_metric_rows, DATASET_METRIC_FIELDS)
        write_csv_rows(matrix_path, matrix_rows, DIAGNOSTIC_MATRIX_FIELDS)
        write_csv_rows(
            training_curve_path,
            training_curve_rows,
            ["dataset", "subject_id", "model", "seed", "protocol", "epoch", "train_loss", "val_score", "best_epoch", "epochs_completed", "is_best_epoch"],
        )
        write_csv_rows(
            coverage_summary_path,
            coverage_rows,
            ["model", "recording_count", "coverage_ratio_min", "coverage_ratio_max", "num_valid_samples_min", "num_valid_samples_max", "model_family_contract"],
        )
        write_json(model_entries_path, model_run_entries)
        write_model_identity_audit(config, output_dir)
        write_json(completed_path, completed_jobs)
        write_json(
            run_state_path,
            {
                "protocol": config["protocol"],
                "dataset": dataset_id,
                "seed": int(config["seed"]),
                "completed_job_count": len(completed_jobs["completed_jobs"]),
                "pending_job_count": len(planned_jobs) - len(completed_jobs["completed_jobs"]),
                "last_completed_job_key": job["job_key"],
            },
        )
        write_json(failure_path, {"failures": failure_entries})
        write_json(schema_path, schema)
        aggregate_summary_cfg = summary_config(config, dataset_id, "all_subjects")
        write_training_summary_mixed(
            training_summary_path,
            config=aggregate_summary_cfg,
            matrix_rows=matrix_rows,
            model_run_entries=model_run_entries,
            recording_rows=recording_rows,
        )
        shape_sections = []
        for row in matrix_rows:
            shape_sections.append(f"## `{row['model']}`\n\n- status: `{row['status']}`\n- model_family_contract: `{row['model_family_contract']}`\n")
        write_shape_audit(shape_audit_path, {
            "protocol": config["protocol"],
            "dataset": {"dataset_id": dataset_id, "subject_id": "all_subjects"},
            "seed": int(config["seed"]),
        }, matrix_rows, shape_sections)
        write_full_eval_diagnostic(diagnostic_path, matrix_rows, recording_rows)
        comparison_rows = [
            {
                "subject_id": row["subject_id"],
                "model": row["model"],
                "family": "local_full_eval",
                "metric_value": row["subject_metric"],
                "reference_source": repo_relative(subject_metrics_path),
                "checkpoint_id": entry.get("checkpoint_id", ""),
            }
            for row, entry in zip(matrix_rows, model_run_entries)
            if row["status"] == "success"
        ]
        write_result_summary_all_subjects(
            result_summary_path,
            aggregate_summary_cfg,
            comparison_rows,
        )
        run_manifest = {
            "protocol": config["protocol"],
            "dataset": dataset_id,
            "seed": int(config["seed"]),
            "subjects": subjects,
            "models": list(config["models"]),
            "planned_jobs": planned_jobs,
            "protocol_audit_summary": comparable_states,
            "notes": [
                "linear/lasso/elasticnet are budgeted linear-family baselines with max_fit_samples_per_split=12000",
                "happyquokka remains a distinct 10s_chunk contract",
                "P00 is intentionally rerun for directory consistency",
                "dataset_metrics.csv is grouped by dataset/model/seed and is not a cross-model aggregate",
                "model_identity_audit.md documents local adaptation / not yet reference-protocol parity status",
            ],
        }
        write_json(output_dir / "run_manifest.json", run_manifest)
        append_log(log_lines, f"job_completed {job['job_key']} metric={result.matrix_row['subject_metric']}")
        write_log(log_path, log_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
