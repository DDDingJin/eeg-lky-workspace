from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from repro.reference_baselines import list_reference_subjects
from run_gate0_gate2_full_subject_single_seed import run_job as run_reference_job
from run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1 import (
    run_happyquokka_full_eval,
    run_linear_family_full_eval,
    run_vlaai_full_eval,
)
from run_gate0_gate2_subject_specific_local_model_smoke_v1 import (
    ensure_dir,
    load_config,
    print_event,
    repo_relative,
    resolve_dataset_path,
    write_json,
)


MODELS = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "cca",
    "fcnn",
    "dnn",
    "cnn",
    "eegnet",
    "adt",
    "vlaai",
    "happyquokka",
]

PROTOCOL_AUDIT_FIELDS = [
    "dataset",
    "model",
    "subject_count",
    "seed",
    "input_contract",
    "target_alignment",
    "normalization",
    "training_budget",
    "checkpoint_selection",
    "scorer",
    "recording_level_aggregation",
    "full_test_evaluation",
    "uses_true_envelope_context_as_input",
    "train_split_only_for_training",
    "val_split_only_for_selection",
    "test_split_only_for_final_metric",
    "identity_status",
    "coverage",
]

JOB_PLAN_FIELDS = ["job_index", "dataset", "subject_id", "model", "seed", "job_key", "status", "checkpoint_path_planned"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--shape-check-only", action="store_true")
    parser.add_argument("--job-plan-only", action="store_true")
    parser.add_argument("--subjects", default="", help="Comma-separated subject filter.")
    parser.add_argument("--models", default="", help="Comma-separated model filter.")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def to_dict_rows(rows: list[Any]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if isinstance(row, dict):
            output.append(row)
        else:
            output.append(asdict(row))
    return output


def checkpoint_path(config: dict[str, Any], subject_id: str, model: str) -> str:
    key = job_key(config, subject_id, model).replace(":", "__")
    return f"{config['checkpoint_dir']}/{key}.pt"


def job_key(config: dict[str, Any], subject_id: str, model: str) -> str:
    return f"{config['dataset']['dataset_id']}:{subject_id}:{model}:seed{int(config['seed'])}"


def build_jobs(config: dict[str, Any], subjects: list[str], models: list[str]) -> list[dict[str, Any]]:
    jobs = []
    for subject_id in subjects:
        for model in models:
            jobs.append(
                {
                    "job_index": len(jobs),
                    "dataset": config["dataset"]["dataset_id"],
                    "subject_id": subject_id,
                    "model": model,
                    "seed": int(config["seed"]),
                    "job_key": job_key(config, subject_id, model),
                    "status": "pending",
                    "checkpoint_path_planned": checkpoint_path(config, subject_id, model),
                }
            )
    return jobs


def job_dir(output_dir: Path, subject_id: str, model: str, seed: int) -> Path:
    return output_dir / "jobs" / subject_id / model / f"seed_{seed}"


def job_manifest_path(output_dir: Path, job: dict[str, Any]) -> Path:
    return job_dir(output_dir, str(job["subject_id"]), str(job["model"]), int(job["seed"])) / "job_manifest.json"


def completed_job_keys_from_manifests(output_dir: Path, jobs: list[dict[str, Any]]) -> set[str]:
    completed = set()
    for job in jobs:
        manifest = job_manifest_path(output_dir, job)
        if not manifest.exists():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "success" and payload.get("job_key") == job["job_key"]:
            completed.add(str(job["job_key"]))
    return completed


def contract_category(model: str) -> str:
    if model in {"linear", "ridge", "lasso", "elasticnet", "cca"}:
        return "lagged_linear_family"
    if model in {"fcnn", "dnn", "cnn", "eegnet", "vlaai"}:
        return "50_sample_window"
    if model == "adt":
        return "320_sample_window_hop64"
    if model == "happyquokka":
        return "10s_chunk"
    raise ValueError(model)


def identity_status(model: str) -> str:
    if model in {"linear", "lasso", "elasticnet"}:
        return "budgeted baseline / local adaptation"
    if model in {"vlaai", "happyquokka", "dnn", "cnn"}:
        return "local adaptation / not yet reference-protocol parity"
    return "reference-compatible local subject-specific run"


def build_protocol_audit(config: dict[str, Any], subjects: list[str]) -> list[dict[str, Any]]:
    rows = []
    scorer = "benchmark.scoring.pearson_on_valid"
    aggregation = "full recording prediction aggregation -> recording Pearson -> subject mean_recording_pearson_r"
    for model in config["models"]:
        cfg = config[model]
        rows.append(
            {
                "dataset": config["dataset"]["dataset_id"],
                "model": model,
                "subject_count": len(subjects),
                "seed": int(config["seed"]),
                "input_contract": cfg["input_contract"],
                "target_alignment": "model-native EEG-only target alignment; no true envelope context input",
                "normalization": cfg["normalization"],
                "training_budget": cfg["training_budget"],
                "checkpoint_selection": cfg["selection_rule"],
                "scorer": scorer,
                "recording_level_aggregation": aggregation,
                "full_test_evaluation": "full available test recordings; no 256-window cap",
                "uses_true_envelope_context_as_input": False,
                "train_split_only_for_training": True,
                "val_split_only_for_selection": True,
                "test_split_only_for_final_metric": True,
                "identity_status": identity_status(model),
                "coverage": "model-native coverage recorded per recording; HappyQuokka may drop short tail due to 10s chunks",
            }
        )
    return rows


def write_protocol_audit(output_dir: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "protocol_audit.csv", rows, PROTOCOL_AUDIT_FIELDS)
    lines = [
        "# Etard Subject-Specific Modelset Protocol Audit",
        "",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- seed: `{config['seed']}`",
        "- scope: pure EEG-only subject-specific models",
        "- excluded: complete DECAF TwoBranch and LSTM",
        "- scorer: `benchmark.scoring.pearson_on_valid`",
        "- aggregation: full recording prediction aggregation -> recording Pearson -> subject mean",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## `{row['model']}`",
                "",
                f"- input_contract: {row['input_contract']}",
                f"- normalization: {row['normalization']}",
                f"- training_budget: {row['training_budget']}",
                f"- checkpoint_selection: {row['checkpoint_selection']}",
                f"- identity_status: `{row['identity_status']}`",
                f"- uses_true_envelope_context_as_input: `{row['uses_true_envelope_context_as_input']}`",
                "",
            ]
        )
    (output_dir / "protocol_audit.md").write_text("\n".join(lines), encoding="utf-8")


def build_shape_audit(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in config["models"]:
        category = contract_category(model)
        if category == "lagged_linear_family":
            eeg_input = "lagged EEG design matrix from 64 channels x lags 0..50"
            target = "trimmed envelope vector aligned after lag valid-range trim"
            output = "1D envelope prediction vector"
        elif category == "50_sample_window":
            eeg_input = "[batch, 64, 50] or adapter-equivalent 50-sample EEG window"
            target = "scalar last-sample envelope target"
            output = "scalar envelope prediction per window"
        elif category == "320_sample_window_hop64":
            eeg_input = "[batch, 64, 320] ADT window"
            target = "window-level envelope target under ADT exact adapter"
            output = "window-level envelope prediction"
        else:
            eeg_input = "[batch, 640, 64] 10s EEG chunk at 64 Hz"
            target = "[batch, 640, 1] 10s envelope chunk"
            output = "[batch, 640, 1] envelope prediction"
        rows.append(
            {
                "model": model,
                "contract_category": category,
                "eeg_input": eeg_input,
                "target": target,
                "output": output,
                "true_envelope_context_input": False,
                "shape_check_type": "contract preflight; no model training",
            }
        )
    return rows


def write_shape_audit(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "shape_audit.json", rows)
    lines = ["# Shape Audit", "", "- This is a contract preflight; no model training was started.", ""]
    for row in rows:
        lines.extend(
            [
                f"## `{row['model']}`",
                f"- category: `{row['contract_category']}`",
                f"- EEG input: {row['eeg_input']}",
                f"- target: {row['target']}",
                f"- output: {row['output']}",
                f"- true_envelope_context_input: `{row['true_envelope_context_input']}`",
                "",
            ]
        )
    (output_dir / "shape_audit.md").write_text("\n".join(lines), encoding="utf-8")


def build_manual_commands(config: dict[str, Any], subjects: list[str]) -> list[str]:
    script = "scripts\\run_gate0_gate2_subject_specific_modelset_etard_full_v1.py"
    cfg = "configs\\benchmark\\gate0_gate2_subject_specific_modelset_etard_full_v1.json"
    chunks = [
        ("P00-P04", subjects[0:5]),
        ("P05-P09", subjects[5:10]),
        ("P10-P14", subjects[10:15]),
        ("P15-P19", subjects[15:20]),
    ]
    commands = ["cd E:\\decode\\_fix_fixed_split_pooled20_modelset_v1"]
    for label, chunk in chunks:
        commands.append(
            "F:\\miniconda\\envs\\decode-torch\\python.exe "
            f"{script} --config {cfg} --device auto --resume --subjects {','.join(chunk)}"
            f"  # {label}"
        )
    return commands


def summarize_dataset_metrics(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in subject_rows:
        key = (str(row["dataset"]), str(row["model"]), int(row["seed"]))
        grouped.setdefault(key, []).append(float(row["metric_value"]))
    output = []
    for (dataset, model, seed), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "n_subjects": len(values),
                "mean_pearson": sum(values) / len(values),
                "std_pearson": statistics.stdev(values) if len(values) > 1 else 0.0,
                "median_pearson": statistics.median(values),
                "min_pearson": min(values),
                "max_pearson": max(values),
            }
        )
    return output


def build_post_run_schema(
    *,
    completed_jobs: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    recording_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    expected_job_count: int,
    expected_subject_count: int,
) -> dict[str, Any]:
    subject_keys = [
        (str(row["dataset"]), str(row["subject_id"]), str(row["model"]), str(row["seed"]))
        for row in subject_rows
    ]
    completed_keys = [
        (str(row["dataset"]), str(row["subject_id"]), str(row["model"]), str(row["seed"]))
        for row in completed_jobs
        if row.get("status") == "success"
    ]
    dataset_keys = [(str(row["dataset"]), str(row["model"]), str(row["seed"])) for row in dataset_rows]
    expected_dataset_keys = sorted({(dataset, model, seed) for dataset, _subject, model, seed in subject_keys})
    recording_job_keys = {
        (str(row["dataset"]), str(row["subject_id"]), str(row["model"]), str(row["seed"]))
        for row in recording_rows
    }
    dataset_subject_counts = {
        (str(row["dataset"]), str(row["model"]), str(row["seed"])): int(row["n_subjects"])
        for row in dataset_rows
    }
    subject_counts = {
        key: sum(1 for dataset, _subject, model, seed in subject_keys if (dataset, model, seed) == key)
        for key in expected_dataset_keys
    }
    checks = {
        "completed_job_count_expected": len(completed_jobs) == expected_job_count,
        "failure_report_empty": len(failures) == 0,
        "subject_metrics_one_row_per_completed_job": sorted(subject_keys) == sorted(completed_keys),
        "subject_metric_keys_unique": len(subject_keys) == len(set(subject_keys)),
        "dataset_metrics_one_row_per_dataset_model_seed": sorted(dataset_keys) == expected_dataset_keys,
        "dataset_metric_keys_unique": len(dataset_keys) == len(set(dataset_keys)),
        "dataset_metrics_subject_counts_match": dataset_subject_counts == subject_counts,
        "dataset_metrics_expected_subject_count": all(count == expected_subject_count for count in dataset_subject_counts.values()),
        "recording_metrics_cover_completed_jobs": set(completed_keys).issubset(recording_job_keys),
        "recording_metric_rows_present": len(recording_rows) > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "completed_job_count": len(completed_jobs),
        "pending_job_count": expected_job_count - len(completed_jobs),
        "failure_count": len(failures),
        "expected_job_count": expected_job_count,
        "subject_metric_rows": len(subject_rows),
        "dataset_metric_rows": len(dataset_rows),
        "recording_metric_rows": len(recording_rows),
    }


def load_support_configs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = load_config(ROOT / "configs" / "benchmark" / "gate0_gate2_model_expansion_v1.json")
    reference["protocol"] = config["protocol"]
    reference["artifact_scope"] = config["artifact_scope"]
    reference["seed"] = int(config["seed"])
    local = load_config(ROOT / "configs" / "benchmark" / "gate0_gate2_subject_specific_local_models_weissbart_full_v1.json")
    local["protocol"] = config["protocol"]
    local["artifact_scope"] = config["artifact_scope"]
    local["seed"] = int(config["seed"])
    local["dataset"] = config["dataset"]
    local["output_dir"] = config["output_dir"]
    return reference, local


def run_one_job(
    *,
    config: dict[str, Any],
    reference_config: dict[str, Any],
    local_config: dict[str, Any],
    dataset_dir: Path,
    dataset_cfg: dict[str, Any],
    subject_id: str,
    model: str,
    device: str,
    output_dir: Path,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if model in {"ridge", "cca", "fcnn", "dnn", "cnn", "eegnet", "adt"}:
        recording_rows, subject_rows, meta = run_reference_job(
            config=reference_config,
            dataset_cfg=dataset_cfg,
            dataset_dir=dataset_dir,
            subject_id=subject_id,
            model_name=model,
            device=device,
            output_dir=output_dir,
        )
        return recording_rows, subject_rows, meta
    if model in {"linear", "lasso", "elasticnet"}:
        result = run_linear_family_full_eval(
            model_name=model,
            config=local_config,
            dataset_dir=dataset_dir,
            dataset_id=config["dataset"]["dataset_id"],
            subject_id=subject_id,
            sampling_rate=int(config["dataset"]["sampling_rate"]),
        )
        if result.failure_entry is not None:
            raise RuntimeError(result.failure_entry["failure_reason"])
        return result.recording_rows, result.subject_rows, result.model_run_entry
    if model == "vlaai":
        result = run_vlaai_full_eval(local_config, dataset_dir, config["dataset"]["dataset_id"], subject_id, int(config["dataset"]["sampling_rate"]), device)
        if result.failure_entry is not None:
            raise RuntimeError(result.failure_entry["failure_reason"])
        return result.recording_rows, result.subject_rows, result.model_run_entry
    if model == "happyquokka":
        result = run_happyquokka_full_eval(local_config, dataset_dir, config["dataset"]["dataset_id"], subject_id, int(config["dataset"]["sampling_rate"]), device)
        if result.failure_entry is not None:
            raise RuntimeError(result.failure_entry["failure_reason"])
        return result.recording_rows, result.subject_rows, result.model_run_entry
    raise ValueError(f"unknown model {model}")


def run_execution(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dataset_dir: Path,
    subjects: list[str],
    models: list[str],
    device: str,
    resume: bool,
    force_rerun: bool,
    max_jobs: int | None,
) -> int:
    jobs = build_jobs(config, subjects, models)
    completed_lookup = completed_job_keys_from_manifests(output_dir, jobs)
    completed_path = output_dir / "completed_jobs.json"
    if completed_path.exists() and resume and not force_rerun:
        completed_lookup.update(str(item.get("job_key")) for item in json.loads(completed_path.read_text(encoding="utf-8")) if item.get("status") == "success")
    pending_jobs = [job for job in jobs if force_rerun or str(job["job_key"]) not in completed_lookup]
    if max_jobs is not None:
        pending_jobs = pending_jobs[: int(max_jobs)]

    reference_config, local_config = load_support_configs(config)
    dataset_cfg = {
        "dataset_id": config["dataset"]["dataset_id"],
        "dataset_locator": config["dataset"]["dataset_locator"],
        "sampling_rate": int(config["dataset"]["sampling_rate"]),
    }
    ensure_dir(output_dir / "jobs")
    failures = []
    for index, job in enumerate(pending_jobs, start=1):
        print_event("JOB START", f"{index}/{len(pending_jobs)} {job['job_key']}")
        jd = job_dir(output_dir, str(job["subject_id"]), str(job["model"]), int(job["seed"]))
        ensure_dir(jd)
        manifest = jd / "job_manifest.json"
        try:
            recording_rows, subject_rows, meta = run_one_job(
                config=config,
                reference_config=reference_config,
                local_config=local_config,
                dataset_dir=dataset_dir,
                dataset_cfg=dataset_cfg,
                subject_id=str(job["subject_id"]),
                model=str(job["model"]),
                device=device,
                output_dir=output_dir,
            )
            write_csv(jd / "recording_metrics.csv", to_dict_rows(recording_rows), RECORDING_METRIC_FIELDS)
            write_csv(jd / "subject_metrics.csv", to_dict_rows(subject_rows), SUBJECT_METRIC_FIELDS)
            metric = float(to_dict_rows(subject_rows)[0]["metric_value"]) if subject_rows else float("nan")
            payload = {
                **job,
                "status": "success",
                "subject_metric": metric,
                "meta": meta,
            }
            write_json(manifest, payload)
            print_event("JOB DONE", f"{job['job_key']} metric={metric:.6f}")
        except Exception as exc:
            payload = {**job, "status": "failed", "failure_reason": repr(exc)}
            failures.append(payload)
            write_json(manifest, payload)
            print_event("JOB FAILED", f"{job['job_key']} error={repr(exc)}")

    completed_jobs = []
    all_recording_rows: list[dict[str, Any]] = []
    all_subject_rows: list[dict[str, Any]] = []
    for job in build_jobs(config, list_reference_subjects(dataset_dir, split="test"), list(config["models"])):
        jd = job_dir(output_dir, str(job["subject_id"]), str(job["model"]), int(job["seed"]))
        manifest = jd / "job_manifest.json"
        if not manifest.exists():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if "checkpoint_path" in payload and "checkpoint_path_planned" not in payload:
            payload["checkpoint_path_planned"] = payload.pop("checkpoint_path")
        if payload.get("status") != "success":
            failures.append(payload)
            continue
        completed_jobs.append(payload)
        with (jd / "recording_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            all_recording_rows.extend(csv.DictReader(handle))
        with (jd / "subject_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            all_subject_rows.extend(csv.DictReader(handle))

    write_csv(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_csv(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
    dataset_rows = summarize_dataset_metrics(all_subject_rows)
    write_csv(output_dir / "dataset_metrics.csv", dataset_rows)
    write_json(output_dir / "completed_jobs.json", completed_jobs)
    write_json(output_dir / "failure_report.json", {"failures": failures})
    total_jobs = 20 * 12
    schema = build_post_run_schema(
        completed_jobs=completed_jobs,
        failures=failures,
        subject_rows=all_subject_rows,
        recording_rows=all_recording_rows,
        dataset_rows=dataset_rows,
        expected_job_count=total_jobs,
        expected_subject_count=20,
    )
    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "completed" if len(completed_jobs) == total_jobs and not failures else "partial",
            "completed_job_count": len(completed_jobs),
            "pending_job_count": total_jobs - len(completed_jobs),
            "last_completed_job_key": completed_jobs[-1]["job_key"] if completed_jobs else "",
        },
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "artifact_scope": config["artifact_scope"],
            "dataset": config["dataset"]["dataset_id"],
            "subjects": list_reference_subjects(dataset_dir, split="test"),
            "models": config["models"],
            "excluded_models": config["excluded_models"],
            "seed": int(config["seed"]),
            "planned_jobs": total_jobs,
            "completed_job_count": len(completed_jobs),
            "pending_job_count": total_jobs - len(completed_jobs),
            "failure_count": len(failures),
            "status": "completed" if len(completed_jobs) == total_jobs and not failures else "partial",
            "training_started": bool(completed_jobs),
            "checkpoint_policy": "local_checkpoints only; checkpoint files are not committed",
            "metric_policy": "existing metrics aggregated without rerun during post-run closure",
        },
    )
    return 0 if not failures else 1


def build_schema(
    *,
    config: dict[str, Any],
    subjects: list[str],
    jobs: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
    dataset_dir: Path,
) -> dict[str, Any]:
    checks = {
        "dataset_is_etard_tf64": config["dataset"]["dataset_id"] == "etard_tf64",
        "dataset_path_exists": dataset_dir.exists(),
        "subject_count_20": len(subjects) == 20,
        "models_exact_12": list(config["models"]) == MODELS,
        "excluded_decaf_twobranch": "decaf_twobranch" in config["excluded_models"],
        "excluded_lstm": "lstm" in config["excluded_models"],
        "planned_jobs_240": len(jobs) == 240,
        "seed_zero": int(config["seed"]) == 0,
        "protocol_audit_all_models": len(protocol_rows) == 12,
        "shape_audit_all_models": len(shape_rows) == 12,
        "no_true_envelope_context_inputs": all(not bool(row["uses_true_envelope_context_as_input"]) for row in protocol_rows),
        "checkpoint_path_local_only": all(str(job["checkpoint_path_planned"]).startswith("local_checkpoints/") for job in jobs),
        "no_training_started": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "planned_job_count": len(jobs),
        "dataset": config["dataset"]["dataset_id"],
        "subjects": subjects,
        "models": config["models"],
        "seed": int(config["seed"]),
        "manual_command_count": 4,
    }


def write_preflight_artifacts(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dataset_dir: Path,
    subjects: list[str],
    jobs: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
    schema: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    write_protocol_audit(output_dir, config, protocol_rows)
    write_shape_audit(output_dir, shape_rows)
    write_csv(output_dir / "job_plan.csv", jobs, JOB_PLAN_FIELDS)
    write_json(output_dir / "job_plan.json", jobs)
    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(output_dir / "failure_report.json", {"failures": [] if schema["passed"] else [{"failure_reason": "schema failed", "checks": schema["checks"]}]})
    write_json(
        output_dir / "run_state.json",
        {
            "status": "preflight_only",
            "dataset_dir": str(dataset_dir),
            "planned_job_count": len(jobs),
            "completed_job_count": 0,
            "pending_job_count": len(jobs),
            "training_started": False,
        },
    )
    write_json(output_dir / "completed_jobs.json", [])
    commands = build_manual_commands(config, subjects)
    (output_dir / "manual_commands.md").write_text(
        "# Manual Chunk Commands\n\n"
        "Run the smoke command first. If it passes, run the four chunk commands with `--resume`.\n\n"
        "## Smoke\n\n"
        "```powershell\n"
        "cd E:\\decode\\_fix_fixed_split_pooled20_modelset_v1\n"
        "F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_subject_specific_modelset_etard_full_v1.py --config configs\\benchmark\\gate0_gate2_subject_specific_modelset_etard_full_v1.json --device auto --resume --subjects P00 --models linear,ridge,cca --max-jobs 3\n"
        "```\n\n"
        "## Full chunks\n\n"
        "```powershell\n" + "\n".join(commands) + "\n```\n",
        encoding="utf-8",
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "artifact_scope": config["artifact_scope"],
            "dataset": config["dataset"]["dataset_id"],
            "subjects": subjects,
            "models": config["models"],
            "excluded_models": config["excluded_models"],
            "seed": int(config["seed"]),
            "planned_jobs": len(jobs),
            "no_training_started": True,
            "manual_commands": commands,
        },
    )


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    subjects = list_reference_subjects(dataset_dir, split="test")
    subject_filter = parse_filter(args.subjects)
    model_filter = parse_filter(args.models)
    filtered_subjects = filter_ordered(subjects, subject_filter, "subject")
    filtered_models = filter_ordered(list(config["models"]), model_filter, "model")
    jobs = build_jobs(config, filtered_subjects, filtered_models)
    if args.max_jobs is not None:
        jobs = jobs[: int(args.max_jobs)]
    protocol_rows = build_protocol_audit(config, subjects)
    shape_rows = build_shape_audit(config)
    schema = build_schema(
        config=config,
        subjects=subjects,
        jobs=build_jobs(config, subjects, list(config["models"])),
        protocol_rows=protocol_rows,
        shape_rows=shape_rows,
        dataset_dir=dataset_dir,
    )

    print_event("STARTUP", f"config={args.config}")
    print_event("STARTUP", f"dataset={config['dataset']['dataset_id']} subjects={len(subjects)} models={len(config['models'])} device={args.device}")
    print_event("STARTUP", f"output_dir={repo_relative(output_dir)}")
    if args.startup_only:
        return 0
    if args.dry_run_plan:
        all_jobs = build_jobs(config, subjects, list(config["models"]))
        first_job = jobs[0]["job_key"] if jobs else "none"
        last_job = jobs[-1]["job_key"] if jobs else "none"
        print_event("DRY RUN", f"planned_jobs={len(all_jobs)} filtered_execution_jobs={len(jobs)}")
        print_event("DRY RUN", f"first_execution_job={first_job}")
        print_event("DRY RUN", f"last_execution_job={last_job}")
        return 0
    if args.job_plan_only:
        for job in jobs:
            print_event("JOB PLAN", str(job["job_key"]))
    if args.shape_check_only:
        print_event("SHAPE", f"shape_contracts={len(shape_rows)} no_training_started=true")
    if args.job_plan_only or args.shape_check_only:
        write_preflight_artifacts(
            config=config,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            subjects=subjects,
            jobs=build_jobs(config, subjects, list(config["models"])),
            protocol_rows=protocol_rows,
            shape_rows=shape_rows,
            schema=schema,
        )
        return 0 if schema["passed"] else 1
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    return run_execution(
        config=config,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        subjects=filtered_subjects,
        models=filtered_models,
        device=device,
        resume=args.resume,
        force_rerun=args.force_rerun,
        max_jobs=args.max_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
