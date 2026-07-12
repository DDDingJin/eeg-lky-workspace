from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.scoring import WindowPrediction
from repro.mldecoders.cca import trim_valid_range
from repro.reference_baselines import concatenate_reference_split, load_reference_recordings

import run_gate0_gate2_negative_metric_audit_v1 as negative_audit
import run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1 as local_full_eval


HQ_AUDIT_FIELDS = [
    "dataset",
    "subject_id",
    "model",
    "seed",
    "recording_id",
    "metric_value",
    "num_valid_samples",
    "recording_length",
    "coverage_ratio",
    "dropped_samples_reason",
    "checkpoint_id",
]

REPEAT_FIELDS = [
    "run_index",
    "best_epoch",
    "epochs_completed",
    "best_val_score",
    "test_metric",
    "independent_pearson_metric",
    "first_test_input_hash",
    "prediction_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--hq-readonly-audit", action="store_true")
    parser.add_argument("--vlaai-repeat", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=None)
    parser.add_argument("--elasticnet-standardized", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_event(label: str, message: str) -> None:
    print(f"[{label}] {message}", flush=True)


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def source_config(job: dict[str, Any]) -> dict[str, Any]:
    return negative_audit.source_support_config(load_json(ROOT / job["source_config"]), str(job["model"]))


def dataset_dir_for(job: dict[str, Any]) -> Path:
    cfg = source_config(job)
    return negative_audit.resolve_dataset_path(cfg["dataset"]["dataset_locator"])


def job_key(job: dict[str, Any]) -> str:
    return f"{job['dataset']}:{job['subject_id']}:{job['model']}:seed{job['seed']}"


def build_preflight(config: dict[str, Any], output_dir: Path, device: str) -> dict[str, Any]:
    vlaai_job = config["vlaai_repeatability"]
    hq_job = config["happyquokka_p11_readonly"]
    enet_job = config["elasticnet_standardized"]
    hq_manifest = ROOT / hq_job["source_result_dir"] / "jobs" / hq_job["subject_id"] / hq_job["model"] / f"seed_{hq_job['seed']}" / "job_manifest.json"
    enet_cfg = source_config(enet_job)
    enet_missing = negative_audit.validate_runtime_config(enet_cfg, "elasticnet")
    formal_linear_normalization_claim = load_json(ROOT / enet_job["source_config"])["elasticnet"].get("normalization", "")
    runtime_standardization_applied = False
    checks = {
        "vlaai_repeat_job_declared": job_key(vlaai_job) == "weissbart_tf64:P06:vlaai:seed0",
        "hq_manifest_exists": hq_manifest.exists(),
        "elasticnet_runtime_config_hydrated": not enet_missing,
        "no_training_executed": True,
        "no_checkpoint_written": True,
        "no_prediction_dump_written": True,
        "independent_output_dir": config["output_dir"] not in {vlaai_job["source_result_dir"], hq_job["source_result_dir"], enet_job["source_result_dir"]},
    }
    schema = {
        "passed": all(checks.values()),
        "checks": checks,
        "device": device,
        "planned_manual_actions": [
            "vlaai_p06_repeatability_two_independent_repeats",
            "elasticnet_p11_train_only_standardized_diagnostic",
        ],
        "hq_readonly_audit_safe_to_run": hq_manifest.exists(),
        "elasticnet_preflight": {
            "formal_config_normalization_claim": formal_linear_normalization_claim,
            "published_runtime_standardization_applied": runtime_standardization_applied,
            "targeted_diagnostic_standardization": "StandardScaler fit on P11 train lag matrix only",
            "missing_runtime_fields": enet_missing,
        },
        "forbidden_outputs": config["forbidden_outputs"],
    }
    write_json(output_dir / "schema_validation_report.json", schema)
    failures = [] if schema["passed"] else [{"failure_reason": key} for key, ok in checks.items() if not ok]
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_json(
        output_dir / "run_state.json",
        {
            "status": "preflight_only",
            "training_started": False,
            "manual_jobs_completed": 0,
            "hq_readonly_audit_available": hq_manifest.exists(),
        },
    )
    lines = [
        "# Subject-Specific Targeted Audit Preflight",
        "",
        "Status: implementation/preflight only unless an explicit manual diagnostic flag is supplied.",
        "",
        "## Scope",
        f"- VLAAI repeatability job: `{job_key(vlaai_job)}`",
        f"- HappyQuokka read-only job: `{job_key(hq_job)}`",
        f"- ElasticNet standardized diagnostic job: `{job_key(enet_job)}`",
        "",
        "## ElasticNet Runtime Normalization",
        f"- formal_config_normalization_claim: `{formal_linear_normalization_claim}`",
        f"- published_runtime_standardization_applied: `{runtime_standardization_applied}`",
        "- targeted diagnostic will fit `StandardScaler` on P11 train lag matrix only, then transform train/val/test.",
        "",
        "## Policy",
        "- published subject metrics are read-only.",
        "- all new outputs stay in this independent output directory.",
        "- no checkpoint, raw data, prediction dump, or model weights are written by preflight/HQ audit.",
    ]
    (output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return schema


def hq_readonly_audit(config: dict[str, Any], output_dir: Path) -> None:
    job = config["happyquokka_p11_readonly"]
    source_dir = ROOT / job["source_result_dir"]
    manifest_path = source_dir / "jobs" / job["subject_id"] / job["model"] / f"seed_{job['seed']}" / "job_manifest.json"
    manifest = load_json(manifest_path)
    meta = manifest.get("meta", {})
    recording_rows = [
        row
        for row in read_csv(source_dir / "recording_metrics.csv")
        if row.get("dataset") == job["dataset"]
        and row.get("subject_id") == job["subject_id"]
        and row.get("model") == job["model"]
        and int(row.get("seed", 0)) == int(job["seed"])
    ]
    lengths = {
        recording_id: int(len(env))
        for recording_id, _eeg, env in load_reference_recordings(dataset_dir_for(job), "test", job["subject_id"], channels=range(64))
    }
    audit_rows: list[dict[str, Any]] = []
    for row in recording_rows:
        valid = int(row["num_valid_samples"])
        length = lengths.get(row["recording_id"], 0)
        audit_rows.append(
            {
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "model": row["model"],
                "seed": row["seed"],
                "recording_id": row["recording_id"],
                "metric_value": row["metric_value"],
                "num_valid_samples": valid,
                "recording_length": length,
                "coverage_ratio": float(valid / length) if length else "",
                "dropped_samples_reason": "tail shorter than 10s chunk",
                "checkpoint_id": row["checkpoint_id"],
            }
        )
    write_csv(output_dir / "hq_p11_audit.csv", audit_rows, HQ_AUDIT_FIELDS)
    history_train = meta.get("history_train_loss")
    history_val = meta.get("history_val_score")
    history_available = isinstance(history_train, list) and isinstance(history_val, list)
    if history_available:
        history_rows = [
            {"epoch": index + 1, "train_loss": train, "val_score": val}
            for index, (train, val) in enumerate(zip(history_train, history_val))
        ]
        write_csv(output_dir / "hq_p11_validation_history.csv", history_rows, ["epoch", "train_loss", "val_score"])
    lines = [
        "# HappyQuokka P11 Read-Only Audit",
        "",
        "No HappyQuokka training was run by this audit. This file only summarizes existing Etard P11 job artifacts and compact diagnostic artifacts.",
        "",
        f"- job_key: `{job_key(job)}`",
        f"- subject_metric: `{manifest.get('subject_metric')}`",
        f"- best_epoch: `{meta.get('best_epoch', '')}`",
        f"- epochs_completed: `{meta.get('epochs_completed', '')}`",
        f"- best_val_score: `{meta.get('best_val_score', '')}`",
        f"- validation_history: `{'available' if history_available else 'unavailable'}`",
        f"- recording_rows: `{len(audit_rows)}`",
        "",
        "This audit does not decide whether HappyQuokka should be retained; it only records available facts.",
    ]
    (output_dir / "hq_p11_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_event("HQ AUDIT", f"rows={len(audit_rows)} validation_history={'available' if history_available else 'unavailable'}")


def run_vlaai_repeat(config: dict[str, Any], output_dir: Path, device: str, repeat_count: int | None) -> None:
    job = dict(config["vlaai_repeatability"])
    repeats = int(repeat_count or job.get("repeat_count", 2))
    rows: list[dict[str, Any]] = []
    for run_index in range(1, repeats + 1):
        print_event("JOB START", f"{job_key(job)} repeat={run_index}")
        negative_audit.set_all_rng(int(job["seed"]))
        summary, _lag_rows = negative_audit.run_one_diagnostic(job, device)
        rows.append(
            {
                "run_index": run_index,
                "best_epoch": summary.get("best_epoch", ""),
                "epochs_completed": summary.get("epochs_completed", ""),
                "best_val_score": summary.get("best_val", ""),
                "test_metric": summary.get("current_scorer_metric", ""),
                "independent_pearson_metric": summary.get("independent_pearson_metric", ""),
                "first_test_input_hash": summary.get("input_hash", ""),
                "prediction_hash": summary.get("prediction_hash", ""),
            }
        )
        write_csv(output_dir / "vlaai_p06_repeatability.csv", rows, REPEAT_FIELDS)
        print_event("JOB DONE", f"{job_key(job)} repeat={run_index}")


def run_elasticnet_standardized(config: dict[str, Any], output_dir: Path) -> None:
    job = config["elasticnet_standardized"]
    negative_audit.set_all_rng(int(job["seed"]))
    cfg = source_config(job)
    missing = negative_audit.validate_runtime_config(cfg, "elasticnet")
    if missing:
        raise ValueError(f"runtime config missing required fields before training: {missing}")
    model_cfg = cfg["elasticnet"]
    fit_budget = cfg["fit_budget"]
    dataset_dir = dataset_dir_for(job)
    dataset_id = str(job["dataset"])
    subject_id = str(job["subject_id"])
    start_lag = int(model_cfg["start_lag"])
    end_lag = int(model_cfg["end_lag"])
    x_train, y_train = concatenate_reference_split(dataset_dir, "train", subject_id, channels=range(64))
    x_val, y_val = concatenate_reference_split(dataset_dir, "val", subject_id, channels=range(64))
    x_train_lag_full, _train_idx, y_train_center_full = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag_full, _val_idx, y_val_center_full = trim_valid_range(x_val, y_val, start_lag, end_lag)
    x_train_lag, y_train_center = local_full_eval.cap_fit_samples(x_train_lag_full, y_train_center_full, int(fit_budget["max_fit_samples_per_split"]))
    x_val_lag, y_val_center = local_full_eval.cap_fit_samples(x_val_lag_full, y_val_center_full, int(fit_budget["max_fit_samples_per_split"]))
    scaler = StandardScaler().fit(x_train_lag)
    x_train_scaled = scaler.transform(x_train_lag)
    x_val_scaled = scaler.transform(x_val_lag)
    best_model = None
    best_summary: dict[str, Any] | None = None
    warning_count = 0
    for alpha in model_cfg["alphas"]:
        for l1_ratio in model_cfg["l1_ratios"]:
            model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=int(model_cfg["max_iter"]))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train_scaled, y_train_center)
            warning_count += sum(1 for item in caught if issubclass(item.category, ConvergenceWarning))
            val_pred = model.predict(x_val_scaled).astype(np.float32)
            val_score = local_full_eval.correlation(val_pred, y_val_center.astype(np.float32))
            if best_summary is None or val_score > float(best_summary["best_val_score"]):
                best_model = model
                best_summary = {"best_alpha": float(alpha), "best_l1_ratio": float(l1_ratio), "best_val_score": float(val_score)}
    if best_model is None or best_summary is None:
        raise RuntimeError("elasticnet standardized diagnostic failed to select a model")
    windows: list[WindowPrediction] = []
    offset = end_lag - 1
    checkpoint_id = f"elasticnet_standardized_alpha_{best_summary['best_alpha']}_l1_{best_summary['best_l1_ratio']}"
    for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
        x_lag, _idx, y_center = trim_valid_range(eeg, env, start_lag, end_lag)
        pred = best_model.predict(scaler.transform(x_lag)).astype(np.float32)
        windows.append(
            local_full_eval.build_series_window(
                dataset=dataset_id,
                model="elasticnet_standardized_diagnostic",
                protocol=config["protocol"],
                seed=int(job["seed"]),
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=int(cfg["dataset"]["sampling_rate"]),
                checkpoint_id=checkpoint_id,
                full_length=len(env),
                offset=offset,
                prediction=pred,
                target=y_center.astype(np.float32),
            )
        )
    recording_rows, subject_rows = local_full_eval.aggregate_model_outputs(windows, config["artifact_scope"])
    summary = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": "elasticnet_standardized_diagnostic",
        "seed": int(job["seed"]),
        "selected_alpha": best_summary["best_alpha"],
        "selected_l1_ratio": best_summary["best_l1_ratio"],
        "best_validation_score": best_summary["best_val_score"],
        "final_test_metric": float(subject_rows[0].metric_value) if subject_rows else float("nan"),
        "convergence_warning_count": warning_count,
        "published_result_replaced": False,
        "standardizer_fit_scope": "P11 train lag matrix only",
    }
    write_csv(output_dir / "elasticnet_p11_standardized_recording_metrics.csv", [vars(row) for row in recording_rows])
    write_json(output_dir / "elasticnet_p11_standardized_summary.json", summary)
    (output_dir / "elasticnet_p11_standardized_diagnostic.md").write_text(
        "\n".join(
            [
                "# ElasticNet P11 Standardized Diagnostic",
                "",
                "This is an independent diagnostic result and does not replace the published ElasticNet subject metric.",
                f"- selected_alpha: `{summary['selected_alpha']}`",
                f"- selected_l1_ratio: `{summary['selected_l1_ratio']}`",
                f"- best_validation_score: `{summary['best_validation_score']}`",
                f"- final_test_metric: `{summary['final_test_metric']}`",
                f"- convergence_warning_count: `{summary['convergence_warning_count']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print_event("JOB DONE", f"{job_key(job)} standardized_metric={summary['final_test_metric']}")


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.startup_only:
        print_event("STARTUP", f"config={args.config}")
        print_event("STARTUP", f"output_dir={repo_relative(output_dir)} device={device}")
        print_event("STARTUP", "manual_jobs=vlaai_repeatability,elasticnet_standardized")
        return 0
    schema = build_preflight(config, output_dir, device)
    if args.dry_run:
        print_event("DRY RUN", f"vlaai_repeat_manual={job_key(config['vlaai_repeatability'])}")
        print_event("DRY RUN", f"hq_readonly_audit={job_key(config['happyquokka_p11_readonly'])}")
        print_event("DRY RUN", f"elasticnet_standardized_manual={job_key(config['elasticnet_standardized'])}")
        print_event("DRY RUN", "no_training_executed=true")
    if args.hq_readonly_audit:
        hq_readonly_audit(config, output_dir)
    if args.vlaai_repeat:
        run_vlaai_repeat(config, output_dir, device, args.repeat_count)
    if args.elasticnet_standardized:
        run_elasticnet_standardized(config, output_dir)
    if args.preflight or args.dry_run or args.hq_readonly_audit:
        print_event("PREFLIGHT", f"passed={schema['passed']} output_dir={repo_relative(output_dir)}")
    return 0 if schema["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
