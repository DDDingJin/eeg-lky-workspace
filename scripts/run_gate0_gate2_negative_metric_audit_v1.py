from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")
SHARED_UPSTREAM = SHARED_WORKSPACE / "external" / "upstream" / "mldecoders"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SHARED_UPSTREAM.exists() and str(SHARED_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(SHARED_UPSTREAM))

from repro.reference_baselines import list_reference_subjects, load_reference_recordings
from repro.simple_models import FCNNBaseline
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows
import run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1 as local_full_eval

try:
    from pipeline.dnn import FCNN as UpstreamFCNN
except Exception:
    UpstreamFCNN = None


JOBS_FIELDS = [
    "dataset",
    "subject_id",
    "model",
    "seed",
    "source_config",
    "source_result_dir",
    "dataset_dir",
    "source_metric",
    "train_recordings",
    "val_recordings",
    "test_recordings",
    "adapter_status",
    "manual_command",
]

SHAPE_FIELDS = [
    "dataset",
    "subject_id",
    "model",
    "seed",
    "split",
    "recording_id",
    "eeg_shape",
    "target_shape",
    "input_contract",
    "shape_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shape-preflight", action="store_true")
    parser.add_argument("--identity-audit-only", action="store_true")
    parser.add_argument("--run-diagnostics", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_dataset_path(locator: str) -> Path:
    candidates = [ROOT / locator, SHARED_WORKSPACE / locator]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_shapes_and_value_hash(model: torch.nn.Module) -> tuple[int, str, str]:
    params = [parameter.detach().cpu().contiguous() for parameter in model.parameters()]
    param_count = sum(int(parameter.numel()) for parameter in params)
    shapes = [tuple(parameter.shape) for parameter in params]
    shape_text = json.dumps(shapes)
    digest = hashlib.sha256()
    for parameter in params:
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        digest.update(str(parameter.dtype).encode("utf-8"))
        digest.update(parameter.numpy().tobytes())
    return param_count, shape_text, digest.hexdigest()


def hash_tensor(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    array = np.asarray(array).astype(np.float32, copy=False)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def set_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def lag_sweep(prediction: np.ndarray, target: np.ndarray, min_lag: int = -128, max_lag: int = 128) -> dict[str, Any]:
    zero_lag = pearson_np(prediction, target)
    best_lag = 0
    best_r = -float("inf")
    for lag in range(min_lag, max_lag + 1):
        if lag < 0:
            pred = prediction[-lag:]
            tgt = target[: len(pred)]
        elif lag > 0:
            pred = prediction[: len(prediction) - lag]
            tgt = target[lag : lag + len(pred)]
        else:
            pred = prediction
            tgt = target
        r = pearson_np(pred, tgt)
        if np.isfinite(r) and r > best_r:
            best_lag = lag
            best_r = r
    return {"zero_lag_r": zero_lag, "best_lag_samples": best_lag, "best_lag_r": best_r}


def source_support_config(source_config: dict[str, Any], model: str) -> dict[str, Any]:
    cfg = dict(source_config)
    if model in {"happyquokka", "vlaai", "linear", "lasso", "elasticnet"}:
        local_cfg = load_json(ROOT / "configs" / "benchmark" / "gate0_gate2_subject_specific_local_models_weissbart_full_v1.json")
        for key in ("fit_budget", "linear", "lasso", "elasticnet", "vlaai", "happyquokka"):
            if key in local_cfg and (key not in cfg or model == key and key == "happyquokka" and "win_len_seconds" not in cfg.get(key, {})):
                cfg[key] = local_cfg[key]
        if model == "happyquokka" and "win_len_seconds" not in cfg.get("happyquokka", {}):
            cfg["happyquokka"] = local_cfg["happyquokka"]
    return cfg


def group_captured_windows(windows: list[WindowPrediction]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, list[WindowPrediction]] = {}
    for window in windows:
        grouped.setdefault(window.recording_id, []).append(window)
    output = {}
    for recording_id, recording_windows in grouped.items():
        first = recording_windows[0]
        output[recording_id] = aggregate_overlapping_windows(first.recording_length, recording_windows)
    return output


def first_test_input_hash(*, dataset_dir: Path, subject_id: str, model: str, source_config: dict[str, Any]) -> str:
    _recording_id, eeg, env = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    if model == "vlaai":
        window_size = int(effective_model_config(source_config, "vlaai")["window_size"])
        return hash_array(eeg[:window_size].T[None, :, :])
    if model == "happyquokka":
        cfg = effective_model_config(source_config, "happyquokka")
        input_length = int(cfg["win_len_seconds"]) * int(cfg["sample_rate"])
        return hash_array(eeg[:input_length][None, :, :])
    if model == "elasticnet":
        from repro.mldecoders.cca import trim_valid_range

        cfg = effective_model_config(source_config, "elasticnet")
        start_lag = int(cfg["start_lag"])
        end_lag = int(cfg["end_lag"])
        x_lag, _lag_indexes, _y = trim_valid_range(eeg, env, start_lag, end_lag)
        return hash_array(x_lag[:1])
    return hash_array(eeg[:1])


def run_one_diagnostic(job: dict[str, Any], device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_all_rng(int(job["seed"]))
    source_config = source_support_config(load_json(ROOT / job["source_config"]), str(job["model"]))
    dataset_cfg = source_config["dataset"]
    dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
    dataset_id = str(job["dataset"])
    subject_id = str(job["subject_id"])
    model = str(job["model"])
    sampling_rate = int(dataset_cfg["sampling_rate"])
    captured_windows: list[WindowPrediction] = []
    original_aggregate = local_full_eval.aggregate_model_outputs

    def capture_aggregate(windows: list[WindowPrediction], artifact_scope: str):
        captured_windows.extend(windows)
        return original_aggregate(windows, artifact_scope)

    local_full_eval.aggregate_model_outputs = capture_aggregate
    try:
        if model == "vlaai":
            result = local_full_eval.run_vlaai_full_eval(source_config, dataset_dir, dataset_id, subject_id, sampling_rate, device)
        elif model == "happyquokka":
            result = local_full_eval.run_happyquokka_full_eval(source_config, dataset_dir, dataset_id, subject_id, sampling_rate, device)
        elif model == "elasticnet":
            result = local_full_eval.run_linear_family_full_eval(
                model_name="elasticnet",
                config=source_config,
                dataset_dir=dataset_dir,
                dataset_id=dataset_id,
                subject_id=subject_id,
                sampling_rate=sampling_rate,
            )
        else:
            raise ValueError(f"unsupported diagnostic model: {model}")
    finally:
        local_full_eval.aggregate_model_outputs = original_aggregate

    source_metric = source_subject_metric(ROOT / job["source_result_dir"], dataset_id, subject_id, model, int(job["seed"]))
    current_metric = float(result.subject_rows[0].metric_value) if result.subject_rows else float("nan")
    grouped = group_captured_windows(captured_windows)
    recording_rows = []
    independent_values = []
    for recording_id, (prediction, target, valid_mask) in sorted(grouped.items()):
        valid = valid_mask.astype(bool)
        pred_valid = prediction[valid]
        target_valid = target[valid]
        independent_r = pearson_np(pred_valid, target_valid)
        independent_values.append(independent_r)
        lag = lag_sweep(pred_valid, target_valid)
        recording_rows.append(
            {
                "dataset": dataset_id,
                "subject_id": subject_id,
                "model": model,
                "seed": int(job["seed"]),
                "recording_id": recording_id,
                "zero_lag_pearson": independent_r,
                "num_valid_samples": int(valid.sum()),
                "zero_lag_r": lag["zero_lag_r"],
                "best_lag_samples": lag["best_lag_samples"],
                "best_lag_r": lag["best_lag_r"],
                "prediction_hash": hash_array(pred_valid),
                "target_hash": hash_array(target_valid),
            }
        )
    model_entry = result.model_run_entry
    first_input_hash = first_test_input_hash(
        dataset_dir=dataset_dir,
        subject_id=subject_id,
        model=model,
        source_config=source_config,
    )
    summary = {
        "dataset": dataset_id,
        "subject_id": subject_id,
        "model": model,
        "seed": int(job["seed"]),
        "source_metric": source_metric,
        "current_scorer_metric": current_metric,
        "independent_pearson_metric": float(np.nanmean(independent_values)) if independent_values else float("nan"),
        "current_minus_independent": current_metric - float(np.nanmean(independent_values)) if independent_values else float("nan"),
        "source_minus_current": (source_metric - current_metric) if source_metric is not None else None,
        "train_split_ids": recording_ids(dataset_dir, "train", subject_id),
        "val_split_ids": recording_ids(dataset_dir, "val", subject_id),
        "test_split_ids": recording_ids(dataset_dir, "test", subject_id),
        "best_epoch": model_entry.get("best_epoch", ""),
        "best_val": model_entry.get("best_val_score", ""),
        "epochs_completed": model_entry.get("epochs_completed", ""),
        "input_hash": first_input_hash,
        "target_hash": hash_array(np.concatenate([window.target.reshape(-1) for window in captured_windows])) if captured_windows else "",
        "prediction_hash": hash_array(np.concatenate([window.prediction.reshape(-1) for window in captured_windows])) if captured_windows else "",
        "lag_sweep_policy": "diagnostic only; not used for checkpoint, hyperparameter, or final score selection",
    }
    return summary, recording_rows


def source_subject_metric(source_dir: Path, dataset: str, subject_id: str, model: str, seed: int) -> float | None:
    path = source_dir / "subject_metrics.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") == dataset and row.get("subject_id") == subject_id and row.get("model") == model and int(row.get("seed", 0)) == seed:
                return float(row["metric_value"])
    return None


def recording_ids(dataset_dir: Path, split: str, subject_id: str) -> list[str]:
    return [item[0] for item in load_reference_recordings(dataset_dir, split, subject_id, channels=range(64))]


def input_contract(source_config: dict[str, Any], model: str) -> str:
    if model == "happyquokka":
        cfg = effective_model_config(source_config, model)
        return f"10s_chunk; input_length={int(cfg['win_len_seconds']) * int(cfg['sample_rate'])}; g_con={bool(cfg.get('g_con', False))}"
    if model == "vlaai":
        cfg = effective_model_config(source_config, model)
        return f"window_size={int(cfg['window_size'])}; target_index=last"
    if model == "elasticnet":
        return "lagged EEG matrix; start_lag=0; end_lag=50; budgeted train/val fit samples"
    return "unknown"


def effective_model_config(source_config: dict[str, Any], model: str) -> dict[str, Any]:
    cfg = dict(source_config.get(model, {}))
    if model in {"happyquokka", "vlaai", "linear", "lasso", "elasticnet"} and (
        model not in cfg or model == "happyquokka" and "win_len_seconds" not in cfg
    ):
        local_cfg = load_json(ROOT / "configs" / "benchmark" / "gate0_gate2_subject_specific_local_models_weissbart_full_v1.json")
        cfg.update(local_cfg[model])
    return cfg


def build_job_plan(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for job in config["diagnostic_jobs"]:
        source_config_path = ROOT / job["source_config"]
        source_config = load_json(source_config_path)
        dataset_cfg = source_config["dataset"]
        dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
        source_dir = ROOT / job["source_result_dir"]
        train_ids: list[str] = []
        val_ids: list[str] = []
        test_ids: list[str] = []
        adapter_status = "ok"
        try:
            available = set(list_reference_subjects(dataset_dir, split="test"))
            if job["subject_id"] not in available:
                failures.append(f"{job['dataset']}:{job['subject_id']} missing from test subjects")
                adapter_status = "missing_subject"
            train_ids = recording_ids(dataset_dir, "train", job["subject_id"])
            val_ids = recording_ids(dataset_dir, "val", job["subject_id"])
            test_ids = recording_ids(dataset_dir, "test", job["subject_id"])
            first_split, first_id, first_eeg, first_env = ("test", *load_reference_recordings(dataset_dir, "test", job["subject_id"], channels=range(64))[0])
            shape_rows.append(
                {
                    "dataset": job["dataset"],
                    "subject_id": job["subject_id"],
                    "model": job["model"],
                    "seed": job["seed"],
                    "split": first_split,
                    "recording_id": first_id,
                    "eeg_shape": str(tuple(first_eeg.shape)),
                    "target_shape": str(tuple(first_env.shape)),
                    "input_contract": input_contract(source_config, job["model"]),
                    "shape_status": "ok",
                }
            )
        except Exception as exc:
            failures.append(f"{job['dataset']}:{job['subject_id']}:{job['model']} shape preflight failed: {exc!r}")
            adapter_status = "failed"
        manual_command = (
            "F:\\miniconda\\envs\\decode-torch\\python.exe "
            "scripts\\run_gate0_gate2_negative_metric_audit_v1.py "
            "--config configs\\benchmark\\gate0_gate2_negative_metric_audit_v1.json "
            "--device auto --run-diagnostics"
        )
        rows.append(
            {
                "dataset": job["dataset"],
                "subject_id": job["subject_id"],
                "model": job["model"],
                "seed": job["seed"],
                "source_config": job["source_config"],
                "source_result_dir": job["source_result_dir"],
                "dataset_dir": repo_relative(dataset_dir),
                "source_metric": source_subject_metric(source_dir, job["dataset"], job["subject_id"], job["model"], int(job["seed"])),
                "train_recordings": ";".join(train_ids),
                "val_recordings": ";".join(val_ids),
                "test_recordings": ";".join(test_ids),
                "adapter_status": adapter_status,
                "manual_command": manual_command,
            }
        )
    return rows, shape_rows, failures


def identity_audit(config: dict[str, Any], device: str) -> tuple[list[dict[str, Any]], str]:
    ref_config = load_json(ROOT / config["identity_audit"]["reference_config"])
    fixed_shape = tuple(int(item) for item in config["identity_audit"]["fixed_input_shape"])
    seed = int(config["identity_audit"]["comparison_seed"])
    set_all_rng(seed)
    fixed_input = torch.randn(*fixed_shape, dtype=torch.float32).to(device)
    audit_rows: list[dict[str, Any]] = []
    model_specs = [
        ("fcnn", FCNNBaseline, ROOT / "src" / "repro" / "simple_models.py", ref_config["fcnn"]),
        ("dnn", UpstreamFCNN, SHARED_UPSTREAM / "pipeline" / "dnn.py", ref_config["dnn"]),
    ]
    for model_name, model_class, source_path, cfg in model_specs:
        if model_class is None:
            audit_rows.append(
                {
                    "model": model_name,
                    "class_name": "",
                    "source_path": repo_relative(source_path),
                    "source_sha256": "",
                    "config": json.dumps(cfg, sort_keys=True),
                    "initial_state_hash": "",
                    "parameter_count": "",
                    "parameter_tensor_shapes": "",
                    "parameter_value_hash_ignore_keys": "",
                    "fixed_input_output_hash": "",
                    "status": "class_unavailable",
                }
            )
            continue
        set_all_rng(seed)
        kwargs = {
            "num_hidden": int(cfg["hidden_layers"]),
            "dropout_rate": float(cfg["dropout_rate"]),
            "input_length": int(cfg["window_size"]),
            "num_input_channels": 64,
        }
        model = model_class(**kwargs).to(device)
        parameter_count, parameter_shapes, parameter_value_hash = parameter_shapes_and_value_hash(model)
        model.eval()
        with torch.no_grad():
            output = model(fixed_input)
        audit_rows.append(
            {
                "model": model_name,
                "class_name": f"{model_class.__module__}.{model_class.__qualname__}",
                "source_path": repo_relative(source_path),
                "source_sha256": sha256_file(source_path) if source_path.exists() else "",
                "config": json.dumps(cfg, sort_keys=True),
                "initial_state_hash": hash_state_dict(model.state_dict()),
                "parameter_count": parameter_count,
                "parameter_tensor_shapes": parameter_shapes,
                "parameter_value_hash_ignore_keys": parameter_value_hash,
                "fixed_input_output_hash": hash_tensor(output),
                "status": "ok",
            }
        )
    fcnn = next(row for row in audit_rows if row["model"] == "fcnn")
    dnn = next(row for row in audit_rows if row["model"] == "dnn")
    if fcnn["status"] == "ok" and dnn["status"] == "ok":
        alias = (
            fcnn["parameter_count"] == dnn["parameter_count"]
            and fcnn["parameter_tensor_shapes"] == dnn["parameter_tensor_shapes"]
            and fcnn["parameter_value_hash_ignore_keys"] == dnn["parameter_value_hash_ignore_keys"]
            and fcnn["fixed_input_output_hash"] == dnn["fixed_input_output_hash"]
        )
        conclusion = "functional_alias_of=fcnn" if alias else "dnn not proven functional_alias_of=fcnn; parameter/output evidence differs"
    else:
        conclusion = "identity audit incomplete because one class was unavailable"
    return audit_rows, conclusion


def write_preflight_artifacts(config: dict[str, Any], output_dir: Path, device: str, *, include_identity: bool) -> dict[str, Any]:
    job_rows, shape_rows, failures = build_job_plan(config)
    identity_rows: list[dict[str, Any]] = []
    identity_conclusion = "not_run"
    if include_identity:
        identity_rows, identity_conclusion = identity_audit(config, device)
        write_csv(output_dir / "fcnn_dnn_identity_audit.csv", identity_rows)
    write_csv(output_dir / "diagnostic_job_plan.csv", job_rows, JOBS_FIELDS)
    write_csv(output_dir / "shape_preflight.csv", shape_rows, SHAPE_FIELDS)
    checks = {
        "diagnostic_job_count_is_3": len(job_rows) == 3,
        "shape_preflight_all_jobs": len(shape_rows) == 3,
        "source_metrics_found": all(row["source_metric"] not in {None, ""} for row in job_rows),
        "split_ids_recorded": all(row["train_recordings"] and row["val_recordings"] and row["test_recordings"] for row in job_rows),
        "identity_audit_available": identity_conclusion != "not_run",
        "no_training_executed": True,
        "no_prediction_dump_written": True,
        "no_checkpoint_written": True,
    }
    schema = {
        "passed": all(checks.values()) and not failures,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "identity_conclusion": identity_conclusion,
        "roster_decision": {
            "dnn_main_table_independent_model": False,
            "raw_dnn_rows_retained_for": "implementation-equivalence audit",
            "cross_subject_roster_distinct_methods": 11,
            "cross_subject_roster_excludes": ["dnn"],
        },
        "device": device,
    }
    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(output_dir / "failure_report.json", {"failures": [{"failure_reason": item} for item in failures]})
    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "artifact_scope": config["artifact_scope"],
            "diagnostic_jobs": config["diagnostic_jobs"],
            "diagnostic_policy": config["diagnostic_policy"],
            "manual_command": job_rows[0]["manual_command"] if job_rows else "",
            "preflight_only": True,
            "training_started": False,
            "identity_conclusion": identity_conclusion,
            "roster_decision": schema["roster_decision"],
        },
    )
    lines = [
        "# Negative Metric Diagnostic Preflight",
        "",
        "Status: preflight only. No diagnostic rerun, checkpoint, prediction dump, raw data export, or cross-subject training was executed.",
        "",
        "## Jobs",
    ]
    for row in job_rows:
        lines.append(f"- `{row['dataset']} / {row['subject_id']} / {row['model']} / seed{row['seed']}` source_metric=`{row['source_metric']}`")
    lines += [
        "",
        "## Required Manual Diagnostic Command",
        "```powershell",
        "cd E:\\decode\\_fix_fixed_split_pooled20_modelset_v1",
        job_rows[0]["manual_command"] if job_rows else "",
        "```",
        "",
        "## FCNN/DNN Identity",
        f"- conclusion: `{identity_conclusion}`",
        "- `dnn` must not be treated as an independent main-table model when `functional_alias_of=fcnn`.",
        "- raw `dnn` rows are retained only for implementation-equivalence audit.",
        "- later cross-subject rosters should use 11 distinct methods and exclude `dnn`.",
        "",
        "## Policy",
        "- lag sweep is diagnostic only: -128..+128 samples; it must not select checkpoint, hyperparameter, or final score.",
        "- current scorer metric and independent Pearson must both be reported in the manual diagnostic run.",
    ]
    (output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return schema


def run_diagnostics(config: dict[str, Any], output_dir: Path, device: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    recording_rows = []
    failures = []
    for job in config["diagnostic_jobs"]:
        try:
            summary, rows = run_one_diagnostic(job, device)
            summaries.append(summary)
            recording_rows.extend(rows)
        except Exception as exc:
            failures.append({**job, "failure_reason": repr(exc)})
    write_csv(output_dir / "diagnostic_summary.csv", summaries)
    write_csv(output_dir / "recording_lag_sweep.csv", recording_rows)
    write_json(output_dir / "diagnostic_summary.json", summaries)
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_json(
        output_dir / "schema_validation_report.json",
        {
            "passed": not failures and len(summaries) == len(config["diagnostic_jobs"]),
            "diagnostic_job_count": len(config["diagnostic_jobs"]),
            "completed_diagnostic_count": len(summaries),
            "failure_count": len(failures),
            "prediction_dump_written": False,
            "checkpoint_written": False,
            "lag_sweep_policy": "diagnostic only; not used for checkpoint, hyperparameter, or final score selection",
        },
    )
    lines = [
        "# Negative Metric Diagnostic Run",
        "",
        "This artifact is generated only when `--run-diagnostics` is explicitly supplied.",
        "No prediction dump, raw data, checkpoint, or model weight is written.",
        "",
        "## Jobs",
    ]
    for item in summaries:
        lines.append(
            f"- `{item['dataset']} / {item['subject_id']} / {item['model']}`: "
            f"source=`{item['source_metric']}`, current=`{item['current_scorer_metric']}`, "
            f"independent=`{item['independent_pearson_metric']}`"
        )
    (output_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if not failures else 1


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_diagnostics:
        return run_diagnostics(config, output_dir, device)
    include_identity = args.identity_audit_only or args.shape_preflight or args.dry_run
    schema = write_preflight_artifacts(config, output_dir, device, include_identity=include_identity)
    print(f"[PREFLIGHT] output_dir={repo_relative(output_dir)} passed={schema['passed']} device={device}")
    print(f"[PREFLIGHT] identity_conclusion={schema['identity_conclusion']}")
    print("[PREFLIGHT] no_training_executed=true")
    return 0 if schema["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
