from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repro.mldecoders.models import EEGNetRegressor


ATOMIC_RETRY_ATTEMPTS = 5
ATOMIC_RETRY_SLEEP_SECONDS = 0.05
COMPARISON_FIELDS = [
    "source_dataset",
    "target_dataset",
    "model",
    "seed",
    "stage",
    "subject_id",
    "zero_shot_metric",
    "calibrated_metric",
    "delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source-dataset", nargs="*", default=None)
    parser.add_argument("--target-dataset", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--stage", choices=["cross_dataset_zero_shot", "pooled10_target_calibration", "all"], default="all")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--duration-audit-only", action="store_true")
    parser.add_argument("--shape-check-only", action="store_true")
    parser.add_argument("--checkpoint-path-check-only", action="store_true")
    parser.add_argument("--job-plan-only", action="store_true")
    parser.add_argument("--rebuild-preflight-artifacts-only", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_tmp_path(path: Path) -> Path:
    unique = f"{os.getpid()}_{threading.get_ident()}_{int(time.time() * 1000)}_{uuid.uuid4().hex}"
    return path.with_name(f"{path.name}.{unique}.tmp")


def atomic_replace_with_retry(tmp: Path, path: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, ATOMIC_RETRY_ATTEMPTS + 1):
        try:
            tmp.replace(path)
            return
        except (PermissionError, OSError) as exc:
            last_error = exc
            if attempt == ATOMIC_RETRY_ATTEMPTS:
                raise
            time.sleep(ATOMIC_RETRY_SLEEP_SECONDS * attempt)
    if last_error is not None:
        raise last_error


def atomic_write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    atomic_replace_with_retry(tmp, path)


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    tmp.write_text(content, encoding="utf-8")
    atomic_replace_with_retry(tmp, path)


def atomic_write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    tmp = build_tmp_path(path)
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    atomic_replace_with_retry(tmp, path)


def read_json(path: Path, default: object) -> object:
    if not path.exists() or path.stat().st_size == 0:
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_relative(path: Path) -> str:
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_candidate = Path(ROOT.drive + "\\decode") / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def current_git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        return None
    return None


def current_git_branch_name() -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        return None
    return None


def stdout_message(message: str) -> None:
    print(message, flush=True)


def stdout_block(title: str, lines: list[str]) -> None:
    print(title, flush=True)
    for line in lines:
        print(line, flush=True)


def load_split_manifest(path: Path) -> dict[str, object]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        raise ValueError(f"invalid split manifest: {path}")
    required = {"split_id", "dataset_id", "train_subjects", "val_subjects", "test_subjects", "subject_ids_all", "split_seed", "split_rule", "created_at", "no_overlap"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"split manifest missing fields: {sorted(missing)}")
    return payload


def model_contract(config: dict, model_name: str) -> dict[str, object]:
    if model_name != "eegnet":
        raise ValueError(f"unsupported model for cross-dataset transfer preflight: {model_name}")
    return {
        "family": "window_scalar",
        "input_tensor_shape": [1, 64, int(config["eegnet"]["window_size"])],
        "raw_output_shape": [1],
        "postprocessed_prediction_shape": [1],
        "scorer_input_shape": [1],
        "target_index": str(config["eegnet"]["target_index"]),
        "target_contract": "scalar target per window",
    }


def instantiate_model(config: dict, model_name: str) -> tuple[object, dict[str, object]]:
    if model_name != "eegnet":
        raise ValueError(f"unsupported model for cross-dataset transfer preflight: {model_name}")
    cfg = config["eegnet"]
    return EEGNetRegressor, {
        "num_input_channels": 64,
        "input_length": int(cfg["window_size"]),
        "temporal_filters": int(cfg["temporal_filters"]),
        "depth_multiplier": int(cfg["depth_multiplier"]),
        "separable_filters": int(cfg["separable_filters"]),
        "dropout_rate": float(cfg["dropout_rate"]),
    }


def discover_recordings(dataset_dir: Path, *, split: str, subject_id: str, sampling_rate: int, window_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eeg_path in sorted(dataset_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        num_samples = int(np.load(eeg_path, mmap_mode="r").shape[0])
        window_count = max(0, num_samples - window_size + 1)
        rows.append(
            {
                "recording_id": recording_id,
                "num_samples": num_samples,
                "seconds": float(num_samples / sampling_rate),
                "window_count": int(window_count),
            }
        )
    return rows


def duration_prefix(rows: list[dict[str, object]], budget_seconds: float) -> tuple[list[dict[str, object]], float]:
    selected: list[dict[str, object]] = []
    total = 0.0
    for row in rows:
        selected.append(row)
        total += float(row["seconds"])
        if total >= budget_seconds:
            break
    return selected, total


def dataset_cfg_map(config: dict) -> dict[str, dict[str, object]]:
    return {str(item["dataset_id"]): item for item in config["datasets"]}


def resolve_transfer_pairs(config: dict, requested_sources: list[str] | None, requested_targets: list[str] | None) -> list[dict[str, str]]:
    pairs = [dict(item) for item in config["transfer_pairs"]]
    if requested_sources:
        source_set = set(requested_sources)
        pairs = [pair for pair in pairs if pair["source_dataset"] in source_set]
    if requested_targets:
        target_set = set(requested_targets)
        pairs = [pair for pair in pairs if pair["target_dataset"] in target_set]
    if not pairs:
        raise ValueError("no transfer pairs selected")
    return pairs


def resolve_selected_models(config: dict, requested: list[str] | None) -> list[str]:
    models = list(config["models"])
    if not requested:
        return models
    missing = sorted(set(requested) - set(models))
    if missing:
        raise ValueError(f"unknown models requested: {missing}")
    return list(requested)


def build_transfer_duration_audit(config: dict, pair: dict[str, str], datasets_by_id: dict[str, dict[str, object]]) -> dict[str, object]:
    source_cfg = datasets_by_id[pair["source_dataset"]]
    target_cfg = datasets_by_id[pair["target_dataset"]]
    source_manifest = load_split_manifest(ROOT / str(source_cfg["split_manifest_path"]))
    target_manifest = load_split_manifest(ROOT / str(target_cfg["split_manifest_path"]))
    target_dir = resolve_dataset_path(str(target_cfg["dataset_locator"]))
    sampling_rate = int(target_cfg["sampling_rate"])
    window_size = int(config["target_calibration"]["calibration_window_size_for_audit"])
    subjects_report: list[dict[str, object]] = []
    pooled = {
        "train_seconds": 0.0,
        "val_seconds": 0.0,
        "calibration_total_seconds": 0.0,
        "final_test_seconds": 0.0,
        "train_recording_count": 0,
        "val_recording_count": 0,
        "final_test_recording_count": 0,
        "train_window_count": 0,
        "val_window_count": 0,
        "final_test_window_count": 0,
    }
    for subject_id in target_manifest["test_subjects"]:
        train_rows = discover_recordings(target_dir, split="train", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=window_size)
        val_rows = discover_recordings(target_dir, split="val", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=window_size)
        test_rows = discover_recordings(target_dir, split="test", subject_id=str(subject_id), sampling_rate=sampling_rate, window_size=window_size)
        train_selected, train_seconds = duration_prefix(train_rows, float(config["target_calibration"]["train_budget_seconds"]))
        val_selected, val_seconds = duration_prefix(val_rows, float(config["target_calibration"]["val_budget_seconds"]))
        test_seconds = float(sum(row["seconds"] for row in test_rows))
        test_windows = int(sum(row["window_count"] for row in test_rows))
        subject_row = {
            "subject_id": str(subject_id),
            "train_total_seconds": float(sum(row["seconds"] for row in train_rows)),
            "val_total_seconds": float(sum(row["seconds"] for row in val_rows)),
            "test_total_seconds": test_seconds,
            "calibration": {
                "required_seconds": float(config["target_calibration"]["calibration_budget_seconds"]),
                "train_budget_seconds": float(config["target_calibration"]["train_budget_seconds"]),
                "val_budget_seconds": float(config["target_calibration"]["val_budget_seconds"]),
                "actual_train_seconds": train_seconds,
                "actual_val_seconds": val_seconds,
                "actual_total_seconds": train_seconds + val_seconds,
                "train_recording_count": len(train_selected),
                "val_recording_count": len(val_selected),
                "train_window_count": int(sum(row["window_count"] for row in train_selected)),
                "val_window_count": int(sum(row["window_count"] for row in val_selected)),
                "train_recording_ids": [str(row["recording_id"]) for row in train_selected],
                "val_recording_ids": [str(row["recording_id"]) for row in val_selected],
                "supports_10min": train_seconds >= float(config["target_calibration"]["train_budget_seconds"]) and val_seconds >= float(config["target_calibration"]["val_budget_seconds"]),
            },
            "final_test": {
                "recording_count": len(test_rows),
                "seconds": test_seconds,
                "window_count": test_windows,
                "recording_ids": [str(row["recording_id"]) for row in test_rows],
                "non_empty": len(test_rows) > 0 and test_windows > 0,
            },
        }
        subjects_report.append(subject_row)
        pooled["train_seconds"] += train_seconds
        pooled["val_seconds"] += val_seconds
        pooled["calibration_total_seconds"] += train_seconds + val_seconds
        pooled["final_test_seconds"] += test_seconds
        pooled["train_recording_count"] += len(train_selected)
        pooled["val_recording_count"] += len(val_selected)
        pooled["final_test_recording_count"] += len(test_rows)
        pooled["train_window_count"] += int(sum(row["window_count"] for row in train_selected))
        pooled["val_window_count"] += int(sum(row["window_count"] for row in val_selected))
        pooled["final_test_window_count"] += test_windows
    return {
        "source_dataset": str(pair["source_dataset"]),
        "target_dataset": str(pair["target_dataset"]),
        "source_split_id": str(source_manifest["split_id"]),
        "target_split_id": str(target_manifest["split_id"]),
        "source_train_subjects": list(source_manifest["train_subjects"]),
        "source_val_subjects": list(source_manifest["val_subjects"]),
        "source_test_subjects_unused_for_training": list(source_manifest["test_subjects"]),
        "target_train_subjects_unused_for_source_training": list(target_manifest["train_subjects"]),
        "target_val_subjects_unused_for_source_training": list(target_manifest["val_subjects"]),
        "target_test_subjects": list(target_manifest["test_subjects"]),
        "subjects": subjects_report,
        "pooled_target_calibration": pooled,
        "all_target_test_subjects_support_10min": all(
            bool(subject["calibration"]["supports_10min"]) and bool(subject["final_test"]["non_empty"]) for subject in subjects_report
        ),
        "final_test_uses_target_original_test_split": True,
    }


def shape_check_for_model(config: dict, model_name: str, device: str) -> dict[str, object]:
    model_handle, model_kwargs = instantiate_model(config, model_name)
    contract = model_contract(config, model_name)
    model = model_handle(**model_kwargs).to(device)
    model.eval()
    x = torch.zeros(*contract["input_tensor_shape"], device=device, dtype=torch.float32)
    with torch.no_grad():
        raw = model(x)
    return {
        "model": model_name,
        "input_shape": list(contract["input_tensor_shape"]),
        "raw_output_shape": list(raw.shape),
        "postprocessed_prediction_shape": [int(raw.reshape(-1).shape[0])],
        "scorer_input_shape": contract["scorer_input_shape"],
        "target_index": contract["target_index"],
        "target_contract": contract["target_contract"],
        "passed": list(raw.shape) == contract["raw_output_shape"],
    }


def cross_zero_shot_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "cross_dataset" / "zero_shot"


def cross_calibration_checkpoint_root() -> Path:
    return ROOT / "local_checkpoints" / "cross_dataset" / "pooled10_target_calibration"


def cross_zero_shot_checkpoint_path_pattern(*, source_dataset: str, target_dataset: str, model: str, seed: int) -> Path:
    return cross_zero_shot_checkpoint_root() / model / source_dataset / target_dataset / f"seed{seed}" / "best_epoch_PENDING.pt"


def cross_calibration_checkpoint_path_pattern(*, source_dataset: str, target_dataset: str, model: str, seed: int) -> Path:
    return cross_calibration_checkpoint_root() / model / source_dataset / target_dataset / f"seed{seed}" / "best_epoch_PENDING.pt"


def load_latest_local_checkpoint(path: Path) -> Path | None:
    if not path.parent.exists():
        return None
    candidates = sorted(path.parent.glob("best_epoch_*.pt"))
    return candidates[-1] if candidates else None


def checkpoint_path_check(config: dict, transfer_audits: list[dict[str, object]], selected_models: list[str], device: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in transfer_audits:
        for model_name in selected_models:
            zero_pattern = cross_zero_shot_checkpoint_path_pattern(
                source_dataset=str(audit["source_dataset"]),
                target_dataset=str(audit["target_dataset"]),
                model=model_name,
                seed=int(config["seed"]),
            )
            cal_pattern = cross_calibration_checkpoint_path_pattern(
                source_dataset=str(audit["source_dataset"]),
                target_dataset=str(audit["target_dataset"]),
                model=model_name,
                seed=int(config["seed"]),
            )
            zero_existing = load_latest_local_checkpoint(zero_pattern)
            model_handle, model_kwargs = instantiate_model(config, model_name)
            model = model_handle(**model_kwargs)
            smoke_path = zero_pattern.parent / "smoke_checkpoint_tmp.pt"
            ensure_dir(smoke_path.parent)
            torch.save(
                {
                    "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "source_dataset": str(audit["source_dataset"]),
                    "target_dataset": str(audit["target_dataset"]),
                    "model": model_name,
                    "seed": int(config["seed"]),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                },
                smoke_path,
            )
            smoke_payload = torch.load(smoke_path, map_location="cpu")
            model.load_state_dict(smoke_payload["model_state_dict"], strict=True)
            smoke_path.unlink(missing_ok=True)
            load_ok = True
            strict_ok = True
            if zero_existing is not None and zero_existing.exists():
                payload = torch.load(zero_existing, map_location="cpu")
                model.load_state_dict(payload["model_state_dict"], strict=True)
            rows.append(
                {
                    "source_dataset": str(audit["source_dataset"]),
                    "target_dataset": str(audit["target_dataset"]),
                    "model": model_name,
                    "seed": int(config["seed"]),
                    "cross_dataset_zero_shot_checkpoint_path_pattern": repo_relative(zero_pattern),
                    "pooled10_target_calibration_checkpoint_path_pattern": repo_relative(cal_pattern),
                    "existing_zero_shot_checkpoint": repo_relative(zero_existing) if zero_existing is not None else None,
                    "torch_load_smoke": load_ok,
                    "strict_load_state_dict_smoke": strict_ok,
                }
            )
    return rows


def build_job_key(source_dataset: str, target_dataset: str, model: str, seed: int, stage: str) -> str:
    return f"{source_dataset}->{target_dataset}:{model}:seed{seed}:{stage}"


def build_job_plan(config: dict, transfer_audits: list[dict[str, object]], selected_models: list[str], stage: str) -> dict[str, object]:
    jobs: list[dict[str, object]] = []
    for audit in transfer_audits:
        source_dataset = str(audit["source_dataset"])
        target_dataset = str(audit["target_dataset"])
        for model_name in selected_models:
            if stage in {"cross_dataset_zero_shot", "all"}:
                jobs.append(
                    {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "stage": "cross_dataset_zero_shot",
                        "job_key": build_job_key(source_dataset, target_dataset, model_name, int(config["seed"]), "cross_dataset_zero_shot"),
                    }
                )
            if stage in {"pooled10_target_calibration", "all"}:
                jobs.append(
                    {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "seed": int(config["seed"]),
                        "stage": "pooled10_target_calibration",
                        "job_key": build_job_key(source_dataset, target_dataset, model_name, int(config["seed"]), "pooled10_target_calibration"),
                        "depends_on_zero_shot": True,
                    }
                )
    return {
        "planned_jobs": jobs,
        "cross_dataset_zero_shot_job_count": sum(1 for job in jobs if job["stage"] == "cross_dataset_zero_shot"),
        "pooled10_target_calibration_job_count": sum(1 for job in jobs if job["stage"] == "pooled10_target_calibration"),
    }


def build_manual_commands(config_path: Path, transfer_audits: list[dict[str, object]], selected_models: list[str]) -> list[str]:
    sources = " ".join(str(item["source_dataset"]) for item in transfer_audits)
    targets = " ".join(str(item["target_dataset"]) for item in transfer_audits)
    models_arg = " ".join(selected_models)
    config_path_ps = str(config_path).replace("/", "\\")
    return [
        f"cd {ROOT}",
        f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_cross_dataset_transfer_v1.py --config {config_path_ps} --device auto --resume --source-dataset {sources} --target-dataset {targets} --models {models_arg} --stage cross_dataset_zero_shot --max-jobs 1",
        f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_cross_dataset_transfer_v1.py --config {config_path_ps} --device auto --resume --source-dataset {sources} --target-dataset {targets} --models {models_arg} --stage pooled10_target_calibration --max-jobs 1",
        f"F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_cross_dataset_transfer_v1.py --config {config_path_ps} --device auto --resume --source-dataset {sources} --target-dataset {targets} --models {models_arg} --stage all --max-jobs 2",
    ]


def build_job_plan_only_summary(transfer_audits: list[dict[str, object]], selected_models: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in transfer_audits:
        for model_name in selected_models:
            rows.append(
                {
                    "source_dataset": audit["source_dataset"],
                    "target_dataset": audit["target_dataset"],
                    "model": model_name,
                    "source_train_subjects": audit["source_train_subjects"],
                    "source_val_subjects": audit["source_val_subjects"],
                    "target_test_subjects": audit["target_test_subjects"],
                    "target_pooled_train_recording_count": audit["pooled_target_calibration"]["train_recording_count"],
                    "target_pooled_val_recording_count": audit["pooled_target_calibration"]["val_recording_count"],
                    "target_final_test_recording_count": audit["pooled_target_calibration"]["final_test_recording_count"],
                    "target_pooled_train_window_count": audit["pooled_target_calibration"]["train_window_count"],
                    "target_pooled_val_window_count": audit["pooled_target_calibration"]["val_window_count"],
                    "input_tensor_contract": "[batch, 64, 50] -> [batch]",
                    "target_scorer_alignment": "scalar target per window, Pearson on valid overlap",
                }
            )
    return rows


def preflight_output_dir(output_dir: Path) -> Path:
    return output_dir / "preflight"


def write_preflight_artifacts(
    *,
    config: dict,
    config_path: Path,
    output_dir: Path,
    transfer_audits: list[dict[str, object]],
    shape_checks: list[dict[str, object]],
    checkpoint_checks: list[dict[str, object]],
    job_plan: dict[str, object],
    job_plan_only: list[dict[str, object]],
) -> None:
    target_dir = preflight_output_dir(output_dir)
    ensure_dir(target_dir)
    run_manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "base_branch": config["base_branch"],
        "base_commit": config["base_commit"],
        "current_branch": current_git_branch_name(),
        "config_path": repo_relative(config_path),
        "commit_sha": current_git_commit_sha(),
        "stage_scope": "preflight_only_no_training_started",
        "artifacts": {
            "transfer_plan": repo_relative(target_dir / "transfer_plan.json"),
            "calibration_plan": repo_relative(target_dir / "calibration_plan.json"),
            "checkpoint_manifest": repo_relative(target_dir / "checkpoint_manifest.json"),
            "model_run_entries": repo_relative(target_dir / "model_run_entries.json"),
            "schema_validation_report": repo_relative(target_dir / "schema_validation_report.json"),
            "failure_report": repo_relative(target_dir / "failure_report.json"),
            "job_plan_only": repo_relative(target_dir / "job_plan_only.json"),
        },
    }
    calibration_plan = {
        "protocol": config["protocol"],
        "selection_rule": config["target_calibration"]["selection_rule"],
        "transfer_pairs": transfer_audits,
    }
    transfer_plan = {
        "protocol": config["protocol"],
        "pairs": transfer_audits,
    }
    checkpoint_manifest = {"checkpoints": checkpoint_checks}
    model_run_entries = {"model_run_entries": shape_checks}
    failures = {"failures": []}
    schema_validation = {
        "protocol": config["protocol"],
        "passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits) and all(item["passed"] for item in shape_checks),
        "checks": {
            "split_manifests_loaded": True,
            "target_calibration_audit_passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits),
            "shape_checks_passed": all(item["passed"] for item in shape_checks),
            "failure_report_empty": True,
            "cross_dataset_stage_requires_training": True,
        },
        "notes": ["preflight only; no cross-dataset zero-shot or pooled10 target calibration jobs executed"],
    }
    atomic_write_json(target_dir / "run_manifest.json", run_manifest)
    atomic_write_json(target_dir / "transfer_plan.json", transfer_plan)
    atomic_write_json(target_dir / "calibration_plan.json", calibration_plan)
    atomic_write_json(target_dir / "checkpoint_manifest.json", checkpoint_manifest)
    atomic_write_json(target_dir / "model_run_entries.json", model_run_entries)
    atomic_write_json(target_dir / "completed_jobs.json", {"completed_jobs": []})
    atomic_write_json(target_dir / "run_state.json", {"phase": "preflight_only", "pending_jobs": job_plan["planned_jobs"], "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
    atomic_write_json(target_dir / "failure_report.json", failures)
    atomic_write_json(target_dir / "schema_validation_report.json", schema_validation)
    atomic_write_json(target_dir / "job_plan_only.json", {"jobs": job_plan_only})
    atomic_write_text(target_dir / "transfer_or_leakage_summary.md", "# Preflight Only\n\nNo training started.\n")


def write_runtime_state(
    *,
    config: dict,
    config_path: Path,
    output_dir: Path,
    transfer_audits: list[dict[str, object]],
    job_plan: dict[str, object],
) -> None:
    completed_job_keys = list(read_json(output_dir / "completed_jobs.json", {"completed_jobs": []}).get("completed_jobs", []))
    completed_keys = {str(item["job_key"]) for item in completed_job_keys}
    pending_jobs = [job for job in job_plan["planned_jobs"] if str(job["job_key"]) not in completed_keys]
    leakage_lines = [
        "# Transfer / Leakage Summary",
        "",
        f"- protocol: `{config['protocol']}`",
        "- source training uses only source dataset train subjects",
        "- source checkpoint selection uses only source dataset val subjects",
        "- target pooled calibration uses only target original train/val recordings of target fixed test subjects",
        "- target original test split remains final test only",
        "",
    ]
    for audit in transfer_audits:
        leakage_lines.append(f"## {audit['source_dataset']} -> {audit['target_dataset']}")
        leakage_lines.append(f"- source_train_subjects: `{', '.join(audit['source_train_subjects'])}`")
        leakage_lines.append(f"- source_val_subjects: `{', '.join(audit['source_val_subjects'])}`")
        leakage_lines.append(f"- target_test_subjects: `{', '.join(audit['target_test_subjects'])}`")
        leakage_lines.append(f"- target_pooled_train_seconds: `{audit['pooled_target_calibration']['train_seconds']}`")
        leakage_lines.append(f"- target_pooled_val_seconds: `{audit['pooled_target_calibration']['val_seconds']}`")
        leakage_lines.append(f"- target_final_test_seconds: `{audit['pooled_target_calibration']['final_test_seconds']}`")
        leakage_lines.append("")
    schema_validation = {
        "protocol": config["protocol"],
        "passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits),
        "checks": {
            "target_calibration_audit_passed": all(item["all_target_test_subjects_support_10min"] for item in transfer_audits),
            "failure_report_empty": True,
            "runtime_jobs_executed": False,
        },
        "notes": ["preflight closure only; runtime files initialized without training"],
    }
    run_manifest = {
        "protocol": config["protocol"],
        "artifact_scope": config["artifact_scope"],
        "base_branch": config["base_branch"],
        "base_commit": config["base_commit"],
        "current_branch": current_git_branch_name(),
        "config_path": repo_relative(config_path),
        "commit_sha": current_git_commit_sha(),
        "artifacts": {
            "transfer_plan": repo_relative(output_dir / "transfer_plan.json"),
            "calibration_plan": repo_relative(output_dir / "calibration_plan.json"),
            "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
            "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
            "dataset_metrics": repo_relative(output_dir / "dataset_metrics.csv"),
            "comparison": repo_relative(output_dir / "cross_dataset_zero_shot_vs_calibrated_comparison.csv"),
            "checkpoint_manifest": repo_relative(output_dir / "checkpoint_manifest.json"),
            "model_run_entries": repo_relative(output_dir / "model_run_entries.json"),
            "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
            "run_state": repo_relative(output_dir / "run_state.json"),
            "failure_report": repo_relative(output_dir / "failure_report.json"),
            "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
            "transfer_or_leakage_summary": repo_relative(output_dir / "transfer_or_leakage_summary.md"),
            "logs": repo_relative(output_dir / "logs"),
        },
    }
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    atomic_write_json(output_dir / "transfer_plan.json", {"pairs": transfer_audits})
    atomic_write_json(output_dir / "calibration_plan.json", {"selection_rule": config["target_calibration"]["selection_rule"], "pairs": transfer_audits})
    atomic_write_json(output_dir / "checkpoint_manifest.json", {"checkpoints": []})
    atomic_write_json(output_dir / "model_run_entries.json", {"model_run_entries": []})
    atomic_write_json(output_dir / "completed_jobs.json", {"completed_jobs": []})
    atomic_write_json(output_dir / "failure_report.json", {"failures": []})
    atomic_write_json(output_dir / "schema_validation_report.json", schema_validation)
    atomic_write_json(
        output_dir / "run_state.json",
        {
            "phase": "preflight_only",
            "completed_job_keys": [],
            "pending_jobs": pending_jobs,
            "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )
    atomic_write_text(output_dir / "transfer_or_leakage_summary.md", "\n".join(leakage_lines) + "\n")
    atomic_write_csv(output_dir / "subject_metrics.csv", [], ["source_dataset", "target_dataset", "model", "seed", "stage", "subject_id", "metric_value", "checkpoint_id"])
    atomic_write_csv(output_dir / "recording_metrics.csv", [], ["source_dataset", "target_dataset", "model", "seed", "stage", "subject_id", "recording_id", "metric_value", "checkpoint_id"])
    atomic_write_csv(output_dir / "dataset_metrics.csv", [], ["source_dataset", "target_dataset", "model", "seed", "stage", "n_subjects", "mean_pearson", "std_pearson", "min_pearson", "max_pearson"])
    atomic_write_csv(output_dir / "cross_dataset_zero_shot_vs_calibrated_comparison.csv", [], COMPARISON_FIELDS)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    device = resolve_device(args.device)
    datasets_by_id = dataset_cfg_map(config)
    selected_models = resolve_selected_models(config, args.models)
    transfer_pairs = resolve_transfer_pairs(config, args.source_dataset, args.target_dataset)
    transfer_audits = [build_transfer_duration_audit(config, pair, datasets_by_id) for pair in transfer_pairs]
    shape_checks = [shape_check_for_model(config, model_name, device) for model_name in selected_models]
    checkpoint_checks = checkpoint_path_check(config, transfer_audits, selected_models, device)
    job_plan = build_job_plan(config, transfer_audits, selected_models, args.stage)
    job_plan_only = build_job_plan_only_summary(transfer_audits, selected_models)
    manual_commands = build_manual_commands(config_path, transfer_audits, selected_models)
    preflight_mode = any(
        [
            args.duration_audit_only,
            args.shape_check_only,
            args.checkpoint_path_check_only,
            args.startup_only,
            args.dry_run_plan,
            args.job_plan_only,
            args.rebuild_preflight_artifacts_only,
        ]
    )
    if preflight_mode:
        write_preflight_artifacts(
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            transfer_audits=transfer_audits,
            shape_checks=shape_checks,
            checkpoint_checks=checkpoint_checks,
            job_plan=job_plan,
            job_plan_only=job_plan_only,
        )
    if args.duration_audit_only:
        stdout_block("[DURATION AUDIT]", [json.dumps({"transfer_pairs": transfer_audits}, indent=2)])
        return 0
    if args.shape_check_only:
        stdout_block("[SHAPE CHECK]", [json.dumps({"models": shape_checks}, indent=2)])
        return 0
    if args.checkpoint_path_check_only:
        stdout_block("[CHECKPOINT PATH CHECK]", [json.dumps({"checkpoints": checkpoint_checks}, indent=2)])
        return 0
    if args.job_plan_only:
        stdout_block("[JOB PLAN ONLY]", [json.dumps({"jobs": job_plan_only}, indent=2)])
        return 0
    if args.startup_only or args.dry_run_plan:
        transfer_pair_labels = [f"{item['source_dataset']}->{item['target_dataset']}" for item in transfer_audits]
        startup_lines = [
            f"config path: {config_path}",
            f"output_dir: {output_dir}",
            f"device: {device}",
            f"transfer_pairs: {transfer_pair_labels}",
            f"models: {selected_models}",
            f"stage: {args.stage}",
            f"planned_jobs: {len(job_plan['planned_jobs'])}",
            f"cross_dataset_zero_shot_jobs: {job_plan['cross_dataset_zero_shot_job_count']}",
            f"pooled10_target_calibration_jobs: {job_plan['pooled10_target_calibration_job_count']}",
            "no training started: True",
        ]
        for audit in transfer_audits:
            startup_lines.append(
                f"{audit['source_dataset']}->{audit['target_dataset']} target_test_subjects: {audit['target_test_subjects']}"
            )
        stdout_block("[STARTUP SUMMARY]", startup_lines)
        if args.dry_run_plan:
            stdout_block("[DRY RUN PLAN]", [json.dumps(job_plan, indent=2), "", "Recommended manual commands:"] + manual_commands)
        return 0
    if args.rebuild_preflight_artifacts_only:
        stdout_block("[REBUILD DONE]", ["preflight artifacts rewritten", "no training started"])
        return 0

    # This closure intentionally stops at preflight.
    write_runtime_state(
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        transfer_audits=transfer_audits,
        job_plan=job_plan,
    )
    stdout_block(
        "[PRECHECK COMPLETE]",
        [
            "cross-dataset transfer preflight prepared",
            "no training started",
            "runtime placeholders initialized for future manual runs",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
