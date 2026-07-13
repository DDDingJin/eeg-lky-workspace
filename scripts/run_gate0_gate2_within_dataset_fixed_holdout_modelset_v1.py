from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")


SMOKE_FIELDS = [
    "job_key",
    "stage",
    "dataset",
    "model",
    "seed",
    "split_id",
    "train_subject_count",
    "val_subject_count",
    "test_subject_count",
    "status",
]

PREFLIGHT_FIELDS = [
    "dataset",
    "model",
    "stage",
    "split_id",
    "train_subjects",
    "val_subjects",
    "test_subjects",
    "train_val_intersection",
    "train_test_intersection",
    "val_test_intersection",
    "first_train_recording",
    "first_train_eeg_shape",
    "first_train_target_shape",
    "input_contract",
    "scorer",
    "coverage_check",
    "checkpoint_path",
    "zero_shot_status",
]

MODELS = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "cca",
    "fcnn",
    "cnn",
    "eegnet",
    "adt",
    "vlaai",
    "happyquokka",
]

LINEAR_HELD = {"linear", "ridge", "lasso", "elasticnet", "cca"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter.")
    parser.add_argument("--models", default="", help="Comma-separated model filter.")
    parser.add_argument("--stage", choices=["smoke", "zero_shot"], default="smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_filter(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_candidate = SHARED_WORKSPACE / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def output_dir(config: dict[str, Any], stage: str) -> Path:
    key = "smoke_output_dir" if stage == "smoke" else "zero_shot_output_dir"
    return ROOT / config[key]


def completed_keys(out_dir: Path) -> set[str]:
    path = out_dir / "completed_jobs.json"
    if not path.exists():
        return set()
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return set()
    return {str(item.get("job_key", "")) for item in payload}


def load_split(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    return load_json(ROOT / config["datasets"][dataset]["split_manifest_path"])


def validate_split(split: dict[str, Any], dataset: str) -> dict[str, Any]:
    train = set(split["train_subjects"])
    val = set(split["val_subjects"])
    test = set(split["test_subjects"])
    return {
        "dataset": dataset,
        "split_id": split["split_id"],
        "train_subjects": list(split["train_subjects"]),
        "val_subjects": list(split["val_subjects"]),
        "test_subjects": list(split["test_subjects"]),
        "train_val_intersection": sorted(train & val),
        "train_test_intersection": sorted(train & test),
        "val_test_intersection": sorted(val & test),
        "no_overlap": not (train & val or train & test or val & test) and bool(split.get("no_overlap", False)),
    }


def job_key(dataset: str, model: str, seed: int, stage: str) -> str:
    return f"{dataset}:{model}:seed{seed}:stage={stage}"


def build_jobs(config: dict[str, Any], stage: str, dataset_filter: set[str] | None, model_filter: set[str] | None) -> list[dict[str, Any]]:
    datasets = [item for item in config["datasets"] if dataset_filter is None or item in dataset_filter]
    models = [item for item in config["models"] if model_filter is None or item in model_filter]
    unknown_datasets = sorted((dataset_filter or set()) - set(config["datasets"]))
    unknown_models = sorted((model_filter or set()) - set(config["models"]))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}")
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        split = load_split(config, dataset)
        for model in models:
            jobs.append(
                {
                    "job_key": job_key(dataset, model, int(config["seed"]), stage),
                    "stage": stage,
                    "dataset": dataset,
                    "model": model,
                    "seed": int(config["seed"]),
                    "split_id": split["split_id"],
                    "train_subject_count": len(split["train_subjects"]),
                    "val_subject_count": len(split["val_subjects"]),
                    "test_subject_count": len(split["test_subjects"]),
                    "status": "pending",
                }
            )
    return jobs


def first_recording_shape(dataset_dir: Path, split_name: str, subject: str) -> tuple[str, str, str]:
    eeg_paths = sorted(dataset_dir.glob(f"{split_name}_-_{subject}_-_*_-_eeg.npy"))
    if not eeg_paths:
        return "", "", ""
    eeg_path = eeg_paths[0]
    recording_id = eeg_path.stem.replace("_-_eeg", "")
    env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
    eeg = np.load(eeg_path, mmap_mode="r")
    env = np.load(env_path, mmap_mode="r") if env_path.exists() else np.empty((0,))
    return recording_id, str(tuple(eeg.shape)), str(tuple(env.shape))


def checkpoint_path(config: dict[str, Any], dataset: str, model: str, seed: int, stage: str) -> str:
    split_id = config["split_id"]
    return f"{config['checkpoint_dir']}/{stage}/{dataset}/{model}/{split_id}/seed{seed}/best_model.pt"


def build_preflight_rows(config: dict[str, Any], stage: str, jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset in config["datasets"]:
        split_check = validate_split(load_split(config, dataset), dataset)
        split_rows.append(split_check)
        if not split_check["no_overlap"]:
            failures.append(f"{dataset} split overlap check failed")
    for job in jobs:
        dataset = job["dataset"]
        model = job["model"]
        split = load_split(config, dataset)
        dataset_dir = resolve_dataset_path(config["datasets"][dataset]["dataset_locator"])
        first_id, eeg_shape, target_shape = first_recording_shape(dataset_dir, "train", split["train_subjects"][0])
        if not first_id:
            failures.append(f"{dataset}:{model} first train recording missing")
        zero_shot_status = "deferred_pending_linear_validation_fix_scope" if stage == "zero_shot" and model in LINEAR_HELD else "ready_for_manual_execution_after_smoke"
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "stage": stage,
                "split_id": split["split_id"],
                "train_subjects": "|".join(split["train_subjects"]),
                "val_subjects": "|".join(split["val_subjects"]),
                "test_subjects": "|".join(split["test_subjects"]),
                "train_val_intersection": "|".join(sorted(set(split["train_subjects"]) & set(split["val_subjects"]))),
                "train_test_intersection": "|".join(sorted(set(split["train_subjects"]) & set(split["test_subjects"]))),
                "val_test_intersection": "|".join(sorted(set(split["val_subjects"]) & set(split["test_subjects"]))),
                "first_train_recording": first_id,
                "first_train_eeg_shape": eeg_shape,
                "first_train_target_shape": target_shape,
                "input_contract": config["contracts"][model]["input_contract"],
                "scorer": "recording-level Pearson over full test-subject recordings",
                "coverage_check": "preflight shape only; smoke will check train batch, val forward, test aggregation",
                "checkpoint_path": checkpoint_path(config, dataset, model, int(config["seed"]), stage),
                "zero_shot_status": zero_shot_status,
            }
        )
    return rows, split_rows, failures


def write_reports(config: dict[str, Any], stage: str, jobs: list[dict[str, Any]], pending: list[dict[str, Any]], preflight_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]], failures: list[str], device: str) -> None:
    out_dir = output_dir(config, stage)
    write_csv(out_dir / f"{stage}_job_plan.csv", jobs, SMOKE_FIELDS)
    write_csv(out_dir / f"{stage}_execution_plan.csv", pending, SMOKE_FIELDS)
    write_csv(out_dir / "split_preflight.csv", split_rows, ["dataset", "split_id", "train_subjects", "val_subjects", "test_subjects", "train_val_intersection", "train_test_intersection", "val_test_intersection", "no_overlap"])
    write_csv(out_dir / "shape_checkpoint_preflight.csv", preflight_rows, PREFLIGHT_FIELDS)
    linear_deferred = [job for job in jobs if stage == "zero_shot" and job["model"] in LINEAR_HELD]
    schema = {
        "passed": not failures and len(jobs) == 22,
        "stage": stage,
        "planned_job_count": len(jobs),
        "pending_execution_count": len(pending),
        "expected_full_stage_job_count": 22,
        "split_no_overlap": all(bool(row["no_overlap"]) for row in split_rows),
        "dnn_excluded_as_fcnn_alias": True,
        "linear_zero_shot_deferred_count": len(linear_deferred),
        "checkpoint_paths_local_only": True,
        "metrics_written": False,
        "training_started": False,
        "device": device,
        "failures": failures,
    }
    write_json(out_dir / "schema_validation_report.json", schema)
    write_json(out_dir / "failure_report.json", {"failures": [{"failure_reason": item} for item in failures]})
    write_json(
        out_dir / "run_state.json",
        {
            "status": "preflight_only",
            "stage": stage,
            "planned_job_count": len(jobs),
            "completed_job_count": 0,
            "pending_job_count": len(pending),
            "training_started": False,
            "metrics_written": False,
        },
    )
    lines = [
        f"# Within-Dataset Fixed Holdout Modelset {stage} Preflight",
        "",
        "No smoke or zero-shot training was executed.",
        "",
        f"- stage: `{stage}`",
        f"- planned jobs: `{len(jobs)}`",
        f"- pending jobs after resume/max-jobs filtering: `{len(pending)}`",
        "- DNN excluded: `functional_alias_of=fcnn`",
        f"- linear-family zero-shot deferred jobs: `{len(linear_deferred)}`",
        "",
        "## Split Subjects",
    ]
    for row in split_rows:
        lines.append(f"- `{row['dataset']}` train=`{','.join(row['train_subjects'])}` val=`{','.join(row['val_subjects'])}` test=`{','.join(row['test_subjects'])}`")
    lines += [
        "",
        "## Manual Commands",
        "```powershell",
        "cd E:\\decode\\_fix_fixed_split_pooled20_modelset_v1",
        "F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py --config configs\\benchmark\\gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json --stage smoke --device auto --dry-run-plan",
        "```",
    ]
    (out_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config))
    device = resolve_device(args.device)
    dataset_filter = parse_filter(args.datasets)
    model_filter = parse_filter(args.models)
    jobs = build_jobs(config, args.stage, dataset_filter, model_filter)
    out_dir = output_dir(config, args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.startup_only:
        print(f"[STARTUP] config={args.config}", flush=True)
        print(f"[STARTUP] stage={args.stage} datasets={','.join(config['datasets'].keys())} models={len(config['models'])} device={device}", flush=True)
        print(f"[STARTUP] output_dir={out_dir.relative_to(ROOT).as_posix()} planned_jobs={len(jobs)}", flush=True)
        return 0
    completed = completed_keys(out_dir) if args.resume else set()
    pending = [job for job in jobs if job["job_key"] not in completed]
    if args.max_jobs is not None:
        pending = pending[: int(args.max_jobs)]
    preflight_rows, split_rows, failures = build_preflight_rows(config, args.stage, jobs)
    write_reports(config, args.stage, jobs, pending, preflight_rows, split_rows, failures, device)
    if args.dry_run_plan:
        print(f"[DRY RUN] stage={args.stage} planned_jobs={len(jobs)} completed={len(completed)} pending_execution={len(pending)}", flush=True)
        for job in pending:
            print(f"[DRY RUN] execution={job['job_key']}", flush=True)
        if args.stage == "zero_shot":
            held = [job["job_key"] for job in pending if job["model"] in LINEAR_HELD]
            print(f"[DRY RUN] zero_shot_deferred_linear_jobs={len(held)}", flush=True)
            for key in held:
                print(f"[DRY RUN] deferred={key}", flush=True)
    print(f"[PREFLIGHT] output_dir={out_dir.relative_to(ROOT).as_posix()} passed={not failures and len(jobs) == 22}", flush=True)
    print("[PREFLIGHT] no_training_executed=true metrics_written=false", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
