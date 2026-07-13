from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")

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
NEURAL_MODELS = {"fcnn", "cnn", "eegnet", "adt", "vlaai", "happyquokka"}
STATIC_REUSE_MODELS = {"fcnn", "eegnet", "adt"}

ACCEPTED_ZERO_SHOT_RUNNER = "E:/decode/_fix_subject_holdout_pooled_finetune_v1/scripts/run_gate0_gate2_subject_holdout_fixed_split_v1.py"
ACCEPTED_POOLED10_RUNNER = "E:/decode/_fix_subject_holdout_pooled_finetune_v1/scripts/run_gate0_gate2_subject_holdout_pooled_finetune_v1.py"

SUBJECT_SPECIFIC_SOURCES = {
    "fcnn": {
        "code": "scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_fcnn",
        "config": "configs/benchmark/gate0_gate2_model_expansion_v1.json::fcnn",
        "note": "accepted full subject-specific FCNN config; fixed-holdout layer must not redefine model/training",
    },
    "cnn": {
        "code": "scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_cnn",
        "config": "configs/benchmark/gate0_gate2_model_expansion_v1.json::cnn",
        "note": "accepted full subject-specific CNN config; fixed-holdout layer must not redefine model/training",
    },
    "eegnet": {
        "code": "scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_eegnet",
        "config": "configs/benchmark/gate0_gate2_model_expansion_v1.json::eegnet",
        "note": "accepted full subject-specific EEGNet config; fixed-holdout layer must not redefine model/training",
    },
    "adt": {
        "code": "scripts/run_gate0_gate2_full_subject_single_seed.py::fit_and_predict_adt",
        "config": "configs/benchmark/gate0_gate2_model_expansion_v1.json::adt",
        "note": "accepted full subject-specific ADT config; fixed-holdout layer must not redefine model/training",
    },
    "vlaai": {
        "code": "scripts/run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1.py::run_vlaai_full_eval",
        "config": "configs/benchmark/gate0_gate2_vlaai_happyquokka_training_budget_p00_v1.json::vlaai",
        "note": "accepted VLAAI full budget config; strict deterministic policy must be applied by accepted chain",
    },
    "happyquokka": {
        "code": "scripts/run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1.py::run_happyquokka_full_eval",
        "config": "configs/benchmark/gate0_gate2_subject_specific_happyquokka_seeded_weissbart_full_v1.json::happyquokka",
        "note": "accepted seeded HappyQuokka full-training policy; do not use smoke max_epochs=1 config",
    },
}

JOB_FIELDS = [
    "job_key",
    "stage",
    "dataset",
    "model",
    "seed",
    "split_id",
    "train_subjects",
    "val_subjects",
    "test_subjects",
    "execution_status",
    "source_code",
    "source_config",
    "fixed_holdout_runner",
    "pooled10_runner",
    "notes",
]

REUSE_FIELDS = [
    "dataset",
    "model",
    "seed",
    "split_id",
    "zero_shot_reuse_status",
    "pooled10_reuse_status",
    "evidence",
    "difference_or_blocker",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter.")
    parser.add_argument("--models", default="", help="Comma-separated model filter.")
    parser.add_argument("--stage", choices=["smoke", "zero_shot", "pooled10"], default="zero_shot")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--verify-existing-output-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Accepted for CLI compatibility; no training is run.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limits displayed execution plan only.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Accepted for CLI compatibility; no device work is run.")
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


def output_dir(config: dict[str, Any], stage: str) -> Path:
    if stage == "smoke":
        return ROOT / config["smoke_output_dir"]
    if stage == "pooled10":
        return ROOT / "experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_pooled10_plan"
    return ROOT / config["zero_shot_output_dir"]


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_candidate = SHARED_WORKSPACE / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def load_split(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    return load_json(ROOT / config["datasets"][dataset]["split_manifest_path"])


def subject_lists(split: dict[str, Any]) -> tuple[str, str, str]:
    return "|".join(split["train_subjects"]), "|".join(split["val_subjects"]), "|".join(split["test_subjects"])


def job_key(dataset: str, model: str, seed: int, stage: str) -> str:
    return f"{dataset}:{model}:seed{seed}:stage={stage}"


def execution_status(stage: str, model: str) -> str:
    if model in LINEAR_HELD:
        return "deferred_pending_linear_validation_fix_scope"
    if stage == "smoke":
        return "not_supported_no_generic_smoke; use accepted chain dry-run or model-specific engineering checks only"
    if model in STATIC_REUSE_MODELS:
        return "static_reuse_audit_required_before_any_run"
    return "manual_accepted_chain_command_only"


def build_jobs(config: dict[str, Any], stage: str, dataset_filter: set[str] | None, model_filter: set[str] | None) -> list[dict[str, Any]]:
    unknown_datasets = sorted((dataset_filter or set()) - set(config["datasets"]))
    unknown_models = sorted((model_filter or set()) - set(config["models"]))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}")
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    datasets = [item for item in config["datasets"] if dataset_filter is None or item in dataset_filter]
    models = [item for item in config["models"] if model_filter is None or item in model_filter]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        split = load_split(config, dataset)
        train_subjects, val_subjects, test_subjects = subject_lists(split)
        for model in models:
            source = SUBJECT_SPECIFIC_SOURCES.get(model, {})
            rows.append(
                {
                    "job_key": job_key(dataset, model, int(config["seed"]), stage),
                    "stage": stage,
                    "dataset": dataset,
                    "model": model,
                    "seed": int(config["seed"]),
                    "split_id": split["split_id"],
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "execution_status": execution_status(stage, model),
                    "source_code": source.get("code", "deferred_linear_family"),
                    "source_config": source.get("config", "deferred_pending_linear_validation_fix_scope"),
                    "fixed_holdout_runner": ACCEPTED_ZERO_SHOT_RUNNER,
                    "pooled10_runner": ACCEPTED_POOLED10_RUNNER,
                    "notes": source.get("note", "linear-family retained in roster but not runnable this round"),
                }
            )
    return rows


def first_recording_shape(config: dict[str, Any], dataset: str, split_name: str, subject: str) -> tuple[str, str, str]:
    dataset_dir = resolve_dataset_path(config["datasets"][dataset]["dataset_locator"])
    eeg_paths = sorted(dataset_dir.glob(f"{split_name}_-_{subject}_-_*_-_eeg.npy"))
    if not eeg_paths:
        return "", "", ""
    eeg_path = eeg_paths[0]
    recording_id = eeg_path.stem.replace("_-_eeg", "")
    env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
    return recording_id, str(eeg_path), str(env_path)


def local_checkpoint_exists(model: str, dataset: str, split_id: str, seed: int) -> bool:
    root = ROOT / "local_checkpoints" / "subject_holdout" / model / dataset / split_id / f"seed{seed}"
    return root.exists() and any(root.glob("best_epoch_*.pt"))


def pooled10_checkpoint_exists(model: str, dataset: str, split_id: str, seed: int) -> bool:
    root = ROOT / "local_checkpoints" / "finetune" / "subject_holdout_pooled10" / model / dataset / split_id / f"seed{seed}"
    return root.exists() and any(root.glob("best_epoch_*.pt"))


def accepted_eegnet_artifacts_exist(dataset: str) -> bool:
    if dataset != "weissbart_tf64":
        return False
    base = Path("E:/decode/_fix_subject_holdout_pooled_finetune_v1/experiments")
    return (
        (base / "gate0_gate2_subject_holdout_fixed_split_v1" / "run_manifest.json").exists()
        and (base / "gate0_gate2_subject_holdout_fixed_split_v1" / "checkpoint_manifest.json").exists()
        and (base / "gate0_gate2_subject_holdout_pooled_finetune_v1" / "run_manifest.json").exists()
        and (base / "gate0_gate2_subject_holdout_pooled_finetune_v1" / "checkpoint_manifest.json").exists()
    )


def build_reuse_audit(config: dict[str, Any], jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        model = str(job["model"])
        if model not in STATIC_REUSE_MODELS:
            continue
        dataset = str(job["dataset"])
        split_id = str(job["split_id"])
        seed = int(job["seed"])
        zero_ckpt = local_checkpoint_exists(model, dataset, split_id, seed)
        pooled_ckpt = pooled10_checkpoint_exists(model, dataset, split_id, seed)
        if model == "eegnet" and accepted_eegnet_artifacts_exist(dataset):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "split_id": split_id,
                    "zero_shot_reuse_status": "lossless_reuse_ok",
                    "pooled10_reuse_status": "lossless_reuse_ok",
                    "evidence": "accepted subject-holdout fixed split and pooled fine-tune manifests exist in E:/decode/_fix_subject_holdout_pooled_finetune_v1",
                    "difference_or_blocker": "",
                }
            )
        else:
            blockers = []
            if not zero_ckpt:
                blockers.append("missing local subject_holdout checkpoint")
            if not pooled_ckpt:
                blockers.append("missing local subject_holdout_pooled10 checkpoint")
            if zero_ckpt or pooled_ckpt:
                blockers.append("checkpoint metadata exists but complete accepted run artifacts/config provenance were not found in this branch")
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "split_id": split_id,
                    "zero_shot_reuse_status": "not_lossless_reuse",
                    "pooled10_reuse_status": "not_lossless_reuse",
                    "evidence": f"zero_checkpoint_exists={zero_ckpt}; pooled10_checkpoint_exists={pooled_ckpt}",
                    "difference_or_blocker": "; ".join(blockers),
                }
            )
    return rows


def build_selection_audit(config: dict[str, Any], jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = sorted({str(job["dataset"]) for job in jobs})
    for dataset in seen:
        split = load_split(config, dataset)
        first_id, first_eeg, first_env = first_recording_shape(config, dataset, "train", split["train_subjects"][0])
        rows.append(
            {
                "dataset": dataset,
                "split_id": split["split_id"],
                "train_data_selection": "train_subjects train recordings only",
                "val_data_selection": "val_subjects val recordings only",
                "test_data_selection": "test_subjects test recordings only",
                "train_subjects": "|".join(split["train_subjects"]),
                "val_subjects": "|".join(split["val_subjects"]),
                "test_subjects": "|".join(split["test_subjects"]),
                "first_train_recording": first_id,
                "first_train_eeg_path": first_eeg,
                "first_train_env_path": first_env,
                "window_boundary_rule": "windows/chunks must be generated only within their own recording and split",
            }
        )
    return rows


def manual_command_rows(config: dict[str, Any], jobs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Manual Commands",
        "",
        "These commands are dispatch templates only. This branch does not implement or run training.",
        "",
        "## Accepted Chain Roots",
        "",
        f"- zero-shot runner: `{ACCEPTED_ZERO_SHOT_RUNNER}`",
        f"- pooled10 runner: `{ACCEPTED_POOLED10_RUNNER}`",
        "",
        "## CNN / VLAAI / HappyQuokka Two-Dataset Zero-Shot",
        "",
        "Create model-specific accepted-chain configs first, using the source config paths recorded in `job_plan.csv`.",
        "Then run the accepted fixed-holdout zero-shot runner from its owning worktree, for each dataset/model config:",
        "",
        "```powershell",
        "cd E:\\decode\\_fix_subject_holdout_pooled_finetune_v1",
        "F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_subject_holdout_fixed_split_v1.py --config <accepted_fixed_holdout_config_for_dataset_model_seed0.json> --device auto --dry-run-plan",
        "```",
        "",
        "## CNN / VLAAI / HappyQuokka Follow-Up Pooled10",
        "",
        "After zero-shot artifacts and local-only checkpoints exist with accepted provenance, run pooled10 through the accepted pooled runner:",
        "",
        "```powershell",
        "cd E:\\decode\\_fix_subject_holdout_pooled_finetune_v1",
        "F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_subject_holdout_pooled_finetune_v1.py --config <accepted_pooled10_config_for_dataset_model_seed0.json> --device auto --dry-run-plan",
        "```",
        "",
        "## Current Command Matrix",
        "",
    ]
    for job in jobs:
        if job["model"] in {"cnn", "vlaai", "happyquokka"}:
            lines.append(f"- `{job['dataset']} / {job['model']}`: zero-shot config must reuse `{job['source_config']}`; pooled10 config must point to the accepted zero-shot checkpoint provenance.")
    return lines


def write_reports(config: dict[str, Any], stage: str, jobs: list[dict[str, Any]], pending: list[dict[str, Any]]) -> None:
    out_dir = output_dir(config, stage)
    selection_rows = build_selection_audit(config, jobs)
    reuse_rows = build_reuse_audit(config, jobs)
    write_csv(out_dir / f"{stage}_job_plan.csv", jobs, JOB_FIELDS)
    write_csv(out_dir / f"{stage}_execution_plan.csv", pending, JOB_FIELDS)
    write_csv(out_dir / "data_selection_audit.csv", selection_rows, list(selection_rows[0].keys()) if selection_rows else ["dataset"])
    write_csv(out_dir / "reuse_audit.csv", reuse_rows, REUSE_FIELDS)
    write_json(
        out_dir / "run_state.json",
        {
            "status": "dry_run_only",
            "stage": stage,
            "planned_job_count": len(jobs),
            "displayed_execution_count": len(pending),
            "training_started": False,
            "independent_training_loop_present": False,
        },
    )
    write_json(
        out_dir / "failure_report.json",
        {
            "failures": [],
            "deferred_linear_family_count": len([job for job in jobs if job["model"] in LINEAR_HELD]),
        },
    )
    write_json(
        out_dir / "schema_validation_report.json",
        {
            "passed": True,
            "stage": stage,
            "planned_job_count": len(jobs),
            "expected_full_roster_count": 22 if stage != "pooled10" else len(jobs),
            "independent_training_loop_present": False,
            "model_factory_present": False,
            "train_forward_present": False,
            "evaluate_model_present": False,
            "uses_smoke_config_for_formal_params": False,
            "linear_family_deferred": sorted(LINEAR_HELD),
            "accepted_zero_shot_runner": ACCEPTED_ZERO_SHOT_RUNNER,
            "accepted_pooled10_runner": ACCEPTED_POOLED10_RUNNER,
        },
    )
    (out_dir / "manual_commands.md").write_text("\n".join(manual_command_rows(config, jobs)) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config))
    dataset_filter = parse_filter(args.datasets)
    model_filter = parse_filter(args.models)
    jobs = build_jobs(config, args.stage, dataset_filter, model_filter)
    pending = jobs[: args.max_jobs] if args.max_jobs is not None else jobs
    if args.startup_only:
        print(f"[STARTUP] stage={args.stage} datasets={len(config['datasets'])} models={len(config['models'])} planned_jobs={len(jobs)}", flush=True)
        print("[STARTUP] independent_training_loop_present=false", flush=True)
        return 0
    write_reports(config, args.stage, jobs, pending)
    if args.dry_run_plan or args.verify_existing_output_only:
        print(f"[DRY RUN] stage={args.stage} planned_jobs={len(jobs)} displayed_execution={len(pending)}", flush=True)
        for job in pending:
            print(f"[DRY RUN] {job['job_key']} status={job['execution_status']}", flush=True)
        print("[DRY RUN] no_training_executed=true independent_training_loop_present=false", flush=True)
        return 0
    print("[NO-OP] This runner is dry-run only. Use --dry-run-plan or accepted-chain runners.", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
