from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS
from repro.reference_baselines import list_reference_subjects
from run_gate0_gate2_full_subject_single_seed import (
    ensure_dir,
    repo_relative,
    resolve_dataset_path,
    run_job,
    summarize_dataset_metrics,
    summarize_etard_condition_metrics,
    write_generic_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_job_ledger_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "subject_id",
        "model",
        "seed",
        "action",
        "status",
        "job_dir",
        "checkpoint_id",
        "subject_metric",
        "num_recordings",
        "error",
        "source_protocol",
    ]
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_subjects(dataset_cfg: dict) -> list[str]:
    explicit = list(dataset_cfg.get("subject_ids", []))
    if explicit:
        return explicit
    dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
    return list_reference_subjects(dataset_dir, split="test")


def single_seed_job_dir(config: dict, dataset_id: str, subject_id: str, model: str, seed: int) -> Path:
    return ROOT / config["single_seed_job_root"] / dataset_id / subject_id / model / f"seed_{seed}"


def single_seed_job_manifest(config: dict, dataset_id: str, subject_id: str, model: str, seed: int) -> Path:
    return single_seed_job_dir(config, dataset_id, subject_id, model, seed) / "job_manifest.json"


def single_seed_job_exists(config: dict, dataset_id: str, subject_id: str, model: str, seed: int) -> bool:
    manifest = single_seed_job_manifest(config, dataset_id, subject_id, model, seed)
    if not manifest.exists():
        return False
    payload = read_json(manifest)
    return payload.get("status") == "success"


def load_job_artifacts(config: dict, dataset_id: str, subject_id: str, model: str, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    job_dir = single_seed_job_dir(config, dataset_id, subject_id, model, seed)
    recording_rows = read_csv_rows(job_dir / "recording_metrics.csv")
    subject_rows = read_csv_rows(job_dir / "subject_metrics.csv")
    manifest = read_json(job_dir / "job_manifest.json")
    return recording_rows, subject_rows, manifest


def build_single_seed_subconfig(config: dict, dataset_cfg: dict, seed: int) -> dict:
    subconfig = {
        "protocol": config["base_single_seed_protocol"],
        "artifact_scope": "full_subject_single_seed_validation",
        "config_fingerprint_version": config["config_fingerprint_version"],
        "scorer_id": config["scorer_id"],
        "aggregation_rule": "subject_metric=mean_recording_pearson_r; dataset_metric=mean_subject_metric",
        "seed": seed,
        "models": config["models"],
        "ridge": config["ridge"],
        "cca": config["cca"],
        "fcnn": config["fcnn"],
        "adt": config["adt"],
    }
    return subconfig


def subject_metric_lookup(subject_rows: list[dict[str, str]]) -> dict[tuple[str, str, int, str], float]:
    lookup = {}
    for row in subject_rows:
        key = (row["dataset"], row["model"], int(row["seed"]), row["subject_id"])
        lookup[key] = float(row["metric_value"])
    return lookup


def dataset_metrics_with_seed(subject_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in subject_rows:
        if row["metric_name"] != "mean_recording_pearson_r":
            continue
        grouped[(row["dataset"], row["model"], int(row["seed"]))].append(float(row["metric_value"]))

    output = []
    for (dataset, model, seed), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        clipped = np.clip(arr, -0.999999, 0.999999)
        fisher_z = np.arctanh(clipped)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "n_subjects": int(arr.size),
                "mean_pearson": float(np.mean(arr)),
                "std_pearson": float(np.std(arr, ddof=0)),
                "median_pearson": float(np.median(arr)),
                "min_pearson": float(np.min(arr)),
                "max_pearson": float(np.max(arr)),
                "fisher_z_mean": float(np.mean(fisher_z)),
                "backtransformed_mean_r": float(np.tanh(np.mean(fisher_z))),
            }
        )
    return output


def dataset_metrics_seed_average(dataset_seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in dataset_seed_rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)

    output = []
    for (dataset, model), rows in sorted(grouped.items()):
        mean_values = np.asarray([float(row["mean_pearson"]) for row in rows], dtype=np.float64)
        fisher_values = np.asarray([float(row["fisher_z_mean"]) for row in rows], dtype=np.float64)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "n_seeds": len(rows),
                "seed_list": ",".join(str(int(row["seed"])) for row in rows),
                "mean_of_seed_means": float(np.mean(mean_values)),
                "std_of_seed_means": float(np.std(mean_values, ddof=0)),
                "min_seed_mean": float(np.min(mean_values)),
                "max_seed_mean": float(np.max(mean_values)),
                "mean_of_fisher_z_means": float(np.mean(fisher_values)),
                "backtransformed_mean_r": float(np.tanh(np.mean(fisher_values))),
            }
        )
    return output


def etard_condition_metrics_with_seed(recording_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in recording_rows:
        if row["dataset"] != "etard_tf64":
            continue
        recording_id = row["recording_id"]
        tail = recording_id.split("_-_")[2]
        if "_part_" not in tail:
            continue
        condition = tail.split("_part_", 1)[0]
        if condition not in {"clean", "fM", "fW", "hb", "lb", "mb"}:
            continue
        grouped[(row["model"], int(row["seed"]), condition)].append(row)

    output = []
    for (model, seed, condition), rows in sorted(grouped.items()):
        values = np.asarray([float(item["metric_value"]) for item in rows], dtype=np.float64)
        output.append(
            {
                "dataset": "etard_tf64",
                "model": model,
                "seed": seed,
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


def etard_condition_seed_average(condition_seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in condition_seed_rows:
        grouped[(str(row["model"]), str(row["condition"]))].append(row)

    output = []
    for (model, condition), rows in sorted(grouped.items()):
        mean_values = np.asarray([float(row["mean_pearson"]) for row in rows], dtype=np.float64)
        median_values = np.asarray([float(row["median_pearson"]) for row in rows], dtype=np.float64)
        std_values = np.asarray([float(row["std_pearson"]) for row in rows], dtype=np.float64)
        output.append(
            {
                "dataset": "etard_tf64",
                "model": model,
                "condition": condition,
                "n_seeds": len(rows),
                "seed_list": ",".join(str(int(row["seed"])) for row in rows),
                "mean_pearson_across_seeds": float(np.mean(mean_values)),
                "std_pearson_across_seeds": float(np.std(mean_values, ddof=0)),
                "mean_median_pearson": float(np.mean(median_values)),
                "mean_within_seed_std": float(np.mean(std_values)),
                "mean_n_subjects": float(np.mean([float(row["n_subjects"]) for row in rows])),
                "mean_n_recordings": float(np.mean([float(row["n_recordings"]) for row in rows])),
                "mean_negative_recording_count": float(np.mean([float(row["negative_recording_count"]) for row in rows])),
            }
        )
    return output


def write_result_summary(
    path: Path,
    *,
    config: dict,
    device: str,
    dataset_subjects: dict[str, list[str]],
    total_jobs: int,
    successful_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    dataset_seed_rows: list[dict[str, object]],
    dataset_seed_averages: list[dict[str, object]],
) -> None:
    lines = [
        "# Full-Subject Multi-Seed v1",
        "",
        "This directory extends the full-subject unified benchmark to three seeds while reusing the existing seed 0 jobs.",
        "",
        "## Protocol",
        f"- `protocol`: `{config['protocol']}`",
        f"- `device`: `{device}`",
        f"- `seeds`: `{', '.join(str(seed) for seed in config['seeds'])}`",
        f"- `models`: `{', '.join(config['models'])}`",
        "",
        "## Subject inventory",
    ]
    for dataset_id, subjects in sorted(dataset_subjects.items()):
        lines.append(f"- `{dataset_id}` ({len(subjects)} subjects): `{', '.join(subjects)}`")
    lines.extend(
        [
            "",
            "## Job accounting",
            f"- total jobs across all requested seeds: `{total_jobs}`",
            f"- successful jobs available: `{successful_jobs}`",
            f"- failed jobs: `{failed_jobs}`",
            f"- skipped/reused jobs: `{skipped_jobs}`",
            "",
            "## Dataset metrics by seed",
            "| dataset | model | seed | n_subjects | mean_pearson | std_pearson | median_pearson | min_pearson | max_pearson | fisher_z_mean | backtransformed_mean_r |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_seed_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['seed']} | {row['n_subjects']} | "
            f"{row['mean_pearson']:.6f} | {row['std_pearson']:.6f} | {row['median_pearson']:.6f} | "
            f"{row['min_pearson']:.6f} | {row['max_pearson']:.6f} | {row['fisher_z_mean']:.6f} | "
            f"{row['backtransformed_mean_r']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset mean ± std across seeds",
            "| dataset | model | n_seeds | seed_list | mean_of_seed_means | std_of_seed_means | min_seed_mean | max_seed_mean | mean_of_fisher_z_means | backtransformed_mean_r |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_seed_averages:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_seeds']} | {row['seed_list']} | "
            f"{row['mean_of_seed_means']:.6f} | {row['std_of_seed_means']:.6f} | "
            f"{row['min_seed_mean']:.6f} | {row['max_seed_mean']:.6f} | "
            f"{row['mean_of_fisher_z_means']:.6f} | {row['backtransformed_mean_r']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dataset_subjects = {dataset_cfg["dataset_id"]: infer_subjects(dataset_cfg) for dataset_cfg in config["datasets"]}
    dataset_cfg_lookup = {dataset_cfg["dataset_id"]: dataset_cfg for dataset_cfg in config["datasets"]}
    total_jobs = sum(len(subjects) * len(config["models"]) * len(config["seeds"]) for subjects in dataset_subjects.values())

    job_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    skipped_jobs = 0

    for dataset_id, subjects in dataset_subjects.items():
        dataset_cfg = dataset_cfg_lookup[dataset_id]
        dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
        if not dataset_dir.exists():
            failures.append(
                {
                    "dataset": dataset_id,
                    "subject_id": "",
                    "model": "",
                    "seed": "",
                    "status": "missing_dataset_directory",
                    "expected_locator": dataset_cfg["dataset_locator"],
                }
            )
            continue

        for seed in config["seeds"]:
            for subject_id in subjects:
                for model in config["models"]:
                    if single_seed_job_exists(config, dataset_id, subject_id, model, int(seed)):
                        _, subject_rows, manifest = load_job_artifacts(config, dataset_id, subject_id, model, int(seed))
                        metric_value = float(subject_rows[0]["metric_value"])
                        num_recordings = int(subject_rows[0]["num_recordings"])
                        job_rows.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model,
                                "seed": int(seed),
                                "action": "reuse_existing" if int(seed) == 0 else "skip_existing",
                                "status": "success",
                                "job_dir": repo_relative(single_seed_job_dir(config, dataset_id, subject_id, model, int(seed))),
                                "checkpoint_id": manifest.get("checkpoint_id", ""),
                                "subject_metric": metric_value,
                                "num_recordings": num_recordings,
                                "error": "",
                                "source_protocol": manifest.get("protocol", config["base_single_seed_protocol"]),
                            }
                        )
                        skipped_jobs += 1
                        continue

                    subconfig = build_single_seed_subconfig(config, dataset_cfg, int(seed))
                    try:
                        recording_rows, subject_rows, meta = run_job(
                            config=subconfig,
                            dataset_cfg=dataset_cfg,
                            dataset_dir=dataset_dir,
                            subject_id=subject_id,
                            model_name=model,
                            device=device,
                            output_dir=ROOT / "experiments" / "gate0_gate2_full_subject_single_seed",
                        )
                        job_dir = single_seed_job_dir(config, dataset_id, subject_id, model, int(seed))
                        ensure_dir(job_dir)
                        write_generic_csv(job_dir / "recording_metrics.csv", [row.__dict__ if hasattr(row, "__dict__") else row for row in recording_rows], RECORDING_METRIC_FIELDS)
                        write_generic_csv(job_dir / "subject_metrics.csv", [row.__dict__ if hasattr(row, "__dict__") else row for row in subject_rows], SUBJECT_METRIC_FIELDS)
                        metric_value = float(subject_rows[0].metric_value)
                        num_recordings = int(subject_rows[0].num_recordings)
                        write_json(
                            job_dir / "job_manifest.json",
                            {
                                "dataset": dataset_id,
                                "dataset_locator": dataset_cfg["dataset_locator"],
                                "subject_id": subject_id,
                                "model": model,
                                "seed": int(seed),
                                "protocol": config["base_single_seed_protocol"],
                                "artifact_scope": "full_subject_single_seed_validation",
                                "checkpoint_id": meta.get("checkpoint_id"),
                                "subject_metric": metric_value,
                                "num_recordings": num_recordings,
                                "status": "success",
                                "meta": meta,
                                "scorer_id": config["scorer_id"],
                                "aggregation_rule": "subject_metric=mean_recording_pearson_r; dataset_metric=mean_subject_metric",
                                "artifacts": {
                                    "recording_metrics": repo_relative(job_dir / "recording_metrics.csv"),
                                    "subject_metrics": repo_relative(job_dir / "subject_metrics.csv")
                                }
                            },
                        )
                        job_rows.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model,
                                "seed": int(seed),
                                "action": "run_missing_seed",
                                "status": "success",
                                "job_dir": repo_relative(job_dir),
                                "checkpoint_id": meta.get("checkpoint_id", ""),
                                "subject_metric": metric_value,
                                "num_recordings": num_recordings,
                                "error": "",
                                "source_protocol": config["base_single_seed_protocol"],
                            }
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model,
                                "seed": int(seed),
                                "status": "failed",
                                "error": repr(exc),
                                "recoverable": True,
                            }
                        )
                        job_rows.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model,
                                "seed": int(seed),
                                "action": "run_missing_seed",
                                "status": "failed",
                                "job_dir": repo_relative(single_seed_job_dir(config, dataset_id, subject_id, model, int(seed))),
                                "checkpoint_id": "",
                                "subject_metric": "",
                                "num_recordings": "",
                                "error": repr(exc),
                                "source_protocol": config["base_single_seed_protocol"],
                            }
                        )

    all_recording_rows: list[dict[str, str]] = []
    all_subject_rows: list[dict[str, str]] = []
    for row in job_rows:
        if row["status"] != "success":
            continue
        recording_rows, subject_rows, _ = load_job_artifacts(config, str(row["dataset"]), str(row["subject_id"]), str(row["model"]), int(row["seed"]))
        all_recording_rows.extend(recording_rows)
        all_subject_rows.extend(subject_rows)

    write_generic_csv(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_generic_csv(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)

    dataset_seed_rows = dataset_metrics_with_seed(all_subject_rows)
    write_generic_csv(
        output_dir / "dataset_metrics_by_seed.csv",
        dataset_seed_rows,
        [
            "dataset",
            "model",
            "seed",
            "n_subjects",
            "mean_pearson",
            "std_pearson",
            "median_pearson",
            "min_pearson",
            "max_pearson",
            "fisher_z_mean",
            "backtransformed_mean_r",
        ],
    )
    dataset_seed_averages = dataset_metrics_seed_average(dataset_seed_rows)
    write_generic_csv(
        output_dir / "dataset_metrics_across_seeds.csv",
        dataset_seed_averages,
        [
            "dataset",
            "model",
            "n_seeds",
            "seed_list",
            "mean_of_seed_means",
            "std_of_seed_means",
            "min_seed_mean",
            "max_seed_mean",
            "mean_of_fisher_z_means",
            "backtransformed_mean_r",
        ],
    )

    condition_seed_rows = etard_condition_metrics_with_seed(all_recording_rows)
    write_generic_csv(
        output_dir / "etard_condition_metrics_by_seed.csv",
        condition_seed_rows,
        [
            "dataset",
            "model",
            "seed",
            "condition",
            "n_subjects",
            "n_recordings",
            "mean_pearson",
            "median_pearson",
            "std_pearson",
            "negative_recording_count",
        ],
    )
    condition_seed_averages = etard_condition_seed_average(condition_seed_rows)
    write_generic_csv(
        output_dir / "etard_condition_metrics_across_seeds.csv",
        condition_seed_averages,
        [
            "dataset",
            "model",
            "condition",
            "n_seeds",
            "seed_list",
            "mean_pearson_across_seeds",
            "std_pearson_across_seeds",
            "mean_median_pearson",
            "mean_within_seed_std",
            "mean_n_subjects",
            "mean_n_recordings",
            "mean_negative_recording_count",
        ],
    )

    write_job_ledger_csv(output_dir / "job_ledger.csv", job_rows)
    write_json(
        output_dir / "job_ledger.json",
        {
            "protocol": config["protocol"],
            "device": device,
            "rows": job_rows,
        },
    )

    successful_jobs = len([row for row in job_rows if row["status"] == "success"])
    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "artifact_scope": config["artifact_scope"],
            "previous_fix_branch": config["previous_fix_branch"],
            "previous_fix_commit": config["previous_fix_commit"],
            "current_branch": config["current_branch"],
            "device": device,
            "seeds": config["seeds"],
            "datasets_requested": [item["dataset_id"] for item in config["datasets"]],
            "models_requested": list(config["models"]),
            "dataset_subjects": dataset_subjects,
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "failed_jobs": len(failures),
            "skipped_jobs": skipped_jobs,
            "failures": failures,
            "artifacts": {
                "job_ledger_csv": repo_relative(output_dir / "job_ledger.csv"),
                "job_ledger_json": repo_relative(output_dir / "job_ledger.json"),
                "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
                "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
                "dataset_metrics_by_seed": repo_relative(output_dir / "dataset_metrics_by_seed.csv"),
                "dataset_metrics_across_seeds": repo_relative(output_dir / "dataset_metrics_across_seeds.csv"),
                "etard_condition_metrics_by_seed": repo_relative(output_dir / "etard_condition_metrics_by_seed.csv"),
                "etard_condition_metrics_across_seeds": repo_relative(output_dir / "etard_condition_metrics_across_seeds.csv"),
                "failure_report": repo_relative(output_dir / "failure_report.json"),
                "result_summary": repo_relative(output_dir / "result_summary.md"),
                "incremental_comparison": repo_relative(output_dir / "incremental_comparison.md")
            }
        },
    )
    write_json(output_dir / "failure_report.json", {"failures": failures})

    write_result_summary(
        output_dir / "result_summary.md",
        config=config,
        device=device,
        dataset_subjects=dataset_subjects,
        total_jobs=total_jobs,
        successful_jobs=successful_jobs,
        failed_jobs=len(failures),
        skipped_jobs=skipped_jobs,
        dataset_seed_rows=dataset_seed_rows,
        dataset_seed_averages=dataset_seed_averages,
    )

    incremental_lines = [
        "# Incremental Comparison",
        "",
        f"- previous branch: `{config['previous_fix_branch']}`",
        f"- previous commit: `{config['previous_fix_commit']}`",
        f"- current branch: `{config['current_branch']}`",
        "- this round extends the full-subject benchmark from single-seed to multi-seed v1.",
        "- seed 0 was reused from the existing single-seed run.",
        "- only missing seed jobs for 42 and 2026 were executed.",
        "",
        "## New in this round",
        "- `configs/benchmark/gate0_gate2_full_subject_multi_seed_v1.json`",
        "- `scripts/run_gate0_gate2_full_subject_multi_seed_v1.py`",
        "- seed-specific dataset metrics and seed-averaged summaries",
        "- Etard condition-level seed-averaged summary",
        "",
        "## Historical results not rerun",
        "- `experiments/summary_figures/*`",
        "- pilot / focused rerun diagnostic bundles",
        "- specialized models outside `ridge / cca / fcnn / adt`",
    ]
    (output_dir / "incremental_comparison.md").write_text("\n".join(incremental_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
