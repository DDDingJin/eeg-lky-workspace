from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS
from run_gate0_gate2_full_subject_multi_seed_v1 import (
    dataset_metrics_seed_average,
    dataset_metrics_with_seed,
    etard_condition_seed_average,
    etard_condition_metrics_with_seed,
    infer_subjects,
    load_job_artifacts,
    read_csv_rows,
    write_job_ledger_csv,
)
from run_gate0_gate2_full_subject_single_seed import (
    UpstreamCNN,
    UpstreamFCNN,
    ensure_dir,
    repo_relative,
    resolve_dataset_path,
    run_job,
    write_generic_csv,
    write_json,
)


EXPECTED_DATASETS = ["weissbart_tf64", "etard_tf64"]
EXPECTED_SEEDS = [0, 42, 2026]


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


def read_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_single_seed_subconfig(config: dict, dataset_cfg: dict, seed: int, effective_models: list[str]) -> dict:
    subconfig = {
        "protocol": config["base_single_seed_protocol"],
        "artifact_scope": "full_subject_single_seed_validation",
        "config_fingerprint_version": config["config_fingerprint_version"],
        "scorer_id": config["scorer_id"],
        "aggregation_rule": "subject_metric=mean_recording_pearson_r; dataset_metric=mean_subject_metric",
        "seed": seed,
        "models": effective_models,
    }
    for model_name in effective_models:
        subconfig[model_name] = config[model_name]
    return subconfig


def resolve_model_support(config: dict) -> tuple[list[str], list[dict[str, str]]]:
    effective_models = list(config["base_models_reused"])
    skipped: list[dict[str, str]] = []
    for model_name in config["expansion_models_requested"]:
        if model_name == "dnn" and UpstreamFCNN is None:
            skipped.append(
                {
                    "model": model_name,
                    "reason": "shared upstream FCNN implementation is unavailable in this worktree and shared workspace",
                }
            )
            continue
        if model_name == "cnn" and UpstreamCNN is None:
            skipped.append(
                {
                    "model": model_name,
                    "reason": "shared upstream CNN implementation is unavailable in this worktree and shared workspace",
                }
            )
            continue
        if model_name not in config:
            skipped.append(
                {
                    "model": model_name,
                    "reason": "model config is missing from gate0_gate2_model_expansion_v1.json",
                }
            )
            continue
        effective_models.append(model_name)
    return effective_models, skipped


def write_skipped_models(path: Path, skipped: list[dict[str, str]]) -> None:
    if not skipped:
        path.write_text("# Skipped Models\n\nNone.\n", encoding="utf-8")
        return
    lines = [
        "# Skipped Models",
        "",
        "These requested expansion models were intentionally not run in this round.",
        "",
    ]
    for item in skipped:
        lines.append(f"- `{item['model']}`: {item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_summary(
    path: Path,
    *,
    config: dict,
    device: str,
    dataset_subjects: dict[str, list[str]],
    effective_models: list[str],
    skipped_models: list[dict[str, str]],
    total_jobs: int,
    successful_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    dataset_seed_rows: list[dict[str, object]],
    dataset_seed_averages: list[dict[str, object]],
) -> None:
    lines = [
        "# Gate0-Gate2 Model Expansion v1",
        "",
        "This directory extends the full-subject multi-seed benchmark with additional baseline models under the same split, scorer, and aggregation schema.",
        "It is structured execution evidence and writing material rather than final paper prose.",
        "",
        "## Protocol",
        f"- `protocol`: `{config['protocol']}`",
        f"- `device`: `{device}`",
        f"- `seeds`: `{', '.join(str(seed) for seed in config['seeds'])}`",
        f"- reused base models: `{', '.join(config['base_models_reused'])}`",
        f"- requested expansion models: `{', '.join(config['expansion_models_requested'])}`",
        f"- effective models in this run: `{', '.join(effective_models)}`",
        "",
        "## Job accounting",
        f"- total jobs across effective models: `{total_jobs}`",
        f"- successful jobs available: `{successful_jobs}`",
        f"- failed jobs: `{failed_jobs}`",
        f"- skipped/reused jobs: `{skipped_jobs}`",
        "",
        "## Subject inventory",
    ]
    for dataset_id, subjects in sorted(dataset_subjects.items()):
        lines.append(f"- `{dataset_id}` ({len(subjects)} subjects): `{', '.join(subjects)}`")

    if skipped_models:
        lines.extend(["", "## Skipped Models"])
        for item in skipped_models:
            lines.append(f"- `{item['model']}`: {item['reason']}")

    lines.extend(
        [
            "",
            "## Dataset metrics by seed",
            "| dataset | model | seed | n_subjects | mean_pearson | std_pearson | min_pearson | max_pearson |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_seed_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['seed']} | {row['n_subjects']} | "
            f"{row['mean_pearson']:.6f} | {row['std_pearson']:.6f} | {row['min_pearson']:.6f} | {row['max_pearson']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset mean across seeds",
            "| dataset | model | n_seeds | mean_of_seed_means | std_of_seed_means | min_seed_mean | max_seed_mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_seed_averages:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_seeds']} | "
            f"{row['mean_of_seed_means']:.6f} | {row['std_of_seed_means']:.6f} | "
            f"{row['min_seed_mean']:.6f} | {row['max_seed_mean']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_incremental_comparison(
    path: Path,
    *,
    config: dict,
    effective_models: list[str],
    skipped_models: list[dict[str, str]],
) -> None:
    lines = [
        "# Incremental Comparison",
        "",
        f"- base branch: `{config['previous_fix_branch']}`",
        f"- base commit: `{config['previous_fix_commit']}`",
        f"- current branch target: `{config['current_branch']}`",
        "- this round reuses the completed multi-seed aggregate for `ridge`, `cca`, `fcnn`, and `adt`.",
        "- this round only runs missing full-subject jobs for requested expansion models under the same split, scorer, and aggregation schema.",
        "- this round does not promote final paper wording and does not regenerate heavyweight prediction dumps or checkpoints in the aggregate output directory.",
        "",
        "## Effective model set",
        f"- `{', '.join(effective_models)}`",
        "",
        "## Requested expansion set",
        f"- `{', '.join(config['expansion_models_requested'])}`",
    ]
    if skipped_models:
        lines.extend(["", "## Skipped in this round"])
        for item in skipped_models:
            lines.append(f"- `{item['model']}`: {item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_ready_tables(
    output_dir: Path,
    *,
    dataset_seed_rows: list[dict[str, object]],
    dataset_seed_averages: list[dict[str, object]],
    condition_seed_averages: list[dict[str, object]],
) -> None:
    table_dir = output_dir / "figure_ready_tables"
    ensure_dir(table_dir)

    main_seed_rows = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": row["seed"],
            "mean_pearson": row["mean_pearson"],
            "std_subject_pearson": row["std_pearson"],
            "n_subjects": row["n_subjects"],
        }
        for row in dataset_seed_rows
    ]
    write_generic_csv(
        table_dir / "main_dataset_model_seed_summary.csv",
        main_seed_rows,
        ["dataset", "model", "seed", "mean_pearson", "std_subject_pearson", "n_subjects"],
    )

    main_across_rows = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "mean_across_seeds": row["mean_of_seed_means"],
            "std_across_seeds": row["std_of_seed_means"],
            "min_seed_mean": row["min_seed_mean"],
            "max_seed_mean": row["max_seed_mean"],
        }
        for row in dataset_seed_averages
    ]
    write_generic_csv(
        table_dir / "main_dataset_model_across_seed_summary.csv",
        main_across_rows,
        ["dataset", "model", "mean_across_seeds", "std_across_seeds", "min_seed_mean", "max_seed_mean"],
    )

    condition_rows = [
        {
            "model": row["model"],
            "condition": row["condition"],
            "mean_across_seeds": row["mean_pearson_across_seeds"],
            "std_across_seeds": row["std_pearson_across_seeds"],
            "n_subjects": row["mean_n_subjects"],
            "n_recordings": row["mean_n_recordings"],
        }
        for row in condition_seed_averages
    ]
    write_generic_csv(
        table_dir / "etard_condition_model_summary.csv",
        condition_rows,
        ["model", "condition", "mean_across_seeds", "std_across_seeds", "n_subjects", "n_recordings"],
    )


def load_csv_fieldnames(path: Path) -> list[str]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def validate_schema(
    *,
    output_dir: Path,
    config: dict,
    effective_models: list[str],
    dataset_subjects: dict[str, list[str]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    subject_rows = read_csv_rows(output_dir / "subject_metrics.csv")
    recording_rows = read_csv_rows(output_dir / "recording_metrics.csv")
    dataset_by_seed_rows = read_csv_rows(output_dir / "dataset_metrics_by_seed.csv")
    dataset_across_rows = read_csv_rows(output_dir / "dataset_metrics_across_seeds.csv")
    condition_by_seed_rows = read_csv_rows(output_dir / "etard_condition_metrics_by_seed.csv")
    condition_across_rows = read_csv_rows(output_dir / "etard_condition_metrics_across_seeds.csv")

    expected_subject_jobs = {
        (dataset_id, subject_id, model_name, seed)
        for dataset_id, subjects in dataset_subjects.items()
        for subject_id in subjects
        for model_name in effective_models
        for seed in config["seeds"]
    }
    actual_subject_jobs = {
        (row["dataset"], row["subject_id"], row["model"], int(row["seed"]))
        for row in subject_rows
        if row["metric_name"] == "mean_recording_pearson_r"
    }
    missing_subject_jobs = sorted(expected_subject_jobs - actual_subject_jobs)

    expected_dataset_model_seed = {
        (dataset_id, model_name, seed)
        for dataset_id in dataset_subjects.keys()
        for model_name in effective_models
        for seed in config["seeds"]
    }
    actual_dataset_model_seed = {
        (row["dataset"], row["model"], int(row["seed"]))
        for row in dataset_by_seed_rows
    }
    missing_dataset_model_seed = sorted(expected_dataset_model_seed - actual_dataset_model_seed)

    def pearson_issues(rows: list[dict[str, str]], field_names: list[str]) -> list[dict[str, object]]:
        problems: list[dict[str, object]] = []
        for row in rows:
            for field_name in field_names:
                value = float(row[field_name])
                if value < -1.000001 or value > 1.000001:
                    problems.append({"field": field_name, "row": row})
        return problems

    subject_metric_fields_ok = load_csv_fieldnames(output_dir / "subject_metrics.csv") == SUBJECT_METRIC_FIELDS
    recording_metric_fields_ok = load_csv_fieldnames(output_dir / "recording_metrics.csv") == RECORDING_METRIC_FIELDS
    dataset_by_seed_fields_ok = load_csv_fieldnames(output_dir / "dataset_metrics_by_seed.csv") == [
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
    ]
    dataset_across_fields_ok = load_csv_fieldnames(output_dir / "dataset_metrics_across_seeds.csv") == [
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
    ]
    condition_by_seed_fields_ok = load_csv_fieldnames(output_dir / "etard_condition_metrics_by_seed.csv") == [
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
    ]
    condition_across_fields_ok = load_csv_fieldnames(output_dir / "etard_condition_metrics_across_seeds.csv") == [
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
    ]

    failure_rows_have_reasons = all(
        item.get("status") not in {"", None} and (item.get("error") not in {"", None} or item.get("status") == "missing_dataset_directory")
        for item in failures
    )

    checks = {
        "seeds_exact": sorted(config["seeds"]) == EXPECTED_SEEDS,
        "datasets_exact": [item["dataset_id"] for item in config["datasets"]] == EXPECTED_DATASETS,
        "subject_model_seed_complete": len(missing_subject_jobs) == 0,
        "dataset_model_seed_complete": len(missing_dataset_model_seed) == 0,
        "pearson_ranges_subject_metrics": len(pearson_issues(subject_rows, ["metric_value"])) == 0,
        "pearson_ranges_recording_metrics": len(pearson_issues(recording_rows, ["metric_value"])) == 0,
        "pearson_ranges_dataset_metrics_by_seed": len(pearson_issues(dataset_by_seed_rows, ["mean_pearson", "std_pearson", "median_pearson", "min_pearson", "max_pearson", "backtransformed_mean_r"])) == 0,
        "pearson_ranges_dataset_metrics_across_seeds": len(pearson_issues(dataset_across_rows, ["mean_of_seed_means", "std_of_seed_means", "min_seed_mean", "max_seed_mean", "backtransformed_mean_r"])) == 0,
        "pearson_ranges_condition_metrics_by_seed": len(pearson_issues(condition_by_seed_rows, ["mean_pearson", "median_pearson", "std_pearson"])) == 0,
        "pearson_ranges_condition_metrics_across_seeds": len(pearson_issues(condition_across_rows, ["mean_pearson_across_seeds", "std_pearson_across_seeds", "mean_median_pearson", "mean_within_seed_std"])) == 0,
        "subject_metric_fields_consistent": subject_metric_fields_ok,
        "recording_metric_fields_consistent": recording_metric_fields_ok,
        "dataset_metrics_by_seed_fields_consistent": dataset_by_seed_fields_ok,
        "dataset_metrics_across_seeds_fields_consistent": dataset_across_fields_ok,
        "condition_metrics_by_seed_fields_consistent": condition_by_seed_fields_ok,
        "condition_metrics_across_seeds_fields_consistent": condition_across_fields_ok,
        "failure_report_empty_or_reasoned": len(failures) == 0 or failure_rows_have_reasons,
    }

    return {
        "protocol": config["protocol"],
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "expected_subject_jobs": len(expected_subject_jobs),
            "actual_subject_jobs": len(actual_subject_jobs),
            "missing_subject_jobs": [
                {
                    "dataset": dataset,
                    "subject_id": subject_id,
                    "model": model,
                    "seed": seed,
                }
                for dataset, subject_id, model, seed in missing_subject_jobs[:50]
            ],
            "expected_dataset_model_seed_rows": len(expected_dataset_model_seed),
            "actual_dataset_model_seed_rows": len(actual_dataset_model_seed),
            "missing_dataset_model_seed_rows": [
                {"dataset": dataset, "model": model, "seed": seed}
                for dataset, model, seed in missing_dataset_model_seed[:50]
            ],
            "failure_count": len(failures),
            "effective_models": effective_models,
        },
    }


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)

    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    effective_models, skipped_models = resolve_model_support(config)
    dataset_subjects = {dataset_cfg["dataset_id"]: infer_subjects(dataset_cfg) for dataset_cfg in config["datasets"]}
    dataset_cfg_lookup = {dataset_cfg["dataset_id"]: dataset_cfg for dataset_cfg in config["datasets"]}
    total_jobs = sum(len(subjects) * len(effective_models) * len(config["seeds"]) for subjects in dataset_subjects.values())

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
                    "error": "",
                    "expected_locator": dataset_cfg["dataset_locator"],
                }
            )
            continue

        for seed in config["seeds"]:
            for subject_id in subjects:
                for model_name in effective_models:
                    job_dir = single_seed_job_dir(config, dataset_id, subject_id, model_name, int(seed))
                    if single_seed_job_exists(config, dataset_id, subject_id, model_name, int(seed)) and not args.force_rerun:
                        _, subject_rows, manifest = load_job_artifacts(config, dataset_id, subject_id, model_name, int(seed))
                        metric_value = float(subject_rows[0]["metric_value"])
                        num_recordings = int(subject_rows[0]["num_recordings"])
                        job_rows.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model_name,
                                "seed": int(seed),
                                "action": "reuse_existing",
                                "status": "success",
                                "job_dir": repo_relative(job_dir),
                                "checkpoint_id": manifest.get("checkpoint_id", ""),
                                "subject_metric": metric_value,
                                "num_recordings": num_recordings,
                                "error": "",
                                "source_protocol": manifest.get("protocol", config["base_single_seed_protocol"]),
                            }
                        )
                        skipped_jobs += 1
                        continue

                    subconfig = build_single_seed_subconfig(config, dataset_cfg, int(seed), effective_models)
                    try:
                        recording_rows, subject_rows, meta = run_job(
                            config=subconfig,
                            dataset_cfg=dataset_cfg,
                            dataset_dir=dataset_dir,
                            subject_id=subject_id,
                            model_name=model_name,
                            device=device,
                            output_dir=ROOT / "experiments" / "gate0_gate2_full_subject_single_seed",
                        )
                        ensure_dir(job_dir)
                        write_generic_csv(
                            job_dir / "recording_metrics.csv",
                            [row.__dict__ if hasattr(row, "__dict__") else row for row in recording_rows],
                            RECORDING_METRIC_FIELDS,
                        )
                        write_generic_csv(
                            job_dir / "subject_metrics.csv",
                            [row.__dict__ if hasattr(row, "__dict__") else row for row in subject_rows],
                            SUBJECT_METRIC_FIELDS,
                        )
                        metric_value = float(subject_rows[0].metric_value)
                        num_recordings = int(subject_rows[0].num_recordings)
                        write_json(
                            job_dir / "job_manifest.json",
                            {
                                "dataset": dataset_id,
                                "dataset_locator": dataset_cfg["dataset_locator"],
                                "subject_id": subject_id,
                                "model": model_name,
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
                                    "subject_metrics": repo_relative(job_dir / "subject_metrics.csv"),
                                },
                            },
                        )
                        job_rows.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model_name,
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
                                "model": model_name,
                                "seed": int(seed),
                                "status": "failed",
                                "error": repr(exc),
                                "job_dir": repo_relative(job_dir),
                            }
                        )
                        job_rows.append(
                            {
                                "dataset": dataset_id,
                                "subject_id": subject_id,
                                "model": model_name,
                                "seed": int(seed),
                                "action": "run_missing_seed",
                                "status": "failed",
                                "job_dir": repo_relative(job_dir),
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
        recording_rows, subject_rows, _ = load_job_artifacts(
            config,
            str(row["dataset"]),
            str(row["subject_id"]),
            str(row["model"]),
            int(row["seed"]),
        )
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
    write_json(output_dir / "job_ledger.json", {"protocol": config["protocol"], "device": device, "rows": job_rows})
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_skipped_models(output_dir / "skipped_models.md", skipped_models)
    write_figure_ready_tables(
        output_dir,
        dataset_seed_rows=dataset_seed_rows,
        dataset_seed_averages=dataset_seed_averages,
        condition_seed_averages=condition_seed_averages,
    )

    successful_jobs = len([row for row in job_rows if row["status"] == "success"])
    schema_validation = validate_schema(
        output_dir=output_dir,
        config=config,
        effective_models=effective_models,
        dataset_subjects=dataset_subjects,
        failures=failures,
    )
    write_json(output_dir / "schema_validation_report.json", schema_validation)

    write_result_summary(
        output_dir / "result_summary.md",
        config=config,
        device=device,
        dataset_subjects=dataset_subjects,
        effective_models=effective_models,
        skipped_models=skipped_models,
        total_jobs=total_jobs,
        successful_jobs=successful_jobs,
        failed_jobs=len(failures),
        skipped_jobs=skipped_jobs,
        dataset_seed_rows=dataset_seed_rows,
        dataset_seed_averages=dataset_seed_averages,
    )
    write_incremental_comparison(
        output_dir / "incremental_comparison.md",
        config=config,
        effective_models=effective_models,
        skipped_models=skipped_models,
    )

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
            "base_models_reused": config["base_models_reused"],
            "expansion_models_requested": config["expansion_models_requested"],
            "effective_models": effective_models,
            "skipped_models": skipped_models,
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
                "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
                "failure_report": repo_relative(output_dir / "failure_report.json"),
                "result_summary": repo_relative(output_dir / "result_summary.md"),
                "incremental_comparison": repo_relative(output_dir / "incremental_comparison.md"),
                "skipped_models": repo_relative(output_dir / "skipped_models.md"),
                "figure_ready_tables": repo_relative(output_dir / "figure_ready_tables"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
