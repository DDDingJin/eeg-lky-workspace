from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODELS = ["linear", "ridge", "lasso", "elasticnet", "cca", "fcnn", "cnn", "eegnet", "adt", "vlaai", "happyquokka"]
BANNED_SUFFIXES = {".mat", ".fif", ".pt", ".pth", ".npy", ".npz", ".h5", ".ckpt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/meg_scans_subject_specific_11models_unified_adapters_sub03_v1")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    errors: list[str] = []
    for name in ["run_config.json", "meg_model_set_lock.json", "data_integrity_check.json", "split_manifest.json", "shape_preflight.json", "run_manifest.json"]:
        if not (output_dir / name).exists():
            errors.append(f"missing {name}")
    if not errors:
        config = read_json(output_dir / "run_config.json")
        lock = read_json(output_dir / "meg_model_set_lock.json")
        if lock.get("models") != EXPECTED_MODELS or lock.get("model_count") != 11:
            errors.append("model lock is not the exact 11-model roster")
        subjects = config.get("subjects") or [{"subject_id": config["subject_ids"][0]}]
        planned = [f"{subject['subject_id']}:{model}:seed{config['seed']}" for subject in subjects for model in EXPECTED_MODELS]
        preflight = read_json(output_dir / "shape_preflight.json")
        preflight_rows = preflight.get("rows", [])
        if len(preflight_rows) != len(planned):
            errors.append(f"shape_preflight row count {len(preflight_rows)} != planned jobs {len(planned)}")
        failed_preflight = [row for row in preflight_rows if row.get("status") != "ok"]
        if failed_preflight:
            errors.append("preflight failures: " + json.dumps(failed_preflight, ensure_ascii=False))
        completed_path = output_dir / "completed_jobs.json"
        failure_path = output_dir / "failure_report.json"
        completed = read_json(completed_path).get("completed_jobs", []) if completed_path.exists() else []
        failures = read_json(failure_path).get("failures", []) if failure_path.exists() else []
        completed_keys = [row["job_key"] for row in completed]
        failure_keys = [row["job_key"] for row in failures]
        if len(completed_keys) != len(set(completed_keys)):
            errors.append("completed job key not unique")
        if set(completed_keys) & set(failure_keys):
            errors.append("same job appears as success and failure")
        if any(row.get("status") == "success" for row in failures):
            errors.append("failure masquerades as success")
        rec_rows = csv_rows(output_dir / "recording_metrics.csv")
        subj_rows = csv_rows(output_dir / "subject_metrics.csv")
        for job_key in completed_keys:
            rec_for_job = [row for row in rec_rows if row.get("job_key") == job_key]
            subj_for_job = [row for row in subj_rows if row.get("job_key") == job_key]
            if len(rec_for_job) != 2:
                errors.append(f"{job_key} recording row count {len(rec_for_job)} != 2")
            if len(subj_for_job) != 1:
                errors.append(f"{job_key} subject row count {len(subj_for_job)} != 1")
            if rec_for_job and subj_for_job:
                recomputed = float(np.mean([float(row["metric_value"]) for row in rec_for_job]))
                observed = float(subj_for_job[0]["metric_value"])
                if abs(recomputed - observed) > 1e-9:
                    errors.append(f"{job_key} subject metric mismatch: {observed} vs {recomputed}")
        if args.require_complete and set(completed_keys) != set(planned):
            errors.append("not all planned jobs completed successfully")

    proc = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        errors.append(f"git ls-files failed: {proc.stderr.strip()}")
    else:
        bad = [line for line in proc.stdout.splitlines() if Path(line).suffix.lower() in BANNED_SUFFIXES and "meg_scans_subject_specific_11models" in line]
        if bad:
            errors.append("banned tracked artifacts: " + ", ".join(bad))

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checked_dir": str(output_dir),
        "expected_models": EXPECTED_MODELS,
        "require_complete": args.require_complete,
        "banned_suffixes": sorted(BANNED_SUFFIXES),
    }
    if args.require_complete:
        report["run_status"] = "completed" if report["status"] == "passed" else "incomplete_or_failed"
    else:
        report["run_status"] = "incomplete_or_failed"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema_validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
