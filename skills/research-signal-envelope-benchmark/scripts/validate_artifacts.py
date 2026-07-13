#!/usr/bin/env python3
"""Validate incremental benchmark artifacts and completed-job evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _common import load_json, missing_fields, print_errors


def metric_keys(path: Path, errors: list[str]) -> set[str]:
    if not path.exists():
        errors.append(f"missing metric file: {path.name}")
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "job_key" not in fields:
            errors.append(f"{path.name} lacks job_key column")
            return set()
        rows = list(reader)
    if not rows:
        errors.append(f"{path.name} has no metric rows")
    return {row["job_key"] for row in rows if row.get("job_key")}


def completed_entries(data: object) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("completed_jobs", [])
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-recording-metrics", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    errors: list[str] = []
    required = ("run_manifest.json", "run_state.json", "completed_jobs.json", "subject_metrics.csv", "failure_report.json")
    for name in required:
        if not (root / name).exists():
            errors.append(f"missing artifact: {name}")
    if errors:
        return print_errors(errors)

    try:
        manifest = load_json(root / "run_manifest.json")
        state = load_json(root / "run_state.json")
        completed_raw = load_json(root / "completed_jobs.json")
        failures = load_json(root / "failure_report.json")
    except Exception as exc:
        return print_errors([str(exc)])

    if not isinstance(manifest, dict):
        errors.append("run_manifest.json root must be an object")
    else:
        errors.extend(
            f"run manifest missing: {field}"
            for field in missing_fields(
                manifest,
                ("schema_version", "study_id", "run_id", "run_type", "code_commit", "config_id", "output_dir", "domain_gate"),
            )
        )
    if not isinstance(state, dict):
        errors.append("run_state.json root must be an object")
    entries = completed_entries(completed_raw)
    completed_keys = {str(item.get("job_key", "")) for item in entries if item.get("job_key")}
    if len(completed_keys) != len(entries):
        errors.append("completed jobs contain missing or duplicate job_key values")
    for item in entries:
        for field in ("dataset_id", "subject_or_fold_id", "model_id", "seed", "config_id"):
            if field not in item:
                errors.append(f"completed job {item.get('job_key')} missing {field}")

    subject_keys = metric_keys(root / "subject_metrics.csv", errors)
    missing_subject = completed_keys - subject_keys
    if missing_subject:
        errors.append(f"completed jobs without subject metric rows: {sorted(missing_subject)}")
    recording_path = root / "recording_metrics.csv"
    if args.require_recording_metrics or recording_path.exists():
        recording_keys = metric_keys(recording_path, errors)
        missing_recording = completed_keys - recording_keys
        if missing_recording:
            errors.append(f"completed jobs without recording metric rows: {sorted(missing_recording)}")

    if not isinstance(failures, (dict, list)):
        errors.append("failure_report.json root must be an object or list")
    state_completed = state.get("completed_jobs", []) if isinstance(state, dict) else []
    state_keys = {
        item if isinstance(item, str) else str(item.get("job_key", ""))
        for item in state_completed
        if isinstance(item, (str, dict))
    }
    if state_keys and state_keys != completed_keys:
        errors.append("run_state completed jobs do not match completed_jobs.json")
    return print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
