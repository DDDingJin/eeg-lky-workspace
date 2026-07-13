"""Merge availability, canonical validation, and minimal-closure evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COHORT = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]


def load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--accepted-validation", type=Path, required=True)
    parser.add_argument("--new-validation", type=Path, required=True)
    parser.add_argument("--minimal-closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    availability = load(args.availability, {})
    accepted = {row["subject_id"]: row for row in load(args.accepted_validation, {}).get("subjects", [])}
    generated = {row["subject_id"]: row for row in load(args.new_validation, {}).get("subjects", [])}
    checks = {row["subject_id"]: row for row in load(args.minimal_closure, {}).get("statuses", [])}
    rows = []
    for source in availability.get("subjects", []):
        subject = source["subject_id"]
        validation = accepted.get(subject, generated.get(subject, {}))
        check = checks.get(subject, {})
        canonical = bool(validation.get("canonicalization_passed", False))
        reason = validation.get("invalid_reason", "")
        if not canonical and subject in COHORT and not reason:
            reason = "official MATLAB wrapper produced no validated paired MAT before batch termination"
        rows.append({
            "subject_id": subject,
            "path_complete": bool(source.get("path_complete", False)),
            "canonicalization_passed": canonical,
            "smoke_passed": check.get("ridge_sanity") == "passed" and check.get("eegnet_smoke") == "passed",
            "ridge_sanity": check.get("ridge_sanity", "not_run"),
            "eegnet_smoke": check.get("eegnet_smoke", "not_run"),
            "status": "canonicalization_passed" if canonical else ("missing" if source.get("status") == "missing" else "invalid"),
            "reason": reason or source.get("missing_reason", ""),
            "paired_mat_sha256": validation.get("paired_mat_sha256", ""),
            "envelope_mat_sha256": validation.get("envelope_mat_sha256", ""),
            "fs_hz": validation.get("paired_fs_hz", ""),
            "trial_count": validation.get("trial_count", ""),
            "channel_order_sha256": validation.get("channel_order_sha256", ""),
        })
    cohort_rows = [row for row in rows if row["subject_id"] in COHORT]
    report = {
        "protocol": "meg_scans_preregistered_cohort_v1",
        "fixed_cohort": COHORT,
        "subjects": rows,
        "path_complete_subjects": [row["subject_id"] for row in rows if row["path_complete"]],
        "canonicalization_passed_subjects": [row["subject_id"] for row in rows if row["canonicalization_passed"]],
        "invalid_subjects": [row["subject_id"] for row in rows if row["status"] == "invalid"],
        "missing_subjects": [row["subject_id"] for row in rows if row["status"] == "missing"],
        "cohort_ready_for_full_11model_run": False,
        "full_11model_run_started": False,
        "incomplete_reason": "canonicalization batch terminated after no-progress official wrapper stage; no multi-subject full benchmark was started",
        "cohort_rows": cohort_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
