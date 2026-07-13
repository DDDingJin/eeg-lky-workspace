#!/usr/bin/env python3
"""Validate machine-readable benchmark workflow state."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_json, missing_fields, print_errors


REQUIRED = (
    "schema_version",
    "study_id",
    "phase",
    "active_modules",
    "role",
    "branch",
    "commit",
    "config_id",
    "dataset_version",
    "split_version",
    "last_gate",
    "planned_jobs",
    "completed_jobs",
    "failed_jobs",
    "pending_jobs",
    "resume_enabled",
    "last_update",
)
ROLES = {"reviewer", "implementer", "standalone"}
GATES = {
    "not_evaluated",
    "design_only",
    "approved_for_full_run",
    "approved_as_smoke_only",
    "engineering_closure_required",
    "rejected_until_reproduced_cleanly",
}


def job_key(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("job_key", ""))
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        data = load_json(args.state)
    except Exception as exc:
        return print_errors([str(exc)])
    if not isinstance(data, dict):
        return print_errors(["state root must be an object"])
    errors.extend(f"missing field: {field}" for field in missing_fields(data, REQUIRED))
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("role") not in ROLES:
        errors.append(f"invalid role: {data.get('role')}")
    phase = data.get("phase")
    if not isinstance(phase, str) or not phase.startswith("P"):
        errors.append("phase must be a P-stage string")
    if data.get("last_gate") not in GATES:
        errors.append(f"invalid last_gate: {data.get('last_gate')}")
    if not isinstance(data.get("active_modules"), list):
        errors.append("active_modules must be a list")
    for field in ("planned_jobs", "completed_jobs", "failed_jobs", "pending_jobs"):
        if not isinstance(data.get(field), list):
            errors.append(f"{field} must be a list")
    if not isinstance(data.get("resume_enabled"), bool):
        errors.append("resume_enabled must be boolean")

    sets = {
        name: {job_key(item) for item in data.get(name, []) if job_key(item)}
        for name in ("completed_jobs", "failed_jobs", "pending_jobs")
        if isinstance(data.get(name), list)
    }
    if sets.get("completed_jobs", set()) & sets.get("pending_jobs", set()):
        errors.append("a job cannot be both completed and pending")
    return print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
