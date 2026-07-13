#!/usr/bin/env python3
"""Validate a standardized JSON benchmark configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_json, missing_fields, print_errors


REQUIRED = (
    "schema_version",
    "study_id",
    "run_type",
    "models",
    "datasets",
    "protocol",
    "target",
    "metrics",
    "seeds",
    "runtime",
    "output_dir",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        data = load_json(args.config)
    except Exception as exc:
        return print_errors([str(exc)])
    if not isinstance(data, dict):
        return print_errors(["config root must be an object"])
    errors.extend(f"missing field: {field}" for field in missing_fields(data, REQUIRED))
    for field in ("models", "datasets", "metrics", "seeds"):
        if field in data and (not isinstance(data[field], list) or not data[field]):
            errors.append(f"{field} must be a non-empty list")
    if data.get("run_type") not in {"smoke", "closure", "pilot", "article"}:
        errors.append(f"invalid run_type: {data.get('run_type')}")
    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        errors.append("runtime must be an object")
        runtime = {}
    if args.mode == "full":
        if data.get("run_type") not in {"pilot", "article"}:
            errors.append("full mode requires run_type pilot or article")
        if runtime.get("smoke_only") is True:
            errors.append("full config cannot set smoke_only=true")
        epochs = runtime.get("epochs")
        if isinstance(epochs, int) and epochs <= 1:
            errors.append("full config has epochs <= 1; verify smoke settings did not leak")
        if not data.get("code_commit"):
            errors.append("full config requires code_commit")
        if not data.get("config_id"):
            errors.append("full config requires config_id")
        if not data.get("split_version"):
            errors.append("full config requires split_version")
    output = str(data.get("output_dir", ""))
    if output.startswith("/Users/") or output.startswith("C:\\Users\\"):
        errors.append("publication config contains a machine-specific absolute output path")
    return print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
