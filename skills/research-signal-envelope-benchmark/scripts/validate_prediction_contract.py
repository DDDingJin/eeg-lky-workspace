#!/usr/bin/env python3
"""Validate a model prediction-contract JSON artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_json, missing_fields, print_errors


REQUIRED = (
    "schema_version",
    "model_id",
    "input_shape",
    "raw_output_shape",
    "prediction_shape",
    "target_shape",
    "scorer_input_shape",
    "alignment_rule",
    "overlap_aggregation",
    "mask_rule",
    "variable_length_rule",
)


def shape_ok(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, (str, int)) for item in value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--allow-prediction-target-shape-difference", action="store_true")
    args = parser.parse_args()
    try:
        data = load_json(args.contract)
    except Exception as exc:
        return print_errors([str(exc)])
    if not isinstance(data, dict):
        return print_errors(["contract root must be an object"])
    errors = [f"missing field: {field}" for field in missing_fields(data, REQUIRED)]
    for field in ("input_shape", "raw_output_shape", "prediction_shape", "target_shape", "scorer_input_shape"):
        if field in data and not shape_ok(data[field]):
            errors.append(f"{field} must be a non-empty shape list")
    if (
        not args.allow_prediction_target_shape_difference
        and data.get("prediction_shape") != data.get("target_shape")
    ):
        errors.append("prediction_shape and target_shape differ without an explicit validator override")
    for field in ("alignment_rule", "overlap_aggregation", "mask_rule", "variable_length_rule"):
        if field in data and not str(data[field]).strip():
            errors.append(f"{field} must be explicit")
    return print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
