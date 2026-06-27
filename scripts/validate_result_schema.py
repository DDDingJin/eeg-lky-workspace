from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.registry import load_model_registry, validate_registry_entries
from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, load_dict_rows, validate_required_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-model-registry", action="store_true")
    parser.add_argument("--check-metrics-dir")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {"repo_root": ".", "checks": {}}
    errors: list[str] = []

    if args.check_model_registry:
        entries = load_model_registry()
        registry_report = validate_registry_entries(entries)
        report["checks"]["model_registry"] = registry_report
        errors.extend(registry_report["errors"])

    if args.check_metrics_dir:
        metrics_dir = Path(args.check_metrics_dir)
        try:
            metrics_dir_value = metrics_dir.relative_to(ROOT).as_posix()
        except ValueError:
            metrics_dir_value = metrics_dir.as_posix()
        metrics_report: dict[str, object] = {"metrics_dir": metrics_dir_value, "errors": []}
        recording_path = metrics_dir / "recording_metrics.csv"
        subject_path = metrics_dir / "subject_metrics.csv"
        if not recording_path.exists():
            metrics_report["errors"].append(f"missing {recording_path.name}")
        else:
            recording_rows = load_dict_rows(recording_path)
            metrics_report["recording_rows"] = len(recording_rows)
            metrics_report["errors"].extend(validate_required_fields(recording_rows, RECORDING_METRIC_FIELDS))
        if not subject_path.exists():
            metrics_report["errors"].append(f"missing {subject_path.name}")
        else:
            subject_rows = load_dict_rows(subject_path)
            metrics_report["subject_rows"] = len(subject_rows)
            metrics_report["errors"].extend(validate_required_fields(subject_rows, SUBJECT_METRIC_FIELDS))
        report["checks"]["metrics_dir"] = metrics_report
        errors.extend(metrics_report["errors"])

    report["ok"] = len(errors) == 0
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
