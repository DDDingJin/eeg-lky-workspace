from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.boundaries import safe_cross_boundary_rows, unsafe_cross_boundary_rows
from benchmark.registry import load_model_registry, validate_registry_entries
from benchmark.result_schema import PREDICTION_FIELDS, RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, prediction_rows_from_series, recording_metric_row, subject_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "registry", "scorer", "boundaries"], default="all")
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "gate0_gate2_smoke"))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_registry(output_dir: Path) -> dict[str, object]:
    entries = load_model_registry()
    report = validate_registry_entries(entries)
    with open(output_dir / "registry_validation.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def synthetic_windows() -> list[WindowPrediction]:
    windows: list[WindowPrediction] = []
    blueprint = [
        ("S01", "rec_A", 48, 0, 16),
        ("S01", "rec_A", 48, 8, 16),
        ("S01", "rec_A", 48, 24, 16),
        ("S01", "rec_B", 40, 0, 20),
        ("S01", "rec_B", 40, 10, 20),
        ("S02", "rec_C", 36, 0, 18),
        ("S02", "rec_C", 36, 9, 18)
    ]
    for subject_id, recording_id, recording_length, start_index, window_length in blueprint:
        time = np.arange(window_length, dtype=np.float32)
        target = np.sin((time + start_index) / 5.0).astype(np.float32)
        prediction = (target * 0.92 + 0.03).astype(np.float32)
        windows.append(
            WindowPrediction(
                dataset="synthetic_fixture",
                model="schema_smoke",
                task="reconstruction",
                protocol="gate0_gate2_smoke",
                seed=0,
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=64,
                checkpoint_id="smoke-checkpoint",
                recording_length=recording_length,
                start_index=start_index,
                prediction=prediction,
                target=target,
            )
        )
    return windows


def run_scorer(output_dir: Path) -> dict[str, object]:
    windows = synthetic_windows()
    by_recording: dict[tuple[str, str], list[WindowPrediction]] = {}
    for window in windows:
        by_recording.setdefault((window.subject_id, window.recording_id), []).append(window)

    prediction_rows = []
    recording_rows = []
    for (_, _), recording_windows in sorted(by_recording.items()):
        first = recording_windows[0]
        prediction, target, valid_mask = aggregate_overlapping_windows(first.recording_length, recording_windows)
        prediction_rows.extend(
            prediction_rows_from_series(
                dataset=first.dataset,
                model=first.model,
                task=first.task,
                protocol=first.protocol,
                seed=first.seed,
                subject_id=first.subject_id,
                recording_id=first.recording_id,
                sampling_rate=first.sampling_rate,
                checkpoint_id=first.checkpoint_id,
                artifact_scope="smoke_test_only",
                prediction=prediction,
                target=target,
                valid_mask=valid_mask,
            )
        )
        recording_rows.append(
            recording_metric_row(
                dataset=first.dataset,
                model=first.model,
                task=first.task,
                protocol=first.protocol,
                seed=first.seed,
                subject_id=first.subject_id,
                recording_id=first.recording_id,
                sampling_rate=first.sampling_rate,
                checkpoint_id=first.checkpoint_id,
                artifact_scope="smoke_test_only",
                prediction=prediction,
                target=target,
                valid_mask=valid_mask,
            )
        )

    subject_rows = subject_metric_rows(recording_rows)
    write_rows(output_dir / "prediction_samples.csv", prediction_rows, PREDICTION_FIELDS)
    write_rows(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", subject_rows, SUBJECT_METRIC_FIELDS)

    report = {
        "artifact_scope": "smoke_test_only",
        "dataset": "synthetic_fixture",
        "num_prediction_rows": len(prediction_rows),
        "num_recording_rows": len(recording_rows),
        "num_subject_rows": len(subject_rows),
        "recording_metric_mean": float(np.mean([row.metric_value for row in recording_rows])),
        "subject_metric_mean": float(np.mean([row.metric_value for row in subject_rows]))
    }
    with open(output_dir / "scorer_smoke_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def run_boundaries(output_dir: Path) -> dict[str, object]:
    recording_lengths = [12, 15, 11]
    start_lag = 0
    end_lag = 4
    unsafe_rows = unsafe_cross_boundary_rows(recording_lengths, start_lag, end_lag)
    safe_rows = safe_cross_boundary_rows(recording_lengths, start_lag, end_lag)
    report = {
        "recording_lengths": recording_lengths,
        "start_lag": start_lag,
        "end_lag": end_lag,
        "unsafe_cross_boundary_rows": unsafe_rows,
        "safe_cross_boundary_rows": safe_rows,
        "boundary_fix_passed": unsafe_rows > 0 and safe_rows == 0,
        "artifact_scope": "smoke_test_only"
    }
    with open(output_dir / "boundary_validation.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    try:
        output_dir_value = output_dir.relative_to(ROOT).as_posix()
    except ValueError:
        output_dir_value = output_dir.as_posix()
    report: dict[str, object] = {"output_dir": output_dir_value}
    ok = True

    if args.phase in {"all", "registry"}:
        registry_report = run_registry(output_dir)
        report["registry"] = registry_report
        ok = ok and not registry_report["errors"]
    if args.phase in {"all", "scorer"}:
        report["scorer"] = run_scorer(output_dir)
    if args.phase in {"all", "boundaries"}:
        boundary_report = run_boundaries(output_dir)
        report["boundaries"] = boundary_report
        ok = ok and bool(boundary_report["boundary_fix_passed"])

    report["ok"] = bool(ok)
    with open(output_dir / "smoke_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
