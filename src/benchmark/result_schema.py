from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable


PREDICTION_FIELDS = [
    "dataset",
    "model",
    "task",
    "protocol",
    "seed",
    "subject_id",
    "recording_id",
    "sampling_rate",
    "time_index",
    "prediction",
    "target",
    "valid_mask",
    "checkpoint_id",
    "artifact_scope",
]

RECORDING_METRIC_FIELDS = [
    "dataset",
    "model",
    "task",
    "protocol",
    "seed",
    "subject_id",
    "recording_id",
    "sampling_rate",
    "metric_name",
    "metric_value",
    "num_valid_samples",
    "checkpoint_id",
    "artifact_scope",
]

SUBJECT_METRIC_FIELDS = [
    "dataset",
    "model",
    "task",
    "protocol",
    "seed",
    "subject_id",
    "metric_name",
    "metric_value",
    "num_recordings",
    "checkpoint_id",
    "artifact_scope",
]


@dataclass(frozen=True)
class PredictionSample:
    dataset: str
    model: str
    task: str
    protocol: str
    seed: int
    subject_id: str
    recording_id: str
    sampling_rate: int
    time_index: int
    prediction: float
    target: float
    valid_mask: int
    checkpoint_id: str
    artifact_scope: str


@dataclass(frozen=True)
class RecordingMetricRow:
    dataset: str
    model: str
    task: str
    protocol: str
    seed: int
    subject_id: str
    recording_id: str
    sampling_rate: int
    metric_name: str
    metric_value: float
    num_valid_samples: int
    checkpoint_id: str
    artifact_scope: str


@dataclass(frozen=True)
class SubjectMetricRow:
    dataset: str
    model: str
    task: str
    protocol: str
    seed: int
    subject_id: str
    metric_name: str
    metric_value: float
    num_recordings: int
    checkpoint_id: str
    artifact_scope: str


def write_rows(path: str | Path, rows: Iterable[object], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_dict_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_required_fields(rows: Iterable[dict[str, str]], required_fields: Iterable[str]) -> list[str]:
    required = list(required_fields)
    errors: list[str] = []
    for index, row in enumerate(rows):
        missing = [field for field in required if field not in row or row[field] == ""]
        if missing:
            errors.append(f"row {index} missing required fields: {', '.join(missing)}")
    return errors
