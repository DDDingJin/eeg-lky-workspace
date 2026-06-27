from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .result_schema import PredictionSample, RecordingMetricRow, SubjectMetricRow


@dataclass(frozen=True)
class WindowPrediction:
    dataset: str
    model: str
    task: str
    protocol: str
    seed: int
    subject_id: str
    recording_id: str
    sampling_rate: int
    checkpoint_id: str
    recording_length: int
    start_index: int
    prediction: np.ndarray
    target: np.ndarray


def aggregate_overlapping_windows(recording_length: int, windows: Iterable[WindowPrediction]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_sum = np.zeros(recording_length, dtype=np.float64)
    target_sum = np.zeros(recording_length, dtype=np.float64)
    counts = np.zeros(recording_length, dtype=np.int32)

    for window in windows:
        length = int(len(window.prediction))
        stop = window.start_index + length
        pred_sum[window.start_index:stop] += np.asarray(window.prediction, dtype=np.float64)
        target_sum[window.start_index:stop] += np.asarray(window.target, dtype=np.float64)
        counts[window.start_index:stop] += 1

    valid_mask = counts > 0
    prediction = np.zeros(recording_length, dtype=np.float64)
    target = np.zeros(recording_length, dtype=np.float64)
    prediction[valid_mask] = pred_sum[valid_mask] / counts[valid_mask]
    target[valid_mask] = target_sum[valid_mask] / counts[valid_mask]
    return prediction.astype(np.float32), target.astype(np.float32), valid_mask.astype(np.int32)


def pearson_on_valid(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = valid_mask.astype(bool)
    if int(valid.sum()) < 2:
        return float("nan")
    pred_valid = prediction[valid].astype(np.float64)
    target_valid = target[valid].astype(np.float64)
    pred_center = pred_valid - pred_valid.mean()
    target_center = target_valid - target_valid.mean()
    denom = np.sqrt(np.sum(pred_center**2) * np.sum(target_center**2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(pred_center * target_center) / denom)


def prediction_rows_from_series(
    *,
    dataset: str,
    model: str,
    task: str,
    protocol: str,
    seed: int,
    subject_id: str,
    recording_id: str,
    sampling_rate: int,
    checkpoint_id: str,
    artifact_scope: str,
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> list[PredictionSample]:
    rows: list[PredictionSample] = []
    for index, (pred_value, target_value, mask_value) in enumerate(zip(prediction, target, valid_mask)):
        rows.append(
            PredictionSample(
                dataset=dataset,
                model=model,
                task=task,
                protocol=protocol,
                seed=seed,
                subject_id=subject_id,
                recording_id=recording_id,
                sampling_rate=sampling_rate,
                time_index=index,
                prediction=float(pred_value),
                target=float(target_value),
                valid_mask=int(mask_value),
                checkpoint_id=checkpoint_id,
                artifact_scope=artifact_scope,
            )
        )
    return rows


def recording_metric_row(
    *,
    dataset: str,
    model: str,
    task: str,
    protocol: str,
    seed: int,
    subject_id: str,
    recording_id: str,
    sampling_rate: int,
    checkpoint_id: str,
    artifact_scope: str,
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> RecordingMetricRow:
    return RecordingMetricRow(
        dataset=dataset,
        model=model,
        task=task,
        protocol=protocol,
        seed=seed,
        subject_id=subject_id,
        recording_id=recording_id,
        sampling_rate=sampling_rate,
        metric_name="pearson_r",
        metric_value=pearson_on_valid(prediction, target, valid_mask),
        num_valid_samples=int(valid_mask.astype(bool).sum()),
        checkpoint_id=checkpoint_id,
        artifact_scope=artifact_scope,
    )


def subject_metric_rows(recording_rows: Iterable[RecordingMetricRow]) -> list[SubjectMetricRow]:
    grouped: dict[tuple[str, str, str, str, int, str, str, str], list[float]] = {}
    for row in recording_rows:
        key = (
            row.dataset,
            row.model,
            row.task,
            row.protocol,
            row.seed,
            row.subject_id,
            row.checkpoint_id,
            row.artifact_scope,
        )
        grouped.setdefault(key, []).append(float(row.metric_value))

    subject_rows: list[SubjectMetricRow] = []
    for key, values in sorted(grouped.items()):
        dataset, model, task, protocol, seed, subject_id, checkpoint_id, artifact_scope = key
        subject_rows.append(
            SubjectMetricRow(
                dataset=dataset,
                model=model,
                task=task,
                protocol=protocol,
                seed=seed,
                subject_id=subject_id,
                metric_name="mean_recording_pearson_r",
                metric_value=float(np.mean(values)),
                num_recordings=len(values),
                checkpoint_id=checkpoint_id,
                artifact_scope=artifact_scope,
            )
        )
    return subject_rows
