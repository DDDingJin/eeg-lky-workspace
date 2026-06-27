from __future__ import annotations

import numpy as np


def build_lag_matrix(values: np.ndarray, start_lag: int, end_lag: int) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    n_time, n_feat = values.shape
    lags = list(range(start_lag, end_lag))
    matrix = np.zeros((n_time, n_feat * len(lags)), dtype=np.int32)
    for index, lag in enumerate(lags):
        src_start = max(0, -lag)
        src_end = min(n_time, n_time - lag) if lag >= 0 else n_time
        dst_start = src_start + lag if lag >= 0 else src_start + lag
        dst_end = src_end + lag if lag >= 0 else src_end + lag
        matrix[dst_start:dst_end, index * n_feat : (index + 1) * n_feat] = values[src_start:src_end]
    return matrix


def valid_slice(length: int, start_lag: int, end_lag: int) -> slice:
    max_left = max(0, end_lag - 1)
    max_right = max(0, -start_lag)
    return slice(max_left, length - max_right)


def unsafe_cross_boundary_rows(recording_lengths: list[int], start_lag: int, end_lag: int) -> int:
    origins = np.concatenate(
        [np.full(length, fill_value=recording_index, dtype=np.int32) for recording_index, length in enumerate(recording_lengths)],
        axis=0,
    )
    lagged = build_lag_matrix(origins, start_lag=start_lag, end_lag=end_lag)
    valid = lagged[valid_slice(len(origins), start_lag, end_lag)]
    mixed = 0
    for row in valid:
        row_values = row[row >= 0]
        if row_values.size and np.unique(row_values).size > 1:
            mixed += 1
    return mixed


def safe_cross_boundary_rows(recording_lengths: list[int], start_lag: int, end_lag: int) -> int:
    mixed = 0
    for recording_index, length in enumerate(recording_lengths):
        origins = np.full(length, fill_value=recording_index, dtype=np.int32)
        lagged = build_lag_matrix(origins, start_lag=start_lag, end_lag=end_lag)
        valid = lagged[valid_slice(length, start_lag, end_lag)]
        for row in valid:
            row_values = row[row >= 0]
            if row_values.size and np.unique(row_values).size > 1:
                mixed += 1
    return mixed
