from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from .cca import build_lag_matrix, trim_valid_range


@dataclass
class LinearDecoderModel:
    start_lag: int
    end_lag: int
    method: str
    alpha: float
    coef_: np.ndarray
    intercept_: float


@dataclass
class ForwardEncodingModel:
    start_lag: int
    end_lag: int
    method: str
    alpha: float
    coef_: np.ndarray
    intercept_: np.ndarray


@dataclass
class StandardizedLinearDecoderModel:
    start_lag: int
    end_lag: int
    method: str
    alpha: float
    coef_: np.ndarray
    intercept_: float
    x_mean_: np.ndarray
    x_scale_: np.ndarray
    y_mean_: float


def _lagged_pair(x: np.ndarray, y: np.ndarray, start_lag: int, end_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x_lag, _, y_center = trim_valid_range(x, y, start_lag, end_lag)
    return x_lag, y_center


def _lagged_forward_pair(x: np.ndarray, y: np.ndarray, start_lag: int, end_lag: int) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be (time, channels)")
    if y.ndim != 1:
        raise ValueError("y must be (time,)")
    _, y_lag, _ = trim_valid_range(x, y, start_lag, end_lag)
    max_left = max(0, end_lag - 1)
    max_right = max(0, -start_lag)
    valid = slice(max_left, x.shape[0] - max_right)
    x_center = x[valid]
    return y_lag, x_center


def fit_avgdec_model(
    x_parts: list[np.ndarray],
    y_parts: list[np.ndarray],
    *,
    start_lag: int,
    end_lag: int,
    alpha: float,
    method: str,
) -> LinearDecoderModel:
    coefs = []
    intercepts = []
    for x_part, y_part in zip(x_parts, y_parts):
        x_lag, y_center = _lagged_pair(x_part, y_part, start_lag, end_lag)
        if method == "ridge":
            model = Ridge(alpha=alpha)
        elif method == "lasso":
            model = Lasso(alpha=alpha, max_iter=5000)
        else:
            raise ValueError(f"unknown method: {method}")
        model.fit(x_lag, y_center)
        coefs.append(np.asarray(model.coef_, dtype=np.float64))
        intercepts.append(float(model.intercept_))

    return LinearDecoderModel(
        start_lag=start_lag,
        end_lag=end_lag,
        method=f"avgdec_{method}",
        alpha=alpha,
        coef_=np.mean(np.stack(coefs, axis=0), axis=0),
        intercept_=float(np.mean(intercepts)),
    )


def fit_avgdec_lasso_model_fast(
    x_parts: list[np.ndarray],
    y_parts: list[np.ndarray],
    *,
    start_lag: int,
    end_lag: int,
    alpha: float,
) -> StandardizedLinearDecoderModel:
    coefs = []
    intercepts = []
    x_means = []
    x_scales = []
    y_means = []
    for x_part, y_part in zip(x_parts, y_parts):
        x_lag, y_center = _lagged_pair(x_part, y_part, start_lag, end_lag)
        x_scaler = StandardScaler(with_mean=True, with_std=True)
        x_std = x_scaler.fit_transform(x_lag)
        y_mean = float(y_center.mean())
        y0 = y_center - y_mean
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=3000, tol=1e-3, selection="random")
        model.fit(x_std, y0)
        coefs.append(np.asarray(model.coef_, dtype=np.float64))
        intercepts.append(0.0)
        x_means.append(np.asarray(x_scaler.mean_, dtype=np.float64))
        x_scales.append(np.asarray(x_scaler.scale_, dtype=np.float64))
        y_means.append(y_mean)

    return StandardizedLinearDecoderModel(
        start_lag=start_lag,
        end_lag=end_lag,
        method="avgdec_lasso",
        alpha=alpha,
        coef_=np.mean(np.stack(coefs, axis=0), axis=0),
        intercept_=float(np.mean(intercepts)),
        x_mean_=np.mean(np.stack(x_means, axis=0), axis=0),
        x_scale_=np.mean(np.stack(x_scales, axis=0), axis=0),
        y_mean_=float(np.mean(y_means)),
    )


def fit_forward_ridge_model(
    x_parts: list[np.ndarray],
    y_parts: list[np.ndarray],
    *,
    start_lag: int,
    end_lag: int,
    alpha: float,
) -> ForwardEncodingModel:
    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    y_lag, x_center = _lagged_forward_pair(x_train, y_train, start_lag, end_lag)
    model = Ridge(alpha=alpha)
    model.fit(y_lag, x_center)
    return ForwardEncodingModel(
        start_lag=start_lag,
        end_lag=end_lag,
        method="forward_ridge",
        alpha=alpha,
        coef_=np.asarray(model.coef_, dtype=np.float64),
        intercept_=np.asarray(model.intercept_, dtype=np.float64),
    )


def fit_avgcorr_ridge_model(
    x_parts: list[np.ndarray],
    y_parts: list[np.ndarray],
    *,
    start_lag: int,
    end_lag: int,
    alpha: float,
) -> LinearDecoderModel:
    cov_xx = []
    cov_xy = []
    means_x = []
    means_y = []
    for x_part, y_part in zip(x_parts, y_parts):
        x_lag, y_center = _lagged_pair(x_part, y_part, start_lag, end_lag)
        means_x.append(x_lag.mean(axis=0))
        means_y.append(float(y_center.mean()))
        x0 = x_lag - x_lag.mean(axis=0, keepdims=True)
        y0 = y_center - y_center.mean()
        cov_xx.append((x0.T @ x0) / max(1, x0.shape[0] - 1))
        cov_xy.append((x0.T @ y0) / max(1, x0.shape[0] - 1))

    g = np.mean(np.stack(cov_xx, axis=0), axis=0)
    b = np.mean(np.stack(cov_xy, axis=0), axis=0)
    coef = np.linalg.solve(g + alpha * np.eye(g.shape[0]), b)
    mean_x = np.mean(np.stack(means_x, axis=0), axis=0)
    mean_y = float(np.mean(means_y))
    intercept = mean_y - float(mean_x @ coef)
    return LinearDecoderModel(
        start_lag=start_lag,
        end_lag=end_lag,
        method="avgcorr_ridge",
        alpha=alpha,
        coef_=coef.astype(np.float64),
        intercept_=intercept,
    )


def fit_avgcorr_lasso_model(
    x_parts: list[np.ndarray],
    y_parts: list[np.ndarray],
    *,
    start_lag: int,
    end_lag: int,
    alpha: float,
) -> LinearDecoderModel:
    cov_xx = []
    cov_xy = []
    means_x = []
    means_y = []
    for x_part, y_part in zip(x_parts, y_parts):
        x_lag, y_center = _lagged_pair(x_part, y_part, start_lag, end_lag)
        means_x.append(x_lag.mean(axis=0))
        means_y.append(float(y_center.mean()))
        x0 = x_lag - x_lag.mean(axis=0, keepdims=True)
        y0 = y_center - y_center.mean()
        cov_xx.append((x0.T @ x0) / max(1, x0.shape[0] - 1))
        cov_xy.append((x0.T @ y0) / max(1, x0.shape[0] - 1))

    g = np.mean(np.stack(cov_xx, axis=0), axis=0)
    b = np.mean(np.stack(cov_xy, axis=0), axis=0)

    jitter = 1e-6
    chol = None
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(g + jitter * np.eye(g.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if chol is None:
        eigvals, eigvecs = np.linalg.eigh((g + g.T) * 0.5)
        eigvals = np.clip(eigvals, 1e-8, None)
        g_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        chol = np.linalg.cholesky(g_psd)
    x_surrogate = chol.T
    y_surrogate = np.linalg.solve(chol, b)
    model = Lasso(alpha=alpha / x_surrogate.shape[0], fit_intercept=False, max_iter=10000)
    model.fit(x_surrogate, y_surrogate)
    coef = np.asarray(model.coef_, dtype=np.float64)
    mean_x = np.mean(np.stack(means_x, axis=0), axis=0)
    mean_y = float(np.mean(means_y))
    intercept = mean_y - float(mean_x @ coef)
    return LinearDecoderModel(
        start_lag=start_lag,
        end_lag=end_lag,
        method="avgcorr_lasso",
        alpha=alpha,
        coef_=coef,
        intercept_=intercept,
    )


def predict_linear_model(model: LinearDecoderModel, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_lag, y_center = _lagged_pair(x, y, model.start_lag, model.end_lag)
    pred = x_lag @ model.coef_ + model.intercept_
    return pred.astype(np.float32), y_center.astype(np.float32)


def score_linear_model(model: LinearDecoderModel, x: np.ndarray, y: np.ndarray) -> dict:
    pred, target = predict_linear_model(model, x, y)
    return {
        "prediction": pred,
        "target": target,
        "pearson": float(pearsonr(pred, target)[0]),
    }


def predict_standardized_linear_model(model: StandardizedLinearDecoderModel, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_lag, y_center = _lagged_pair(x, y, model.start_lag, model.end_lag)
    x_std = (x_lag - model.x_mean_) / np.where(model.x_scale_ == 0.0, 1.0, model.x_scale_)
    pred = x_std @ model.coef_ + model.y_mean_ + model.intercept_
    return pred.astype(np.float32), y_center.astype(np.float32)


def score_standardized_linear_model(model: StandardizedLinearDecoderModel, x: np.ndarray, y: np.ndarray) -> dict:
    pred, target = predict_standardized_linear_model(model, x, y)
    return {
        "prediction": pred,
        "target": target,
        "pearson": float(pearsonr(pred, target)[0]),
    }


def predict_forward_model(model: ForwardEncodingModel, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_lag, x_center = _lagged_forward_pair(x, y, model.start_lag, model.end_lag)
    pred = y_lag @ model.coef_.T + model.intercept_
    return pred.astype(np.float32), x_center.astype(np.float32)


def score_forward_model(model: ForwardEncodingModel, x: np.ndarray, y: np.ndarray) -> dict:
    pred, target = predict_forward_model(model, x, y)
    channel_corrs = []
    for ch in range(target.shape[1]):
        channel_corrs.append(float(pearsonr(pred[:, ch], target[:, ch])[0]))
    channel_corrs_arr = np.asarray(channel_corrs, dtype=np.float32)
    valid_corrs = channel_corrs_arr[np.isfinite(channel_corrs_arr)]
    return {
        "prediction": pred,
        "target": target,
        "channel_corrs": channel_corrs_arr,
        "mean_channel_corr": float(valid_corrs.mean()) if valid_corrs.size else float("nan"),
    }
