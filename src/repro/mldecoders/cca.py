from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class CCAModel:
    start_lag: int
    end_lag: int
    n_components: int
    x_pca_components: int
    y_pca_components: int
    recon_alpha: float
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    x_pca: PCA
    y_pca: PCA
    cca: CCA
    recon: Ridge


def fit_cca_correlation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    start_lag: int,
    end_lag: int,
    n_components_grid: list[int],
    x_pca_grid: list[int],
    y_pca_grid: list[int],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[CCAModel, dict]:
    x_train_lag, y_train_lag, y_train_center = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag, y_val_lag, y_val_center = trim_valid_range(x_val, y_val, start_lag, end_lag)

    best_model: CCAModel | None = None
    best_metrics: dict | None = None

    for x_pca_components in x_pca_grid:
        x_scaler = StandardScaler()
        x_train_std = x_scaler.fit_transform(x_train_lag)
        x_val_std = x_scaler.transform(x_val_lag)
        x_pca = PCA(n_components=min(x_pca_components, x_train_std.shape[1]), svd_solver="randomized", random_state=0)
        x_train_pca = x_pca.fit_transform(x_train_std)
        x_val_pca = x_pca.transform(x_val_std)

        for y_pca_components in y_pca_grid:
            y_scaler = StandardScaler()
            y_train_std = y_scaler.fit_transform(y_train_lag)
            y_val_std = y_scaler.transform(y_val_lag)
            y_pca = PCA(n_components=min(y_pca_components, y_train_std.shape[1]), svd_solver="randomized", random_state=0)
            y_train_pca = y_pca.fit_transform(y_train_std)
            y_val_pca = y_pca.transform(y_val_std)

            for n_components in n_components_grid:
                actual_components = min(n_components, x_train_pca.shape[1], y_train_pca.shape[1])
                cca = CCA(n_components=actual_components, max_iter=max_iter, tol=tol)
                cca.fit(x_train_pca, y_train_pca)
                u_train, v_train = cca.transform(x_train_pca, y_train_pca)
                u_val, v_val = cca.transform(x_val_pca, y_val_pca)
                train_canonical_corr = float(np.mean(_component_corrs(u_train, v_train)))
                val_canonical_corr = float(np.mean(_component_corrs(u_val, v_val)))

                if best_metrics is None or val_canonical_corr > best_metrics["val_canonical_corr"]:
                    dummy_recon = Ridge(alpha=1.0)
                    dummy_recon.fit(u_train, y_train_center)
                    val_pred = dummy_recon.predict(u_val)
                    best_model = CCAModel(
                        start_lag=start_lag,
                        end_lag=end_lag,
                        n_components=actual_components,
                        x_pca_components=x_pca.n_components_,
                        y_pca_components=y_pca.n_components_,
                        recon_alpha=1.0,
                        x_scaler=x_scaler,
                        y_scaler=y_scaler,
                        x_pca=x_pca,
                        y_pca=y_pca,
                        cca=cca,
                        recon=dummy_recon,
                    )
                    best_metrics = {
                        "train_canonical_corr": train_canonical_corr,
                        "val_canonical_corr": val_canonical_corr,
                        "val_recon_corr": float(pearsonr(val_pred, y_val_center)[0]),
                        "n_components": actual_components,
                        "x_pca_components": int(x_pca.n_components_),
                        "y_pca_components": int(y_pca.n_components_),
                        "recon_alpha": 1.0,
                    }

    if best_model is None or best_metrics is None:
        raise RuntimeError("CCA correlation search did not produce a valid model")
    return best_model, best_metrics


def build_lag_matrix(x: np.ndarray, start_lag: int, end_lag: int) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    if end_lag <= start_lag:
        raise ValueError("end_lag must be greater than start_lag")

    n_time, n_feat = x.shape
    lags = list(range(start_lag, end_lag))
    out = np.zeros((n_time, n_feat * len(lags)), dtype=np.float32)
    for idx, lag in enumerate(lags):
        src_start = max(0, -lag)
        src_end = min(n_time, n_time - lag) if lag >= 0 else n_time
        dst_start = src_start + lag if lag >= 0 else src_start + lag
        dst_end = src_end + lag if lag >= 0 else src_end + lag
        out[dst_start:dst_end, idx * n_feat : (idx + 1) * n_feat] = x[src_start:src_end]
    return out


def trim_valid_range(
    x: np.ndarray,
    y: np.ndarray,
    start_lag: int,
    end_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_left = max(0, end_lag - 1)
    max_right = max(0, -start_lag)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    if x.shape[0] <= max_left + max_right:
        raise ValueError("signal too short for requested lag range")

    valid = slice(max_left, x.shape[0] - max_right)
    y_center = y[valid]
    x_lag = build_lag_matrix(x, start_lag=start_lag, end_lag=end_lag)[valid]
    y_lag = build_lag_matrix(y[:, None], start_lag=start_lag, end_lag=end_lag)[valid]
    return x_lag, y_lag, y_center


def _component_corrs(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    corrs = []
    for idx in range(u.shape[1]):
        corrs.append(float(pearsonr(u[:, idx], v[:, idx])[0]))
    return np.asarray(corrs, dtype=np.float64)


def fit_cca_reconstruction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    start_lag: int,
    end_lag: int,
    n_components_grid: list[int],
    x_pca_grid: list[int],
    y_pca_grid: list[int],
    alpha_grid: list[float],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[CCAModel, dict]:
    x_train_lag, y_train_lag, y_train_center = trim_valid_range(x_train, y_train, start_lag, end_lag)
    x_val_lag, y_val_lag, y_val_center = trim_valid_range(x_val, y_val, start_lag, end_lag)

    best_model: CCAModel | None = None
    best_metrics: dict | None = None

    for x_pca_components in x_pca_grid:
        x_scaler = StandardScaler()
        x_train_std = x_scaler.fit_transform(x_train_lag)
        x_val_std = x_scaler.transform(x_val_lag)

        x_pca = PCA(n_components=min(x_pca_components, x_train_std.shape[1]), svd_solver="randomized", random_state=0)
        x_train_pca = x_pca.fit_transform(x_train_std)
        x_val_pca = x_pca.transform(x_val_std)

        for y_pca_components in y_pca_grid:
            y_scaler = StandardScaler()
            y_train_std = y_scaler.fit_transform(y_train_lag)
            y_val_std = y_scaler.transform(y_val_lag)

            y_pca = PCA(n_components=min(y_pca_components, y_train_std.shape[1]), svd_solver="randomized", random_state=0)
            y_train_pca = y_pca.fit_transform(y_train_std)
            y_val_pca = y_pca.transform(y_val_std)

            for n_components in n_components_grid:
                actual_components = min(n_components, x_train_pca.shape[1], y_train_pca.shape[1])
                cca = CCA(n_components=actual_components, max_iter=max_iter, tol=tol)
                cca.fit(x_train_pca, y_train_pca)

                u_train, v_train = cca.transform(x_train_pca, y_train_pca)
                u_val, v_val = cca.transform(x_val_pca, y_val_pca)
                train_canonical_corr = float(np.mean(_component_corrs(u_train, v_train)))
                val_canonical_corr = float(np.mean(_component_corrs(u_val, v_val)))

                for alpha in alpha_grid:
                    recon = Ridge(alpha=alpha)
                    recon.fit(u_train, y_train_center)
                    val_pred = recon.predict(u_val)
                    val_recon_corr = float(pearsonr(val_pred, y_val_center)[0])

                    if best_metrics is None or val_recon_corr > best_metrics["val_recon_corr"]:
                        best_model = CCAModel(
                            start_lag=start_lag,
                            end_lag=end_lag,
                            n_components=actual_components,
                            x_pca_components=x_pca.n_components_,
                            y_pca_components=y_pca.n_components_,
                            recon_alpha=float(alpha),
                            x_scaler=x_scaler,
                            y_scaler=y_scaler,
                            x_pca=x_pca,
                            y_pca=y_pca,
                            cca=cca,
                            recon=recon,
                        )
                        best_metrics = {
                            "train_canonical_corr": train_canonical_corr,
                            "val_canonical_corr": val_canonical_corr,
                            "val_recon_corr": val_recon_corr,
                            "n_components": actual_components,
                            "x_pca_components": int(x_pca.n_components_),
                            "y_pca_components": int(y_pca.n_components_),
                            "recon_alpha": float(alpha),
                        }

    if best_model is None or best_metrics is None:
        raise RuntimeError("CCA model search did not produce a valid model")
    return best_model, best_metrics


def transform_pair(model: CCAModel, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_lag, y_lag, y_center = trim_valid_range(x, y, model.start_lag, model.end_lag)
    x_std = model.x_scaler.transform(x_lag)
    y_std = model.y_scaler.transform(y_lag)
    x_pca = model.x_pca.transform(x_std)
    y_pca = model.y_pca.transform(y_std)
    u, v = model.cca.transform(x_pca, y_pca)
    return u, v, y_center


def score_reconstruction(model: CCAModel, x: np.ndarray, y: np.ndarray) -> dict:
    u, v, y_center = transform_pair(model, x, y)
    pred = model.recon.predict(u)
    return {
        "canonical_corr": float(np.mean(_component_corrs(u, v))),
        "recon_corr": float(pearsonr(pred, y_center)[0]),
        "prediction": pred.astype(np.float32),
        "target": y_center.astype(np.float32),
    }


def _window_scores(u: np.ndarray, v: np.ndarray, window: int, stride: int) -> np.ndarray:
    scores = []
    for start in range(0, u.shape[0] - window + 1, stride):
        stop = start + window
        comp_scores = _component_corrs(u[start:stop], v[start:stop])
        scores.append(float(np.mean(comp_scores)))
    return np.asarray(scores, dtype=np.float32)


def score_match_mismatch(
    model: CCAModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int,
    stride: int,
    mismatch_shift: int,
) -> dict:
    u, v, _ = transform_pair(model, x, y)
    if mismatch_shift <= 0:
        raise ValueError("mismatch_shift must be positive")
    if v.shape[0] < window + mismatch_shift:
        raise ValueError("signal too short for requested match/mismatch settings")

    matched = []
    mismatched = []
    for start in range(0, u.shape[0] - window - mismatch_shift + 1, stride):
        stop = start + window
        matched.append(float(np.mean(_component_corrs(u[start:stop], v[start:stop]))))
        mismatch_start = start + mismatch_shift
        mismatch_stop = mismatch_start + window
        mismatched.append(float(np.mean(_component_corrs(u[start:stop], v[mismatch_start:mismatch_stop]))))

    matched_arr = np.asarray(matched, dtype=np.float32)
    mismatched_arr = np.asarray(mismatched, dtype=np.float32)
    return {
        "match_scores": matched_arr,
        "mismatch_scores": mismatched_arr,
        "accuracy": float(np.mean(matched_arr > mismatched_arr)),
        "margin_mean": float(np.mean(matched_arr - mismatched_arr)),
    }
