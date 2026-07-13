from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SHARED_UPSTREAM = Path(ROOT.drive + "\\decode") / "external" / "upstream" / "mldecoders"
if SHARED_UPSTREAM.exists() and str(SHARED_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(SHARED_UPSTREAM))
HAPPYQUOKKA_UPSTREAM = Path(ROOT.drive + "\\decode") / "external" / "upstream" / "HappyQuokka_system_for_EEG_Challenge"
if HAPPYQUOKKA_UPSTREAM.exists() and str(HAPPYQUOKKA_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(HAPPYQUOKKA_UPSTREAM))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow
from benchmark.scoring import pearson_on_valid, subject_metric_rows
from repro.adt_exact import ADTExactRegressor
from repro.mldecoders.models import EEGNetRegressor, VLAAIExactOfficialRegressor
from repro.simple_models import FCNNBaseline

try:
    from pipeline.dnn import CNN as UpstreamCNN
except Exception:
    UpstreamCNN = None

try:
    from models.FFT_block import Decoder as HappyQuokkaDecoder  # type: ignore
except Exception:
    HappyQuokkaDecoder = None


RUN_LABELS = [
    "task-audiobook1_run-01",
    "task-audiobook1_run-02",
    "task-audiobook2_run-01",
    "task-audiobook2_run-02",
]
MODEL_SET = ["linear", "ridge", "lasso", "elasticnet", "cca", "fcnn", "cnn", "eegnet", "adt", "vlaai", "happyquokka"]
EXCLUDED_MODELS = ["dnn"]
BANNED_SUFFIXES = {".mat", ".fif", ".pt", ".pth", ".npy", ".npz", ".h5", ".ckpt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-gate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--generate-all-eligible-config", action="store_true")
    parser.add_argument("--generate-all-canonicalized-config", action="store_true")
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    current = csv_rows(path)
    current.extend([{k: row.get(k, "") for k in fieldnames} for row in rows])
    write_csv(path, current, fieldnames)


def filter_csv_by_job(path: Path, job_key: str, fieldnames: list[str]) -> None:
    if not path.exists():
        return
    rows = [row for row in csv_rows(path) if row.get("job_key", "") != job_key]
    write_csv(path, rows, fieldnames)


def atomic_write_run_lists(output_dir: Path, completed: list[dict[str, Any]], failures: list[dict[str, Any]], entries: list[dict[str, Any]], planned: list[str]) -> None:
    completed_unique = {row["job_key"]: row for row in completed}
    failure_unique = {row["job_key"]: row for row in failures}
    entry_unique = {row["job_key"]: row for row in entries}
    write_json(output_dir / "completed_jobs.json", {"completed_jobs": list(completed_unique.values())})
    write_json(output_dir / "failure_report.json", {"failures": list(failure_unique.values())})
    write_json(output_dir / "model_run_entries.json", list(entry_unique.values()))
    done = set(completed_unique) | set(failure_unique)
    write_json(
        output_dir / "run_state.json",
        {
            "planned_jobs": planned,
            "completed_or_failed": sorted(done),
            "pending_jobs": [job for job in planned if job not in done],
            "ready_for_full_manual_run": False,
        },
    )


def decode_matlab_string(f: h5py.File, ref: Any) -> str:
    return "".join(chr(int(x)) for x in np.array(f[ref]).squeeze())


def deref_cell(f: h5py.File, dataset: h5py.Dataset) -> list[Any]:
    return [f[ref] for ref in np.array(dataset).reshape(-1)]


def subject_from_path(path: Path, fallback: str) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return fallback


def load_paired_mat(path: str | Path, subject_id: str) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as f:
        results = f["results"]
        fs = int(np.array(results["epochs_neuro"]["fsample"]).squeeze())
        labels = [decode_matlab_string(f, ref) for ref in np.array(results["epochs_neuro"]["label"]).reshape(-1)]
        meg_trials = []
        for ds in deref_cell(f, results["epochs_neuro"]["trial"]):
            arr = np.array(ds, dtype=np.float32)
            if arr.shape == (306, 7680):
                arr = arr.T
            if arr.shape != (7680, 306):
                raise RuntimeError(f"unexpected MEG trial shape {arr.shape}")
            meg_trials.append(arr)
        targets = [np.array(ds, dtype=np.float32).reshape(-1) for ds in deref_cell(f, results["epochs_audio"])]
    if len(meg_trials) != len(targets):
        raise RuntimeError("MEG/audio trial count mismatch")
    trials = []
    for idx, (x, y) in enumerate(zip(meg_trials, targets), start=1):
        run_idx = (idx - 1) // 4
        local_idx = (idx - 1) % 4 + 1
        trials.append(
            {
                "global_trial_index": idx,
                "run_label": RUN_LABELS[run_idx] if run_idx < len(RUN_LABELS) else f"run-{run_idx + 1:02d}",
                "trial_index_in_run": local_idx,
                "recording_id": f"{subject_id}_{RUN_LABELS[run_idx] if run_idx < len(RUN_LABELS) else f'run-{run_idx + 1:02d}'}_trial-{local_idx:02d}",
                "x_all306": x,
                "target": y.astype(np.float32),
            }
        )
    return {"fs": fs, "labels": labels, "trials": trials, "subject_id": subject_id}


def subject_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "subjects" in config:
        return config["subjects"]
    return [
        {
            "subject_id": config["subject_ids"][0],
            "dataset_id": config["dataset_id"],
            "paired_mat": config["paired_mat"],
            "envelope_mat": config["envelope_mat"],
            "expected_paired_sha256": config["expected_paired_sha256"],
            "expected_envelope_sha256": config["expected_envelope_sha256"],
            "split": config["split"],
        }
    ]


def validate_and_select(data: dict[str, Any], config: dict[str, Any], subject_cfg: dict[str, Any], output_dir: Path) -> None:
    inventory = list(csv.DictReader(open(ROOT / config["channel_inventory"], newline="", encoding="utf-8")))
    mag_selection = read_json(ROOT / config["mag102_selection"])
    labels = data["labels"]
    if [row["channel_label"] for row in inventory] != labels:
        raise RuntimeError("all306 labels do not match channel inventory")
    mag_indices = [int(v) - 1 for v in mag_selection["source_indices_1based"]]
    mag_labels = [labels[i] for i in mag_indices]
    mag_hash = hashlib.sha256("\n".join(mag_labels).encode("utf-8")).hexdigest()
    if mag_hash != config["expected_mag102_channel_order_sha256"]:
        raise RuntimeError("mag102 channel hash mismatch")
    paired_sha = sha256_file(subject_cfg["paired_mat"])
    envelope_sha = sha256_file(subject_cfg["envelope_mat"])
    if paired_sha != subject_cfg["expected_paired_sha256"] or envelope_sha != subject_cfg["expected_envelope_sha256"]:
        raise RuntimeError(f"MAT SHA256 mismatch for {subject_cfg['subject_id']}")
    if data["fs"] != 64 or len(data["trials"]) != 16:
        raise RuntimeError(f"bad fs/trial count for {subject_cfg['subject_id']}")
    for trial in data["trials"]:
        x, y = trial["x_all306"], trial["target"]
        if x.shape != (7680, 306) or y.shape != (7680,):
            raise RuntimeError(f"bad shape for {trial['recording_id']}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError(f"NaN/Inf in {trial['recording_id']}")
        trial["x_mag102"] = x[:, mag_indices]
    payload_path = output_dir / "data_integrity_check.json"
    current = read_json(payload_path) if payload_path.exists() else {"subjects": []}
    current["subjects"] = [row for row in current.get("subjects", []) if row.get("subject_id") != subject_cfg["subject_id"]]
    current["subjects"].append(
        {
            "status": "passed",
            "subject_id": subject_cfg["subject_id"],
            "dataset_id": subject_cfg["dataset_id"],
            "paired_sha256": paired_sha,
            "envelope_sha256": envelope_sha,
            "n_trials": 16,
            "fs": 64,
            "representation": "mag102",
            "mag102_channel_order_sha256": mag_hash,
            "channel_count": 102,
        }
    )
    current["status"] = "passed"
    write_json(payload_path, current)


def split_trials(data: dict[str, Any], subject_cfg: dict[str, Any], output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    lookup = {t["global_trial_index"]: t for t in data["trials"]}
    split = subject_cfg["split"]
    splits = {
        "train": [lookup[i] for i in split["train_global_trial_index"]],
        "val": [lookup[i] for i in split["val_global_trial_index"]],
        "test": [lookup[i] for i in split["test_global_trial_index"]],
    }
    rows = []
    seen = set()
    for role, trials in splits.items():
        for trial in trials:
            idx = trial["global_trial_index"]
            if idx in seen:
                raise RuntimeError("split overlap")
            seen.add(idx)
            rows.append(
                {
                    "subject_id": subject_cfg["subject_id"],
                    "role": role,
                    "global_trial_index": idx,
                    "recording_id": trial["recording_id"],
                    "run_label": trial["run_label"],
                }
            )
    if sorted(seen) != list(range(1, 17)):
        raise RuntimeError("split does not cover trials 1..16")
    path = output_dir / "split_manifest.json"
    current = read_json(path) if path.exists() else {"rows": []}
    current["rows"] = [row for row in current.get("rows", []) if row.get("subject_id") != subject_cfg["subject_id"]]
    current["rows"].extend(rows)
    current["identity"] = split["identity"]
    current["recording_level"] = True
    write_json(path, current)
    return splits


def write_model_lock(config: dict[str, Any], output_dir: Path) -> None:
    if config["models"] != MODEL_SET or len(config["models"]) != 11:
        raise RuntimeError("model set is not exactly the locked 11-model roster")
    source_cfg = read_json(config["model_lock_source"]["config_path"])
    source_audit = read_json(config["model_lock_source"]["audit_manifest_path"])
    write_json(
        output_dir / "meg_model_set_lock.json",
        {
            "status": "locked",
            "models": config["models"],
            "model_count": len(config["models"]),
            "excluded_models": config.get("excluded_models", EXCLUDED_MODELS),
            "source_config_path": config["model_lock_source"]["config_path"],
            "source_config_commit": config["model_lock_source"]["source_commit"],
            "source_config_models": source_cfg["models"],
            "audit_manifest_path": config["model_lock_source"]["audit_manifest_path"],
            "audit_identity_conclusion": source_audit.get("identity_conclusion"),
            "audit_roster_decision": source_audit.get("roster_decision"),
        },
    )


def fit_scalers(train: list[dict[str, Any]]) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler().fit(np.concatenate([t["x_mag102"] for t in train], axis=0))
    y_scaler = StandardScaler().fit(np.concatenate([t["target"][:, None] for t in train], axis=0))
    return x_scaler, y_scaler


def lag_sparse(x: np.ndarray, start_lag: int, end_lag: int) -> sparse.csr_matrix:
    valid_start = end_lag - 1
    return sparse.hstack([sparse.csr_matrix(x[valid_start - lag : x.shape[0] - lag]) for lag in range(start_lag, end_lag)], format="csr")


def lag_dense(x: np.ndarray, start_lag: int, end_lag: int) -> np.ndarray:
    return lag_sparse(x, start_lag, end_lag).toarray().astype(np.float32)


def linear_xy_for_trials(
    trials: list[dict[str, Any]],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    start_lag: int,
    end_lag: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    xs, ys = [], []
    valid_start = end_lag - 1
    for trial in trials:
        x = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
        y = y_scaler.transform(trial["target"][:, None]).reshape(-1).astype(np.float32)
        xs.append(lag_sparse(x, start_lag, end_lag))
        ys.append(y[valid_start:])
    return sparse.vstack(xs, format="csr"), np.concatenate(ys)


def cca_xy_for_trials(
    trials: list[dict[str, Any]],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    start_lag: int,
    end_lag: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    xs, y_lags, y_centers = [], [], []
    valid_start = end_lag - 1
    for trial in trials:
        x = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
        y = y_scaler.transform(trial["target"][:, None]).reshape(-1).astype(np.float32)
        xs.append(lag_sparse(x, start_lag, end_lag))
        y_lags.append(lag_sparse(y[:, None], start_lag, end_lag))
        y_centers.append(y[valid_start:])
    return sparse.vstack(xs, format="csr"), sparse.vstack(y_lags, format="csr"), np.concatenate(y_centers)


def bounded_rows(x: Any, y: np.ndarray, limit: int | None, seed: int) -> tuple[Any, np.ndarray]:
    if limit is None or limit <= 0 or x.shape[0] <= limit:
        return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(x.shape[0], size=limit, replace=False))
    return x[idx], y[idx]


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    return pearson_on_valid(a[mask], b[mask], np.ones(int(mask.sum()), dtype=np.int32))


def make_recording_row(config: dict[str, Any], subject_cfg: dict[str, Any], model: str, recording_id: str, checkpoint_id: str, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> RecordingMetricRow:
    return RecordingMetricRow(
        dataset=subject_cfg["dataset_id"],
        model=model,
        task="audiobook_envelope_reconstruction",
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=subject_cfg["subject_id"],
        recording_id=recording_id,
        sampling_rate=64,
        metric_name="pearson_r",
        metric_value=pearson_on_valid(pred, target, mask),
        num_valid_samples=int(mask.astype(bool).sum()),
        checkpoint_id=checkpoint_id,
        artifact_scope=config["artifact_scope"],
    )


def append_log(output_dir: Path, model_name: str, message: str) -> None:
    with open(output_dir / f"{model_name}.log", "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def validation_rows_for_linear(model: Any, trials: list[dict[str, Any]], x_scaler: StandardScaler, y_scaler: StandardScaler, cfg: dict[str, Any]) -> tuple[list[float], np.ndarray, np.ndarray]:
    valid_start = int(cfg["end_lag"]) - 1
    scores, pred_cat, target_cat = [], [], []
    for trial in trials:
        x = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
        pred = y_scaler.inverse_transform(model.predict(lag_sparse(x, cfg["start_lag"], cfg["end_lag"])).reshape(-1, 1)).reshape(-1).astype(np.float32)
        target = trial["target"][valid_start:].astype(np.float32)
        scores.append(safe_pearson(pred, target))
        pred_cat.append(pred)
        target_cat.append(target)
    return scores, np.concatenate(pred_cat), np.concatenate(target_cat)


def run_linear_family(model_name: str, config: dict[str, Any], subject_cfg: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[list[RecordingMetricRow], dict[str, Any]]:
    cfg = config["linear_family"]
    x_scaler, y_scaler = fit_scalers(splits["train"])
    x_train, y_train = linear_xy_for_trials(splits["train"], x_scaler, y_scaler, cfg["start_lag"], cfg["end_lag"])
    x_fit, y_fit = bounded_rows(x_train, y_train, cfg.get("max_fit_samples_per_split"), int(config["seed"]))
    if model_name == "linear":
        candidates = [("linear_ols", LinearRegression())]
    elif model_name == "ridge":
        candidates = [(f"ridge_alpha_{a}", Ridge(alpha=float(a), solver="lsqr", tol=0.01, max_iter=80)) for a in cfg["ridge_alphas"]]
    elif model_name == "lasso":
        candidates = [(f"lasso_alpha_{a}", Lasso(alpha=float(a), max_iter=int(cfg["max_iter"]))) for a in cfg["lasso_alphas"]]
    elif model_name == "elasticnet":
        candidates = [
            (f"elasticnet_alpha_{a}_l1_{r}", ElasticNet(alpha=float(a), l1_ratio=float(r), max_iter=int(cfg["max_iter"])))
            for a in cfg["elasticnet_alphas"]
            for r in cfg["elasticnet_l1_ratios"]
        ]
    else:
        raise RuntimeError(model_name)
    diagnostics = []
    best = None
    for checkpoint, model in candidates:
        model.fit(x_fit, y_fit)
        val_scores, val_pred, val_target = validation_rows_for_linear(model, splits["val"], x_scaler, y_scaler, cfg)
        mean_val = float(np.nanmean(val_scores))
        legacy = safe_pearson(val_pred, val_target)
        diagnostics.append(
            {
                "job_key": f"{subject_cfg['subject_id']}:{model_name}:seed{config['seed']}",
                "model": model_name,
                "implementation_identity": "benchmark_local_elasticnet" if model_name == "elasticnet" else f"recording_safe_{model_name}_adapter",
                "candidate": checkpoint,
                "val_recording_pearson": json.dumps(val_scores),
                "val_mean_recording_pearson_r": mean_val,
                "legacy_concatenated_val_pearson_r": legacy,
            }
        )
        if best is None or mean_val > best["val_mean"]:
            best = {"checkpoint": checkpoint, "model": model, "val_mean": mean_val, "legacy": legacy}
    assert best is not None
    append_csv(
        output_dir / "validation_diagnostics.csv",
        diagnostics,
        ["job_key", "model", "implementation_identity", "candidate", "val_recording_pearson", "val_mean_recording_pearson_r", "legacy_concatenated_val_pearson_r"],
    )
    rows = []
    valid_start = int(cfg["end_lag"]) - 1
    for trial in splits["test"]:
        x = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
        pred = y_scaler.inverse_transform(best["model"].predict(lag_sparse(x, cfg["start_lag"], cfg["end_lag"])).reshape(-1, 1)).reshape(-1).astype(np.float32)
        full = np.zeros(7680, dtype=np.float32)
        mask = np.zeros(7680, dtype=np.int32)
        full[valid_start:] = pred
        mask[valid_start:] = 1
        rows.append(make_recording_row(config, subject_cfg, model_name, trial["recording_id"], best["checkpoint"], full, trial["target"], mask))
    return rows, {
        "checkpoint_id": best["checkpoint"],
        "best_val_score": best["val_mean"],
        "legacy_concatenated_val_pearson_r": best["legacy"],
        "implementation_identity": "benchmark_local_elasticnet" if model_name == "elasticnet" else f"recording_safe_{model_name}_adapter",
        "scaler_fit_scope": "train_recordings_only",
    }


def cca_candidate_predictions(model: dict[str, Any], trials: list[dict[str, Any]], x_scaler: StandardScaler, y_scaler: StandardScaler, cfg: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]], list[float], np.ndarray, np.ndarray]:
    out = []
    scores, pred_cat, target_cat = [], [], []
    valid_start = int(cfg["end_lag"]) - 1
    for trial in trials:
        x = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
        x_lag = lag_dense(x, cfg["start_lag"], cfg["end_lag"])
        x_pca = model["x_pca"].transform(model["x_std"].transform(x_lag))
        u = model["cca"].transform(x_pca)
        pred_scaled = model["recon"].predict(u)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1).astype(np.float32)
        target = trial["target"][valid_start:].astype(np.float32)
        full = np.zeros(7680, dtype=np.float32)
        mask = np.zeros(7680, dtype=np.int32)
        full[valid_start:] = pred
        mask[valid_start:] = 1
        out.append((trial, full, trial["target"], mask))
        scores.append(safe_pearson(pred, target))
        pred_cat.append(pred)
        target_cat.append(target)
    return out, scores, np.concatenate(pred_cat), np.concatenate(target_cat)


def run_cca(config: dict[str, Any], subject_cfg: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[list[RecordingMetricRow], dict[str, Any]]:
    cfg = config["cca"]
    x_scaler, y_scaler = fit_scalers(splits["train"])
    x_train, y_train_lag, y_train_center = cca_xy_for_trials(splits["train"], x_scaler, y_scaler, cfg["start_lag"], cfg["end_lag"])
    x_fit, y_center_fit = bounded_rows(x_train, y_train_center, cfg.get("max_fit_samples"), int(config["seed"]))
    if cfg.get("max_fit_samples") and x_train.shape[0] > int(cfg["max_fit_samples"]):
        rng = np.random.default_rng(int(config["seed"]))
        idx = np.sort(rng.choice(x_train.shape[0], size=int(cfg["max_fit_samples"]), replace=False))
        y_lag_fit = y_train_lag[idx]
    else:
        y_lag_fit = y_train_lag
    y_lag_fit = y_lag_fit.toarray() if sparse.issparse(y_lag_fit) else y_lag_fit
    diagnostics = []
    best = None
    for x_pca_n in cfg["x_pca_grid"]:
        for y_pca_n in cfg["y_pca_grid"]:
            x_std = StandardScaler()
            y_std = StandardScaler()
            x_fit_std = x_std.fit_transform(x_fit.toarray() if sparse.issparse(x_fit) else x_fit)
            y_fit_std = y_std.fit_transform(y_lag_fit)
            x_pca = PCA(n_components=min(int(x_pca_n), x_fit_std.shape[1]), svd_solver="randomized", random_state=int(config["seed"]))
            y_pca = PCA(n_components=min(int(y_pca_n), y_fit_std.shape[1]), svd_solver="randomized", random_state=int(config["seed"]))
            x_fit_pca = x_pca.fit_transform(x_fit_std)
            y_fit_pca = y_pca.fit_transform(y_fit_std)
            for n_comp in cfg["n_components_grid"]:
                actual = min(int(n_comp), x_fit_pca.shape[1], y_fit_pca.shape[1])
                cca = CCA(n_components=actual, max_iter=int(cfg["max_iter"]), tol=float(cfg["tol"]))
                cca.fit(x_fit_pca, y_fit_pca)
                u_fit, _ = cca.transform(x_fit_pca, y_fit_pca)
                for alpha in cfg["alpha_grid"]:
                    recon = Ridge(alpha=float(alpha))
                    recon.fit(u_fit, y_center_fit)
                    candidate = {
                        "x_std": x_std,
                        "y_std": y_std,
                        "x_pca": x_pca,
                        "y_pca": y_pca,
                        "cca": cca,
                        "recon": recon,
                    }
                    _, val_scores, val_pred, val_target = cca_candidate_predictions(candidate, splits["val"], x_scaler, y_scaler, cfg)
                    mean_val = float(np.nanmean(val_scores))
                    legacy = safe_pearson(val_pred, val_target)
                    checkpoint = f"cca_xpca_{x_pca.n_components_}_ypca_{y_pca.n_components_}_nc_{actual}_alpha_{alpha}"
                    diagnostics.append(
                        {
                            "job_key": f"{subject_cfg['subject_id']}:cca:seed{config['seed']}",
                            "model": "cca",
                            "implementation_identity": "recording_safe_cca_meg_adapter",
                            "candidate": checkpoint,
                            "val_recording_pearson": json.dumps(val_scores),
                            "val_mean_recording_pearson_r": mean_val,
                            "legacy_concatenated_val_pearson_r": legacy,
                        }
                    )
                    if best is None or mean_val > best["val_mean"]:
                        best = {"checkpoint": checkpoint, "model": candidate, "val_mean": mean_val, "legacy": legacy}
    if best is None:
        raise RuntimeError("CCA model search did not produce a valid model")
    append_csv(
        output_dir / "validation_diagnostics.csv",
        diagnostics,
        ["job_key", "model", "implementation_identity", "candidate", "val_recording_pearson", "val_mean_recording_pearson_r", "legacy_concatenated_val_pearson_r"],
    )
    test_preds, _, _, _ = cca_candidate_predictions(best["model"], splits["test"], x_scaler, y_scaler, cfg)
    rows = [make_recording_row(config, subject_cfg, "cca", trial["recording_id"], best["checkpoint"], pred, target, mask) for trial, pred, target, mask in test_preds]
    return rows, {
        "checkpoint_id": best["checkpoint"],
        "best_val_score": best["val_mean"],
        "legacy_concatenated_val_pearson_r": best["legacy"],
        "implementation_identity": "recording_safe_cca_meg_adapter",
        "metric_definition": "reconstruction_y_hat_pearson_not_canonical_correlation",
        "scaler_fit_scope": "train_recordings_only",
        "cca_inference_uses_target": False,
    }


class WindowDataset:
    def __init__(self, trials: list[dict[str, Any]], window_size: int, hop: int, *, layout: str, target_mode: str, max_items: int | None = None) -> None:
        self.trials = trials
        self.window_size = int(window_size)
        self.layout = layout
        self.target_mode = target_mode
        self.index: list[tuple[int, int]] = []
        for trial_idx, trial in enumerate(trials):
            n = int(trial["x"].shape[0])
            for start in range(0, n - self.window_size + 1, int(hop)):
                self.index.append((trial_idx, start))
        if max_items is not None and max_items > 0:
            self.index = self.index[:max_items]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        trial_idx, start = self.index[idx]
        stop = start + self.window_size
        trial = self.trials[trial_idx]
        x = trial["x"][start:stop]
        y = trial["y"][start:stop]
        if self.layout == "channels_time":
            x = x.T
        elif self.layout == "time_channels":
            pass
        else:
            raise RuntimeError(f"bad layout {self.layout}")
        if self.target_mode == "last":
            target = np.asarray(y[-1], dtype=np.float32)
        elif self.target_mode == "sequence":
            target = y[:, None].astype(np.float32)
        else:
            raise RuntimeError(f"bad target mode {self.target_mode}")
        return x.astype(np.float32), target, np.asarray(0, dtype=np.int64)


def make_scaled_cache(splits: dict[str, list[dict[str, Any]]], x_scaler: StandardScaler, y_scaler: StandardScaler) -> dict[str, list[dict[str, Any]]]:
    cached: dict[str, list[dict[str, Any]]] = {}
    for role, trials in splits.items():
        cached[role] = []
        for trial in trials:
            item = {k: v for k, v in trial.items() if k not in {"x_all306", "x_mag102", "target"}}
            item["x"] = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
            item["y"] = y_scaler.transform(trial["target"][:, None]).reshape(-1).astype(np.float32)
            item["target_raw"] = trial["target"].astype(np.float32)
            cached[role].append(item)
    return cached


class VLAAIMEG102Adapter:
    pass


def build_torch_model(model_name: str, config: dict[str, Any]):
    import torch
    from torch import nn

    if model_name == "fcnn":
        return FCNNBaseline(num_input_channels=102, input_length=int(config["window_models"]["fcnn"]["window_size"]))
    if model_name == "cnn":
        if UpstreamCNN is None:
            raise RuntimeError("upstream CNN import unavailable")
        return UpstreamCNN(num_input_channels=102, input_length=int(config["window_models"]["cnn"]["window_size"]))
    if model_name == "eegnet":
        return EEGNetRegressor(num_input_channels=102, input_length=int(config["window_models"]["eegnet"]["window_size"]))
    if model_name == "adt":
        return ADTExactRegressor(chans=102, seq_len=int(config["adt"]["window_length"]))
    if model_name == "vlaai":
        class _VLAAIMEG102Adapter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.meg102_to_vlaai64 = nn.Linear(102, 64)
                self.vlaai64 = VLAAIExactOfficialRegressor(num_input_channels=64, input_length=int(config["vlaai"]["window_size"]))

            def forward(self, x: Any) -> Any:
                x = self.meg102_to_vlaai64(x.transpose(1, 2)).transpose(1, 2)
                return self.vlaai64(x)

        return _VLAAIMEG102Adapter()
    if model_name == "happyquokka":
        if HappyQuokkaDecoder is None:
            raise RuntimeError("HappyQuokka Decoder import unavailable")
        class _HappyQuokkaMEG102Adapter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.meg102_to_hq64 = nn.Linear(102, 64)
                self.decoder = HappyQuokkaDecoder(
                    in_channel=64,
                    d_model=int(config["happyquokka"]["d_model"]),
                    d_inner=int(config["happyquokka"]["d_inner"]),
                    n_head=int(config["happyquokka"]["n_head"]),
                    n_layers=int(config["happyquokka"]["n_layers"]),
                    fft_conv1d_kernel=(9, 1),
                    fft_conv1d_padding=(4, 0),
                    dropout=float(config["happyquokka"]["dropout"]),
                    g_con=bool(config["happyquokka"]["g_con"]),
                    within_sub_num=1,
                )

            def forward(self, x: Any) -> Any:
                x = self.meg102_to_hq64(x)
                sub_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                return self.decoder(x, sub_id)

        return _HappyQuokkaMEG102Adapter()
    raise RuntimeError(f"no torch model builder for {model_name}")


def torch_model_contract(model_name: str, config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    if model_name in {"fcnn", "cnn", "eegnet"}:
        cfg = config["window_models"][model_name]
        return {
            "window_size": int(cfg["window_size"]),
            "hop": 1,
            "layout": "channels_time",
            "target_mode": "last",
            "batch_size": int(cfg["batch_size"]),
            "max_epochs": int(config["smoke"]["epochs"] if smoke else cfg["max_epochs"]),
            "patience": int(cfg["early_stopping_patience"]),
            "learning_rate": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "device": cfg["device"],
            "max_train_windows": int(config["smoke"]["max_train_windows"]) if smoke else None,
            "optimizer": cfg["optimizer"],
            "loss": cfg["loss"],
            "scheduler": cfg["scheduler"],
            "min_lr": None,
        }
    if model_name == "adt":
        cfg = config["adt"]
        return {
            "window_size": int(cfg["window_length"]),
            "hop": int(cfg["hop_length"]),
            "layout": "time_channels",
            "target_mode": "sequence",
            "batch_size": int(cfg["batch_size"]),
            "max_epochs": int(config["smoke"]["epochs"] if smoke else cfg["max_epochs"]),
            "patience": int(cfg["early_stopping_patience"]),
            "learning_rate": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "device": cfg["device"],
            "max_train_windows": int(config["smoke"]["max_train_windows"]) if smoke else None,
            "optimizer": cfg["optimizer"],
            "loss": cfg["loss"],
            "scheduler": cfg["scheduler"],
            "min_lr": float(cfg["min_lr"]),
        }
    if model_name == "vlaai":
        cfg = config["vlaai"]
        return {
            "window_size": int(cfg["window_size"]),
            "hop": int(cfg["hop"]),
            "layout": "channels_time",
            "target_mode": "last",
            "batch_size": int(cfg["batch_size"]),
            "max_epochs": int(config["smoke"]["epochs"] if smoke else cfg["max_epochs"]),
            "patience": int(cfg["early_stopping_patience"]),
            "learning_rate": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "device": cfg["device"],
            "max_train_windows": int(config["smoke"]["max_train_windows"]) if smoke else None,
            "optimizer": cfg["optimizer"],
            "loss": cfg["loss"],
            "scheduler": cfg["scheduler"],
            "min_lr": float(cfg["min_lr"]),
        }
    if model_name == "happyquokka":
        cfg = config["happyquokka"]
        return {
            "window_size": int(cfg["chunk_length"]),
            "hop": int(cfg["chunk_length"]),
            "layout": "time_channels",
            "target_mode": "sequence",
            "batch_size": int(cfg["batch_size"]),
            "max_epochs": int(config["smoke"]["epochs"] if smoke else cfg["max_epochs"]),
            "patience": int(cfg["early_stopping_patience"]),
            "learning_rate": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "device": cfg["device"],
            "max_train_windows": int(config["smoke"]["max_train_windows"]) if smoke else None,
            "optimizer": cfg["optimizer"],
            "loss": cfg["loss"],
            "scheduler": cfg["scheduler"],
            "min_lr": None,
            "lambda_l1": float(cfg["lambda_l1"]),
        }
    raise RuntimeError(model_name)


def torch_correlation(y_true: Any, y_pred: Any, eps: float = 1e-8) -> Any:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = (y_true.square().sum() * y_pred.square().sum()).sqrt().clamp_min(eps)
    return (y_true * y_pred).sum() / denom


def sequence_pearson_loss(y_true: Any, y_pred: Any, eps: float = 1e-8) -> Any:
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.reshape_as(y_true)
    yt = y_true - y_true.mean(dim=1, keepdim=True)
    yp = y_pred - y_pred.mean(dim=1, keepdim=True)
    denom = (yt.square().sum(dim=1, keepdim=True) * yp.square().sum(dim=1, keepdim=True)).sqrt().clamp_min(eps)
    return -((yt * yp).sum(dim=1, keepdim=True) / denom).mean()


def happyquokka_loss(y_true: Any, y_pred: Any, lambda_l1: float) -> Any:
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.reshape_as(y_true)
    pearson = sequence_pearson_loss(y_true, y_pred)
    l1 = (y_pred - y_true).abs().mean(dim=1).mean()
    return pearson + float(lambda_l1) * l1


def make_optimizer(model_name: str, model: Any, contract: dict[str, Any]):
    import torch

    if contract["optimizer"] == "NAdam":
        return torch.optim.NAdam(model.parameters(), lr=contract["learning_rate"], weight_decay=contract["weight_decay"])
    if contract["optimizer"] == "Adam":
        if model_name == "happyquokka":
            return torch.optim.Adam(model.parameters(), lr=contract["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=contract["weight_decay"])
        return torch.optim.Adam(model.parameters(), lr=contract["learning_rate"], weight_decay=contract["weight_decay"])
    raise RuntimeError(f"unsupported optimizer {contract['optimizer']}")


def make_scheduler(optimizer: Any, contract: dict[str, Any]):
    import torch

    if contract["scheduler"] == "none":
        return None
    if contract["scheduler"] == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=float(contract["min_lr"]))
    if contract["scheduler"] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    raise RuntimeError(f"unsupported scheduler {contract['scheduler']}")


def compute_training_loss(model_name: str, y_true: Any, y_pred: Any, contract: dict[str, Any]) -> Any:
    if model_name in {"fcnn", "cnn", "eegnet", "vlaai"}:
        return -torch_correlation(y_true, y_pred)
    if model_name == "adt":
        return sequence_pearson_loss(y_true, y_pred)
    if model_name == "happyquokka":
        return happyquokka_loss(y_true, y_pred, float(contract["lambda_l1"]))
    raise RuntimeError(model_name)


def reconstruct_torch_recordings_batch(model: Any, trials: list[dict[str, Any]], contract: dict[str, Any], device: Any, y_scaler: StandardScaler) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]:
    import torch

    model.eval()
    out_rows = []
    window_size, hop = int(contract["window_size"]), int(contract["hop"])
    batch_size = int(contract["batch_size"])
    with torch.no_grad():
        for trial in trials:
            starts = list(range(0, trial["x"].shape[0] - window_size + 1, hop))
            pred_sum = np.zeros(trial["x"].shape[0], dtype=np.float64)
            count = np.zeros(trial["x"].shape[0], dtype=np.int32)
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                arrays = []
                for start in batch_starts:
                    x = trial["x"][start : start + window_size]
                    arrays.append(x.T if contract["layout"] == "channels_time" else x)
                xb = torch.from_numpy(np.stack(arrays).astype(np.float32)).to(device)
                pred = model(xb).detach().cpu().numpy()
                for local, start in enumerate(batch_starts):
                    stop = start + window_size
                    if contract["target_mode"] == "last":
                        pred_sum[stop - 1] += float(np.asarray(pred[local]).reshape(-1)[-1])
                        count[stop - 1] += 1
                    else:
                        seq = np.asarray(pred[local]).reshape(window_size, -1)[:, 0]
                        pred_sum[start:stop] += seq
                        count[start:stop] += 1
            valid = count > 0
            pred_scaled = np.zeros(trial["x"].shape[0], dtype=np.float32)
            pred_scaled[valid] = (pred_sum[valid] / count[valid]).astype(np.float32)
            pred_raw = y_scaler.inverse_transform(pred_scaled[:, None]).reshape(-1).astype(np.float32)
            out_rows.append((trial, pred_raw, trial["target_raw"], valid.astype(np.int32)))
    return out_rows


def eval_torch_recordings(model: Any, trials: list[dict[str, Any]], contract: dict[str, Any], device: Any, y_scaler: StandardScaler) -> tuple[list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]], float]:
    rows = reconstruct_torch_recordings_batch(model, trials, contract, device, y_scaler)
    scores = [pearson_on_valid(pred, target, mask) for _, pred, target, mask in rows]
    return rows, float(np.nanmean(scores))


def run_torch_model(model_name: str, config: dict[str, Any], subject_cfg: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path, smoke: bool = False) -> tuple[list[RecordingMetricRow], dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    contract = torch_model_contract(model_name, config, smoke=smoke)
    if contract["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    device = torch.device(contract["device"])
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    x_scaler, y_scaler = fit_scalers(splits["train"])
    cached = make_scaled_cache(splits, x_scaler, y_scaler)
    train_ds = WindowDataset(cached["train"], contract["window_size"], contract["hop"], layout=contract["layout"], target_mode=contract["target_mode"], max_items=contract["max_train_windows"])
    if len(train_ds) == 0:
        raise RuntimeError("empty train window dataset")
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(train_ds, batch_size=contract["batch_size"], shuffle=True, num_workers=0, generator=generator)
    model = build_torch_model(model_name, config).to(device)
    optimizer = make_optimizer(model_name, model, contract)
    scheduler = make_scheduler(optimizer, contract)
    best_state = None
    best_metric = -math.inf
    best_epoch = 0
    stale = 0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    history_fields = ["job_key", "subject_id", "model", "epoch", "train_loss", "val_mean_recording_pearson_r", "elapsed_seconds", "device", "gpu_name"]
    job_key = f"{subject_cfg['subject_id']}:{model_name}:seed{config['seed']}"
    epochs_completed = 0
    for epoch in range(1, int(contract["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.shape != yb.shape:
                pred = pred.reshape_as(yb)
            loss = compute_training_loss(model_name, yb, pred, contract)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, val_metric = eval_torch_recordings(model, cached["val"], contract, device, y_scaler)
        val_loss_for_scheduler = -float(val_metric)
        if scheduler is not None:
            if contract["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_loss_for_scheduler)
            else:
                scheduler.step()
        epochs_completed += 1
        row = {
            "job_key": job_key,
            "subject_id": subject_cfg["subject_id"],
            "model": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_mean_recording_pearson_r": val_metric,
            "elapsed_seconds": time.perf_counter() - epoch_start,
            "device": str(device),
            "gpu_name": gpu_name,
        }
        append_csv(output_dir / "training_history.csv", [row], history_fields)
        msg = f"EPOCH job={job_key} epoch={epoch} train_loss={row['train_loss']:.6f} val_r={val_metric:.6f}"
        print(msg, flush=True)
        append_log(output_dir, model_name, msg)
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(contract["patience"]):
            break
    if best_state is None:
        raise RuntimeError("no best model state selected")
    model.load_state_dict(best_state)
    model.to(device)
    test_recons, _ = eval_torch_recordings(model, cached["test"], contract, device, y_scaler)
    rec_rows = [make_recording_row(config, subject_cfg, model_name, trial["recording_id"], f"best_epoch_{best_epoch}", pred, target, mask) for trial, pred, target, mask in test_recons]
    identities = {
        "fcnn": "accepted_fcnn_mag102_adapter",
        "cnn": "accepted_cnn_mag102_adapter",
        "eegnet": "accepted_eegnet_mag102_adapter",
        "adt": "accepted_adt_mag102_adapter",
        "vlaai": "vlaai_meg102_adapter",
        "happyquokka": "happyquokka_meg102_chunk_adapter",
    }
    return rec_rows, {
        "checkpoint_id": f"best_epoch_{best_epoch}",
        "best_epoch": best_epoch,
        "best_val_score": best_metric,
        "epochs_completed": epochs_completed,
        "device": str(device),
        "gpu_name": gpu_name,
        "implementation_identity": identities[model_name],
        "input_shape": [102, contract["window_size"]] if contract["layout"] == "channels_time" else [contract["window_size"], 102],
        "scaler_fit_scope": "train_recordings_only",
        "training_budget": "smoke" if smoke else "full_config",
        "optimizer": contract["optimizer"],
        "loss": contract["loss"],
        "scheduler": contract["scheduler"],
        "learning_rate": contract["learning_rate"],
        "weight_decay": contract["weight_decay"],
        "batch_size": contract["batch_size"],
        "max_epochs": contract["max_epochs"],
        "patience": contract["patience"],
    }


def preflight_model(model_name: str, config: dict[str, Any], subject_cfg: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        x_scaler, y_scaler = fit_scalers(splits["train"])
        x0 = x_scaler.transform(splits["train"][0]["x_mag102"]).astype(np.float32)
        y0 = y_scaler.transform(splits["train"][0]["target"][:, None]).reshape(-1).astype(np.float32)
        if model_name in {"linear", "ridge", "lasso", "elasticnet", "cca"}:
            lagged = lag_sparse(x0, 0, 50)
            return {"subject_id": subject_cfg["subject_id"], "model": model_name, "status": "ok", "input_shape": list(lagged.shape), "target_shape": [len(y0) - 49], "device_path": "cpu"}
        import torch

        contract = torch_model_contract(model_name, config, smoke=True)
        if contract["device"] == "cuda" and not torch.cuda.is_available():
            return {"subject_id": subject_cfg["subject_id"], "model": model_name, "status": "failed", "failure_reason": "CUDA unavailable"}
        device = torch.device(contract["device"])
        model = build_torch_model(model_name, config).to(device)
        sample_shape = (2, 102, contract["window_size"]) if contract["layout"] == "channels_time" else (2, contract["window_size"], 102)
        sample = torch.zeros(*sample_shape, device=device)
        with torch.no_grad():
            out = model(sample)
        return {"subject_id": subject_cfg["subject_id"], "model": model_name, "status": "ok", "input_shape": list(sample.shape), "output_shape": list(out.shape), "device_path": str(device)}
    except Exception as exc:
        return {"subject_id": subject_cfg["subject_id"], "model": model_name, "status": "failed", "failure_reason": repr(exc)}


def run_model(model_name: str, config: dict[str, Any], subject_cfg: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path, smoke: bool) -> tuple[list[RecordingMetricRow], dict[str, Any]]:
    if model_name in {"linear", "ridge", "lasso", "elasticnet"}:
        return run_linear_family(model_name, config, subject_cfg, splits, output_dir)
    if model_name == "cca":
        return run_cca(config, subject_cfg, splits, output_dir)
    return run_torch_model(model_name, config, subject_cfg, splits, output_dir, smoke=smoke)


def planned_jobs(config: dict[str, Any]) -> list[str]:
    return [f"{subject['subject_id']}:{model}:seed{config['seed']}" for subject in subject_configs(config) for model in config["models"]]


def clear_job_outputs(output_dir: Path, job_key: str) -> None:
    for path, fields in [
        (output_dir / "recording_metrics.csv", ["job_key", *RECORDING_METRIC_FIELDS]),
        (output_dir / "subject_metrics.csv", ["job_key", *SUBJECT_METRIC_FIELDS]),
        (output_dir / "validation_diagnostics.csv", ["job_key", "model", "implementation_identity", "candidate", "val_recording_pearson", "val_mean_recording_pearson_r", "legacy_concatenated_val_pearson_r"]),
        (output_dir / "training_history.csv", ["job_key", "subject_id", "model", "epoch", "train_loss", "val_mean_recording_pearson_r", "elapsed_seconds", "device", "gpu_name"]),
    ]:
        filter_csv_by_job(path, job_key, fields)


def run_resume(config: dict[str, Any], loaded: dict[str, tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]], output_dir: Path, max_jobs: int | None, smoke: bool) -> None:
    planned = planned_jobs(config)
    completed = read_json(output_dir / "completed_jobs.json").get("completed_jobs", []) if (output_dir / "completed_jobs.json").exists() else []
    failures = read_json(output_dir / "failure_report.json").get("failures", []) if (output_dir / "failure_report.json").exists() else []
    entries = read_json(output_dir / "model_run_entries.json") if (output_dir / "model_run_entries.json").exists() else []
    completed_keys = {row["job_key"] for row in completed}
    failure_keys = {row["job_key"] for row in failures}
    ran = 0
    for subject_cfg in subject_configs(config):
        for model_name in config["models"]:
            job_key = f"{subject_cfg['subject_id']}:{model_name}:seed{config['seed']}"
            if job_key in completed_keys or job_key in failure_keys:
                print(f"JOB SKIP completed_or_failed {job_key}", flush=True)
                continue
            if max_jobs is not None and ran >= max_jobs:
                atomic_write_run_lists(output_dir, completed, failures, entries, planned)
                return
            print(f"JOB START {job_key}", flush=True)
            clear_job_outputs(output_dir, job_key)
            start = time.perf_counter()
            try:
                _, splits = loaded[subject_cfg["subject_id"]]
                rec_rows, meta = run_model(model_name, config, subject_cfg, splits, output_dir, smoke=smoke)
                rec_dicts = [{"job_key": job_key, **asdict(row)} for row in rec_rows]
                subj_rows = subject_metric_rows(rec_rows)
                subj_dicts = [{"job_key": job_key, **asdict(row)} for row in subj_rows]
                append_csv(output_dir / "recording_metrics.csv", rec_dicts, ["job_key", *RECORDING_METRIC_FIELDS])
                append_csv(output_dir / "subject_metrics.csv", subj_dicts, ["job_key", *SUBJECT_METRIC_FIELDS])
                metric = float(subj_rows[0].metric_value)
                entry = {"job_key": job_key, "subject_id": subject_cfg["subject_id"], "model": model_name, "status": "success", "metric": metric, "elapsed_seconds": time.perf_counter() - start, **meta}
                entries = [row for row in entries if row.get("job_key") != job_key] + [entry]
                completed = [row for row in completed if row.get("job_key") != job_key] + [{"job_key": job_key, "subject_id": subject_cfg["subject_id"], "model": model_name, "metric": metric}]
                print(f"JOB COMPLETE {job_key} metric={metric:.6f}", flush=True)
            except Exception as exc:
                failure = {"job_key": job_key, "subject_id": subject_cfg["subject_id"], "model": model_name, "status": "failed", "failure_reason": repr(exc), "elapsed_seconds": time.perf_counter() - start}
                failures = [row for row in failures if row.get("job_key") != job_key] + [failure]
                entries = [row for row in entries if row.get("job_key") != job_key] + [failure]
                print(f"JOB FAILED {job_key} reason={repr(exc)}", flush=True)
            atomic_write_run_lists(output_dir, completed, failures, entries, planned)
            ran += 1


def scan_eligible_subjects(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = Path(config["canonical_data_root"])
    subjects = []
    for paired in sorted(base.glob("sub-*/speech/sub-*_preprocessed_audiobooks_decoding.mat")):
        subject_id = subject_from_path(paired, paired.name.split("_")[0])
        envelope = Path(config["envelope_mat"])
        try:
            paired_sha = sha256_file(paired)
            envelope_sha = sha256_file(envelope)
            with h5py.File(paired, "r") as f:
                n_trials = len(np.array(f["results"]["epochs_neuro"]["trial"]).reshape(-1))
                fs = int(np.array(f["results"]["epochs_neuro"]["fsample"]).squeeze())
            status = "eligible" if n_trials == 16 and fs == 64 else "ineligible_bad_trial_or_fs"
        except Exception as exc:
            paired_sha, envelope_sha, n_trials, fs, status = "", "", 0, 0, f"ineligible_error:{exc!r}"
        if status == "eligible":
            subjects.append(
                {
                    "subject_id": subject_id,
                    "dataset_id": f"meg_scans_canonical_05_08hz_64hz_{subject_id}",
                    "paired_mat": str(paired),
                    "envelope_mat": str(envelope),
                    "expected_paired_sha256": paired_sha,
                    "expected_envelope_sha256": envelope_sha,
                    "split": deepcopy(config["split"]),
                }
            )
    return subjects


def training_profiles(config: dict[str, Any]) -> dict[str, Any]:
    source_commit = config["model_lock_source"]["source_commit"]
    profiles: dict[str, Any] = {
        "linear": {"implementation_identity": "recording_safe_linear_adapter", "profile_status": "benchmark_local", "optimizer": "sklearn LinearRegression", "loss": "least_squares", "scheduler": "none"},
        "ridge": {"implementation_identity": "recording_safe_ridge_adapter", "profile_status": "benchmark_local", "optimizer": "sklearn Ridge", "loss": "least_squares", "scheduler": "none"},
        "lasso": {"implementation_identity": "recording_safe_lasso_adapter", "profile_status": "benchmark_local", "optimizer": "sklearn Lasso", "loss": "least_squares_l1", "scheduler": "none"},
        "elasticnet": {"implementation_identity": "benchmark_local_elasticnet", "profile_status": "benchmark_local", "optimizer": "sklearn ElasticNet", "loss": "least_squares_l1_l2", "scheduler": "none"},
        "cca": {"implementation_identity": "recording_safe_cca_meg_adapter", "profile_status": "benchmark_local", "optimizer": "sklearn CCA + Ridge", "loss": "ridge_reconstruction_to_y_hat", "scheduler": "none"},
    }
    for model in ["fcnn", "cnn", "eegnet"]:
        cfg = config["window_models"][model]
        profiles[model] = {
            "implementation_identity": f"accepted_{model}_mag102_adapter",
            "EEG source branch": config["model_lock_source"]["source_commit"],
            "EEG source commit": source_commit,
            "EEG source file": "src/repro/reference_baselines.py",
            "EEG source function": "train_dnn_reference_logged",
            "optimizer": cfg["optimizer"],
            "loss": cfg["loss"],
            "scheduler": cfg["scheduler"],
            "lr": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"],
            "batch_size": cfg["batch_size"],
            "max_epochs": cfg["max_epochs"],
            "patience": cfg["early_stopping_patience"],
            "input adapter": "mag102 native window input",
            "profile_status": "source_faithful_after_input_adapter",
        }
    for model, source_file, source_fn in [
        ("adt", "src/repro/adt_exact.py", "train_adt_exact_reference"),
        ("vlaai", "src/repro/vlaai_exact.py", "train_vlaai_exact_reference"),
    ]:
        cfg = config[model]
        profiles[model] = {
            "implementation_identity": "accepted_adt_mag102_adapter" if model == "adt" else "vlaai_meg102_adapter",
            "EEG source branch": config["model_lock_source"]["source_commit"],
            "EEG source commit": source_commit,
            "EEG source file": source_file,
            "EEG source function": source_fn,
            "optimizer": cfg["optimizer"],
            "loss": cfg["loss"],
            "scheduler": cfg["scheduler"],
            "lr": cfg["learning_rate"],
            "min_lr": cfg.get("min_lr"),
            "weight_decay": cfg.get("weight_decay", 0.0),
            "batch_size": cfg["batch_size"],
            "max_epochs": cfg["max_epochs"],
            "patience": cfg["early_stopping_patience"],
            "input adapter": "mag102 native" if model == "adt" else "trainable 102->64 MEG projection",
            "profile_status": "source_faithful_after_input_adapter",
        }
    cfg = config["happyquokka"]
    profiles["happyquokka"] = {
        "implementation_identity": "happyquokka_meg102_chunk_adapter",
        "EEG source branch": config["model_lock_source"]["source_commit"],
        "EEG source commit": source_commit,
        "EEG source file": "src/repro/happyquokka_reference.py",
        "EEG source function": "train_happyquokka_reference",
        "optimizer": cfg["optimizer"],
        "loss": cfg["loss"],
        "scheduler": cfg["scheduler"],
        "lr": cfg["learning_rate"],
        "weight_decay": cfg.get("weight_decay", 0.0),
        "batch_size": cfg["batch_size"],
        "max_epochs": cfg["max_epochs"],
        "patience": cfg["early_stopping_patience"],
        "input adapter": "trainable 102->64 MEG projection; 10s/640 sample chunks",
        "profile_status": "source_faithful_after_input_adapter",
        "g_con": cfg["g_con"],
    }
    return profiles


def generate_all_canonicalized_config(config: dict[str, Any]) -> dict[str, Any]:
    subjects = scan_eligible_subjects(config)
    out_config = deepcopy(config)
    out_config["protocol"] = "meg_scans_subject_specific_11models_all_canonicalized_v1"
    out_config["artifact_scope"] = "native_meg_subject_specific_11models_all_canonicalized"
    out_config["output_dir"] = "experiments/meg_scans_subject_specific_11models_all_canonicalized_v1"
    out_config["subjects"] = subjects
    out_config["subject_ids"] = [row["subject_id"] for row in subjects]
    out_config["subject_scope_note"] = "Subjects are included only if canonical 0.5-8 Hz / 64 Hz paired MAT already exists locally; this is not a scan of all raw SCANS participants."
    for key in ["paired_mat", "expected_paired_sha256", "dataset_id"]:
        out_config.pop(key, None)
    out_path = ROOT / "configs" / "benchmark" / "meg_scans_subject_specific_11models" / "all_canonicalized_v1.json"
    write_json(out_path, out_config)
    manifest_path = ROOT / "experiments" / "meg_scans_subject_specific_11models_all_canonicalized_v1" / "canonicalized_subject_manifest.json"
    write_json(
        manifest_path,
        {
            "canonicalized_subjects": subjects,
            "canonicalized_subject_count": len(subjects),
            "selection_rule": "canonical paired MAT already exists with 16 trials at 64 Hz",
            "scope_note": out_config["subject_scope_note"],
        },
    )
    return out_config


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg["linear_family"]["max_fit_samples_per_split"] = min(int(cfg["linear_family"].get("max_fit_samples_per_split", 12000)), 512)
    cfg["linear_family"]["ridge_alphas"] = cfg["linear_family"]["ridge_alphas"][:1]
    cfg["linear_family"]["lasso_alphas"] = cfg["linear_family"]["lasso_alphas"][:1]
    cfg["linear_family"]["elasticnet_alphas"] = cfg["linear_family"]["elasticnet_alphas"][:1]
    cfg["linear_family"]["elasticnet_l1_ratios"] = cfg["linear_family"]["elasticnet_l1_ratios"][:1]
    cfg["linear_family"]["max_iter"] = min(int(cfg["linear_family"]["max_iter"]), 200)
    cfg["cca"]["max_fit_samples"] = min(int(cfg["cca"].get("max_fit_samples", 8000)), 512)
    cfg["cca"]["x_pca_grid"] = cfg["cca"]["x_pca_grid"][:1]
    cfg["cca"]["y_pca_grid"] = cfg["cca"]["y_pca_grid"][:1]
    cfg["cca"]["n_components_grid"] = cfg["cca"]["n_components_grid"][:1]
    cfg["cca"]["alpha_grid"] = cfg["cca"]["alpha_grid"][:1]
    cfg["cca"]["max_iter"] = min(int(cfg["cca"]["max_iter"]), 100)
    return cfg


def load_all(config: dict[str, Any], output_dir: Path) -> dict[str, tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]]:
    loaded = {}
    for subject_cfg in subject_configs(config):
        data = load_paired_mat(subject_cfg["paired_mat"], subject_cfg["subject_id"])
        validate_and_select(data, config, subject_cfg, output_dir)
        splits = split_trials(data, subject_cfg, output_dir)
        loaded[subject_cfg["subject_id"]] = (data, splits)
    return loaded


def write_preflight(config: dict[str, Any], loaded: dict[str, tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]], output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for subject_cfg in subject_configs(config):
        _, splits = loaded[subject_cfg["subject_id"]]
        rows.extend([preflight_model(model, config, subject_cfg, splits) for model in config["models"]])
    write_json(output_dir / "shape_preflight.json", {"rows": rows, "all_models_checked": len(rows) == len(subject_configs(config)) * 11})
    return rows


def validate_outputs(config: dict[str, Any], output_dir: Path, *, require_complete: bool) -> dict[str, Any]:
    errors = []
    planned = planned_jobs(config)
    completed = read_json(output_dir / "completed_jobs.json").get("completed_jobs", []) if (output_dir / "completed_jobs.json").exists() else []
    failures = read_json(output_dir / "failure_report.json").get("failures", []) if (output_dir / "failure_report.json").exists() else []
    entries = read_json(output_dir / "model_run_entries.json") if (output_dir / "model_run_entries.json").exists() else []
    if any(row.get("status") == "success" for row in failures):
        errors.append("failure row masquerades as success")
    completed_keys = [row["job_key"] for row in completed]
    if len(completed_keys) != len(set(completed_keys)):
        errors.append("completed job key not unique")
    failure_keys = {row["job_key"] for row in failures}
    success_keys = {row["job_key"] for row in completed}
    if success_keys & failure_keys:
        errors.append("job appears in both success and failure")
    rec_rows = csv_rows(output_dir / "recording_metrics.csv")
    subj_rows = csv_rows(output_dir / "subject_metrics.csv")
    for job_key in success_keys:
        subject_id = job_key.split(":")[0]
        rec_for_job = [row for row in rec_rows if row.get("job_key") == job_key]
        subj_for_job = [row for row in subj_rows if row.get("job_key") == job_key]
        if len(rec_for_job) != 2:
            errors.append(f"{job_key} has {len(rec_for_job)} recording rows, expected 2")
        if len(subj_for_job) != 1:
            errors.append(f"{job_key} has {len(subj_for_job)} subject rows, expected 1")
        if rec_for_job and subj_for_job:
            mean_rec = float(np.mean([float(row["metric_value"]) for row in rec_for_job]))
            subj_val = float(subj_for_job[0]["metric_value"])
            if abs(mean_rec - subj_val) > 1e-9:
                errors.append(f"{job_key} subject metric does not match recording mean")
        if any(row.get("subject_id") != subject_id for row in rec_for_job + subj_for_job):
            errors.append(f"{job_key} subject_id mismatch")
    if require_complete and set(completed_keys) != set(planned):
        errors.append("not all planned jobs completed successfully")
    import subprocess

    proc = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        errors.append(f"git ls-files failed: {proc.stderr.strip()}")
    else:
        bad = [line for line in proc.stdout.splitlines() if Path(line).suffix.lower() in BANNED_SUFFIXES and "meg_scans_subject_specific_11models" in line]
        if bad:
            errors.append("banned tracked artifacts: " + ", ".join(bad))
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "planned_jobs": planned,
        "completed_jobs": sorted(success_keys),
        "failed_jobs": sorted(failure_keys),
        "ready_for_full_manual_run": (not errors and require_complete),
    }
    if require_complete:
        report["run_status"] = "completed" if report["status"] == "passed" else "incomplete_or_failed"
    else:
        report["run_status"] = "incomplete_or_failed"
    write_json(output_dir / "schema_validation_report.json", report)
    return report


def write_cca_x_only_assertion(output_dir: Path) -> None:
    entries_path = output_dir / "model_run_entries.json"
    entries = read_json(entries_path) if entries_path.exists() else []
    cca_entries = [row for row in entries if row.get("model") == "cca" and row.get("status") == "success"]
    status = "passed" if cca_entries and all(row.get("cca_inference_uses_target") is False for row in cca_entries) else "failed"
    write_json(
        output_dir / "cca_x_only_inference_assertion.json",
        {
            "status": status,
            "assertion": "validation/test CCA reconstruction uses X lag -> X scaler/PCA -> CCA.transform(X) -> ridge y_hat; target is used only for final Pearson scoring",
            "checked_entries": cca_entries,
        },
    )


def write_run_manifest(config: dict[str, Any], output_dir: Path, status: str, training_started: bool) -> None:
    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "models": config["models"],
            "subjects": [row["subject_id"] for row in subject_configs(config)],
            "status": status,
            "training_started": training_started,
            "primary_metric": "mean_recording_pearson_r",
            "recording_level_split": True,
            "representation": "mag102",
        },
    )


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    if args.generate_all_eligible_config or args.generate_all_canonicalized_config:
        config = generate_all_canonicalized_config(config)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)
    write_model_lock(config, output_dir)
    write_json(output_dir / "model_training_profiles.json", training_profiles(config))
    loaded = load_all(config, output_dir)
    preflight = write_preflight(config, loaded, output_dir)
    write_run_manifest(config, output_dir, "preflight_complete", False)
    if args.preflight_only or args.dry_run:
        print(json.dumps({"status": "preflight_complete", "output_dir": str(output_dir), "preflight": preflight}, indent=2), flush=True)
        return 0
    if args.smoke_gate:
        config = smoke_config(config)
        smoke_dir = output_dir / "smoke_gate"
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        smoke_dir.mkdir(parents=True, exist_ok=True)
        write_json(smoke_dir / "run_config.json", config)
        write_model_lock(config, smoke_dir)
        write_json(smoke_dir / "model_training_profiles.json", training_profiles(config))
        loaded_smoke = load_all(config, smoke_dir)
        write_preflight(config, loaded_smoke, smoke_dir)
        write_run_manifest(config, smoke_dir, "smoke_running", True)
        run_resume(config, loaded_smoke, smoke_dir, None, smoke=True)
        write_cca_x_only_assertion(smoke_dir)
        report = validate_outputs(config, smoke_dir, require_complete=True)
        state_path = smoke_dir / "run_state.json"
        state = read_json(state_path)
        state["ready_for_full_manual_run"] = report["status"] == "passed"
        write_json(state_path, state)
        print(json.dumps({"status": "smoke_complete", "report": report}, indent=2), flush=True)
        return 0 if report["status"] == "passed" else 1
    if not args.resume:
        raise SystemExit("Use --resume for formal incremental execution.")
    write_run_manifest(config, output_dir, "running_or_resumable", True)
    run_resume(config, loaded, output_dir, args.max_jobs, smoke=False)
    report = validate_outputs(config, output_dir, require_complete=(args.max_jobs is None))
    if args.max_jobs is None and report["status"] != "passed":
        write_run_manifest(config, output_dir, "incomplete_or_failed", True)
        return 1
    if args.max_jobs is None:
        write_run_manifest(config, output_dir, "completed", True)
    else:
        write_run_manifest(config, output_dir, "incomplete_or_failed", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
