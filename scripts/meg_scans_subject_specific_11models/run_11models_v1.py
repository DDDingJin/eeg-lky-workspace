from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SHARED_UPSTREAM = Path(ROOT.drive + "\\decode") / "external" / "upstream" / "mldecoders"
if SHARED_UPSTREAM.exists() and str(SHARED_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(SHARED_UPSTREAM))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow, write_rows
from benchmark.scoring import pearson_on_valid, subject_metric_rows
from repro.adt_exact import ADTExactRegressor
from repro.mldecoders.models import EEGNetRegressor, VLAAIExactOfficialRegressor
from repro.simple_models import FCNNBaseline

try:
    from pipeline.dnn import CNN as UpstreamCNN
except Exception:
    UpstreamCNN = None


RUN_LABELS = [
    "task-audiobook1_run-01",
    "task-audiobook1_run-02",
    "task-audiobook2_run-01",
    "task-audiobook2_run-02",
]
MODEL_SET = ["linear", "ridge", "lasso", "elasticnet", "cca", "fcnn", "cnn", "eegnet", "adt", "vlaai", "happyquokka"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_matlab_string(f: h5py.File, ref: Any) -> str:
    return "".join(chr(int(x)) for x in np.array(f[ref]).squeeze())


def deref_cell(f: h5py.File, dataset: h5py.Dataset) -> list[Any]:
    return [f[ref] for ref in np.array(dataset).reshape(-1)]


def load_paired_mat(path: Path) -> dict[str, Any]:
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
    trials = []
    for idx, (x, y) in enumerate(zip(meg_trials, targets), start=1):
        run_idx = (idx - 1) // 4
        local_idx = (idx - 1) % 4 + 1
        trials.append(
            {
                "global_trial_index": idx,
                "run_label": RUN_LABELS[run_idx],
                "trial_index_in_run": local_idx,
                "recording_id": f"sub-03_{RUN_LABELS[run_idx]}_trial-{local_idx:02d}",
                "x_all306": x,
                "target": y,
            }
        )
    return {"fs": fs, "labels": labels, "trials": trials}


def validate_and_select(data: dict[str, Any], config: dict[str, Any], output_dir: Path) -> None:
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
    paired_sha = sha256_file(Path(config["paired_mat"]))
    envelope_sha = sha256_file(Path(config["envelope_mat"]))
    if paired_sha != config["expected_paired_sha256"] or envelope_sha != config["expected_envelope_sha256"]:
        raise RuntimeError("MAT SHA256 mismatch")
    if data["fs"] != 64 or len(data["trials"]) != 16:
        raise RuntimeError("bad fs/trial count")
    for trial in data["trials"]:
        x, y = trial["x_all306"], trial["target"]
        if x.shape != (7680, 306) or y.shape != (7680,):
            raise RuntimeError(f"bad shape for {trial['recording_id']}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError(f"NaN/Inf in {trial['recording_id']}")
        trial["x_mag102"] = x[:, mag_indices]
    write_json(
        output_dir / "data_integrity_check.json",
        {
            "status": "passed",
            "paired_sha256": paired_sha,
            "envelope_sha256": envelope_sha,
            "n_trials": 16,
            "fs": 64,
            "representation": "mag102",
            "mag102_channel_order_sha256": mag_hash,
            "channel_count": 102,
        },
    )


def split_trials(data: dict[str, Any], config: dict[str, Any], output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    lookup = {t["global_trial_index"]: t for t in data["trials"]}
    split = config["split"]
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
            rows.append({"role": role, "global_trial_index": idx, "recording_id": trial["recording_id"], "run_label": trial["run_label"]})
    if len(seen) != 16:
        raise RuntimeError("split does not cover all trials")
    write_json(output_dir / "split_manifest.json", {"identity": split["identity"], "rows": rows, "recording_level": True})
    return splits


def write_model_lock(config: dict[str, Any], output_dir: Path) -> None:
    if config["models"] != MODEL_SET or len(config["models"]) != 11:
        raise RuntimeError("model set is not exactly the locked 11-model roster")
    source_cfg = read_json(Path(config["model_lock_source"]["config_path"]))
    source_audit = read_json(Path(config["model_lock_source"]["audit_manifest_path"]))
    payload = {
        "status": "locked",
        "models": config["models"],
        "model_count": len(config["models"]),
        "excluded_models": config["excluded_models"],
        "source_config_path": config["model_lock_source"]["config_path"],
        "source_config_commit": config["model_lock_source"]["source_commit"],
        "source_config_models": source_cfg["models"],
        "audit_manifest_path": config["model_lock_source"]["audit_manifest_path"],
        "audit_identity_conclusion": source_audit.get("identity_conclusion"),
        "audit_roster_decision": source_audit.get("roster_decision"),
    }
    write_json(output_dir / "meg_model_set_lock.json", payload)


def fit_scalers(train: list[dict[str, Any]]) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler().fit(np.concatenate([t["x_mag102"] for t in train], axis=0))
    y_scaler = StandardScaler().fit(np.concatenate([t["target"][:, None] for t in train], axis=0))
    return x_scaler, y_scaler


def lag_sparse(x: np.ndarray, start_lag: int, end_lag: int) -> sparse.csr_matrix:
    valid_start = end_lag - 1
    return sparse.hstack([sparse.csr_matrix(x[valid_start - lag : x.shape[0] - lag]) for lag in range(start_lag, end_lag)], format="csr")


def lagged_split(trials: list[dict[str, Any]], x_scaler: StandardScaler, y_scaler: StandardScaler, start_lag: int, end_lag: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    xs, ys = [], []
    valid_start = end_lag - 1
    for trial in trials:
        x = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
        y = y_scaler.transform(trial["target"][:, None]).reshape(-1).astype(np.float32)
        xs.append(lag_sparse(x, start_lag, end_lag))
        ys.append(y[valid_start:])
    return sparse.vstack(xs, format="csr"), np.concatenate(ys)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def append_log(output_dir: Path, model_name: str, message: str) -> None:
    path = output_dir / f"{model_name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


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


def window_index(trials: list[dict[str, Any]], window_size: int, hop: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for trial_idx, trial in enumerate(trials):
        n = int(trial["x"].shape[0])
        for start in range(0, n - window_size + 1, hop):
            rows.append((trial_idx, start))
    return rows


class WindowDataset:
    def __init__(self, trials: list[dict[str, Any]], window_size: int, hop: int, *, layout: str, target_mode: str) -> None:
        self.trials = trials
        self.window_size = int(window_size)
        self.layout = layout
        self.target_mode = target_mode
        self.index = window_index(trials, int(window_size), int(hop))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
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
            target = y[-1]
        elif self.target_mode == "sequence":
            target = y[:, None]
        else:
            raise RuntimeError(f"bad target mode {self.target_mode}")
        return x.astype(np.float32), np.asarray(target, dtype=np.float32)


def build_torch_model(model_name: str, config: dict[str, Any]):
    if model_name == "fcnn":
        return FCNNBaseline(num_input_channels=102, input_length=int(config["window_models"]["window_size"]))
    if model_name == "cnn":
        if UpstreamCNN is None:
            raise RuntimeError("upstream CNN import unavailable")
        return UpstreamCNN(num_input_channels=102, input_length=int(config["window_models"]["window_size"]))
    if model_name == "eegnet":
        return EEGNetRegressor(num_input_channels=102, input_length=int(config["window_models"]["window_size"]))
    if model_name == "adt":
        return ADTExactRegressor(chans=102, seq_len=int(config["adt"]["window_length"]))
    raise RuntimeError(f"no torch model builder for {model_name}")


def torch_model_contract(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    if model_name in {"fcnn", "cnn", "eegnet"}:
        cfg = config["window_models"]
        return {
            "window_size": int(cfg["window_size"]),
            "hop": 1,
            "layout": "channels_time",
            "target_mode": "last",
            "batch_size": int(cfg["batch_size"]),
            "max_epochs": int(cfg["max_epochs"]),
            "patience": int(cfg["early_stopping_patience"]),
            "learning_rate": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "device": cfg["device"],
        }
    if model_name == "adt":
        cfg = config["adt"]
        return {
            "window_size": int(cfg["window_length"]),
            "hop": int(cfg["hop_length"]),
            "layout": "time_channels",
            "target_mode": "sequence",
            "batch_size": int(cfg["batch_size"]),
            "max_epochs": int(cfg["max_epochs"]),
            "patience": int(cfg["early_stopping_patience"]),
            "learning_rate": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "device": cfg["device"],
        }
    raise RuntimeError(model_name)


def reconstruct_torch_recording(model: Any, trial: dict[str, Any], contract: dict[str, Any], device: Any, y_scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    n = int(trial["x"].shape[0])
    pred_sum = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int32)
    window_size = int(contract["window_size"])
    hop = int(contract["hop"])
    starts = list(range(0, n - window_size + 1, hop))
    with torch.no_grad():
        for start in starts:
            stop = start + window_size
            x = trial["x"][start:stop]
            if contract["layout"] == "channels_time":
                arr = x.T[None, :, :]
            else:
                arr = x[None, :, :]
            xb = torch.from_numpy(arr.astype(np.float32)).to(device)
            out = model(xb).detach().cpu().numpy()
            if contract["target_mode"] == "last":
                pred_sum[stop - 1] += float(np.asarray(out).reshape(-1)[-1])
                count[stop - 1] += 1
            else:
                pred_seq = np.asarray(out).reshape(window_size, -1)[:, 0]
                pred_sum[start:stop] += pred_seq
                count[start:stop] += 1
    valid = count > 0
    pred_scaled = np.zeros(n, dtype=np.float32)
    pred_scaled[valid] = (pred_sum[valid] / count[valid]).astype(np.float32)
    pred = y_scaler.inverse_transform(pred_scaled[:, None]).reshape(-1).astype(np.float32)
    return pred, trial["target_raw"], valid.astype(np.int32)


def eval_torch_recordings(model: Any, trials: list[dict[str, Any]], contract: dict[str, Any], device: Any, y_scaler: StandardScaler) -> tuple[list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]], float]:
    rows = []
    scores = []
    for trial in trials:
        pred, target, mask = reconstruct_torch_recording(model, trial, contract, device, y_scaler)
        rows.append((trial, pred, target, mask))
        scores.append(pearson_on_valid(pred, target, mask))
    return rows, float(np.nanmean(scores))


def run_torch_model(model_name: str, config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[list[RecordingMetricRow], dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    contract = torch_model_contract(model_name, config)
    if contract["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    device = torch.device(contract["device"])
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    x_scaler, y_scaler = fit_scalers(splits["train"])
    cached = make_scaled_cache(splits, x_scaler, y_scaler)
    train_ds = WindowDataset(cached["train"], contract["window_size"], contract["hop"], layout=contract["layout"], target_mode=contract["target_mode"])
    val_ds = WindowDataset(cached["val"], contract["window_size"], contract["hop"], layout=contract["layout"], target_mode=contract["target_mode"])
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("empty train/val window dataset")
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(train_ds, batch_size=contract["batch_size"], shuffle=True, num_workers=0, generator=generator)
    model = build_torch_model(model_name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract["learning_rate"], weight_decay=contract["weight_decay"])
    criterion = torch.nn.MSELoss()
    best_state = None
    best_metric = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    append_log(output_dir, model_name, f"JOB TRAIN model={model_name} device={device} gpu={gpu_name} train_windows={len(train_ds)} val_windows={len(val_ds)}")
    for epoch in range(1, int(contract["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.shape != yb.shape:
                pred = pred.reshape_as(yb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_recons, val_metric = eval_torch_recordings(model, cached["val"], contract, device, y_scaler)
        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_mean_recording_pearson_r": val_metric,
            "elapsed_seconds": time.perf_counter() - epoch_start,
            "device": str(device),
            "gpu_name": gpu_name,
        }
        history_rows.append(row)
        append_csv(
            output_dir / "training_history.csv",
            [row],
            ["model", "epoch", "train_loss", "val_mean_recording_pearson_r", "elapsed_seconds", "device", "gpu_name"],
        )
        msg = f"EPOCH model={model_name} epoch={epoch} train_loss={row['train_loss']:.6f} val_r={val_metric:.6f}"
        print(msg, flush=True)
        append_log(output_dir, model_name, msg)
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(contract["patience"]):
            break
    if best_state is None:
        raise RuntimeError("no best model state selected")
    model.load_state_dict(best_state)
    model.to(device)
    test_recons, _ = eval_torch_recordings(model, cached["test"], contract, device, y_scaler)
    rec_rows = [
        make_recording_row(config, model_name, trial["recording_id"], f"best_epoch_{best_epoch}", pred, target, mask)
        for trial, pred, target, mask in test_recons
    ]
    return rec_rows, {
        "checkpoint_id": f"best_epoch_{best_epoch}",
        "best_epoch": best_epoch,
        "best_val_score": best_metric,
        "epochs_completed": len(history_rows),
        "device": str(device),
        "gpu_name": gpu_name,
        "input_shape": [102, contract["window_size"]] if contract["layout"] == "channels_time" else [contract["window_size"], 102],
        "scaler_fit_scope": "train_recordings_only",
    }


def make_recording_row(config: dict[str, Any], model: str, recording_id: str, checkpoint_id: str, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> RecordingMetricRow:
    return RecordingMetricRow(
        dataset=config["dataset_id"],
        model=model,
        task="audiobook_envelope_reconstruction",
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id="sub-03",
        recording_id=recording_id,
        sampling_rate=64,
        metric_name="pearson_r",
        metric_value=pearson_on_valid(pred, target, mask),
        num_valid_samples=int(mask.astype(bool).sum()),
        checkpoint_id=checkpoint_id,
        artifact_scope=config["artifact_scope"],
    )


def run_linear_family(model_name: str, config: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> tuple[list[RecordingMetricRow], dict[str, Any]]:
    cfg = config["linear_family"]
    x_scaler, y_scaler = fit_scalers(splits["train"])
    x_train, y_train = lagged_split(splits["train"], x_scaler, y_scaler, cfg["start_lag"], cfg["end_lag"])
    x_val, y_val = lagged_split(splits["val"], x_scaler, y_scaler, cfg["start_lag"], cfg["end_lag"])
    candidates: list[tuple[str, Any]] = []
    if model_name == "linear":
        candidates = [("linear_ols", LinearRegression())]
    elif model_name == "ridge":
        candidates = [(f"ridge_alpha_{a}", Ridge(alpha=float(a), solver="lsqr", tol=0.01, max_iter=80)) for a in cfg["ridge_alphas"]]
    elif model_name == "lasso":
        candidates = [(f"lasso_alpha_{a}", Lasso(alpha=float(a), max_iter=int(cfg["max_iter"]))) for a in cfg["lasso_alphas"]]
    elif model_name == "elasticnet":
        candidates = [(f"elasticnet_alpha_{a}_l1_{r}", ElasticNet(alpha=float(a), l1_ratio=float(r), max_iter=int(cfg["max_iter"]))) for a in cfg["elasticnet_alphas"] for r in cfg["elasticnet_l1_ratios"]]
    else:
        raise RuntimeError(model_name)
    best = None
    for checkpoint, model in candidates:
        model.fit(x_train, y_train)
        val_pred = y_scaler.inverse_transform(model.predict(x_val).reshape(-1, 1)).reshape(-1)
        val_target = y_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
        val_r = safe_pearson(val_pred, val_target)
        if best is None or val_r > best["val_r"]:
            best = {"checkpoint": checkpoint, "model": model, "val_r": val_r}
    assert best is not None
    rows = []
    valid_start = int(cfg["end_lag"]) - 1
    for trial in splits["test"]:
        x = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
        pred = y_scaler.inverse_transform(best["model"].predict(lag_sparse(x, cfg["start_lag"], cfg["end_lag"])).reshape(-1, 1)).reshape(-1).astype(np.float32)
        full = np.zeros(7680, dtype=np.float32)
        mask = np.zeros(7680, dtype=np.int32)
        full[valid_start:] = pred
        mask[valid_start:] = 1
        rows.append(make_recording_row(config, model_name, trial["recording_id"], best["checkpoint"], full, trial["target"], mask))
    return rows, {"checkpoint_id": best["checkpoint"], "val_pearson": best["val_r"]}


def preflight_model(model_name: str, config: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        x_scaler, y_scaler = fit_scalers(splits["train"])
        x0 = x_scaler.transform(splits["train"][0]["x_mag102"]).astype(np.float32)
        y0 = y_scaler.transform(splits["train"][0]["target"][:, None]).reshape(-1).astype(np.float32)
        if model_name in {"linear", "ridge", "lasso", "elasticnet"}:
            lagged = lag_sparse(x0, 0, 50)
            return {"model": model_name, "status": "ok", "input_shape": list(lagged.shape), "target_shape": [len(y0) - 49], "device_path": "cpu"}
        if model_name == "cca":
            return {"model": model_name, "status": "failed", "failure_reason": config["cca"]["reason"]}
        if model_name == "happyquokka":
            return {"model": model_name, "status": "failed", "failure_reason": config["happyquokka"]["reason"]}
        if model_name == "vlaai":
            return {"model": model_name, "status": "failed", "failure_reason": "VLAAI exact official path has a fixed internal 64-channel contract and no accepted mag102 adapter in this branch"}
        import torch
        if not torch.cuda.is_available():
            return {"model": model_name, "status": "failed", "failure_reason": "CUDA unavailable"}
        if model_name == "fcnn":
            model = FCNNBaseline(num_input_channels=102, input_length=50)
            sample = torch.zeros(2, 102, 50)
        elif model_name == "cnn":
            if UpstreamCNN is None:
                return {"model": model_name, "status": "failed", "failure_reason": "upstream CNN import unavailable"}
            model = UpstreamCNN(num_input_channels=102, input_length=50)
            sample = torch.zeros(2, 102, 50)
        elif model_name == "eegnet":
            model = EEGNetRegressor(num_input_channels=102, input_length=50)
            sample = torch.zeros(2, 102, 50)
        elif model_name == "adt":
            model = ADTExactRegressor(chans=102, seq_len=320)
            sample = torch.zeros(2, 320, 102)
        else:
            raise RuntimeError(model_name)
        with torch.no_grad():
            out = model(sample)
        target_shape = [2, 320, 1] if model_name == "adt" else [2]
        return {"model": model_name, "status": "ok", "input_shape": list(sample.shape), "output_shape": list(out.shape), "target_shape": target_shape, "device_path": "cuda"}
    except Exception as exc:
        return {"model": model_name, "status": "failed", "failure_reason": repr(exc)}


def run_placeholder_failure(model_name: str, reason: str) -> tuple[list[RecordingMetricRow], dict[str, Any]]:
    raise RuntimeError(reason)


def load_completed(output_dir: Path) -> set[str]:
    path = output_dir / "completed_jobs.json"
    if not path.exists():
        return set()
    return {item["job_key"] for item in read_json(path).get("completed_jobs", [])}


def write_incremental_state(output_dir: Path, config: dict[str, Any], completed: list[dict[str, Any]], failures: list[dict[str, Any]], entries: list[dict[str, Any]]) -> None:
    write_json(output_dir / "completed_jobs.json", {"completed_jobs": completed})
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_json(output_dir / "model_run_entries.json", entries)
    planned = [f"sub-03:{m}:seed0" for m in config["models"]]
    done = {j["job_key"] for j in completed} | {f["job_key"] for f in failures}
    write_json(output_dir / "run_state.json", {"planned_jobs": planned, "completed_or_failed": sorted(done), "pending_jobs": [j for j in planned if j not in done]})


def run_resume(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path, max_jobs: int | None) -> None:
    completed_keys = load_completed(output_dir)
    completed = read_json(output_dir / "completed_jobs.json").get("completed_jobs", []) if (output_dir / "completed_jobs.json").exists() else []
    failures = read_json(output_dir / "failure_report.json").get("failures", []) if (output_dir / "failure_report.json").exists() else []
    entries = read_json(output_dir / "model_run_entries.json") if (output_dir / "model_run_entries.json").exists() else []
    ran = 0
    for model_name in config["models"]:
        job_key = f"sub-03:{model_name}:seed0"
        if job_key in completed_keys or any(f["job_key"] == job_key for f in failures):
            print(f"JOB SKIP completed_or_failed {job_key}", flush=True)
            continue
        if max_jobs is not None and ran >= max_jobs:
            break
        print(f"JOB START {job_key}", flush=True)
        start = time.perf_counter()
        try:
            if model_name in {"linear", "ridge", "lasso", "elasticnet"}:
                rec_rows, meta = run_linear_family(model_name, config, splits)
            elif model_name == "cca":
                rec_rows, meta = run_placeholder_failure(model_name, config["cca"]["reason"])
            elif model_name == "happyquokka":
                rec_rows, meta = run_placeholder_failure(model_name, config["happyquokka"]["reason"])
            elif model_name == "vlaai":
                rec_rows, meta = run_placeholder_failure(model_name, "VLAAI exact official path has a fixed internal 64-channel contract and no accepted mag102 adapter in this branch")
            else:
                rec_rows, meta = run_torch_model(model_name, config, splits, output_dir)
            subj_rows = subject_metric_rows(rec_rows)
            append_csv(output_dir / "recording_metrics.csv", [r.__dict__ for r in rec_rows], RECORDING_METRIC_FIELDS)
            append_csv(output_dir / "subject_metrics.csv", [r.__dict__ for r in subj_rows], SUBJECT_METRIC_FIELDS)
            metric = float(subj_rows[0].metric_value)
            entry = {"job_key": job_key, "model": model_name, "status": "success", "run_type": "actual", "metric": metric, "elapsed_seconds": time.perf_counter() - start, **meta}
            entries.append(entry)
            completed.append({"job_key": job_key, "model": model_name, "metric": metric})
            print(f"JOB COMPLETE {job_key} metric={metric:.6f}", flush=True)
        except Exception as exc:
            failure = {"job_key": job_key, "model": model_name, "status": "failed", "failure_reason": repr(exc), "elapsed_seconds": time.perf_counter() - start}
            failures.append(failure)
            entries.append(failure)
            print(f"JOB FAILED {job_key} reason={repr(exc)}", flush=True)
        write_incremental_state(output_dir, config, completed, failures, entries)
        ran += 1


def main() -> int:
    args = parse_args()
    config = read_json(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)
    write_model_lock(config, output_dir)
    data = load_paired_mat(Path(config["paired_mat"]))
    validate_and_select(data, config, output_dir)
    splits = split_trials(data, config, output_dir)
    preflight = [preflight_model(model, config, splits) for model in config["models"]]
    write_json(output_dir / "shape_preflight.json", {"rows": preflight, "all_models_checked": len(preflight) == 11})
    write_json(output_dir / "run_manifest.json", {"protocol": config["protocol"], "models": config["models"], "status": "preflight_complete", "training_started": False})
    if args.preflight_only or args.dry_run:
        print(json.dumps({"status": "preflight_complete", "output_dir": str(output_dir), "preflight": preflight}, indent=2), flush=True)
        return 0
    if not args.resume:
        raise SystemExit("Use --resume for formal incremental execution.")
    write_json(output_dir / "run_manifest.json", {"protocol": config["protocol"], "models": config["models"], "status": "running_or_resumable", "training_started": True})
    run_resume(config, splits, output_dir, args.max_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
