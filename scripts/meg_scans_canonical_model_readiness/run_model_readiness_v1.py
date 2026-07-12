from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow, SubjectMetricRow, write_rows
from benchmark.scoring import pearson_on_valid, subject_metric_rows
from repro.mldecoders.models import EEGNetRegressor


RUN_LABELS = [
    "task-audiobook1_run-01",
    "task-audiobook1_run-02",
    "task-audiobook2_run-01",
    "task-audiobook2_run-02",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_matlab_string(f: h5py.File, ref: Any) -> str:
    data = np.array(f[ref]).squeeze()
    return "".join(chr(int(x)) for x in data)


def deref_cell(f: h5py.File, dataset: h5py.Dataset) -> list[Any]:
    refs = np.array(dataset).reshape(-1)
    return [f[ref] for ref in refs]


def load_paired_mat(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        results = f["results"]
        fs = int(np.array(results["epochs_neuro"]["fsample"]).squeeze())
        labels = [decode_matlab_string(f, ref) for ref in np.array(results["epochs_neuro"]["label"]).reshape(-1)]
        mapping_labels = [decode_matlab_string(f, ref) for ref in np.array(results["mapping_label"]).reshape(-1)]
        mapping_epochs = np.array(results["mapping_epochs"]).astype(int)
        if mapping_epochs.shape == (2, 4):
            mapping_epochs = mapping_epochs.T
        meg_trials = []
        for ds in deref_cell(f, results["epochs_neuro"]["trial"]):
            arr = np.array(ds, dtype=np.float32)
            if arr.shape == (7680, 306):
                meg_trials.append(arr)
            elif arr.shape == (306, 7680):
                meg_trials.append(arr.T)
            else:
                raise ValueError(f"unexpected MEG trial shape {arr.shape}")
        target_trials = []
        for ds in deref_cell(f, results["epochs_audio"]):
            target_trials.append(np.array(ds, dtype=np.float32).reshape(-1))
    trials = []
    for idx, (x, y) in enumerate(zip(meg_trials, target_trials), start=1):
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
    return {"fs": fs, "labels": labels, "mapping_labels": mapping_labels, "mapping_epochs": mapping_epochs, "trials": trials}


def validate_and_select(data: dict[str, Any], config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    inventory_path = ROOT / config["channel_inventory"]
    mag_path = ROOT / config["mag102_selection"]
    inventory = list(csv.DictReader(open(inventory_path, newline="", encoding="utf-8")))
    mag_selection = read_json(mag_path)
    labels = data["labels"]
    inventory_labels = [row["channel_label"] for row in inventory]
    if labels != inventory_labels:
        raise RuntimeError("all306 labels do not match channel_inventory original order")
    mag_indices = [int(v) - 1 for v in mag_selection["source_indices_1based"]]
    mag_labels = [labels[i] for i in mag_indices]
    digest = hashlib.sha256("\n".join(mag_labels).encode("utf-8")).hexdigest()
    if digest != config["expected_mag102_channel_order_sha256"]:
        raise RuntimeError(f"mag102 channel-order hash mismatch: {digest}")
    if len(mag_indices) != 102 or len(set(mag_indices)) != 102:
        raise RuntimeError("mag102 indices are not 102 unique entries")
    if len(labels) != 306 or len(set(labels)) != 306:
        raise RuntimeError("all306 labels are not 306 unique entries")

    for trial in data["trials"]:
        x = trial["x_all306"]
        y = trial["target"]
        if x.shape != (7680, 306):
            raise RuntimeError(f"bad all306 shape for {trial['recording_id']}: {x.shape}")
        if y.shape != (7680,):
            raise RuntimeError(f"bad target shape for {trial['recording_id']}: {y.shape}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError(f"NaN/Inf in {trial['recording_id']}")
        trial["x_mag102"] = x[:, mag_indices]

    check = {
        "status": "passed",
        "paired_mat": config["paired_mat"],
        "envelope_mat": config["envelope_mat"],
        "paired_sha256": sha256_file(Path(config["paired_mat"])),
        "envelope_sha256": sha256_file(Path(config["envelope_mat"])),
        "expected_paired_sha256": config["expected_paired_sha256"],
        "expected_envelope_sha256": config["expected_envelope_sha256"],
        "fs": data["fs"],
        "n_trials": len(data["trials"]),
        "all306_channel_count": len(labels),
        "mag102_channel_count": len(mag_indices),
        "mag102_channel_order_sha256": digest,
        "mapping_labels": data["mapping_labels"],
        "assertions": [
            "paired/envelope MAT SHA256 matched",
            "16 trials loaded",
            "each trial is 306 x 7680 before representation selection",
            "each target is 7680 samples",
            "fs is 64 Hz",
            "no NaN/Inf",
            "all306 label order matches channel_inventory",
            "mag102 label order hash matches mag102_selection",
        ],
    }
    if check["paired_sha256"] != config["expected_paired_sha256"]:
        raise RuntimeError("paired MAT SHA256 mismatch")
    if check["envelope_sha256"] != config["expected_envelope_sha256"]:
        raise RuntimeError("envelope MAT SHA256 mismatch")
    if data["fs"] != 64 or len(data["trials"]) != 16:
        raise RuntimeError("bad fs or trial count")
    write_json(output_dir / "data_integrity_check.json", check)
    return {"mag_indices_0based": mag_indices, "mag_labels": mag_labels, "all306_labels": labels, "integrity": check}


def split_trials(data: dict[str, Any], config: dict[str, Any], output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    lookup = {trial["global_trial_index"]: trial for trial in data["trials"]}
    split_cfg = config["split"]
    splits = {
        "train": [lookup[i] for i in split_cfg["train_global_trial_index"]],
        "val": [lookup[i] for i in split_cfg["val_global_trial_index"]],
        "test": [lookup[i] for i in split_cfg["test_global_trial_index"]],
    }
    seen: set[int] = set()
    for role, trials in splits.items():
        for trial in trials:
            idx = trial["global_trial_index"]
            if idx in seen:
                raise RuntimeError(f"split overlap at trial {idx}")
            seen.add(idx)
    rows = []
    for role, trials in splits.items():
        for trial in trials:
            rows.append({k: trial[k] for k in ["global_trial_index", "run_label", "trial_index_in_run", "recording_id"]} | {"role": role})
    write_json(
        output_dir / "split_manifest.json",
        {
            "identity": split_cfg["identity"],
            "seed": config["seed"],
            "train_global_trial_index": split_cfg["train_global_trial_index"],
            "val_global_trial_index": split_cfg["val_global_trial_index"],
            "test_global_trial_index": split_cfg["test_global_trial_index"],
            "recording_level_split": True,
            "no_window_random_split": True,
            "rows": rows,
        },
    )
    return splits


def fit_raw_scalers(train_trials: list[dict[str, Any]], representation: str) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = np.concatenate([trial[f"x_{representation}"] for trial in train_trials], axis=0)
    y_train = np.concatenate([trial["target"][:, None] for trial in train_trials], axis=0)
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)
    return x_scaler, y_scaler


def lag_sparse_recording(x: np.ndarray, start_lag: int, end_lag: int) -> sparse.csr_matrix:
    if start_lag != 0:
        raise ValueError("this readiness runner currently supports start_lag=0 only")
    n_time, n_ch = x.shape
    lags = list(range(start_lag, end_lag))
    valid_start = end_lag - 1
    n_valid = n_time - valid_start
    blocks = [sparse.csr_matrix(x[valid_start - lag : n_time - lag, :]) for lag in lags]
    mat = sparse.hstack(blocks, format="csr")
    if mat.shape != (n_valid, n_ch * len(lags)):
        raise RuntimeError(f"bad lag matrix shape {mat.shape}")
    return mat


def prepare_lagged_split(
    trials: list[dict[str, Any]],
    representation: str,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    start_lag: int,
    end_lag: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    xs = []
    ys = []
    valid_start = end_lag - 1
    for trial in trials:
        x = x_scaler.transform(trial[f"x_{representation}"]).astype(np.float32)
        y = y_scaler.transform(trial["target"][:, None]).reshape(-1).astype(np.float32)
        xs.append(lag_sparse_recording(x, start_lag, end_lag))
        ys.append(y[valid_start:])
    return sparse.vstack(xs, format="csr"), np.concatenate(ys)


def predict_recording_ridge(model: Ridge, trial: dict[str, Any], representation: str, x_scaler: StandardScaler, y_scaler: StandardScaler, start_lag: int, end_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_start = end_lag - 1
    x = x_scaler.transform(trial[f"x_{representation}"]).astype(np.float32)
    x_lag = lag_sparse_recording(x, start_lag, end_lag)
    pred_scaled = model.predict(x_lag).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1).astype(np.float32)
    full_pred = np.zeros(7680, dtype=np.float32)
    valid_mask = np.zeros(7680, dtype=np.int32)
    full_pred[valid_start:] = pred
    valid_mask[valid_start:] = 1
    return full_pred, trial["target"].astype(np.float32), valid_mask


def run_ridge(model_name: str, representation: str, config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[RecordingMetricRow], list[dict[str, Any]]]:
    start = time.perf_counter()
    log_lines = [f"{model_name} started"]
    ridge_cfg = config["ridge"]
    x_scaler, y_scaler = fit_raw_scalers(splits["train"], representation)
    x_train, y_train = prepare_lagged_split(splits["train"], representation, x_scaler, y_scaler, ridge_cfg["start_lag"], ridge_cfg["end_lag"])
    x_val, y_val = prepare_lagged_split(splits["val"], representation, x_scaler, y_scaler, ridge_cfg["start_lag"], ridge_cfg["end_lag"])
    lambda_rows = []
    best = None
    for alpha in ridge_cfg["alphas"]:
        model = Ridge(
            alpha=float(alpha),
            solver=str(ridge_cfg.get("solver", "lsqr")),
            tol=float(ridge_cfg.get("tol", 1e-4)),
            max_iter=int(ridge_cfg["max_iter"]) if ridge_cfg.get("max_iter") is not None else None,
        )
        model.fit(x_train, y_train)
        val_pred_scaled = model.predict(x_val).reshape(-1, 1)
        val_pred = y_scaler.inverse_transform(val_pred_scaled).reshape(-1)
        val_target = y_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
        val_r = safe_pearson(val_pred, val_target)
        lambda_rows.append({"model": model_name, "representation": representation, "alpha": alpha, "val_pearson_r": val_r})
        if best is None or val_r > best["val_r"]:
            best = {"alpha": float(alpha), "model": model, "val_r": val_r}
    assert best is not None
    predictions = {}
    recording_rows = []
    sorted_rows = []
    for trial in splits["test"]:
        pred, target, valid_mask = predict_recording_ridge(best["model"], trial, representation, x_scaler, y_scaler, ridge_cfg["start_lag"], ridge_cfg["end_lag"])
        predictions[trial["global_trial_index"]] = {"prediction": pred, "target": target, "valid_mask": valid_mask, "recording_id": trial["recording_id"]}
        recording_rows.append(make_recording_row(config, model_name, trial["recording_id"], f"alpha_{best['alpha']}", pred, target, valid_mask, "actual"))
    sorted_rows = build_sorted_mismatched_rows(config, model_name, predictions, config["mismatched_derangement"])
    elapsed = time.perf_counter() - start
    entry = {
        "model": model_name,
        "model_identity": "backward_trf_ridge",
        "representation": representation,
        "input_shape": f"[recording, {102 if representation == 'mag102' else 306}, 7680] -> lagged samples x channels*50",
        "seed": config["seed"],
        "split_identity": config["split"]["identity"],
        "scaler_fit_scope": "raw X scaler and target scaler fit on train recordings only",
        "training_budget": f"alphas={ridge_cfg['alphas']}; solver={ridge_cfg.get('solver', 'lsqr')}; tol={ridge_cfg.get('tol', 1e-4)}; max_iter={ridge_cfg.get('max_iter')}",
        "status": "actual",
        "elapsed_seconds": elapsed,
        "metric": mean_recording_metric(recording_rows),
        "selected_lambda": best["alpha"],
        "validation_pearson_r": best["val_r"],
        "sorted_vs_mismatched_delta": aggregate_delta(sorted_rows),
        "failure_reason": "",
    }
    (output_dir / f"{model_name}.log").write_text("\n".join(log_lines + [json.dumps(entry, indent=2)]) + "\n", encoding="utf-8")
    return entry, lambda_rows, recording_rows, sorted_rows


def make_recording_row(config: dict[str, Any], model: str, recording_id: str, checkpoint_id: str, pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray, artifact_scope: str) -> RecordingMetricRow:
    return RecordingMetricRow(
        dataset=config["dataset_id"],
        model=model,
        task="audiobook_envelope_reconstruction",
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=config["subject_id"],
        recording_id=recording_id,
        sampling_rate=int(config["sampling_rate"]),
        metric_name="pearson_r",
        metric_value=pearson_on_valid(pred, target, valid_mask),
        num_valid_samples=int(valid_mask.astype(bool).sum()),
        checkpoint_id=checkpoint_id,
        artifact_scope=artifact_scope,
    )


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def mean_recording_metric(rows: list[RecordingMetricRow]) -> float:
    vals = [float(row.metric_value) for row in rows if math.isfinite(float(row.metric_value))]
    return float(np.mean(vals)) if vals else float("nan")


def build_sorted_mismatched_rows(config: dict[str, Any], model: str, predictions: dict[int, dict[str, Any]], derangement: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for idx, item in sorted(predictions.items()):
        mismatch_idx = int(derangement[str(idx)])
        if mismatch_idx == idx:
            raise RuntimeError("mismatched derangement maps a test trial to itself")
        pred = item["prediction"]
        mask = item["valid_mask"]
        sorted_target = item["target"]
        mismatch_target = predictions[mismatch_idx]["target"]
        sorted_r = pearson_on_valid(pred, sorted_target, mask)
        mismatch_r = pearson_on_valid(pred, mismatch_target, mask)
        rows.append(
            {
                "dataset": config["dataset_id"],
                "model": model,
                "subject_id": config["subject_id"],
                "recording_id": item["recording_id"],
                "global_trial_index": idx,
                "mismatched_global_trial_index": mismatch_idx,
                "sorted_pearson_r": sorted_r,
                "mismatched_pearson_r": mismatch_r,
                "delta": sorted_r - mismatch_r,
            }
        )
    return rows


def aggregate_delta(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([float(row["delta"]) for row in rows])) if rows else float("nan")


def write_lambda_selection(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, ["model", "representation", "alpha", "val_pearson_r"])


def run_cca_skip(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    reason = (
        "skipped_existing_interface_unavailable: accepted CCA path "
        "src/repro/mldecoders/cca.py::fit_cca_reconstruction accepts concatenated x/y and internally constructs lags, "
        "which would cross canonical recording boundaries; this round forbids cross-recording lag construction and forbids silently replacing the CCA definition."
    )
    entry = {
        "model": "cca_mag102",
        "model_identity": "cca_reconstruction",
        "representation": "mag102",
        "input_shape": "[recording, 102, 7680]",
        "seed": config["seed"],
        "split_identity": config["split"]["identity"],
        "scaler_fit_scope": "not run",
        "training_budget": "not run",
        "status": "skipped",
        "elapsed_seconds": 0.0,
        "metric": None,
        "sorted_vs_mismatched_delta": None,
        "failure_reason": reason,
    }
    (output_dir / "cca_mag102.log").write_text(reason + "\n", encoding="utf-8")
    return entry


class WindowDataset:
    def __init__(self, trials: list[dict[str, Any]], representation: str, x_scaler: StandardScaler, y_scaler: StandardScaler, window: int) -> None:
        self.trials = trials
        self.representation = representation
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.window = window
        self.items = [(ti, start) for ti, trial in enumerate(trials) for start in range(0, len(trial["target"]) - window + 1)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        ti, start = self.items[index]
        trial = self.trials[ti]
        x = self.x_scaler.transform(trial[f"x_{self.representation}"][start : start + self.window]).T.astype(np.float32)
        y = self.y_scaler.transform(np.array([[trial["target"][start + self.window - 1]]], dtype=np.float32)).reshape(-1)[0]
        return x, np.float32(y)


def run_eegnet_smoke(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], list[RecordingMetricRow]]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class TorchWindowDataset(Dataset):
        def __init__(self, base: WindowDataset) -> None:
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            x, y = self.base[index]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    start = time.perf_counter()
    cfg = config["eegnet_smoke"]
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    x_scaler, y_scaler = fit_raw_scalers(splits["train"], "mag102")
    train_ds = TorchWindowDataset(WindowDataset(splits["train"], "mag102", x_scaler, y_scaler, cfg["input_length"]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    model = EEGNetRegressor(num_input_channels=102, input_length=int(cfg["input_length"]))
    if tuple(next(iter(train_loader))[0].shape[1:]) != (102, 50):
        raise RuntimeError("EEGNet smoke input is not [batch, 102, 50]")
    optimizer = torch.optim.NAdam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    model.train()
    losses = []
    for x, y in train_loader:
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    rows = []
    valid_start = int(cfg["input_length"]) - 1
    model.eval()
    with torch.no_grad():
        for trial in splits["test"]:
            preds = []
            x_scaled = x_scaler.transform(trial["x_mag102"]).astype(np.float32)
            for start_idx in range(0, 7680 - int(cfg["input_length"]) + 1, 512):
                stop_idx = min(start_idx + 512, 7680 - int(cfg["input_length"]) + 1)
                batch = np.stack([x_scaled[s : s + int(cfg["input_length"])].T for s in range(start_idx, stop_idx)], axis=0)
                pred_scaled = model(torch.tensor(batch, dtype=torch.float32)).detach().cpu().numpy()
                preds.append(pred_scaled)
            pred_scaled_all = np.concatenate(preds).reshape(-1, 1)
            pred = y_scaler.inverse_transform(pred_scaled_all).reshape(-1).astype(np.float32)
            full_pred = np.zeros(7680, dtype=np.float32)
            mask = np.zeros(7680, dtype=np.int32)
            full_pred[valid_start:] = pred
            mask[valid_start:] = 1
            rows.append(make_recording_row(config, "eegnet_mag102_smoke", trial["recording_id"], "epoch_1_smoke", full_pred, trial["target"], mask, "smoke_only"))
    elapsed = time.perf_counter() - start
    entry = {
        "model": "eegnet_mag102_smoke",
        "model_identity": "EEGNetRegressor",
        "representation": "mag102",
        "input_shape": "[batch, 102, 50] -> [batch]",
        "seed": config["seed"],
        "split_identity": config["split"]["identity"],
        "scaler_fit_scope": "raw X scaler and target scaler fit on train recordings only",
        "training_budget": "1 epoch smoke",
        "status": "smoke_only",
        "elapsed_seconds": elapsed,
        "metric": mean_recording_metric(rows),
        "sorted_vs_mismatched_delta": None,
        "failure_reason": "",
        "train_loss_mean": float(np.mean(losses)) if losses else None,
    }
    (output_dir / "eegnet_mag102_smoke.log").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry, rows


def write_reports(output_dir: Path, config: dict[str, Any], entries: list[dict[str, Any]], failures: list[dict[str, Any]], sorted_rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "run_manifest.json", {"protocol": config["protocol"], "status": "completed", "models": [e["model"] for e in entries], "output_dir": str(output_dir), "scope": "native MEG modeling readiness only"})
    write_json(output_dir / "model_run_entries.json", entries)
    write_json(output_dir / "failure_report.json", {"failures": failures, "skipped": [e for e in entries if e["status"] == "skipped"]})
    write_csv(
        output_dir / "sorted_vs_mismatched.csv",
        sorted_rows,
        ["dataset", "model", "subject_id", "recording_id", "global_trial_index", "mismatched_global_trial_index", "sorted_pearson_r", "mismatched_pearson_r", "delta"],
    )
    lines = [
        "# MEG-SCANS Canonical Model Readiness v1",
        "",
        "Scope: native-MEG sub-03 audiobook modeling readiness only; not a paper benchmark and not cross-modal transfer.",
        "",
        "## Model status",
    ]
    for entry in entries:
        lines.append(f"- `{entry['model']}`: status=`{entry['status']}`, metric=`{entry['metric']}`, delta=`{entry['sorted_vs_mismatched_delta']}`")
        if entry.get("failure_reason"):
            lines.append(f"  reason: {entry['failure_reason']}")
    lines.extend(["", "## Sorted vs mismatched", "Fixed derangement: 4 -> 8 -> 12 -> 16 -> 4."])
    output_dir.joinpath("result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_dir.joinpath("loader_contract_report.md").write_text(
        "\n".join(
            [
                "# Loader Contract",
                "",
                "- MAT files are read only.",
                "- Preflight SHA256 checks are mandatory before training.",
                "- Representations are selected from channel metadata: `mag102` uses mag102_selection hash, `all306` uses channel_inventory order.",
                "- Split is recording-level: train [1,2,5,6,9,10,13,14], val [3,7,11,15], test [4,8,12,16].",
                "- Lag/window construction is contained within each recording.",
                "- Scalers fit on train recordings only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("incremental_comparison.md").write_text(
        "This branch adds model-readiness scripts and compact outputs on top of canonical metadata commit 2837b68. It does not modify canonical MAT data or run OLSA/LOSO.\n",
        encoding="utf-8",
    )


def validate_schema_outputs(output_dir: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    required = [
        "data_integrity_check.json",
        "split_manifest.json",
        "loader_contract_report.md",
        "run_manifest.json",
        "model_run_entries.json",
        "subject_metrics.csv",
        "recording_metrics.csv",
        "sorted_vs_mismatched.csv",
        "result_summary.md",
        "failure_report.json",
        "incremental_comparison.md",
    ]
    checks = {f"exists_{name}": (output_dir / name).exists() for name in required}
    checks["model_entries_present"] = {e["model"] for e in entries} == {"ridge_mag102", "ridge_all306", "cca_mag102", "eegnet_mag102_smoke"}
    checks["ridge_actual"] = all(e["status"] == "actual" for e in entries if e["model"].startswith("ridge_"))
    checks["eegnet_smoke_only"] = next(e for e in entries if e["model"] == "eegnet_mag102_smoke")["status"] == "smoke_only"
    checks["cca_reasoned"] = next(e for e in entries if e["model"] == "cca_mag102")["status"] in {"actual", "skipped", "failed"}
    payload = {"passed": all(checks.values()), "checks": checks}
    write_json(output_dir / "schema_validation_report.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"schema validation failed: {payload}")
    return payload


def main() -> int:
    args = parse_args()
    config = read_json(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)
    data = load_paired_mat(Path(config["paired_mat"]))
    validate_and_select(data, config, output_dir)
    splits = split_trials(data, config, output_dir)

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    recording_rows: list[RecordingMetricRow] = []
    sorted_rows: list[dict[str, Any]] = []

    for model_name, representation in [("ridge_mag102", "mag102"), ("ridge_all306", "all306")]:
        try:
            entry, lambda_part, rec_part, sorted_part = run_ridge(model_name, representation, config, splits, output_dir)
            entries.append(entry)
            lambda_rows.extend(lambda_part)
            recording_rows.extend(rec_part)
            sorted_rows.extend(sorted_part)
        except Exception as exc:  # noqa: BLE001
            failure = {"model": model_name, "status": "failed", "error": repr(exc)}
            failures.append(failure)
            entries.append({"model": model_name, "status": "failed", "representation": representation, "metric": None, "sorted_vs_mismatched_delta": None, "failure_reason": repr(exc)})

    entries.append(run_cca_skip(config, output_dir))

    try:
        entry, rec_part = run_eegnet_smoke(config, splits, output_dir)
        entries.append(entry)
        recording_rows.extend(rec_part)
    except Exception as exc:  # noqa: BLE001
        failure = {"model": "eegnet_mag102_smoke", "status": "failed", "error": repr(exc)}
        failures.append(failure)
        entries.append({"model": "eegnet_mag102_smoke", "status": "failed", "representation": "mag102", "metric": None, "sorted_vs_mismatched_delta": None, "failure_reason": repr(exc)})

    write_lambda_selection(output_dir / "lambda_selection.csv", lambda_rows)
    write_rows(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
    subj_rows = subject_metric_rows(recording_rows)
    write_rows(output_dir / "subject_metrics.csv", subj_rows, SUBJECT_METRIC_FIELDS)
    write_reports(output_dir, config, entries, failures, sorted_rows)
    validate_schema_outputs(output_dir, entries)
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "models": entries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
