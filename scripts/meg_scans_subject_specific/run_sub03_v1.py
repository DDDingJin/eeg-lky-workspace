from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from copy import deepcopy
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

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow, write_rows
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
    return [f[ref] for ref in np.array(dataset).reshape(-1)]


def load_paired_mat(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        results = f["results"]
        fs = int(np.array(results["epochs_neuro"]["fsample"]).squeeze())
        labels = [decode_matlab_string(f, ref) for ref in np.array(results["epochs_neuro"]["label"]).reshape(-1)]
        mapping_labels = [decode_matlab_string(f, ref) for ref in np.array(results["mapping_label"]).reshape(-1)]
        meg_trials = []
        for ds in deref_cell(f, results["epochs_neuro"]["trial"]):
            arr = np.array(ds, dtype=np.float32)
            if arr.shape == (306, 7680):
                arr = arr.T
            if arr.shape != (7680, 306):
                raise RuntimeError(f"unexpected MEG trial shape {arr.shape}")
            meg_trials.append(arr)
        target_trials = [np.array(ds, dtype=np.float32).reshape(-1) for ds in deref_cell(f, results["epochs_audio"])]
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
    return {"fs": fs, "labels": labels, "mapping_labels": mapping_labels, "trials": trials}


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
        raise RuntimeError("mag102 channel-order SHA256 mismatch")
    if len(labels) != 306 or len(mag_indices) != 102 or len(set(mag_indices)) != 102:
        raise RuntimeError("bad channel counts")

    for trial in data["trials"]:
        x = trial["x_all306"]
        y = trial["target"]
        if x.shape != (7680, 306) or y.shape != (7680,):
            raise RuntimeError(f"bad trial shape for {trial['recording_id']}: {x.shape}, {y.shape}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError(f"NaN/Inf in {trial['recording_id']}")
        trial["x_mag102"] = x[:, mag_indices]

    paired_sha = sha256_file(Path(config["paired_mat"]))
    envelope_sha = sha256_file(Path(config["envelope_mat"]))
    if paired_sha != config["expected_paired_sha256"] or envelope_sha != config["expected_envelope_sha256"]:
        raise RuntimeError("MAT SHA256 mismatch")
    if data["fs"] != 64 or len(data["trials"]) != 16:
        raise RuntimeError("bad fs or trial count")
    write_json(
        output_dir / "data_integrity_check.json",
        {
            "status": "passed",
            "paired_mat": config["paired_mat"],
            "paired_sha256": paired_sha,
            "envelope_mat": config["envelope_mat"],
            "envelope_sha256": envelope_sha,
            "fs": data["fs"],
            "n_trials": len(data["trials"]),
            "all306_channel_count": 306,
            "mag102_channel_count": 102,
            "mag102_channel_order_sha256": mag_hash,
            "assertions": [
                "SHA256 matched",
                "16 trials",
                "306 all channels",
                "102 magnetometers selected by channel hash",
                "each recording is 7680 samples",
                "no NaN/Inf",
            ],
        },
    )


def split_trials(data: dict[str, Any], config: dict[str, Any], output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    lookup = {trial["global_trial_index"]: trial for trial in data["trials"]}
    split_cfg = config["split"]
    splits = {
        "train": [lookup[i] for i in split_cfg["train_global_trial_index"]],
        "val": [lookup[i] for i in split_cfg["val_global_trial_index"]],
        "test": [lookup[i] for i in split_cfg["test_global_trial_index"]],
    }
    seen = set()
    rows = []
    for role, trials in splits.items():
        for trial in trials:
            idx = trial["global_trial_index"]
            if idx in seen:
                raise RuntimeError(f"split overlap at trial {idx}")
            seen.add(idx)
            rows.append({k: trial[k] for k in ["global_trial_index", "run_label", "trial_index_in_run", "recording_id"]} | {"role": role})
    if len(seen) != 16:
        raise RuntimeError("split does not cover all 16 trials")
    write_json(
        output_dir / "split_manifest.json",
        {
            "identity": split_cfg["identity"],
            "train_global_trial_index": split_cfg["train_global_trial_index"],
            "val_global_trial_index": split_cfg["val_global_trial_index"],
            "test_global_trial_index": split_cfg["test_global_trial_index"],
            "trial_is_atomic_recording_unit": True,
            "no_random_window_split": True,
            "rows": rows,
        },
    )
    return splits


def fit_raw_scalers(train_trials: list[dict[str, Any]], representation: str) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler().fit(np.concatenate([t[f"x_{representation}"] for t in train_trials], axis=0))
    y_scaler = StandardScaler().fit(np.concatenate([t["target"][:, None] for t in train_trials], axis=0))
    return x_scaler, y_scaler


def lag_sparse_recording(x: np.ndarray, start_lag: int, end_lag: int) -> sparse.csr_matrix:
    if start_lag != 0:
        raise RuntimeError("only start_lag=0 is supported in this adapter")
    valid_start = end_lag - 1
    blocks = [sparse.csr_matrix(x[valid_start - lag : x.shape[0] - lag, :]) for lag in range(start_lag, end_lag)]
    return sparse.hstack(blocks, format="csr")


def prepare_lagged(trials: list[dict[str, Any]], representation: str, x_scaler: StandardScaler, y_scaler: StandardScaler, start_lag: int, end_lag: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    xs, ys = [], []
    valid_start = end_lag - 1
    for trial in trials:
        x = x_scaler.transform(trial[f"x_{representation}"]).astype(np.float32)
        y = y_scaler.transform(trial["target"][:, None]).reshape(-1).astype(np.float32)
        xs.append(lag_sparse_recording(x, start_lag, end_lag))
        ys.append(y[valid_start:])
    return sparse.vstack(xs, format="csr"), np.concatenate(ys)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


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


def mean_metric(rows: list[RecordingMetricRow]) -> float:
    vals = [float(r.metric_value) for r in rows if math.isfinite(float(r.metric_value))]
    return float(np.mean(vals)) if vals else float("nan")


def predict_ridge_recording(model: Ridge, trial: dict[str, Any], representation: str, x_scaler: StandardScaler, y_scaler: StandardScaler, start_lag: int, end_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_start = end_lag - 1
    x = x_scaler.transform(trial[f"x_{representation}"]).astype(np.float32)
    pred_scaled = model.predict(lag_sparse_recording(x, start_lag, end_lag)).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1).astype(np.float32)
    full_pred = np.zeros(7680, dtype=np.float32)
    mask = np.zeros(7680, dtype=np.int32)
    full_pred[valid_start:] = pred
    mask[valid_start:] = 1
    return full_pred, trial["target"].astype(np.float32), mask


def sorted_mismatched_rows(config: dict[str, Any], model_name: str, predictions: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source_idx, item in sorted(predictions.items()):
        mismatch_idx = int(config["mismatched_derangement"][str(source_idx)])
        if mismatch_idx == source_idx:
            raise RuntimeError("mismatched target equals source")
        sorted_r = pearson_on_valid(item["prediction"], item["target"], item["valid_mask"])
        mismatch_r = pearson_on_valid(item["prediction"], predictions[mismatch_idx]["target"], item["valid_mask"])
        rows.append(
            {
                "dataset": config["dataset_id"],
                "model": model_name,
                "subject_id": config["subject_id"],
                "recording_id": item["recording_id"],
                "global_trial_index": source_idx,
                "mismatched_global_trial_index": mismatch_idx,
                "sorted_pearson_r": sorted_r,
                "mismatched_pearson_r": mismatch_r,
                "delta": sorted_r - mismatch_r,
            }
        )
    return rows


def run_ridge(model_name: str, representation: str, config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[RecordingMetricRow], list[dict[str, Any]]]:
    start = time.perf_counter()
    cfg = config["ridge"]
    x_scaler, y_scaler = fit_raw_scalers(splits["train"], representation)
    x_train, y_train = prepare_lagged(splits["train"], representation, x_scaler, y_scaler, cfg["start_lag"], cfg["end_lag"])
    x_val, y_val = prepare_lagged(splits["val"], representation, x_scaler, y_scaler, cfg["start_lag"], cfg["end_lag"])
    lambda_rows, best = [], None
    for alpha in cfg["alphas"]:
        model = Ridge(alpha=float(alpha), solver=cfg["solver"], tol=float(cfg["tol"]), max_iter=int(cfg["max_iter"]))
        model.fit(x_train, y_train)
        val_scaled = model.predict(x_val).reshape(-1, 1)
        val_pred = y_scaler.inverse_transform(val_scaled).reshape(-1)
        val_target = y_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
        val_r = safe_pearson(val_pred, val_target)
        lambda_rows.append({"model": model_name, "representation": representation, "alpha": alpha, "val_pearson_r": val_r})
        if best is None or val_r > best["val_r"]:
            best = {"alpha": float(alpha), "model": model, "val_r": val_r}
    assert best is not None
    rec_rows, predictions = [], {}
    for trial in splits["test"]:
        pred, target, mask = predict_ridge_recording(best["model"], trial, representation, x_scaler, y_scaler, cfg["start_lag"], cfg["end_lag"])
        predictions[trial["global_trial_index"]] = {"prediction": pred, "target": target, "valid_mask": mask, "recording_id": trial["recording_id"]}
        rec_rows.append(make_recording_row(config, model_name, trial["recording_id"], f"alpha_{best['alpha']}", pred, target, mask, "actual"))
    mismatch_rows = sorted_mismatched_rows(config, model_name, predictions)
    entry = {
        "model": model_name,
        "model_identity": "recording_safe_ridge_adapter",
        "representation": representation,
        "input_shape": f"[recording, {102 if representation == 'mag102' else 306}, 7680] -> recording-safe lagged samples x channels*50",
        "seed": config["seed"],
        "split_identity": config["split"]["identity"],
        "scaler_fit_scope": "train recordings only",
        "training_budget": f"alphas={cfg['alphas']}; solver={cfg['solver']}; tol={cfg['tol']}; max_iter={cfg['max_iter']}",
        "status": "actual",
        "elapsed_seconds": time.perf_counter() - start,
        "metric": mean_metric(rec_rows),
        "selected_lambda": best["alpha"],
        "validation_pearson_r": best["val_r"],
        "sorted_vs_mismatched_delta": float(np.mean([r["delta"] for r in mismatch_rows])),
        "failure_reason": "",
    }
    (output_dir / f"{model_name}.log").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry, lambda_rows, rec_rows, mismatch_rows


def cache_scaled_recordings(trials: list[dict[str, Any]], x_scaler: StandardScaler, y_scaler: StandardScaler) -> list[dict[str, Any]]:
    cached = []
    for trial in trials:
        cached.append(
            {
                "global_trial_index": trial["global_trial_index"],
                "recording_id": trial["recording_id"],
                "x": x_scaler.transform(trial["x_mag102"]).astype(np.float32),
                "y": y_scaler.transform(trial["target"][:, None]).reshape(-1).astype(np.float32),
                "target_raw": trial["target"].astype(np.float32),
            }
        )
    return cached


class CachedWindowDataset:
    def __init__(self, recordings: list[dict[str, Any]], window: int) -> None:
        self.recordings = recordings
        self.window = window
        self.items = [(ri, start) for ri, rec in enumerate(recordings) for start in range(0, len(rec["y"]) - window + 1)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        ri, start = self.items[index]
        rec = self.recordings[ri]
        return rec["x"][start : start + self.window].T, np.float32(rec["y"][start + self.window - 1])


def eval_eegnet_recordings(model, cached: list[dict[str, Any]], y_scaler: StandardScaler, config: dict[str, Any], artifact_scope: str) -> list[RecordingMetricRow]:
    import torch

    cfg = config["eegnet"]
    device = torch.device("cuda")
    window = int(cfg["input_length"])
    rows = []
    model.eval()
    with torch.no_grad():
        for rec in cached:
            preds = []
            n_windows = 7680 - window + 1
            for start_idx in range(0, n_windows, 1024):
                stop_idx = min(start_idx + 1024, n_windows)
                batch = np.stack([rec["x"][s : s + window].T for s in range(start_idx, stop_idx)], axis=0)
                pred_scaled = model(torch.tensor(batch, dtype=torch.float32, device=device)).detach().cpu().numpy()
                preds.append(pred_scaled)
            pred = y_scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).reshape(-1).astype(np.float32)
            full_pred = np.zeros(7680, dtype=np.float32)
            mask = np.zeros(7680, dtype=np.int32)
            full_pred[window - 1 :] = pred
            mask[window - 1 :] = 1
            rows.append(make_recording_row(config, "eegnet_mag102", rec["recording_id"], "best_val_checkpoint_in_memory", full_pred, rec["target_raw"], mask, artifact_scope))
    return rows


def run_eegnet(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[dict[str, Any], list[RecordingMetricRow], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class TorchDataset(Dataset):
        def __init__(self, base: CachedWindowDataset) -> None:
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int):
            x, y = self.base[index]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for eegnet_mag102 but is not available")
    start = time.perf_counter()
    cfg = config["eegnet"]
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    x_scaler, y_scaler = fit_raw_scalers(splits["train"], "mag102")
    train_cached = cache_scaled_recordings(splits["train"], x_scaler, y_scaler)
    val_cached = cache_scaled_recordings(splits["val"], x_scaler, y_scaler)
    test_cached = cache_scaled_recordings(splits["test"], x_scaler, y_scaler)
    train_ds = TorchDataset(CachedWindowDataset(train_cached, int(cfg["input_length"])))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    first_batch = next(iter(train_loader))[0]
    if tuple(first_batch.shape[1:]) != (102, 50):
        raise RuntimeError(f"EEGNet input shape is not [batch, 102, 50]: {tuple(first_batch.shape)}")
    model = EEGNetRegressor(num_input_channels=102, input_length=int(cfg["input_length"])).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    best_val, best_epoch, best_state = -np.inf, -1, deepcopy(model.state_dict())
    history = []
    log_path = output_dir / "eegnet_mag102.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"device=cuda gpu={gpu_name}\n")
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            model.train()
            losses = []
            for x, y in train_loader:
                x = x.to(device=device, non_blocking=False)
                y = y.to(device=device, non_blocking=False)
                pred = model(x)
                loss = torch.mean((pred - y) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            val_rows = eval_eegnet_recordings(model, val_cached, y_scaler, config, "validation_internal")
            val_score = mean_metric(val_rows)
            train_loss = float(np.mean(losses)) if losses else float("nan")
            improved = val_score > best_val
            if improved:
                best_val, best_epoch, best_state = val_score, epoch, deepcopy(model.state_dict())
            line = f"epoch={epoch} train_loss={train_loss:.6f} val_mean_recording_pearson={val_score:.6f} best_epoch={best_epoch}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            history.append({"model": "eegnet_mag102", "epoch": epoch, "train_loss": train_loss, "val_mean_recording_pearson": val_score, "best_epoch_so_far": best_epoch})
            if epoch >= best_epoch + int(cfg["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    test_rows = eval_eegnet_recordings(model, test_cached, y_scaler, config, "actual")
    elapsed = time.perf_counter() - start
    entry = {
        "model": "eegnet_mag102",
        "model_identity": "EEGNetRegressor",
        "representation": "mag102",
        "input_shape": "[batch, 102, 50] -> [batch]",
        "seed": config["seed"],
        "split_identity": config["split"]["identity"],
        "scaler_fit_scope": "train recordings only; train/val/test recordings cached after train-scaler transform",
        "training_budget": f"max_epochs={cfg['max_epochs']}; patience={cfg['early_stopping_patience']}; batch_size={cfg['batch_size']}",
        "status": "actual",
        "elapsed_seconds": elapsed,
        "metric": mean_metric(test_rows),
        "device": "cuda",
        "gpu_name": gpu_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "epochs_completed": len(history),
        "failure_reason": "",
    }
    with open(output_dir / "eegnet_mag102.log", "a", encoding="utf-8") as log:
        log.write(json.dumps(entry, indent=2) + "\n")
    return entry, test_rows, history


def write_reports(output_dir: Path, config: dict[str, Any], entries: list[dict[str, Any]], failures: list[dict[str, Any]], sorted_rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "run_manifest.json", {"protocol": config["protocol"], "branch": "fix/ar-20260712-meg-scans-subject-specific-sub03-v1", "status": "completed", "models": [e["model"] for e in entries], "scope": "subject-specific native MEG pilot only"})
    write_json(output_dir / "model_run_entries.json", entries)
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_csv(output_dir / "sorted_vs_mismatched.csv", sorted_rows, ["dataset", "model", "subject_id", "recording_id", "global_trial_index", "mismatched_global_trial_index", "sorted_pearson_r", "mismatched_pearson_r", "delta"])
    lines = ["# MEG-SCANS Subject-Specific sub-03 v1", "", "Scope: native MEG subject-specific pilot only; no OLSA/LOSO/CCA/cross-modal transfer.", "", "## Models"]
    for entry in entries:
        lines.append(f"- `{entry['model']}`: status=`{entry['status']}`, metric=`{entry.get('metric')}`, best_epoch=`{entry.get('best_epoch', '')}`")
        if entry.get("failure_reason"):
            lines.append(f"  reason: {entry['failure_reason']}")
    output_dir.joinpath("result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_schema_outputs(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    required = ["run_manifest.json", "data_integrity_check.json", "split_manifest.json", "model_run_entries.json", "recording_metrics.csv", "subject_metrics.csv", "sorted_vs_mismatched.csv", "training_history.csv", "result_summary.md", "failure_report.json"]
    checks = {name: (output_dir / name).exists() for name in required}
    checks["model_set"] = {e["model"] for e in entries} == {"recording_safe_ridge_adapter_mag102", "recording_safe_ridge_adapter_all306", "eegnet_mag102"}
    checks["all_actual"] = all(e["status"] == "actual" for e in entries)
    write_json(output_dir / "schema_validation_report.json", {"passed": all(checks.values()), "checks": checks})
    if not all(checks.values()):
        raise RuntimeError(f"schema validation failed: {checks}")


def main() -> int:
    args = parse_args()
    config = read_json(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)
    data = load_paired_mat(Path(config["paired_mat"]))
    validate_and_select(data, config, output_dir)
    splits = split_trials(data, config, output_dir)
    entries, failures, lambda_rows, recording_rows, sorted_rows, history_rows = [], [], [], [], [], []
    for model_name, rep in [("recording_safe_ridge_adapter_mag102", "mag102"), ("recording_safe_ridge_adapter_all306", "all306")]:
        entry, lambda_part, rec_part, sorted_part = run_ridge(model_name, rep, config, splits, output_dir)
        entries.append(entry)
        lambda_rows.extend(lambda_part)
        recording_rows.extend(rec_part)
        sorted_rows.extend(sorted_part)
    entry, eegnet_rows, history_rows = run_eegnet(config, splits, output_dir)
    entries.append(entry)
    recording_rows.extend(eegnet_rows)
    write_csv(output_dir / "lambda_selection.csv", lambda_rows, ["model", "representation", "alpha", "val_pearson_r"])
    write_csv(output_dir / "training_history.csv", history_rows, ["model", "epoch", "train_loss", "val_mean_recording_pearson", "best_epoch_so_far"])
    write_rows(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", subject_metric_rows(recording_rows), SUBJECT_METRIC_FIELDS)
    write_reports(output_dir, config, entries, failures, sorted_rows)
    validate_schema_outputs(output_dir, entries)
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "models": entries}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
