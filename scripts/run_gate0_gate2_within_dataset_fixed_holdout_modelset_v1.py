from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import Adam, NAdam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")
SHARED_UPSTREAM = SHARED_WORKSPACE / "external" / "upstream" / "mldecoders"
SHARED_HAPPYQUOKKA = SHARED_WORKSPACE / "external" / "upstream" / "HappyQuokka_system_for_EEG_Challenge"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SHARED_UPSTREAM.exists() and str(SHARED_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(SHARED_UPSTREAM))
if SHARED_HAPPYQUOKKA.exists() and str(SHARED_HAPPYQUOKKA) not in sys.path:
    sys.path.insert(0, str(SHARED_HAPPYQUOKKA))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row
from repro.adt_exact import ADTExactRegressor
from repro.mldecoders.models import EEGNetRegressor, VLAAIExactOfficialRegressor
from repro.reference_baselines import correlation as torch_correlation
from repro.reference_baselines import load_reference_recordings
from repro.simple_models import FCNNBaseline

try:
    from pipeline.dnn import CNN as UpstreamCNN
except Exception:
    UpstreamCNN = None

try:
    from models.FFT_block import Decoder as HappyQuokkaDecoder
except Exception:
    HappyQuokkaDecoder = None


MODELS = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "cca",
    "fcnn",
    "cnn",
    "eegnet",
    "adt",
    "vlaai",
    "happyquokka",
]
LINEAR_HELD = {"linear", "ridge", "lasso", "elasticnet", "cca"}
NONLINEAR_MODELS = {"fcnn", "cnn", "eegnet", "adt", "vlaai", "happyquokka"}

SMOKE_FIELDS = [
    "job_key",
    "stage",
    "dataset",
    "model",
    "seed",
    "split_id",
    "train_subject_count",
    "val_subject_count",
    "test_subject_count",
    "status",
]

PREFLIGHT_FIELDS = [
    "dataset",
    "model",
    "stage",
    "split_id",
    "train_subjects",
    "val_subjects",
    "test_subjects",
    "train_val_intersection",
    "train_test_intersection",
    "val_test_intersection",
    "first_train_recording",
    "first_train_eeg_shape",
    "first_train_target_shape",
    "input_contract",
    "scorer",
    "coverage_check",
    "checkpoint_path",
    "zero_shot_status",
]

SMOKE_RESULT_FIELDS = [
    "job_key",
    "dataset",
    "model",
    "seed",
    "status",
    "train_batch_loss",
    "val_forward_score",
    "test_subject_count",
    "test_recording_count",
    "test_mean_recording_pearson_r",
    "checkpoint_written",
    "notes",
]

DATASET_METRIC_FIELDS = [
    "dataset",
    "model",
    "stage",
    "seed",
    "n_subjects",
    "mean_pearson",
    "std_pearson",
    "median_pearson",
    "min_pearson",
    "max_pearson",
    "fisher_z_mean",
    "backtransformed_mean_r",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter.")
    parser.add_argument("--models", default="", help="Comma-separated model filter.")
    parser.add_argument("--stage", choices=["smoke", "zero_shot"], default="smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true", help="Run a capped zero_shot engineering proof in an isolated output directory.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def append_log(out_dir: Path, message: str) -> None:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "runner.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def parse_filter(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_candidate = SHARED_WORKSPACE / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def output_dir(config: dict[str, Any], stage: str) -> Path:
    if stage == "zero_shot" and bool(config.get("engineering_smoke", False)):
        return ROOT / config.get("zero_shot_engineering_smoke_output_dir", "experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot_engineering_smoke")
    key = "smoke_output_dir" if stage == "smoke" else "zero_shot_output_dir"
    return ROOT / config[key]


def load_completed_jobs(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "completed_jobs.json"
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("completed_jobs", [])
    return [item for item in payload if isinstance(item, dict)]


def completed_keys(out_dir: Path) -> set[str]:
    return {str(item.get("job_key", "")) for item in load_completed_jobs(out_dir) if item.get("status") == "success"}


def load_split(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    return load_json(ROOT / config["datasets"][dataset]["split_manifest_path"])


def validate_split(split: dict[str, Any], dataset: str) -> dict[str, Any]:
    train = set(split["train_subjects"])
    val = set(split["val_subjects"])
    test = set(split["test_subjects"])
    return {
        "dataset": dataset,
        "split_id": split["split_id"],
        "train_subjects": list(split["train_subjects"]),
        "val_subjects": list(split["val_subjects"]),
        "test_subjects": list(split["test_subjects"]),
        "train_val_intersection": sorted(train & val),
        "train_test_intersection": sorted(train & test),
        "val_test_intersection": sorted(val & test),
        "no_overlap": not (train & val or train & test or val & test) and bool(split.get("no_overlap", False)),
    }


def job_key(dataset: str, model: str, seed: int, stage: str) -> str:
    return f"{dataset}:{model}:seed{seed}:stage={stage}"


def checkpoint_stage(config: dict[str, Any], stage: str) -> str:
    if stage == "zero_shot" and bool(config.get("engineering_smoke", False)):
        return "zero_shot_engineering_smoke"
    return stage


def checkpoint_path(config: dict[str, Any], dataset: str, model: str, seed: int, stage: str) -> str:
    split_id = config["split_id"]
    return f"{config['checkpoint_dir']}/{checkpoint_stage(config, stage)}/{dataset}/{model}/{split_id}/seed{seed}/best_model.pt"


def job_checkpoint_path(config: dict[str, Any], job: dict[str, Any]) -> Path:
    return ROOT / checkpoint_path(config, str(job["dataset"]), str(job["model"]), int(job["seed"]), str(job["stage"]))


def build_jobs(config: dict[str, Any], stage: str, dataset_filter: set[str] | None, model_filter: set[str] | None) -> list[dict[str, Any]]:
    datasets = [item for item in config["datasets"] if dataset_filter is None or item in dataset_filter]
    models = [item for item in config["models"] if model_filter is None or item in model_filter]
    unknown_datasets = sorted((dataset_filter or set()) - set(config["datasets"]))
    unknown_models = sorted((model_filter or set()) - set(config["models"]))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}")
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        split = load_split(config, dataset)
        for model in models:
            jobs.append(
                {
                    "job_key": job_key(dataset, model, int(config["seed"]), stage),
                    "stage": stage,
                    "dataset": dataset,
                    "model": model,
                    "seed": int(config["seed"]),
                    "split_id": split["split_id"],
                    "train_subject_count": len(split["train_subjects"]),
                    "val_subject_count": len(split["val_subjects"]),
                    "test_subject_count": len(split["test_subjects"]),
                    "status": "pending",
                }
            )
    return jobs


def first_recording_shape(dataset_dir: Path, split_name: str, subject: str) -> tuple[str, str, str]:
    eeg_paths = sorted(dataset_dir.glob(f"{split_name}_-_{subject}_-_*_-_eeg.npy"))
    if not eeg_paths:
        return "", "", ""
    eeg_path = eeg_paths[0]
    recording_id = eeg_path.stem.replace("_-_eeg", "")
    env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
    eeg = np.load(eeg_path, mmap_mode="r")
    env = np.load(env_path, mmap_mode="r") if env_path.exists() else np.empty((0,))
    return recording_id, str(tuple(eeg.shape)), str(tuple(env.shape))


def build_preflight_rows(config: dict[str, Any], stage: str, jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset in config["datasets"]:
        split_check = validate_split(load_split(config, dataset), dataset)
        split_rows.append(split_check)
        if not split_check["no_overlap"]:
            failures.append(f"{dataset} split overlap check failed")
    for job in jobs:
        dataset = str(job["dataset"])
        model = str(job["model"])
        split = load_split(config, dataset)
        dataset_dir = resolve_dataset_path(config["datasets"][dataset]["dataset_locator"])
        first_id, eeg_shape, target_shape = first_recording_shape(dataset_dir, "train", split["train_subjects"][0])
        if not first_id:
            failures.append(f"{dataset}:{model} first train recording missing")
        zero_shot_status = "deferred_pending_linear_validation_fix_scope" if stage == "zero_shot" and model in LINEAR_HELD else "ready_for_execution"
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "stage": stage,
                "split_id": split["split_id"],
                "train_subjects": "|".join(split["train_subjects"]),
                "val_subjects": "|".join(split["val_subjects"]),
                "test_subjects": "|".join(split["test_subjects"]),
                "train_val_intersection": "|".join(sorted(set(split["train_subjects"]) & set(split["val_subjects"]))),
                "train_test_intersection": "|".join(sorted(set(split["train_subjects"]) & set(split["test_subjects"]))),
                "val_test_intersection": "|".join(sorted(set(split["val_subjects"]) & set(split["test_subjects"]))),
                "first_train_recording": first_id,
                "first_train_eeg_shape": eeg_shape,
                "first_train_target_shape": target_shape,
                "input_contract": config["contracts"][model]["input_contract"],
                "scorer": "recording-level Pearson over held-out test-subject recordings",
                "coverage_check": "train subjects batch, val subjects checkpoint selection, held-out test subjects aggregation",
                "checkpoint_path": checkpoint_path(config, dataset, model, int(config["seed"]), stage),
                "zero_shot_status": zero_shot_status,
            }
        )
    return rows, split_rows, failures


class MultiSubjectWindowDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, subjects: Iterable[str], *, window_size: int, channels: Iterable[int] = range(64), max_windows_per_subject: int | None = None) -> None:
        self.window_size = int(window_size)
        self.recordings: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self.records: list[tuple[int, int]] = []
        for subject in subjects:
            subject_records = load_reference_recordings(dataset_dir, split, subject, channels=channels)
            subject_record_indices: list[tuple[int, int]] = []
            for recording_id, eeg, env in subject_records:
                rec_idx = len(self.recordings)
                self.recordings.append((subject, recording_id, eeg, env))
                max_start = eeg.shape[0] - self.window_size
                if max_start < 0:
                    continue
                subject_record_indices.extend((rec_idx, start) for start in range(max_start + 1))
            if max_windows_per_subject is not None:
                subject_record_indices = subject_record_indices[: int(max_windows_per_subject)]
            self.records.extend(subject_record_indices)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self.records[idx]
        _, _, eeg, env = self.recordings[rec_idx]
        x = eeg[start : start + self.window_size].T
        y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)


class MultiSubjectSequenceDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, subjects: Iterable[str], *, window_size: int, hop_length: int, max_windows_per_subject: int | None = None) -> None:
        self.window_size = int(window_size)
        self.recordings: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self.records: list[tuple[int, int]] = []
        for subject in subjects:
            subject_record_indices: list[tuple[int, int]] = []
            for recording_id, eeg, env in load_reference_recordings(dataset_dir, split, subject, channels=range(64)):
                rec_idx = len(self.recordings)
                self.recordings.append((subject, recording_id, eeg, env))
                max_start = eeg.shape[0] - self.window_size
                if max_start >= 0:
                    subject_record_indices.extend((rec_idx, start) for start in range(0, max_start + 1, int(hop_length)))
            if max_windows_per_subject is not None:
                subject_record_indices = subject_record_indices[: int(max_windows_per_subject)]
            self.records.extend(subject_record_indices)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec_idx, start = self.records[idx]
        _, _, eeg, env = self.recordings[rec_idx]
        return eeg[start : start + self.window_size].astype(np.float32), env[start : start + self.window_size].astype(np.float32)


class MultiSubjectHappyQuokkaDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, subjects: Iterable[str], *, input_length: int, max_windows_per_subject: int | None = None) -> None:
        self.input_length = int(input_length)
        self.recordings: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self.records: list[tuple[int, int]] = []
        for subject in subjects:
            subject_indices: list[tuple[int, int]] = []
            for recording_id, eeg, env in load_reference_recordings(dataset_dir, split, subject, channels=range(64)):
                rec_idx = len(self.recordings)
                self.recordings.append((subject, recording_id, eeg, env))
                for start in range(0, eeg.shape[0] - self.input_length + 1, self.input_length):
                    subject_indices.append((rec_idx, start))
            if max_windows_per_subject is not None:
                subject_indices = subject_indices[: int(max_windows_per_subject)]
            self.records.extend(subject_indices)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rec_idx, start = self.records[idx]
        _, _, eeg, env = self.recordings[rec_idx]
        return eeg[start : start + self.input_length].astype(np.float32), env[start : start + self.input_length].astype(np.float32)[:, None]


def is_capped_engineering_run(config: dict[str, Any], stage: str) -> bool:
    return stage == "zero_shot" and bool(config.get("engineering_smoke", False))


def model_config(base: dict[str, Any], model: str, stage: str, *, engineering_smoke: bool = False) -> dict[str, Any]:
    cfg = deepcopy(base.get(model, {}))
    cfg["_engineering_smoke"] = bool(engineering_smoke)
    if stage == "smoke" or engineering_smoke:
        cfg["max_epochs"] = 1
        cfg["early_stopping_patience"] = 1
    return cfg


def build_model(model: str, cfg: dict[str, Any], device: str) -> torch.nn.Module:
    if model == "fcnn":
        return FCNNBaseline(num_hidden=int(cfg.get("hidden_layers", 3)), dropout_rate=float(cfg.get("dropout_rate", 0.45)), input_length=int(cfg["window_size"]), num_input_channels=64).to(device)
    if model == "cnn":
        if UpstreamCNN is None:
            raise RuntimeError("upstream CNN implementation unavailable")
        return UpstreamCNN(F1=int(cfg["F1"]), D=int(cfg["D"]), F2=int(cfg["F2"]), dropout_rate=float(cfg["dropout_rate"]), input_length=int(cfg["window_size"]), num_input_channels=64).to(device)
    if model == "eegnet":
        return EEGNetRegressor(num_input_channels=64, input_length=int(cfg["window_size"]), temporal_filters=int(cfg["temporal_filters"]), depth_multiplier=int(cfg["depth_multiplier"]), separable_filters=int(cfg["separable_filters"]), dropout_rate=float(cfg["dropout_rate"])).to(device)
    if model == "vlaai":
        return VLAAIExactOfficialRegressor(num_input_channels=64, input_length=int(cfg["window_size"])).to(device)
    if model == "adt":
        return ADTExactRegressor(seq_len=int(cfg["window_length"])).to(device)
    if model == "happyquokka":
        if HappyQuokkaDecoder is None:
            raise RuntimeError("HappyQuokka upstream Decoder unavailable")
        return HappyQuokkaDecoder(in_channel=64, d_model=128, d_inner=1024, n_head=2, n_layers=8, fft_conv1d_kernel=(9, 1), fft_conv1d_padding=(4, 0), dropout=float(cfg["dropout"]), g_con=bool(cfg.get("g_con", False)), within_sub_num=1).to(device)
    raise ValueError(model)


def train_forward(model: str, net: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = batch
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    if model in {"fcnn", "cnn", "eegnet", "vlaai"}:
        pred = net(x)
        loss = -torch_correlation(y, pred)
    elif model == "adt":
        pred = net(x).squeeze(-1)
        loss = -torch_correlation(y.flatten(), pred.flatten())
    elif model == "happyquokka":
        sub_ids = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        pred = net(x, sub_ids).squeeze(-1)
        loss = -torch_correlation(y.squeeze(-1).flatten(), pred.flatten())
    else:
        raise ValueError(model)
    return pred, y, loss


def make_loader(dataset: Dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator if shuffle else None)


def train_model(
    *,
    model: str,
    net: torch.nn.Module,
    cfg: dict[str, Any],
    dataset_dir: Path,
    split: dict[str, Any],
    stage: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    capped_engineering = is_capped_engineering_run({"engineering_smoke": cfg.get("_engineering_smoke", False)}, stage)
    capped = stage == "smoke" or capped_engineering
    if model in {"fcnn", "cnn", "eegnet", "vlaai"}:
        train_set = MultiSubjectWindowDataset(dataset_dir, "train", split["train_subjects"], window_size=int(cfg["window_size"]), max_windows_per_subject=256 if capped else None)
        val_set = MultiSubjectWindowDataset(dataset_dir, "val", split["val_subjects"], window_size=int(cfg["window_size"]), max_windows_per_subject=128 if capped else None)
        batch_size = int(cfg["batch_size"])
        optimizer = NAdam(net.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    elif model == "adt":
        train_set = MultiSubjectSequenceDataset(dataset_dir, "train", split["train_subjects"], window_size=int(cfg["window_length"]), hop_length=int(cfg["hop_length"]), max_windows_per_subject=32 if capped else None)
        val_set = MultiSubjectSequenceDataset(dataset_dir, "val", split["val_subjects"], window_size=int(cfg["window_length"]), hop_length=int(cfg["hop_length"]), max_windows_per_subject=16 if capped else None)
        batch_size = int(cfg["batch_size"])
        optimizer = Adam(net.parameters(), lr=float(cfg["learning_rate"]))
    elif model == "happyquokka":
        input_length = int(cfg["win_len_seconds"]) * int(cfg["sample_rate"])
        train_set = MultiSubjectHappyQuokkaDataset(dataset_dir, "train", split["train_subjects"], input_length=input_length, max_windows_per_subject=8 if capped else None)
        val_set = MultiSubjectHappyQuokkaDataset(dataset_dir, "val", split["val_subjects"], input_length=input_length, max_windows_per_subject=4 if capped else None)
        batch_size = int(cfg["batch_size"])
        optimizer = Adam(net.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(model)

    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError(f"empty train/val dataset for {model}")
    train_loader = make_loader(train_set, batch_size, seed, shuffle=True)
    val_loader = make_loader(val_set, min(1024, batch_size), seed, shuffle=False)
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg.get("early_stopping_patience", 10))
    best_score = -float("inf")
    best_epoch = 0
    best_state = deepcopy(net.state_dict())
    val_history: list[float] = []
    train_history: list[float] = []
    first_train_loss = float("nan")
    train_batches_completed = 0

    for epoch in range(max_epochs):
        net.train()
        losses = []
        max_train_batches = 1 if capped else None
        for batch_index, batch in enumerate(train_loader):
            _, _, loss = train_forward(model, net, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu().item())
            if np.isnan(first_train_loss):
                first_train_loss = value
            losses.append(value)
            train_batches_completed += 1
            if max_train_batches is not None and batch_index + 1 >= max_train_batches:
                break
        net.eval()
        scores = []
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                pred, y, _ = train_forward(model, net, batch, device)
                scores.append(float(torch_correlation(y.flatten(), pred.flatten()).detach().cpu().item()))
                if capped:
                    break
        val_score = float(np.mean(scores)) if scores else float("-inf")
        train_history.append(float(np.mean(losses)) if losses else float("nan"))
        val_history.append(val_score)
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            best_state = deepcopy(net.state_dict())
        elif stage != "smoke" and epoch + 1 > best_epoch + patience:
            break
    net.load_state_dict(best_state)
    return {
        "best_val_score": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(val_history),
        "train_history": train_history,
        "val_history": val_history,
        "first_train_loss": float(first_train_loss),
        "train_batches_completed": int(train_batches_completed),
        "state_dict": best_state,
    }


def window_prediction(dataset: str, model: str, protocol: str, seed: int, subject_id: str, recording_id: str, sampling_rate: int, checkpoint_id: str, full_length: int, offset: int, pred: np.ndarray, target: np.ndarray) -> WindowPrediction:
    return WindowPrediction(dataset=dataset, model=model, task="reconstruction", protocol=protocol, seed=seed, subject_id=subject_id, recording_id=recording_id, sampling_rate=sampling_rate, checkpoint_id=checkpoint_id, recording_length=int(full_length), start_index=int(offset), prediction=pred.astype(np.float32), target=target.astype(np.float32))


def evaluate_model(
    *,
    model: str,
    net: torch.nn.Module,
    cfg: dict[str, Any],
    config: dict[str, Any],
    dataset_id: str,
    dataset_dir: Path,
    subjects: list[str],
    sampling_rate: int,
    seed: int,
    checkpoint_id: str,
    stage: str,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    net.eval()
    windows: list[WindowPrediction] = []
    with torch.no_grad():
        capped = stage == "smoke" or is_capped_engineering_run(config, stage)
        for subject in subjects:
            for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject, channels=range(64)):
                if model in {"fcnn", "cnn", "eegnet", "vlaai"}:
                    input_length = int(cfg["window_size"])
                    max_windows = min(eeg.shape[0] - input_length + 1, 64 if capped else eeg.shape[0])
                    preds, targets = [], []
                    eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
                    for start in range(max(0, max_windows)):
                        batch = eeg_tensor[start : start + input_length].T.unsqueeze(0).to(device)
                        preds.append(float(net(batch).detach().cpu().item()))
                        targets.append(float(env[start + input_length - 1]))
                    if preds:
                        windows.append(window_prediction(dataset_id, model, config["protocol"], seed, subject, recording_id, sampling_rate, checkpoint_id, len(env), input_length - 1, np.asarray(preds), np.asarray(targets)))
                elif model == "adt":
                    input_length = int(cfg["window_length"])
                    hop = int(cfg["hop_length"])
                    starts = list(range(0, max(0, eeg.shape[0] - input_length + 1), hop))
                    if capped:
                        starts = starts[:4]
                    eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
                    for start in starts:
                        batch = eeg_tensor[start : start + input_length].unsqueeze(0).to(device)
                        pred = net(batch).squeeze(0).squeeze(-1).detach().cpu().numpy()
                        target = env[start : start + input_length]
                        windows.append(WindowPrediction(dataset_id, model, "reconstruction", config["protocol"], seed, subject, recording_id, sampling_rate, checkpoint_id, len(env), start, pred, target))
                elif model == "happyquokka":
                    input_length = int(cfg["win_len_seconds"]) * int(cfg["sample_rate"])
                    starts = list(range(0, eeg.shape[0] - input_length + 1, input_length))
                    if capped:
                        starts = starts[:2]
                    for start in starts:
                        batch = torch.from_numpy(eeg[start : start + input_length].astype(np.float32)).unsqueeze(0).to(device)
                        sub_id = torch.zeros(1, dtype=torch.long, device=device)
                        pred = net(batch, sub_id).squeeze(0).squeeze(-1).detach().cpu().numpy()
                        target = env[start : start + input_length]
                        windows.append(WindowPrediction(dataset_id, model, "reconstruction", config["protocol"], seed, subject, recording_id, sampling_rate, checkpoint_id, len(env), start, pred, target))
    recording_rows = []
    grouped: dict[tuple[str, str], list[WindowPrediction]] = {}
    for item in windows:
        grouped.setdefault((item.subject_id, item.recording_id), []).append(item)
    for (_, _), recording_windows in sorted(grouped.items()):
        first = recording_windows[0]
        pred, target, valid_mask = aggregate_overlapping_windows(first.recording_length, recording_windows)
        recording_rows.append(asdict(recording_metric_row(dataset=first.dataset, model=first.model, task=first.task, protocol=first.protocol, seed=first.seed, subject_id=first.subject_id, recording_id=first.recording_id, sampling_rate=first.sampling_rate, checkpoint_id=first.checkpoint_id, artifact_scope=config["artifact_scope"], prediction=pred, target=target, valid_mask=valid_mask)))
    subject_rows: list[dict[str, Any]] = []
    grouped_subject: dict[str, list[float]] = {}
    subject_recordings: dict[str, int] = {}
    for row in recording_rows:
        grouped_subject.setdefault(row["subject_id"], []).append(float(row["metric_value"]))
        subject_recordings[row["subject_id"]] = subject_recordings.get(row["subject_id"], 0) + 1
    for subject, values in sorted(grouped_subject.items()):
        first_row = next(row for row in recording_rows if row["subject_id"] == subject)
        subject_rows.append(
            {
                "dataset": first_row["dataset"],
                "model": first_row["model"],
                "task": first_row["task"],
                "protocol": first_row["protocol"],
                "seed": int(first_row["seed"]),
                "subject_id": subject,
                "metric_name": "mean_recording_pearson_r",
                "metric_value": float(np.mean(values)),
                "num_recordings": subject_recordings[subject],
                "checkpoint_id": first_row["checkpoint_id"],
                "artifact_scope": first_row["artifact_scope"],
            }
        )
    return recording_rows, subject_rows


def summarize_dataset_metrics(subject_rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in subject_rows:
        if row["metric_name"] == "mean_recording_pearson_r":
            grouped.setdefault((row["dataset"], row["model"], int(row["seed"])), []).append(float(row["metric_value"]))
    rows = []
    for (dataset, model, seed), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        clipped = np.clip(arr, -0.999999, 0.999999)
        fisher = np.arctanh(clipped)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "stage": stage,
                "seed": seed,
                "n_subjects": int(arr.size),
                "mean_pearson": float(np.mean(arr)),
                "std_pearson": float(np.std(arr, ddof=0)),
                "median_pearson": float(np.median(arr)),
                "min_pearson": float(np.min(arr)),
                "max_pearson": float(np.max(arr)),
                "fisher_z_mean": float(np.mean(fisher)),
                "backtransformed_mean_r": float(np.tanh(np.mean(fisher))),
            }
        )
    return rows


def load_support_config() -> dict[str, Any]:
    base = load_json(ROOT / "configs" / "benchmark" / "gate0_gate2_model_expansion_v1.json")
    smoke = load_json(ROOT / "configs" / "benchmark" / "gate0_gate2_subject_specific_local_model_smoke_v1.json")
    base["vlaai"] = smoke["vlaai"]
    base["happyquokka"] = smoke["happyquokka"]
    return base


def execute_job(config: dict[str, Any], base_config: dict[str, Any], job: dict[str, Any], device: str) -> dict[str, Any]:
    stage = str(job["stage"])
    dataset_id = str(job["dataset"])
    model = str(job["model"])
    seed = int(job["seed"])
    dataset_cfg = config["datasets"][dataset_id]
    dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
    split = load_split(config, dataset_id)
    sampling_rate = int(dataset_cfg["sampling_rate"])
    cfg = model_config(base_config, model, stage, engineering_smoke=is_capped_engineering_run(config, stage))
    net = build_model(model, cfg, device)
    train_summary = train_model(model=model, net=net, cfg=cfg, dataset_dir=dataset_dir, split=split, stage=stage, device=device, seed=seed)
    checkpoint_id = f"{model}_epoch_{train_summary['best_epoch']}"
    checkpoint_file = job_checkpoint_path(config, job)
    if stage == "zero_shot":
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": train_summary["state_dict"],
                "provenance": {
                    "job_key": job["job_key"],
                    "dataset": dataset_id,
                    "model": model,
                    "seed": seed,
                    "stage": stage,
                    "split_id": split["split_id"],
                    "train_subjects": split["train_subjects"],
                    "val_subjects": split["val_subjects"],
                    "test_subjects": split["test_subjects"],
                    "artifact_policy": "local_checkpoints only; do not commit",
                },
            },
            checkpoint_file,
        )
    recording_rows, subject_rows = evaluate_model(model=model, net=net, cfg=cfg, config=config, dataset_id=dataset_id, dataset_dir=dataset_dir, subjects=list(split["test_subjects"]), sampling_rate=sampling_rate, seed=seed, checkpoint_id=checkpoint_id, stage=stage, device=device)
    mean_metric = float(np.mean([float(row["metric_value"]) for row in subject_rows])) if subject_rows else float("nan")
    return {
        "job": {**job, "status": "success", "checkpoint_path": checkpoint_path(config, dataset_id, model, seed, stage) if stage == "zero_shot" else ""},
        "recording_rows": recording_rows,
        "subject_rows": subject_rows,
        "dataset_rows": summarize_dataset_metrics(subject_rows, stage),
        "smoke_result": {
            "job_key": job["job_key"],
            "dataset": dataset_id,
            "model": model,
            "seed": seed,
            "status": "success",
            "train_batch_loss": train_summary["first_train_loss"],
            "val_forward_score": train_summary["best_val_score"],
            "test_subject_count": len(split["test_subjects"]),
            "test_recording_count": len(recording_rows),
            "test_mean_recording_pearson_r": mean_metric,
            "checkpoint_written": stage == "zero_shot",
            "notes": "real train forward/backward/optimizer step, val forward, held-out test aggregation"
            + ("; engineering smoke evidence only, not a scientific result" if is_capped_engineering_run(config, stage) else ""),
        },
        "meta": {k: v for k, v in train_summary.items() if k != "state_dict"},
    }


def write_reports(config: dict[str, Any], stage: str, jobs: list[dict[str, Any]], pending: list[dict[str, Any]], preflight_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]], failures: list[str], device: str, *, training_started: bool = False, metrics_written: bool = False, completed_count: int = 0, status: str = "preflight_only") -> None:
    out_dir = output_dir(config, stage)
    write_csv(out_dir / f"{stage}_job_plan.csv", jobs, SMOKE_FIELDS)
    write_csv(out_dir / f"{stage}_execution_plan.csv", pending, SMOKE_FIELDS)
    write_csv(out_dir / "split_preflight.csv", split_rows, ["dataset", "split_id", "train_subjects", "val_subjects", "test_subjects", "train_val_intersection", "train_test_intersection", "val_test_intersection", "no_overlap"])
    write_csv(out_dir / "shape_checkpoint_preflight.csv", preflight_rows, PREFLIGHT_FIELDS)
    linear_deferred = [job for job in jobs if stage == "zero_shot" and job["model"] in LINEAR_HELD]
    expected_job_count = 22
    schema_passed = not failures
    schema = {
        "passed": schema_passed,
        "stage": stage,
        "engineering_smoke": bool(config.get("engineering_smoke", False)),
        "engineering_smoke_policy": "engineering evidence only; not a scientific result" if bool(config.get("engineering_smoke", False)) else "",
        "planned_job_count": len(jobs),
        "pending_execution_count": len(pending),
        "expected_full_stage_job_count": expected_job_count,
        "full_stage_plan_count_matches": len(jobs) == expected_job_count,
        "split_no_overlap": all(bool(row["no_overlap"]) for row in split_rows),
        "dnn_excluded_as_fcnn_alias": True,
        "linear_zero_shot_deferred_count": len(linear_deferred),
        "checkpoint_paths_local_only": True,
        "metrics_written": metrics_written,
        "training_started": training_started,
        "device": device,
        "failures": failures,
    }
    write_json(out_dir / "schema_validation_report.json", schema)
    if not (out_dir / "failure_report.json").exists():
        write_json(out_dir / "failure_report.json", {"failures": [{"failure_reason": item} for item in failures]})
    write_json(
        out_dir / "run_state.json",
        {
            "status": status,
            "stage": stage,
            "planned_job_count": len(jobs),
            "completed_job_count": completed_count,
            "pending_job_count": len(pending),
            "training_started": training_started,
            "metrics_written": metrics_written,
        },
    )
    lines = [
        f"# Within-Dataset Fixed Holdout Modelset {stage}",
        "",
        f"- stage: `{stage}`",
        f"- planned jobs: `{len(jobs)}`",
        f"- pending jobs after resume/max-jobs filtering: `{len(pending)}`",
        f"- training started: `{str(training_started).lower()}`",
        f"- metrics written: `{str(metrics_written).lower()}`",
        f"- engineering smoke: `{str(bool(config.get('engineering_smoke', False))).lower()}`",
        "- DNN excluded: `functional_alias_of=fcnn`",
        f"- linear-family zero-shot deferred jobs: `{len(linear_deferred)}`",
        "",
        "## Split Subjects",
    ]
    for row in split_rows:
        lines.append(f"- `{row['dataset']}` train=`{','.join(row['train_subjects'])}` val=`{','.join(row['val_subjects'])}` test=`{','.join(row['test_subjects'])}`")
    lines += [
        "",
        "## Manual Smoke Command",
        "```powershell",
        "cd E:\\decode\\_fix_fixed_split_pooled20_modelset_v1",
        "F:\\miniconda\\envs\\decode-torch\\python.exe scripts\\run_gate0_gate2_within_dataset_fixed_holdout_modelset_v1.py --config configs\\benchmark\\gate0_gate2_within_dataset_fixed_holdout_modelset_v1.json --stage smoke --device auto --models fcnn,cnn,eegnet,adt,vlaai,happyquokka --resume",
        "```",
    ]
    (out_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_execution(config: dict[str, Any], jobs: list[dict[str, Any]], pending: list[dict[str, Any]], stage: str, device: str) -> int:
    out_dir = output_dir(config, stage)
    base_config = load_support_config()
    completed = load_completed_jobs(out_dir)
    failures: list[dict[str, Any]] = []
    deferred_jobs: list[dict[str, Any]] = []
    if (out_dir / "failure_report.json").exists():
        previous = load_json(out_dir / "failure_report.json")
        failures.extend(item for item in previous.get("failures", []) if item.get("status") != "deferred")
    if (out_dir / "deferred_jobs.json").exists():
        previous_deferred = load_json(out_dir / "deferred_jobs.json")
        if isinstance(previous_deferred, dict):
            previous_deferred = previous_deferred.get("deferred_jobs", [])
        deferred_jobs.extend(item for item in previous_deferred if isinstance(item, dict))
    all_subject_rows: list[dict[str, Any]] = []
    all_recording_rows: list[dict[str, Any]] = []
    if stage == "zero_shot":
        for path, target in ((out_dir / "subject_metrics.csv", all_subject_rows), (out_dir / "recording_metrics.csv", all_recording_rows)):
            if path.exists():
                with path.open("r", encoding="utf-8", newline="") as handle:
                    target.extend(csv.DictReader(handle))
    smoke_rows: list[dict[str, Any]] = []
    if (out_dir / "smoke_job_results.csv").exists():
        with (out_dir / "smoke_job_results.csv").open("r", encoding="utf-8", newline="") as handle:
            smoke_rows.extend(csv.DictReader(handle))

    for index, job in enumerate(pending, start=1):
        model = str(job["model"])
        if model in LINEAR_HELD:
            reason = "linear-family execution deferred pending validation-fix scope confirmation"
            entry = {**job, "status": "deferred", "failure_reason": reason}
            deferred_jobs = [item for item in deferred_jobs if item.get("job_key") != job["job_key"]] + [entry]
            write_json(out_dir / "deferred_jobs.json", {"deferred_jobs": deferred_jobs})
            write_json(out_dir / "failure_report.json", {"failures": failures})
            append_log(out_dir, f"DEFERRED {job['job_key']} {reason}")
            continue
        if stage == "zero_shot" and model not in NONLINEAR_MODELS:
            reason = "zero_shot executable roster is currently restricted to nonlinear models"
            entry = {**job, "status": "deferred", "failure_reason": reason}
            deferred_jobs = [item for item in deferred_jobs if item.get("job_key") != job["job_key"]] + [entry]
            write_json(out_dir / "deferred_jobs.json", {"deferred_jobs": deferred_jobs})
            write_json(out_dir / "failure_report.json", {"failures": failures})
            continue
        append_log(out_dir, f"JOB START {index}/{len(pending)} {job['job_key']}")
        try:
            result = execute_job(config, base_config, job, device)
            completed = [item for item in completed if item.get("job_key") != job["job_key"]] + [result["job"]]
            if stage == "smoke":
                smoke_rows = [item for item in smoke_rows if item.get("job_key") != job["job_key"]] + [result["smoke_result"]]
                write_csv(out_dir / "smoke_job_results.csv", smoke_rows, SMOKE_RESULT_FIELDS)
            else:
                all_subject_rows = [row for row in all_subject_rows if not (row.get("dataset") == job["dataset"] and row.get("model") == job["model"] and str(row.get("seed")) == str(job["seed"]))] + result["subject_rows"]
                all_recording_rows = [row for row in all_recording_rows if not (row.get("dataset") == job["dataset"] and row.get("model") == job["model"] and str(row.get("seed")) == str(job["seed"]))] + result["recording_rows"]
                dataset_rows = summarize_dataset_metrics(all_subject_rows, stage)
                write_csv(out_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
                write_csv(out_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
                write_csv(out_dir / "dataset_metrics.csv", dataset_rows, DATASET_METRIC_FIELDS)
            write_json(out_dir / "completed_jobs.json", completed)
            write_json(out_dir / "failure_report.json", {"failures": failures})
            accounted = len(completed) + len(deferred_jobs)
            write_json(out_dir / "run_state.json", {"status": "completed_with_deferred" if accounted >= len(jobs) and deferred_jobs and not failures else ("completed" if accounted >= len(jobs) and not failures else "partial"), "stage": stage, "planned_job_count": len(jobs), "completed_job_count": len(completed), "deferred_job_count": len(deferred_jobs), "pending_job_count": max(0, len(jobs) - accounted), "last_completed_job_key": job["job_key"], "training_started": True, "metrics_written": stage == "zero_shot"})
            append_log(out_dir, f"JOB DONE {job['job_key']}")
        except Exception as exc:
            failures.append({**job, "status": "failed", "failure_reason": repr(exc)})
            write_json(out_dir / "failure_report.json", {"failures": failures})
            append_log(out_dir, f"JOB FAILED {job['job_key']} {repr(exc)}")
    write_json(out_dir / "completed_jobs.json", completed)
    write_json(out_dir / "failure_report.json", {"failures": failures})
    write_json(out_dir / "deferred_jobs.json", {"deferred_jobs": deferred_jobs})
    accounted = len(completed) + len(deferred_jobs)
    write_json(
        out_dir / "run_state.json",
        {
            "status": "completed_with_deferred" if accounted >= len(jobs) and deferred_jobs and not failures else ("completed" if accounted >= len(jobs) and not failures else "partial"),
            "stage": stage,
            "planned_job_count": len(jobs),
            "completed_job_count": len(completed),
            "deferred_job_count": len(deferred_jobs),
            "pending_job_count": max(0, len(jobs) - accounted),
            "last_completed_job_key": completed[-1]["job_key"] if completed else "",
            "training_started": bool(completed),
            "metrics_written": stage == "zero_shot" and bool(all_subject_rows),
        },
    )
    schema = load_json(out_dir / "schema_validation_report.json") if (out_dir / "schema_validation_report.json").exists() else {}
    schema.update({"training_started": bool(completed), "metrics_written": stage == "zero_shot" and bool(all_subject_rows), "completed_job_count": len(completed), "deferred_job_count": len(deferred_jobs), "pending_job_count": max(0, len(jobs) - accounted), "failure_count": len(failures), "nonlinear_executable_job_count": len([j for j in jobs if j["model"] in NONLINEAR_MODELS])})
    write_json(out_dir / "schema_validation_report.json", schema)
    return 0 if not [item for item in failures if item.get("status") == "failed"] else 1


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config))
    if args.engineering_smoke:
        if args.stage != "zero_shot":
            raise ValueError("--engineering-smoke is only valid with --stage zero_shot")
        config["engineering_smoke"] = True
        config["zero_shot_engineering_smoke_output_dir"] = "experiments/gate0_gate2_within_dataset_fixed_holdout_modelset_v1_zero_shot_engineering_smoke"
    device = resolve_device(args.device)
    dataset_filter = parse_filter(args.datasets)
    model_filter = parse_filter(args.models)
    jobs = build_jobs(config, args.stage, dataset_filter, model_filter)
    out_dir = output_dir(config, args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.startup_only:
        print(f"[STARTUP] config={args.config}", flush=True)
        print(f"[STARTUP] stage={args.stage} datasets={','.join(config['datasets'].keys())} models={len(config['models'])} device={device}", flush=True)
        print(f"[STARTUP] output_dir={out_dir.relative_to(ROOT).as_posix()} planned_jobs={len(jobs)}", flush=True)
        return 0
    completed = completed_keys(out_dir) if args.resume else set()
    pending = [job for job in jobs if job["job_key"] not in completed]
    if args.max_jobs is not None:
        pending = pending[: int(args.max_jobs)]
    preflight_rows, split_rows, failures = build_preflight_rows(config, args.stage, jobs)
    write_reports(config, args.stage, jobs, pending, preflight_rows, split_rows, failures, device)
    if args.dry_run_plan:
        print(f"[DRY RUN] stage={args.stage} planned_jobs={len(jobs)} completed={len(completed)} pending_execution={len(pending)}", flush=True)
        for job in pending:
            print(f"[DRY RUN] execution={job['job_key']}", flush=True)
        if args.stage == "zero_shot":
            held = [job["job_key"] for job in pending if job["model"] in LINEAR_HELD]
            print(f"[DRY RUN] zero_shot_deferred_linear_jobs={len(held)}", flush=True)
            for key in held:
                print(f"[DRY RUN] deferred={key}", flush=True)
        print(f"[PREFLIGHT] output_dir={out_dir.relative_to(ROOT).as_posix()} passed={not failures and len(jobs) == 22}", flush=True)
        return 0 if not failures else 1
    if failures:
        print(f"[PREFLIGHT] failed before execution failure_count={len(failures)}", flush=True)
        return 1
    print(f"[EXECUTION] stage={args.stage} jobs={len(pending)} device={device}", flush=True)
    return run_execution(config, jobs, pending, args.stage, device)


if __name__ == "__main__":
    raise SystemExit(main())
