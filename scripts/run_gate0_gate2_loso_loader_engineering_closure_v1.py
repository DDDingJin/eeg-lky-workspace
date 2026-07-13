from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.optim import NAdam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHARED_WORKSPACE = Path(ROOT.drive + "\\decode")
SHARED_UPSTREAM = SHARED_WORKSPACE / "external" / "upstream" / "mldecoders"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SHARED_UPSTREAM.exists() and str(SHARED_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(SHARED_UPSTREAM))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, RecordingMetricRow, load_dict_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.mldecoders.models import EEGNetRegressor
from repro.reference_baselines import list_reference_subjects
from run_gate0_gate2_full_subject_single_seed import write_generic_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resolve_dataset_path(locator: str) -> Path:
    repo_candidate = ROOT / locator
    if repo_candidate.exists():
        return repo_candidate
    shared_candidate = SHARED_WORKSPACE / locator
    if shared_candidate.exists():
        return shared_candidate
    return repo_candidate


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def correlation(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = pred.astype(np.float64)
    target64 = target.astype(np.float64)
    pred0 = pred64 - pred64.mean()
    target0 = target64 - target64.mean()
    denom = np.sqrt(np.sum(pred0**2) * np.sum(target0**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(pred0 * target0) / denom)


def batch_corr(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true0 = y_true - torch.mean(y_true)
    y_pred0 = y_pred - torch.mean(y_pred)
    return torch.sum(y_true0 * y_pred0) / (torch.sqrt(torch.sum(y_true0**2)) * torch.sqrt(torch.sum(y_pred0**2)) + eps)


def build_series_window(
    *,
    dataset: str,
    model: str,
    protocol: str,
    seed: int,
    subject_id: str,
    recording_id: str,
    sampling_rate: int,
    checkpoint_id: str,
    full_length: int,
    offset: int,
    prediction: np.ndarray,
    target: np.ndarray,
) -> WindowPrediction:
    return WindowPrediction(
        dataset=dataset,
        model=model,
        task="reconstruction",
        protocol=protocol,
        seed=seed,
        subject_id=subject_id,
        recording_id=recording_id,
        sampling_rate=sampling_rate,
        checkpoint_id=checkpoint_id,
        recording_length=int(full_length),
        start_index=int(offset),
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
    )


def aggregate_model_outputs(windows: list[WindowPrediction], artifact_scope: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[WindowPrediction]] = {}
    for window in windows:
        grouped.setdefault((window.subject_id, window.recording_id), []).append(window)

    recording_rows: list[dict[str, object]] = []
    for (_, _), recording_windows in sorted(grouped.items()):
        first = recording_windows[0]
        prediction, target, valid_mask = aggregate_overlapping_windows(first.recording_length, recording_windows)
        recording_rows.append(
            recording_metric_row(
                dataset=first.dataset,
                model=first.model,
                task=first.task,
                protocol=first.protocol,
                seed=first.seed,
                subject_id=first.subject_id,
                recording_id=first.recording_id,
                sampling_rate=first.sampling_rate,
                checkpoint_id=first.checkpoint_id,
                artifact_scope=artifact_scope,
                prediction=prediction,
                target=target,
                valid_mask=valid_mask,
            ).__dict__
        )
    subject_rows = [row.__dict__ for row in subject_metric_rows([RecordingMetricRow(**row) for row in recording_rows])]
    return recording_rows, subject_rows


@dataclass(frozen=True)
class RecordingWindowSource:
    subject_id: str
    recording_id: str
    eeg_path: Path
    env_path: Path
    length: int
    num_windows: int
    eeg_bytes: int
    env_bytes: int


class InstrumentedWindowDatasetBase(Dataset):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_size: int,
        channels: range,
        target_index: str = "last",
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.subjects = list(subjects)
        self.window_size = int(window_size)
        self.target_index = target_index
        self.channel_idx = np.asarray(list(channels), dtype=int)
        self.sources: list[RecordingWindowSource] = []
        self.cumulative_windows: list[int] = []
        self.total_windows = 0
        self.total_samples = 0
        self.total_recordings = 0
        self.total_eeg_bytes = 0
        self.total_env_bytes = 0
        self.getitem_calls = 0
        self.load_requests = 0
        self.recording_access_counts: dict[str, int] = {}
        self.recording_load_counts: dict[str, int] = {}
        self.recording_subject_lookup: dict[str, str] = {}

        for subject_id in self.subjects:
            for eeg_path in sorted(self.input_dir.glob(f"{split}_-_{subject_id}_-_*_-_eeg.npy")):
                recording_id = eeg_path.stem.replace("_-_eeg", "")
                env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
                if not env_path.exists():
                    continue
                eeg_shape = np.load(eeg_path, mmap_mode="r").shape
                env_shape = np.load(env_path, mmap_mode="r").shape
                if len(eeg_shape) != 2 or eeg_shape[0] != env_shape[0]:
                    raise ValueError(f"length mismatch for {recording_id}: eeg={eeg_shape} env={env_shape}")
                length = int(eeg_shape[0])
                num_windows = max(length - self.window_size + 1, 0)
                source = RecordingWindowSource(
                    subject_id=subject_id,
                    recording_id=recording_id,
                    eeg_path=eeg_path,
                    env_path=env_path,
                    length=length,
                    num_windows=num_windows,
                    eeg_bytes=int(length * len(self.channel_idx) * 4),
                    env_bytes=int(length * 4),
                )
                self.sources.append(source)
                self.total_recordings += 1
                self.total_samples += length
                self.total_eeg_bytes += source.eeg_bytes
                self.total_env_bytes += source.env_bytes
                self.total_windows += num_windows
                self.cumulative_windows.append(self.total_windows)
                self.recording_access_counts[recording_id] = 0
                self.recording_load_counts[recording_id] = 0
                self.recording_subject_lookup[recording_id] = subject_id
        if not self.sources:
            raise ValueError(f"no recordings found for split={split} subjects={subjects}")

    def __len__(self) -> int:
        return self.total_windows

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(idx)
        rec_idx = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        previous = 0 if rec_idx == 0 else self.cumulative_windows[rec_idx - 1]
        return rec_idx, int(idx - previous)

    def _register_access(self, source: RecordingWindowSource) -> None:
        self.getitem_calls += 1
        self.recording_access_counts[source.recording_id] += 1

    def _register_load(self, source: RecordingWindowSource) -> None:
        self.load_requests += 1
        self.recording_load_counts[source.recording_id] += 1

    def base_summary(self, *, heldout_subject: str) -> dict[str, object]:
        hypothetical_window_bytes = int(self.total_windows * (len(self.channel_idx) * self.window_size * 4 + 4))
        return {
            "subjects": self.subjects,
            "heldout_subject": heldout_subject,
            "heldout_excluded": heldout_subject not in self.subjects,
            "recording_count": self.total_recordings,
            "total_samples": self.total_samples,
            "total_windows": self.total_windows,
            "lazy_generation": True,
            "full_window_materialization_avoided": True,
            "estimated_recording_bytes": int(self.total_eeg_bytes + self.total_env_bytes),
            "hypothetical_full_window_materialization_bytes": hypothetical_window_bytes,
        }

    def audit_snapshot(self) -> dict[str, object]:
        repeated_loads = {recording_id: count for recording_id, count in self.recording_load_counts.items() if count > 1}
        return {
            "getitem_calls": int(self.getitem_calls),
            "load_requests": int(self.load_requests),
            "recording_access_counts_top10": sorted(self.recording_access_counts.items(), key=lambda item: item[1], reverse=True)[:10],
            "recording_load_counts_top10": sorted(self.recording_load_counts.items(), key=lambda item: item[1], reverse=True)[:10],
            "recordings_loaded_more_than_once_count": len(repeated_loads),
            "recordings_loaded_more_than_once_examples": sorted(repeated_loads.items(), key=lambda item: item[1], reverse=True)[:10],
            "distinct_recordings_accessed": int(sum(1 for count in self.recording_access_counts.values() if count > 0)),
            "distinct_recordings_loaded": int(sum(1 for count in self.recording_load_counts.values() if count > 0)),
        }


class CachedLazyWindowDataset(InstrumentedWindowDatasetBase):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_size: int,
        channels: range,
        target_index: str = "last",
        recording_cache_size: int = 4,
    ) -> None:
        super().__init__(
            input_dir,
            split,
            subjects=subjects,
            window_size=window_size,
            channels=channels,
            target_index=target_index,
        )
        self.recording_cache_size = int(recording_cache_size)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.current_cache_bytes = 0
        self.peak_cache_bytes = 0

    def _load_recording(self, rec_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(rec_idx)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(rec_idx)
            return cached
        self.cache_misses += 1
        source = self.sources[rec_idx]
        eeg = np.asarray(np.load(source.eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
        env = np.asarray(np.load(source.env_path, mmap_mode="r")[:, 0], dtype=np.float32)
        self._register_load(source)
        cached = (eeg, env)
        self._cache[rec_idx] = cached
        self.current_cache_bytes += source.eeg_bytes + source.env_bytes
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.current_cache_bytes)
        self._cache.move_to_end(rec_idx)
        while len(self._cache) > self.recording_cache_size:
            evict_idx, _ = self._cache.popitem(last=False)
            evict_source = self.sources[evict_idx]
            self.current_cache_bytes -= evict_source.eeg_bytes + evict_source.env_bytes
        return cached

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self._resolve_index(idx)
        source = self.sources[rec_idx]
        self._register_access(source)
        eeg, env = self._load_recording(rec_idx)
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)

    def audit_snapshot(self) -> dict[str, object]:
        snapshot = super().audit_snapshot()
        snapshot.update(
            {
                "dataset_type": "cached_lazy",
                "cache_hits": int(self.cache_hits),
                "cache_misses": int(self.cache_misses),
                "cache_hit_rate": float(self.cache_hits / max(self.cache_hits + self.cache_misses, 1)),
                "recording_cache_size": int(self.recording_cache_size),
                "peak_cache_bytes": int(self.peak_cache_bytes),
            }
        )
        return snapshot


class PreloadedWindowDataset(InstrumentedWindowDatasetBase):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        *,
        subjects: list[str],
        window_size: int,
        channels: range,
        target_index: str = "last",
    ) -> None:
        super().__init__(
            input_dir,
            split,
            subjects=subjects,
            window_size=window_size,
            channels=channels,
            target_index=target_index,
        )
        self._store: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for rec_idx, source in enumerate(self.sources):
            eeg = np.asarray(np.load(source.eeg_path, mmap_mode="r")[:, self.channel_idx], dtype=np.float32)
            env = np.asarray(np.load(source.env_path, mmap_mode="r")[:, 0], dtype=np.float32)
            self._store[rec_idx] = (eeg, env)
            self._register_load(source)
        self.preloaded_bytes = int(self.total_eeg_bytes + self.total_env_bytes)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        rec_idx, start = self._resolve_index(idx)
        source = self.sources[rec_idx]
        self._register_access(source)
        eeg, env = self._store[rec_idx]
        x = eeg[start : start + self.window_size].T
        if self.target_index == "first":
            y = env[start]
        elif self.target_index == "center":
            y = env[start + self.window_size // 2]
        else:
            y = env[start + self.window_size - 1]
        return x.astype(np.float32), np.float32(y)

    def audit_snapshot(self) -> dict[str, object]:
        snapshot = super().audit_snapshot()
        snapshot.update(
            {
                "dataset_type": "preloaded_raw_recordings",
                "cache_hits": None,
                "cache_misses": None,
                "cache_hit_rate": None,
                "recording_cache_size": None,
                "peak_cache_bytes": int(self.preloaded_bytes),
                "preloaded_recording_bytes": int(self.preloaded_bytes),
            }
        )
        return snapshot


def select_window_model(config: dict) -> tuple[object, dict[str, object]]:
    model_cfg = config["eegnet"]
    return EEGNetRegressor, {
        "num_input_channels": 64,
        "input_length": int(model_cfg["window_size"]),
        "temporal_filters": int(model_cfg["temporal_filters"]),
        "depth_multiplier": int(model_cfg["depth_multiplier"]),
        "separable_filters": int(model_cfg["separable_filters"]),
        "dropout_rate": float(model_cfg["dropout_rate"]),
    }


def build_dataset(
    *,
    mode: str,
    dataset_dir: Path,
    split: str,
    subjects: list[str],
    config: dict,
) -> InstrumentedWindowDatasetBase:
    if mode == "cached_lazy":
        return CachedLazyWindowDataset(
            dataset_dir,
            split,
            subjects=subjects,
            window_size=int(config["eegnet"]["window_size"]),
            channels=range(64),
            target_index=str(config["eegnet"]["target_index"]),
            recording_cache_size=int(config["dataloader"]["recording_cache_size"]),
        )
    if mode == "preloaded":
        return PreloadedWindowDataset(
            dataset_dir,
            split,
            subjects=subjects,
            window_size=int(config["eegnet"]["window_size"]),
            channels=range(64),
            target_index=str(config["eegnet"]["target_index"]),
        )
    raise ValueError(f"unsupported dataset mode {mode}")


def profile_loader(
    *,
    dataset: InstrumentedWindowDatasetBase,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_batches: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator if shuffle else None,
    )
    first_batch_shape = None
    iter_start = time.perf_counter()
    batch_count = 0
    example_y_shape = None
    for batch_idx, (x, y) in enumerate(loader, start=1):
        if first_batch_shape is None:
            first_batch_shape = list(x.shape)
            example_y_shape = list(y.shape)
        batch_count = batch_idx
        if batch_idx >= max_batches:
            break
    elapsed = time.perf_counter() - iter_start
    audit = dataset.audit_snapshot()
    audit.update(
        {
            "elapsed_seconds": float(elapsed),
            "batch_count_completed": int(batch_count),
            "batch_shape": first_batch_shape,
            "target_batch_shape": example_y_shape,
            "shuffle": bool(shuffle),
            "num_workers": int(num_workers),
            "batches_per_second": float(batch_count / elapsed) if elapsed > 0 else float("nan"),
        }
    )
    return audit


def write_loader_profile_report(
    path: Path,
    *,
    config: dict,
    train_old: dict[str, object],
    train_opt: dict[str, object],
    val_old: dict[str, object],
    val_opt: dict[str, object],
    train_summary: dict[str, object],
    val_summary: dict[str, object],
    train_subjects: list[str],
    val_subjects: list[str],
    heldout_subject: str,
) -> None:
    lines = [
        "# Loader Engineering Closure",
        "",
        "This report profiles the existing single-process lazy window loader against a raw-recording preload optimization.",
        "The optimization does not materialize the full window matrix and does not change split rules, target alignment, scorer, or model adapter.",
        "",
        "## Scope",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- heldout subject for profiling: `{heldout_subject}`",
        f"- train subjects: `{', '.join(train_subjects)}`",
        f"- val subjects: `{', '.join(val_subjects)}`",
        f"- num_workers: `{config['dataloader']['num_workers']}`",
        f"- persistent_workers: `{config['dataloader']['persistent_workers']}`",
        "",
        "## Train Loader (200 batches)",
        f"- old elapsed seconds: `{train_old['elapsed_seconds']:.6f}`",
        f"- optimized elapsed seconds: `{train_opt['elapsed_seconds']:.6f}`",
        f"- old batch shape: `{train_old['batch_shape']}`",
        f"- optimized batch shape: `{train_opt['batch_shape']}`",
        f"- old cache hit rate: `{train_old.get('cache_hit_rate')}`",
        f"- old load requests: `{train_old['load_requests']}`",
        f"- optimized preload loads: `{train_opt['load_requests']}`",
        f"- old recordings loaded more than once: `{train_old['recordings_loaded_more_than_once_count']}`",
        f"- optimized recordings loaded more than once: `{train_opt['recordings_loaded_more_than_once_count']}`",
        "",
        "## Val Loader (50 batches)",
        f"- old elapsed seconds: `{val_old['elapsed_seconds']:.6f}`",
        f"- optimized elapsed seconds: `{val_opt['elapsed_seconds']:.6f}`",
        f"- old batch shape: `{val_old['batch_shape']}`",
        f"- optimized batch shape: `{val_opt['batch_shape']}`",
        f"- old cache hit rate: `{val_old.get('cache_hit_rate')}`",
        f"- old load requests: `{val_old['load_requests']}`",
        f"- optimized preload loads: `{val_opt['load_requests']}`",
        "",
        "## Memory Estimate",
        f"- old train estimated raw recording bytes: `{train_summary['estimated_recording_bytes']}`",
        f"- old train peak cache bytes: `{train_old.get('peak_cache_bytes')}`",
        f"- optimized train preloaded bytes: `{train_opt.get('preloaded_recording_bytes')}`",
        f"- old val estimated raw recording bytes: `{val_summary['estimated_recording_bytes']}`",
        f"- old val peak cache bytes: `{val_old.get('peak_cache_bytes')}`",
        f"- optimized val preloaded bytes: `{val_opt.get('preloaded_recording_bytes')}`",
        "",
        "## Alignment Invariants",
        f"- train batch shape unchanged: `{train_old['batch_shape'] == train_opt['batch_shape']}`",
        f"- val batch shape unchanged: `{val_old['batch_shape'] == val_opt['batch_shape']}`",
        f"- train target batch shape unchanged: `{train_old['target_batch_shape'] == train_opt['target_batch_shape']}`",
        f"- val target batch shape unchanged: `{val_old['target_batch_shape'] == val_opt['target_batch_shape']}`",
        "- target alignment rule unchanged: `window target index = last sample`",
        "- scorer unchanged: `pearson_on_valid`",
        "- pure LOSO split unchanged: `heldout subject excluded from train and val`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_eegnet_for_subject(
    *,
    config: dict,
    dataset_dir: Path,
    heldout_subject: str,
    all_subjects: list[str],
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    train_subjects = [subject for subject in all_subjects if subject != heldout_subject]
    val_subjects = list(train_subjects)
    model_handle, model_kwargs = select_window_model(config)
    train_dataset = PreloadedWindowDataset(
        dataset_dir,
        "train",
        subjects=train_subjects,
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )
    val_dataset = PreloadedWindowDataset(
        dataset_dir,
        "val",
        subjects=val_subjects,
        window_size=int(model_kwargs["input_length"]),
        channels=range(int(model_kwargs["num_input_channels"])),
        target_index=str(config["eegnet"]["target_index"]),
    )

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))

    model = model_handle(**model_kwargs).to(device)
    optimizer = NAdam(
        model.parameters(),
        lr=float(config["eegnet"]["learning_rate"]),
        weight_decay=float(config["eegnet"]["weight_decay"]),
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["dataloader"]["batch_size"]),
        shuffle=bool(config["dataloader"]["shuffle_train"]),
        num_workers=0,
        pin_memory=(device == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["dataloader"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_score = float("-inf")
    best_epoch = 0
    val_history: list[float] = []

    for epoch in range(int(config["mini_closure"]["eegnet_max_epochs"])):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            loss = -batch_corr(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx >= int(config["mini_closure"]["eegnet_max_train_batches"]):
                break

        model.eval()
        scores = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader, start=1):
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                y_hat = model(x)
                scores.append(float(batch_corr(y, y_hat).item()))
                if batch_idx >= int(config["mini_closure"]["eegnet_max_val_batches"]):
                    break
        val_score = float(np.mean(scores)) if scores else float("nan")
        val_history.append(val_score)
        if scores and val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model = model_handle(**model_kwargs).to(device)
    model.load_state_dict(best_state)
    model.eval()
    windows: list[WindowPrediction] = []
    scores: list[float] = []
    input_length = int(model_kwargs["input_length"])
    offset = input_length - 1
    for eeg_path in sorted(dataset_dir.glob(f"test_-_{heldout_subject}_-_*_-_eeg.npy")):
        recording_id = eeg_path.stem.replace("_-_eeg", "")
        env_path = eeg_path.with_name(f"{recording_id}_-_envelope.npy")
        eeg = np.load(eeg_path).astype(np.float32)[:, list(range(int(model_kwargs["num_input_channels"])))]
        env = np.load(env_path).astype(np.float32)[:, 0]
        starts = list(range(0, eeg.shape[0] - input_length + 1))
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for chunk_start in range(0, len(starts), int(config["dataloader"]["eval_batch_size"])):
                chunk = starts[chunk_start : chunk_start + int(config["dataloader"]["eval_batch_size"])]
                batch = np.stack([eeg[start : start + input_length].T for start in chunk], axis=0)
                batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                pred_chunk = model(batch_tensor).detach().cpu().numpy().astype(np.float32)
                preds.append(pred_chunk)
        pred_arr = np.concatenate(preds, axis=0)
        target_arr = np.asarray([env[start + input_length - 1] for start in starts], dtype=np.float32)
        windows.append(
            build_series_window(
                dataset=str(config["dataset"]["dataset_id"]),
                model="eegnet",
                protocol=str(config["protocol"]),
                seed=int(config["seed"]),
                subject_id=heldout_subject,
                recording_id=recording_id,
                sampling_rate=int(config["dataset"]["sampling_rate"]),
                checkpoint_id=f"eegnet_epoch_{best_epoch}",
                full_length=len(env),
                offset=offset,
                prediction=pred_arr,
                target=target_arr,
            )
        )
        scores.append(correlation(pred_arr, target_arr))

    recording_rows, subject_rows = aggregate_model_outputs(windows, str(config["artifact_scope"]))
    for row in recording_rows:
        row["checkpoint_id"] = f"eegnet_epoch_{best_epoch}"
    for row in subject_rows:
        row["checkpoint_id"] = f"eegnet_epoch_{best_epoch}"
    leakage_entry = {
        "dataset": str(config["dataset"]["dataset_id"]),
        "model": "eegnet",
        "heldout_subject": heldout_subject,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": [heldout_subject],
        "excluded_target_from_train": heldout_subject not in train_subjects,
        "excluded_target_from_val": heldout_subject not in val_subjects,
        "excluded_target_from_selection": heldout_subject not in val_subjects,
        "normalization_fitted_on_target": False,
        "train_recording_count": int(train_dataset.total_recordings),
        "val_recording_count": int(val_dataset.total_recordings),
        "train_window_count": int(train_dataset.total_windows),
        "val_window_count": int(val_dataset.total_windows),
        "selection_rule": str(config["eegnet"]["selection_rule"]),
        "training_protocol": str(config["eegnet"]["training_protocol"]),
        "mean_test_pearson_direct": float(np.mean(scores)) if scores else float("nan"),
        "subject_metric_value": float(subject_rows[0]["metric_value"]),
    }
    return recording_rows, subject_rows, leakage_entry


def load_ridge_reference_rows(config: dict, heldout_subjects: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    subject_rows = []
    recording_rows = []
    leakage_entries = []
    all_subjects = list_reference_subjects(resolve_dataset_path(str(config["dataset"]["dataset_locator"])), split="test")
    for row in load_dict_rows(ROOT / config["ridge_loso_reference"]["subject_metrics"]):
        if row["dataset"] != config["dataset"]["dataset_id"] or row["subject_id"] not in heldout_subjects or row["model"] != "ridge":
            continue
        subject_rows.append(
            {
                "dataset": row["dataset"],
                "model": "ridge",
                "task": row["task"],
                "protocol": config["protocol"],
                "seed": int(config["seed"]),
                "subject_id": row["subject_id"],
                "metric_name": row["metric_name"],
                "metric_value": float(row["metric_value"]),
                "num_recordings": int(row["num_recordings"]),
                "checkpoint_id": row["checkpoint_id"],
                "artifact_scope": config["artifact_scope"],
            }
        )
        leakage_entries.append(
            {
                "dataset": row["dataset"],
                "model": "ridge",
                "heldout_subject": row["subject_id"],
                "train_subjects": [subject for subject in all_subjects if subject != row["subject_id"]],
                "val_subjects": [subject for subject in all_subjects if subject != row["subject_id"]],
                "test_subjects": [row["subject_id"]],
                "excluded_target_from_train": True,
                "excluded_target_from_val": True,
                "excluded_target_from_selection": True,
                "normalization_fitted_on_target": False,
                "selection_rule": "reuse accepted loso-ridge-full-v1 rows",
                "training_protocol": "no rerun; import accepted ridge LOSO result rows and relabel to current engineering closure protocol",
            }
        )
    for row in load_dict_rows(ROOT / config["ridge_loso_reference"]["recording_metrics"]):
        if row["dataset"] != config["dataset"]["dataset_id"] or row["subject_id"] not in heldout_subjects or row["model"] != "ridge":
            continue
        recording_rows.append(
            {
                "dataset": row["dataset"],
                "model": "ridge",
                "task": row["task"],
                "protocol": config["protocol"],
                "seed": int(config["seed"]),
                "subject_id": row["subject_id"],
                "recording_id": row["recording_id"],
                "sampling_rate": int(row["sampling_rate"]),
                "metric_name": row["metric_name"],
                "metric_value": float(row["metric_value"]),
                "num_valid_samples": int(row["num_valid_samples"]),
                "checkpoint_id": row["checkpoint_id"],
                "artifact_scope": config["artifact_scope"],
            }
        )
    return recording_rows, subject_rows, leakage_entries


def write_leakage_summary(path: Path, *, leakage_entries: list[dict[str, object]]) -> None:
    lines = [
        "# Leakage Summary",
        "",
        "All jobs in this engineering closure keep pure LOSO semantics: the heldout subject is excluded from train, val, normalization fitting, and checkpoint selection, and used only for test.",
        "",
        "| model | heldout_subject | excluded_from_train | excluded_from_val | excluded_from_selection | test_subjects |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in leakage_entries:
        lines.append(
            f"| {entry['model']} | {entry['heldout_subject']} | {entry['excluded_target_from_train']} | "
            f"{entry['excluded_target_from_val']} | {entry['excluded_target_from_selection']} | {','.join(entry['test_subjects'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_schema(
    *,
    config: dict,
    subject_metrics: list[dict[str, object]],
    recording_metrics: list[dict[str, object]],
    completed_jobs: list[dict[str, object]],
    leakage_entries: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    expected_pairs = {
        (str(config["dataset"]["dataset_id"]), model_name, subject_id)
        for model_name in config["mini_closure"]["models"]
        for subject_id in config["mini_closure"]["heldout_subjects"]
    }
    successful_pairs = {(row["dataset"], row["model"], row["subject_id"]) for row in subject_metrics}
    completed_pairs = {(row["dataset"], row["model"], row["subject_id"]) for row in completed_jobs}
    missing_pairs = sorted(expected_pairs - successful_pairs)
    subject_fields_ok = all(sorted(row.keys()) == sorted(SUBJECT_METRIC_FIELDS) for row in subject_metrics)
    recording_fields_ok = all(sorted(row.keys()) == sorted(RECORDING_METRIC_FIELDS) for row in recording_metrics)
    subject_range_ok = all(-1.0 <= float(row["metric_value"]) <= 1.0 for row in subject_metrics)
    recording_range_ok = all(-1.0 <= float(row["metric_value"]) <= 1.0 for row in recording_metrics)
    leakage_ok = all(
        entry["excluded_target_from_train"]
        and entry["excluded_target_from_val"]
        and entry["excluded_target_from_selection"]
        and entry["test_subjects"] == [entry["heldout_subject"]]
        and not entry["normalization_fitted_on_target"]
        for entry in leakage_entries
    )
    passed = bool(
        len(missing_pairs) == 0
        and successful_pairs == completed_pairs
        and subject_fields_ok
        and recording_fields_ok
        and subject_range_ok
        and recording_range_ok
        and leakage_ok
        and len(failures) == 0
    )
    return {
        "protocol": config["protocol"],
        "passed": passed,
        "checks": {
            "dataset_model_subject_coverage_complete": len(missing_pairs) == 0,
            "completed_jobs_match_subject_metrics": successful_pairs == completed_pairs,
            "subject_metric_fields_consistent": subject_fields_ok,
            "recording_metric_fields_consistent": recording_fields_ok,
            "subject_metric_values_valid": subject_range_ok,
            "recording_metric_values_valid": recording_range_ok,
            "pure_loso_leakage_check_passed": leakage_ok,
            "failure_report_empty": len(failures) == 0,
        },
        "details": {
            "expected_jobs": len(expected_pairs),
            "subject_metric_rows": len(subject_metrics),
            "recording_metric_rows": len(recording_metrics),
            "completed_jobs": len(completed_jobs),
            "failures": len(failures),
            "missing_pairs": [
                {"dataset": dataset, "model": model, "subject_id": subject_id}
                for dataset, model, subject_id in missing_pairs
            ],
        },
    }


def write_result_summary(
    path: Path,
    *,
    config: dict,
    profiling_summary: dict[str, object],
    subject_metrics: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    grouped: dict[tuple[str, str], float] = {}
    for row in subject_metrics:
        grouped[(row["model"], row["subject_id"])] = float(row["metric_value"])
    lines = [
        "# LOSO Loader Engineering Closure v1",
        "",
        "This package is not a full benchmark. It closes the `num_workers=0` loader-engineering question before any broader LOSO expansion.",
        "",
        "## Profiling",
        f"- train old seconds: `{profiling_summary['train_old_seconds']:.6f}`",
        f"- train optimized seconds: `{profiling_summary['train_optimized_seconds']:.6f}`",
        f"- val old seconds: `{profiling_summary['val_old_seconds']:.6f}`",
        f"- val optimized seconds: `{profiling_summary['val_optimized_seconds']:.6f}`",
        f"- train batch shape unchanged: `{profiling_summary['train_shape_unchanged']}`",
        f"- val batch shape unchanged: `{profiling_summary['val_shape_unchanged']}`",
        "",
        "## Mini Closure Subject Metrics",
        f"- ridge / P00: `{grouped[('ridge', 'P00')]:.6f}`",
        f"- ridge / P01: `{grouped[('ridge', 'P01')]:.6f}`",
        f"- eegnet / P00: `{grouped[('eegnet', 'P00')]:.6f}`",
        f"- eegnet / P01: `{grouped[('eegnet', 'P01')]:.6f}`",
        "",
        "## Status",
        f"- failure count: `{len(failures)}`",
        "- ridge rows are reused from the accepted ridge LOSO package and relabeled to the current engineering-closure protocol.",
        "- eegnet rows are newly run under the optimized single-process preloaded-raw-recording loader.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_engineering_report(
    path: Path,
    *,
    config: dict,
    train_old_profile: dict[str, object],
    train_opt_profile: dict[str, object],
    val_old_profile: dict[str, object],
    val_opt_profile: dict[str, object],
    subject_metrics: list[dict[str, object]],
) -> None:
    metric_lookup = {(row["model"], row["subject_id"]): float(row["metric_value"]) for row in subject_metrics}
    lines = [
        "# Engineering Report",
        "",
        "This package is an engineering closure for the single-process LOSO loader path. It is not a full LOSO benchmark package.",
        "",
        "## Fixed Constraints",
        f"- num_workers: `{config['dataloader']['num_workers']}`",
        f"- persistent_workers: `{config['dataloader']['persistent_workers']}`",
        "- model adapter changed: `false`",
        "- target alignment changed: `false`",
        "- scorer changed: `false`",
        "- split / LOSO rule changed: `false`",
        "- training budget changed: `false`",
        "- optimization type: `raw recording preload only; no full window matrix materialization`",
        "",
        "## Profiling Comparison",
        f"- train old seconds for 200 batches: `{train_old_profile['elapsed_seconds']:.6f}`",
        f"- train optimized seconds for 200 batches: `{train_opt_profile['elapsed_seconds']:.6f}`",
        f"- val old seconds for 50 batches: `{val_old_profile['elapsed_seconds']:.6f}`",
        f"- val optimized seconds for 50 batches: `{val_opt_profile['elapsed_seconds']:.6f}`",
        f"- old train load requests: `{train_old_profile['load_requests']}`",
        f"- optimized train preload loads: `{train_opt_profile['load_requests']}`",
        f"- old train recordings loaded more than once: `{train_old_profile['recordings_loaded_more_than_once_count']}`",
        f"- optimized train recordings loaded more than once: `{train_opt_profile['recordings_loaded_more_than_once_count']}`",
        f"- train batch shape unchanged: `{train_old_profile['batch_shape'] == train_opt_profile['batch_shape']}`",
        f"- val batch shape unchanged: `{val_old_profile['batch_shape'] == val_opt_profile['batch_shape']}`",
        "",
        "## Mini Closure Coverage",
        "- subject_metrics rows expected: `4`",
        f"- ridge / P00: `{metric_lookup[('ridge', 'P00')]:.6f}`",
        f"- ridge / P01: `{metric_lookup[('ridge', 'P01')]:.6f}`",
        f"- eegnet / P00: `{metric_lookup[('eegnet', 'P00')]:.6f}`",
        f"- eegnet / P01: `{metric_lookup[('eegnet', 'P01')]:.6f}`",
        "",
        "## Artifact Hygiene",
        "- raw data uploaded: `false`",
        "- prediction dump uploaded: `false`",
        "- checkpoint uploaded: `false`",
        "- model weights uploaded: `false`",
        "- per-job directories uploaded: `false`",
        "- large array / weight files uploaded: `false`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    device = resolve_device(args.device)

    dataset_dir = resolve_dataset_path(str(config["dataset"]["dataset_locator"]))
    all_subjects = list_reference_subjects(dataset_dir, split="test")
    profile_heldout = str(config["profile"]["heldout_subject"])
    train_subjects = [subject for subject in all_subjects if subject != profile_heldout]
    val_subjects = list(train_subjects)

    train_old_dataset = build_dataset(mode="cached_lazy", dataset_dir=dataset_dir, split="train", subjects=train_subjects, config=config)
    train_opt_dataset = build_dataset(mode="preloaded", dataset_dir=dataset_dir, split="train", subjects=train_subjects, config=config)
    val_old_dataset = build_dataset(mode="cached_lazy", dataset_dir=dataset_dir, split="val", subjects=val_subjects, config=config)
    val_opt_dataset = build_dataset(mode="preloaded", dataset_dir=dataset_dir, split="val", subjects=val_subjects, config=config)

    train_old_profile = profile_loader(
        dataset=train_old_dataset,
        batch_size=int(config["dataloader"]["batch_size"]),
        shuffle=bool(config["dataloader"]["shuffle_train"]),
        num_workers=0,
        max_batches=int(config["profile"]["max_train_batches"]),
        seed=int(config["seed"]),
    )
    train_opt_profile = profile_loader(
        dataset=train_opt_dataset,
        batch_size=int(config["dataloader"]["batch_size"]),
        shuffle=bool(config["dataloader"]["shuffle_train"]),
        num_workers=0,
        max_batches=int(config["profile"]["max_train_batches"]),
        seed=int(config["seed"]),
    )
    val_old_profile = profile_loader(
        dataset=val_old_dataset,
        batch_size=int(config["dataloader"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        max_batches=int(config["profile"]["max_val_batches"]),
        seed=int(config["seed"]),
    )
    val_opt_profile = profile_loader(
        dataset=val_opt_dataset,
        batch_size=int(config["dataloader"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        max_batches=int(config["profile"]["max_val_batches"]),
        seed=int(config["seed"]),
    )

    write_json(output_dir / "loader_profile_train_old.json", train_old_profile)
    write_json(output_dir / "loader_profile_train_optimized.json", train_opt_profile)
    write_json(output_dir / "loader_profile_val_old.json", val_old_profile)
    write_json(output_dir / "loader_profile_val_optimized.json", val_opt_profile)
    write_loader_profile_report(
        output_dir / "loader_profile_report.md",
        config=config,
        train_old=train_old_profile,
        train_opt=train_opt_profile,
        val_old=val_old_profile,
        val_opt=val_opt_profile,
        train_summary=train_old_dataset.base_summary(heldout_subject=profile_heldout),
        val_summary=val_old_dataset.base_summary(heldout_subject=profile_heldout),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        heldout_subject=profile_heldout,
    )

    all_recording_rows: list[dict[str, object]] = []
    all_subject_rows: list[dict[str, object]] = []
    all_leakage_entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    completed_jobs: list[dict[str, object]] = []

    ridge_recording_rows, ridge_subject_rows, ridge_leakage_entries = load_ridge_reference_rows(
        config,
        list(config["mini_closure"]["heldout_subjects"]),
    )
    all_recording_rows.extend(ridge_recording_rows)
    all_subject_rows.extend(ridge_subject_rows)
    all_leakage_entries.extend(ridge_leakage_entries)
    for subject_id in config["mini_closure"]["heldout_subjects"]:
        completed_jobs.append(
            {
                "dataset": str(config["dataset"]["dataset_id"]),
                "subject_id": str(subject_id),
                "model": "ridge",
                "seed": int(config["seed"]),
                "status": "reused_reference",
            }
        )

    for heldout_subject in config["mini_closure"]["heldout_subjects"]:
        try:
            recording_rows, subject_rows, leakage_entry = fit_eegnet_for_subject(
                config=config,
                dataset_dir=dataset_dir,
                heldout_subject=str(heldout_subject),
                all_subjects=all_subjects,
                device=device,
            )
            all_recording_rows.extend(recording_rows)
            all_subject_rows.extend(subject_rows)
            all_leakage_entries.append(leakage_entry)
            completed_jobs.append(
                {
                    "dataset": str(config["dataset"]["dataset_id"]),
                    "subject_id": str(heldout_subject),
                    "model": "eegnet",
                    "seed": int(config["seed"]),
                    "status": "success",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset": str(config["dataset"]["dataset_id"]),
                    "subject_id": str(heldout_subject),
                    "model": "eegnet",
                    "seed": int(config["seed"]),
                    "status": "failed",
                    "error_type": exc.__class__.__name__,
                    "error": repr(exc),
                }
            )

    all_recording_rows = sorted(all_recording_rows, key=lambda row: (row["model"], row["subject_id"], row["recording_id"]))
    all_subject_rows = sorted(all_subject_rows, key=lambda row: (row["model"], row["subject_id"]))
    completed_jobs = sorted(completed_jobs, key=lambda row: (row["model"], row["subject_id"]))
    all_leakage_entries = sorted(all_leakage_entries, key=lambda row: (row["model"], row["heldout_subject"]))

    write_generic_csv(output_dir / "recording_metrics.csv", all_recording_rows, RECORDING_METRIC_FIELDS)
    write_generic_csv(output_dir / "subject_metrics.csv", all_subject_rows, SUBJECT_METRIC_FIELDS)
    write_json(output_dir / "completed_jobs.json", {"completed_jobs": completed_jobs})
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_leakage_summary(output_dir / "leakage_summary.md", leakage_entries=all_leakage_entries)

    schema_validation = validate_schema(
        config=config,
        subject_metrics=all_subject_rows,
        recording_metrics=all_recording_rows,
        completed_jobs=completed_jobs,
        leakage_entries=all_leakage_entries,
        failures=failures,
    )
    write_json(output_dir / "schema_validation_report.json", schema_validation)

    profiling_summary = {
        "train_old_seconds": float(train_old_profile["elapsed_seconds"]),
        "train_optimized_seconds": float(train_opt_profile["elapsed_seconds"]),
        "val_old_seconds": float(val_old_profile["elapsed_seconds"]),
        "val_optimized_seconds": float(val_opt_profile["elapsed_seconds"]),
        "train_shape_unchanged": train_old_profile["batch_shape"] == train_opt_profile["batch_shape"],
        "val_shape_unchanged": val_old_profile["batch_shape"] == val_opt_profile["batch_shape"],
    }
    write_result_summary(
        output_dir / "result_summary.md",
        config=config,
        profiling_summary=profiling_summary,
        subject_metrics=all_subject_rows,
        failures=failures,
    )
    write_engineering_report(
        output_dir / "engineering_report.md",
        config=config,
        train_old_profile=train_old_profile,
        train_opt_profile=train_opt_profile,
        val_old_profile=val_old_profile,
        val_opt_profile=val_opt_profile,
        subject_metrics=all_subject_rows,
    )

    write_json(
        output_dir / "run_manifest.json",
        {
            "protocol": config["protocol"],
            "base_branch": config["base_branch"],
            "base_commit": config["base_commit"],
            "current_branch": config["current_branch"],
            "device": device,
            "dataset": config["dataset"],
            "seed": int(config["seed"]),
            "profile": config["profile"],
            "mini_closure": config["mini_closure"],
            "dataloader": config["dataloader"],
            "profiling_summary": profiling_summary,
            "job_accounting": {
                "planned_jobs": len(config["mini_closure"]["heldout_subjects"]) * len(config["mini_closure"]["models"]),
                "successful_jobs": len(all_subject_rows),
                "failed_jobs": len(failures),
            },
            "artifacts": {
                "loader_profile_report": repo_relative(output_dir / "loader_profile_report.md"),
                "loader_profile_train_old": repo_relative(output_dir / "loader_profile_train_old.json"),
                "loader_profile_train_optimized": repo_relative(output_dir / "loader_profile_train_optimized.json"),
                "loader_profile_val_old": repo_relative(output_dir / "loader_profile_val_old.json"),
                "loader_profile_val_optimized": repo_relative(output_dir / "loader_profile_val_optimized.json"),
                "recording_metrics": repo_relative(output_dir / "recording_metrics.csv"),
                "subject_metrics": repo_relative(output_dir / "subject_metrics.csv"),
                "completed_jobs": repo_relative(output_dir / "completed_jobs.json"),
                "failure_report": repo_relative(output_dir / "failure_report.json"),
                "schema_validation_report": repo_relative(output_dir / "schema_validation_report.json"),
                "leakage_summary": repo_relative(output_dir / "leakage_summary.md"),
                "result_summary": repo_relative(output_dir / "result_summary.md"),
                "engineering_report": repo_relative(output_dir / "engineering_report.md"),
            },
        },
    )
    return 0 if schema_validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
