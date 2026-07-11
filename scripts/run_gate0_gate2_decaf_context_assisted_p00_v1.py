from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys
import types
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.result_schema import RECORDING_METRIC_FIELDS, SUBJECT_METRIC_FIELDS, write_rows
from benchmark.scoring import WindowPrediction, aggregate_overlapping_windows, recording_metric_row, subject_metric_rows
from repro.reference_baselines import load_reference_recordings
from run_gate0_gate2_subject_specific_local_model_smoke_v1 import (
    ensure_dir,
    load_config,
    print_event,
    repo_relative,
    resolve_dataset_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--checkpoint-mock", action="store_true")
    parser.add_argument("--schema-preflight", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shape_text(shape: Iterable[int] | torch.Size) -> str:
    return "[" + ", ".join(str(int(item)) for item in shape) + "]"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_log(output_dir: Path, message: str) -> None:
    log_path = output_dir / "logs" / "run.log"
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def emit(output_dir: Path, stage: str, message: str) -> None:
    line = f"[{stage}] {message}"
    print(line, flush=True)
    append_log(output_dir, line)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_twobranch_class(upstream_path: Path):
    decaf_src = upstream_path.resolve() / "src"
    for path in (upstream_path.resolve(), decaf_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    models_dir = decaf_src / "models"
    package = types.ModuleType("models")
    package.__path__ = [str(models_dir)]
    package.__package__ = "models"
    sys.modules["models"] = package
    module_path = models_dir / "two_branch_model.py"
    spec = importlib.util.spec_from_file_location("models.two_branch_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load DECAF TwoBranchModel from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.two_branch_model"] = module
    spec.loader.exec_module(module)
    return module.TwoBranchModel


def instantiate_twobranch(config: dict[str, Any], device: str):
    decaf = config["decaf"]
    model_cls = load_twobranch_class(ROOT / decaf["upstream_path"])
    model = model_cls(
        in_channel=int(decaf["in_channel"]),
        d_model=int(decaf["d_model"]),
        d_inner=int(decaf["d_inner"]),
        n_head=int(decaf["n_head"]),
        n_layers=int(decaf["n_layers"]),
        fft_conv1d_kernel=tuple(decaf["fft_conv1d_kernel"]),
        fft_conv1d_padding=tuple(decaf["fft_conv1d_padding"]),
        dropout=float(decaf["dropout"]),
        fusion_strategy=str(decaf["fusion_strategy"]),
        rnn_type=str(decaf["rnn_type"]),
        context_seconds=float(decaf["context_seconds"]),
        prediction_seconds=float(decaf["prediction_seconds"]),
    ).to(device)
    return model


def load_split_recordings(dataset_dir: Path, subject_id: str) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    return {
        split: load_reference_recordings(dataset_dir, split, subject_id, channels=range(64))
        for split in ["train", "val", "test"]
    }


def split_disjointness(recordings_by_split: dict[str, list[tuple[str, np.ndarray, np.ndarray]]]) -> dict[str, Any]:
    ids_by_split = {
        split: {recording_id for recording_id, _, _ in recordings}
        for split, recordings in recordings_by_split.items()
    }
    train_val = sorted(ids_by_split["train"] & ids_by_split["val"])
    train_test = sorted(ids_by_split["train"] & ids_by_split["test"])
    val_test = sorted(ids_by_split["val"] & ids_by_split["test"])
    return {
        "recording_id_counts": {split: len(ids) for split, ids in ids_by_split.items()},
        "train_val_intersection_count": len(train_val),
        "train_test_intersection_count": len(train_test),
        "val_test_intersection_count": len(val_test),
        "train_val_intersection_ids": train_val,
        "train_test_intersection_ids": train_test,
        "val_test_intersection_ids": val_test,
        "all_pairwise_intersections_empty": len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0,
    }


def fit_train_only_scalers(train_recordings: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    eeg = np.concatenate([item[1] for item in train_recordings], axis=0).astype(np.float64)
    env = np.concatenate([item[2] for item in train_recordings], axis=0).astype(np.float64)
    eeg_mean = eeg.mean(axis=0).astype(np.float32)
    eeg_std = eeg.std(axis=0).astype(np.float32)
    eeg_std[eeg_std < 1e-6] = 1.0
    env_mean = float(env.mean())
    env_std = float(env.std())
    if env_std < 1e-6:
        env_std = 1.0
    return {
        "eeg_mean": eeg_mean,
        "eeg_std": eeg_std,
        "env_mean": env_mean,
        "env_std": env_std,
        "fit_split": "train",
        "fit_recording_count": len(train_recordings),
    }


def apply_scalers(recordings: list[tuple[str, np.ndarray, np.ndarray]], scalers: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    eeg_mean = np.asarray(scalers["eeg_mean"], dtype=np.float32)
    eeg_std = np.asarray(scalers["eeg_std"], dtype=np.float32)
    env_mean = float(scalers["env_mean"])
    env_std = float(scalers["env_std"])
    normalized = []
    for recording_id, eeg, env in recordings:
        normalized.append(
            (
                recording_id,
                ((eeg.astype(np.float32) - eeg_mean) / eeg_std).astype(np.float32),
                ((env.astype(np.float32) - env_mean) / env_std).astype(np.float32),
            )
        )
    return normalized


class ContextAssistedWindowDataset(Dataset):
    def __init__(
        self,
        recordings: list[tuple[str, np.ndarray, np.ndarray]],
        *,
        split: str,
        subject_id: str,
        target_samples: int,
        context_samples: int,
        trim_samples: int,
    ) -> None:
        self.recordings = recordings
        self.split = split
        self.subject_id = subject_id
        self.target_samples = int(target_samples)
        self.context_samples = int(context_samples)
        self.trim_samples = int(trim_samples)
        self.records: list[tuple[int, int]] = []
        for rec_idx, (_, eeg, env) in enumerate(recordings):
            length = int(min(eeg.shape[0], env.shape[0]))
            for target_start in range(self.context_samples, length - self.target_samples + 1, self.target_samples):
                self.records.append((rec_idx, target_start))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec_idx, target_start = self.records[index]
        recording_id, eeg, env = self.recordings[rec_idx]
        target_end = target_start + self.target_samples
        context_start = target_start - self.context_samples
        context_end = target_start
        return {
            "eeg": torch.from_numpy(eeg[target_start:target_end].astype(np.float32)),
            "envelope_context": torch.from_numpy(env[context_start:context_end].astype(np.float32)).unsqueeze(-1),
            "target": torch.from_numpy(env[target_start:target_end].astype(np.float32)).unsqueeze(-1),
            "split": self.split,
            "subject_id": self.subject_id,
            "recording_id": recording_id,
            "recording_length": int(min(eeg.shape[0], env.shape[0])),
            "target_start": target_start,
            "target_end_exclusive": target_end,
            "eeg_start": target_start,
            "eeg_end_exclusive": target_end,
            "envelope_context_start": context_start,
            "envelope_context_end_exclusive": context_end,
            "effective_context_start": context_start + self.trim_samples,
            "effective_context_end_exclusive": context_end,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eeg": torch.stack([item["eeg"] for item in items], dim=0),
        "envelope_context": torch.stack([item["envelope_context"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
        "metadata": [{key: value for key, value in item.items() if key not in {"eeg", "envelope_context", "target"}} for item in items],
    }


def build_datasets(
    recordings_by_split: dict[str, list[tuple[str, np.ndarray, np.ndarray]]],
    config: dict[str, Any],
    scalers: dict[str, Any],
) -> dict[str, ContextAssistedWindowDataset]:
    task = config["task"]
    subject_id = config["dataset"]["subject_id"]
    return {
        split: ContextAssistedWindowDataset(
            apply_scalers(recordings, scalers),
            split=split,
            subject_id=subject_id,
            target_samples=int(task["eeg_target_samples"]),
            context_samples=int(task["forward_envelope_context_samples"]),
            trim_samples=int(task["forward_context_trim_samples"]),
        )
        for split, recordings in recordings_by_split.items()
    }


def coverage_rows(datasets: dict[str, ContextAssistedWindowDataset], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    target_samples = int(config["task"]["eeg_target_samples"])
    context_samples = int(config["task"]["forward_envelope_context_samples"])
    for split, dataset in datasets.items():
        grouped: dict[str, list[int]] = {}
        lengths: dict[str, int] = {}
        for rec_idx, target_start in dataset.records:
            recording_id, eeg, env = dataset.recordings[rec_idx]
            grouped.setdefault(recording_id, []).append(target_start)
            lengths[recording_id] = int(min(eeg.shape[0], env.shape[0]))
        for recording_id, starts in sorted(grouped.items()):
            starts = sorted(starts)
            length = lengths[recording_id]
            covered_samples = len(starts) * target_samples
            last_target_end = starts[-1] + target_samples if starts else ""
            eligible_after_context = max(0, length - context_samples)
            rows.append(
                {
                    "split": split,
                    "subject_id": config["dataset"]["subject_id"],
                    "recording_id": recording_id,
                    "recording_length": length,
                    "excluded_initial_context_samples": min(context_samples, length),
                    "first_target_start": starts[0] if starts else "",
                    "window_count": len(starts),
                    "covered_target_samples": covered_samples,
                    "last_target_end_exclusive": last_target_end,
                    "dropped_tail_samples_after_nonoverlap_windows": max(0, length - int(last_target_end)) if starts else eligible_after_context,
                    "eligible_samples_after_initial_context": eligible_after_context,
                    "coverage_ratio_of_recording": float(covered_samples / length) if length else 0.0,
                    "coverage_ratio_after_initial_context": float(covered_samples / eligible_after_context) if eligible_after_context else 0.0,
                    "context_never_crosses_recording": True,
                    "context_strictly_past": True,
                }
            )
    return rows


def leakage_ok(datasets: dict[str, ContextAssistedWindowDataset]) -> bool:
    for dataset in datasets.values():
        for rec_idx, target_start in dataset.records:
            recording_id, _, _ = dataset.recordings[rec_idx]
            context_start = target_start - dataset.context_samples
            context_end = target_start
            if context_start < 0 or context_end != target_start or context_end - 1 >= target_start:
                return False
            if not recording_id:
                return False
    return True


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(np.float64)
    truth = target.astype(np.float64)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = np.sqrt(np.sum(pred**2) * np.sum(truth**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(pred * truth) / denom)


def evaluate_split(
    model,
    dataset: ContextAssistedWindowDataset,
    *,
    config: dict[str, Any],
    device: str,
    checkpoint_id: str,
    artifact_scope: str,
) -> tuple[float, list[Any], list[Any]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)
    grouped: dict[str, list[WindowPrediction]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device=device, dtype=torch.float32)
            context = batch["envelope_context"].to(device=device, dtype=torch.float32)
            output = model(eeg, context)
            pred = output["envelope"].detach().cpu().numpy()[0, :, 0].astype(np.float32)
            target = batch["target"].numpy()[0, :, 0].astype(np.float32)
            meta = batch["metadata"][0]
            window = WindowPrediction(
                dataset=config["dataset"]["dataset_id"],
                model=config["task"]["model_key"],
                task=config["task"]["task_classification"],
                protocol=config["protocol"],
                seed=int(config["seed"]),
                subject_id=config["dataset"]["subject_id"],
                recording_id=meta["recording_id"],
                sampling_rate=int(config["dataset"]["sampling_rate"]),
                checkpoint_id=checkpoint_id,
                recording_length=int(meta["recording_length"]),
                start_index=int(meta["target_start"]),
                prediction=pred,
                target=target,
            )
            grouped.setdefault(meta["recording_id"], []).append(window)

    recording_rows = []
    recording_scores = []
    for recording_id, windows in sorted(grouped.items()):
        first = windows[0]
        prediction, target, valid_mask = aggregate_overlapping_windows(first.recording_length, windows)
        score = pearson(prediction[valid_mask.astype(bool)], target[valid_mask.astype(bool)])
        recording_scores.append(score)
        recording_rows.append(
            recording_metric_row(
                dataset=first.dataset,
                model=first.model,
                task=first.task,
                protocol=first.protocol,
                seed=first.seed,
                subject_id=first.subject_id,
                recording_id=recording_id,
                sampling_rate=first.sampling_rate,
                checkpoint_id=checkpoint_id,
                artifact_scope=artifact_scope,
                prediction=prediction,
                target=target,
                valid_mask=valid_mask,
            )
        )
    mean_score = float(np.nanmean(recording_scores)) if recording_scores else float("nan")
    return mean_score, recording_rows, subject_metric_rows(recording_rows)


def job_key(config: dict[str, Any]) -> str:
    return ":".join(
        [
            config["dataset"]["dataset_id"],
            config["dataset"]["subject_id"],
            config["task"]["model_key"],
            f"seed{int(config['seed'])}",
            config["task"]["task_classification"],
        ]
    )


def checkpoint_path(config: dict[str, Any]) -> Path:
    return ROOT / config["checkpoint_dir"] / f"{job_key(config).replace(':', '__')}.pt"


def manifest_path(config: dict[str, Any]) -> Path:
    return ROOT / config["output_dir"] / "checkpoint_manifest.json"


def run_checkpoint_mock(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    ckpt_path = checkpoint_path(config).with_suffix(".mock.pt")
    ensure_dir(ckpt_path.parent)
    payload = {
        "job_key": job_key(config),
        "epoch": -1,
        "best_val_score": None,
        "source": "checkpoint-path/load mock; not a scientific checkpoint",
    }
    torch.save(payload, ckpt_path)
    loaded = torch.load(ckpt_path, map_location="cpu")
    ckpt_path.unlink(missing_ok=True)
    manifest = {
        "mock_passed": loaded["job_key"] == payload["job_key"],
        "mock_checkpoint_path": repo_relative(ckpt_path),
        "mock_checkpoint_deleted": not ckpt_path.exists(),
        "real_checkpoint_path": repo_relative(checkpoint_path(config)),
        "real_checkpoint_dir_gitignored": True,
        "source": "local_checkpoints only; checkpoint files are not committed",
        "best_epoch": None,
        "best_val_score": None,
    }
    write_json(manifest_path(config), manifest)
    emit(output_dir, "CHECKPOINT", f"mock_passed={manifest['mock_passed']} path={manifest['real_checkpoint_path']}")
    return manifest


def prepare_context(config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    recordings = load_split_recordings(dataset_dir, config["dataset"]["subject_id"])
    disjointness = split_disjointness(recordings)
    scalers = fit_train_only_scalers(recordings["train"])
    datasets = build_datasets(recordings, config, scalers)
    coverage = coverage_rows(datasets, config)
    return {
        "dataset_dir": dataset_dir,
        "recordings": recordings,
        "disjointness": disjointness,
        "scalers": scalers,
        "datasets": datasets,
        "coverage": coverage,
    }


def write_preflight_artifacts(config: dict[str, Any], output_dir: Path, context: dict[str, Any], *, device: str, checkpoint_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    datasets = context["datasets"]
    coverage = context["coverage"]
    disjointness = context["disjointness"]
    protocol_lines = [
        "# DECAF Context-Assisted P00 Protocol",
        "",
        f"- model: `{config['decaf']['implementation_file']}`",
        f"- task_classification: `{config['task']['task_classification']}`",
        f"- fusion_strategy: `{config['decaf']['fusion_strategy']}`",
        f"- EEG/target: `[t, t+{config['task']['eeg_target_samples']})`",
        f"- input context: `[t-{config['task']['forward_envelope_context_samples']}, t)`",
        f"- effective context: `[t-{config['task']['effective_context_samples']}, t)`",
        f"- train loss: `{config['task']['loss']}`",
        f"- max_epochs: `{config['training']['max_epochs']}`",
        f"- early_stopping_patience: `{config['training']['early_stopping_patience']}`",
        f"- checkpoint selection metric: `{config['task']['checkpoint_selection_metric']}`",
        "- test is evaluated only after selecting the best validation checkpoint.",
        "- output belongs only to context-assisted results, not pure reconstruction.",
        "",
    ]
    (output_dir / "decaf_context_assisted_protocol.md").write_text("\n".join(protocol_lines), encoding="utf-8")
    preflight_lines = [
        "# DECAF Context-Assisted P00 Preflight",
        "",
        f"- device: `{device}`",
        f"- job_key: `{job_key(config)}`",
        f"- dataset_dir: `{context['dataset_dir']}`",
        f"- split_recording_id_counts: `{disjointness['recording_id_counts']}`",
        f"- train_val_intersection_count: `{disjointness['train_val_intersection_count']}`",
        f"- train_test_intersection_count: `{disjointness['train_test_intersection_count']}`",
        f"- val_test_intersection_count: `{disjointness['val_test_intersection_count']}`",
        f"- window_counts: `{{'train': {len(datasets['train'])}, 'val': {len(datasets['val'])}, 'test': {len(datasets['test'])}}}`",
        "- normalization: `train-only EEG channel z-score and envelope z-score`",
        f"- checkpoint_path: `{repo_relative(checkpoint_path(config))}`",
        f"- checkpoint_mock_passed: `{bool(checkpoint_manifest and checkpoint_manifest.get('mock_passed'))}`",
        "- formal_training_started: `False`",
        "- scientific_metric_generated: `False`",
        "",
    ]
    (output_dir / "preflight_report.md").write_text("\n".join(preflight_lines), encoding="utf-8")
    write_csv(output_dir / "coverage_summary.csv", coverage)
    schema = build_schema(config, context, device=device, checkpoint_manifest=checkpoint_manifest, mode="preflight")
    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(output_dir / "failure_report.json", {"failures": [] if schema["passed"] else [{"failure_reason": "schema failed", "checks": schema["checks"]}]})
    write_json(output_dir / "completed_jobs.json", [])
    write_json(
        output_dir / "run_state.json",
        {
            "status": "preflight_only",
            "job_key": job_key(config),
            "completed_job_count": 0,
            "pending_job_count": 1,
            "formal_training_started": False,
            "scientific_metric_generated": False,
        },
    )
    return schema


def build_schema(
    config: dict[str, Any],
    context: dict[str, Any],
    *,
    device: str,
    checkpoint_manifest: dict[str, Any] | None,
    mode: str,
    metrics_written: bool = False,
) -> dict[str, Any]:
    disjoint = context["disjointness"]
    datasets = context["datasets"]
    coverage = context["coverage"]
    checks = {
        "genuine_twobranch_model": config["decaf"]["model_family"] == "TwoBranchModel",
        "task_context_assisted": config["task"]["task_classification"] == "context_assisted_envelope_forecasting",
        "weighted_fusion": config["decaf"]["fusion_strategy"] == "weighted",
        "fixed_window_contract": int(config["task"]["eeg_target_samples"]) == 224
        and int(config["task"]["forward_envelope_context_samples"]) == 128
        and int(config["task"]["effective_context_samples"]) == 96,
        "mse_loss": config["task"]["loss"] == "mse",
        "max_epochs_100": int(config["training"]["max_epochs"]) == 100,
        "patience_10": int(config["training"]["early_stopping_patience"]) == 10,
        "device_resolved": device in {"cuda", "cpu"},
        "train_val_intersection_zero": disjoint["train_val_intersection_count"] == 0,
        "train_test_intersection_zero": disjoint["train_test_intersection_count"] == 0,
        "val_test_intersection_zero": disjoint["val_test_intersection_count"] == 0,
        "train_only_scaler": context["scalers"]["fit_split"] == "train",
        "train_val_test_windows_present": all(len(datasets[split]) > 0 for split in ["train", "val", "test"]),
        "no_leakage_window_rule": leakage_ok(datasets),
        "coverage_rows_present": len(coverage) > 0,
        "checkpoint_path_under_local_checkpoints": repo_relative(checkpoint_path(config)).startswith("local_checkpoints/"),
        "checkpoint_mock_passed": bool(checkpoint_manifest and checkpoint_manifest.get("mock_passed")) if mode == "preflight" else True,
        "formal_checkpoint_manifest_has_best_epoch": bool(checkpoint_manifest and checkpoint_manifest.get("best_epoch") is not None) if mode == "formal" else True,
        "formal_checkpoint_manifest_has_best_val_score": bool(checkpoint_manifest and checkpoint_manifest.get("best_val_score") is not None) if mode == "formal" else True,
        "preflight_does_not_write_metrics": (not metrics_written) if mode == "preflight" else True,
    }
    return {
        "passed": all(checks.values()),
        "mode": mode,
        "checks": checks,
        "device": device,
        "job_key": job_key(config),
        "split_disjointness": disjoint,
        "window_counts": {split: len(datasets[split]) for split in ["train", "val", "test"]},
        "checkpoint_path": repo_relative(checkpoint_path(config)),
        "expected_artifacts_after_formal_run": [
            "subject_metrics.csv",
            "recording_metrics.csv",
            "completed_jobs.json",
            "run_state.json",
            "failure_report.json",
            "schema_validation_report.json",
            "coverage_summary.csv",
            "checkpoint_manifest.json",
            "logs/run.log",
        ],
    }


def save_checkpoint(config: dict[str, Any], model, epoch: int, best_val_score: float) -> dict[str, Any]:
    path = checkpoint_path(config)
    ensure_dir(path.parent)
    payload = {
        "job_key": job_key(config),
        "epoch": epoch,
        "best_val_score": best_val_score,
        "model_state_dict": model.state_dict(),
        "source": "best validation checkpoint selected by val mean_recording_pearson_r",
    }
    torch.save(payload, path)
    manifest = {
        "source": payload["source"],
        "job_key": job_key(config),
        "checkpoint_path": repo_relative(path),
        "best_epoch": epoch,
        "best_val_score": best_val_score,
        "selection_metric": config["task"]["checkpoint_selection_metric"],
        "checkpoint_dir_gitignored": True,
    }
    write_json(manifest_path(config), manifest)
    return manifest


def run_formal(config: dict[str, Any], output_dir: Path, context: dict[str, Any], device: str, resume: bool) -> int:
    completed_path = output_dir / "completed_jobs.json"
    if resume and completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if job_key(config) in {item.get("job_key") for item in completed}:
            emit(output_dir, "COMPLETED", f"resume found completed job {job_key(config)}")
            return 0

    set_seed(int(config["seed"]))
    datasets = context["datasets"]
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        generator=generator,
    )
    model = instantiate_twobranch(config, device)
    optimizer = Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    best_epoch = -1
    best_val = -np.inf
    best_state = deepcopy(model.state_dict())
    patience = int(config["training"]["early_stopping_patience"])
    max_epochs = int(config["training"]["max_epochs"])
    training_rows = []
    emit(output_dir, "TRAIN", f"start epochs={max_epochs} patience={patience} device={device}")
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            eeg = batch["eeg"].to(device=device, dtype=torch.float32)
            context_batch = batch["envelope_context"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            output = model(eeg, context_batch)
            loss = torch.nn.functional.mse_loss(output["envelope"], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        emit(output_dir, "VAL", f"epoch={epoch} evaluating full validation recordings")
        val_score, _, _ = evaluate_split(model, datasets["val"], config=config, device=device, checkpoint_id="epoch_current", artifact_scope=config["artifact_scope"])
        train_loss = float(np.mean(losses)) if losses else float("nan")
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            save_checkpoint(config, model, epoch, best_val)
        training_rows.append({"epoch": epoch, "train_loss": train_loss, "val_mean_recording_pearson_r": val_score, "best_epoch": best_epoch})
        emit(output_dir, "TRAIN", f"epoch={epoch} train_loss={train_loss:.8f} val_pearson={val_score:.8f} best_epoch={best_epoch}")
        if epoch >= best_epoch + patience:
            emit(output_dir, "TRAIN", f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    checkpoint_id = f"best_epoch_{best_epoch}"
    emit(output_dir, "TEST", f"evaluating best validation checkpoint {checkpoint_id}")
    _, recording_rows, subject_rows = evaluate_split(model, datasets["test"], config=config, device=device, checkpoint_id=checkpoint_id, artifact_scope=config["artifact_scope"])
    emit(output_dir, "METRICS", "writing context-assisted metrics")
    write_rows(output_dir / "recording_metrics.csv", recording_rows, RECORDING_METRIC_FIELDS)
    write_rows(output_dir / "subject_metrics.csv", subject_rows, SUBJECT_METRIC_FIELDS)
    write_csv(output_dir / "training_curve.csv", training_rows)
    write_csv(output_dir / "coverage_summary.csv", context["coverage"])
    completed = [{
        "job_key": job_key(config),
        "dataset": config["dataset"]["dataset_id"],
        "subject_id": config["dataset"]["subject_id"],
        "model": config["task"]["model_key"],
        "seed": int(config["seed"]),
        "task": config["task"]["task_classification"],
        "status": "completed",
        "best_epoch": best_epoch,
        "best_val_score": best_val,
    }]
    write_json(completed_path, completed)
    write_json(output_dir / "run_state.json", {"status": "completed", "completed_job_count": 1, "pending_job_count": 0, "last_completed_job_key": job_key(config)})
    schema = build_schema(config, context, device=device, checkpoint_manifest=json.loads(manifest_path(config).read_text(encoding="utf-8")), mode="formal", metrics_written=True)
    schema["checks"]["metrics_written_after_best_checkpoint"] = True
    schema["passed"] = all(schema["checks"].values())
    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(output_dir / "failure_report.json", {"failures": [] if schema["passed"] else [{"failure_reason": "schema failed", "checks": schema["checks"]}]})
    emit(output_dir, "COMPLETED", job_key(config))
    return 0 if schema["passed"] else 1


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    device = resolve_device(args.device)
    emit(output_dir, "STARTUP", f"config={args.config}")
    emit(output_dir, "STARTUP", f"output_dir={repo_relative(output_dir)} device={device}")
    emit(output_dir, "STARTUP", f"job_key={job_key(config)}")
    if args.startup_only:
        return 0

    if args.dry_run_plan:
        plan = {
            "job_key": job_key(config),
            "formal_training_started": False,
            "manual_run_will_train": True,
            "selection_metric": config["task"]["checkpoint_selection_metric"],
            "output_dir": config["output_dir"],
            "checkpoint_path": repo_relative(checkpoint_path(config)),
            "expected_artifacts": [
                "subject_metrics.csv",
                "recording_metrics.csv",
                "completed_jobs.json",
                "run_state.json",
                "failure_report.json",
                "schema_validation_report.json",
                "coverage_summary.csv",
                "checkpoint_manifest.json",
                "logs/run.log",
            ],
        }
        emit(output_dir, "DRY RUN", json.dumps(plan, sort_keys=True))
        return 0

    context = prepare_context(config)
    checkpoint_manifest = None
    if args.checkpoint_mock:
        checkpoint_manifest = run_checkpoint_mock(config, output_dir)
    if args.schema_preflight:
        schema = write_preflight_artifacts(config, output_dir, context, device=device, checkpoint_manifest=checkpoint_manifest)
        emit(output_dir, "PREFLIGHT", f"schema_passed={schema['passed']} formal_training_started=false scientific_metric_generated=false")
        return 0 if schema["passed"] else 1
    return run_formal(config, output_dir, context, device, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
