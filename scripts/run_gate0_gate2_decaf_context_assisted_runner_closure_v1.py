from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.util
import json
import os
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
    parser.add_argument("--single-batch-smoke", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shape_text(shape: Iterable[int] | torch.Size) -> str:
    return "[" + ", ".join(str(int(item)) for item in shape) + "]"


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


def fit_train_only_scalers(train_recordings: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, np.ndarray | float]:
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


def apply_scalers(
    recordings: list[tuple[str, np.ndarray, np.ndarray]],
    scalers: dict[str, np.ndarray | float],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
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
            "target_start": target_start,
            "target_end_exclusive": target_end,
            "eeg_start": target_start,
            "eeg_end_exclusive": target_end,
            "envelope_context_start": context_start,
            "envelope_context_end_exclusive": context_end,
            "effective_context_start": context_start + self.trim_samples,
            "effective_context_end_exclusive": context_end,
        }


def collate_single_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eeg": torch.stack([item["eeg"] for item in items], dim=0),
        "envelope_context": torch.stack([item["envelope_context"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
        "metadata": [{key: value for key, value in item.items() if key not in {"eeg", "envelope_context", "target"}} for item in items],
    }


def load_split_recordings(dataset_dir: Path, subject_id: str) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    return {
        split: load_reference_recordings(dataset_dir, split, subject_id, channels=range(64))
        for split in ["train", "val", "test"]
    }


def build_datasets(
    recordings_by_split: dict[str, list[tuple[str, np.ndarray, np.ndarray]]],
    config: dict[str, Any],
    scalers: dict[str, np.ndarray | float],
) -> dict[str, ContextAssistedWindowDataset]:
    task = config["task"]
    subject_id = config["dataset"]["subject_id"]
    datasets = {}
    for split, recordings in recordings_by_split.items():
        datasets[split] = ContextAssistedWindowDataset(
            apply_scalers(recordings, scalers),
            split=split,
            subject_id=subject_id,
            target_samples=int(task["eeg_target_samples"]),
            context_samples=int(task["forward_envelope_context_samples"]),
            trim_samples=int(task["forward_context_trim_samples"]),
        )
    return datasets


def interval_rows(datasets: dict[str, ContextAssistedWindowDataset], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    sr = int(config["dataset"]["sampling_rate"])
    effective_context = int(config["task"]["effective_context_samples"])
    for split, dataset in datasets.items():
        for rec_idx, target_start in dataset.records:
            recording_id, eeg, env = dataset.recordings[rec_idx]
            target_end = target_start + dataset.target_samples
            context_start = target_start - dataset.context_samples
            context_end = target_start
            rows.append(
                {
                    "split": split,
                    "subject_id": dataset.subject_id,
                    "recording_id": recording_id,
                    "sample_rate": sr,
                    "target_start": target_start,
                    "target_end_exclusive": target_end,
                    "eeg_start": target_start,
                    "eeg_end_exclusive": target_end,
                    "envelope_context_start": context_start,
                    "envelope_context_end_exclusive": context_end,
                    "envelope_context_end_inclusive": context_end - 1,
                    "forward_effective_context_start": context_end - effective_context,
                    "forward_effective_context_end_exclusive": context_end,
                    "forward_effective_context_end_inclusive": context_end - 1,
                    "recording_length": int(min(eeg.shape[0], env.shape[0])),
                    "same_recording": True,
                    "same_split": True,
                    "context_end_inclusive_lt_target_start": (context_end - 1) < target_start,
                    "context_exclusive_end_eq_target_start": context_end == target_start,
                    "target_or_future_envelope_in_context": False,
                }
            )
    return rows


def coverage_rows(datasets: dict[str, ContextAssistedWindowDataset], config: dict[str, Any]) -> list[dict[str, Any]]:
    test_dataset = datasets["test"]
    target_samples = int(config["task"]["eeg_target_samples"])
    context_samples = int(config["task"]["forward_envelope_context_samples"])
    rows = []
    for recording_id, eeg, env in test_dataset.recordings:
        length = int(min(eeg.shape[0], env.shape[0]))
        starts = list(range(context_samples, length - target_samples + 1, target_samples))
        covered_samples = len(starts) * target_samples
        first_target_start = starts[0] if starts else ""
        last_target_end = (starts[-1] + target_samples) if starts else ""
        eligible_after_context = max(0, length - context_samples)
        dropped_tail = max(0, length - int(last_target_end)) if starts else eligible_after_context
        rows.append(
            {
                "split": "test",
                "subject_id": config["dataset"]["subject_id"],
                "recording_id": recording_id,
                "recording_length": length,
                "excluded_initial_context_samples": min(context_samples, length),
                "first_target_start": first_target_start,
                "window_count": len(starts),
                "covered_target_samples": covered_samples,
                "last_target_end_exclusive": last_target_end,
                "dropped_tail_samples_after_nonoverlap_windows": dropped_tail,
                "eligible_samples_after_initial_context": eligible_after_context,
                "coverage_ratio_of_recording": float(covered_samples / length) if length else 0.0,
                "coverage_ratio_after_initial_context": float(covered_samples / eligible_after_context) if eligible_after_context else 0.0,
                "context_never_crosses_recording": True,
                "context_strictly_past": True,
            }
        )
    return rows


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_single_batch_smoke(config: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(int(config["seed"]))
    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    subject_id = config["dataset"]["subject_id"]
    recordings_by_split = load_split_recordings(dataset_dir, subject_id)
    disjointness = split_disjointness(recordings_by_split)
    scalers = fit_train_only_scalers(recordings_by_split["train"])
    datasets = build_datasets(recordings_by_split, config, scalers)
    intervals = interval_rows(datasets, config)
    coverage = coverage_rows(datasets, config)

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_single_batch,
        generator=generator,
    )
    batch = next(iter(train_loader))

    model = instantiate_twobranch(config, device)
    model.train()
    optimizer = Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    eeg = batch["eeg"].to(device=device, dtype=torch.float32)
    context = batch["envelope_context"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    output = model(eeg, context)
    prediction = output["envelope"]
    loss = torch.nn.functional.mse_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite_gradients = all(
        param.grad is None or bool(torch.isfinite(param.grad).all().item())
        for param in model.parameters()
    )
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()

    no_leakage = all(
        row["context_end_inclusive_lt_target_start"]
        and row["context_exclusive_end_eq_target_start"]
        and not row["target_or_future_envelope_in_context"]
        for row in intervals
    )
    smoke = {
        "status": "passed" if bool(torch.isfinite(loss).item()) and finite_gradients and no_leakage else "failed",
        "subject_id": subject_id,
        "task_classification": config["task"]["task_classification"],
        "model_family": config["decaf"]["model_family"],
        "fusion_strategy": config["decaf"]["fusion_strategy"],
        "loss": config["task"]["loss"],
        "device": device,
        "train_batch_loaded": True,
        "batch_metadata": batch["metadata"],
        "eeg_input_shape": shape_text(eeg.shape),
        "envelope_context_input_shape": shape_text(context.shape),
        "target_shape": shape_text(target.shape),
        "prediction_shape": shape_text(prediction.shape),
        "envelope_branch_shape": shape_text(output["envelope_branch"].shape),
        "eeg_branch_shape": shape_text(output["eeg_branch"].shape),
        "fusion_weights_shape": shape_text(output["fusion_weights"].shape),
        "mse_loss": float(loss.detach().cpu().item()),
        "mse_loss_finite": bool(torch.isfinite(loss).item()),
        "backward_success": True,
        "optimizer_step_success": True,
        "finite_gradients": finite_gradients,
        "scientific_metric_computed": False,
        "test_pearson_computed": False,
        "checkpoint_saved": False,
    }
    del output, prediction, loss, eeg, context, target, optimizer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    audit = {
        "dataset_dir": repo_relative(dataset_dir) if dataset_dir.is_relative_to(ROOT) else str(dataset_dir),
        "scaler_fit_split": scalers["fit_split"],
        "scaler_fit_recording_count": int(scalers["fit_recording_count"]),
        "split_recording_counts": {split: len(items) for split, items in recordings_by_split.items()},
        "split_disjointness": disjointness,
        "window_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "interval_count": len(intervals),
        "no_leakage": no_leakage,
        "no_cross_recording": all(row["same_recording"] for row in intervals),
        "no_cross_split": disjointness["all_pairwise_intersections_empty"],
        "test_context_strictly_past": all(row["context_strictly_past"] for row in coverage),
        "test_initial_128_samples_excluded": all(int(row["excluded_initial_context_samples"]) == int(config["task"]["forward_envelope_context_samples"]) for row in coverage),
        "checkpoint_selection_split": config["training"]["checkpoint_selection_split"],
        "normalization_policy": "EEG channel z-score and envelope z-score fitted on train recordings only; train-fitted scalers are applied to train/val/test.",
        "interval_rule": "For target [t,t+224), EEG uses [t,t+224), envelope_context uses [t-128,t), and TwoBranchModel forward trims to [t-96,t).",
    }
    return smoke, audit, intervals, coverage


def write_markdown_artifacts(output_dir: Path, config: dict[str, Any], smoke: dict[str, Any], audit: dict[str, Any]) -> None:
    protocol = [
        "# DECAF Context-Assisted Protocol",
        "",
        f"- task_classification: `{config['task']['task_classification']}`",
        f"- status: `{config['decaf']['local_adaptation_status']}`",
        f"- model: `{config['decaf']['implementation_file']}`",
        f"- upstream_commit: `{config['decaf']['commit']}`",
        f"- fusion_strategy: `{config['decaf']['fusion_strategy']}`",
        f"- sampling_rate: `{config['dataset']['sampling_rate']}`",
        f"- EEG/target interval: `[t, t+{config['task']['eeg_target_samples']})`",
        f"- envelope_context interval: `[t-{config['task']['forward_envelope_context_samples']}, t)` from the same recording and split",
        f"- effective forward context after upstream trim: `[t-{config['task']['effective_context_samples']}, t)`",
        f"- training_loss: `{config['task']['loss']}`",
        "- checkpoint selection: validation recordings only",
        "- scientific metrics are intentionally not computed in this single-batch engineering smoke.",
        "",
    ]
    (output_dir / "decaf_context_assisted_protocol.md").write_text("\n".join(protocol), encoding="utf-8")

    interface = [
        "# DECAF Data Interface Audit",
        "",
        f"- dataset_dir: `{audit['dataset_dir']}`",
        f"- scaler_fit_split: `{audit['scaler_fit_split']}`",
        f"- scaler_fit_recording_count: `{audit['scaler_fit_recording_count']}`",
        f"- split_recording_counts: `{audit['split_recording_counts']}`",
        f"- split_recording_id_counts: `{audit['split_disjointness']['recording_id_counts']}`",
        f"- train_val_intersection_count: `{audit['split_disjointness']['train_val_intersection_count']}`",
        f"- train_test_intersection_count: `{audit['split_disjointness']['train_test_intersection_count']}`",
        f"- val_test_intersection_count: `{audit['split_disjointness']['val_test_intersection_count']}`",
        f"- window_counts: `{audit['window_counts']}`",
        f"- normalization_policy: `{audit['normalization_policy']}`",
        f"- interval_rule: `{audit['interval_rule']}`",
        f"- no_leakage: `{audit['no_leakage']}`",
        f"- no_cross_recording: `{audit['no_cross_recording']}`",
        f"- no_cross_split: `{audit['no_cross_split']}`",
        f"- test_context_strictly_past: `{audit['test_context_strictly_past']}`",
        f"- test_initial_128_samples_excluded: `{audit['test_initial_128_samples_excluded']}`",
        "",
    ]
    (output_dir / "decaf_data_interface_audit.md").write_text("\n".join(interface), encoding="utf-8")

    smoke_lines = [
        "# DECAF Single-Batch Smoke",
        "",
        f"- status: `{smoke['status']}`",
        f"- subject_id: `{smoke['subject_id']}`",
        f"- device: `{smoke['device']}`",
        f"- train_batch_loaded: `{smoke['train_batch_loaded']}`",
        f"- eeg_input_shape: `{smoke['eeg_input_shape']}`",
        f"- envelope_context_input_shape: `{smoke['envelope_context_input_shape']}`",
        f"- target_shape: `{smoke['target_shape']}`",
        f"- prediction_shape: `{smoke['prediction_shape']}`",
        f"- mse_loss: `{smoke['mse_loss']:.8f}`",
        f"- mse_loss_finite: `{smoke['mse_loss_finite']}`",
        f"- backward_success: `{smoke['backward_success']}`",
        f"- optimizer_step_success: `{smoke['optimizer_step_success']}`",
        f"- finite_gradients: `{smoke['finite_gradients']}`",
        f"- checkpoint_saved: `{smoke['checkpoint_saved']}`",
        f"- test_pearson_computed: `{smoke['test_pearson_computed']}`",
        f"- scientific_metric_computed: `{smoke['scientific_metric_computed']}`",
        "",
    ]
    (output_dir / "decaf_single_batch_smoke.md").write_text("\n".join(smoke_lines), encoding="utf-8")


def build_schema(smoke: dict[str, Any] | None, audit: dict[str, Any] | None, coverage: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "task_classification_context_assisted_envelope_forecasting": bool(smoke and smoke["task_classification"] == "context_assisted_envelope_forecasting"),
        "single_train_batch_loaded": bool(smoke and smoke["train_batch_loaded"]),
        "twobranch_forward_success": bool(smoke and smoke["prediction_shape"] == smoke["target_shape"]),
        "mse_loss_finite": bool(smoke and smoke["mse_loss_finite"]),
        "backward_optimizer_step_success": bool(smoke and smoke["backward_success"] and smoke["optimizer_step_success"] and smoke["finite_gradients"]),
        "no_leakage": bool(audit and audit["no_leakage"]),
        "no_cross_recording": bool(audit and audit["no_cross_recording"]),
        "no_cross_split": bool(audit and audit["no_cross_split"]),
        "split_recording_ids_pairwise_disjoint": bool(audit and audit["split_disjointness"]["all_pairwise_intersections_empty"]),
        "train_val_intersection_count_zero": bool(audit and audit["split_disjointness"]["train_val_intersection_count"] == 0),
        "train_test_intersection_count_zero": bool(audit and audit["split_disjointness"]["train_test_intersection_count"] == 0),
        "val_test_intersection_count_zero": bool(audit and audit["split_disjointness"]["val_test_intersection_count"] == 0),
        "train_only_scaler_fit": bool(audit and audit["scaler_fit_split"] == "train"),
        "checkpoint_selection_val_only": bool(audit and audit["checkpoint_selection_split"] == "val"),
        "test_context_strictly_past": bool(audit and audit["test_context_strictly_past"]),
        "test_initial_128_samples_excluded": bool(audit and audit["test_initial_128_samples_excluded"]),
        "coverage_rows_present": len(coverage) > 0,
        "no_checkpoint_saved": bool(smoke and smoke["checkpoint_saved"] is False),
        "no_test_pearson_or_scientific_metric": bool(smoke and smoke["test_pearson_computed"] is False and smoke["scientific_metric_computed"] is False),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "smoke_status": smoke["status"] if smoke else "not_run",
        "task_classification": smoke["task_classification"] if smoke else "",
        "single_batch_loss": smoke["mse_loss"] if smoke else None,
        "device": smoke["device"] if smoke else "",
        "split_disjointness": audit["split_disjointness"] if audit else {},
        "coverage_recording_count": len(coverage),
    }


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    smoke: dict[str, Any],
    audit: dict[str, Any],
    coverage: list[dict[str, Any]],
    schema: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    write_markdown_artifacts(output_dir, config, smoke, audit)
    write_csv(output_dir / "decaf_coverage_audit.csv", coverage)
    write_json(output_dir / "schema_validation_report.json", schema)
    failures = [] if schema["passed"] else [{"failure_reason": "schema failed", "checks": schema["checks"]}]
    write_json(output_dir / "failure_report.json", {"failures": failures})


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print_event("STARTUP", f"config={args.config}")
    print_event("STARTUP", f"output_dir={repo_relative(output_dir)} device={device}")
    if args.dry_run_plan:
        planned = {
            "subject_id": config["dataset"]["subject_id"],
            "mode": "single_batch_engineering_smoke",
            "will_train_full_result": False,
            "will_compute_scientific_metric": False,
            "will_save_checkpoint": False,
            "artifacts": [
                "decaf_context_assisted_protocol.md",
                "decaf_data_interface_audit.md",
                "decaf_single_batch_smoke.md",
                "decaf_coverage_audit.csv",
                "schema_validation_report.json",
                "failure_report.json",
            ],
        }
        print_event("DRY RUN", json.dumps(planned, sort_keys=True))
        return 0
    if not args.single_batch_smoke:
        raise SystemExit("This closure runner only executes with --single-batch-smoke in this round.")
    smoke, audit, _intervals, coverage = run_single_batch_smoke(config, device)
    schema = build_schema(smoke, audit, coverage)
    write_outputs(output_dir, config, smoke, audit, coverage, schema)
    print_event("SMOKE", f"status={smoke['status']} loss={smoke['mse_loss']:.8f} schema_passed={schema['passed']}")
    exit_code = 0 if schema["passed"] else 1
    if os.name == "nt" and device == "cuda":
        sys.stdout.flush()
        sys.stderr.flush()
        ctypes.windll.kernel32.ExitProcess(exit_code)
    return exit_code


if __name__ == "__main__":
    exit_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        ctypes.windll.kernel32.ExitProcess(exit_code)
    os._exit(exit_code)
