from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import platform
import random
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import NAdam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.scoring import WindowPrediction
from repro.mldecoders.models import VLAAIExactOfficialRegressor
from repro.reference_baselines import ReferenceWindowDataset, correlation, load_reference_recordings

import run_gate0_gate2_subject_specific_local_model_full_eval_p00_v1 as local_full_eval
from run_gate0_gate2_subject_specific_local_model_smoke_v1 import resolve_dataset_path


RUN_FIELDS = [
    "run_index",
    "config_sha256",
    "runner_source_sha256",
    "python_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "device",
    "deterministic_settings",
    "initial_state_hash",
    "first_train_batch_hash",
    "after_first_optimizer_step_hash",
    "best_state_hash",
    "prediction_hash",
    "best_epoch",
    "epochs_completed",
    "best_val_score",
    "test_metric",
]


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray, np.float32]:
        x, y = self.base[index]
        return int(index), x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--single-batch-smoke", action="store_true")
    parser.add_argument("--run-repeats", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=None)
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


def print_event(label: str, message: str) -> None:
    print(f"[{label}] {message}", flush=True)


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    array = np.asarray(array).astype(np.float32, copy=False)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def hash_tensor(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def batch_hash(indices: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(hash_tensor(indices.to(dtype=torch.int64)).encode("utf-8"))
    digest.update(hash_tensor(x).encode("utf-8"))
    digest.update(hash_tensor(y).encode("utf-8"))
    return digest.hexdigest()


def set_deterministic(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_manual_seed": seed,
        "torch_cuda_manual_seed_all": bool(torch.cuda.is_available()),
        "dataloader_generator_seed": seed,
        "num_workers": 0,
        "torch_backends_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_backends_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": False,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def env_audit(config_path: Path, runner_path: Path, device: str, deterministic_settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_sha256": sha256_file(config_path),
        "runner_source_sha256": sha256_file(runner_path),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "deterministic_settings": deterministic_settings,
    }


def source_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    source = load_json(ROOT / config["source_config"])
    source["output_dir"] = config["output_dir"]
    source["artifact_scope"] = config["artifact_scope"]
    source["protocol"] = config["protocol"]
    return source


def build_loaders(
    dataset_dir: Path,
    subject_id: str,
    window_size: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = IndexedDataset(ReferenceWindowDataset(dataset_dir, "train", subject_id, window_size=window_size, channels=range(64)))
    val_dataset = IndexedDataset(ReferenceWindowDataset(dataset_dir, "val", subject_id, window_size=window_size, channels=range(64)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    return train_loader, val_loader


def train_one_run(config_path: Path, config: dict[str, Any], device: str, run_index: int, *, max_batches: int | None = None) -> dict[str, Any]:
    seed = int(config["seed"])
    deterministic_settings = set_deterministic(seed)
    source_config = source_runtime_config(config)
    dataset_cfg = config["dataset"]
    dataset_dir = resolve_dataset_path(dataset_cfg["dataset_locator"])
    subject_id = str(config["subject_id"])
    model_cfg = source_config["vlaai"]
    model_kwargs = {"num_input_channels": 64, "input_length": int(model_cfg["window_size"])}
    model = VLAAIExactOfficialRegressor(**model_kwargs).to(device)
    optimizer = NAdam(model.parameters(), lr=float(model_cfg["learning_rate"]), weight_decay=float(model_cfg["weight_decay"]))
    train_loader, val_loader = build_loaders(
        dataset_dir,
        subject_id,
        int(model_cfg["window_size"]),
        int(model_cfg["batch_size"]),
        device,
        seed,
    )
    initial_state_hash = hash_state_dict(model.state_dict())
    first_train_batch_hash = ""
    after_first_optimizer_step_hash = ""
    best_val = -float("inf")
    best_epoch = -1
    best_state = deepcopy(model.state_dict())
    val_history: list[float] = []
    batches_seen = 0
    for epoch in range(int(model_cfg["max_epochs"])):
        if best_epoch >= 0 and epoch > best_epoch + int(model_cfg["early_stopping_patience"]):
            break
        model.train()
        for indices, x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            indices = indices.to(device=device, dtype=torch.int64)
            if not first_train_batch_hash:
                first_train_batch_hash = batch_hash(indices, x, y)
            y_hat = model(x)
            loss = -correlation(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches_seen += 1
            if not after_first_optimizer_step_hash:
                after_first_optimizer_step_hash = hash_state_dict(model.state_dict())
            if max_batches is not None and batches_seen >= max_batches:
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                best_val = float("nan")
                val_history.append(float("nan"))
                return build_result_row(
                    config_path=config_path,
                    config=config,
                    device=device,
                    run_index=run_index,
                    deterministic_settings=deterministic_settings,
                    initial_state_hash=initial_state_hash,
                    first_train_batch_hash=first_train_batch_hash,
                    after_first_optimizer_step_hash=after_first_optimizer_step_hash,
                    best_state=best_state,
                    best_epoch=best_epoch + 1,
                    epochs_completed=len(val_history),
                    best_val_score=best_val,
                    test_metric=float("nan"),
                    prediction_hash="",
                )
        model.eval()
        scores = []
        with torch.no_grad():
            for _indices, x, y in val_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                scores.append(correlation(y, model(x)).item())
        val_score = float(np.mean(scores))
        val_history.append(val_score)
        print_event("EPOCH", f"run={run_index} epoch={epoch + 1} val={val_score:.9f} best_epoch={best_epoch + 1 if best_epoch >= 0 else ''}")
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    test_metric, prediction_hash = evaluate_test(config, source_config, dataset_dir, subject_id, best_state, device)
    return build_result_row(
        config_path=config_path,
        config=config,
        device=device,
        run_index=run_index,
        deterministic_settings=deterministic_settings,
        initial_state_hash=initial_state_hash,
        first_train_batch_hash=first_train_batch_hash,
        after_first_optimizer_step_hash=after_first_optimizer_step_hash,
        best_state=best_state,
        best_epoch=best_epoch + 1,
        epochs_completed=len(val_history),
        best_val_score=best_val,
        test_metric=test_metric,
        prediction_hash=prediction_hash,
    )


def evaluate_test(
    config: dict[str, Any],
    source_config: dict[str, Any],
    dataset_dir: Path,
    subject_id: str,
    state_dict: dict[str, torch.Tensor],
    device: str,
) -> tuple[float, str]:
    model_cfg = source_config["vlaai"]
    input_length = int(model_cfg["window_size"])
    model = VLAAIExactOfficialRegressor(num_input_channels=64, input_length=input_length).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    windows: list[WindowPrediction] = []
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for recording_id, eeg, env in load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64)):
            preds: list[float] = []
            targets: list[float] = []
            eeg_tensor = torch.from_numpy(eeg.astype(np.float32))
            for start in range(0, eeg.shape[0] - input_length + 1):
                batch = eeg_tensor[start : start + input_length].T.unsqueeze(0).to(device)
                preds.append(float(model(batch).item()))
                targets.append(float(env[start + input_length - 1]))
            pred_arr = np.asarray(preds, dtype=np.float32)
            target_arr = np.asarray(targets, dtype=np.float32)
            all_predictions.append(pred_arr)
            windows.append(
                local_full_eval.build_series_window(
                    dataset=config["dataset"]["dataset_id"],
                    model="vlaai",
                    protocol=config["protocol"],
                    seed=int(config["seed"]),
                    subject_id=subject_id,
                    recording_id=recording_id,
                    sampling_rate=int(config["dataset"]["sampling_rate"]),
                    checkpoint_id="vlaai_deterministic_best_state",
                    full_length=len(env),
                    offset=input_length - 1,
                    prediction=pred_arr,
                    target=target_arr,
                )
            )
    _recording_rows, subject_rows = local_full_eval.aggregate_model_outputs(windows, config["artifact_scope"])
    metric = float(subject_rows[0].metric_value) if subject_rows else float("nan")
    return metric, hash_array(np.concatenate(all_predictions)) if all_predictions else ""


def build_result_row(
    *,
    config_path: Path,
    config: dict[str, Any],
    device: str,
    run_index: int,
    deterministic_settings: dict[str, Any],
    initial_state_hash: str,
    first_train_batch_hash: str,
    after_first_optimizer_step_hash: str,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    epochs_completed: int,
    best_val_score: float,
    test_metric: float,
    prediction_hash: str,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    return {
        "run_index": run_index,
        "config_sha256": sha256_file(config_path),
        "runner_source_sha256": sha256_file(runner_path),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device,
        "deterministic_settings": json.dumps(deterministic_settings, sort_keys=True),
        "initial_state_hash": initial_state_hash,
        "first_train_batch_hash": first_train_batch_hash,
        "after_first_optimizer_step_hash": after_first_optimizer_step_hash,
        "best_state_hash": hash_state_dict(best_state),
        "prediction_hash": prediction_hash,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "best_val_score": best_val_score,
        "test_metric": test_metric,
    }


def write_preflight(config_path: Path, config: dict[str, Any], output_dir: Path, device: str) -> dict[str, Any]:
    deterministic_settings = {
        "required_CUBLAS_WORKSPACE_CONFIG": config["deterministic_mode"]["requires_cublas_workspace_config_before_python"],
        "actual_CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "torch_use_deterministic_algorithms": "enabled during smoke/repeats",
        "warn_only": False,
        "num_workers": 0,
    }
    audit = env_audit(config_path, Path(__file__).resolve(), device, deterministic_settings)
    write_json(output_dir / "environment_and_seed_audit.json", audit)
    schema = {
        "passed": True,
        "checks": {
            "subject_is_p06": config["subject_id"] == "P06",
            "dataset_is_weissbart_tf64": config["dataset"]["dataset_id"] == "weissbart_tf64",
            "model_is_vlaai": config["model"] == "vlaai",
            "seed_is_0": int(config["seed"]) == 0,
            "independent_output_dir": config["output_dir"] != "experiments/gate0_gate2_subject_specific_local_models_weissbart_full_v1",
            "no_checkpoint_written": True,
            "no_prediction_dump_written": True,
            "no_raw_data_written": True,
        },
        "manual_repeat_policy": "not run by preflight; user must run with CUBLAS_WORKSPACE_CONFIG set before Python starts",
        "forbidden_outputs": config["forbidden_outputs"],
    }
    schema["passed"] = all(schema["checks"].values())
    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(output_dir / "failure_report.json", {"failures": [] if schema["passed"] else [{"failure_reason": "schema check failed"}]})
    write_json(
        output_dir / "run_state.json",
        {
            "status": "preflight_only",
            "formal_repeats_completed": 0,
            "single_batch_smoke_completed": False,
            "checkpoint_written": False,
            "prediction_dump_written": False,
            "raw_data_written": False,
        },
    )
    return schema


def run_single_batch_smoke(config_path: Path, config: dict[str, Any], output_dir: Path, device: str) -> int:
    output_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    try:
        row = train_one_run(config_path, config, device, run_index=0, max_batches=1)
        smoke = {
            "passed": True,
            "device": device,
            "initial_state_hash": row["initial_state_hash"],
            "first_train_batch_hash": row["first_train_batch_hash"],
            "after_first_optimizer_step_hash": row["after_first_optimizer_step_hash"],
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "checkpoint_written": False,
            "prediction_dump_written": False,
            "raw_data_written": False,
        }
        write_json(output_dir / "single_batch_smoke_report.json", smoke)
        state = load_json(output_dir / "run_state.json")
        state["single_batch_smoke_completed"] = True
        write_json(output_dir / "run_state.json", state)
        print_event("SMOKE", "passed=true forward_backward_optimizer_step=true")
        return 0
    except Exception as exc:
        write_json(output_dir / "failure_report.json", {"failures": [{"failure_reason": repr(exc)}]})
        print_event("SMOKE FAILED", repr(exc))
        return 1


def run_repeats(config_path: Path, config: dict[str, Any], output_dir: Path, device: str, repeat_count: int) -> int:
    output_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run_index in range(1, repeat_count + 1):
        print_event("JOB START", f"weissbart_tf64:P06:vlaai:seed0 repeat={run_index}")
        try:
            row = train_one_run(config_path, config, device, run_index=run_index)
            rows.append(row)
            write_csv(output_dir / "reproducibility_runs.csv", rows, RUN_FIELDS)
            print_event("JOB DONE", f"repeat={run_index} test_metric={row['test_metric']}")
        except Exception as exc:
            failures.append({"run_index": run_index, "failure_reason": repr(exc)})
            print_event("JOB FAILED", f"repeat={run_index} error={repr(exc)}")
            break
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_json(
        output_dir / "run_state.json",
        {
            "status": "completed" if len(rows) == repeat_count and not failures else "failed",
            "formal_repeats_completed": len(rows),
            "single_batch_smoke_completed": (output_dir / "single_batch_smoke_report.json").exists(),
            "checkpoint_written": False,
            "prediction_dump_written": False,
            "raw_data_written": False,
        },
    )
    schema = load_json(output_dir / "schema_validation_report.json")
    schema["manual_repeats"] = {"completed": len(rows), "expected": repeat_count, "failure_count": len(failures)}
    schema["passed"] = bool(schema["passed"] and len(rows) == repeat_count and not failures)
    write_json(output_dir / "schema_validation_report.json", schema)
    return 0 if not failures and len(rows) == repeat_count else 1


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.startup_only:
        print_event("STARTUP", f"config={args.config}")
        print_event("STARTUP", f"output_dir={repo_relative(output_dir)} device={device}")
        print_event("STARTUP", "job=weissbart_tf64:P06:vlaai:seed0 deterministic=true")
        return 0
    schema = write_preflight(config_path, config, output_dir, device)
    if args.dry_run:
        print_event("DRY RUN", "planned_manual_repeats=2")
        print_event("DRY RUN", "execution=weissbart_tf64:P06:vlaai:seed0 repeat=1")
        print_event("DRY RUN", "execution=weissbart_tf64:P06:vlaai:seed0 repeat=2")
        print_event("DRY RUN", "no_training_executed=true")
    if args.single_batch_smoke:
        return run_single_batch_smoke(config_path, config, output_dir, device)
    if args.run_repeats:
        repeat_count = int(args.repeat_count or config["manual_repeat_count"])
        return run_repeats(config_path, config, output_dir, device, repeat_count)
    print_event("PREFLIGHT", f"passed={schema['passed']} output_dir={repo_relative(output_dir)}")
    return 0 if schema["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
