from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shape-check", action="store_true")
    return parser.parse_args()


def shape_text(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(item) for item in shape) + "]"


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
    model.eval()
    return model


def build_causality_rows(config: dict[str, Any], dataset_dir: Path) -> list[dict[str, Any]]:
    subject_id = config["dataset"]["subject_id"]
    sr = int(config["dataset"]["sampling_rate"])
    forecaster_context_samples = int(float(config["decaf"]["context_seconds"]) * sr)
    input_context_samples = int(float(config["decaf"]["forward_input_context_seconds"]) * sr)
    prediction_samples = int(float(config["decaf"]["prediction_seconds"]) * sr)
    trim = int(config["decaf"]["forward_context_trim_samples"])
    effective_context_samples = input_context_samples - trim
    rows: list[dict[str, Any]] = []
    for split in ["test"]:
        for recording_id, eeg, env in load_reference_recordings(dataset_dir, split, subject_id, channels=range(64)):
            recording_length = int(min(eeg.shape[0], env.shape[0]))
            for target_start in range(input_context_samples, recording_length - prediction_samples + 1, prediction_samples):
                target_end = target_start + prediction_samples
                context_start = target_start - input_context_samples
                context_end = target_start
                rows.append(
                    {
                        "split": split,
                        "subject_id": subject_id,
                        "recording_id": str(recording_id),
                        "sample_rate": sr,
                        "target_start": target_start,
                        "target_end_exclusive": target_end,
                        "eeg_start": target_start,
                        "eeg_end_exclusive": target_end,
                        "envelope_context_start": context_start,
                        "envelope_context_end_exclusive": context_end,
                        "envelope_context_end_inclusive": context_end - 1,
                        "forward_effective_context_start": context_start + trim,
                        "forward_effective_context_end_exclusive": context_end,
                        "forward_effective_context_end_inclusive": context_end - 1,
                        "context_end_inclusive_lt_target_start": (context_end - 1) < target_start,
                        "exclusive_context_end_eq_target_start": context_end == target_start,
                        "effective_context_end_le_target_start": context_end <= target_start,
                        "target_length": prediction_samples,
                        "forward_input_context_samples": input_context_samples,
                        "forecaster_expected_context_samples": forecaster_context_samples,
                        "effective_context_samples_after_trim": effective_context_samples,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_shape_check(config: dict[str, Any], device: str) -> dict[str, Any]:
    decaf = config["decaf"]
    sr = int(config["dataset"]["sampling_rate"])
    forecaster_context_samples = int(float(decaf["context_seconds"]) * sr)
    input_context_samples = int(float(decaf["forward_input_context_seconds"]) * sr)
    prediction_samples = int(float(decaf["prediction_seconds"]) * sr)
    model = instantiate_twobranch(config, device)
    eeg = torch.zeros(1, prediction_samples, int(decaf["in_channel"]), device=device)
    envelope_context = torch.zeros(1, input_context_samples, 1, device=device)
    with torch.no_grad():
        output = model(eeg, envelope_context)
    return {
        "shape_check_context": "synthetic_zero_context_only_no_target_envelope_used",
        "eeg_input_shape": shape_text(tuple(eeg.shape)),
        "envelope_context_input_shape": shape_text(tuple(envelope_context.shape)),
        "forward_context_trim_samples": int(decaf["forward_context_trim_samples"]),
        "forecaster_expected_context_shape": shape_text((1, forecaster_context_samples, 1)),
        "effective_envelope_context_shape_after_forward_trim": shape_text((1, input_context_samples - int(decaf["forward_context_trim_samples"]), 1)),
        "output_keys": sorted(output.keys()),
        "envelope_shape": shape_text(tuple(output["envelope"].shape)),
        "envelope_branch_shape": shape_text(tuple(output["envelope_branch"].shape)),
        "eeg_branch_shape": shape_text(tuple(output["eeg_branch"].shape)),
        "fusion_weights_shape": shape_text(tuple(output["fusion_weights"].shape)),
        "scientific_metric_computed": False,
    }


def write_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    shape: dict[str, Any] | None,
    schema: dict[str, Any],
) -> None:
    decaf = config["decaf"]
    interface_lines = [
        "# DECAF TwoBranch Interface Audit",
        "",
        f"- source: `{decaf['implementation_file']}`",
        f"- pinned upstream commit: `{decaf['commit']}`",
        f"- status: `{decaf['local_adaptation_status']}`",
        "- true forward signature: `forward(eeg, envelope_context)`.",
        f"- EEG input interval: target/prediction interval, length `{int(float(decaf['prediction_seconds']) * int(config['dataset']['sampling_rate']))}` samples.",
        f"- envelope_context input interval: immediately preceding context interval, forward input length `{int(float(decaf['forward_input_context_seconds']) * int(config['dataset']['sampling_rate']))}` samples.",
        f"- envelope forecaster expected context after trim: `{int(float(decaf['context_seconds']) * int(config['dataset']['sampling_rate']))}` samples.",
        f"- forward trims first `{decaf['forward_context_trim_samples']}` context samples via `envelope_context = envelope_context[:,32:,:]` before forecasting.",
        "- prediction output interval: same duration as EEG input / target interval.",
        "- target interval: same sample interval as model output; used only for evaluation, not for synthetic shape-check context.",
        "- fusion strategy: combines envelope-branch and EEG-branch predictions; `weighted` uses softmax-normalized learned two-element weights.",
        "- upstream trainer/data path indicates context models can receive ground-truth past envelope context; this changes task class relative to pure EEG reconstruction.",
        "",
    ]
    if shape:
        interface_lines.extend([f"- {key}: `{value}`" for key, value in shape.items()])
    (output_dir / "decaf_twobranch_interface_audit.md").write_text("\n".join(interface_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# DECAF Task Compatibility Decision",
        "",
        "- pure_reconstruction_compatible: `false`",
        "- task_classification: `context_assisted_or_forecasting`",
        "- reason: `TwoBranchModel.forward` requires an `envelope_context` input for the envelope branch. If that context is the true past target envelope at test time, the condition is context-assisted forecasting, not pure EEG-only envelope reconstruction.",
        "- reporting rule: do not mix TwoBranchModel results into the pure EEG envelope reconstruction main table.",
        "- allowed next step: reviewer may approve a separate context-assisted/forecasting condition with explicit context availability assumptions.",
        "",
    ]
    (output_dir / "decaf_task_compatibility_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    shape_lines = ["# Adapter Shape Audit", ""]
    if shape:
        shape_lines.extend([f"- {key}: `{value}`" for key, value in shape.items()])
    else:
        shape_lines.append("- shape_check: `not_run`")
    (output_dir / "adapter_shape_audit.md").write_text("\n".join(shape_lines) + "\n", encoding="utf-8")
    write_csv(output_dir / "decaf_causality_table.csv", rows)
    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(output_dir / "failure_report.json", {"failures": [] if schema["passed"] else [{"failure_reason": "schema failed", "checks": schema["checks"]}]})


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    print_event("STARTUP", f"config={args.config}")
    print_event("STARTUP", f"output_dir={repo_relative(output_dir)} device={device}")
    rows = build_causality_rows(config, dataset_dir)
    shape = run_shape_check(config, device) if args.shape_check else None
    schema_checks = {
        "twobranch_forward_contract_documented": True,
        "causality_table_has_rows": len(rows) > 0,
        "context_end_inclusive_lt_target_start": all(bool(row["context_end_inclusive_lt_target_start"]) for row in rows),
        "exclusive_context_end_equals_target_start_documented": all(bool(row["exclusive_context_end_eq_target_start"]) for row in rows),
        "pure_reconstruction_compatible_false": True,
        "task_classification_context_assisted_or_forecasting": True,
        "shape_check_synthetic_zero_context_no_metric": bool(shape and shape["scientific_metric_computed"] is False),
    }
    schema = {
        "passed": all(schema_checks.values()),
        "checks": schema_checks,
        "causality_rows": len(rows),
        "pure_reconstruction_compatible": False,
        "task_classification": "context_assisted_or_forecasting",
        "shape_check": shape,
    }
    if args.dry_run:
        print_event("DRY RUN", json.dumps(schema, sort_keys=True))
        return 0 if schema["passed"] else 1
    write_artifacts(output_dir, config, rows, shape, schema)
    print_event("AUDIT", f"schema_passed={schema['passed']} pure_reconstruction_compatible=false")
    return 0 if schema["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
