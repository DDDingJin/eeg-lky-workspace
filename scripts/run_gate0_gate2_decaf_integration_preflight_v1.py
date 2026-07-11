from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.scoring import recording_metric_row
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


def load_decaf_model_class(upstream_path: Path):
    decaf_root = upstream_path.resolve()
    if not decaf_root.exists():
        raise FileNotFoundError(f"DECAF upstream path does not exist: {decaf_root}")
    decaf_src = decaf_root / "src"
    for path in (decaf_root, decaf_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = decaf_src / "models" / "happyquoka.py"
    spec = importlib.util.spec_from_file_location("decaf_upstream_happyquoka", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load DECAF module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HappyQuoka


def run_import_checks(upstream_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    for module_name in ["torch", "numpy", "scipy", "sklearn"]:
        try:
            importlib.import_module(module_name)
            checks.append({"name": module_name, "status": "available", "required": True})
        except Exception as exc:
            checks.append({"name": module_name, "status": "missing", "required": True, "error": repr(exc)})
            missing.append(module_name)
    try:
        importlib.import_module("wandb")
        checks.append({"name": "wandb", "status": "available", "required": "training_only"})
    except Exception as exc:
        checks.append({"name": "wandb", "status": "missing", "required": "training_only", "error": repr(exc)})
        missing.append("wandb(training_only)")
    try:
        load_decaf_model_class(upstream_path)
        checks.append({"name": "DECAF HappyQuoka import", "status": "available", "required": True})
    except Exception as exc:
        checks.append({"name": "DECAF HappyQuoka import", "status": "missing", "required": True, "error": repr(exc)})
        missing.append("DECAF HappyQuoka import")
    return checks, missing


def split_isolation(dataset_dir: Path, subject_id: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    split_recordings: dict[str, list[str]] = {}
    for split in ["train", "val", "test"]:
        rows = load_reference_recordings(dataset_dir, split, subject_id, channels=range(64))
        split_recordings[split] = [str(item[0]) for item in rows]
    checks = [
        {
            "check": "P00 train/val/test subject isolation",
            "passed": all(all(subject_id in recording_id for recording_id in ids) for ids in split_recordings.values()),
            "detail": {split: len(ids) for split, ids in split_recordings.items()},
        },
        {
            "check": "P00 train/val/test recording disjointness",
            "passed": (
                set(split_recordings["train"]).isdisjoint(split_recordings["val"])
                and set(split_recordings["train"]).isdisjoint(split_recordings["test"])
                and set(split_recordings["val"]).isdisjoint(split_recordings["test"])
            ),
            "detail": "recording ids are disjoint across splits",
        },
    ]
    return checks, split_recordings


def instantiate_decaf(config: dict[str, Any], device: str):
    decaf_cfg = config["decaf"]
    model_cls = load_decaf_model_class(ROOT / decaf_cfg["upstream_path"])
    model = model_cls(
        in_channel=int(decaf_cfg["in_channel"]),
        d_model=int(decaf_cfg["d_model"]),
        d_inner=int(decaf_cfg["d_inner"]),
        n_head=int(decaf_cfg["n_head"]),
        n_layers=int(decaf_cfg["n_layers"]),
        fft_conv1d_kernel=tuple(decaf_cfg["fft_conv1d_kernel"]),
        fft_conv1d_padding=tuple(decaf_cfg["fft_conv1d_padding"]),
        dropout=float(decaf_cfg["dropout"]),
        g_con=bool(decaf_cfg["g_con"]),
        within_sub_num=int(decaf_cfg["within_sub_num"]),
    ).to(device)
    model.eval()
    return model


def run_shape_audit(config: dict[str, Any], dataset_dir: Path, device: str) -> dict[str, Any]:
    decaf_cfg = config["decaf"]
    subject_id = config["dataset"]["subject_id"]
    sampling_rate = int(config["dataset"]["sampling_rate"])
    shape_samples = int(decaf_cfg["shape_check_samples"])
    recording_id, eeg, env = load_reference_recordings(dataset_dir, "test", subject_id, channels=range(64))[0]
    check_samples = min(shape_samples, int(eeg.shape[0]))
    model = instantiate_decaf(config, device)
    eeg_tensor = torch.from_numpy(eeg[:check_samples].astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(eeg_tensor)
    envelope = raw["envelope"].detach().cpu().numpy()
    post = envelope.squeeze(0).squeeze(-1).astype(np.float32)
    target = env[: post.shape[0]].astype(np.float32)
    scorer_row = recording_metric_row(
        dataset=config["dataset"]["dataset_id"],
        model="decaf_happyquoka",
        task="reconstruction_preflight_shape_only",
        protocol=config["protocol"],
        seed=int(config["seed"]),
        subject_id=subject_id,
        recording_id=str(recording_id),
        sampling_rate=sampling_rate,
        checkpoint_id="untrained_preflight",
        artifact_scope=config["artifact_scope"],
        prediction=post,
        target=target,
        valid_mask=np.ones_like(post, dtype=bool),
    )
    return {
        "recording_id": str(recording_id),
        "input_shape": shape_text(tuple(eeg_tensor.shape)),
        "raw_output_type": "dict",
        "raw_output_keys": sorted(raw.keys()),
        "raw_envelope_shape": shape_text(tuple(envelope.shape)),
        "postprocessed_prediction_shape": shape_text(tuple(post.shape)),
        "target_shape": shape_text(tuple(target.shape)),
        "scorer_input_shape": shape_text(tuple(post.shape)),
        "scorer_smoke_metric_name": scorer_row.metric_name,
        "native_contract": "HappyQuoka: EEG tensor (batch,time,channels) -> dict with envelope (batch,time,1)",
        "adapter_note": "Shape/scorer smoke uses an untrained model only; no scientific metric is reported.",
    }


def coverage_audit(config: dict[str, Any], dataset_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subject_id = config["dataset"]["subject_id"]
    for split in ["train", "val", "test"]:
        for recording_id, eeg, env in load_reference_recordings(dataset_dir, split, subject_id, channels=range(64)):
            valid_samples = int(min(eeg.shape[0], env.shape[0]))
            rows.append(
                {
                    "split": split,
                    "subject_id": subject_id,
                    "recording_id": str(recording_id),
                    "recording_length": int(len(env)),
                    "valid_samples": valid_samples,
                    "coverage_ratio": float(valid_samples / len(env)) if len(env) else 0.0,
                    "coverage_policy": "time-preserving native DECAF HappyQuoka output; no 50-sample cap",
                }
            )
    return rows


def write_markdown_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    import_checks: list[dict[str, Any]],
    missing: list[str],
    isolation_checks: list[dict[str, Any]],
    split_recordings: dict[str, list[str]],
    shape: dict[str, Any] | None,
    coverage: list[dict[str, Any]],
    schema: dict[str, Any],
) -> None:
    decaf_cfg = config["decaf"]
    (output_dir / "decaf_provenance.md").write_text(
        "\n".join(
            [
                "# DECAF Provenance",
                "",
                f"- source URL: `{decaf_cfg['source_url']}`",
                f"- pinned commit: `{decaf_cfg['commit']}`",
                f"- license: `{decaf_cfg['license']}`",
                f"- upstream path: `{decaf_cfg['upstream_path']}`",
                "- upstream source, checkpoints, HuggingFace data, and model weights are not committed to this repository.",
                f"- status: `{decaf_cfg['local_adaptation_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    shape_lines = [
        "# DECAF Interface Audit",
        "",
        f"- model family: `{decaf_cfg['model_family']}`",
        f"- implementation: `{decaf_cfg['implementation_file']}`",
        "- native HappyQuoka input: `(batch, time, channels)` EEG tensor.",
        "- native HappyQuoka output: dict with `envelope` shaped `(batch, time, 1)` and `dec_output` shaped `(batch, time, d_model)`.",
        "- two-branch DECAF variants additionally require envelope context and fusion logic; this preflight does not claim reference-protocol parity for those variants.",
        "- adapter stance: align output/target/scorer after auditing native context/input/output; do not force existing 50-sample windows.",
        f"- status: `{decaf_cfg['local_adaptation_status']}`",
        "",
    ]
    if shape:
        shape_lines.extend([f"- {key}: `{value}`" for key, value in shape.items()])
    (output_dir / "decaf_interface_audit.md").write_text("\n".join(shape_lines) + "\n", encoding="utf-8")
    adapter_lines = ["# Adapter Shape Audit", ""]
    if shape:
        adapter_lines.extend(
            [
                f"- input_shape: `{shape['input_shape']}`",
                f"- raw_output_keys: `{','.join(shape['raw_output_keys'])}`",
                f"- raw_envelope_shape: `{shape['raw_envelope_shape']}`",
                f"- postprocessed_prediction_shape: `{shape['postprocessed_prediction_shape']}`",
                f"- target_shape: `{shape['target_shape']}`",
                f"- scorer_input_shape: `{shape['scorer_input_shape']}`",
            ]
        )
    else:
        adapter_lines.append("- shape check not run")
    (output_dir / "adapter_shape_audit.md").write_text("\n".join(adapter_lines) + "\n", encoding="utf-8")
    report_lines = [
        "# DECAF Integration Preflight Report",
        "",
        f"- protocol: `{config['protocol']}`",
        f"- dataset: `{config['dataset']['dataset_id']}`",
        f"- subject: `{config['dataset']['subject_id']}`",
        f"- sampling_rate: `{config['dataset']['sampling_rate']}`",
        f"- status: `{decaf_cfg['local_adaptation_status']}`",
        f"- schema_passed: `{schema['passed']}`",
        f"- dependency_missing: `{','.join(missing) if missing else 'none'}`",
        "",
        "## Split Isolation",
    ]
    for check in isolation_checks:
        report_lines.append(f"- {check['check']}: `{check['passed']}` ({check['detail']})")
    report_lines.extend(["", "## Coverage"])
    for split in ["train", "val", "test"]:
        split_rows = [row for row in coverage if row["split"] == split]
        if split_rows:
            report_lines.append(
                f"- {split}: recordings=`{len(split_rows)}`, coverage_min=`{min(row['coverage_ratio'] for row in split_rows):.6f}`, coverage_max=`{max(row['coverage_ratio'] for row in split_rows):.6f}`"
            )
    report_lines.extend(["", "## Import Checks"])
    for check in import_checks:
        report_lines.append(f"- {check['name']}: `{check['status']}` required=`{check['required']}`")
    report_lines.extend(["", "## Next Step"])
    report_lines.append("- If schema remains passed and required imports are available, this can enter P00 formal smoke; do not treat this preflight as a scientific result.")
    (output_dir / "preflight_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = ROOT / config["output_dir"]
    ensure_dir(output_dir)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dataset_dir = resolve_dataset_path(config["dataset"]["dataset_locator"])
    upstream_path = ROOT / config["decaf"]["upstream_path"]

    print_event("STARTUP", f"config={args.config}")
    print_event("STARTUP", f"output_dir={repo_relative(output_dir)} dataset={config['dataset']['dataset_id']} subject={config['dataset']['subject_id']} device={device}")
    print_event("STARTUP", f"decaf_upstream={repo_relative(upstream_path)} commit={config['decaf']['commit']}")

    import_checks, missing = run_import_checks(upstream_path)
    isolation_checks, split_recordings = split_isolation(dataset_dir, config["dataset"]["subject_id"])
    shape = None
    if args.shape_check:
        shape = run_shape_audit(config, dataset_dir, device)
    coverage = coverage_audit(config, dataset_dir)
    required_imports_ok = all(check["status"] == "available" for check in import_checks if check["required"] is True)
    schema_checks = {
        "required_imports_available": required_imports_ok,
        "p00_split_isolation_passed": all(item["passed"] for item in isolation_checks),
        "shape_check_present": bool(shape),
        "coverage_full_recording_policy_defined": all(row["coverage_ratio"] == 1.0 for row in coverage),
        "no_training_metrics_reported": True,
        "local_adaptation_not_reference_parity": config["decaf"]["local_adaptation_status"] == "DECAF local adaptation / not yet reference-protocol parity",
    }
    schema = {
        "passed": all(schema_checks.values()),
        "checks": schema_checks,
        "dependency_missing": missing,
        "subject_id": config["dataset"]["subject_id"],
        "coverage_rows": len(coverage),
        "split_recording_counts": {split: len(ids) for split, ids in split_recordings.items()},
    }
    failures = []
    if not schema["passed"]:
        failures.append({"status": "preflight_failed", "failure_reason": "one or more schema checks failed", "checks": schema_checks})

    if args.dry_run:
        print_event("DRY RUN", json.dumps(schema, sort_keys=True))
        return 0 if schema["passed"] else 1

    write_json(output_dir / "schema_validation_report.json", schema)
    write_json(output_dir / "failure_report.json", {"failures": failures})
    write_markdown_artifacts(output_dir, config, import_checks, missing, isolation_checks, split_recordings, shape, coverage, schema)
    print_event("PREFLIGHT", f"schema_passed={schema['passed']} missing={','.join(missing) if missing else 'none'}")
    return 0 if schema["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
