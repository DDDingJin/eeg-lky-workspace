from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "meg_scans_subject_specific_sub03_v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    required = [
        "run_manifest.json",
        "data_integrity_check.json",
        "split_manifest.json",
        "model_run_entries.json",
        "recording_metrics.csv",
        "subject_metrics.csv",
        "sorted_vs_mismatched.csv",
        "training_history.csv",
        "result_summary.md",
        "failure_report.json",
        "schema_validation_report.json",
        "recording_safe_ridge_adapter_mag102.log",
        "recording_safe_ridge_adapter_all306.log",
        "eegnet_mag102.log",
        "lambda_selection.csv",
        "run_config.json",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    integrity = read_json(OUT / "data_integrity_check.json")
    if integrity["status"] != "passed":
        raise SystemExit("integrity check failed")
    if integrity["paired_sha256"] != "d7c189af6ba44031ce67745f1511162259af0a1ef07ea19eb8618b33367bf91e":
        raise SystemExit("paired SHA mismatch")
    if integrity["envelope_sha256"] != "0aed00fa9a9990d45378627911255d98cd09afc70cdff07372837d296e2b269d":
        raise SystemExit("envelope SHA mismatch")
    if integrity["n_trials"] != 16 or integrity["all306_channel_count"] != 306 or integrity["mag102_channel_count"] != 102:
        raise SystemExit("bad integrity counts")

    split = read_json(OUT / "split_manifest.json")
    expected = {
        "train": {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15},
        "val": {4, 12},
        "test": {8, 16},
    }
    for role, indices in expected.items():
        actual = {int(row["global_trial_index"]) for row in split["rows"] if row["role"] == role}
        if actual != indices:
            raise SystemExit(f"bad split for {role}: {actual}")

    entries = read_json(OUT / "model_run_entries.json")
    by_model = {row["model"]: row for row in entries}
    expected_models = {"recording_safe_ridge_adapter_mag102", "recording_safe_ridge_adapter_all306", "eegnet_mag102"}
    if set(by_model) != expected_models:
        raise SystemExit(f"bad model set: {set(by_model)}")
    for model in expected_models:
        if by_model[model]["status"] != "actual":
            raise SystemExit(f"{model} did not complete as actual")
        metric = float(by_model[model]["metric"])
        if metric < -1.000001 or metric > 1.000001:
            raise SystemExit(f"{model} metric out of range")
    eegnet = by_model["eegnet_mag102"]
    if eegnet.get("device") != "cuda" or not eegnet.get("gpu_name") or int(eegnet.get("epochs_completed", 0)) < 1:
        raise SystemExit("bad EEGNet CUDA metadata")

    recording_rows = read_csv(OUT / "recording_metrics.csv")
    subject_rows = read_csv(OUT / "subject_metrics.csv")
    if len(recording_rows) != 6:
        raise SystemExit(f"expected 6 recording metrics, got {len(recording_rows)}")
    if len(subject_rows) != 3:
        raise SystemExit(f"expected 3 subject metrics, got {len(subject_rows)}")
    for row in recording_rows + subject_rows:
        value = float(row["metric_value"])
        if value < -1.000001 or value > 1.000001:
            raise SystemExit(f"metric out of range: {row}")

    sorted_rows = read_csv(OUT / "sorted_vs_mismatched.csv")
    if len(sorted_rows) != 4:
        raise SystemExit(f"expected 4 sorted/mismatched rows, got {len(sorted_rows)}")
    mapping = {8: 16, 16: 8}
    for row in sorted_rows:
        source = int(row["global_trial_index"])
        target = int(row["mismatched_global_trial_index"])
        if mapping[source] != target or source == target:
            raise SystemExit(f"bad mismatch row: {row}")

    history = read_csv(OUT / "training_history.csv")
    if not history:
        raise SystemExit("empty EEGNet training history")
    schema = read_json(OUT / "schema_validation_report.json")
    if not schema["passed"]:
        raise SystemExit("schema report failed")

    banned_ext = {".mat", ".fif", ".gz", ".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".pkl"}
    allowed_prefixes = (
        "configs/benchmark/meg_scans_subject_specific/",
        "scripts/meg_scans_subject_specific/",
        "experiments/meg_scans_subject_specific_sub03_v1/",
        "docs/meg_scans_subject_specific_sub03_zh.md",
        "workflow/meg_scans_subject_specific_START_HERE.md",
    )
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(allowed_prefixes):
            if path.suffix.lower() in banned_ext:
                raise SystemExit(f"banned artifact: {rel}")
            if path.stat().st_size > 5_000_000:
                raise SystemExit(f"unexpected large artifact: {rel}")
    print("subject-specific sub03 compact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
