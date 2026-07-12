from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "meg_scans_canonical_model_readiness_v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    required = [
        OUT / "run_config.json",
        OUT / "data_integrity_check.json",
        OUT / "split_manifest.json",
        OUT / "loader_contract_report.md",
        OUT / "run_manifest.json",
        OUT / "model_run_entries.json",
        OUT / "subject_metrics.csv",
        OUT / "recording_metrics.csv",
        OUT / "sorted_vs_mismatched.csv",
        OUT / "result_summary.md",
        OUT / "failure_report.json",
        OUT / "schema_validation_report.json",
        OUT / "incremental_comparison.md",
        OUT / "ridge_mag102.log",
        OUT / "ridge_all306.log",
        OUT / "cca_mag102.log",
        OUT / "eegnet_mag102_smoke.log",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    integrity = read_json(OUT / "data_integrity_check.json")
    if integrity["status"] != "passed":
        raise SystemExit("data integrity did not pass")
    if integrity["paired_sha256"] != "d7c189af6ba44031ce67745f1511162259af0a1ef07ea19eb8618b33367bf91e":
        raise SystemExit("paired SHA mismatch")
    if integrity["envelope_sha256"] != "0aed00fa9a9990d45378627911255d98cd09afc70cdff07372837d296e2b269d":
        raise SystemExit("envelope SHA mismatch")
    if integrity["fs"] != 64 or integrity["n_trials"] != 16:
        raise SystemExit("bad fs or trial count")
    if integrity["all306_channel_count"] != 306 or integrity["mag102_channel_count"] != 102:
        raise SystemExit("bad channel counts")

    split = read_json(OUT / "split_manifest.json")
    expected = {
        "train": {1, 2, 5, 6, 9, 10, 13, 14},
        "val": {3, 7, 11, 15},
        "test": {4, 8, 12, 16},
    }
    for role, indices in expected.items():
        actual = {int(row["global_trial_index"]) for row in split["rows"] if row["role"] == role}
        if actual != indices:
            raise SystemExit(f"bad {role} split: {actual}")
    if len({int(row["global_trial_index"]) for row in split["rows"]}) != 16:
        raise SystemExit("split overlap or missing trial")

    entries = read_json(OUT / "model_run_entries.json")
    by_model = {row["model"]: row for row in entries}
    expected_models = {"ridge_mag102", "ridge_all306", "cca_mag102", "eegnet_mag102_smoke"}
    if set(by_model) != expected_models:
        raise SystemExit(f"bad model set: {set(by_model)}")
    for model in ["ridge_mag102", "ridge_all306"]:
        if by_model[model]["status"] != "actual":
            raise SystemExit(f"{model} is not actual")
        metric = float(by_model[model]["metric"])
        if metric < -1 or metric > 1:
            raise SystemExit(f"{model} metric out of range")
    if by_model["eegnet_mag102_smoke"]["status"] != "smoke_only":
        raise SystemExit("EEGNet is not smoke_only")
    if by_model["cca_mag102"]["status"] == "skipped" and not by_model["cca_mag102"].get("failure_reason"):
        raise SystemExit("skipped CCA lacks reason")

    recording_rows = read_csv(OUT / "recording_metrics.csv")
    subject_rows = read_csv(OUT / "subject_metrics.csv")
    if not recording_rows or not subject_rows:
        raise SystemExit("missing metric rows")
    for row in recording_rows:
        value = float(row["metric_value"])
        if value < -1.000001 or value > 1.000001:
            raise SystemExit(f"recording metric out of range: {row}")

    sorted_rows = read_csv(OUT / "sorted_vs_mismatched.csv")
    if len(sorted_rows) != 8:
        raise SystemExit(f"expected 8 sorted/mismatched rows for two Ridge models, got {len(sorted_rows)}")
    mapping = {4: 8, 8: 12, 12: 16, 16: 4}
    for row in sorted_rows:
        source = int(row["global_trial_index"])
        target = int(row["mismatched_global_trial_index"])
        if mapping[source] != target or source == target:
            raise SystemExit(f"bad mismatched mapping: {row}")

    schema = read_json(OUT / "schema_validation_report.json")
    if not schema["passed"]:
        raise SystemExit("schema validation report failed")

    banned_ext = {".mat", ".fif", ".gz", ".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".pkl"}
    allowed_prefixes = (
        "configs/benchmark/meg_scans_canonical_model_readiness/",
        "scripts/meg_scans_canonical_model_readiness/",
        "experiments/meg_scans_canonical_model_readiness_v1/",
        "docs/meg_scans_canonical_model_readiness_sub03_zh.md",
        "workflow/meg_scans_canonical_model_readiness_START_HERE.md",
        "scripts/meg_scans_canonical/",
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
    print("model readiness compact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
