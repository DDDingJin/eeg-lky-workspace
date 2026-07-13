from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BANNED_SUFFIXES = {".mat", ".fif", ".pt", ".pth", ".npy", ".npz", ".h5", ".ckpt"}
EXPECTED_MODELS = ["linear", "ridge", "lasso", "elasticnet", "cca", "fcnn", "cnn", "eegnet", "adt", "vlaai", "happyquokka"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    output_dir = ROOT / "experiments" / "meg_scans_subject_specific_11models_v1"
    errors: list[str] = []
    required = [
        "run_config.json",
        "meg_model_set_lock.json",
        "data_integrity_check.json",
        "split_manifest.json",
        "shape_preflight.json",
        "run_manifest.json",
    ]
    for name in required:
        if not (output_dir / name).exists():
            errors.append(f"missing {name}")

    if not errors:
        lock = read_json(output_dir / "meg_model_set_lock.json")
        if lock.get("models") != EXPECTED_MODELS or lock.get("model_count") != 11:
            errors.append("meg_model_set_lock does not contain exactly the expected 11-model roster")
        if "dnn" not in lock.get("excluded_models", []):
            errors.append("meg_model_set_lock does not record dnn exclusion")

        integrity = read_json(output_dir / "data_integrity_check.json")
        expected_integrity = {
            "status": "passed",
            "n_trials": 16,
            "fs": 64,
            "representation": "mag102",
            "channel_count": 102,
            "paired_sha256": "d7c189af6ba44031ce67745f1511162259af0a1ef07ea19eb8618b33367bf91e",
            "envelope_sha256": "0aed00fa9a9990d45378627911255d98cd09afc70cdff07372837d296e2b269d",
        }
        for key, expected in expected_integrity.items():
            if integrity.get(key) != expected:
                errors.append(f"data_integrity_check {key}={integrity.get(key)!r}, expected {expected!r}")

        split = read_json(output_dir / "split_manifest.json")
        rows = split.get("rows", [])
        roles = {role: [r for r in rows if r.get("role") == role] for role in ["train", "val", "test"]}
        if len(roles["train"]) != 12 or len(roles["val"]) != 2 or len(roles["test"]) != 2:
            errors.append("split_manifest does not have 12/2/2 recording-level split")
        indices = [int(r["global_trial_index"]) for r in rows]
        if sorted(indices) != list(range(1, 17)) or len(indices) != len(set(indices)):
            errors.append("split_manifest does not uniquely cover trials 1..16")

        preflight = read_json(output_dir / "shape_preflight.json")
        preflight_rows = preflight.get("rows", [])
        if len(preflight_rows) != 11:
            errors.append("shape_preflight does not contain 11 rows")
        seen_models = [r.get("model") for r in preflight_rows]
        if seen_models != EXPECTED_MODELS:
            errors.append("shape_preflight model order does not match lock")
        for model in ["fcnn", "cnn", "eegnet", "adt"]:
            row = next((r for r in preflight_rows if r.get("model") == model), {})
            if row.get("status") != "ok":
                errors.append(f"{model} shape preflight is not ok: {row.get('failure_reason')}")

    tracked = []
    import subprocess

    proc = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        errors.append(f"git ls-files failed: {proc.stderr.strip()}")
    else:
        tracked = [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
        bad = [str(path) for path in tracked if path.suffix.lower() in BANNED_SUFFIXES and "experiments/meg_scans_subject_specific_11models_v1" in path.as_posix()]
        if bad:
            errors.append("banned tracked artifact(s): " + ", ".join(bad))

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checked_dir": str(output_dir),
        "expected_models": EXPECTED_MODELS,
        "banned_suffixes": sorted(BANNED_SUFFIXES),
    }
    (output_dir / "schema_validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
