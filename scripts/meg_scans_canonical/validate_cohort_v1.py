"""Validate local-only canonical cohort MAT artifacts without training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matlab_string(f: h5py.File, ref) -> str:
    return "".join(chr(int(x)) for x in np.asarray(f[ref]).reshape(-1))


def validate_paired(path: Path) -> dict:
    result = {"paired_mat_sha256": sha256(path), "paired_fs_hz": None, "trial_count": 0, "trial_shapes": [], "nan_inf": False, "channel_order_sha256": "", "reason": ""}
    try:
        with h5py.File(path, "r") as f:
            root = f["results"]
            result["paired_fs_hz"] = int(np.asarray(root["epochs_neuro"]["fsample"]).squeeze())
            labels = [matlab_string(f, ref) for ref in np.asarray(root["epochs_neuro"]["label"]).reshape(-1)]
            result["channel_order_sha256"] = hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()
            trials = np.asarray(root["epochs_neuro"]["trial"]).reshape(-1)
            result["trial_count"] = len(trials)
            for ref in trials:
                arr = np.asarray(f[ref])
                shape = list(arr.shape)
                if shape == [306, 7680]:
                    shape = [7680, 306]
                result["trial_shapes"].append(shape)
                result["nan_inf"] = result["nan_inf"] or bool(np.isnan(arr).any() or np.isinf(arr).any())
        if result["paired_fs_hz"] != 64:
            result["reason"] = "paired fs is not 64 Hz"
        elif result["trial_count"] != 16:
            result["reason"] = "paired trial count is not 16"
        elif any(shape != [7680, 306] for shape in result["trial_shapes"]):
            result["reason"] = "paired trial shape is not [7680, 306]"
        elif result["nan_inf"]:
            result["reason"] = "paired data contains NaN or Inf"
    except Exception as exc:
        result["reason"] = f"paired MAT invalid: {type(exc).__name__}: {exc}"
    result["paired_passed"] = not result["reason"]
    return result


def validate_envelope(path: Path) -> dict:
    result = {"envelope_mat_sha256": sha256(path), "envelope_trial_count": 0, "envelope_shapes": [], "nan_inf": False, "reason": ""}
    try:
        with h5py.File(path, "r") as f:
            epochs = np.asarray(f["audio_envelopes"]["epochs_audio"]).reshape(-1)
            trials = []
            for epoch_ref in epochs:
                trials.extend(np.asarray(f[epoch_ref]["trial"]).reshape(-1))
            result["envelope_trial_count"] = len(trials)
            for ref in trials:
                arr = np.asarray(f[ref])
                result["envelope_shapes"].append(list(arr.shape))
                result["nan_inf"] = result["nan_inf"] or bool(np.isnan(arr).any() or np.isinf(arr).any())
        if result["envelope_trial_count"] not in {16, 20}:
            result["reason"] = "envelope trial count is neither 16 (paired) nor 20 (official source before cleanup)"
        elif any(np.prod(shape) <= 0 for shape in result["envelope_shapes"]):
            result["reason"] = "envelope trial has non-positive length"
        elif result["nan_inf"]:
            result["reason"] = "envelope data contains NaN or Inf"
    except Exception as exc:
        result["reason"] = f"envelope MAT invalid: {type(exc).__name__}: {exc}"
    result["envelope_passed"] = not result["reason"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    envelope = args.derived_root / "stimuli" / "sub-others_preprocessed_audiobook_envelopes_decoding.mat"
    rows = []
    for subject in args.subjects:
        paired = args.derived_root / subject / "speech" / f"{subject}_preprocessed_audiobooks_decoding.mat"
        row = {"subject_id": subject, "canonicalization_passed": False, "invalid_reason": ""}
        if not paired.is_file():
            row["invalid_reason"] = "paired MAT not generated"
        elif not envelope.is_file():
            row["invalid_reason"] = "canonical envelope MAT not generated"
        else:
            row.update(validate_paired(paired))
            row.update(validate_envelope(envelope))
            row["canonicalization_passed"] = bool(row.get("paired_passed", False) and row.get("envelope_passed", False))
            if not row["canonicalization_passed"]:
                row["invalid_reason"] = row.get("reason", "canonical validation failed")
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "canonicalization_validation.json").write_text(json.dumps({"subjects": rows}, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "canonicalization_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["canonicalization_passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
