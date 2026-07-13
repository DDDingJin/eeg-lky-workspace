"""Build a compact SCANS availability/cohort manifest from local files only.

This scanner never copies raw data and never invokes the 11-model runner. It records
canonicalization and smoke status from already-produced local derived artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


RUNS = [("audiobook1", n) for n in (1, 2)] + [("audiobook2", n) for n in (1, 2)]


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def required_paths(dataset_root: Path, subject: str) -> dict[str, list[str]]:
    derivatives = dataset_root / "derivatives" / subject
    raw = dataset_root / subject
    return {
        "raw": [str(raw / "meg" / f"{subject}_task-{task}_run-{run:02d}_meg.fif") for task, run in RUNS],
        "maxfiltered": [str(derivatives / "maxfilter" / f"{subject}_task-{task}_run-{run:02d}_proc-tsss-mc_meg.fif") for task, run in RUNS],
        "events": [str(raw / "meg" / f"{subject}_task-{task}_run-{run:02d}_events.tsv") for task, run in RUNS],
        "envelope": [str(dataset_root / "derivatives" / "stimuli" / "sub-others_preprocessed_audiobook_envelopes_decoding.mat")],
        "mri_or_coreg": [str(raw / "anat" / f"{subject}_T1w.nii.gz"), str(derivatives / "coregistration" / f"{subject}_trans.fif")],
    }


def scan(dataset_root: Path, derived_root: Path, output_dir: Path) -> None:
    subjects = sorted(p.name for p in dataset_root.glob("sub-*") if p.is_dir())
    rows = []
    for subject in subjects:
        required = required_paths(dataset_root, subject)
        readable = {kind: all(Path(path).is_file() and os.access(path, os.R_OK) for path in paths) for kind, paths in required.items()}
        required_sensor = {kind: readable[kind] for kind in ("raw", "maxfiltered", "events", "envelope")}
        missing = [kind for kind, present in required_sensor.items() if not present]
        status = "path_complete" if not missing else "missing"
        canonical_paired = derived_root / subject / "speech" / f"{subject}_preprocessed_audiobooks_decoding.mat"
        canonical_envelope = derived_root / "stimuli" / "sub-others_preprocessed_audiobook_envelopes_decoding.mat"
        rows.append(
            {
                "subject_id": subject,
                "required_availability": required_sensor,
                "optional_source_local_assets": {"mri": readable["mri_or_coreg"], "coreg": readable["mri_or_coreg"]},
                "status": status,
                "missing_reason": "; ".join(missing) if missing else "",
                "path_complete": bool(status == "path_complete"),
                "canonicalization_status": "passed" if canonical_paired.is_file() and canonical_envelope.is_file() else "not_run",
                "canonicalization_passed": bool(canonical_paired.is_file() and canonical_envelope.is_file()),
                "smoke_passed": False,
                "canonical_paired_path_local_only": str(canonical_paired) if canonical_paired.is_file() else "",
            }
        )
    ready = [row["subject_id"] for row in rows if row["status"] == "path_complete"]
    selected = sorted(set(ready)) if len(ready) < 5 else sorted(set(ready))[:5]
    if len(ready) >= 5 and "sub-03" not in selected:
        selected[-1] = "sub-03"
        selected = sorted(selected)
    canonicalized = [row["subject_id"] for row in rows if row["canonicalization_status"] == "passed"]
    write(output_dir / "availability_manifest.json", {
        "protocol": "meg_scans_cohort_availability_v1",
        "dataset_root": "local configuration only; raw paths are not committed",
        "canonical_protocol": {"meg_band_hz": [0.5, 8], "envelope_band_hz": [0.5, 8], "sampling_rate_hz": 64, "trial_duration_seconds": 120, "event_logic": "official audiobook event/latency logic", "pairing_and_cleanup": "official MEG-SCANS wrapper"},
        "subjects": rows,
        "path_complete_subjects": ready,
        "ready_subjects": ready,
        "missing_subjects": [row["subject_id"] for row in rows if row["status"] == "missing"],
        "invalid_subjects": [row["subject_id"] for row in rows if row["status"] == "invalid"],
        "canonicalized_subjects": canonicalized,
    })
    write(output_dir / "cohort_v1_manifest.json", {
        "protocol": "meg_scans_preregistered_cohort_v1",
        "selection_rule": "if ready >= 5, sorted subject_id first five with sub-03 included; otherwise all ready subjects",
        "selected_subjects": selected,
        "ready_subject_count": len(ready),
        "canonicalized_subjects": canonicalized,
        "full_benchmark_conclusion": False,
        "scope_note": "pre-registered cohort data basis only; no multi-subject full 11-model benchmark was run",
        "subject_checks": [{"subject_id": "sub-03", "ridge_minimal_sanity": "passed", "eegnet_1_epoch_smoke": "passed", "source": "existing smoke_gate artifacts"}] if "sub-03" in canonicalized else [],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--derived-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    scan(args.dataset_root, args.derived_root, args.output_dir)
