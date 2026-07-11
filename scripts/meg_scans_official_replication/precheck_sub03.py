from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("MNE_USE_NUMBA", "false")
import mne


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path("E:/decode/data/raw/meg_scans_ds006468")
OFFICIAL_REPO = Path("E:/decode/external/upstream/MEG-SCANS")
OUT = ROOT / "experiments" / "meg_scans_official_preprocessing_sub03_v1"
SUBJECT = "sub-03"
RUNS = [
    "audiobook1_run-01",
    "audiobook1_run-02",
    "audiobook2_run-01",
    "audiobook2_run-02",
]
BASE_BRANCH = "fix/ar-20260711-meg-scans-representation-preflight-v1"
BASE_COMMIT = "69ed86096951e40728348dc1f7fc9d0b59c8556c"


def run_to_paths(run_id: str) -> dict[str, Path]:
    task, run = run_id.split("_")
    stem = f"{SUBJECT}_task-{task}_{run}"
    return {
        "maxfiltered_meg": DATASET_ROOT / "derivatives" / SUBJECT / "maxfilter" / f"{stem}_proc-tsss-mc_meg.fif",
        "raw_meg": DATASET_ROOT / SUBJECT / "meg" / f"{stem}_meg.fif",
        "events": DATASET_ROOT / SUBJECT / "meg" / f"{stem}_events.tsv",
        "envelope": DATASET_ROOT / "derivatives" / "stimuli" / "sub-others_preprocessed_audiobook_envelopes_decoding.mat",
    }


def official_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={OFFICIAL_REPO.as_posix()}",
                "-C",
                str(OFFICIAL_REPO),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except Exception as exc:
        head = OFFICIAL_REPO / ".git" / "HEAD"
        if head.exists():
            return f"git_rev_parse_blocked; HEAD={head.read_text(encoding='utf-8', errors='replace').strip()}; error={exc}"
        return f"unavailable: {exc}"


def read_channel_counts(path: Path) -> tuple[int, int, int]:
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, misc=False, exclude=[])
    mag = mne.pick_types(raw.info, meg="mag", eeg=False, eog=False, stim=False, misc=False, exclude=[])
    grad = mne.pick_types(raw.info, meg="grad", eeg=False, eog=False, stim=False, misc=False, exclude=[])
    return len(meg), len(mag), len(grad)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for run_id in RUNS:
        paths = run_to_paths(run_id)
        counts = (None, None, None)
        readable = False
        if paths["maxfiltered_meg"].exists():
            counts = read_channel_counts(paths["maxfiltered_meg"])
            readable = True
        row = {
            "subject_id": SUBJECT,
            "run_id": run_id,
            "maxfiltered_meg_path": paths["maxfiltered_meg"].as_posix(),
            "maxfiltered_meg_exists": paths["maxfiltered_meg"].exists(),
            "maxfiltered_meg_readable": readable,
            "meg_channels": counts[0],
            "mag_channels": counts[1],
            "grad_channels": counts[2],
            "raw_event_fif_path": paths["raw_meg"].as_posix(),
            "raw_event_fif_exists": paths["raw_meg"].exists(),
            "events_tsv_path": paths["events"].as_posix(),
            "events_tsv_exists": paths["events"].exists(),
            "envelope_path": paths["envelope"].as_posix(),
            "envelope_exists": paths["envelope"].exists(),
        }
        rows.append(row)
        if not (row["maxfiltered_meg_exists"] and row["maxfiltered_meg_readable"]):
            failures.append(f"{run_id}: maxfiltered MEG missing/unreadable")
        if counts != (306, 102, 204):
            failures.append(f"{run_id}: channel counts {counts} != (306,102,204)")
        if not (row["raw_event_fif_exists"] and row["events_tsv_exists"] and row["envelope_exists"]):
            failures.append(f"{run_id}: event/envelope input missing")

    ancillary = {
        "t1w_mri": DATASET_ROOT / SUBJECT / "anat" / f"{SUBJECT}_T1w.nii.gz",
        "coreg_transform": DATASET_ROOT / "derivatives" / SUBJECT / "coregistration" / f"{SUBJECT}_trans.fif",
        "empty_room_noise_01": DATASET_ROOT / "derivatives" / SUBJECT / "maxfilter" / f"{SUBJECT}_task-noise_run-01_proc-sss_meg.fif",
        "empty_room_noise_02": DATASET_ROOT / "derivatives" / SUBJECT / "maxfilter" / f"{SUBJECT}_task-noise_run-02_proc-sss_meg.fif",
        "calibration": DATASET_ROOT / SUBJECT / "meg" / f"{SUBJECT}_acq-calibration_meg.dat",
        "crosstalk": DATASET_ROOT / SUBJECT / "meg" / f"{SUBJECT}_acq-crosstalk_meg.fif",
    }
    ancillary_rows = [
        {"item": key, "path": path.as_posix(), "exists": path.exists()}
        for key, path in ancillary.items()
    ]
    for row in ancillary_rows:
        if not row["exists"]:
            failures.append(f"{row['item']} missing")

    write_csv(
        OUT / "sub03_input_precheck.csv",
        rows,
        [
            "subject_id",
            "run_id",
            "maxfiltered_meg_path",
            "maxfiltered_meg_exists",
            "maxfiltered_meg_readable",
            "meg_channels",
            "mag_channels",
            "grad_channels",
            "raw_event_fif_path",
            "raw_event_fif_exists",
            "events_tsv_path",
            "events_tsv_exists",
            "envelope_path",
            "envelope_exists",
        ],
    )
    write_csv(OUT / "sub03_ancillary_precheck.csv", ancillary_rows, ["item", "path", "exists"])

    provenance = {
        "artifact": "official_preprocessing_provenance.json",
        "base_branch": BASE_BRANCH,
        "base_commit": BASE_COMMIT,
        "official_meg_scans_repo": OFFICIAL_REPO.as_posix(),
        "official_meg_scans_commit": official_commit(),
        "official_files": {
            "settings": (OFFICIAL_REPO / "speech" / "settings_speech.m").as_posix(),
            "preprocessing": (OFFICIAL_REPO / "speech" / "decoding" / "preprocessing_audiobooks_decoding.m").as_posix(),
            "trialfun": (OFFICIAL_REPO / "helper_functions" / "my_trialfun_audiobook.m").as_posix(),
        },
        "locked_protocol": {
            "use_maxfilter": True,
            "apply_latency_correction": True,
            "audio_latency_seconds": 0.003,
            "bandpass_hz": [0.5, 4],
            "trialdur_seconds": 120,
            "fs_neuro_hz": 1000,
            "fs_down_hz": 64,
            "envelope_mat": run_to_paths(RUNS[0])["envelope"].as_posix(),
        },
        "precheck_failures": failures,
    }
    (OUT / "official_preprocessing_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit("precheck failed: " + "; ".join(failures))
    print("sub-03 official preprocessing precheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
