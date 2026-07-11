from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import h5py

os.environ.setdefault("MNE_USE_NUMBA", "false")
import mne


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path("E:/decode/data/raw/meg_scans_ds006468")
DERIVED_ROOT = Path("E:/decode/data/derived/meg_scans_official_replication_v1")
OFFICIAL_REPO = Path("E:/decode/external/upstream/MEG-SCANS")
OUT = ROOT / "experiments" / "meg_scans_official_preprocessing_sub03_v1"
SUBJECT = "sub-03"
OFFICIAL_COMMIT = "32bfc690e28e7591b45d96615c59b2d53b6a7165"


def official_commit() -> str:
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


def read_channel_counts(path: Path) -> tuple[int, int, int]:
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, misc=False, exclude=[])
    mag = mne.pick_types(raw.info, meg="mag", eeg=False, eog=False, stim=False, misc=False, exclude=[])
    grad = mne.pick_types(raw.info, meg="grad", eeg=False, eog=False, stim=False, misc=False, exclude=[])
    return len(meg), len(mag), len(grad)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = official_commit()
    failures: list[str] = []
    if commit != OFFICIAL_COMMIT:
        failures.append(f"official commit mismatch: {commit} != {OFFICIAL_COMMIT}")

    olsa_maxfilter = DATASET_ROOT / "derivatives" / SUBJECT / "maxfilter" / f"{SUBJECT}_task-olsa_proc-tsss-mc_meg.fif"
    olsa_raw = DATASET_ROOT / SUBJECT / "meg" / f"{SUBJECT}_task-olsa_meg.fif"
    events_tsv = DATASET_ROOT / SUBJECT / "meg" / f"{SUBJECT}_task-olsa_events.tsv"
    olsa_env = DATASET_ROOT / "derivatives" / "stimuli" / "preprocessed_olsa_envelopes_decoding.mat"
    audiobook_mat = DERIVED_ROOT / SUBJECT / "speech" / f"{SUBJECT}_preprocessed_audiobooks_decoding.mat"

    counts = (None, None, None)
    readable = False
    if olsa_maxfilter.exists():
        counts = read_channel_counts(olsa_maxfilter)
        readable = True
    else:
        failures.append(f"missing OLSA maxfilter: {olsa_maxfilter}")
    if counts != (306, 102, 204):
        failures.append(f"bad OLSA channel counts: {counts}")

    events_columns = []
    events_rows = 0
    if events_tsv.exists():
        events = pd.read_csv(events_tsv, sep="\t")
        events_columns = list(events.columns)
        events_rows = len(events)
        for col in ["SNR", "intelligibility", "stim_file"]:
            if col not in events.columns:
                failures.append(f"missing events.tsv column: {col}")
        if events_rows != 120:
            failures.append(f"unexpected OLSA events row count: {events_rows}")
    else:
        failures.append(f"missing OLSA events.tsv: {events_tsv}")

    for label, path in [
        ("OLSA raw event FIF", olsa_raw),
        ("OLSA envelope mat", olsa_env),
        ("audiobook paired mat", audiobook_mat),
    ]:
        if not path.exists():
            failures.append(f"missing {label}: {path}")

    olsa_envelope_readable = False
    if olsa_env.exists():
        with h5py.File(olsa_env, "r") as handle:
            olsa_envelope_readable = "audio_envelopes" in handle
        if not olsa_envelope_readable:
            failures.append("OLSA envelope mat is readable but missing audio_envelopes")

    audiobook_paired_trials = 0
    if audiobook_mat.exists():
        with h5py.File(audiobook_mat, "r") as handle:
            audiobook_paired_trials = int(handle["results"]["epochs_audio"].shape[0])
        if audiobook_paired_trials != 16:
            failures.append(f"expected 16 audiobook paired trials, found {audiobook_paired_trials}")

    rows = [
        {
            "subject_id": SUBJECT,
            "official_commit": commit,
            "olsa_maxfilter_path": olsa_maxfilter.as_posix(),
            "olsa_maxfilter_exists": olsa_maxfilter.exists(),
            "olsa_maxfilter_readable": readable,
            "meg_channels": counts[0],
            "mag_channels": counts[1],
            "grad_channels": counts[2],
            "olsa_raw_event_fif_path": olsa_raw.as_posix(),
            "olsa_raw_event_fif_exists": olsa_raw.exists(),
            "events_tsv_path": events_tsv.as_posix(),
            "events_tsv_exists": events_tsv.exists(),
            "events_rows": events_rows,
            "events_has_SNR": "SNR" in events_columns,
            "events_has_intelligibility": "intelligibility" in events_columns,
            "events_has_stim_file": "stim_file" in events_columns,
            "olsa_envelope_path": olsa_env.as_posix(),
            "olsa_envelope_exists": olsa_env.exists(),
            "olsa_envelope_readable": olsa_envelope_readable,
            "audiobook_paired_mat_path": audiobook_mat.as_posix(),
            "audiobook_paired_mat_exists": audiobook_mat.exists(),
            "audiobook_paired_trials": audiobook_paired_trials,
        }
    ]
    write_csv(
        OUT / "sub03_official_anchor_precheck.csv",
        rows,
        list(rows[0].keys()),
    )

    report = {
        "artifact": "official_anchor_precheck.json",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "official_commit": commit,
        "expected_official_commit": OFFICIAL_COMMIT,
        "olsa_maxfilter": olsa_maxfilter.as_posix(),
        "olsa_envelope": olsa_env.as_posix(),
        "audiobook_paired_mat": audiobook_mat.as_posix(),
    }
    (OUT / "official_anchor_precheck.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit("anchor precheck failed: " + "; ".join(failures))
    print("sub-03 official anchor precheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
