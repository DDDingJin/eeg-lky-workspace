from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "meg_scans_official_preprocessing_sub03_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    required = [
        ROOT / "configs" / "benchmark" / "meg_scans_official_replication" / "sub03_preprocessing_v1.json",
        ROOT / "scripts" / "meg_scans_official_replication" / "precheck_sub03.py",
        ROOT / "scripts" / "meg_scans_official_replication" / "run_official_preprocessing_sub03.m",
        ROOT / "scripts" / "meg_scans_official_replication" / "validate_official_preprocessing_sub03.m",
        ROOT / "docs" / "meg_scans_official_replication_sub03_zh.md",
        ROOT / "workflow" / "meg_scans_official_replication_START_HERE.md",
        OUT / "official_preprocessing_provenance.json",
        OUT / "sub03_input_precheck.csv",
        OUT / "sub03_ancillary_precheck.csv",
        OUT / "sub03_official_trial_validation.csv",
        OUT / "sub03_official_validation_summary.json",
        OUT / "sub03_official_preprocessing_log_summary.md",
        OUT / "preprocessing_validation_report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing compact artifacts: {missing}")
    for path in [
        ROOT / "configs" / "benchmark" / "meg_scans_official_replication" / "sub03_preprocessing_v1.json",
        OUT / "official_preprocessing_provenance.json",
        OUT / "sub03_official_validation_summary.json",
    ]:
        json.loads(path.read_text(encoding="utf-8"))
    precheck = read_csv(OUT / "sub03_input_precheck.csv")
    if len(precheck) != 4:
        raise SystemExit("expected four audiobook precheck rows")
    for row in precheck:
        if row["maxfiltered_meg_exists"] != "True" or row["maxfiltered_meg_readable"] != "True":
            raise SystemExit(f"bad maxfilter precheck row: {row}")
        if (row["meg_channels"], row["mag_channels"], row["grad_channels"]) != ("306", "102", "204"):
            raise SystemExit(f"bad channel counts: {row}")
    trials = read_csv(OUT / "sub03_official_trial_validation.csv")
    if len(trials) != 4:
        raise SystemExit("expected four trial validation rows")
    for row in trials:
        if int(row["paired_trials_kept"]) <= 0:
            raise SystemExit(f"empty paired run: {row}")
        if row["min_neuro_len"] != row["max_neuro_len"] or row["min_audio_len"] != row["max_audio_len"]:
            raise SystemExit(f"length mismatch spread: {row}")
        if row["min_neuro_len"] != row["min_audio_len"]:
            raise SystemExit(f"audio/neuro length mismatch: {row}")
    banned_ext = {".mat", ".fif", ".gz", ".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth", ".ckpt"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("configs/benchmark/meg_scans_official_replication/", "scripts/meg_scans_official_replication/", "experiments/meg_scans_official_preprocessing_sub03_v1/")):
            if path.suffix.lower() in banned_ext:
                raise SystemExit(f"banned large/raw artifact in compact outputs: {rel}")
            if path.stat().st_size > 5_000_000:
                raise SystemExit(f"unexpected large artifact: {rel}")
    print("compact artifact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
