from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "meg_scans_canonical_05_08hz_64hz_sub03_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def main() -> int:
    required = [
        ROOT / "configs" / "benchmark" / "meg_scans_canonical" / "sub03_05_08hz_64hz_v1.json",
        ROOT / "scripts" / "meg_scans_canonical" / "run_canonical_sub03_05_08hz_64hz.m",
        ROOT / "scripts" / "meg_scans_canonical" / "validate_canonical_sub03_05_08hz_64hz.m",
        OUT / "protocol_lock.json",
        OUT / "trial_manifest.csv",
        OUT / "validation_report.md",
        OUT / "psd_summary.csv",
        OUT / "run_manifest.json",
        OUT / "run_manifest.csv",
        OUT / "invalid_trials.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing compact artifacts: {missing}")
    for path in [
        ROOT / "configs" / "benchmark" / "meg_scans_canonical" / "sub03_05_08hz_64hz_v1.json",
        OUT / "protocol_lock.json",
        OUT / "run_manifest.json",
    ]:
        json.loads(path.read_text(encoding="utf-8"))

    trials = read_csv(OUT / "trial_manifest.csv")
    if len(trials) != 16:
        raise SystemExit(f"expected 16 trial rows, found {len(trials)}")
    for row in trials:
        if row["meg_channels"] != "306" or row["meg_samples"] != "7680" or row["envelope_samples"] != "7680" or row["fs_hz"] != "64":
            raise SystemExit(f"bad trial row: {row}")

    runs = read_csv(OUT / "run_manifest.csv")
    if len(runs) != 4:
        raise SystemExit(f"expected four run rows, found {len(runs)}")
    for row in runs:
        if not truthy(row["meg_source_exists"]) or not truthy(row["wav_source_exists"]):
            raise SystemExit(f"missing source row: {row}")
        if row["paired_trials_kept"] != "4":
            raise SystemExit(f"expected four kept trials per run: {row}")

    psd = read_csv(OUT / "psd_summary.csv")
    expected_bands = {"0.5-4", "4-8", "8-16"}
    if {row["band_hz"] for row in psd} != expected_bands:
        raise SystemExit("PSD summary missing expected bands")
    env_4_8 = [float(row["relative_power_0p5_16"]) for row in psd if row["signal"] == "envelope" and row["band_hz"] == "4-8"]
    if not env_4_8 or max(env_4_8) <= 0.01:
        raise SystemExit("canonical envelope 4-8 Hz relative power is unexpectedly tiny; possible old 0.5-4 copy")

    banned_ext = {".mat", ".fif", ".gz", ".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".pkl"}
    allowed_prefixes = (
        "configs/benchmark/meg_scans_canonical/",
        "scripts/meg_scans_canonical/",
        "experiments/meg_scans_canonical_05_08hz_64hz_sub03_v1/",
        "workflow/meg_scans_canonical_START_HERE.md",
        "docs/meg_scans_canonical_05_08hz_64hz_sub03_zh.md",
    )
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(allowed_prefixes):
            if path.suffix.lower() in banned_ext:
                raise SystemExit(f"banned raw/large artifact in compact outputs: {rel}")
            if path.stat().st_size > 5_000_000:
                raise SystemExit(f"unexpected large compact artifact: {rel}")
    print("canonical compact artifact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
