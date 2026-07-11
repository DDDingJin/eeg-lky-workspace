from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "meg_scans_representation_preflight_v1"
ALLOWED_PREFIXES = {
    "configs/benchmark/meg_scans_crossmodal/",
    "scripts/meg_scans_crossmodal/",
    "experiments/meg_scans_representation_preflight_v1/",
    "docs/meg_scans_crossmodal_protocol_zh.md",
    "workflow/meg_scans_crossmodal_START_HERE.md",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    required = [
        ROOT / "docs" / "meg_scans_crossmodal_protocol_zh.md",
        ROOT / "workflow" / "meg_scans_crossmodal_START_HERE.md",
        OUT / "protocol_lock.json",
        OUT / "scans_data_inventory.csv",
        OUT / "eeg_meg_protocol_compatibility.md",
        OUT / "scans_block_manifest.csv",
        OUT / "representation_manifest.json",
        OUT / "meg102_to_target_eeg64_mapping.csv",
        OUT / "preflight_validation_report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    for path in [OUT / "protocol_lock.json", OUT / "representation_manifest.json"]:
        json.loads(path.read_text(encoding="utf-8"))

    inventory = read_csv(OUT / "scans_data_inventory.csv")
    if not inventory:
        raise SystemExit("empty scans_data_inventory.csv")
    for row in inventory:
        if int(row["meg_total_channels"]) != 306 or int(row["mag_channel_count"]) != 102 or int(row["grad_channel_count"]) != 204:
            raise SystemExit(f"bad channel count: {row}")

    mapping = read_csv(OUT / "meg102_to_target_eeg64_mapping.csv")
    if len(mapping) != 102:
        raise SystemExit(f"expected 102 mag mapping rows, found {len(mapping)}")
    selected = [row["selected_meg_mag_channel"] for row in mapping]
    if len(selected) != len(set(selected)):
        raise SystemExit("duplicate selected MEG mag channel")
    if any(row["mapping_status"] != "blocked_no_confirmed_target_eeg64_coordinates" for row in mapping):
        raise SystemExit("R3 mapping should remain blocked without target EEG64 coordinates")

    blocks = read_csv(OUT / "scans_block_manifest.csv")
    if not blocks:
        raise SystemExit("empty scans_block_manifest.csv")
    by_run: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row in blocks:
        if int(row["expected_samples_at_64hz"]) != 7680:
            raise SystemExit(f"bad expected sample count: {row}")
        by_run.setdefault((row["subject_id"], row["run_id"]), []).append((int(row["start_sample_raw_1000hz"]), int(row["end_sample_raw_1000hz"])))
    for key, intervals in by_run.items():
        intervals = sorted(intervals)
        for prev, cur in zip(intervals, intervals[1:]):
            if prev[1] > cur[0]:
                raise SystemExit(f"overlapping blocks for {key}: {prev} {cur}")

    large_ext = {".fif", ".gz", ".npy", ".npz", ".h5", ".hdf5", ".mat", ".pkl", ".pt", ".pth", ".ckpt"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            if path.suffix.lower() in large_ext:
                raise SystemExit(f"large/raw artifact under allowed output: {rel}")
            if path.stat().st_size > 5_000_000:
                raise SystemExit(f"unexpected large compact artifact: {rel}")
    print("preflight validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
