from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "meg_scans_canonical_05_08hz_64hz_sub03_v1"
EXPECTED_UPSTREAM = "32bfc690e28e7591b45d96615c59b2d53b6a7165"


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
        OUT / "upstream_version_check.json",
        OUT / "channel_inventory.csv",
        OUT / "mag102_selection.json",
        OUT / "channelwise_psd_summary.csv",
        OUT / "data_provenance.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing compact artifacts: {missing}")
    json_payloads = {}
    for path in [
        ROOT / "configs" / "benchmark" / "meg_scans_canonical" / "sub03_05_08hz_64hz_v1.json",
        OUT / "protocol_lock.json",
        OUT / "run_manifest.json",
        OUT / "upstream_version_check.json",
        OUT / "mag102_selection.json",
        OUT / "data_provenance.json",
    ]:
        json_payloads[path.name] = json.loads(path.read_text(encoding="utf-8"))

    upstream = json_payloads["upstream_version_check.json"]
    if upstream["expected_commit"] != EXPECTED_UPSTREAM or upstream["observed_commit"] != EXPECTED_UPSTREAM or upstream["status"] != "passed":
        raise SystemExit(f"bad upstream version check: {upstream}")

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

    inventory = read_csv(OUT / "channel_inventory.csv")
    if len(inventory) != 306:
        raise SystemExit(f"expected 306 channel inventory rows, found {len(inventory)}")
    labels = [row["channel_label"] for row in inventory]
    if len(set(labels)) != 306:
        raise SystemExit("channel inventory labels are not unique")
    mag_rows = [row for row in inventory if truthy(row["is_magnetometer"])]
    grad_rows = [row for row in inventory if truthy(row["is_gradiometer"])]
    if len(mag_rows) != 102 or len(grad_rows) != 204:
        raise SystemExit(f"bad channel counts: mag={len(mag_rows)} grad={len(grad_rows)}")

    mag_selection = json_payloads["mag102_selection.json"]
    mag_indices = [int(v) for v in mag_selection["source_indices_1based"]]
    mag_labels = list(mag_selection["ordered_channel_labels"])
    if mag_selection["source_channel_count"] != 306 or mag_selection["selected_channel_count"] != 102:
        raise SystemExit(f"bad mag102 counts: {mag_selection}")
    if len(mag_indices) != 102 or len(set(mag_indices)) != 102 or len(mag_labels) != 102 or len(set(mag_labels)) != 102:
        raise SystemExit("mag102 selection must have 102 unique indices and labels")
    inventory_by_index = {int(row["native_index_1based"]): row for row in inventory}
    for idx, label in zip(mag_indices, mag_labels):
        row = inventory_by_index.get(idx)
        if row is None or row["channel_label"] != label or not truthy(row["is_magnetometer"]):
            raise SystemExit(f"mag102 selection does not match inventory at index {idx}: {label}")

    channelwise = read_csv(OUT / "channelwise_psd_summary.csv")
    cw_sensors = {row["sensor_type"] for row in channelwise}
    cw_bands = {row["band_hz"] for row in channelwise}
    if not {"mag", "grad"}.issubset(cw_sensors) or cw_bands != expected_bands:
        raise SystemExit("channelwise PSD must include mag and grad for all expected bands")
    for row in channelwise:
        expected_count = "102" if row["sensor_type"] == "mag" else "204"
        if row["channel_count"] != expected_count:
            raise SystemExit(f"bad channelwise PSD channel count: {row}")

    provenance = json_payloads["data_provenance.json"]
    if provenance["preprocessing_rerun"] or provenance["training_run"] or provenance["data_body_modified"]:
        raise SystemExit(f"bad provenance flags: {provenance}")
    for key in ["local_paired_mat", "local_envelope_mat"]:
        record = provenance[key]
        if len(record["sha256"]) != 64 or int(record["bytes"]) <= 0:
            raise SystemExit(f"bad provenance file record: {record}")

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
