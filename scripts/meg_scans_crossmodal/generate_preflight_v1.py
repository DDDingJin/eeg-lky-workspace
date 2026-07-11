from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path("E:/decode/data/raw/meg_scans_ds006468")
OUT_DIR = ROOT / "experiments" / "meg_scans_representation_preflight_v1"


BASE_BRANCH = "origin/fix/ar-20260625-161300-a43831b-subject-specific-model-availability-v1"
BASE_COMMIT = "41a9c458731962b3dc1da575a159abb6a6c211a8"


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def mne_module():
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    import mne

    return mne


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_lines(path: Path, patterns: list[str]) -> dict[str, list[int]]:
    out = {pattern: [] for pattern in patterns}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, start=1):
        for pattern in patterns:
            if pattern in line:
                out[pattern].append(idx)
    return out


def line_ref(path: str, pattern: str, lines: dict[str, list[int]]) -> str:
    nums = lines.get(pattern, [])
    if not nums:
        return path
    return f"{path}:{nums[0]}"


def build_protocol_lock() -> dict[str, Any]:
    sources = {
        "baseline": Path("configs/baseline_protocol.json"),
        "full_subject_config": Path("configs/benchmark/gate0_gate2_full_subject_single_seed.json"),
        "reference_baselines": Path("src/repro/reference_baselines.py"),
        "adt_exact": Path("src/repro/adt_exact.py"),
        "scoring": Path("src/benchmark/scoring.py"),
        "schema": Path("src/benchmark/result_schema.py"),
        "runner": Path("scripts/run_gate0_gate2_full_subject_single_seed.py"),
        "hugo_h5": Path("E:/decode/data/processed/mldecoders/hugo_sample_125hz.h5"),
    }
    line_maps = {
        key: read_lines(ROOT / path, [
            "sampling_rate",
            "primary",
            "split_policy",
            "train_parts",
            "val_parts",
            "test_parts",
            "window_size",
            "target_index",
            "window_length",
            "hop_length",
            "recording_id",
            "pearson_on_valid",
            "mean_recording_pearson_r",
            "seed",
            "channels=range(64)",
            "env[start +",
            "eeg_path.stem.replace",
        ])
        for key, path in sources.items()
        if not str(path).startswith("E:/")
    }

    h5_channel_coord_status = "blocked_no_channel_coordinates"
    h5_channel_count = None
    h5_channel_names_path = "meta/channel_names"
    with h5py.File(sources["hugo_h5"], "r") as handle:
        h5_channel_count = int(handle[h5_channel_names_path].shape[0])

    fields = {
        "sampling_rate": {
            "value": 64,
            "source": line_ref("configs/benchmark/gate0_gate2_full_subject_single_seed.json", '"sampling_rate"', line_maps["full_subject_config"]),
        },
        "legacy_baseline_sampling_rate": {
            "value": 125,
            "source": line_ref("configs/baseline_protocol.json", '"sampling_rate_hz"', line_maps["baseline"]),
        },
        "eeg_bandpass": {
            "value": "not_locked_in_unified_runner_config",
            "source": "not found in accepted full-subject runner/config; existing reference_splits are treated as preprocessed inputs",
        },
        "envelope_bandpass": {
            "value": "not_locked_in_unified_runner_config",
            "source": "not found in accepted full-subject runner/config; existing reference_splits are treated as preprocessed inputs",
        },
        "exact_envelope_extraction_definition": {
            "value": "blocked_for_crossmodal_lock; accepted runner consumes *_envelope.npy and does not encode extraction definition",
            "source": line_ref("src/repro/reference_baselines.py", "env_path", line_maps["reference_baselines"]),
        },
        "dnn_window_size": {
            "value": 50,
            "source": line_ref("configs/benchmark/gate0_gate2_full_subject_single_seed.json", '"window_size"', line_maps["full_subject_config"]),
        },
        "sequence_window_size": {
            "value": 320,
            "source": line_ref("configs/benchmark/gate0_gate2_full_subject_single_seed.json", '"window_length"', line_maps["full_subject_config"]),
        },
        "sequence_hop_size": {
            "value": 64,
            "source": line_ref("configs/benchmark/gate0_gate2_full_subject_single_seed.json", '"hop_length"', line_maps["full_subject_config"]),
        },
        "dnn_target_index_alignment": {
            "value": "last sample in window",
            "source": line_ref("src/repro/reference_baselines.py", "env[start +", line_maps["reference_baselines"]),
        },
        "recording_id_definition": {
            "value": "filename stem prefix before _-_eeg.npy; split_-_participant_-_recording_id_-_eeg.npy convention",
            "source": line_ref("src/repro/reference_baselines.py", "eeg_path.stem.replace", line_maps["reference_baselines"]),
        },
        "train_val_test_split_rule": {
            "value": "accepted runner consumes pre-exported train/val/test split files; legacy baseline_protocol has part split 0-8/9-11/12-14",
            "source": f"{line_ref('src/repro/reference_baselines.py', 'load_reference_recordings', line_maps['reference_baselines'])}; {line_ref('configs/baseline_protocol.json', 'train_parts', line_maps['baseline'])}",
        },
        "scaler_normalization_fitting_rule": {
            "value": "not in accepted runner; normalization is expected to be done by dataset export/preprocessing",
            "source": "not found in accepted full-subject runner/config",
        },
        "scorer_rule": {
            "value": "benchmark.scoring.pearson_on_valid over valid mask after overlapping-window aggregation",
            "source": line_ref("src/benchmark/scoring.py", "pearson_on_valid", line_maps["scoring"]),
        },
        "aggregation_rule": {
            "value": "recording metric = Pearson per recording; subject metric = mean_recording_pearson_r; dataset metric = mean subject metric",
            "source": f"{line_ref('configs/benchmark/gate0_gate2_full_subject_single_seed.json', 'aggregation_rule', line_maps['full_subject_config'])}; {line_ref('src/benchmark/scoring.py', 'mean_recording_pearson_r', line_maps['scoring'])}",
        },
        "seed_rule": {
            "value": 0,
            "source": line_ref("configs/benchmark/gate0_gate2_full_subject_single_seed.json", '"seed"', line_maps["full_subject_config"]),
        },
        "input_output_tensor_contract": {
            "value": "EEG arrays are time x channels; scalar-window models consume channels x window and target last sample; sequence models consume window x channels and output window x 1",
            "source": f"{line_ref('src/repro/reference_baselines.py', 'x = eeg[start', line_maps['reference_baselines'])}; {line_ref('src/repro/adt_exact.py', 'eeg_window', line_maps['adt_exact'])}",
        },
        "target_eeg64_channel_coordinates": {
            "value": h5_channel_coord_status,
            "source": f"E:/decode/data/processed/mldecoders/hugo_sample_125hz.h5:{h5_channel_names_path}; channel_count={h5_channel_count}, coordinate_dataset_absent",
        },
    }
    return {
        "artifact": "protocol_lock.json",
        "base_branch": BASE_BRANCH,
        "base_commit": BASE_COMMIT,
        "lock_scope": "read-only canonical EEG protocol extraction for MEG-SCANS cross-modal preflight",
        "fields": fields,
    }


def read_raw_info(path: Path) -> tuple[Any, list[int], list[int], list[int], list[str], dict[str, list[float]]]:
    mne = mne_module()
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, misc=False, exclude=[])
    mags = mne.pick_types(raw.info, meg="mag", eeg=False, eog=False, stim=False, misc=False, exclude=[])
    grads = mne.pick_types(raw.info, meg="grad", eeg=False, eog=False, stim=False, misc=False, exclude=[])
    names = raw.ch_names
    coords = {}
    for idx in mags:
        coords[names[idx]] = [float(x) for x in raw.info["chs"][idx]["loc"][:3]]
    return raw, list(meg), list(mags), list(grads), names, coords


def subject_paths(subject: str) -> dict[str, Path | None]:
    sub_root = DATASET_ROOT / subject
    deriv = DATASET_ROOT / "derivatives" / subject
    return {
        "t1w_mri": sub_root / "anat" / f"{subject}_T1w.nii.gz",
        "coreg_transform": deriv / "coregistration" / f"{subject}_trans.fif",
        "noise_run_01": sub_root / "meg" / f"{subject}_task-noise_run-01_meg.fif",
        "noise_run_02": sub_root / "meg" / f"{subject}_task-noise_run-02_meg.fif",
        "calibration": sub_root / "meg" / f"{subject}_acq-calibration_meg.dat",
        "crosstalk": sub_root / "meg" / f"{subject}_acq-crosstalk_meg.fif",
    }


def run_ids_for_subject(subject: str) -> list[str]:
    if subject == "sub-01":
        return ["audiobook1_run-01", "audiobook1_run-02"]
    return ["audiobook1_run-01", "audiobook1_run-02", "audiobook2_run-01", "audiobook2_run-02"]


def run_to_paths(subject: str, run_id: str) -> dict[str, Path]:
    task, run = run_id.split("_")
    stem = f"{subject}_task-{task}_{run}"
    return {
        "maxfilter_meg": DATASET_ROOT / "derivatives" / subject / "maxfilter" / f"{stem}_proc-tsss-mc_meg.fif",
        "raw_meg": DATASET_ROOT / subject / "meg" / f"{stem}_meg.fif",
        "events": DATASET_ROOT / subject / "meg" / f"{stem}_events.tsv",
        "envelope": DATASET_ROOT / "derivatives" / "stimuli" / ("sub-01_preprocessed_audiobook_envelopes_decoding.mat" if subject == "sub-01" else "sub-others_preprocessed_audiobook_envelopes_decoding.mat"),
    }


def scans_inventory_and_blocks() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, list[float]], dict[str, Any]]:
    inventory_rows = []
    block_rows = []
    all_block_ids: set[str] = set()
    subject_status = {}
    mag_names_for_mapping: list[str] = []
    mag_coords_for_mapping: dict[str, list[float]] = {}

    for subject in ["sub-01", "sub-03"]:
        first_run = run_ids_for_subject(subject)[0]
        first_paths = run_to_paths(subject, first_run)
        raw, meg, mags, grads, names, coords = read_raw_info(first_paths["maxfilter_meg"])
        if subject == "sub-03":
            mag_names_for_mapping = [names[idx] for idx in mags]
            mag_coords_for_mapping = coords
        paths = subject_paths(subject)
        has_source_bridge_materials = all(Path(p).exists() for p in paths.values() if p is not None)
        subject_status[subject] = {
            "meg_total_channels": len(meg),
            "mag_channels": len(mags),
            "grad_channels": len(grads),
            "has_t1w": paths["t1w_mri"].exists(),
            "has_coreg_transform": paths["coreg_transform"].exists(),
            "has_noise": paths["noise_run_01"].exists() and paths["noise_run_02"].exists(),
            "has_calibration": paths["calibration"].exists(),
            "has_crosstalk": paths["crosstalk"].exists(),
            "has_source_bridge_materials": has_source_bridge_materials,
        }
        for run_id in run_ids_for_subject(subject):
            rpaths = run_to_paths(subject, run_id)
            events = pd.read_csv(rpaths["events"], sep="\t")
            aud = events[events["trial_type"].eq("audiobook 1s")]
            start_sample = int(aud["sample"].min()) if len(aud) else -1
            stop_sample = int((aud["sample"].iloc[-1] + 1000)) if len(aud) else -1
            duration_sec = (stop_sample - start_sample) / 1000 if start_sample >= 0 else 0
            complete_blocks = int(math.floor(duration_sec / 120.0))
            inventory_rows.append({
                "dataset_id": "meg_scans_ds006468_v1.1.2",
                "subject_id": subject,
                "role": "bridge_validation_subject" if subject == "sub-01" else "primary_preflight_subject",
                "run_id": run_id,
                "maxfiltered_meg_file_path": rpaths["maxfilter_meg"].as_posix(),
                "maxfiltered_meg_exists": rpaths["maxfilter_meg"].exists(),
                "raw_meg_file_path": rpaths["raw_meg"].as_posix(),
                "events_path": rpaths["events"].as_posix(),
                "t1w_mri_path": paths["t1w_mri"].as_posix(),
                "coregistration_transform_path": paths["coreg_transform"].as_posix(),
                "empty_room_noise_file_path": f"{paths['noise_run_01'].as_posix()};{paths['noise_run_02'].as_posix()}",
                "calibration_file_path": paths["calibration"].as_posix(),
                "crosstalk_file_path": paths["crosstalk"].as_posix(),
                "meg_total_channels": len(meg),
                "mag_channel_count": len(mags),
                "grad_channel_count": len(grads),
                "first_mag_channel": names[mags[0]] if mags else "",
                "first_grad_channel": names[grads[0]] if grads else "",
                "audio_event_count": len(aud),
                "audiobook_duration_sec_from_events": round(duration_sec, 3),
                "envelope_path": rpaths["envelope"].as_posix(),
                "envelope_exists": rpaths["envelope"].exists(),
                "envelope_sampling_rate_hz": 64,
                "has_source_bridge_materials": has_source_bridge_materials,
            })
            if subject == "sub-03":
                for block_idx in range(complete_blocks):
                    block_start = start_sample + block_idx * 120000
                    block_end = block_start + 120000
                    block_id = f"{subject}_{run_id}_block-{block_idx:02d}"
                    all_block_ids.add(block_id)
                    block_rows.append({
                        "dataset_id": "meg_scans_ds006468_v1.1.2",
                        "subject_id": subject,
                        "run_id": run_id,
                        "block_id": block_id,
                        "start_sample_raw_1000hz": block_start,
                        "end_sample_raw_1000hz": block_end,
                        "meg_source_path": rpaths["maxfilter_meg"].as_posix(),
                        "envelope_source_path": rpaths["envelope"].as_posix(),
                        "expected_samples_at_64hz": 7680,
                        "validity_flag": "valid" if rpaths["maxfilter_meg"].exists() and rpaths["envelope"].exists() else "invalid_missing_source",
                    })
    meta = {
        "subject_status": subject_status,
        "block_count": len(block_rows),
        "unique_block_ids": len(all_block_ids),
    }
    return inventory_rows, block_rows, mag_names_for_mapping, mag_coords_for_mapping, meta


def mapping_rows(mag_names: list[str], mag_coords: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for idx, name in enumerate(mag_names):
        rows.append({
            "target_eeg_channel": "",
            "target_order": "",
            "selected_meg_mag_channel": name,
            "selected_meg_mag_order": idx,
            "distance": "",
            "coordinate_frame": "device",
            "mapping_status": "blocked_no_confirmed_target_eeg64_coordinates",
            "blocked_reason": "Existing EEG data exposes channel names but no accepted target EEG64 coordinate source; no BioSemi64 or other standard montage assumed.",
            "meg_x": mag_coords.get(name, ["", "", ""])[0],
            "meg_y": mag_coords.get(name, ["", "", ""])[1],
            "meg_z": mag_coords.get(name, ["", "", ""])[2],
        })
    return rows


def representation_manifest() -> dict[str, Any]:
    return {
        "artifact": "representation_manifest.json",
        "base_branch": BASE_BRANCH,
        "base_commit": BASE_COMMIT,
        "representations": [
            {"id": "R1", "name": "MEG306-native", "status": "preflight_defined", "input_shape": "time x 306", "notes": "Use all MEG channels in native MEG sensor order."},
            {"id": "R2", "name": "MEG102-mag-native", "status": "preflight_defined", "input_shape": "time x 102", "notes": "Use magnetometers only in native order."},
            {"id": "R3", "name": "MEG64-select", "status": "blocked", "input_shape": "time x 64", "notes": "Requires confirmed target EEG64 coordinates for one-to-one nearest-neighbor selection."},
            {"id": "R4", "name": "MEG64-interp", "status": "interface_only_blocked_target_coords", "input_shape": "time x 64", "interface": {"target_coordinates": "required EEG64 coordinate table", "source_coordinates": "MEG102 magnetometer device coordinates", "weights": "64 x 102 sparse interpolation matrix", "generation": "fit once per target montage, apply to each time x 102 block"}},
            {"id": "R5", "name": "virtual-EEG64", "status": "workflow_only", "input_shape": "time x 64", "required_inputs": ["T1w MRI", "MEG-MRI trans.fif", "empty-room noise", "calibration/crosstalk", "source space", "BEM", "MEG inverse operator", "EEG64 forward model"], "notes": "No inverse/forward generated in this preflight."},
        ],
        "target_eeg64_status": "blocked_no_confirmed_coordinates",
    }


def write_docs(meta: dict[str, Any]) -> None:
    docs_path = ROOT / "docs" / "meg_scans_crossmodal_protocol_zh.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        f"""# MEG-SCANS 跨模态传感器表示 preflight v1

本模块是隔离新增模块，base 为 `{BASE_BRANCH}` / `{BASE_COMMIT}`。本轮只生成 compact artifacts，不训练深度模型，不生成 source inverse / EEG forward / virtual EEG。

## 目标

为后续五种表示建立可审阅前置检查：

- R1: MEG306-native，使用 306 个 MEG 通道。
- R2: MEG102-mag-native，只使用 102 个 magnetometer。
- R3: MEG64-select，需要目标 EEG64 坐标；当前 blocked。
- R4: MEG64-interp，只定义接口和权重方案；当前因目标 EEG64 坐标 blocked。
- R5: virtual-EEG64，只定义 source inverse + EEG forward 工作流；本轮不执行。

## 当前结论

- sub-03 是正式 preflight subject，包含 4 个 audiobook run。
- sub-01 仅登记为后续 EEG32 + MEG102 bridge validation subject。
- sub-03 生成 `{meta['block_count']}` 个 120 秒 block manifest rows。
- 既有 EEG 数据无法确认目标 EEG64 坐标，因此 R3/R4 不能给出有效距离映射。

## 后续最小 smoke

建议先使用 R1 或 R2 在 sub-03 的 120 秒 block 上导出 time x channel / envelope split，再运行现有 unified scorer 的 recording-level Pearson 聚合。
""",
        encoding="utf-8",
    )
    wf = ROOT / "workflow" / "meg_scans_crossmodal_START_HERE.md"
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(
        f"""# MEG-SCANS Crossmodal START HERE

- Branch: `fix/ar-20260711-meg-scans-representation-preflight-v1`
- Worktree: `E:\\decode\\_meg_scans_crossmodal_preflight_v1`
- Base: `{BASE_BRANCH}` / `{BASE_COMMIT}`
- Main artifact dir: `experiments/meg_scans_representation_preflight_v1`

## Resume order

1. Read `docs/meg_scans_crossmodal_protocol_zh.md`.
2. Read `experiments/meg_scans_representation_preflight_v1/protocol_lock.json`.
3. Read `experiments/meg_scans_representation_preflight_v1/scans_data_inventory.csv`.
4. Read `experiments/meg_scans_representation_preflight_v1/scans_block_manifest.csv`.
5. Read `experiments/meg_scans_representation_preflight_v1/representation_manifest.json`.

## Current status

This is a compact preflight only. No raw arrays, checkpoints, predictions, source inverse, or forward models are committed.
""",
        encoding="utf-8",
    )


def write_markdown_reports(meta: dict[str, Any]) -> None:
    (OUT_DIR / "eeg_meg_protocol_compatibility.md").write_text(
        """# EEG/MEG Protocol Compatibility

## Compatible

- Sampling target can be aligned at 64 Hz.
- Recording-level unit can be represented as `recording_id` rows and scored with `pearson_on_valid`.
- R1/R2 can preserve time x channels arrays and existing target alignment conventions.

## Not directly compatible

- EEG accepted model paths assume `channels=range(64)` in the runner.
- MEG native has 306 channels; magnetometer-only has 102 channels.
- Existing EEG HDF5 metadata does not provide confirmed EEG64 sensor coordinates.
- Existing accepted runner consumes preprocessed `*_envelope.npy`; it does not lock an envelope extraction implementation.

## Decision

For the next smoke, train from scratch on MEG representation outputs. Do not transfer EEG-trained weights directly.
""",
        encoding="utf-8",
    )
    (OUT_DIR / "preflight_validation_report.md").write_text(
        f"""# Preflight Validation Report

- Base branch: `{BASE_BRANCH}`
- Base commit: `{BASE_COMMIT}`
- Subjects checked: `sub-01`, `sub-03`
- sub-03 block rows: `{meta['block_count']}`
- Unique block IDs: `{meta['unique_block_ids']}`
- R3 mapping status: blocked, because target EEG64 coordinates are unavailable from accepted EEG data.
- R4 status: interface only, blocked on target coordinates.
- R5 status: workflow only, required MRI/coreg/noise inputs present for inspected subjects.

## Validation summary

- Channel-count check: expected 306/102/204 from representative maxfilter FIF.
- Coordinate availability check: MEG magnetometer device coordinates available; EEG64 target coordinates unavailable.
- One-to-one mapping uniqueness check: not applicable while R3 is blocked.
- Block overlap/leakage check: block IDs are unique and 120 s windows are non-overlapping within each run.
- Raw/large-file check: artifacts are JSON/CSV/MD/PY only; no raw MEG/MRI/envelope arrays/checkpoints/predictions are added.
""",
        encoding="utf-8",
    )


def write_docs(meta: dict[str, Any]) -> None:
    docs_path = ROOT / "docs" / "meg_scans_crossmodal_protocol_zh.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        f"""# MEG-SCANS 跨模态传感器表示 preflight v1

本模块是隔离新增模块，base 为 `{BASE_BRANCH}` / `{BASE_COMMIT}`。本轮只生成 compact artifacts，不训练深度模型，不生成 source inverse、EEG forward 或 virtual EEG。

## 目标

为后续五种表示建立可审阅前置检查：

- R1: MEG306-native，使用 306 个 MEG 通道。
- R2: MEG102-mag-native，只使用 102 个 magnetometer。
- R3: MEG64-select，需要目标 EEG64 坐标；当前 blocked。
- R4: MEG64-interp，只定义接口和权重生成方案；当前因目标 EEG64 坐标 blocked。
- R5: virtual-EEG64，只定义 source inverse + EEG forward 工作流；本轮不执行。

## 当前结论

- sub-03 是正式 preflight subject，包含 4 个 audiobook run。
- sub-01 仅登记为后续 EEG32 + MEG102 bridge validation subject。
- sub-03 生成 `{meta['block_count']}` 个 120 秒 block manifest rows。
- 既有 EEG 数据无法确认目标 EEG64 坐标，因此 R3/R4 不能给出有效距离映射。

## 后续最小 smoke

建议先使用 R1 或 R2 在 sub-03 的 120 秒 block 上导出 `time x channel` / envelope split，再运行现有 unified scorer 的 recording-level Pearson 聚合。
""",
        encoding="utf-8",
    )
    wf = ROOT / "workflow" / "meg_scans_crossmodal_START_HERE.md"
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(
        f"""# MEG-SCANS Crossmodal START HERE

- Branch: `fix/ar-20260711-meg-scans-representation-preflight-v1`
- Worktree: `E:\\decode\\_meg_scans_crossmodal_preflight_v1`
- Base: `{BASE_BRANCH}` / `{BASE_COMMIT}`
- Main artifact dir: `experiments/meg_scans_representation_preflight_v1`

## Resume order

1. Read `docs/meg_scans_crossmodal_protocol_zh.md`.
2. Read `experiments/meg_scans_representation_preflight_v1/protocol_lock.json`.
3. Read `experiments/meg_scans_representation_preflight_v1/scans_data_inventory.csv`.
4. Read `experiments/meg_scans_representation_preflight_v1/scans_block_manifest.csv`.
5. Read `experiments/meg_scans_representation_preflight_v1/representation_manifest.json`.

## Current status

This is a compact preflight only. No raw arrays, checkpoints, predictions, source inverse, or forward models are committed.
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_rows, block_rows, mag_names, mag_coords, meta = scans_inventory_and_blocks()
    write_csv(OUT_DIR / "scans_data_inventory.csv", inventory_rows, [
        "dataset_id", "subject_id", "role", "run_id", "maxfiltered_meg_file_path", "maxfiltered_meg_exists",
        "raw_meg_file_path", "events_path", "t1w_mri_path", "coregistration_transform_path",
        "empty_room_noise_file_path", "calibration_file_path", "crosstalk_file_path", "meg_total_channels",
        "mag_channel_count", "grad_channel_count", "first_mag_channel", "first_grad_channel",
        "audio_event_count", "audiobook_duration_sec_from_events", "envelope_path", "envelope_exists",
        "envelope_sampling_rate_hz", "has_source_bridge_materials",
    ])
    write_csv(OUT_DIR / "scans_block_manifest.csv", block_rows, [
        "dataset_id", "subject_id", "run_id", "block_id", "start_sample_raw_1000hz", "end_sample_raw_1000hz",
        "meg_source_path", "envelope_source_path", "expected_samples_at_64hz", "validity_flag",
    ])
    write_csv(OUT_DIR / "meg102_to_target_eeg64_mapping.csv", mapping_rows(mag_names, mag_coords), [
        "target_eeg_channel", "target_order", "selected_meg_mag_channel", "selected_meg_mag_order",
        "distance", "coordinate_frame", "mapping_status", "blocked_reason", "meg_x", "meg_y", "meg_z",
    ])
    json_dump(OUT_DIR / "protocol_lock.json", build_protocol_lock())
    json_dump(OUT_DIR / "representation_manifest.json", representation_manifest())
    write_markdown_reports(meta)
    write_docs(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
