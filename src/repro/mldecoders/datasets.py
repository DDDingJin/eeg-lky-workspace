from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from torch.utils.data import Dataset


SPLITS = {
    "train": tuple(range(0, 9)),
    "val": tuple(range(9, 12)),
    "test": tuple(range(12, 15)),
}


@dataclass(frozen=True)
class WindowRecord:
    part_idx: int
    start_idx: int


class HugoH5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        participant: str = "P00",
        split: str = "train",
        window_size: int = 50,
        stride: int = 1,
        channels: Iterable[int] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.participant = participant
        self.split = split
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.channels = None if channels is None else np.asarray(list(channels), dtype=int)

        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split}")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.part_indices = SPLITS[self.split]
        self._build_index()

    def _build_index(self) -> None:
        self.index: list[WindowRecord] = []
        self.part_lengths: dict[int, int] = {}

        with h5py.File(self.h5_path, "r") as f:
            if self.participant not in f["eeg"]:
                raise KeyError(f"participant {self.participant} not found in h5")

            for part_idx in self.part_indices:
                eeg = f[f"eeg/{self.participant}/part{part_idx}"]
                stim = f[f"stim/part{part_idx}"]

                eeg_len = eeg.shape[1]
                stim_len = stim.shape[0]
                if eeg_len != stim_len:
                    raise ValueError(
                        f"length mismatch for {self.participant} part{part_idx}: eeg={eeg_len}, stim={stim_len}"
                    )

                self.part_lengths[part_idx] = eeg_len
                max_start = eeg_len - self.window_size
                if max_start < 0:
                    continue

                for start_idx in range(0, max_start + 1, self.stride):
                    self.index.append(WindowRecord(part_idx=part_idx, start_idx=start_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        record = self.index[idx]
        with h5py.File(self.h5_path, "r") as f:
            eeg = f[f"eeg/{self.participant}/part{record.part_idx}"][:, record.start_idx : record.start_idx + self.window_size]
            stim = f[f"stim/part{record.part_idx}"][record.start_idx]

        if self.channels is not None:
            eeg = eeg[self.channels]

        return eeg.astype(np.float32), np.float32(stim)

    def describe(self) -> dict:
        return {
            "h5_path": str(self.h5_path),
            "participant": self.participant,
            "split": self.split,
            "part_indices": list(self.part_indices),
            "window_size": self.window_size,
            "stride": self.stride,
            "num_channels": None if self.channels is None else int(len(self.channels)),
            "part_lengths": self.part_lengths,
            "num_windows": len(self.index),
        }


def load_hugo_continuous_parts(
    h5_path: str | Path,
    participant: str = "P00",
    parts: Iterable[int] | None = None,
    channels: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h5_path = Path(h5_path)
    if parts is None:
        parts = range(15)

    channels_arr = None if channels is None else np.asarray(list(channels), dtype=int)

    with h5py.File(h5_path, "r") as f:
        eeg_parts = []
        stim_parts = []

        if participant not in f["eeg"]:
            raise KeyError(f"participant {participant} not found in h5")

        for part_idx in parts:
            eeg = f[f"eeg/{participant}/part{part_idx}"][:]
            stim = f[f"stim/part{part_idx}"][:]
            if channels_arr is not None:
                eeg = eeg[channels_arr]
            if eeg.shape[1] != stim.shape[0]:
                raise ValueError(
                    f"length mismatch for {participant} part{part_idx}: eeg={eeg.shape[1]}, stim={stim.shape[0]}"
                )
            eeg_parts.append(eeg)
            stim_parts.append(stim)

    return np.hstack(eeg_parts), np.hstack(stim_parts)
