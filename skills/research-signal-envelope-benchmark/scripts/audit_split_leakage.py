#!/usr/bin/env python3
"""Audit a CSV split manifest for cross-split source overlap."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from _common import print_errors


REQUIRED = {"split", "dataset_id", "subject_id", "recording_id", "source_start", "source_end"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--protocol", choices=("within-subject", "cross-subject", "cross-dataset"), default="within-subject"
    )
    args = parser.parse_args()
    errors: list[str] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED - fields
        if missing:
            return print_errors([f"missing columns: {', '.join(sorted(missing))}"])
        rows = list(reader)

    groups: dict[tuple[str, str, str], list[tuple[int, int, str, int]]] = defaultdict(list)
    seen_source: dict[str, tuple[str, int]] = {}
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    datasets_by_split: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows, 2):
        split = row["split"].strip().lower()
        if split not in {"train", "val", "test"}:
            errors.append(f"line {index}: invalid split {split!r}")
            continue
        try:
            start = int(row["source_start"])
            end = int(row["source_end"])
        except ValueError:
            errors.append(f"line {index}: source_start/source_end must be integers")
            continue
        if start < 0 or end <= start:
            errors.append(f"line {index}: invalid interval [{start}, {end})")
            continue
        key = (row["dataset_id"], row["subject_id"], row["recording_id"])
        groups[key].append((start, end, split, index))
        subjects_by_split[split].add(row["subject_id"])
        datasets_by_split[split].add(row["dataset_id"])
        source_id = row.get("source_id", "").strip()
        if source_id:
            previous = seen_source.get(source_id)
            if previous and previous[0] != split:
                errors.append(f"line {index}: source_id {source_id!r} also occurs in {previous[0]} at line {previous[1]}")
            else:
                seen_source[source_id] = (split, index)

    for key, intervals in groups.items():
        intervals.sort()
        active: list[tuple[int, int, str, int]] = []
        for current in intervals:
            start, end, split, line = current
            active = [item for item in active if item[1] > start]
            for other_start, other_end, other_split, other_line in active:
                if other_split != split and min(end, other_end) > max(start, other_start):
                    errors.append(
                        f"lines {other_line}/{line}: cross-split overlap for {key}: "
                        f"{other_split}[{other_start},{other_end}) vs {split}[{start},{end})"
                    )
            active.append(current)

    if args.protocol == "cross-subject":
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = subjects_by_split[left] & subjects_by_split[right]
            if overlap:
                errors.append(f"subjects shared by {left}/{right}: {sorted(overlap)}")
    if args.protocol == "cross-dataset":
        overlap = datasets_by_split["train"] & datasets_by_split["test"]
        if overlap:
            errors.append(f"datasets shared by train/test in cross-dataset protocol: {sorted(overlap)}")
    return print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
