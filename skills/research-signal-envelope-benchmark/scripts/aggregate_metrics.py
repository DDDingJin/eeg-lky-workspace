#!/usr/bin/env python3
"""Aggregate raw metric CSV rows for reviewer-side summaries."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from _common import atomic_write_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--group-by", default="dataset_id,model_id,metric_name")
    parser.add_argument("--value-column", default="metric_value")
    parser.add_argument("--status-column", default="status")
    parser.add_argument("--include-status", action="append", default=["new", "reused", "success"])
    args = parser.parse_args()
    groups = [item.strip() for item in args.group_by.split(",") if item.strip()]

    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        needed = set(groups) | {args.value_column}
        missing = needed - fields
        if missing:
            raise SystemExit(f"missing columns: {', '.join(sorted(missing))}")
        buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for line, row in enumerate(reader, 2):
            status = row.get(args.status_column, "success")
            if status not in args.include_status:
                continue
            try:
                value = float(row[args.value_column])
            except ValueError as exc:
                raise SystemExit(f"line {line}: invalid metric value") from exc
            if not math.isfinite(value):
                raise SystemExit(f"line {line}: non-finite metric value")
            buckets[tuple(row[field] for field in groups)].append(value)

    rows = []
    for key in sorted(buckets):
        values = buckets[key]
        rows.append(
            dict(
                zip(groups, key),
                count=str(len(values)),
                mean=f"{statistics.fmean(values):.12g}",
                std=f"{statistics.stdev(values):.12g}" if len(values) > 1 else "",
                min=f"{min(values):.12g}",
                max=f"{max(values):.12g}",
            )
        )
    fieldnames = groups + ["count", "mean", "std", "min", "max"]
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(args.output_csv, output.getvalue())
    print(f"WROTE: {args.output_csv} ({len(rows)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
