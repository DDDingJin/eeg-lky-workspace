#!/usr/bin/env python3
"""Check a benchmark repository for minimum release structure and hazards."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from _common import print_errors


REQUIRED = ("README.md", "LICENSE", "CITATION.cff", "configs", "src", "scripts", "splits", "tests")
BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".h5", ".hdf5", ".mat"}
SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}")
ABSOLUTE_PATTERN = re.compile(r"(?:/Users/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--max-file-mb", type=float, default=20.0)
    args = parser.parse_args()
    root = args.project_root.resolve()
    errors: list[str] = []
    for name in REQUIRED:
        if not (root / name).exists():
            errors.append(f"missing release item: {name}")
    if not any((root / name).exists() for name in ("pyproject.toml", "requirements.txt", "requirements-lock.txt", "environment.yml")):
        errors.append("missing environment or dependency file")

    limit = int(args.max_file_mb * 1024 * 1024)
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root)
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"cannot stat {relative}: {exc}")
            continue
        if size > limit:
            errors.append(f"large file ({size / 1024 / 1024:.1f} MiB): {relative}")
        if path.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"binary model/data artifact in repository: {relative}")
        if size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if SECRET_PATTERN.search(text):
            errors.append(f"possible secret in {relative}")
        if ABSOLUTE_PATTERN.search(text):
            errors.append(f"machine-specific absolute path in {relative}")
    return print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
