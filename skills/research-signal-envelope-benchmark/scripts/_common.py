#!/usr/bin/env python3
"""Shared standard-library helpers for benchmark workflow scripts."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def missing_fields(data: dict[str, Any], fields: Iterable[str]) -> list[str]:
    return [field for field in fields if field not in data]


def git_identity(root: Path) -> tuple[str | None, str | None]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else None

    return run("branch", "--show-current"), run("rev-parse", "HEAD")


def print_errors(errors: list[str]) -> int:
    if not errors:
        print("OK")
        return 0
    for error in errors:
        print(f"ERROR: {error}")
    return 1
