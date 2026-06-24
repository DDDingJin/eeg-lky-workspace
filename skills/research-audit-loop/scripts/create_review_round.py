#!/usr/bin/env python3
"""Create a version-locked review round from bundled templates."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TEMPLATE_NAMES = (
    "manifest.json",
    "issues.json",
    "review.md",
    "implementation_response.md",
    "verification.md",
)
TERMINAL_ROUND_STATUSES = {"verified", "superseded"}


def git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def replace_tokens(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(f"__{key}__", value)
    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    if unresolved:
        raise ValueError(f"unresolved template tokens: {', '.join(unresolved)}")
    return text


def active_rounds(repo_root: Path, target_branch: str) -> list[str]:
    active: list[str] = []
    cycles_dir = repo_root / "review_cycles"
    if not cycles_dir.exists():
        return active
    for manifest_path in cycles_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("target_branch") == target_branch
            and manifest.get("status") not in TERMINAL_ROUND_STATUSES
        ):
            active.append(str(manifest.get("round_id") or manifest_path.parent.name))
    return sorted(active)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-repository", required=True)
    parser.add_argument("--target-branch", default="audit/reproduction-note")
    parser.add_argument("--target-commit", help="Commit or ref to lock; defaults to target branch.")
    parser.add_argument("--round-id")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--allow-parallel", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    existing_rounds = active_rounds(repo_root, args.target_branch)
    if existing_rounds and not args.allow_parallel:
        raise RuntimeError(
            "active round already targets "
            f"{args.target_branch}: {', '.join(existing_rounds)}; "
            "close it or pass --allow-parallel after both roles approve"
        )
    target_ref = args.target_commit or args.target_branch
    target_commit = git(repo_root, "rev-parse", f"{target_ref}^{{commit}}").lower()
    if not SHA_RE.fullmatch(target_commit):
        raise ValueError(f"expected a full commit SHA, got {target_commit!r}")

    now = datetime.now(ZoneInfo(args.timezone)).replace(microsecond=0)
    short_sha = target_commit[:7]
    round_id = args.round_id or f"AR-{now:%Y%m%d-%H%M%S}-{short_sha}"
    if not re.fullmatch(r"AR-[A-Za-z0-9-]+", round_id):
        raise ValueError("round ID must start with AR- and contain only letters, numbers, and hyphens")

    offset = now.strftime("%z")
    folder_time = f"{now:%Y-%m-%d_%H%M%S}{offset}"
    round_dir = repo_root / "review_cycles" / f"{folder_time}_{round_id}"
    if round_dir.exists():
        raise FileExistsError(round_dir)
    round_dir.mkdir(parents=True)

    values = {
        "ROUND_ID": round_id,
        "ROUND_ID_LOWER": round_id.lower(),
        "CREATED_AT": now.isoformat(),
        "TARGET_REPOSITORY": args.target_repository,
        "TARGET_BRANCH": args.target_branch,
        "TARGET_COMMIT": target_commit,
    }
    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    for name in TEMPLATE_NAMES:
        template = (assets_dir / name).read_text(encoding="utf-8")
        output = replace_tokens(template, values)
        (round_dir / name).write_text(output, encoding="utf-8")

    print(round_dir)
    print(f"round_id={round_id}")
    print(f"target_commit={target_commit}")
    print(f"review_branch=review/{round_id.lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
