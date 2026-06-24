#!/usr/bin/env python3
"""Validate review-round artifacts and their version references."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ROUND_STATUSES = {
    "review_open",
    "review_merged",
    "implementation_in_progress",
    "fix_pending_verification",
    "verification_in_progress",
    "reopened",
    "verified",
    "superseded",
}
ISSUE_STATUSES = {
    "open",
    "accepted",
    "in_progress",
    "blocked",
    "disputed",
    "fixed_pending_verification",
    "verified",
    "reopened",
    "withdrawn",
}
SEVERITIES = {"blocker", "major", "minor"}
CATEGORIES = {
    "benchmark_design",
    "dataset_scope",
    "task_definition",
    "implementation_bug",
    "data_leakage",
    "protocol_mismatch",
    "metric_mismatch",
    "comparison_fairness",
    "claim_support",
    "reproducibility",
    "statistical_analysis",
    "reporting",
    "new_experiment",
    "other",
}
REQUIRED_FILES = {
    "manifest.json",
    "issues.json",
    "review.md",
    "implementation_response.md",
    "verification.md",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def git_commit_exists(repo_root: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
    )
    return result.returncode == 0


def changed_paths(repo_root: Path, base: str) -> set[str]:
    diff = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", base, "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in (diff.stdout + "\n" + untracked.stdout).splitlines()
        if line.strip()
    }


def validate_role_diff(
    errors: list[str],
    repo_root: Path,
    round_dir: Path,
    phase: str,
    base: str,
) -> None:
    try:
        paths = changed_paths(repo_root, base)
        round_rel = round_dir.relative_to(repo_root).as_posix()
    except (subprocess.CalledProcessError, ValueError) as exc:
        fail(errors, f"cannot validate role diff: {exc}")
        return

    if phase == "review":
        outside = sorted(path for path in paths if not path.startswith(f"{round_rel}/"))
        if outside:
            fail(errors, f"review phase changes files outside its round directory: {', '.join(outside)}")
    elif phase == "fix":
        forbidden = sorted(
            path
            for path in paths
            if path.startswith("review_cycles/")
            and (path.endswith("/review.md") or path.endswith("/verification.md"))
        )
        if forbidden:
            fail(errors, f"fix phase changes reviewer-owned files: {', '.join(forbidden)}")
    elif phase == "verification":
        allowed = {
            f"{round_rel}/manifest.json",
            f"{round_rel}/issues.json",
            f"{round_rel}/verification.md",
        }
        outside = sorted(path for path in paths if path not in allowed)
        if outside:
            fail(errors, f"verification phase changes non-reviewer files: {', '.join(outside)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--check-git", action="store_true")
    parser.add_argument("--phase", choices=("review", "fix", "verification"))
    parser.add_argument("--base", help="Base commit used for role-owned diff validation.")
    args = parser.parse_args()

    round_dir = args.round_dir.resolve()
    repo_root = (args.repo_root or round_dir.parents[1]).resolve()
    errors: list[str] = []

    missing = sorted(name for name in REQUIRED_FILES if not (round_dir / name).is_file())
    if missing:
        fail(errors, f"missing files: {', '.join(missing)}")

    try:
        manifest = json.loads((round_dir / "manifest.json").read_text(encoding="utf-8"))
        issues_doc = json.loads((round_dir / "issues.json").read_text(encoding="utf-8"))
    except Exception as exc:
        fail(errors, f"cannot parse JSON: {exc}")
        manifest = {}
        issues_doc = {}

    required_manifest = {
        "protocol_version",
        "round_id",
        "created_at",
        "target_repository",
        "target_branch",
        "target_commit",
        "input_tag",
        "review_branch",
        "fix_branch",
        "verification_branch",
        "verified_tag",
        "status",
    }
    if isinstance(manifest, dict):
        missing_keys = sorted(required_manifest - set(manifest))
        if missing_keys:
            fail(errors, f"manifest missing keys: {', '.join(missing_keys)}")
        target_commit = str(manifest.get("target_commit", ""))
        round_id = str(manifest.get("round_id", ""))
        if not SHA_RE.fullmatch(target_commit):
            fail(errors, "manifest target_commit must be a full lowercase 40-character SHA")
        if manifest.get("status") not in ROUND_STATUSES:
            fail(errors, f"invalid round status: {manifest.get('status')!r}")
        if round_id and round_id not in round_dir.name:
            fail(errors, "round directory name does not include manifest round_id")
    else:
        fail(errors, "manifest must be a mapping")
        target_commit = ""
        round_id = ""

    if isinstance(issues_doc, dict):
        if issues_doc.get("round_id") != round_id:
            fail(errors, "issues.json round_id does not match manifest")
        issues = issues_doc.get("issues")
        if not isinstance(issues, list):
            fail(errors, "issues.json issues must be a list")
            issues = []
        seen_ids: set[str] = set()
        for index, issue in enumerate(issues):
            prefix = f"issue[{index}]"
            if not isinstance(issue, dict):
                fail(errors, f"{prefix} must be a mapping")
                continue
            issue_id = issue.get("id")
            if not issue_id:
                fail(errors, f"{prefix} has no id")
            elif issue_id in seen_ids:
                fail(errors, f"duplicate issue id: {issue_id}")
            else:
                seen_ids.add(issue_id)
            if issue.get("severity") not in SEVERITIES:
                fail(errors, f"{prefix} has invalid severity")
            if issue.get("category") not in CATEGORIES:
                fail(errors, f"{prefix} has invalid category")
            if issue.get("status") not in ISSUE_STATUSES:
                fail(errors, f"{prefix} has invalid status")
            for key in ("title", "evidence", "risk", "required_actions", "acceptance_checks"):
                if not issue.get(key):
                    fail(errors, f"{prefix} missing {key}")
    else:
        fail(errors, "issues.json must be a mapping")

    for name in ("review.md", "implementation_response.md", "verification.md"):
        path = round_dir / name
        if path.is_file() and target_commit:
            content = path.read_text(encoding="utf-8")
            if target_commit not in content:
                fail(errors, f"{name} does not repeat target commit {target_commit}")

    if args.check_git and target_commit and not git_commit_exists(repo_root, target_commit):
        fail(errors, f"target commit does not exist in {repo_root}")
    if bool(args.phase) != bool(args.base):
        fail(errors, "--phase and --base must be provided together")
    elif args.phase and args.base:
        validate_role_diff(errors, repo_root, round_dir, args.phase, args.base)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"OK: {round_dir}")
    print(f"round_id={round_id}")
    print(f"target_commit={target_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
