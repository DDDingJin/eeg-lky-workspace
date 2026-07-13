#!/usr/bin/env python3
"""Initialize persistent benchmark workflow state in a project repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import atomic_write_json, atomic_write_text, git_identity, now_iso


def write_new(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        print(f"SKIP: {path} already exists")
        return
    atomic_write_text(path, text)
    print(f"WRITE: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--role", choices=("reviewer", "implementer", "standalone"), default="standalone"
    )
    parser.add_argument("--force", action="store_true", help="Replace existing workflow files")
    args = parser.parse_args()

    root = args.project_root.resolve()
    workflow = root / "workflow"
    knowledge = workflow / "knowledge"
    manifests = workflow / "manifests"
    reports = workflow / "reports"
    for directory in (knowledge, manifests, reports):
        directory.mkdir(parents=True, exist_ok=True)

    branch, commit = git_identity(root)
    state_path = workflow / "state.json"
    state = {
        "schema_version": 1,
        "study_id": args.study_id,
        "phase": "P1",
        "active_modules": ["C0", "C1", "C2", "C3", "C5", "P1"],
        "role": args.role,
        "branch": branch,
        "commit": commit,
        "config_id": None,
        "dataset_version": None,
        "split_version": None,
        "last_gate": "not_evaluated",
        "planned_jobs": [],
        "completed_jobs": [],
        "failed_jobs": [],
        "pending_jobs": [],
        "resume_enabled": False,
        "last_update": now_iso(),
    }
    if not state_path.exists() or args.force:
        atomic_write_json(state_path, state)
        print(f"WRITE: {state_path}")
    else:
        print(f"SKIP: {state_path} already exists")

    start_here = f"""# START HERE

- Study: `{args.study_id}`
- Role: `{args.role}`
- Phase: `P1`
- Branch: `{branch or 'unborn-or-unavailable'}`
- Commit: `{commit or 'unborn-or-unavailable'}`
- Formal config: not selected
- Data version: not selected
- Split version: not selected
- Last gate: `not_evaluated`
- Open incidents: none recorded
- Next safe action: define the research question and experiment matrix
- Updated: {state['last_update']}
"""
    write_new(workflow / "START_HERE.md", start_here, args.force)
    write_new(workflow / "decisions.md", "# Decisions\n\n", args.force)
    write_new(workflow / "backlog.md", "# Backlog\n\n", args.force)
    write_new(knowledge / "incidents.jsonl", "", args.force)
    write_new(knowledge / "error-patterns.md", "# Error Patterns\n\n", args.force)
    write_new(knowledge / "learned-rules.md", "# Learned Rules\n\n", args.force)
    write_new(knowledge / "review-queue.md", "# Review Queue\n\n", args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
