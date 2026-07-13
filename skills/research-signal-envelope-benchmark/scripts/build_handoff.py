#!/usr/bin/env python3
"""Build a concise reviewer/implementer handoff from workflow state."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import atomic_write_text, git_identity, load_json, now_iso


def render_jobs(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "none"
    values = []
    for item in items:
        values.append(item if isinstance(item, str) else str(item.get("job_key", item)))
    return ", ".join(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--status", default="local_only_incomplete")
    parser.add_argument("--base-branch", default="unknown")
    parser.add_argument("--base-commit", default="unknown")
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--result-dir", default="unknown")
    parser.add_argument("--next-action", default="inspect and update workflow/START_HERE.md")
    args = parser.parse_args()

    root = args.project_root.resolve()
    state = load_json(root / "workflow" / "state.json")
    branch, commit = git_identity(root)
    commands = "\n".join(f"- `{command}`" for command in args.command) or "- none recorded"
    current_branch = branch or state.get("branch") or "unknown"
    current_commit = commit or state.get("commit") or "unknown"
    text = f"""# Benchmark Handoff

- Status: `{args.status}`
- Role: `{state.get('role')}`
- Phase: `{state.get('phase')}`
- Domain gate: `{state.get('last_gate')}`
- Branch: `{current_branch}`
- Commit: `{current_commit}`
- Base branch: `{args.base_branch}`
- Base commit: `{args.base_commit}`
- Config: `{state.get('config_id') or 'unknown'}`
- Data version: `{state.get('dataset_version') or 'unknown'}`
- Split version: `{state.get('split_version') or 'unknown'}`
- Result directory: `{args.result_dir}`
- Updated: {now_iso()}

## Commands

{commands}

## Jobs

- Completed: {render_jobs(state.get('completed_jobs'))}
- Failed: {render_jobs(state.get('failed_jobs'))}
- Pending: {render_jobs(state.get('pending_jobs'))}

## Required Checks

- Schema validation: record status
- Leakage validation: record status
- Checkpoint load: record status
- Large-file check: record status
- Open incidents: inspect `workflow/knowledge/incidents.jsonl`

## Next Safe Action

{args.next_action}
"""
    atomic_write_text(args.output, text)
    print(f"WROTE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
