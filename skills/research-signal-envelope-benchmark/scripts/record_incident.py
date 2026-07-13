#!/usr/bin/env python3
"""Append a structured incident while detecting recurring signatures."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from _common import atomic_write_text, now_iso


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{number}: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--symptom", required=True)
    parser.add_argument("--root-cause", default="unknown")
    parser.add_argument("--fix", default="not fixed")
    parser.add_argument("--verification", default="not verified")
    parser.add_argument("--prevention", default="not defined")
    parser.add_argument("--command", default="")
    parser.add_argument("--job-key", default="")
    parser.add_argument("--environment", default="")
    parser.add_argument("--affected-version", action="append", default=[])
    parser.add_argument("--scope", choices=("project", "transferable"), default="project")
    parser.add_argument(
        "--status",
        choices=("new", "reproduced", "diagnosed", "fixed", "verified", "archived"),
        default="new",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    path = root / "workflow" / "knowledge" / "incidents.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(path)
    matches = [r for r in records if r.get("signature") == args.signature]
    incident_id = f"INC-{now_iso()[:10].replace('-', '')}-{uuid.uuid4().hex[:8]}"
    record = {
        "incident_id": incident_id,
        "date": now_iso(),
        "signature": args.signature,
        "stage": args.stage,
        "category": args.category,
        "environment": args.environment,
        "command": args.command,
        "job_key": args.job_key,
        "symptom": args.symptom,
        "root_cause": args.root_cause,
        "fix": args.fix,
        "verification": args.verification,
        "prevention": args.prevention,
        "affected_versions": args.affected_version,
        "recurrence_count": len(matches) + 1,
        "duplicate_of": matches[0].get("incident_id") if matches else None,
        "scope": args.scope,
        "status": args.status,
    }
    records.append(record)
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    atomic_write_text(path, text)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if record["recurrence_count"] >= 3:
        print("PROMOTION_CANDIDATE: recurrence threshold reached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
