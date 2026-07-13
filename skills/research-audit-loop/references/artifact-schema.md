# Round Artifact Schema

## Contents

1. Manifest and optional domain gate
2. Issues
3. Review Markdown
4. Implementation response
5. Verification

Use JSON for machine-readable artifacts so both endpoints can validate them with the Python standard library.

## Manifest

Required fields:

```json
{
  "protocol_version": 1,
  "round_id": "AR-20260624-103000-9bf2f20",
  "created_at": "2026-06-24T10:30:00+08:00",
  "target_repository": "DDDingJin/eeg-lky-workspace",
  "target_branch": "audit/reproduction-note",
  "target_commit": "9bf2f200885346914fe6de33b3b3e6ffcf82b834",
  "input_tag": "audit-input/AR-20260624-103000-9bf2f20",
  "review_branch": "review/ar-20260624-103000-9bf2f20",
  "review_commit": null,
  "review_pr": null,
  "fix_branch": "fix/ar-20260624-103000-9bf2f20",
  "fix_base_commit": null,
  "fix_commit": null,
  "fix_pr": null,
  "verification_branch": "verify/ar-20260624-103000-9bf2f20",
  "verification_commit": null,
  "verification_pr": null,
  "verified_tag": "audit-verified/AR-20260624-103000-9bf2f20",
  "domain_gate": null,
  "status": "review_open"
}
```

For an experiment-style round, replace `domain_gate: null` with:

```json
{
  "skill": "research-signal-envelope-benchmark",
  "study_id": "study-id",
  "run_id": "run-id",
  "gate_status": "approved_as_smoke_only",
  "evidence_path": "workflow/reports/domain_gate.json",
  "evidence_commit": "9bf2f200885346914fe6de33b3b3e6ffcf82b834"
}
```

Allowed domain gate statuses:

- `not_applicable`
- `design_only`
- `approved_as_smoke_only`
- `engineering_closure_required`
- `approved_for_full_run`
- `rejected_until_reproduced_cleanly`

The implementer may produce or update domain evidence on an implementation
branch. Only the reviewer may independently accept that evidence for issue
verification or result promotion.

Allowed round statuses:

- `review_open`
- `review_merged`
- `implementation_in_progress`
- `fix_pending_verification`
- `verification_in_progress`
- `reopened`
- `verified`
- `superseded`

## Issues

Keep stable issue IDs. Do not renumber an issue after publication.

```json
{
  "protocol_version": 1,
  "round_id": "AR-20260624-103000-9bf2f20",
  "issues": [
    {
      "id": "AUDIT-20260624-001",
      "severity": "blocker",
      "category": "implementation_bug",
      "status": "open",
      "title": "Short falsifiable finding",
      "evidence": [
        {
          "path": "src/example.py",
          "lines": "20-28",
          "observation": "What was observed without assuming the fix."
        }
      ],
      "risk": "Why this can invalidate or weaken a result.",
      "required_actions": [
        "Required implementation action."
      ],
      "acceptance_checks": [
        "Observable condition the reviewer can reproduce."
      ],
      "reviewer": {
        "created_at": "2026-06-24T10:30:00+08:00",
        "notes": ""
      },
      "implementation": {
        "decision": null,
        "response": null,
        "commits": [],
        "tests": []
      },
      "verification": {
        "result": null,
        "verified_commit": null,
        "notes": null
      }
    }
  ]
}
```

Allowed severities:

- `blocker`
- `major`
- `minor`

Allowed categories:

- `benchmark_design`
- `dataset_scope`
- `task_definition`
- `implementation_bug`
- `data_leakage`
- `protocol_mismatch`
- `metric_mismatch`
- `comparison_fairness`
- `claim_support`
- `reproducibility`
- `statistical_analysis`
- `reporting`
- `new_experiment`
- `other`

Allowed issue statuses:

- `open`
- `accepted`
- `in_progress`
- `blocked`
- `disputed`
- `fixed_pending_verification`
- `verified`
- `reopened`
- `withdrawn`

## Review Markdown

Write findings in severity order. Each finding must include:

- issue ID
- affected files or artifacts
- observed evidence
- scientific or engineering risk
- required action
- acceptance checks

Avoid prescribing a specific implementation when multiple valid fixes exist.

## Implementation Response

For each issue, record:

- decision: accepted, disputed, or blocked
- exact changed files
- fix commits
- tests and commands
- regenerated outputs
- deviations from the requested action
- remaining limitations

Do not claim verification.

## Verification

For each fixed issue, record:

- fix commit reviewed
- acceptance checks performed
- command or artifact evidence
- result: verified or reopened
- residual risk

Verification must evaluate the merged implementation, not an uncommitted local state.
