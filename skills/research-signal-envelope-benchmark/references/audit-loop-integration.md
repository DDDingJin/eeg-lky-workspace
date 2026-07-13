# Audit Loop Integration

## Contents

1. Double-gate purpose
2. Authority split
3. Activation
4. Domain gate record
5. Reviewer behavior
6. Implementer behavior
7. Conflict and blocking rules

## Double-Gate Purpose

Use two independent gates without duplicating rules:

1. this Skill decides whether the benchmark is scientifically and operationally valid;
2. `research-audit-loop` decides whether roles, version identity, evidence ownership, Git state, and independent verification are valid.

Both must pass before article-grade results are verified or promoted.

## Authority Split

This Skill owns:

- research question and experiment matrix;
- model and dataset contracts;
- target, lag, window, prediction, metric, and aggregation definitions;
- leakage and fairness;
- smoke, closure, full-run, resume, and result-schema gates;
- statistical and claim evidence;
- reproducible release content;
- benchmark-specific error learning.

`research-audit-loop` owns:

- reviewer and implementer role boundaries;
- target branch and full commit identity;
- input and verified tags;
- review, fix, and verification branches and PR order;
- reviewer and implementer file ownership;
- issue IDs, statuses, and allowed transitions;
- branch-register ownership;
- published, push-blocked, and local-only completion states;
- independent final verification.

## Activation

Apply both Skills when a task includes any of:

- formal review findings;
- implementation response;
- verification;
- review, fix, or verify branches;
- audit tags;
- branch-register updates;
- PR publication;
- promotion of benchmark results to a protected branch.

For standalone design or local diagnostics without a formal review round, use this Skill alone while preserving state and evidence.

## Domain Gate Record

For experiment-style audit rounds, add a domain-gate object to the audit manifest:

```json
{
  "skill": "research-signal-envelope-benchmark",
  "study_id": "study-id",
  "run_id": "run-id",
  "gate_status": "approved_as_smoke_only",
  "evidence_path": "workflow/reports/domain_gate.json",
  "evidence_commit": "full-40-character-sha"
}
```

Use one of:

- `not_applicable`;
- `design_only`;
- `approved_as_smoke_only`;
- `engineering_closure_required`;
- `approved_for_full_run`;
- `rejected_until_reproduced_cleanly`.

The evidence commit must match the version being reviewed or be an explicitly identified ancestor whose rules remain unchanged.

## Reviewer Behavior

1. Lock the reviewed commit through the audit loop.
2. Load the domain gate and its evidence.
3. Inspect code and artifacts required by the active benchmark phase.
4. Express scientific failures as structured audit issues when they affect validity or claims.
5. Do not repair implementation code on the review or verification branch.
6. Do not mark a benchmark issue verified when the domain acceptance check fails.
7. Record the final verified commit and tag only after both gates pass.

## Implementer Behavior

1. Read the accepted audit issues and domain evidence.
2. Modify only implementer-owned code, tests, configs, and results.
3. Re-run the smallest required domain gate after a fix.
4. Record commands, results, incidents, and remaining limits.
5. Set issues only to implementer-owned states.
6. Do not self-approve the domain gate as independent verification.
7. Publish the branch and verify the remote commit before requesting formal review.

## Conflict And Blocking Rules

1. If the domain gate blocks, do not proceed because Git state is clean.
2. If the audit loop blocks, do not proceed because scientific checks passed locally.
3. If role ownership conflicts with a project instruction, stop and reconcile the protocol.
4. If the reviewed branch moves in implementation, data, metrics, or results, follow audit-loop target reconciliation and re-evaluate affected domain gates.
5. Do not maintain two competing branch, issue, or publication state machines.
6. Negotiate changes to either live protocol through `rule/*` proposals before adoption.
