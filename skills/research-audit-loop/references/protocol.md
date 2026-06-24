# Version-Locked Review Protocol

## 1. Long-Lived Branches

- `master`: stable public material.
- `audit/reproduction-note`: protected integration branch for the current audit package.

Do not create permanent date-stamped copies of either branch. Preserve important states with full commit SHAs and annotated tags.

## 2. Temporary Branches

For round `AR-20260624-103000-9bf2f20`, use:

- `review/ar-20260624-103000-9bf2f20`
- `fix/ar-20260624-103000-9bf2f20`
- `verify/ar-20260624-103000-9bf2f20`

Delete temporary branches only after their PRs are merged and the relevant tags exist.

## 3. Tags

- `audit-input/AR-...`: exact implementation commit reviewed.
- `audit-verified/AR-...`: exact commit after verification artifacts are merged.

Use annotated tags. Never move or reuse a published tag.

## 4. Round Directory

Use:

```text
review_cycles/
└── 2026-06-24_103000+0800_AR-20260624-103000-9bf2f20/
    ├── manifest.json
    ├── review.md
    ├── issues.json
    ├── implementation_response.md
    └── verification.md
```

Every Markdown artifact must repeat:

- round ID
- target branch
- target commit
- the upstream review or fix commit when applicable
- creation or update time with UTC offset

The timestamp helps people find the latest round. The commit SHA proves which version the artifact describes.

## 5. Pull Request Sequence

### PR 1: Review

```text
review/<round-id> -> audit/reproduction-note
```

Contains review artifacts only. It must not change implementation code or generated benchmark results.

### PR 2: Fix

```text
fix/<round-id> -> audit/reproduction-note
```

Starts after PR 1 is merged. Contains implementation changes, tests, regenerated artifacts, issue status updates, and `implementation_response.md`.

### PR 3: Verification

```text
verify/<round-id> -> audit/reproduction-note
```

Starts after PR 2 is merged. Contains reviewer verification, final reviewer-owned statuses, and no new implementation fix.

If verification finds a failure, mark the issue `reopened`. Start another fix branch for the same round or a successor round; do not silently repair it on the verification branch.

## 6. Target Branch Movement

Prefer freezing implementation changes on the target branch from input tagging until the fix PR is merged.

If the target branch advances:

1. Compare `target_commit..current_target`.
2. If only review artifacts changed, continue.
3. If implementation, data processing, model, metric, test, or result files changed, stop.
4. Ask the reviewer to classify the round as:
   - still applicable,
   - updated with an amended review, or
   - superseded by a new round.

Never apply findings to "the latest version" without this reconciliation.

## 7. File Ownership

Reviewer-owned:

- `review.md`
- issue identity, evidence, risk, required actions, and acceptance checks
- `verification.md`
- final statuses `verified`, `reopened`, and `withdrawn`
- input and verified tags

Implementer-owned:

- implementation code and tests
- regenerated result artifacts
- `implementation_response.md`
- implementation decision, response, commits, and test records
- statuses `accepted`, `in_progress`, `blocked`, `disputed`, and `fixed_pending_verification`

Shared but field-owned:

- `manifest.json`
- `issues.json`

Neither role may rewrite the other role's historical content. Correct mistakes by appending an amendment with author, time, and reason.

## 8. Issue State Machine

```text
open
  -> accepted -> in_progress -> fixed_pending_verification
  -> disputed
  -> withdrawn

fixed_pending_verification
  -> verified
  -> reopened -> accepted

accepted | in_progress
  -> blocked -> accepted

disputed
  -> open
  -> withdrawn
```

Reviewer transitions:

- create `open`
- `disputed -> open`
- any reviewer-owned issue -> `withdrawn`
- `fixed_pending_verification -> verified`
- `fixed_pending_verification -> reopened`

Implementer transitions:

- `open -> accepted`
- `open -> disputed`
- `accepted -> in_progress`
- `accepted|in_progress -> blocked`
- `blocked -> accepted`
- `accepted|in_progress -> fixed_pending_verification`

## 9. Branch Protection

Configure the integration branch to:

- reject force pushes and deletion
- require pull requests
- require current status checks
- dismiss stale approvals after new commits
- require conversation resolution

The Skill cannot substitute for repository branch protection.

## 10. Validation Commands

Run from the repository root:

```text
python skills/research-audit-loop/scripts/validate_round.py <round-dir> --check-git --phase review --base <target-commit>
python skills/research-audit-loop/scripts/validate_round.py <round-dir> --check-git --phase fix --base <fix-base-commit>
python skills/research-audit-loop/scripts/validate_round.py <round-dir> --check-git --phase verification --base <merged-fix-commit>
```

The role diff check is a guardrail, not a substitute for PR review.

## 11. Promotion To Stable

Promote from `audit/reproduction-note` to `master` only through a separate PR that identifies:

- verified round IDs
- verified tags
- unresolved non-blocking issues
- exact benchmark artifacts being promoted
