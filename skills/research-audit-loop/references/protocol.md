# Version-Locked Review Protocol

## 1. Long-Lived Branches

- `master`: stable public material.
- `audit/reproduction-note`: protected integration branch for the current audit package.

Do not create permanent date-stamped copies of either branch. Preserve important states with full commit SHAs and annotated tags.

## 2. Review Scope

Review is not limited to implementation debugging. A reviewer must also inspect:

- whether the benchmark question is scientifically coherent
- whether each dataset has an explicit role
- whether task definitions and metrics match the claims
- whether baseline groups are compared fairly
- whether train/validation/test protocols are comparable
- whether reported conclusions are supported by the actual evidence
- whether the paper narrative distinguishes development evidence from article-grade evidence

Record these concerns as structured issues with acceptance checks when they can affect scientific validity or publication claims.

## 3. Temporary Branches

For round `AR-20260624-103000-9bf2f20`, use:

- `review/ar-20260624-103000-9bf2f20`
- `fix/ar-20260624-103000-9bf2f20`
- `verify/ar-20260624-103000-9bf2f20`

Delete temporary branches only after their PRs are merged and the relevant tags exist.

## 4. Tags

- `audit-input/AR-...`: exact implementation commit reviewed.
- `audit-verified/AR-...`: exact commit after verification artifacts are merged.

Use annotated tags. Never move or reuse a published tag.

## 5. Round Directory

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

## 6. Pull Request Sequence

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

## 7. Target Branch Movement

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

## 8. File Ownership

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

## 9. Issue State Machine

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

## 10. Branch Protection

Configure the integration branch to:

- reject force pushes and deletion
- require pull requests
- require current status checks
- dismiss stale approvals after new commits
- require conversation resolution

The Skill cannot substitute for repository branch protection.

## 11. Validation Commands

Run from the repository root:

```text
python skills/research-audit-loop/scripts/validate_round.py <round-dir> --check-git --phase review --base <target-commit>
python skills/research-audit-loop/scripts/validate_round.py <round-dir> --check-git --phase fix --base <fix-base-commit>
python skills/research-audit-loop/scripts/validate_round.py <round-dir> --check-git --phase verification --base <merged-fix-commit>
```

The role diff check is a guardrail, not a substitute for PR review.

## 12. Rule Adoption And Legacy Documents

When this Skill is adopted:

- make it the normative workflow
- replace older informal review-loop instructions with a short pointer to this Skill
- do not maintain two competing branch or status protocols
- preserve old documents only as historical context if clearly marked non-normative

## 13. Promotion To Stable

Promote from `audit/reproduction-note` to `master` only through a separate PR that identifies:

- verified round IDs
- verified tags
- unresolved non-blocking issues
- exact benchmark artifacts being promoted

## 14. Experiment Round Completion Gate

This section applies to every lightweight experiment, benchmark, model expansion,
analysis pilot, paper-material update, and result-summary update, even when the
work is not part of a formal review/fix/verification round.

The implementer must not report such a round as completed unless one of the
states below is reached.

### State A: `published_for_review`

This is the only state in which the implementer may ask the reviewer endpoint
to review the round.

All conditions must be true:

1. The round has a clearly named branch.
2. All intended code, configs, compact result artifacts, reports, and
   paper-material notes are committed locally.
3. The branch is pushed to `origin`.
4. The implementer verifies that the remote branch exists and that the remote
   commit matches the local commit.
5. The local worktree is clean except ignored cache files.
6. The completion report includes:
   - branch name;
   - full commit SHA;
   - base branch and base commit;
   - changed-file summary;
   - result directory;
   - commands actually run;
   - validation commands and validation status;
   - large-file, prediction-dump, checkpoint, and model-weight check;
   - skipped or failed items with reasons.

### State B: `local_committed_push_blocked`

Use this state if local work is complete and committed, but pushing to GitHub
fails.

The implementer must:

1. Commit all intended files locally.
2. Record the local commit SHA.
3. Record the exact push command attempted.
4. Record the complete push error.
5. Check and report local worktree status.
6. Stop the next experiment round and request manual user help to publish the
   branch.

The completion report must explicitly say:

```text
Status: local_committed_push_blocked
This round has been committed locally but has not been pushed to GitHub.
Because the reviewer endpoint cannot read an unpushed local branch, this round
is not yet reviewable.
Manual user help is required to publish the branch.
```

In this state, the implementer may report a local result summary, but must not
claim that the round is reviewable. The implementer must not continue to the next
experiment round unless the user explicitly approves local-only continuation and
the risk is recorded.

### State C: `local_only_incomplete`

Use this state if result files exist only in the local filesystem and have not
been committed.

In this state, the implementer may report preliminary local observations, but
must not:

1. claim that the round is completed;
2. ask the reviewer for formal review;
3. ask the reviewer to accept, verify, or register the result;
4. ask the reviewer to build the next round from the result;
5. continue to the next experiment round.

The completion report must explicitly say:

```text
Status: local_only_incomplete
This round still exists only as uncommitted local files.
This round is not complete and is not reviewable.
The next action is to commit the intended files. If push then fails, request
manual user help under local_committed_push_blocked.
```

### Reviewer Rule

The reviewer must treat uncommitted or unpushed results as not formally
reviewable. The reviewer may give preliminary comments on a local summary, but
must not mark such a round as accepted, verified, or the baseline for the next
round.

Formal review requires a GitHub-accessible branch and commit.

### Completion Report Template

Every experiment-style round must end with this report:

```text
Status:
published_for_review / local_committed_push_blocked / local_only_incomplete

Branch:
...

Commit:
...

Base branch:
...

Base commit:
...

Result directory:
...

Changed files summary:
...

Commands run:
...

Validation:
...

Large-file check:
prediction dump: yes/no
checkpoint: yes/no
.pt/.pth/.npy/.npz/.h5/.mat: yes/no

Skipped or failed items:
...

Reviewer entry:
GitHub branch/commit/PR link if available
```

### 中文说明

“本地跑完”不等于“本轮完成”。

“本地有结果”不等于“审阅端可审阅”。

“push 失败”不是错误，但必须进入 `local_committed_push_blocked` 状态，并请求用户手动帮助，而不是直接继续下一轮。
