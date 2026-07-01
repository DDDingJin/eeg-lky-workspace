---
name: research-audit-loop
description: Coordinate a version-locked Git review loop between an independent reviewer endpoint and an implementation endpoint. Use when publishing audit findings, converting findings into code changes, responding to review issues, verifying fixes, creating review/fix/verification branches, or maintaining review_cycles artifacts without allowing either endpoint to overwrite the other's work.
---

# Research Audit Loop

Use one protocol with two explicit roles:

- `reviewer`: inspect, report, and independently verify; never implement the reviewed fix.
- `implementer`: change code, tests, results, and implementation responses; never rewrite reviewer evidence or verification.

Determine the role before changing files. If the role is not explicit, inspect the requested action and stop before writing if ownership remains ambiguous.

## Load The Protocol

Read:

- `references/protocol.md` before any branch, tag, commit, or PR action.
- `references/artifact-schema.md` before creating or editing a round artifact.

Use `scripts/create_review_round.py` to initialize a reviewer round. Use `scripts/validate_round.py` before every review, fix, or verification commit.

## Experiment Round Completion Gate

For any experiment, benchmark, model expansion, analysis pilot, paper-material update, or result-summary update, read `references/protocol.md#14-experiment-round-completion-gate` before reporting completion.

The short rule is:

- `published_for_review`: committed, pushed, remote commit verified, and reviewable.
- `local_committed_push_blocked`: locally committed, push failed, user help required; not reviewable yet.
- `local_only_incomplete`: local files exist but are not committed; not complete and not reviewable.

Never ask the reviewer to accept, verify, or build the next round from uncommitted or unpushed local-only results.

## Reviewer Workflow

1. Fetch the target branch and record its full 40-character commit SHA.
2. Confirm no other active round owns the same target branch.
3. Create the immutable input tag `audit-input/<round-id>` at the target commit.
4. Create `review/<round-id>` from that exact commit.
5. Run `scripts/create_review_round.py`.
6. Inspect code, tests, results, protocols, and documentation.
7. Explicitly assess benchmark question design, dataset roles, task definitions, baseline grouping fairness, train/test comparability, claim support, and paper narrative consistency.
8. Write findings in `review.md` and structured actions in `issues.json`.
9. Do not edit implementation code on the review branch.
10. Validate with `--phase review --base <target_commit>` and open a PR into the target branch.

After the fix PR is merged:

1. Create `verify/<round-id>` from the target branch.
2. Reproduce the acceptance checks against the merged fix commit.
3. Write `verification.md`.
4. Set each issue to `verified`, `reopened`, or `withdrawn`.
5. Validate with `--phase verification --base <merged_fix_commit>`.
6. Open a verification PR.
7. After that PR is merged, create `audit-verified/<round-id>` at the merge commit.

Only the reviewer may set `verified`, `reopened`, or `withdrawn`.

## Implementer Workflow

1. Read the merged round manifest, review, and issue register.
2. Confirm the round targets the expected branch and commit.
3. If the target branch contains implementation changes after `target_commit`, stop and request reviewer reconciliation before applying fixes.
4. Create `fix/<round-id>` from the target branch after the review PR is merged.
5. For each issue, set `accepted`, `in_progress`, `blocked`, or `disputed` with a reason.
6. Modify implementation code, tests, generated results, and documentation only as required by accepted issues.
7. Record changed files, commits, commands, tests, reruns, and remaining limitations in `implementation_response.md`.
8. Set completed issues to `fixed_pending_verification`.
9. Do not edit `review.md`, reviewer evidence, acceptance checks, or `verification.md`.
10. Validate with `--phase fix --base <fix_base_commit>` and open a PR into the target branch.

Only the implementer may set `accepted`, `in_progress`, `blocked`, `disputed`, or `fixed_pending_verification`.

## Invariants

- Treat commit SHA and annotated tags as version identity; timestamps are navigation aids only.
- Never push review or fix commits directly to the protected target branch.
- Never force-push a shared review, fix, verification, or target branch.
- Keep one active round per target branch unless both roles explicitly approve parallel rounds.
- Preserve original findings. Add responses and verification instead of rewriting history.
- Do not claim an issue is resolved because code changed. Resolution requires reviewer verification.
- Do not merge the target branch into `master` until required issues are verified or explicitly withdrawn.
- Treat benchmark design and unsupported scientific claims as auditable findings, not as optional editorial comments.

## Rule Negotiation

Before the protocol is adopted, treat `rule/*` branches as proposals:

1. The reviewer publishes the proposed Skill on a `rule/*` branch.
2. The implementer creates a separate feedback branch from that rule branch.
3. The implementer proposes edits without overwriting the reviewer branch.
4. Both roles review the diff.
5. Merge the agreed rule through a PR before starting a live audit round.

Do not use production `review/*` or `fix/*` rounds to negotiate the protocol itself.
