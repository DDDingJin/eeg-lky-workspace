# Research Audit Loop Rule Proposal

Status: reviewer revision after informal implementer feedback; implementer acceptance still required

Proposal branch: `rule/research-audit-loop-v1`

Based on branch: `audit/reproduction-note`

Based on commit: `9bf2f200885346914fe6de33b3b3e6ffcf82b834`

Created: 2026-06-24

Latest reviewer revision considered:

- implementer-side branch: `audit/reproduction-note`
- implementer-side commit: `a43831b83598c82520be320c21b56e92b73b7dcd`
- relevant file: `docs/GITHUB_REVIEW_LOOP.md`

## Purpose

This branch proposes a shared Skill for two independent AI endpoints:

- reviewer endpoint: inspect and verify
- implementer endpoint: change code and produce evidence

The proposal prevents version drift and cross-role overwrites by requiring:

- full target commit SHAs
- immutable input and verified tags
- separate review, fix, and verification branches
- PR-only integration
- field-level artifact ownership
- reviewer verification before closure

## Proposal Files

- `AGENTS.md`
- `skills/research-audit-loop/SKILL.md`
- `skills/research-audit-loop/references/protocol.md`
- `skills/research-audit-loop/references/artifact-schema.md`
- `skills/research-audit-loop/scripts/`
- `skills/research-audit-loop/assets/`

## Implementer Feedback Procedure

Do not modify `rule/research-audit-loop-v1` directly.

Create a feedback branch from this exact rule commit:

```text
rule/research-audit-loop-v1-implementer-feedback
```

On that branch:

1. review the Skill from the implementer perspective
2. propose concrete edits
3. explain any workflow that cannot be executed in the implementer's environment
4. run the Skill validator and script tests
5. open a PR back to `rule/research-audit-loop-v1`

The production loop must not start until both endpoints accept the rule diff and the agreed rule is merged into the repository's integration branch.

See `docs/IMPLEMENTER_FEEDBACK_ASSESSMENT.md` for the reviewer's point-by-point decision.
