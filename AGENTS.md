# Repository Agent Instructions

For review, fix, response, or verification work:

1. Read `skills/research-audit-loop/SKILL.md`.
2. Declare the active role as `reviewer` or `implementer`.
3. Follow the role-owned files and issue status transitions.
4. Bind every review round to a full commit SHA.
5. Do not push implementation or review changes directly to the protected target branch.
6. For experiment/model/result/paper-material rounds, apply the Experiment Round Completion Gate in `skills/research-audit-loop/references/protocol.md`: local-only results are not reviewable, and push failures must be reported as `local_committed_push_blocked` with user help requested.

On a `rule/*` branch, treat the protocol as a proposal. Propose changes from a separate feedback branch instead of overwriting the published rule branch.
