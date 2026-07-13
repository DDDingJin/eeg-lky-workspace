# Reviewer And Implementer Collaboration

## Contents

1. Shared rule
2. Reviewer role
3. Implementer role
4. Instruction contract
5. Handoff contract
6. Push failure

## Shared Rule

Use the same Skill package on both endpoints. Determine role before writes. For formal Git review rounds, defer role-owned files and issue transitions to `research-audit-loop`.

Start endpoint-specific work from `assets/reviewer-invocation-template.md` or `assets/implementer-invocation-template.md` when a copyable role prompt is useful.

## Reviewer Role

The reviewer:

1. defines or confirms the active scientific stage;
2. inspects protocol, code, configuration, raw evidence, statistics, and claims;
3. recomputes mean, variance, and other summaries from raw results;
4. balances mainline contribution against token, time, and compute cost;
5. requests only necessary files;
6. checks submodule correctness directly;
7. gives phased, implementable instructions;
8. defines inputs, actions, outputs, acceptance checks, and stop conditions;
9. owns reviewer evidence and verification under the audit loop;
10. records low-contribution work in the backlog.

## Implementer Role

The implementer:

1. uses sourced model implementations;
2. integrates code through shared contracts;
3. runs short preflight, unit, and smoke tests;
4. prepares exact long-run commands for the user;
5. writes incremental state and raw results;
6. uploads necessary code, configs, manifests, raw metrics, and compact logs;
7. does not perform the reviewer's final aggregation unless asked for a diagnostic;
8. reports actual changes, files, commands, and blockers;
9. avoids speculative next-step advice unless a clear problem is visible;
10. does not claim independent verification.

## Instruction Contract

Every reviewer instruction should include:

1. current phase;
2. objective;
3. accepted inputs;
4. ordered steps;
5. files to create or change;
6. commands when determinable;
7. required intermediate artifacts;
8. acceptance checks;
9. stop conditions;
10. prohibited changes;
11. expected handoff.

Do not issue a broad instruction such as “implement cross-subject training” without mapping it to the current code and staged outputs.

## Handoff Contract

Every implementer handoff should include:

1. status;
2. role;
3. phase and domain gate;
4. branch and full commit;
5. base branch and commit;
6. exact commands;
7. resolved config;
8. successful, failed, skipped, reused, and pending jobs;
9. changed files;
10. result and checkpoint-manifest paths;
11. validation and leakage status;
12. relevant logs;
13. open incidents;
14. large-file check;
15. next safe action.

Use `scripts/build_handoff.py` to generate a starting report. Under a formal audit round, include the round ID and audit publication state.

## Push Failure

Attempt push once. If it fails:

1. keep the local commit;
2. record the full commit SHA;
3. record the exact push command;
4. preserve the complete error;
5. report worktree status;
6. set `local_committed_push_blocked`;
7. stop dependent rounds;
8. give the user the exact manual push command.

Do not repeatedly retry, force-push, rewrite remote history, or ask the reviewer to treat an unpushed branch as formal evidence.
