# Learning And Error Memory

## Contents

1. Purpose
2. Signal types
3. Session lifecycle
4. Incident workflow
5. Deduplication
6. Promotion
7. Compaction and review
8. Skill evolution

## Purpose

Preserve enough state that a new session does not repeat an already diagnosed error. Keep project-specific experience in the project and promote only validated transferable rules into the Skill.

## Signal Types

- `ERR`: command, code, data, runtime, or evidence failure;
- `LRN`: confirmed learning;
- `FEAT`: requested capability;
- `PREF`: stable user or project preference;
- `PATTERN`: recurring error or successful method.

## Session Lifecycle

### Start

1. Read `workflow/START_HERE.md`.
2. Validate `workflow/state.json` against Git and artifacts.
3. Read `workflow/knowledge/error-patterns.md`.
4. Read open incidents and review candidates relevant to the task.
5. Load only patterns matching the current project, platform, phase, or command.

### During Work

1. Capture failures, user corrections, new requirements, and stable preferences.
2. Search for an existing semantic or category match.
3. Increase recurrence count instead of duplicating a pattern.
4. Keep raw evidence in the incident and the reusable rule concise.

### Handoff

1. Save new incidents.
2. Update open and closed statuses.
3. Update active patterns.
4. Update state and next safe action.
5. Add promotion candidates to the review queue.

## Incident Workflow

```text
new → reproduced → diagnosed → fixed → verified → pattern_or_archived
```

On failure:

1. preserve exact traceback and failing line;
2. preserve command, working directory, environment, branch, commit, and config;
3. identify the job and artifacts affected;
4. classify the failure as scientific, data, implementation, runtime, state, artifact, Git, or environment;
5. search known patterns;
6. reproduce the smallest failing case;
7. identify root cause without conflating symptoms;
8. apply a bounded fix;
9. verify the original failure is resolved;
10. run the affected contract check, regression test, or smoke;
11. record remaining risk;
12. update state before continuing.

Use `scripts/record_incident.py` for append-only machine-readable capture.

## Deduplication

Check in order:

1. exact signature;
2. normalized error signature;
3. same root cause with different message;
4. same category and prevention rule;
5. new incident.

Merge only when the verified fix and prevention rule are genuinely compatible. Do not hide distinct root causes under one broad pattern.

## Promotion

Promote when:

1. the pattern recurs at least three times; or
2. one verified event can corrupt splits, metrics, checkpoints, or article evidence; or
3. a deterministic guard can prevent a high-cost rerun; or
4. the user confirms it as a stable workflow preference.

Classify the destination:

- global hard rule → `SKILL.md`;
- phase-specific rule → one reference;
- deterministic prevention → script or test;
- output structure → asset or schema;
- project-specific convention → project `workflow/knowledge`;
- one-off environment incident → archive.

Before promotion:

1. confirm evidence;
2. deduplicate;
3. check for conflicting rules;
4. define scope and exceptions;
5. add a validation or forward-test case;
6. update the smallest authoritative location.

## Compaction And Review

Keep active context short:

1. preserve recent or high-severity incidents in detail;
2. summarize older recurring patterns;
3. archive one-time inactive errors;
4. mark superseded rules;
5. remove resolved feature requests from active context;
6. keep `START_HERE.md` below a practical session-loading size;
7. review the queue after a milestone or approximately weekly during active work.

Share validated patterns across endpoints through version control. Keep raw private signals, secrets, and machine-local context out of shared history.

## Skill Evolution

Use this change path:

```text
discover → record → deduplicate → reproduce_or_validate → classify → promote → regression_test → publish
```

Treat text-only clarification as a patch, a new reference or script as a minor version, and a changed core router or scientific contract as a major version. Track Skill versions with Git commits and tags, not extra frontmatter fields. Negotiate live protocol changes through the rule branches required by `research-audit-loop`.
