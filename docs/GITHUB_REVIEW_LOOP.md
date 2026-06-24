# GitHub Review Loop

Last updated: 2026-06-24

This document defines the intended review-and-iteration loop for this project.

## Goal

The project should evolve as a public benchmark and paper-oriented open-source repository.

The loop is not only for debugging code.
It is also the mechanism for tightening:

- benchmark question design
- fairness of model comparison
- reporting discipline
- paper-grade narrative consistency

The working loop is:

1. run or update local benchmark code
2. generate reviewable result tables, figures, and notes
3. push a review branch to GitHub
4. collect external reviewer comments on another client / another terminal
5. sync those comments back into this workspace
6. update code, protocol, and documentation locally
7. push the revised branch again

## Recommended Branch Roles

- `master`
  - stable public branch
  - should only receive cleaner, better-audited material
- `audit/reproduction-note`
  - active review branch
  - carries method audits, benchmark figures, and response iterations
- optional future branches
  - `feature/neuroconformer-integration`
  - `feature/fair-baseline-table`
  - `feature/cross-subject-protocol`

## Where Reviewer Feedback Should Go

To make the loop machine-readable and easy to iterate, reviewer feedback should be copied into local markdown files under:

- `docs/reviewer_feedback/`

Recommended naming:

- `docs/reviewer_feedback/2026-06-24_reviewer_a.md`
- `docs/reviewer_feedback/2026-06-24_reviewer_b.md`
- `docs/reviewer_feedback/2026-06-25_round2_summary.md`

## Feedback Template

Each feedback file should use this structure:

```md
# Reviewer Feedback

Reviewer: <name or anonymous label>
Date: YYYY-MM-DD
Branch reviewed: <branch name>
Commit reviewed: <commit hash if known>

## Major Concerns
- ...

## Methodology Concerns
- ...

## Implementation Concerns
- ...

## Reporting / Figure Concerns
- ...

## Requested Actions
- ...

## Optional Line-Level Notes
- file/path: short note
```

## Expected Assistant Workflow

When new feedback appears in `docs/reviewer_feedback/`, the local assistant should:

1. read the new feedback files
2. classify each item into:
   - implementation bug
   - protocol mismatch
   - reporting / wording problem
   - new experiment request
3. decide which changes require code, which require reruns, and which require documentation updates
4. implement the local changes
5. regenerate affected figures / CSV / notes
6. update audit documents
7. commit and push the revised review branch

In addition, the assistant should explicitly check whether the requested change affects:

- benchmark scope
- dataset role definitions
- task definitions
- fairness of baseline grouping
- train/test protocol comparability

## What Should Be Pushed For Review

At each review round, prefer pushing:

- core method documents
- benchmark-question documents
- result CSVs
- publication-style figures
- exact code files that produced the results
- short audit notes explaining comparability boundaries

Avoid pushing:

- raw datasets
- temporary caches
- unstable scratch files
- unrelated exploratory outputs

## Review Priority Order

External reviewers should be asked to inspect in this order:

1. `README.md`
2. `docs/CURRENT_EVALUATION_STATUS.md`
3. `docs/reproduction_audit_note.tex`
4. `docs/NEUROCONFORMER_AUDIT.md`
5. benchmark summary CSVs and figures
6. core implementation files

## Decision Rule

At this stage, reviewer comments should be interpreted in the following order of urgency:

1. correctness bug
2. unfair comparison
3. protocol ambiguity
4. missing strong baseline
5. weak benchmark narrative or unsupported claim
6. plotting / presentation refinement

## Two-Terminal Closed Loop

This project now assumes a practical two-terminal workflow:

- execution terminal
  - runs code
  - updates documents
  - regenerates tables and figures
  - commits and pushes benchmark revisions
- review terminal
  - inspects GitHub branch state
  - leaves structured methodological comments
  - requests protocol clarification or new benchmark rows

The key rule is:

- the review terminal should criticize the current branch state
- the execution terminal should translate that criticism into concrete code, protocol, rerun, or documentation actions

This should remain a tight loop rather than an informal chat stream.

This keeps the project aligned with the final goal:

- publishable benchmark logic
- reproducible open-source code
- defensible comparison table
