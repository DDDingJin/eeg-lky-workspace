# Reviewer Feedback Folder

Put reviewer comments here as plain markdown files.

Recommended filename format:

- `YYYY-MM-DD_reviewer_label.md`

Recommended content structure:

```md
# Reviewer Feedback

Reviewer: <label>
Date: YYYY-MM-DD
Branch reviewed: <branch>
Commit reviewed: <commit if known>

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
```

Once a file is placed here, the local assistant can read it and turn it into:

- code changes
- reruns
- figure regeneration
- documentation updates
- response summaries
