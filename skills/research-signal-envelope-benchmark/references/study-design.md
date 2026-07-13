# Study Design

## Contents

1. Entry conditions
2. Research questions and claims
3. Experiment matrix
4. Fair comparison
5. Statistics and evidence
6. Scope control
7. Required outputs

## Entry Conditions

Read this reference when starting an article, changing the main question, adding an experiment level, changing a split or evaluation protocol, or revising a scientific claim.

Do not require this entire module for a local implementation-only fix that leaves the accepted protocol unchanged.

## Research Questions And Claims

Define before implementation:

1. neural input modality and representation;
2. target envelope definition;
3. reconstruction unit and time scale;
4. model families being compared;
5. dataset roles;
6. primary generalization question;
7. primary claim candidates;
8. secondary claim candidates;
9. experiments needed for each claim;
10. questions explicitly outside the article.

Maintain a claim-to-evidence map with:

- `claim_id`;
- claim wording;
- confirmatory or exploratory status;
- required dataset, protocol, model group, metric, and statistic;
- expected table or figure;
- evidence path;
- current support status;
- known limitations.

Do not add a strong claim because a convenient test happened to be significant.

## Experiment Matrix

Represent the benchmark as:

```text
models × datasets × protocols × subjects_or_folds × seeds
```

Classify each cell as required, optional, pilot-only, unavailable, failed, or complete.

### Single Dataset, Single Subject

1. Train and test a separate model for each subject.
2. Split recordings or trials before sliding windows.
3. Keep train, validation, and test temporal or trial boundaries explicit.
4. Store each subject result before dataset aggregation.

### Single Dataset, Cross Subject

1. Split at subject level.
2. Example: train A-D, validate E, test F.
3. Exclude the held-out subject from fitting, tuning, normalization, and checkpoint selection.
4. State whether the split is fixed or rotating.
5. Store each held-out fold independently.

### Cross Dataset, Same Modality

1. Train on source dataset G and test on target dataset H.
2. Keep target test labels out of tuning.
3. Declare channel, sampling-rate, envelope, label, and preprocessing alignment.
4. Distinguish direct transfer from target-domain adaptation.
5. Report source-only, adapted, and target-trained results separately when present.

### Cross Dataset, Cross Modality

1. Declare EEG, MEG, or other modality roles.
2. Identify directly comparable inputs.
3. Define modality adapters explicitly.
4. Separate modality shift from ordinary dataset shift.
5. Mark settings that are not scientifically comparable.

### Optional Extensions

- single-source to single-target;
- multi-source to single-target;
- joint multi-dataset training;
- zero-shot transfer;
- few-shot target adaptation;
- cross-subject plus cross-dataset evaluation;
- model ablation;
- preprocessing ablation;
- robustness or sensitivity analyses.

Do not let optional cells block the primary article path unless they can change the main conclusion.

## Fair Comparison

1. Use identical splits for all comparable models.
2. Use the same target construction and scorer path.
3. Use validation-only hyperparameter and checkpoint selection.
4. Declare each model's search space and search budget.
5. Use a common seed list for each comparison group.
6. Keep training budget rules explicit.
7. Explain unavoidable model-specific budgets.
8. Report parameter count, training time, inference time, and memory when cost is part of the claim.
9. Separate predictive quality from computational efficiency.
10. Include simple, linear, and null baselines appropriate to the task.
11. Mark reused baseline rows and filter them to the active matrix.
12. Do not compare cells with incompatible target, split, or aggregation definitions as if they were equivalent.

## Statistics And Evidence

1. Predeclare the primary metric.
2. Predeclare the primary aggregation unit.
3. Store recording-, subject-, fold-, and seed-level observations.
4. Use subject-level paired comparisons when subjects are the scientific sampling unit.
5. Report mean and standard deviation when appropriate.
6. Report confidence intervals or another uncertainty estimate when useful.
7. Report effect size separately from statistical significance.
8. Correct for multiple comparisons across broad model or dataset families when required.
9. Predeclare how failed or missing jobs affect analysis.
10. Keep excluded subjects and reasons visible.
11. Separate confirmatory and exploratory statistics.
12. Preserve negative results.

## Scope Control

Prioritize:

1. leakage and evidence corruption;
2. issues that can change the main conclusion;
3. issues blocking the primary experiment matrix;
4. reproducibility failures;
5. article-required comparisons;
6. optional refinements.

Move low-contribution, high-cost work to `workflow/backlog.md`. Record why it was deferred and what evidence would justify reopening it.

## Required Outputs

Exit design work with:

1. research-question list;
2. claim-to-evidence map;
3. required and optional experiment matrix;
4. dataset-role table;
5. envelope and prediction contract draft;
6. primary and secondary metric plan;
7. baseline plan;
8. statistical analysis plan;
9. compute-budget notes;
10. explicit non-goals.
