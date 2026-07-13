# Artifact Schemas

## Contents

1. Workflow state
2. Run manifest
3. Prediction contract
4. Metric rows
5. Completed jobs
6. Failure report
7. Incident
8. Handoff

Use JSON for mutable machine state and CSV for tabular metric observations. Add fields when needed, but never remove identity or provenance required for audit.

## Workflow State

Require:

```json
{
  "schema_version": 1,
  "study_id": "study-id",
  "phase": "P1",
  "active_modules": ["C0", "C1", "C2", "C3", "C5", "P1"],
  "role": "standalone",
  "branch": null,
  "commit": null,
  "config_id": null,
  "dataset_version": null,
  "split_version": null,
  "last_gate": "not_evaluated",
  "planned_jobs": [],
  "completed_jobs": [],
  "failed_jobs": [],
  "pending_jobs": [],
  "resume_enabled": false,
  "last_update": "ISO-8601 timestamp"
}
```

## Run Manifest

Require:

- schema version;
- study and run IDs;
- run type: smoke, closure, pilot, or article;
- branch and full commit;
- config ID and resolved config path;
- dataset and split versions;
- model-source versions;
- target and metric protocol IDs;
- planned model, dataset, subject or fold, and seed scope;
- command;
- environment identity;
- output directory;
- resume enabled;
- accepted smoke or closure evidence;
- domain gate;
- audit round and publication state when applicable;
- creation and update times.

## Prediction Contract

Require:

```json
{
  "schema_version": 1,
  "model_id": "model",
  "input_shape": ["batch", "channels", "time"],
  "raw_output_shape": ["batch", "time", 1],
  "prediction_shape": ["batch", "time"],
  "target_shape": ["batch", "time"],
  "scorer_input_shape": ["time"],
  "alignment_rule": "explicit rule",
  "overlap_aggregation": "mean",
  "mask_rule": "explicit rule",
  "variable_length_rule": "explicit rule"
}
```

## Metric Rows

Require columns sufficient to identify:

- study and run;
- dataset;
- subject and fold;
- recording when recording-level;
- model;
- seed;
- protocol, target, and config IDs;
- metric name and value;
- aggregation level;
- observation count;
- status: new, reused, failed, or diagnostic;
- code commit;
- result path.

Do not store only a pre-aggregated mean.

## Completed Jobs

Each entry must include:

- job key;
- dataset;
- subject or fold;
- model;
- seed;
- config ID;
- completion time;
- required metric row counts;
- metadata path;
- checkpoint-manifest path when applicable.

Write a completed entry only after required rows are durable.

## Failure Report

Each failure should include:

- job key;
- status;
- stage;
- exception type and message;
- traceback or log path;
- branch, commit, config, and environment;
- time;
- retry status and reason;
- related incident ID.

Never delete a historical failure silently.

## Incident

Require:

- incident ID;
- date;
- normalized signature;
- stage and category;
- environment;
- command and job key;
- symptom;
- root cause;
- fix;
- verification;
- prevention;
- affected versions;
- recurrence count;
- scope;
- status.

## Handoff

Require:

- role and status;
- phase and active modules;
- domain gate;
- audit round and publication state when applicable;
- branch, commit, base branch, and base commit;
- commands and resolved config;
- completed, failed, skipped, reused, and pending jobs;
- changed files and artifact paths;
- validation, leakage, failure, and checkpoint status;
- open incidents;
- caveats;
- next safe action.
