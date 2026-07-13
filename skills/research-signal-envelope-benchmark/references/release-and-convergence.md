# Release And Branch Convergence

## Contents

1. Entry conditions
2. Branch inventory
3. Convergence
4. Repository shape
5. Release layers
6. Reproduction entry points
7. Release gate

## Entry Conditions

Enter this phase only after primary interfaces, protocols, runner behavior, and most article evidence are stable. Do not make branch cleanup block active scientific work prematurely.

## Branch Inventory

Freeze creation of new long-lived branches. For each existing branch record:

1. name;
2. base and head commit;
3. purpose;
4. changed modules;
5. unique code;
6. unique configs;
7. unique evidence;
8. uncommitted work;
9. data and split versions;
10. protocol;
11. smoke and full-run status;
12. caveats;
13. keep, extract commits, port logic, preserve evidence, archive, supersede, deprecate, or investigate decision.

Let the reviewer own the branch/version register under `research-audit-loop`. Let the implementer report facts but not rewrite the register.

## Convergence

1. Select a stable common base.
2. Create a release-candidate branch.
3. Do not choose the newest branch automatically.
4. Do not merge every branch wholesale.
5. Port one clear module or change at a time.
6. Integrate shared interfaces first.
7. Integrate dataset adapters.
8. Integrate model adapters.
9. Integrate protocols and runners.
10. Integrate aggregation and release tooling.
11. Test and smoke after each port.
12. Record source branch and commit.
13. Keep legacy results outside the active result directory.
14. Require engineering closure after runner changes.
15. Promote the release candidate only after clean validation.

Use branches for code changes, not experiment identity. Store experiment identity in manifests and commits. Prefer short-lived `feat/*`, `fix/*`, `exp/*`, and `release/*` branches plus a stable protected target branch.

## Repository Shape

Aim for:

```text
project/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml or environment lock
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── protocols/
│   └── releases/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── protocols/
│   ├── metrics/
│   └── runners/
├── scripts/
├── splits/
├── tests/
├── docs/
├── third_party/
├── workflow/
└── release_artifacts/
```

Keep formal implementation in `src/`, formal experiment definitions in `configs/`, fixed partitions in `splits/`, fast correctness checks in `tests/`, and compact final evidence in `release_artifacts/`.

## Release Layers

### Code

Publish model adapters, data adapters, preprocessing, protocols, training, evaluation, metrics, aggregation, table and figure generation, smoke tests, leakage checks, and release verification.

### Environment

Publish Python, framework, CUDA, operating-system assumptions, locked dependencies, environment checks, hardware requirements, and optional container files.

### Data Recipe

Publish source, license, acquisition or application instructions, download scripts, checksums, directory layout, cleaning, preprocessing, exclusions, fixed splits, label and channel mapping, modality mapping, and data version. Do not redistribute restricted or identifying raw data.

### Model Artifacts

Publish necessary best weights or external links, configs, selection metrics, seeds, code and data versions, checksums, license, and loading example. Keep large weights out of normal Git history.

### Evidence

Publish compact raw recording, subject, fold, and seed metrics; aggregate results; failure and reuse declarations; run manifests; schema and leakage reports; tables; figures; and generation scripts.

Exclude secrets, personal data, absolute local paths, machine caches, temporary checkpoints, full debug logs, duplicate result trees, unlicensed third-party code, unexplained historical scripts, and mixed failed-branch results.

## Reproduction Entry Points

Support:

1. from-scratch data acquisition, preprocessing, training, testing, aggregation, and figure generation;
2. checkpoint-based evaluation;
3. result-only audit from raw metrics to statistics and figures.

Provide a fixed command chain such as setup, check environment, fetch data, verify data, prepare data, verify splits, smoke, benchmark, resume, evaluate, aggregate, build tables, build figures, and verify release. Give every entry explicit input, output, config, failure state, and `--help`.

## Release Gate

1. Freeze code commit.
2. Freeze environment, data, split, target, config, budget, metric, checkpoint, runner, and schema versions.
3. Validate a fresh clone and clean environment.
4. Run a minimal reproduction from a new directory.
5. Load and evaluate a published checkpoint.
6. Rebuild aggregate tables and figures from raw metrics.
7. Verify licenses, third-party provenance, secrets, large files, and checksums.
8. Create an immutable version tag.
9. Bind final article evidence to that tag.
10. Publish corrections as a new version.
