# `codex/open-source-latest` branch status

## Purpose

This is the single integration branch for the latest runnable EEG and MEG code available on 2026-07-13.  It is meant to become the basis of the public repository, while keeping the original experiment branches as immutable provenance records.

## Integrated scope

The branch includes every current `origin/fix/*` branch, including these independent protocol lines:

- MEG-SCANS preprocessing, representation checks, the native `sub-03` single-subject runner, and the unified 11-model runner.
- EEG subject-specific and fixed-holdout model runners.
- EEG cross-dataset transfer, LOSO, and subject-holdout fine-tuning runners.
- DECAF integration, causality audit, and context-assisted runner checks.

When the same active-worktree handoff file conflicted, the version from the more recent integration line was retained.  This preserves code and compact artifacts; it does not claim that the retained handoff note is a general public entry point.

## What this branch deliberately does not contain

- Raw EEG/MEG recordings, derived MAT/HDF5 data, checkpoints, model weights, prediction arrays, or other large artifacts.
- A portable environment lockfile or installation package.
- Bundled copies of all external upstream implementations used by some adapters.
- A public-release licence, citation file, dataset-access instructions, or release-quality tutorial.

Some configurations therefore contain workstation-local Windows paths.  They document the executed run, but cannot be used unchanged on another machine.

## Required work before making the repository public

1. Replace absolute data/upstream paths with documented environment variables or a data-root configuration layer.
2. Add a reproducible environment specification, licence, citation metadata, and a clear data-access/download guide.
3. Decide whether external implementations (for example the upstream CNN and HappyQuokka adapter) are vendored, declared as pinned dependencies, or replaced by local implementations.
4. Replace historical `workflow/START_HERE*.md` handoff notes with a neutral public quick-start and protocol index.
5. Run a clean-machine smoke test using only the documented setup, then archive the exact commit/tag used for the release.

Until those gates are complete, this branch should be treated as the latest code consolidation, not as a reproducible public release.
