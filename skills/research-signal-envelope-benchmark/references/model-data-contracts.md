# Model And Data Contracts

## Contents

1. Architecture boundaries
2. Model provenance and adapter
3. Dataset adapter
4. Configuration
5. Reproducibility
6. Integration order
7. Required outputs

## Architecture Boundaries

Use separate modules for:

- models;
- datasets;
- preprocessing;
- protocols;
- metrics;
- runners;
- configuration;
- result schemas.

Require common interfaces for loading, transforming, training, validating, predicting, scoring, saving, and restoring. Use adapters for unavoidable differences. Never embed dataset-specific logic inside a model or model-specific scientific logic inside a protocol runner.

## Model Provenance And Adapter

Before implementation:

1. locate the paper;
2. prefer the author's official repository;
3. otherwise select a credible open implementation;
4. record repository URL and full commit;
5. record license;
6. record original dependencies and expected input;
7. compare code with the paper;
8. record known deviations;
9. avoid rewriting the core model from memory;
10. mark unavailable or unreliable implementations honestly.

Keep third-party source, project adapter, and project modification distinct. For each adapter define:

1. constructor/config mapping;
2. input shape and dtype;
3. target shape;
4. forward output;
5. loss calculation;
6. prediction postprocessing;
7. checkpoint save/load;
8. device and mixed-precision behavior;
9. variable-length and mask behavior;
10. unsupported protocols.

## Dataset Adapter

Expose a common sample and metadata schema. Preserve at least:

- dataset ID and version;
- subject ID;
- recording or trial ID;
- source sample range;
- window ID and source range;
- split membership;
- modality;
- channel or sensor metadata;
- sampling rate;
- input and target shapes;
- exclusion and cleaning flags.

Keep download, license, raw validation, cleaning, preprocessing, split, windowing, and caching as visible stages. Bind every cache key to data version, preprocessing config, and split version.

## Configuration

Separate:

1. model config;
2. dataset config;
3. protocol config;
4. target and metric config;
5. runtime config;
6. release config.

Give each resolved config a stable ID or digest. Save the fully resolved config with every run. Never overwrite a formal config with smoke-only values. Keep output directories unique by study, protocol, model, dataset, subject or fold, and seed.

Validate:

- required fields;
- unknown fields;
- conflicting fields;
- invalid ranges;
- smoke-only flags in full configs;
- missing version identity;
- absolute local paths intended for publication.

## Reproducibility

Lock and record:

1. Python random seed;
2. NumPy seed;
3. framework CPU seed;
4. framework GPU and multi-GPU seeds;
5. DataLoader worker seed;
6. sampler seed;
7. split seed;
8. augmentation seed;
9. deterministic-algorithm settings;
10. known nondeterministic operators;
11. operating system and hardware;
12. Python, framework, CUDA, and dependency versions;
13. code commit;
14. exact command;
15. resolved config;
16. fixed multi-run seed list.

Distinguish the seed used to reproduce one run from the planned seed set used to estimate variability.

## Integration Order

1. Connect one reliable dataset and one model.
2. Validate sample, target, and prediction contracts.
3. Complete one subject-specific path.
4. Add cross-subject protocol.
5. Add cross-dataset protocol.
6. Add further models through the same interfaces.
7. Add further datasets through the same interfaces.
8. Run interface and smoke tests after every adapter addition.

Check:

- finite loss;
- parameter updates;
- gradient flow;
- metric reset;
- train/eval mode;
- optimizer and scheduler order;
- final incomplete batch;
- checkpoint load parity;
- exceptions not being swallowed.

## Required Outputs

Exit implementation with:

1. model-source inventory;
2. dataset-source and license inventory;
3. model adapters;
4. dataset adapters;
5. prediction contracts;
6. config schema and resolved examples;
7. environment lock;
8. interface tests;
9. minimal smoke configuration.
