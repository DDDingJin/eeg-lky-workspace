# DECAF Context-Assisted Protocol

- task_classification: `context_assisted_envelope_forecasting`
- status: `DECAF local adaptation / not yet reference-protocol parity`
- model: `external/upstream/DECAF/src/models/two_branch_model.py::TwoBranchModel`
- upstream_commit: `f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332`
- fusion_strategy: `weighted`
- sampling_rate: `64`
- EEG/target interval: `[t, t+224)`
- envelope_context interval: `[t-128, t)` from the same recording and split
- effective forward context after upstream trim: `[t-96, t)`
- training_loss: `mse`
- checkpoint selection: validation recordings only
- scientific metrics are intentionally not computed in this single-batch engineering smoke.
