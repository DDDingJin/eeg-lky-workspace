# DECAF Context-Assisted P00 Protocol

- model: `external/upstream/DECAF/src/models/two_branch_model.py::TwoBranchModel`
- task_classification: `context_assisted_envelope_forecasting`
- fusion_strategy: `weighted`
- EEG/target: `[t, t+224)`
- input context: `[t-128, t)`
- effective context: `[t-96, t)`
- train loss: `mse`
- max_epochs: `100`
- early_stopping_patience: `10`
- checkpoint selection metric: `val_mean_recording_pearson_r`
- test is evaluated only after selecting the best validation checkpoint.
- output belongs only to context-assisted results, not pure reconstruction.
