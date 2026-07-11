# DECAF Task Compatibility Decision

- pure_reconstruction_compatible: `false`
- task_classification: `context_assisted_or_forecasting`
- reason: `TwoBranchModel.forward` requires an `envelope_context` input for the envelope branch. If that context is the true past target envelope at test time, the condition is context-assisted forecasting, not pure EEG-only envelope reconstruction.
- reporting rule: do not mix TwoBranchModel results into the pure EEG envelope reconstruction main table.
- allowed next step: reviewer may approve a separate context-assisted/forecasting condition with explicit context availability assumptions.
