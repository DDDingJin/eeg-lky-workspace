# Archived Status

- status: `deferred_not_in_pure_eeg_main_benchmark`
- Current state: preflight only; no training was run and no scientific metric was generated.
- The genuine DECAF `TwoBranchModel` requires causally preceding true envelope context at inference time.
- This task is `context_assisted_envelope_forecasting`, not pure EEG-only reconstruction.
- HappyQuokka EEG-only is already used as the DECAF-family representative in the pure EEG main benchmark table.
- Do not execute the formal P00 command unless the review endpoint explicitly reopens this condition.
- If reopened later, it must be reported as a separate context-assisted condition, and the current runner must first fix envelope scale handling, checkpoint strict reload, and interruption state handling.
