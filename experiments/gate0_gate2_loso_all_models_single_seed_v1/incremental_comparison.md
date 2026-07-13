# Incremental Comparison

- base branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1`
- base commit: `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- current branch: `fix/ar-20260625-161300-a43831b-loso-all-models-single-seed-v1`
- this round expands LOSO from ridge-only to the current unified subject-specific model set.
- ridge is reused from the verified `loso-ridge-full-v1` branch instead of being rerun.
- executed models in this branch: `adt, cca, cnn, dnn, eegnet, fcnn, ridge`
- subject-level failures or skips recorded: `198`
- models without successful LOSO result rows in this round: `adt, cca, cnn, dnn, eegnet, fcnn`
