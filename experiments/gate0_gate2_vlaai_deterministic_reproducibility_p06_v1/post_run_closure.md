# VLAAI P06 Deterministic Reproducibility Closure

Status: completed.

## Scope
- Dataset: `weissbart_tf64`.
- Subject: `P06`.
- Model: `vlaai`.
- Seed: `0`.
- Device: `cuda`.
- Output directory: `experiments/gate0_gate2_vlaai_deterministic_reproducibility_p06_v1`.
- Published benchmark metrics were not modified.

## Deterministic Policy
- `random.seed(0)`, `numpy.seed(0)`, `torch.manual_seed(0)`, and `torch.cuda.manual_seed_all(0)` were applied.
- `DataLoader` used `torch.Generator(seed=0)`.
- `num_workers=0`.
- `torch.backends.cudnn.benchmark=False`.
- `torch.backends.cudnn.deterministic=True`.
- `torch.use_deterministic_algorithms(True, warn_only=False)`.
- Python was launched with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Result
- Repeat count: `2`.
- Reproducible: `true`.
- best_epoch: `4`.
- epochs_completed: `14`.
- best_val_score: `0.09118621892606218`.
- test_metric: `0.028696070905947592`.
- initial_state_hash: `a254b3d1e89088188f5ab3ecd9680f54913f590087b0bcaf7c7873712895930c`.
- first_train_batch_hash: `ba71d466c563bbe1d1dc904a6cf156c9bf0a6d89b819d3670b489c04f22c6533`.
- after_first_optimizer_step_hash: `d7d1aefb3314ec6019bdaf6836d364b605b9848a4454a5bb591969eca8035e51`.
- best_state_hash: `2acc29fcab40d080764e5a28fe1e8c05015016cb6cd7b1f219574fe9233a75b1`.
- prediction_hash: `291e32ff4a0fe793b50af20e1674c85239d3b2e8e17eafdb46295f33d21f350f`.

No checkpoint, prediction dump, raw data, model weights, or large binary artifact was written.
