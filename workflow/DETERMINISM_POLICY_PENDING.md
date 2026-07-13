# Determinism Policy Pending

Status: pending integration into formal within-dataset cross-subject neural runners.

VLAAI P06 deterministic closure established that strict CUDA determinism can make the Weissbart P06 VLAAI training path reproducible:

- branch: `fix/ar-20260625-161300-a43831b-vlaai-deterministic-reproducibility-p06-v1`
- commit: `ae53d62`
- required launch environment: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- required runtime settings: Python, NumPy, Torch and CUDA seeds; fixed `DataLoader` generator; `num_workers=0`; `torch.backends.cudnn.benchmark=False`; `torch.backends.cudnn.deterministic=True`; `torch.use_deterministic_algorithms(True, warn_only=False)`

Decision for this branch:

- Do not retrofit strict deterministic execution into the new within-dataset modelset runner during this smoke/preflight round.
- After smoke passes, formal neural-model `zero_shot` jobs should receive a separate patch that applies the strict deterministic policy consistently.
- Linear-family validation/normalization issues are tracked separately and must not be mixed with deterministic-policy rollout.
