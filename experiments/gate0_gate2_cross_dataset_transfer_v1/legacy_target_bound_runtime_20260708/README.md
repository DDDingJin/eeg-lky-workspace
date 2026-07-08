This directory preserves an older cross-dataset smoke/runtime snapshot whose checkpoint path was incorrectly target-bound.

It is retained only as legacy engineering evidence.

It is not the formal cross-dataset result scope.

Formal cross-dataset source-only checkpoint design now uses:

- source train job: `source_dataset + model + seed + stage=source_zero_shot`
- target eval job: `source_dataset + target_dataset + model + seed + stage=cross_dataset_zero_shot_eval`

Source-only checkpoint path pattern:

- `local_checkpoints/cross_dataset/source_only/<model>/<source_dataset>/seed<seed>/best_epoch_<N>.pt`
