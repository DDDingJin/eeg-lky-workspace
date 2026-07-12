# Loader Contract

- MAT files are read only.
- Preflight SHA256 checks are mandatory before training.
- Representations are selected from channel metadata: `mag102` uses mag102_selection hash, `all306` uses channel_inventory order.
- Split is recording-level: train [1,2,5,6,9,10,13,14], val [3,7,11,15], test [4,8,12,16].
- Lag/window construction is contained within each recording.
- Scalers fit on train recordings only.
