# Etard Split Sanity

## Dataset entry
- `dataset_id`: `etard_tf64_p00`
- `dataset_locator`: `data/processed/reference_splits/etard_tf64`
- `subject_id`: `P00`

## Split counts for P00
- `train` recordings: `24`
- `val` recordings: `24`
- `test` recordings: `24`

## Participant presence
- `P00` appears in splits: `test, train, val`

## Condition distribution
- `train`: `clean:4, fM:4, fW:4, hb:4, lb:4, mb:4`
- `val`: `clean:4, fM:4, fW:4, hb:4, lb:4, mb:4`
- `test`: `clean:4, fM:4, fW:4, hb:4, lb:4, mb:4`

## Shape and signal sanity
- `train` example EEG shape: `(7900, 64)`, envelope shape: `(7900, 1)`
- `val` example EEG shape: `(987, 64)`, envelope shape: `(987, 1)`
- `test` example EEG shape: `(989, 64)`, envelope shape: `(989, 1)`
- The export summary records `target_fs = 64`, so the pilot's 64 Hz assumption matches the processed export metadata.
- The Etard `test` split for `P00` mixes `clean`, `fM`, `fW`, `hb`, `lb`, and `mb` conditions.

## Issues found
- No obvious shape mismatch, empty envelope, or zero-variance envelope was found in the P00 split scan.
