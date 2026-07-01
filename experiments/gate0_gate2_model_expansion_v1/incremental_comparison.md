# Incremental Comparison

- base branch: `fix/ar-20260625-161300-a43831b-multi-seed-v1`
- base commit: `dee590b26aca180637b920b9579a59daa9ba176c`
- current branch target: `fix/ar-20260625-161300-a43831b-model-expansion-v1`
- this round reuses the completed multi-seed aggregate for `ridge`, `cca`, `fcnn`, and `adt`.
- this round only runs missing full-subject jobs for requested expansion models under the same split, scorer, and aggregation schema.
- this round does not promote final paper wording and does not regenerate heavyweight prediction dumps or checkpoints in the aggregate output directory.

## Effective model set
- `ridge, cca, fcnn, adt, dnn, cnn, eegnet`

## Requested expansion set
- `dnn, cnn, eegnet`
