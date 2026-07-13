# DECAF Integration Preflight Report

- protocol: `gate0_gate2_decaf_integration_preflight_v1`
- dataset: `weissbart_tf64`
- subject: `P00`
- sampling_rate: `64`
- status: `DECAF local adaptation / not yet reference-protocol parity`
- schema_passed: `True`
- dependency_missing: `wandb(training_only)`

## Split Isolation
- P00 train/val/test subject isolation: `True` ({'train': 15, 'val': 15, 'test': 15})
- P00 train/val/test recording disjointness: `True` (recording ids are disjoint across splits)

## Coverage
- train: recordings=`15`, coverage_min=`1.000000`, coverage_max=`1.000000`
- val: recordings=`15`, coverage_min=`1.000000`, coverage_max=`1.000000`
- test: recordings=`15`, coverage_min=`1.000000`, coverage_max=`1.000000`

## Import Checks
- torch: `available` required=`True`
- numpy: `available` required=`True`
- scipy: `available` required=`True`
- sklearn: `available` required=`True`
- wandb: `missing` required=`training_only`
- DECAF HappyQuoka import: `available` required=`True`

## Next Step
- If schema remains passed and required imports are available, this can enter P00 formal smoke; do not treat this preflight as a scientific result.
