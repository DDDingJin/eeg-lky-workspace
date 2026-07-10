# Accepted Protocol Audit

- dataset: `weissbart_tf64`
- subject count: `13`
- seed: `0`

## `happyquokka`

- comparable: `partial`
- reason: same split/scorer/aggregation/final-test usage, but model family contract and coverage differ from accepted 50-sample window families
- model_family_contract: `10s_chunk`
- checkpoint_selection_rule: `best validation Pearson`
- known_caveats: distinct 10s_chunk contract; per-recording coverage can be <1 due to tail shorter than 10s chunk
