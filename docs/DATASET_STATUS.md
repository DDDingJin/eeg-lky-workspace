# Dataset Status

Last checked: 2026-06-17

## 0. Dataset Lineage And Benchmark Role

The current workspace mixes several related but not identical auditory EEG data lines.

They should be interpreted as follows.

### `Thornton / mldecoders / hugo_sample`

- local source:
  - `external/data_download/sample_data/sample_data`
- local processed export:
  - `data/processed/reference_splits/hugo_sample_tf64`
- role:
  - development dataset
  - smoke-test dataset
  - sample-only cross-method panorama
- important note:
  - this is not the same thing as `SparrKULee`
  - this is not the same thing as the controlled `ICASSP 2023 challenge split`
  - this should not be treated as the final article-grade benchmark core

### `Weissbart`

- raw/original family:
  - `weissbart_7086168`
- aligned easier-to-use family:
  - `weissbart_7775260_hdf5`
- local benchmark export:
  - `data/processed/reference_splits/weissbart_tf64`
- role:
  - first serious public reconstruction benchmark in the current branch

### `Etard / Reichenbach`

- raw/original family:
  - `etard_reichenbach_7086209`
- aligned easier-to-use family:
  - `etard_reichenbach_7778289`
- local benchmark export:
  - `data/processed/reference_splits/etard_tf64`
- role:
  - second serious public reconstruction benchmark in the current branch

### `KU Leuven / AAD / SparrKULee / challenge ecosystem`

- public large-scale family:
  - `SparrKULee`
- controlled benchmark layers:
  - `ICASSP 2023 challenge split`
  - `ICASSP 2024 challenge split`
- role:
  - this is the natural large benchmark target for article-grade comparison
  - it is related to the challenge ecosystem and should be treated separately from the current `weissbart` and `etard` tables

### `Fuglsang / DTU`

- current local state:
  - upstream reference evaluation files are present under `external/upstream/vlaai/evaluation_datasets/DTU`
  - a public Fuglsang archive is also tracked separately
- role:
  - external generalization dataset family
  - useful after the main public benchmark layer is stable

## 1. Already complete locally

### `weissbart_7775260_hdf5`

Path: `E:\decode\data\raw\weissbart_7775260_hdf5`

Status: complete

Files confirmed:

- `P00.h5` to `P12.h5`
- `audiobooks.zip`
- `stimulus_order.csv`
- `preprocess.py`
- `align_data.py`

Notes:

- This is the easier-to-use HDF5 release of the Weissbart dataset.
- It corresponds to the paper on cortical tracking of surprisal during continuous speech comprehension.
- Local processed exports now exist under:
  - `data/processed/reference_splits/weissbart_tf64`
  - `data/processed/reference_splits/weissbart_tf64_p00`
- `weissbart_tf64` contains all 13 exported participants and is now the first serious non-sample benchmark target.

## 2. Already present locally, but as a different release

### `weissbart_7086168`

Path: `E:\decode\external\data_download\WeissbartSurprisal.zip`

Status: present as archive, not unpacked into `data/raw/`

Notes:

- This is the older/original raw release.
- It contains raw EEG plus stimuli and linguistic features.
- `7775260` is the more accessible HDF5-aligned version of the same dataset family.
- Do not prioritize downloading this again unless you specifically need:
  - raw VHDR files
  - the original folder layout
  - the original derived linguistic-feature files

## 3. Already present locally from upstream code

### `DTU / Fuglsang`

Path: `E:\decode\external\upstream\vlaai\evaluation_datasets\DTU`

Status: already present inside upstream VLAAI repository

Notes:

- These are `.npz` evaluation files bundled with the upstream VLAAI repo.
- This likely covers immediate external-evaluation smoke tests.
- Still treat this as upstream reference data, not your canonical raw-data store.

## 4. Still should be downloaded

### `etard_reichenbach_7778289`

Recommended path: `E:\decode\data\raw\etard_7778289_hdf5`

Status: downloaded locally and adapted into unified split exports

Why it matters:

- Different dataset from Weissbart.
- 18 participants.
- Includes English clean/noisy/competing-speaker conditions.
- Includes Dutch session for 12 participants.
- HDF5-aligned and easier to use than the older raw release.

Expected files:

- `eeg.zip`
- `audiobooks.zip`
- `align_data.py`
- `session_info.json`

Current local state:

- all four expected files are present under `data/raw/etard_7778289_hdf5`
- aligned `eeg/Pxx.h5` files are present under `data/raw/etard_7778289_hdf5/eeg`
- `P00` unified export smoke test completed under:
  - `data/processed/reference_splits/etard_tf64_p00_test`
- full unified export now exists under:
  - `data/processed/reference_splits/etard_tf64`
- exported participants:
  - `P00` to `P19`

### `etard_reichenbach_7086209`

Recommended path: `E:\decode\data\raw\etard_7086209_raw`

Status: missing locally

Why it matters:

- Older/original raw release corresponding to `7778289`.
- Much larger single archive.
- Only download if you need raw VHDR/CND structures instead of the newer aligned release.

Expected file:

- `EtardBrainstemAndComprehension.zip`

## 5. Public datasets referenced by current upstream repos

### `hugo_sample` full-set clarification

Status: no separate larger official local release currently identified inside this workspace

What is currently known from the upstream `mldecoders` materials:

- the tutorial notebooks download a ready-made `thornton_data/data.h5`
- the local sample package contains:
  - 13 participants
  - 15 story parts
  - EEG recordings plus aligned story audio and linguistic side information
- upstream preprocessing code also assumes:
  - participants `1,2,3,4,5,6,7,8,9,10,12,13,14`
  - `15` story parts

Interpretation:

- the local `sample_data` package already appears to match the full tutorial-scale `hugo` data layout used by the `mldecoders` code
- however, the upstream repository does not clearly document a larger openly downloadable `hugo` benchmark release comparable to `SparrKULee`
- therefore this branch should treat `hugo_sample_tf64` as a tutorial/development dataset, not as a missing large-scale benchmark that still needs to be "completed"

If a larger raw or official research-grade `hugo` release exists, it is not clearly linked in the current local upstream documentation and would require a separate acquisition step.

### `SparrKULee`

Recommended path: `E:\decode\data\raw\sparrkulee`

Status: missing locally

Why it matters:

- This is the large benchmark-scale dataset for the KU Leuven ecosystem.
- It is the natural target if you want a sustained multi-model benchmark.
- Upstream repos that reference it:
  - `auditory-eeg-dataset`
  - `auditory-eeg-challenge-2023-code`
  - `auditory-eeg-challenge-2024-code`
  - `ADT_Network` preprocessing

### `ICASSP 2023 challenge split`

Recommended path: `E:\decode\data\raw\challenge_2023`

Status: missing locally

Notes:

- Access is controlled / challenge-style.
- Not the first thing to download if your goal is method build-out.

### `ICASSP 2024 challenge split`

Recommended path: `E:\decode\data\raw\challenge_2024`

Status: missing locally

Notes:

- Access pattern is similar to the 2023 challenge release.
- Better treated as a later standardized benchmark layer.

### `Fuglsang hearing dataset`

Recommended path: `E:\decode\data\raw\fuglsang_hearing_3618205`

Status: downloaded locally as archive, not yet adapted

Notes:

- Public and potentially useful as another external generalization dataset.
- Lower priority than `7778289` and `SparrKULee`.

## 6. Practical integration priority

Recommended order:

1. increase `weissbart_tf64` and `etard_tf64` exact-port training budget
2. one Fuglsang / DTU family dataset as external generalization
3. `SparrKULee` if access or preprocessed route becomes practical
4. raw-format backfills only if paper replication specifically needs them

## 7. Method work that can continue before downloads finish

These do not need the next dataset downloads to start:

- finish more linear baselines
- finish stronger CCA variants
- scale exact VLAAI benchmark track
- scale exact ADT benchmark track
- unify result tables and plotting
- build per-dataset adapters so new datasets plug into one training/eval API
