# Incremental Comparison

Previous fix branch:

- `fix/ar-20260625-161300-a43831b-gate0-gate2`

Previous commit:

- `3d335839e9c9631c163ff48fb670d2e03ba2d558`

Current fix branch:

- `fix/ar-20260625-161300-a43831b-gate0-gate2-real-evidence-and-paper`

Current commit:

- `0a5d70173c0621f89cef43b53db7c87c33ef3f46`

Commit note:

- `0a5d70173c0621f89cef43b53db7c87c33ef3f46` is the real-sample result commit.
- `a4b94640d42bc2bd5ddaec604254f6fc4d4e8b73` is the incremental cleanup and artifact-tracking commit.
- `db7368601fca0b3177e73ad20f61066f03101785` is the skill-sync metadata commit that aligns the repository with the now-installed local `research-audit-loop` skill.

## What Changed In This Round

| Item | Previous fix branch | Current fix branch | Incremental status |
|---|---|---|---|
| Model registry base structure | Present | Present and expanded | Updated |
| Unified schema layer | Present | Present | Carried forward |
| Unified scorer layer | Present | Present | Carried forward |
| Synthetic smoke test | Present | Present | Carried forward |
| Review response / fix manifest base structure | Present | Present and refined | Updated |
| Real sample config | Not present | `configs/benchmark/gate0_gate2_real_sample.json` | New |
| Real sample runner | Not present | `scripts/run_gate0_gate2_real_sample.py` | New |
| Real sample simple FCNN implementation | Not present | `src/repro/simple_models.py` | New |
| Real sample outputs | Not present | `experiments/gate0_gate2_real_sample/` | New |
| Unified `recording_metrics.csv` on real sample | Not present | Present | New |
| Unified `subject_metrics.csv` on real sample | Not present | Present | New |
| Real-sample scorer note | Not present | Present | New |
| Real-sample boundary note | Not present | Present | New |
| Manuscript scaffold | Not present | `paper/draft_zh/` present | New |

## Evidence Added In This Round

| Evidence | Path | Status |
|---|---|---|
| Real sample config | `configs/benchmark/gate0_gate2_real_sample.json` | New |
| Real sample runner | `scripts/run_gate0_gate2_real_sample.py` | New |
| Ridge output | `experiments/gate0_gate2_real_sample/recording_metrics.csv` | New |
| CCA output | `experiments/gate0_gate2_real_sample/recording_metrics.csv` | New |
| FCNN output | `experiments/gate0_gate2_real_sample/recording_metrics.csv` | New |
| ADT output | `experiments/gate0_gate2_real_sample/recording_metrics.csv` | New |
| Unified subject metrics | `experiments/gate0_gate2_real_sample/subject_metrics.csv` | New |
| Unified run manifest | `experiments/gate0_gate2_real_sample/run_manifest.json` | New |
| Paper scaffold | `paper/draft_zh/` | New |

## What Already Existed In The Previous Round

| Existing item from previous round | Status in current round |
|---|---|
| Registry framework | Retained and expanded |
| Result schema | Retained |
| Shared scorer | Retained |
| Synthetic smoke test | Retained |
| Review response / fix manifest base structure | Retained and refined |

## What Results Were Newly Run In This Round

| New run family | Location |
|---|---|
| Minimal real-sample pipeline validation | `experiments/gate0_gate2_real_sample/` |
| Real-sample subject-level metrics | `experiments/gate0_gate2_real_sample/subject_metrics.csv` |
| Real-sample recording-level metrics | `experiments/gate0_gate2_real_sample/recording_metrics.csv` |
| Real-sample schema validation | `experiments/gate0_gate2_real_sample/schema_validation_report.json` |
| Real-sample scorer/boundary notes | `experiments/gate0_gate2_real_sample/scorer_validation.md`, `experiments/gate0_gate2_real_sample/boundary_validation.md` |

## What Was Not Re-Run In This Round

| Not re-run in this round | Reason |
|---|---|
| `experiments/summary_figures/*` historical summaries | Legacy results retained for context only |
| Weissbart benchmark runs | Out of scope for this incremental repair |
| Etard benchmark runs | Out of scope for this incremental repair |
| HappyQuokka benchmark runs | Out of scope for this incremental repair |
| VLAAI benchmark runs | Out of scope for this incremental repair |
| Older cross-dataset panoramas | Not part of the current real-sample closure |

## What Must Not Be Treated As New Conclusions

| Artifact or claim | Reason |
|---|---|
| Old summary figures under `experiments/summary_figures/` | Pre-existing outputs, not rerun in this round |
| Old cross-dataset results | Not part of this incremental run |
| Old all-model panorama figures | Legacy aggregation, not newly validated here |
| Current single-sample Pearson ordering | Minimal real-sample evidence only, not article-grade ranking |

## Incremental Review Rule For Future Rounds

- Future review responses should compare only against the immediately previous accepted fix round.
- Legacy result bundles should not be restated as if they were rerun.
- This round should be read as a delta from `3d335839e9c9631c163ff48fb670d2e03ba2d558`, not as a full benchmark refresh.
