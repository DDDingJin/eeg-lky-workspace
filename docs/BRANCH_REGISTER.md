# Branch Register

Last updated: 2026-07-01

This file is maintained by the review endpoint. The implementer endpoint should not edit it during experiment execution. Each execution round should report its branch and commit; the review endpoint will update this register after review.

## Source Of Truth

- Rule branch: `rule/research-audit-loop-v1`
- Rule commit: `032c81484f1ccf141e5991a0a9d70a3294d8d532`
- Publish-gate rule branch: `rule/research-audit-loop-v1.1-publish-gate`
- Publish-gate rule commit: `365c7c2b034af2abd1e6283adde980de6de7eeed`
- Review package branch: `review/ar-20260625-161300-a43831b-blueprint`
- Original reviewed implementation baseline: `audit/reproduction-note @ a43831b83598c82520be320c21b56e92b73b7dcd`
- Current accepted execution base for the next code round: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure @ d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`
- Latest full-subject result branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1 @ 68f87a4c0005e4556a51b83cd42e73908edc44ad`
- Latest reviewable full-subject artifact branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure @ d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`

## Branches

| Branch | Tip commit | Role | Base / predecessor | Review status | Notes |
|---|---|---|---|---|---|
| `audit/reproduction-note` | `a43831b83598c82520be320c21b56e92b73b7dcd` | Original reviewed implementation branch | Earlier reproduction-note work | Locked baseline | Target commit for review round `AR-20260625-161300-a43831b`. |
| `rule/research-audit-loop-v1` | `032c81484f1ccf141e5991a0a9d70a3294d8d532` | Shared reviewer/implementer rule and skill branch | Rule proposal iterations | Accepted rule lock | Contains `AGENTS.md` and `skills/research-audit-loop/`. |
| `rule/research-audit-loop-v1.1-publish-gate` | `bc8ccfd550e9ad94c04d89c3a5c73ddca84c3cf8` | Shared rule update for experiment completion status | `rule/research-audit-loop-v1` | Accepted rule update | Adds the publish gate: local runs are not review-complete until committed, pushed, and remotely inspectable. |
| `review/ar-20260625-161300-a43831b-blueprint` | `45a19ebe8f336ad074fdda1dfc18b09f0327d1cc` | Review package with long-term blueprint and concrete review | `a43831b` | Accepted review package | Contains `review_cycles/2026-06-25_161300+0800_AR-20260625-161300-a43831b/`. |
| `fix/ar-20260625-161300-a43831b-gate0-gate2` | `3d335839e9c9631c163ff48fb670d2e03ba2d558` | Gate 0-2 framework closure | `a43831b` | Partially accepted | Established registry, schema, scorer, and smoke-test framework. |
| `fix/ar-20260625-161300-a43831b-gate0-gate2-real-evidence-and-paper` | `39b78b6a9f2cab94f7665a80efe794d3c5b99671` | Real-sample closure and paper scaffold | `3d335839` | Accepted as real-sample minimal closure | Added Hugo real-sample evidence, compact artifacts, skill alignment, and incremental comparison. |
| `fix/ar-20260625-161300-a43831b-article-pilot-weissbart-etard-p00` | `827e66d9c703595ec65c2abda2e6e7c1a6da8807` | Article-grade P00 pilot attempt | `39b78b6` | Partially accepted | Weissbart P00 succeeded; Etard initially failed as missing `etard_tf64_p00` alias. |
| `fix/ar-20260625-161300-a43831b-etard-p00-pilot-closure` | `85fbd5a22dcad9df19d51d8d2369ccd1c27bce9b` | Etard P00 pilot closure | `827e66d` | Accepted as engineering closure; required diagnostics | Etard P00 reads from full `etard_tf64` export. Low FCNN/ADT triggered diagnostics. |
| `fix/ar-20260625-161300-a43831b-article-pilot-diagnostic` | `58db691d2c08c2b22ce3cfa50119c1a240c41a18` | Old-vs-current and condition diagnostic | `85fbd5a` | Accepted diagnostic | No model rerun. Added legacy comparison, Etard condition diagnostic, split sanity, and result consistency check. |
| `fix/ar-20260625-161300-a43831b-etard-p00-focused-rerun` | `2840e5a0b8d6caff1c31765f7ca73ed429ea98ec` | Etard P00 FCNN/ADT focused budget rerun | `58db691d` | Accepted for ADT; FCNN still flagged | ADT recovered to old ADT scale; FCNN did not improve. |
| `fix/ar-20260625-161300-a43831b-fcnn-protocol-audit` | `c32114a1d919604341458ca3da5d9288814f0107` | FCNN protocol identity audit | `2840e5a` | Accepted; current execution base | FCNN should be labeled `local_fcnn_baseline` / `architecture_baseline`, not a historical parity baseline. |
| `fix/ar-20260625-161300-a43831b-full-subject-single-seed-v1` | `e459f2ca4807c9f72334c20fa93b2d3dfc67c253` | Full-subject single-seed v1 | `c32114a` | Accepted as seed-0 baseline | Covers Weissbart 13 subjects and Etard 20 subjects with `ridge`, `cca`, `fcnn`, and `adt` at seed 0. Execution reported 132/132 successful jobs, no failures, compact result artifacts only. |
| `fix/ar-20260625-161300-a43831b-multi-seed-v1` | `dee590b26aca180637b920b9579a59daa9ba176c` | Full-subject multi-seed v1 | `e459f2c` | Accepted with minor reporting caveats | Covers Weissbart 13 subjects and Etard 20 subjects with `ridge`, `cca`, `fcnn`, and `adt` at seeds `0`, `42`, and `2026`. Execution reported 396/396 successful jobs and no failures. No full prediction dumps or checkpoints were detected. Caveats: manuscript scaffold was not fully updated for multi-seed results, and many per-job metric files were tracked; future rounds should keep artifacts more compact. |
| `fix/ar-20260625-161300-a43831b-model-expansion-v1` | `68f87a4c0005e4556a51b83cd42e73908edc44ad` | Full-subject model expansion v1 | `dee590b2` | Accepted for result use; metadata cleanup required | Covers Weissbart and Etard with seeds `0`, `42`, and `2026`; carries forward `ridge`, `cca`, `fcnn`, `adt` and adds `dnn`, `cnn`, `eegnet`. Schema report passed with 693/693 subject-level jobs and 0 failures. Caveats: `run_manifest.json` references unpushed local-only files (`subject_metrics.csv`, `recording_metrics.csv`, `job_ledger.*`, `skipped_models.md`), and `workflow/START_HERE.md` still describes pre-push state. These must be cleaned before using this branch as a polished audit package. |
| `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure` | `d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae` | Metadata closure for model expansion v1 | `68f87a4` | Accepted as reviewable artifact closure; local hygiene caveat | No model rerun. Publishes the missing compact manifest-referenced artifacts: `subject_metrics.csv`, `recording_metrics.csv`, `job_ledger.csv`, `job_ledger.json`, and `skipped_models.md`; updates `workflow/START_HERE.md`. Remote checks found no prediction dumps, checkpoints, model weights, raw data, or per-job directories in this commit. Caveat: implementer reported unrelated local residue remains in the worktree, so the next execution round must first clean, ignore, or isolate those files before claiming `published_for_review`. Scientific result baseline remains `68f87a4`; practical next-branch base should be this closure commit. |

## Manuscript Notes

- The richer local manuscript draft is in the workspace under `paper/main.tex` and `paper/sections/`.
- The GitHub `paper/draft_zh/` files produced by the implementer are pipeline-validation scaffolds and should be treated as source material, not the primary manuscript.
- Future manuscript consolidation should migrate validated methods/results from GitHub outputs into the richer local `paper/main.tex` structure.

## Next Expected Execution Round

Proposed branch:

- `fix/ar-20260625-161300-a43831b-reconstruction-analysis-pilot-v1`

Base:

- `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure @ d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`

Scope:

- Begin with worktree hygiene: remove, ignore, or isolate unrelated local residue before starting the new analysis branch. Do not commit stale `README.md` edits, per-job directories, smoke directories, caches, prediction dumps, checkpoints, model weights, or raw data.
- Do not add more models in this immediate round. Use the validated full-subject reconstruction result set to build the paper analysis spine.
- Produce compact analysis outputs for: main dataset-by-model ranking, seed stability, per-subject variability, and Etard condition robustness.
- Add a first subject-independent / leave-one-subject-out feasibility plan or minimal pilot using a small fast subset before scaling.
- Generate figure-ready tables and manuscript-writing notes, but do not replace the reviewer-owned primary manuscript without review.
