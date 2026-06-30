# Branch Register

Last updated: 2026-06-30

This file is maintained by the review endpoint. The implementer endpoint should not edit it during experiment execution. Each execution round should report its branch and commit; the review endpoint will update this register after review.

## Source Of Truth

- Rule branch: `rule/research-audit-loop-v1`
- Rule commit: `032c81484f1ccf141e5991a0a9d70a3294d8d532`
- Review package branch: `review/ar-20260625-161300-a43831b-blueprint`
- Original reviewed implementation baseline: `audit/reproduction-note @ a43831b83598c82520be320c21b56e92b73b7dcd`
- Current accepted execution base for the next code round: `fix/ar-20260625-161300-a43831b-multi-seed-v1 @ dee590b26aca180637b920b9579a59daa9ba176c`
- Latest full-subject result branch: `fix/ar-20260625-161300-a43831b-multi-seed-v1 @ dee590b26aca180637b920b9579a59daa9ba176c`

## Branches

| Branch | Tip commit | Role | Base / predecessor | Review status | Notes |
|---|---|---|---|---|---|
| `audit/reproduction-note` | `a43831b83598c82520be320c21b56e92b73b7dcd` | Original reviewed implementation branch | Earlier reproduction-note work | Locked baseline | Target commit for review round `AR-20260625-161300-a43831b`. |
| `rule/research-audit-loop-v1` | `032c81484f1ccf141e5991a0a9d70a3294d8d532` | Shared reviewer/implementer rule and skill branch | Rule proposal iterations | Accepted rule lock | Contains `AGENTS.md` and `skills/research-audit-loop/`. |
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

## Manuscript Notes

- The richer local manuscript draft is in the workspace under `paper/main.tex` and `paper/sections/`.
- The GitHub `paper/draft_zh/` files produced by the implementer are pipeline-validation scaffolds and should be treated as source material, not the primary manuscript.
- Future manuscript consolidation should migrate validated methods/results from GitHub outputs into the richer local `paper/main.tex` structure.

## Next Expected Execution Round

Proposed branch:

- `fix/ar-20260625-161300-a43831b-model-expansion-v1`

Base:

- `fix/ar-20260625-161300-a43831b-multi-seed-v1 @ dee590b26aca180637b920b9579a59daa9ba176c`

Scope:

- Expand the already validated full-subject multi-seed setting to a small, interpretable next model family before attempting large specialized models.
- Datasets: `weissbart_tf64` and `etard_tf64`.
- Baseline models carried forward: `ridge`, `cca`, `fcnn`, `adt`.
- New model candidates for this round: `cnn` and `eegnet` if both are already implemented and registered; otherwise start with `cnn` only.
- Seeds: use `0`, `42`, and `2026` for new stochastic models. Reuse existing results for `ridge`, `cca`, `fcnn`, and `adt`; do not rerun completed jobs unless a schema-breaking bug is found.
- Required outputs: compact per-subject/per-dataset metrics with `seed`, model-expansion summary tables, updated Etard condition summary for new models, incremental comparison against multi-seed v1, and manuscript-ready notes.
- Artifact policy: keep aggregate metrics, ledgers, manifests, and concise reports; do not upload prediction dumps, checkpoints, or per-job directories unless needed for a specific audit.
- No VLAAI, HappyQuokka, NULL, KUL/SparrKULee, or cross-dataset transfer in this immediate round unless explicitly approved after the first expansion check.
