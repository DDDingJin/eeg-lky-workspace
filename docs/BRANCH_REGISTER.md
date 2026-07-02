# Branch Register

Last updated: 2026-07-02

This file is maintained by the review endpoint. The implementer endpoint should not edit it during experiment execution. Each execution round should report its branch and commit; the review endpoint will update this register after review.

## Source Of Truth

- Rule branch: `rule/research-audit-loop-v1`
- Rule commit: `032c81484f1ccf141e5991a0a9d70a3294d8d532`
- Publish-gate rule branch: `rule/research-audit-loop-v1.1-publish-gate`
- Publish-gate rule commit: `365c7c2b034af2abd1e6283adde980de6de7eeed`
- Review package branch: `review/ar-20260625-161300-a43831b-blueprint`
- Original reviewed implementation baseline: `audit/reproduction-note @ a43831b83598c82520be320c21b56e92b73b7dcd`
- Current accepted execution base for the next code round: `fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1 @ ce804cf328981bf43f9ebb35b7ca2a3e75cd6c63` after the branch is cleaned of unintentionally included `loso_all_models_single_seed_v1` blocked artifacts; use `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a` as the clean fallback base if hygiene is uncertain.
- Latest full-subject result branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1 @ 68f87a4c0005e4556a51b83cd42e73908edc44ad`
- Latest reviewable full-subject artifact branch: `fix/ar-20260625-161300-a43831b-model-expansion-v1-metadata-closure @ d0ea16fe57c7a92bcbc459ed5f5e67fd529c86ae`
- Latest subject-independent LOSO pilot branch: `fix/ar-20260625-161300-a43831b-loso-pilot-v1 @ fb3494d6edb3b54d793288e739d6c37c4fb34e5f`
- Latest subject-independent Ridge LOSO branch: `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1 @ f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a`
- Latest non-Ridge LOSO adapter smoke branch: `fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1 @ ce804cf328981bf43f9ebb35b7ca2a3e75cd6c63`

## Branches

| Branch | Tip commit | Role | Base / predecessor | Review status | Notes |
|---|---|---|---|---|---|
| `audit/reproduction-note` | `a43831b83598c82520be320c21b56e92b73b7dcd` | Original reviewed implementation branch | Earlier reproduction-note work | Locked baseline | Target commit for review round `AR-20260625-161300-a43831b`. |
| `rule/research-audit-loop-v1` | `032c81484f1ccf141e5991a0a9d70a3294d8d532` | Shared reviewer/implementer rule and skill branch | Rule proposal iterations | Accepted rule lock | Contains `AGENTS.md` and `skills/research-audit-loop/`. |
| `rule/research-audit-loop-v1.1-publish-gate` | `365c7c2b034af2abd1e6283adde980de6de7eeed` | Shared rule update for experiment completion status | `rule/research-audit-loop-v1` | Accepted rule update | Adds the publish gate and branch-register ownership rule: local runs are not review-complete until committed, pushed, remotely inspectable, and reported for reviewer-owned register maintenance. |
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
| `fix/ar-20260625-161300-a43831b-loso-pilot-v1` | `fb3494d6edb3b54d793288e739d6c37c4fb34e5f` | Minimal subject-independent LOSO pilot | `d0ea16f` | Accepted as interface pilot; not article-grade full LOSO | Covers `weissbart_tf64`, `ridge`, seed `0`, held-out subjects `P00`, `P01`, and `P02`. Schema validation passed with 3/3 jobs and 0 failures. Leakage check documents pure LOSO: non-heldout train split for fitting, nonheldout val split for alpha selection, heldout test split for final evaluation. Results are lower than subject-specific Ridge as expected. Caveat: `workflow/START_HERE.md` still says the next step is commit and push; next branch should refresh handoff metadata after publication. |
| `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1` | `f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a` | Full all-subject subject-independent Ridge LOSO | `fb3494d` | Accepted as Ridge LOSO baseline; alpha-grid caveat | Covers `weissbart_tf64` 13 subjects and `etard_tf64` 20 subjects with pure LOSO Ridge at seed `0`. Schema validation passed with 33/33 jobs, 867 recording rows, 33 subject rows, 2 dataset summaries, 6 Etard condition summaries, and 33 subject-specific comparison rows. Mean LOSO performance is lower than subject-specific Ridge by `-0.051570`, supporting the expected subject-independent difficulty gap. Caveats: all jobs selected the smallest alpha in the tested grid (`0.001`), so run a Ridge alpha-grid boundary diagnostic before treating the LOSO Ridge row as final; `workflow/START_HERE.md` still contains pre-push wording. |
| `fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1` | `ce804cf328981bf43f9ebb35b7ca2a3e75cd6c63` | Non-Ridge pure LOSO adapter smoke matrix | `f546c08a` | Accepted as adapter-readiness evidence; cleanup required before next base | Covers `weissbart_tf64`, heldout `P00`, seed `0`, and requested models `eegnet`, `fcnn`, `cnn`, `adt`, `dnn`, `cca`. Schema validation passed; `eegnet`, `fcnn`, `cnn`, `adt`, and `dnn` succeeded under lazy pure-LOSO adapters, while `cca` was skipped with a documented lag-matrix memory scaling reason. Leakage check excludes heldout `P00` from train, val, normalization/scaler fitting, hyperparameter selection, and checkpoint selection. Caveat: this branch also includes the previously blocked `gate0_gate2_loso_all_models_single_seed_v1` config, runner, and result directory; those are not part of the accepted smoke evidence and must be removed or isolated before using this branch as the base for a full non-Ridge LOSO run. |

## Manuscript Notes

- The richer local manuscript draft is in the workspace under `paper/main.tex` and `paper/sections/`.
- The GitHub `paper/draft_zh/` files produced by the implementer are pipeline-validation scaffolds and should be treated as source material, not the primary manuscript.
- Future manuscript consolidation should migrate validated methods/results from GitHub outputs into the richer local `paper/main.tex` structure.

## Next Expected Execution Round

Proposed branch:

- `fix/ar-20260625-161300-a43831b-loso-nonridge-full-single-seed-v1`

Base:

- Preferred clean base: `fix/ar-20260625-161300-a43831b-loso-nonridge-data-interface-smoke-v1 @ ce804cf328981bf43f9ebb35b7ca2a3e75cd6c63` after removing or isolating the unintended `gate0_gate2_loso_all_models_single_seed_v1` artifacts.
- If cleanup is cumbersome, start from `fix/ar-20260625-161300-a43831b-loso-ridge-full-v1 @ f546c08ab6e7a3e073e0b35ac9324d7b9bd3a57a` and port only the validated lazy non-Ridge adapter smoke code.

Scope:

- First do a small hygiene closure if needed: remove the unintentionally included blocked `gate0_gate2_loso_all_models_single_seed_v1` config, runner, and result directory from the next execution base, or clearly isolate them as `blocked_context` outside the active benchmark artifact path.
- Run full subject-independent pure LOSO at seed `0` for both `weissbart_tf64` and `etard_tf64`, all held-out subjects, using the lazy non-Ridge data interfaces validated by the smoke matrix.
- Main models for this round: `eegnet`, `fcnn`, `cnn`, `adt`, and `dnn`. Reuse the already accepted Ridge LOSO result for comparison; do not rerun Ridge unless the runner needs a consistency check.
- Do not force `cca` through the current pooled lag-matrix path. Keep `cca` as a separate scalability/path-design issue unless a memory-safe CCA implementation is introduced and separately smoke-tested.
- Output compact artifacts only: `run_manifest.json`, `failure_report.json`, `schema_validation_report.json`, `subject_metrics.csv`, `recording_metrics.csv`, `dataset_metrics.csv`, `model_dataset_summary.csv`, `etard_condition_metrics.csv`, `loso_vs_subject_specific_comparison.*`, `loso_vs_ridge_loso_comparison.csv`, `leakage_summary.md`, `result_summary.md`, and `incremental_comparison.md`.
- The report must state total planned/success/failed/skipped jobs, exact branch/base commits, whether any results are reused, and which artifacts are newly generated. No raw data, prediction dumps, checkpoints, model weights, per-job directories, cache files, or `.pt/.pth/.npy/.npz/.h5/.mat` files should be pushed.
