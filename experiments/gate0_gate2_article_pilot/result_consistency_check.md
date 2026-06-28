# Result Consistency Check

This file checks subject-level Pearson consistency across:
- `experiments/gate0_gate2_article_pilot/subject_metrics.csv`
- `experiments/gate0_gate2_article_pilot/run_manifest.json`
- `experiments/gate0_gate2_article_pilot/result_summary.md`
- `paper/draft_zh/sections/results_placeholder_zh.tex`

| dataset | model | subject_metrics.csv | run_manifest.json | result_summary.md | paper | status | note |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| etard_tf64_p00 | adt | 0.056455132402975 | 0.056455132402975 | 0.056455 | 0.056 | PASS | consistent after fix |
| etard_tf64_p00 | cca | 0.063780967582749 | 0.063780967649936 | 0.063781 | 0.064 | PASS | consistent after fix |
| etard_tf64_p00 | fcnn | 0.030899725920548 | 0.030899725920548 | 0.030900 | 0.031 | PASS | consistent after fix |
| etard_tf64_p00 | ridge | 0.088560154515784 | 0.088560154515784 | 0.088560 | 0.089 | PASS | consistent after fix |
| weissbart_tf64_p00 | adt | 0.162950746012409 | 0.162950746012409 | 0.162951 | 0.163 | PASS | consistent after fix |
| weissbart_tf64_p00 | cca | 0.064111402207058 | 0.064111402035428 | 0.064111 | 0.064 | PASS | consistent after fix |
| weissbart_tf64_p00 | fcnn | 0.050608068098467 | 0.050608068098467 | 0.050608 | 0.051 | PASS | consistent after fix |
| weissbart_tf64_p00 | ridge | 0.132547334469791 | 0.132547334469791 | 0.132547 | 0.133 | PASS | consistent after fix |

## Conclusion
- All subject-level Pearson values are now synchronized across CSV, manifest, summary, and paper text.
- The previous Etard ADT paper value (`0.070`) has been corrected to `0.056` in the paper draft.
