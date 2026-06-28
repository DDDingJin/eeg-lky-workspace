# Focused vs Pilot Comparison

This comparison is diagnostic-only. It contrasts the current Etard `P00` article pilot with a focused rerun under a larger training budget, while keeping the same split, scorer, and result schema.

## Direct answers
1. `FCNN` did not improve: pilot `0.030900` vs focused `0.030900`. `ADT` improved clearly: pilot `0.056455` vs focused `0.085343`.
2. The ADT improvement indicates that training budget is a major contributor for ADT. The FCNN non-improvement indicates that budget alone is not the main explanation for FCNN.
3. Because FCNN did not improve, the next suspects are model protocol mismatch, window definition, target alignment, or implementation differences relative to the old baseline run, not simply insufficient epochs.
4. Focused rerun ADT (`0.085343`) is now very close to the old Etard ADT-exact mean (`0.085971`). Focused rerun FCNN (`0.030900`) remains well below the old FCNN mean (`0.062108`).
5. Even after focused rerun, this is still a single-subject P00 diagnostic and should not be written as a formal benchmark conclusion.
6. The result supports a mixed recommendation: ADT is now less concerning and could be taken into a larger single-seed rollout, but FCNN still needs a protocol-focused check before promoting the whole suite to full-subject.

## Recommendation
- Recommendation `C`: continue focused protocol checking before full-subject. Specifically, ADT no longer shows the same level of concern, but FCNN still does.
