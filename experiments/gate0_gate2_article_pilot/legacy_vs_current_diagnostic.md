# Legacy vs Current Diagnostic

This diagnostic compares historical summary tables against the current single-subject article pilot.
It is diagnostic-only and should not be interpreted as a new benchmark result.

## Direct answers
1. The user's memory that some old Etard bars were around `0.1+` is correct. In the old summaries, `eegnet, happyquokka_gcon, null_gcon, vlaai_exact` were at or above `0.1`.
2. Old Etard models clearly above `0.1` include `eegnet` (`0.1003`), `vlaai_exact` (`0.1128`), `happyquokka_gcon` (`0.1287`), and `null_gcon` (`0.1480`). `ridge` was close at `0.0982` and `cnn` at `0.0956`.
3. Old Etard `ADT-exact` itself was `0.08597070755065966`, not `0.1+`.
4. Current Etard P00 `ridge` (`0.0886`) and `cca` (`0.0638`) are in the same rough magnitude as the old all-subject means (`0.0982` and `0.0680`), though still not directly comparable.
5. Current Etard P00 `fcnn` (`0.0309`) and `adt` (`0.0565`) are lower than the historical means (`0.0621` and `0.0860`). The strongest likely reasons are: single-subject P00 evaluation, only 3--5 pilot epochs for FCNN and 3 epochs for ADT in the current pilot, current aggregation over 24 mixed-condition recordings, and mismatch between pilot protocol and the older longer-budget summaries.
6. The current P00 pilot and old summaries are not directly comparable. The old tables aggregate 13 or 20 subjects, often with longer training budgets or exact-port-specific protocols, while the current pilot is a single-subject closure run meant to validate pipeline connectivity.
7. The observed gap does not by itself prove a pipeline bug. The ridge/CCA values staying near the historical scale argues against a gross scorer or alignment failure. The more plausible explanations are single-subject variance, mixed-condition composition in Etard, short pilot budgets, and protocol differences between the current runner and older summary-generating runs.

## Preliminary recommendation
- This evidence supports `C`: do not jump to full-subject immediately. First do a focused rerun that makes FCNN/ADT closer to the old protocol or epoch budget on Etard P00 (and optionally Weissbart P00 for symmetry).
