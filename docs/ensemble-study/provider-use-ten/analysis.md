# Reading this study

Each standard score gives equal weight to the 30 fixed scenario families. Every paired interval resamples source seeds within families with 10,000 draws at 93000 plus the round number. Each focal estimate averages over source seeds within one source design. Its Brier score integrates over all 16 report patterns and both possible labels under the specified target law.

The two maxima answer different questions. Worst Brier is the larger of the losses when shortcut activity is 0% and 100%. The loss above the Bayes oracle, or regret, first subtracts the best possible predictor's Brier in that same world. Maximum regret takes the larger of these two excess losses. Both maxima are computed separately for every procedure on every source seed before averaging or estimating uncertainty. A method can lower the harder world's absolute Brier yet increase its shortfall in a world where reliable shortcut information is available. The binding-endpoint table records which world sets each maximum.

The source diagnostic can flag a provider without attenuating it. Strength zero preserves its native forecast, half averages the native forecast with a forecast marginalized over donor reports, and full replaces the native forecast with the marginalized one. The hard baseline independently refits a fusion model without the provider. Provider counts on the soft paths describe the original fitted tree. They do not measure extra tree evaluations for donor reports.

| Table | Interpretation |
|---|---|
| `rXX-levels.csv`, `rXX-families.csv` | Absolute risks, secondary outcomes and per-family means. The `metric` field identifies Brier, worst Brier or maximum regret in the `risk` column. |
| `rXX-contrasts.csv` | Method minus reference, with a paired 95% interval. A negative risk difference favors the method. |
| `rXX-diagnostics.csv` | Bank counts, flags, direct and routing gains, chosen strengths and changes from the native forecast. Each source is counted once regardless of the number of worlds scored. |
| `rXX-role-diagnostics.csv` | Which simulated role was flagged and whether direct gain, routing gain, both or neither exceeded .02 on that source. Role labels are available only for the synthetic focus panels. |
| `rXX-binding.csv` | Source counts where each endpoint determines the maximum. Differences no larger than 1e-12 are marked as ties for display. Selection always uses the unrounded risk. |
| `r03-gate-utility.csv`, `r10-gate-utility.csv` | Paired direct and routing gains of the known selector with its original reports and after shuffling. These gains are computed even when the screen abstains. |

Round 9 compares the three prespecified rules on mixed maximum regret after checking their empirical average standard Brier against the +.005 margin. The baseline native method is the fallback if no rule qualifies. Round 10 verifies the committed decision and reports the primary and each safeguard separately. The unconditional candidate tables remain interpretable when the fallback is selected.
