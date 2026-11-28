# Round 4. Choosing one native tree by cross-validation

A four-fold source comparison chose between the regularized native-pattern search and a regularized simplex-weighted tree. Both saw the same three training folds for each held fold. The decision used the difference in their out-of-fold full-class Brier losses on each source row. A positive average gain selected native-pattern under the zero-margin rule. The more conservative rule required the gain to exceed one paired standard error. Otherwise each rule selected the simplex. The row-based margin is heuristic because training sets overlap across folds. Each deployed model was fitted on the full source bank and produced predictions on its own. [Protocol](protocol.md) · [Implementation](r4.py)

| Evaluation | Banks | Selected native-pattern, one-SE | One-SE Brier | Original native-pattern Brier | One-SE minus original, 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Equal weighting | 16 | 0 | .48382 | .54442 | −.06060 [−.07905, −.04391] |
| Weak specialization (primary) | 16 | 0 | .48002 | .53893 | −.05891 [−.07954, −.03995] |
| Strong specialization | 16 | 2 | .44300 | .45692 | −.01393 [−.04288, +.00711] |
| Standard-v5 | 54 | 46 | .25498 | .25607 | −.00109 [−.00181, −.00038] |
| Digits (449 fusion rows) | 4 | 4 | .02926 | .02926 | .00000, descriptive |

On weak specialization the one-SE rule selected the simplex on all 16 banks, beating arithmetic averaging by .00186 Brier [−.00335, −.00016]. At equal weighting the arithmetic mean is the exact population posterior and won on all 16 banks. The zero-margin rule chose native-pattern on one additional weak bank, which brought its mean weak-specialization loss to .48195. The other difference between the two rules was on one strong-specialization bank. On digits both retained native-pattern every time. Probability logistic stacking had lower mean loss there (.02443). [All method levels](results/round4-levels.csv) · [Paired contrasts](results/round4-contrasts.csv)

On the 27 fixed standard families the one-SE rule reduced average loss in two families, matched the original model in 22 and increased it in three. The largest improvements were in `stratified_pool` (−.01960) and `redundant_pool` (−.01863). Regressions were `native_close_switch` (+.00433), `environment_shortcut` (+.00377) and `rare_specialist` (+.00067). Each family contributed two data seeds, so the interval describes seed variation within this suite. The digits splits overlap and represent one dataset. [Family differences](results/round4-families.csv)

The 106 banks used one numerical thread for 254 seconds, including references. Each decision required four fits of **each** candidate on three quarters of the source rows and one full fit of the selected algorithm. Evaluating both standalone controls added one full fit per bank. Both decision rules shared all their fits and every deployed prediction matched the selected standalone model exactly by SHA-256. Regenerate the per-bank selection table with `python r4_decisions.py` after running `python run.py --round 4` in this directory. [Decisions and realized losses](results/round4-decisions.csv) · [Fold sizes and fit times](results/round4-fits.jsonl)
