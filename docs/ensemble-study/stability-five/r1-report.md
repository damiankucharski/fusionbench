# Round 1 · Limiting the GP programs eligible for validation

On the prespecified weak-specialization task, restricting the validation menu
to eight discovery-ranked GP programs reduced full-class Brier from **0.56858
to 0.50533**. The paired change was **−0.06325** (95% bootstrap interval
−0.09079 to −0.03522), with a reduction in 14 of 16 independent training
banks. Arithmetic averaging still scored **0.48188** and won on all 16
banks. The eight-program rule also had higher Brier than ranked Top-N by
0.01367 (0.00510 to 0.02352). [Protocol](protocol.md) · [Paired contrasts](results/round1-contrasts.csv)

## Where the validation gain went

Each 128-row bank supplied 89 discovery rows and 39 validation rows. The
unrestricted validation search compared an average of 367 distinct GP
programs. Its winning GP candidate had median **discovery rank 305**. It beat
both affine heads on validation in all 16 weak-specialization banks. Its
average validation Brier was 0.41300, but on the exact independent population
the same tree scored 0.56858.

Under the eight-program cap, validation Brier rose to 0.50834. On the exact
population the same pre-refit winners scored 0.50654. The signed difference
between population and validation Brier changed from +0.15558 to −0.00180,
a paired reduction of 0.15738 (0.10755 to 0.20589). Ten winners were GP
programs, five were flat heads, and one was an augmented head. Refitting
the selected head when applicable gave the deployment score of 0.50533.
Ranking the archive by discovery loss modifies both the number and the
quality of eligible candidates. The experiment cannot allocate the gain
between those two changes. [Validation and family decisions](results/round1-validation.csv)

## Controls and transfer

Absolute full-class Brier is averaged across banks. The standard column gives
equal weight to all 27 fixed families, with two seeds each. Digits uses four
overlapping partitions of one dataset.

| Decision | Equal | Weak | Strong | Standard | Digits, 449 fusion rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing regularized OKAPI | .54817 | .56858 | .45567 | .26347 | .03424 |
| Discovery top 8 | .51981 | .50533 | .42708 | .26345 | .03309 |
| Discovery top 1 | .52626 | .50250 | .43085 | .26372 | .03325 |
| Arithmetic mean | .48188 | .48188 | .48188 | .40172 | .06136 |
| Ranked Top-N | .50695 | .49166 | .43430 | .38168 | .03573 |
| Probability logistic stacking | .52111 | .50421 | .43667 | .28933 | .02690 |

The eight-program rule also improved relative to unrestricted OKAPI with
equal competence (−0.02836, interval −0.04865 to −0.00811) and strong
specialization (−0.02859, −0.05764 to −0.00604). On the standard panel the
mean change was −0.00002 (−0.00297 to +0.00294). Five family means
improved, one worsened, and 21 tied within $10^{-12}$. The largest changes
went in opposite directions on `rare_shortcut_noise` (−0.02962) and
`environment_shortcut` (+0.04721). On digits the change was −0.00115 across
the four paired partitions. [Family results](results/round1-families.csv)
· [All six methods](results/round1-levels.csv)

## Reproduction and scope

All three native policies share a single 600-evaluation GP trajectory, the
same discovery-trained heads and their full-data refits. Within a shortlist,
validation ties are resolved by discovery loss and then first visit. Choosing
among flat, augmented and GP families preserves the historical order on ties.
The runner captures a tree's discovery predictions when native GP first
scores its unique structure. It records the full baseline and replays the
validation pass for comparison. This measures the effect of restricting
eligibility at fixed discovery cost. The normal implementation can omit
validation of excluded candidates by setting `gp_validation_top_k=8` on
either native-pattern adapter. It keeps one cached validation result per
eligible candidate across checkpoints. The timings archived for this study
include the unrestricted comparison and its diagnostic replay.

On every bank the unrestricted replay had the same GP winner, family and all
three family validation scores as the existing adapter. Additional binary
and five-class regression tests compared the observer against an independent
ordinary fit, including bitwise equality of final predictions. The completed
one-thread run took 68.1 seconds across 106 banks and archived 636 scored
forecasts. [Implementation](r1.py) · [Per-fit audit](results/round1-fits.jsonl)
· [Runtime](results/round1-run.json)

After confirmation, the ordinary public adapter with
`gp_validation_top_k=8` reproduced all 220 fifth-round audit prediction
hashes, GP winners and chosen families. GP validation used **3,500** native
forwards across initial and final checkpoints, compared with **80,180**
for unrestricted GP validation on the same archives. Discovery search and
both heads still run. [Independent API replay](results/round5-shortlist-api-verified.json)

```sh
PYTHONPATH=.:/path/to/OKAPI BACKEND=numpy OKAPI_LOG_LEVEL=ERROR python docs/ensemble-study/stability-five/run.py --round 1
python docs/ensemble-study/stability-five/r1_diagnostics.py
```
