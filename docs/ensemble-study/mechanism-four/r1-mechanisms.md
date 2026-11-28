# How shared errors change the value of averaging

With equally noisy providers and independent provider errors, the eight-provider mean scored .13670 in conditional expected Brier. The full-data, mean-seeded OKAPI incumbent scored .14185. Turning on heterogeneous noise, shared errors and a local specialist on the **same examples** reversed their order. The mean scored .15001 and the incumbent .14151. The paired incumbent-minus-mean difference changed from +.00515 (95% interval +.00259 to +.00805) to −.00851 (−.01397 to −.00149). The average difference over all eight settings was only +.00035 (−.00249 to +.00381). An aggregate close to zero can conceal opposite conclusions in controlled settings. [Paired estimates](results/r1-contrasts.csv)

Each of eight independent draws supplied 16 training scenes and 256 fresh test scenes. The table integrates out the Bernoulli test labels using their known probabilities on these scenes. The H setting introduces differences in the providers' residual noise scales. C gives all eight providers a shared component explaining 65% of residual logit variance. S improves one provider inside the specialist region, which occurs in 20% of scenes in expectation.

| H | C | S | Arithmetic mean | Full-data OKAPI incumbent | Ordinary native GP | Top-N | Logistic stacking |
|---|---|---|---:|---:|---:|---:|---:|
| Off | Off | Off | .13670 | .14185 | .14477 | .14434 | .16016 |
| Off | Off | On | .13609 | .13931 | .14376 | .14338 | .16241 |
| Off | On | Off | .15585 | .16232 | .15647 | .16393 | .17281 |
| Off | On | On | .15450 | .16056 | .15236 | .16190 | .17446 |
| On | Off | Off | .13566 | .13438 | .13365 | .13549 | .15349 |
| On | Off | On | .13520 | .13350 | .13363 | .13463 | .15457 |
| On | On | Off | .15100 | .14435 | .13991 | .14283 | .15541 |
| On | On | On | .15001 | .14151 | .14001 | .14084 | .15485 |

The mean had lower conditional expected Brier than LR in all eight cell averages. Its average advantage over LR was .01664 (.00545 to .02912). Switching H on narrowed the advantage by .01006 (.00380 to .01674) when averaged over C and S. Switching C on narrowed it by .01020 (.00117 to .01861). With both H and C enabled, the interval for the mean-minus-LR contrast included zero at either level of S. Across all 30 standard-v6 families LR performed better than the mean (.26381 versus .37214 observed Brier). The historical split-based model, ordinary GP, Top-N and compact pool were refit in each condition. [All absolute levels](results/r1-levels.csv) · [Factorial effects](results/r1-factorial-effects.csv) · [All standard families](results/r1-families.csv)

## An exact calculation for the mean

For one binary example let the true event probability be `q`, the provider probabilities `p_j`, and their arithmetic mean `m`. The excess full-class Brier of the mean is `2 (m − q)²`. Averaged over test examples, this equals the average provider's excess Brier minus the gain from disagreement, `2 × average[(p_j − m)²]`. The identity holds for every fixed set of forecasts and event probabilities. The program checks it draw by draw and reproduces the recorded arithmetic mean risk on all 64 factorial banks. [Calculation and reproducible code](geometry_four.py) · [Per-draw quantities](results/r1-geometry.csv)

Turning C on increased the mean's risk by **.01693** (.01544 to .01913). The amount of loss removed by averaging fell by **.01833** (.01689 to .01981), while the average individual provider's loss changed by −.00140 (−.00413 to +.00202). The shared shock primarily reduced the benefit of combining the reports. Turning S on lowered the specialist's excess loss by .03916 inside its region and left it identical outside. The eight-provider mean captured .00464 of excess-loss reduction inside that region, or just .00085 across the full audit. These are consequences of the specified data law and eight sampled banks. [Paired decomposition](results/r1-geometry-effects.csv)

## Repeat on the 24 confirmation draws

The original comparison of uncalibrated incumbent versus provider mean could be read again on the independent round-four draws. With all switches off, the incumbent had higher conditional Brier by .00630 (.00388 to .00958). With all switches on, it had lower conditional Brier by .01643 (.01460 to .01823). In the latter setting the mean and logistic stacker were within .00001 of each other (mean minus LR interval −.00732 to +.00631). The reversal of the mean's ordering relative to the incumbent thus recurred while the previously small difference from LR was unresolved. These controls are separate from the prespecified temperature-calibration confirmation. [Round 4 paired comparisons](results/r4-contrasts.csv)
