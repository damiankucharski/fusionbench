# Could random test labels be hiding the gain on 16 scenes?

This post hoc diagnostic revisits the 64 completed scene fits in round five. All 192 test forecast hashes matched the archived values for ordinary full-data OKAPI, the confirmed candidate and arithmetic averaging. The public simulator provides the positive probability of each label conditional on the latent scene and row signal. For each original test row we evaluated the same frozen forecast against both possible labels and weighted the two full-class Brier losses by those probabilities. This integrates away test-label noise while preserving the fitted tree, the 2,048 realized test inputs, and the variation between training draws.

| Training design | Candidate minus full-data OKAPI, observed labels | Same forecasts, conditional expected Brier | Candidate minus averaging, conditional expected Brier |
| --- | ---: | ---: | ---: |
| 16 scenes × 8 rows | −.00235 [−.00580, +.00089] | −.00175 [−.00481, +.00111] | +.00657 [+.00478, +.00878] |
| 128 scenes × 1 row | −.00405 [−.00826, −.00060] | −.00462 [−.00835, −.00176] | +.00440 [+.00319, +.00567] |

Negative values favor the candidate. Intervals use the same 10,000 paired resamples of the 32 draws as the primary round-five comparisons. With 16 training scenes, the gain estimate moved closer to zero after integrating over the possible test labels and its interval still spans zero. With 128 training scenes the interval remained below zero. The observed ambiguity therefore persists with exact conditional evaluation on these particular test inputs. The original primary decision stands. [Paired contrasts](results/round5-scene-posterior-contrasts.csv) · [All 192 matched forecasts](results/round5-scene-posterior-scores.csv) · [Reproduction](scene_posterior.py)
