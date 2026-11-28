# Reading the five-round results

All score differences subtract the reference from the named method on the same evaluation data. Lower full-class Brier favors the method. Higher macro-F1 and positive-class average precision (AP) favor the method. The three metrics measure different properties. A change in probability calibration can improve Brier while leaving predicted labels and therefore F1 unchanged.

## What each table measures

* `roundN-levels.csv` gives absolute scores. The standard row weights each of the 27 fixed scenarios equally. Weak, equal and strong specialization use exact population risk. Scene results use independent evaluation scenes from the training law. Digits uses held-out images from resplits of one real dataset.
* `roundN-families.csv` gives the same scores for each standard scenario. The `delta_full` column subtracts ordinary OKAPI trained and selected on all available fusion rows. Positive values identify Brier regressions. AP is available for binary tasks. The 27-family aggregate leaves AP blank because it includes multiclass tasks.
* `roundN-contrasts.csv` reports paired differences, uncertainty intervals and counts of better, tied and worse draws. The `gap` metric is the audit Brier minus the Brier measured across all supplied fusion rows. In the reduced and split controls from round 1, that latter quantity includes rows omitted from evolution. The performance comparisons between those controls use audit Brier.
* `round5-scene-interaction.csv` asks whether a method's gain relative to a reference changes when 128 observations come from 16 or 128 scenes. For each seed, both fits use the same underlying pool and are scored on identical evaluation scenes. First take the paired method difference within each condition, then subtract the 128-scene difference from the 16-scene difference. Resampling entire seed pairs preserves this dependence.

## Uncertainty

Each prespecified round uses new data seeds and a fixed optimizer seed. A two-sided percentile interval uses 10,000 resamples with seed `91700 + round`. Each draw resamples paired banks within every fixed scenario. Row and pixel counts do not enter the number of independent repetitions. The 95% intervals are marginal and reflect the benchmark and optimizer setting in this study. Only the prespecified confirmation criteria govern the overall research decision. The digits comparisons are descriptive because images recur across partitions. Absolute differences up to 1e−12 count as ties.

## Reproduction

From the FusionBench checkout, after each round's score file has been generated, use the corresponding number in the first command. Render the figure after the fifth round using the frozen selection record.

```sh
python docs/ensemble-study/full-data-five/analyze_full.py 1
python docs/ensemble-study/full-data-five/plot_full.py
```

For example, `method=split`, `reference=reduced` and `metric=brier` in the first round compare two terminal choices for the same evolved population. In the fifth round, filter the same columns to the confirmed method and `reference=mean` to compare directly to arithmetic averaging.
