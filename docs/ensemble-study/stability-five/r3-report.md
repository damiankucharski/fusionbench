# Round 3. Choosing one representative bootstrap tree

On weak specialization, the representative tree had mean population Brier **0.51817**, compared with **0.53119** for ordinary regularized OKAPI. Its paired difference was **−0.01302** (95% interval **−0.03156 to +0.00314**). It improved on seven of the 16 training banks, tied on six, and worsened on three. Arithmetic averaging scored **0.48188** and beat the representative on all 16. The single-tree candidate met the prespecified point-estimate screening thresholds on strong specialization, the standard panel and digits. [Protocol](protocol.md) · [Paired comparisons](results/round3-contrasts.csv)

The procedure fitted the regularized native pipeline on the full source bank and three equally sized, within-class bootstrap samples. Among these four programs it selected the one closest to their arithmetic prediction on the original source inputs, minimizing squared full-class distance averaged over rows. The original model won exact ties. Deployment then evaluated only the chosen native tree. The reported four-model average evaluates all four trees and is a separate diagnostic.

| Evaluation | Banks | Ordinary regularized Brier | Representative Brier | Representative minus regularized (95% interval) | Four-model average Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal competence | 16 | 0.53294 | 0.53707 | +0.00413 (−0.00609, +0.01537) | 0.51313 |
| Weak specialization | 16 | 0.53119 | 0.51817 | −0.01302 (−0.03156, +0.00314) | 0.50156 |
| Strong specialization | 16 | 0.44347 | 0.44641 | +0.00293 (−0.00467, +0.01076) | 0.44324 |
| Standard-v5, 27 families | 54 | 0.26632 | 0.26179 | −0.00453 (−0.00951, +0.00049) | 0.25620 |
| Digits, 449 fusion rows | 4 | 0.03557 | 0.03562 | +0.00004, descriptive | 0.03608 |

On weak specialization, selecting a single tree cost **0.01661 Brier** relative to using the four-model average (0.00802 to 0.02562). The representative had higher risk on 14/16 banks. On the standard panel, the single-tree cost was **0.00559** (0.00397 to 0.00721). Compared with ordinary regularized OKAPI, nine standard-family means improved, seven tied and eleven worsened. The largest improvement was on compositional switch (−0.11685) and the largest regression was on rare shortcut noise (+0.02328). Descriptively, the other 26 families averaged −0.00021 after omitting compositional switch. The digits estimates come from four resplits of the same dataset. [Single-tree versus ensemble](results/round3-stability.csv) · [Family results](results/round3-families.csv)

## Fits and decisions

The full-data tree remained selected on 50/106 banks, including six of 16 weak-specialization banks. Seeds 1061, 1062 and 1063 produced the bootstrap row samples and native search seeds 1, 2 and 3 produced their candidate trees. The full-data control used native seed zero. All four used split seed zero. Sampling each class into its original row positions preserved the source label sequence. Across the bootstrap models, 16,089 of 32,184 internal validation positions were copies of original source rows also present in discovery. The reported comparisons use separate test observations or exact population risks. [Fit records](results/round3-fits.jsonl) · [Single-tree choices](results/round3-decisions.csv)

There were 424 native fits across 106 banks. Together they used 2,359,913 native objective calls, **4.02 times** the 586,778 calls in the 106 full-data fits alone. The complete one-thread run, including reference methods and scoring, took 207.10 seconds. The controlled intervals resample 16 training banks. The standard interval resamples the two banks within each of 27 fixed families. [Implementation](r3.py) · [All method levels](results/round3-levels.csv) · [Individual scores](results/round3-scores.csv) · [Run manifest](results/round3-run.json)
