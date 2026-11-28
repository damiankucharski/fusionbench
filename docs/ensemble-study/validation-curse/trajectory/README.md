# What survives six generations of OKAPI?

With 48 support labels and 32 candidate providers, the selection of typical individuals and the preservation of the best available predictor can move in opposite directions. In the null arm, the calibrated anchor appeared in **all eight** active populations after the first generation. Ordinary Brier fitness retained an audit-prediction-equivalent anchor in **three of eight** final populations. The population's median fresh-test Brier nevertheless improved in seven of eight runs. In the matched positive arm, the informative provider survived in seven of eight final populations, while the support-selected champion grew worse in five. These are two different failure modes of native genetic programming.

## Independent evaluation

Each bank used 48 support rows for evolution, 1,024 independent source rows for retrospective selection, and 2,048 fresh test rows for reporting. No held-out predictions or labels entered the search. The historical search used one stock OKAPI restart and six complete generations. The study concerns individual native GP trees. The higher-level OKAPI method that also chooses flat or augmented heads is evaluated in the [standard-suite study](../panel/). The [frozen protocol](protocol.md) specifies data seeds and budgets.

An **active champion** maximizes rounded native support fitness in the survivor pool, breaking ties by tree size and original order. An **audit pick** minimizes empirical source-audit Brier within that pool and is then scored on fresh test rows. The **test minimum** uses test labels retrospectively and measures available headroom. The **archive** retains the first occurrence of every distinct structure generated so far, including those later discarded. Generation zero precedes native population truncation.

Median fresh-test Brier across eight banks for ordinary search is shown below. Lower values are better. Entries within cells follow generations 0 → 3 → 6. The median of paired bank differences can differ from subtraction of these marginal medians.

| Condition | Active median | Active champion | Active audit pick | Active test minimum | Distinct active behaviors |
|---|---|---|---|---|---|
| Redundant null | .527 → .449 → .448 | .505 → .445 → .450 | .398 → .421 → .427 | .398 → .412 → .425 | 60 → 38 → 29.5 |
| Single signal | .526 → .324 → .313 | .278 → .315 → .307 | .274 → .273 → .273 | .274 → .273 → .273 | 60 → 34.5 → 28.5 |
| Committee | .340 → .263 → .267 | .304 → .266 → .269 | .255 → .252 → .256 | .254 → .250 → .252 | 39 → 36.5 → 35 |
| Native switch | .592 → .363 → .347 | .427 → .354 → .356 | .401 → .338 → .342 | .401 → .338 → .342 | 33 → 20.5 → 15 |

In the null arm, the final-minus-initial paired median was −.0806 for a typical individual and +.0234 for the fresh-test risk of the audit pick. Selecting retrospectively from the **final survivors** cost +.0346 relative to selecting from the **cumulative archive**. On the positive arm the final active audit pick and archive audit pick had the same median paired test risk, while the final champion was .0343 worse than the active audit pick. On native switch the ordinary search improved the active audit pick by .0402 from initialization. These measurements separate survivor loss, champion selection and productive search.

## Penalizing uncertain support gains

The paired objective adds one standard error of the tree's rowwise loss differences from arithmetic averaging. It uses the same support rows, operators, initialization, seed and native rounding as ordinary search. The objectives generate different descendants after selection diverges. Values below are median **penalized minus ordinary** fresh-test Brier on matched banks. Parentheses give banks where the penalized champion improved, tied or worsened, in that order. At 240 score calls the comparison is between champions of equally sized scored prefixes, counting every reevaluation, regardless of generation. At generation six the comparison uses actual survivors after six completed selection steps.

| Condition | Final active median | Final active audit pick | Final active champion | Archive champion at 240 calls |
|---|---:|---:|---:|---:|
| Redundant null | −.0046 | −.0177 | −.0058 (5/0/3) | −.0033 (5/0/3) |
| Single signal | +.0205 | .0000 | +.0182 (1/0/7) | +.0064 (2/2/4) |
| Committee | −.0047 | −.0011 | −.0001 (4/1/3) | −.0011 (5/2/1) |
| Native switch | +.0034 | −.0001 | +.0006 (4/0/4) | +.0053 (3/1/4) |

The penalty retained the null anchor's audit prediction in **six of eight** final populations, compared with three under ordinary fitness. In the single-signal arm both rules retained the true posterior in seven of eight. The penalized champion's median gap to the final audit pick increased to .0549 from .0343. The penalty helped preserve a simple available rule in the null and made the search more reluctant to choose real additional signal in the matched alternative. This panel supplies no basis for enabling it by default.

## Controls

Fresh-test Brier, median over the same eight banks, using all 48 support rows for fitting. The posterior uses known generator parameters. In the two anonymous-pool controls it coincides with one provider. On committee the posterior combines calibrated experts in log-odds space. A probability-feature logistic stacker fit to 48 rows substantially beat the native tree family studied here.

| Condition | Arithmetic mean | Ranked Top-N | Probability LR | Generator posterior |
|---|---:|---:|---:|---:|
| Redundant null | .4018 | .4214 | .6272 | .3978 |
| Single signal | .3942 | .3006 | .4437 | .2702 |
| Committee | .2376 | .2530 | .0470 | .0270 |
| Native switch | .4509 | .3500 | .3787 | .2645 |

## Reproduce and inspect

Use FusionBench with its research dependencies and OKAPI commit `ce6ae177a4058612b7ffa33194a13e3fda86f5cb` on the Python path. Run from the FusionBench repository root.

```sh
out=docs/ensemble-study/validation-curse/trajectory
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m fusionbench.eval.population_trajectory --output "$out" --workers 2 --verify-observer
PYTHONPATH=. python "$out/summarize.py"
```

The 64 runs contained 26,257 distinct structures counted separately per run, 44,688 native score visits, 53,382 offered offspring and 18,112 active population slots across seven snapshots. Each generational pool offered to deduplication is the previous active list followed by that generation's offspring list. All fitness calls and survivor identities are preserved. Replaying bank zero in all four conditions under both objectives with ordinary uninstrumented OKAPI gave identical ordered survivor genotypes at every generation. Repeating all 64 searches with one worker reproduced the compressed traces, stage and control tables byte for byte.

- [Compressed complete traces](traces.jsonl.gz) contain the ordered offered and retained program indices for every generation and every native score visit, exact serialized trees, support/audit/test risks, fitness and a hash of each program's predictions on 1,024 audit inputs after rounding to 10 decimals.
- [Per-bank pool scores](stages.csv), [baseline scores](controls.csv), [median checkpoints](medians.csv), [population medians on all three splits](split-medians.csv) and [paired comparisons](paired.csv) include both objectives and the 240-call prefixes. Sign counts use an absolute tolerance of 1e−12.
- [Posterior matches](posterior-retention.csv) test exact equality of these rounded audit vectors. Equality on these inputs does not prove functional identity on every possible input. Eight banks, binary stationary laws, a 48-label regime and six generations bound the scope of this study. Test minima benefit from selecting over test labels. The independent audit-picked test column quantifies what transfers from a separate source sample.
