# Validation and search overfitting on standard-v3

Round one measures how much the internal validation gain of the historical OKAPI procedure survives on an independent sample from the same source. Round two tests a training intervention at the same label budget and a separate deployment-only guard on 144 fresh banks. Average risk rose with both interventions on the independent source audit. [Round one](round1/README.md) · [Round two](round2/README.md)

The opt-in FusionBench method `okapi_search_guard` subtracts one standard error of paired improvement over the arithmetic mean during GP discovery, native head fitting and feature screening. It retains raw Brier selection on the internal validation rows. `okapi_terminal_guard` fits the incumbent on 70% of the development rows and uses the remaining 30% to choose from its frozen heads, GP champion and a native mean tree. The latter method's fresh selection labels were never exposed to the GP. A fitted instance exposes `raw_choice_`, `chosen_` and `guard_decision_` for inspecting that comparison. Both return one native tree and consume provider predictions at deployment. The methods are optional research comparators.

From the FusionBench research checkout with the OKAPI extra installed, use one numerical thread per worker. Each round has six independent banks on each of the 24 standard scenarios. The public relative-optimism endpoint measures apparent gain over the mean on selection rows minus gain on independent source rows. The actual deployment tests can have a different law; their raw score changes measure transfer.

```sh
export BACKEND=numpy LOGURU_LEVEL=ERROR OKAPI_LOG_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
study=docs/ensemble-study/validation-curse/panel
python "$study/run.py" --round 1 --workers 3 --out results/curse/round1
python "$study/analyze.py" results/curse/round1
python "$study/run.py" --round 2 --workers 3 --out results/curse/round2
python "$study/analyze.py" results/curse/round2
```

An independent check regenerated 1,728 feature and label digests and all 32 reported contrasts and intervals across the two rounds. End-to-end training replays on committee, similar_classes_38 and environment_shortcut matched all 189 recorded score pairs and the archived search path hashes.
