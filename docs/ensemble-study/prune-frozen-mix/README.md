# Pruning removed more tree structure than base models

In the eight-bank CompositionalSwitch test, the previously selected balanced-start trees shrank from 18.56 to 10.94 nodes on average. Their full-class test Brier went from .26931 to .27027. The paired difference was +.00097 (95% interval −.00503 to +.00696), within the prespecified +.010 quality margin. Two of 16 trees dropped a provider. Mean distinct providers went from 4.3125 to 4.1250, saving .1875 (95% interval −.1235 to +.4985). The combined primary criterion required positive evidence of provider savings as well as retained quality. Only the quality condition passed. [Frozen protocol](protocol.md) · [All paired estimates](paired.csv)

| Fresh IID tests | CompositionalSwitch Brier | 19-family mean Brier | 19-family mean providers | 19-family mean nodes |
| --- | ---: | ---: | ---: | ---: |
| Balanced-start OKAPI, intact | .26931 | .25861 | 4.73 | 15.97 |
| Balanced-start OKAPI, pruned | .27027 | .26114 | 4.09 | 10.09 |
| Standard-start OKAPI, intact | .35250 | .27600 | 3.95 | 9.28 |
| Standard-start OKAPI, pruned | .35038 | .27741 | 3.56 | 6.68 |
| Balanced-start tree reduced to its root provider | .62476 | .47295 | 1.00 | 1.00 |
| All-provider mean | .40357 | .37749 | 11.84 | |
| Ranked-prefix Top-N, 896 labels | .35437 | .33204 | 4.61 | |
| Regularized logit stack, 896 labels | .32848 | .21749 | 11.84 | |

The new tests use fresh examples, so the intact losses can differ from the previous round. The table weights each family equally. Full results include chain-start winners, all nine specified shifts, and 640-label baselines. Among the 160 balanced-start trees, 109 dropped nodes and 62 dropped at least one provider. Against nine shifts their average Brier went from .31769 to .32002. On the focal interior-grid shift the difference was +.00045 (−.00160 to +.00249). On MulticlassCommittee the observed IID cost was +.00840 (+.00242 to +.01439), an exploratory four-bank interval. [Per-family means](means.csv) · [Per-run test scores](scores.csv.gz)

An update can use a provider already referenced elsewhere in a tree. Removing that update shortens the program while the provider's prediction is still needed. We count distinct referenced IDs under the assumption that each provider is run once and its outputs are reused. This experiment measured neither inference latency nor provider-specific cost. The path retains the root provider and cannot restore an earlier deletion; a different pruning policy could produce different trade-offs.

## Reproduction

The source was all 480 archived winners from FusionBench d063ec0, with native OKAPI 3eae0e5. Each winner's Brier was replayed on its original validation set (largest absolute difference 5.56e-17). At each step native single-update bypass candidates were compared on the original fit rows. A fresh 256-row pruning set selected the cheapest path member whose loss stayed within .005 of its own intact tree. All choices were frozen before new test generation. The campaign made 8,110 fit evaluations (including intact roots of paths) and scored 2,332 path entries on the pruning sets; it took 4.7 seconds on three local CPU workers. No new evolutionary trajectories were trained. [Ordered cuts and costs](paths.jsonl.gz) · [All 160 Top-N prefix rankings](topn-prefixes.jsonl.gz)

An independent NumPy implementation reproduced all 8,110 fit evaluations, all 2,332 path stages, each validation decision and all 2,160 held-out tree prediction fingerprints. It checked the 252 paired contrasts against the individual runs. The round took about three minutes from protocol publication through verification and reporting. [Independent verifier](verify.py)

From the FusionBench repository root, with the pinned OKAPI checkout at ../OKAPI, run the following commands.

```sh
export PYTHONPATH=.:../OKAPI OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python docs/ensemble-study/prune-frozen-mix/run.py
python docs/ensemble-study/prune-frozen-mix/verify.py
```

An open question is whether an explicit move that removes all occurrences of a given provider, allowing the root to change, can disconnect more models at the same quality margin.
