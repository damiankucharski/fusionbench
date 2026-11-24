# Reproducing the multiclass head comparisons

There are 792 held-out scores: eleven methods on 48 archived real prediction banks and 24 matched synthetic banks. Each row records the seed, an input/label fingerprint, full-class Brier, accuracy and fit time. For the 24 synthetic banks each method also has conditional Brier and the underlying sum of losses and number of observations for true classes 3/8 and the remaining classes. All methods within a bank use identical provider probabilities. The four synthetic conditions share labels and example-level randomness at a given seed. Class permutations leave the true-class mass and the entire multiset of each provider's probabilities intact. The untouched eight classes retain their original input tensors exactly. [Protocol](../protocol.md) · [Per-bank scores](scores.csv) · [Method means and conditional totals](means.csv) · [Paired contrasts](contrasts.csv)

The runner imports 336 previously measured real scores with their original fit times. It verifies both prediction archives, source score files and fit manifests against the published checksums. The other 456 combinations are fresh fits, including all 264 synthetic scores. `origin` distinguishes the two groups. The real cohorts reuse four datasets. Several digits cohorts reuse images, and the original heterogeneous digits and logistic-bag cohorts share the same original partitions. The eight real partitions per cohort can overlap. Six synthetic replicates use independent seeds, with the four interventions paired within each replicate. [Inputs and configuration](manifest.json)

Both standalone native heads have a ceiling of 4,096 scalar objective evaluations. Ten-class fits used 568 calls for diagonal heads and 4,096 for full heads. Three-class fits used 176 and 428 respectively. Per-fit JSON records the raw fit score, variance-weighted penalty, penalized score and actual call count. Timing excludes test prediction and scoring, includes lazy imports on first use, and can depend on concurrent workers. Historical native GP search timings come from the respective source archives.

A separate verification checked all 792 bank-method keys, reproduced every one of the 336 imported scores and fit times exactly, and checked the 264 synthetic conditional decompositions and all 144 standalone native objective budgets and penalty identities. It then independently refit all eleven methods on the first replicate from each cohort (110 fits) and reproduced Brier to a maximum discrepancy of 2.1 × 10⁻¹⁷, with identical accuracy throughout.

Run from the FusionBench checkout with its matching OKAPI checkout and the CPU NumPy backend. Three workers are allowed, each with a single numerical thread. The runner writes each completed bank atomically to one shared scores table. Repeating the command resumes missing bank-method pairs and regenerates the summaries. Use a single invocation at a time per output directory.

```sh
export BACKEND=numpy OKAPI_LOG_LEVEL=ERROR LOGURU_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
python docs/ensemble-study/class-capacity/run.py --out results/class-capacity --workers 3
```

For a separate fit on one bank, import `load_bank(cohort, replicate)` from [the study driver](../run.py). It returns a FusionBench Dataset and the frozen search seed. A negative paired contrast means the method in `left` has lower Brier than the method in `right`; ties for split counts use an absolute tolerance of 10⁻¹². Summary conditional Briers average split means equally. Divide the total sum of losses by the displayed number of examples for a pooled conditional Brier.
