## How to interpret the three rounds

Each bank contains predictions and labels for fitting a fusion method and a separate draw for scoring it. The same bank is used for every method in a comparison. Each of the 30 standard scenarios gets equal weight. The 24 development scenarios supplied the data for fitting the fingerprint rule and choosing the fixed configuration in round 1, then for choosing between the fingerprint and cross-validation rules in round 2. All seeds change between rounds. The other six scenarios enter the round 3 transfer check. These six comprise grouped rare events, shared-error blocs, rare-scene specialists, cloned shortcuts, parity and similar classes 3/8.

### Reading the scores

The main outcome is observed multiclass Brier loss, the sum of squared errors across classes averaged over observations. Smaller values are better. Expected Brier is available where the simulation provides the true conditional class probabilities. Macro-F1 and, for binary banks, average precision give a separate view of decisions and ranking. Each round records the number of deployed providers and tree nodes. The rules can have different training costs even when they deploy the same tree.

In each `r*-contrasts.csv`, a difference is the first method's score minus the reference's score. Percentile intervals use 10,000 paired bootstrap draws, resampling data seeds within fixed scenarios. They quantify variation over new seeds for this benchmark. In the eight factorial variants, each seed creates eight matched worlds, so we average all eight within a seed before resampling. For a single factorial variant or scarcity control, a bootstrap draw samples independent data seeds from that variant or control. The per-family table gives the raw mean for each of the 30 standard scenarios.

### Is the rule actually matching methods to banks?

A routed rule may improve by choosing a better configuration more often, by choosing it on the right banks, or both. The action-mix table measures the return to matching. Keep all its selected actions and reassign them uniformly at random across banks. We calculate the expected loss after shuffling exactly. For each action, multiply its observed selection frequency by its mean test loss on all banks, then sum the products. Subtract that expected loss from actual routed loss. Negative values indicate a benefit from matching. The interval resamples action frequencies and candidate losses together. A second column confines reassignment to each named scenario, so it measures whether the rule can distinguish draws within the same scenario, beyond its ability to distinguish different scenarios.

The portfolio oracle chooses the lowest observed test Brier separately for each bank. It illustrates the headroom among the five actions and includes hindsight from finite test samples. The reported fraction of the gap closed is the gain relative to the fixed round-1 action divided by the gap from that action to this oracle. A negative fraction means the rule lost to the fixed action.

### Confirmation and files

The primary round-3 endpoint compares the selected rule with the fixed round-1 action across all 30 scenarios. A negative upper 95% endpoint meets the primary criterion. The separate transfer claim additionally requires a negative upper endpoint on the six excluded scenarios and an upper endpoint of at most +0.005 on the 16-scene scarcity control. These checks were written before candidate selection. Comparisons with averaging, Top-N and logistic stacking are reported alongside these endpoints.

- `results/rN-levels.csv`, `rN-contrasts.csv` and `rN-families.csv` provide absolute scores, paired contrasts and the standard-scenario breakdown.
- `results/rN-configuration-costs.csv` gives providers, native program size on banks returning a tree, mean average precision on the binary subset, and the number and duration of cross-validation fits. Lookup time excludes fingerprint computation, and CV time excludes the final refit.
- `results/rN-actions.csv` and `rN-action-mix.csv` describe selected actions and the exchange control in rounds 2 and 3.
- `selection.json` records the round-2 decision, the two 24-family scores and the hashes of the source evidence. `results/confirmation.json` gives the final intervals, criteria and oracle gaps.
- `results/autoconfig-three.png` and `autoconfig-three.pdf` show the changes relative to the fixed action and how configuration choices vary across subsets.
