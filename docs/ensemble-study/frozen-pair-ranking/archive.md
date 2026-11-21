# Replaying the frozen-pair audit

The 192 candidate programs, one per bank and GP start, are in [candidates.csv.gz](candidates.csv.gz). [Source losses](source.csv.gz) include the three fitted baselines, their training and validation scores, the complete Top-N ranking and its prefix scores. [Target risks](risks.csv.gz) integrate the two contexts on the same 16,384 independent base examples per bank. [Decisions](selections.csv.gz) identify the two GP starts selected before acquiring target labels. The signed loss difference is always transported tree minus ordinary-validation tree. Negative favors transport.

## Numeric archive

Load [audit-draws.npz](audit-draws.npz) with NumPy. The first three indices in every array denote source bank, source-validation/target-monitor repeat, and target context frequency, in the order written in [config.json](config.json). For example, index `[0, 0, 1]` means bank 7500, repeat zero, target context frequency .50. The configuration names the remaining axes for every array.

- `f` and `g` hold the transported and ordinary trees' predictions on the **same separate pool of 512 audit rows**. `world_labels` holds coupled binary outcomes for its five λ worlds in increasing order. `audit_context` identifies changed rows. `monitor_bit` contains the disagreement feature on a different, unlabeled batch. Actual target contexts and outcomes never enter either tree-ranking criterion. [Input and label fingerprints](pools.csv.gz)
- `estimate` has inner axes proposal, λ, sampler repetition and draw budget. Proposals 0 and 1 denote centered uniform and gap-proportional draws. Each repetition has a single sequence of 128 draws. The first 8 and 32 positions are its earlier checkpoints. The two proposals transform a common set of uniform variates into row indices. The same indices apply to every λ world. `draw_indices` contains the actual input row numbers, with −1 for every slot in a zero-cost identical-prediction pair.
- `n_unique` counts each queried row once per schedule and checkpoint. An identical-prediction pair needs zero labels. When predictions differ somewhere, the implemented uniform schedule reads every distinct drawn index, including indices at which the two predictions happen to agree. An implementation can skip reading such an index and use its known zero residual. Reconstruct this alternative count by retaining only distinct drawn indices with `f[i] != g[i]`. Gap sampling never draws a zero-gap row. Both estimators count all nonzero-residual draws with their original multiplicities.
- `interval_lower` and `interval_upper` apply to gap sampling alone at the fixed budget. `det_lower` and `det_upper` index deterministic budgets 0, 8, 32 and 128. They bound the audit-pool loss difference for every completion of unqueried binary labels. [Per-pool audit performance and independent test regret](audit.csv.gz)

For a paired estimate at 32 draws in the last λ world, using the first bank and repeat at target frequency .50:

```python
import numpy as np

with np.load("audit-draws.npz") as data:
    estimate = data["estimate"][0, 0, 1]
    uniform = estimate[0, -1, :, 1]  # all 64 paired repetitions
    gap = estimate[1, -1, :, 1]
    choose_transport = gap < 0
    pool_difference = data["true_gap"][0, 0, 1, -1]
```

For inference, average the sampler repetitions within each validation/monitor repeat, then average the eight repeats within a source bank before forming the interval across 16 banks. Costs, run counts, split sizes, λ values, seeds and ordered budgets are recorded in [config.json](config.json).
