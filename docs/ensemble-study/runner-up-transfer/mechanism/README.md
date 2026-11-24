# Does the true class predict which wrong class a provider favours?

On historical handwritten digits, knowing the true digit improved cross-half prediction of the strongest wrong class by **0.254 nats per provider event** over uniform guessing. The gain was **0.183** after accounting for which destinations each provider favoured overall. Synthetic panel B showed neither gain. Adding a directed confusion on hard examples in panel E produced a detectable change against shuffled destinations, concentrated among actual top-1 errors. [Frozen protocol](../mechanism-protocol.md) · [Scores and shuffle ranges](summary.csv)

| Historical fit bank | All rows, gain over uniform | All rows, gain over popularity | 95% shuffle range for gain over popularity | Top-1 errors, gain over uniform | Errors counted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Digits, heterogeneous providers | +0.254 | +0.183 | −0.141 to −0.108 | +0.231 | 339 |
| Panel B, finite votes | −0.120 | −0.106 | −0.122 to −0.098 | −0.065 | 302 |
| Panel E, finite votes and directed confusion | −0.066 | −0.054 | −0.126 to −0.103 | +1.042 | 959 |

Each source contributed eight fusion-fit banks, with 449 rows and three providers each (10,776 row/provider appearances). A provider can favour a particular wrong class even when its winning prediction is correct. Tied wrong classes shared one event equally. For the digits logistic regression and forest, knowing the true class gained 0.316 and 0.232 nats over destination popularity. The corresponding gain for the five-neighbour classifier rounded to 0.000. It tied all nine wrong classes on 3,230 of 3,592 rows. [Provider results](banks.csv)

On digits errors alone the gain over uniform was above its shuffled range (−0.256 to −0.108). The gain over estimated destination popularity (+0.299) fell inside its shuffled range (+0.121 to +0.370). The 339 error appearances give a weaker basis for separating class-specific transitions from overall popularity. For panel E the corresponding gain on errors was +1.368 against a shuffled range of −0.056 to +0.115. On *all* panel E rows both simple baselines beat the fitted conditional frequency table, despite its gain over shuffling. [Permutation results](permutations.csv)

The observed digits pattern establishes that a competitor's identity can be predictable from the true class on held-out rows of the same fit bank. The data leave open which image features cause these preferences and how much they contribute to a fusion method's test Brier. Panel E imposes a particular provider-specific wrong destination only on hard rows. The digits partitions reuse images, and the permutation ranges describe randomized destination labels within these banks. [Synthetic mechanism](../../../../fusionbench/scenarios/multiclass_provider_panel.py) · [Reproduction script](../mechanism.py)
