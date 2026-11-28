# Training-bank and optimizer variation under weak specialization

Both the training data and the optimizer seed changed regularized native-pattern OKAPI's predictions. Across 16 fresh training banks and six fixed optimizer seeds, **58.6% of its prediction variance lay between banks** after averaging each bank's six forecasts, and **41.4% lay within banks** as the optimizer seed changed. The discovery/validation split seed stayed at zero for every fit. The source and target distributions were identical. [Frozen secondary protocol](seed-protocol.md) · [Variance components](results/seed-components.csv)

At 128 observations and specialization 0.25, regularized OKAPI's population Brier exceeded arithmetic averaging by **0.03566** (paired 95% interval **0.02725 to 0.04475**). For each bank, this comparison averages the risks of the six individually deployed native models. Arithmetic averaging had lower risk in all 16 bank comparisons. The true event probability weights one provider more heavily, so the arithmetic mean itself had excess Brier of 0.00793. [Bank-level risks](results/seed-bank-means.csv) · [Paired comparisons](results/seed-contrasts.csv)

| Method | Fit Brier | Population Brier | Excess over Bayes | Squared bias | Between banks | Within bank across optimizer seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regularized native-pattern OKAPI | 0.49177 | 0.51754 | 0.04359 | 0.00060 | 0.02518 | 0.01781 |
| Arithmetic mean | 0.48842 | 0.48188 | 0.00793 | 0.00793 | 0 | 0 |
| Brier-ranked Top-N | 0.46803 | 0.49879 | 0.02485 | 0.00053 | 0.02431 | 0 |
| Probability logistic stacking | 0.44970 | 0.50371 | 0.02976 | 0.00102 | 0.02874 | 0 |

The Bayes Brier was **0.47395**. Relative to the arithmetic mean, regularized OKAPI reduced squared bias by 0.00733 while adding 0.04299 of prediction variance. Its population-minus-fit gap was +0.02577. The three reference methods were fitted once per bank and had no optimizer seed variation in this comparison. Regularized OKAPI chose its GP candidate in 95 of 96 fits and its augmented candidate once. [Fit scores and selected families](results/seed-scores.csv)

At each of the 512 support states, we computed the squared difference between the grand mean forecast and the true probability, the population variance of the 16 bank-mean forecasts, and the average of the 16 within-bank forecast variances. We multiplied each by two and integrated over the exact state probabilities to obtain the last three columns. Their sum equals mean full-class Brier excess. Within-bank variation includes any interaction between bank and optimizer seed. These percentages describe the specific 16-by-6 grid and the fixed split rule. [Runner](seeds.py) · [Forecasts and exact population](results/seed-predictions.npz)

The run used one numerical thread for 96 native fits and 48 reference fits, taking 34.1 seconds. Resampling the 16 paired banks 10,000 times kept all six optimizer seeds together. An independent calculation from the archived forecasts reproduced all four methods' components and the paired interval to within 10⁻¹⁴. [Run manifest](results/seed-manifest.json)

## Post-hoc family diagnostic design

The 95 GP selections above motivated a separate diagnostic on 16 previously unused banks, seeds 258100–258115. Fix specialization at 0.25, 128 observations, optimizer seed zero and split seed zero. Construct training rows by the same stratified procedure. Fit the unmodified regularized method once per bank and the same arithmetic mean, Brier-ranked Top-N and probability logistic stacking controls. Freeze the original family choice before evaluating the population.

Replay each of the three stored candidates at discovery and refit. Record its full 128-row empirical Brier, Brier on the 39 original validation rows, and exact population Brier. Verify the discovery validation predictions against the adapter's reported scores. The GP candidate is the same tree at both stages. Compare the ordinary selected family's population risk with the raw-provider flat head, arithmetic mean, and the lowest population risk among the three families in hindsight. For the paired selected-minus-flat and flat-minus-mean differences use 10,000 percentile resamples of the 16 data banks with seed 97947. Count banks in which the flat head beats the mean. This is a retrospective look at existing components and keeps the search procedure fixed.

### What the fresh banks showed

The ordinary fit-only selector chose GP on **16 of 16** banks. The raw-provider flat head improved when refitted on all 128 rows and had lower population Brier than the chosen GP tree on **11 of 16** banks. Always deploying the refit flat head would have reduced average population Brier by **0.02437** (95% paired interval **0.00890 to 0.04231**). Arithmetic averaging still beat the refit flat head on **all 16** banks. Flat minus mean was **+0.02088** (0.01392 to 0.02926). [Bank-level scores](results/family-check.csv) · [Paired comparisons](results/family-check.json)

| Existing family | Discovery Brier on validation rows | Discovery population Brier | Population Brier after refit |
| --- | ---: | ---: | ---: |
| Flat logit head, raw providers | 0.52239 | 0.51885 | 0.50276 |
| Augmented logit head, raw providers and GP features | 0.55906 | 0.55069 | 0.52596 |
| GP validation champion | 0.42380 | 0.52713 | 0.52713 |

Even choosing the best of the three refit candidates separately on each bank **in hindsight** gave mean population Brier 0.50121. The ordinary choice scored 0.52713, arithmetic mean 0.48188, Brier-ranked Top-N 0.49192, and probability logistic stacking 0.50455. Switching families alone would have left a gap to averaging in this weak-specialization setting.

The GP search validated 321–383 distinct programs per bank (mean 350) and carried its validation champion forward. The family decision then compared that champion with the two discovery heads on the same 39 rows. On fresh population states the apparent validation advantage disappeared. The heads also improved with full-data refitting while the selected GP tree stayed fixed. Estimating the causal contribution of validation reuse would require a comparison that holds the candidate set and refitting procedure fixed while varying the selection data. [Replay script](family-check.py)
