# Series 1 · Do eight different MIN proposals transfer better?

Freeze this protocol before scoring the new banks. Use 32 independent data
banks with seeds 241100–241131. Six anonymous provider probabilities are
independent uniform draws from the 63-point grid 1/64 through 63/64. Generate
192 fit observations with `draw_native_switch(seed,1,192)` and 4,096 audit
observations with `draw_native_switch(seed,2,4096)`. The hidden ordered roles
A,B come from the bank's separate `[seed,15]` role stream. For the primary
law set q=min(A,B). For a control with a correct singleton among the founders,
set q=A on exactly the same scores. For both laws generate each label as the
indicator that the same draw's uniform is below q. No learner receives roles,
uniforms, posterior, audit data or proposal correctness.

For each law, run the pooled native path in `fold-promotion/search.py` with
its established six singleton founders plus arithmetic mean, 16 generations,
six native operator families, and seed data seed + 1,000,000. Take
`pooled_final`, the single final survivor with the smallest unrounded fit
Brier under the published tie breaks. Freeze this model before generating or
examining proposals. The three deployment arms are the frozen finalist alone
and two extensions of that same candidate menu.

Use native `okapi.pair_coverage` to build exactly eight one-update MIN
programs per extension. Allowed slots are all 15 unordered pairs i<j,
sorted lexicographically. Use seed data seed + 4,000,000 for the two
schedulers, independently of the hidden roles and training draws. Reuse
the schedules across the two laws. IID draws eight uniform slots with
replacement. Unique takes the first eight positions of a uniform permutation.
Both pay for eight native predictions on the same fit observations, including
repeat slots. Pick the smallest unrounded full-class fit Brier; ties keep the
frozen finalist and then the earliest proposal. The deployment is one program.
This intervention tests coverage at the final decision. There is no breeding
from probes. It prespecifies MIN and can exploit a correct operator family.

The primary is unique minus IID mean conditional excess full-class Brier
on fresh MIN audit rows, $2\,\mathrm{mean}((p-q)^2)$. Calculate within-bank
differences and a two-sided percentile interval from 10,000 bank resamples
with NumPy default_rng(97901). An upper endpoint below zero supports an
improvement on this task. Report directions on individual banks and all
32 observations. Do not count the two laws as separate replicates.

Record coverage, known-pair offers, programs matching q before and after
the intervention, fit and sampled audit Brier, distinct proposals, and
the proportion of offered witnesses deployed. Check the frozen GP archive
and survivors separately. Exact agreement of GP functions on observed
fit and audit vectors uses maximum absolute probability error at most
$10^{-12}$. A probe using A and B computes the true minimum by construction
for every input. For a uniformly located pair its inclusion probability is
$1-(14/15)^8$ with IID draws and $8/15$ with unique draws; fit selection
can still reject it. Under the singleton law the correct program uses only
A, so covering the hidden pair does not offer that correct singleton.

Score simple averaging, Brier-ranked best-prefix Top-N, standardized
probability logistic stacking and regularized native-pattern OKAPI on the
same fit/audit banks. Record GP score counts separately from the extra eight
calls, archive ordered proposals and eligibility/choices, and report elapsed
time with a single worker and one numerical thread. No source files or global
settings from earlier studies change.
