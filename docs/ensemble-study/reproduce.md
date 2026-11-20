# Reproduce the research runs

Use the FusionBench research branch and its pinned OKAPI dependency. The measurements used Python 3.12 on CPU. From the FusionBench checkout run:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[okapi,dev]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q
```

## Validation ranking and tree search

```sh
python -m fusionbench.eval.seed_selection --out results/seed_selection/development --data-seeds 100 101 102 103 104 105 106 107
python -m fusionbench.eval.seed_selection --out results/seed_selection/confirmation --data-seeds 300 301 302 303 304 305 306 307 --gp-seeds 40 41 42 43 44 45 46 47 48 49 50 51
python -m fusionbench.eval.observer --bank results/seed_selection/confirmation --out results/observer
fusionbench stability --out results/stability
python -c "from fusionbench.eval.shrinkage import run_study; run_study('results/shrinkage', workers=2)"
```

[Additional archive, restart and label-budget controls](search-stability.txt)

## Single models

```sh
python -m fusionbench.eval.blind_circuit --out results/blind
python -m fusionbench.eval.blind_selection --out results/blind_confirmation
fusionbench transfer --out results/transfer
fusionbench transfer --out results/diagonal --seeds 100 101 102 103 104 105 106 107 --followup
python -c "from fusionbench.eval.view_alignment import run_study, run_linear_control; run_study('results/rotations'); run_linear_control('results/rotations')"
python -c "from fusionbench.eval.view_alignment import run_study; run_study('results/compact', rescue_rates=(0.,), student_width=4)"
python -c "from fusionbench.eval.bypass_sampling import run_study; run_study('results/sampling',range(100,120))"
python -m fusionbench.eval.reconstruction --out results/reconstruction
python -m fusionbench.eval.reconstruction --out results/reconstruction/confirmation --frontier-only --seeds $(seq 100 119) --n-test 32768 --n-calibration 65536
```

The two published repositories supplied the code base for the historical reconstruction. The earlier share view exposed summaries and file names. Its full code diff, seeds and original run configuration were unavailable. Current measurements and the matched-design reconstruction are in the results archive.

## External predictions

```sh
python examples/digits_predictions.py --out results/data
fusionbench evaluate --dataset results/data/digits --methods best_single simple_average stacking_lr okapi --out results/digits
python examples/digits_stability.py --out results/digits
```

Rebuild the headline figure from the bundled measurements by running `python docs/ensemble-study/plot_results.py`. The compressed archive contains machine-readable results for each study, including the underlying per-seed numerators when recorded. The comparisons use independent data draws as units of replication. Per-example counts and multiple selectors on the same draw are paired measurements.

## Fit the four-feature circuit on your own arrays

```python
from fusionbench.eval.blind_circuit import RawCircuit

model = RawCircuit.fit(unlabelled_inputs, labelled_inputs, binary_labels, seed=0)
decisions = model.score(new_inputs) > 0
specification = model.to_dict()
```

Each input matrix has four columns. The fit routine discovers directions from the unlabeled matrix and chooses the logical rule using binary task labels. The score is a signed decision margin. Use the independent-half and out-of-library controls in the study before interpreting the recovered directions.
