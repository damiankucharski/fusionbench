"""Paired controls on the number of independent, redundant or irrelevant providers."""

from copy import deepcopy
from dataclasses import replace
from numbers import Integral

from fusionbench.scenarios import get_scenario


def _count(value, minimum, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def duplicate_provider(dataset, provider=0, copies=1):
    """Append exact copies on both splits. The observable posterior is preserved."""
    provider = _count(provider, 0, "provider")
    copies = _count(copies, 1, "copies")
    if provider >= dataset.n_models:
        raise ValueError("provider index outside the dataset")
    indices = [*range(dataset.n_models), *([provider] * copies)]
    ids = list(dataset.model_ids)
    base_id = ids[provider]
    suffix = 1
    while len(ids) < len(indices):
        candidate = f"{base_id}_copy_{suffix}"
        if candidate not in ids:
            ids.append(candidate)
        suffix += 1
    meta = deepcopy(dataset.meta)
    meta.setdefault("interventions", []).append(
        {"operation": "duplicate_provider", "provider": provider, "copies": copies}
    )
    return replace(
        dataset,
        name=f"{dataset.name}_clone{provider}x{copies}",
        fit_preds=dataset.fit_preds[:, indices],
        test_preds=dataset.test_preds[:, indices],
        model_ids=ids,
        meta=meta,
    )


def paired_pool_sizes(scenario_id, sizes, *, n_per_split=1024, seed=0, **params):
    """Generate nested pools from one maximal bank. Size counts bad inputs in sparse pools."""
    axis = {
        "committee": "k",
        "multiclass_committee": "k",
        "deadweight": "n_bad",
        "needle": "n_bad",
    }.get(scenario_id)
    if axis is None or axis in params:
        raise ValueError("Choose a supported pool and specify its size through sizes")
    sizes = [_count(size, int(axis == "k"), "size") for size in sizes]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be nonempty and distinct")
    largest = max(sizes)
    source = get_scenario(scenario_id, **params, **{axis: largest})
    bank = source.build(n_per_split=n_per_split, seed=seed)
    paired = {}
    for size in sizes:
        sc = get_scenario(scenario_id, **params, **{axis: size})
        ids = sc.model_ids()
        fit, test = bank.fit_preds[:, : len(ids)], bank.test_preds[:, : len(ids)]
        meta = deepcopy(bank.meta)
        meta["interventions"] = [
            {
                "operation": "prefix",
                "size_parameter": axis,
                "size": size,
                "source_size": largest,
            }
        ]
        paired[size] = replace(
            bank,
            name=sc.name,
            model_ids=ids,
            meta=meta,
            fit_preds=fit,
            test_preds=test,
            fit_oracle=sc.bayes_oracle(fit, {}),
            test_oracle=sc.bayes_oracle(test, {}),
        )
    return paired
