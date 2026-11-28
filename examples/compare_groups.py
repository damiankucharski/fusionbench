"""Score three reference methods on the 30 bundled standard-v6 banks."""

from fusionbench import evaluate_dataset, load_reference_dataset
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR
from fusionbench.taxonomy import APPLICATION_PROFILES, scenario_weights, weighted_score


def compare():
    standard = scenario_weights()
    results = {}
    for scenario in standard:
        methods = [SimpleAverage(), TopN(metric="brier"), StackingLR(c_value=1.0)]
        rows = evaluate_dataset(load_reference_dataset(scenario), methods)
        rows = rows[rows.method.isin([method.name for method in methods])]
        failed = rows[rows.error.ne("")]
        if not failed.empty:
            raise RuntimeError(
                f"{scenario}\n{failed[['method', 'error']].to_string(index=False)}"
            )
        for row in rows.itertuples():
            results.setdefault(row.method, {})[scenario] = row.test_brier

    profiles = {
        "standard-v6": standard,
        "context routing": scenario_weights(groups="context_routing"),
        "operational stress": scenario_weights(purposes="operational_stress"),
        "multi-site": scenario_weights(
            group_weights=APPLICATION_PROFILES["multi_site"]
        ),
    }
    print(f"{'Test Brier':<20}" + "".join(f"{method:>16}" for method in results))
    for label, weights in profiles.items():
        scores = [
            weighted_score(by_scenario, weights) for by_scenario in results.values()
        ]
        print(f"{label:<20}" + "".join(f"{score:>16.5f}" for score in scores))


if __name__ == "__main__":
    compare()
