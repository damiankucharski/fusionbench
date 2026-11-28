"""Discover, generate and benchmark controlled prediction-fusion scenarios."""

from __future__ import annotations

import argparse
import json

from fusionbench.methods.registry import (
    DEFAULT_METHODS,
    FACTORIES,
    comparison_names,
    default_methods,
)
from fusionbench.scenarios.registry import SCENARIOS, STANDARD_SCENARIOS
from fusionbench.taxonomy import (
    APPLICATION_PROFILES,
    MECHANISM_GROUPS,
    PURPOSE_TAGS,
    TAXONOMY_VERSION,
    scenario_weights,
)


def _parameter(text):
    name, separator, value = text.partition("=")
    if not separator or not name.isidentifier():
        raise argparse.ArgumentTypeError(
            "Use --param name=JSON, for example --param accuracy=0.8"
        )
    try:
        return name, json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid JSON value for parameter {name!r}"
        ) from exc


def _scenarios(names, parameters=()):
    from fusionbench import get_scenario

    params = dict(parameters)
    if len(params) != len(parameters):
        raise SystemExit("Specify each scenario parameter once")
    try:
        return [get_scenario(name, **params) for name in (names or STANDARD_SCENARIOS)]
    except (ValueError, TypeError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


def _filters(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--suite",
        choices=("standard-v6", "smoke"),
        help="select a complete versioned panel",
    )
    source.add_argument("--scenarios", nargs="*", help="registered scenario IDs")
    parser.add_argument(
        "--group",
        nargs="+",
        choices=MECHANISM_GROUPS,
        help="include scenarios from these mechanism groups",
    )
    parser.add_argument(
        "--purpose",
        nargs="+",
        choices=PURPOSE_TAGS,
        help="include scenarios with any of these purpose tags",
    )
    parser.add_argument(
        "--application-profile",
        choices=APPLICATION_PROFILES,
        help="select and weight mechanisms for a research question",
    )


def _weights(args, *, discovery=False):
    names = args.scenarios or None
    if args.suite == "smoke":
        from fusionbench.transfer import load_preset

        names = list(load_preset("smoke")["scenarios"])
    elif discovery and args.suite is None and names is None:
        names = list(SCENARIOS)
    try:
        return scenario_weights(
            scenarios=names,
            groups=args.group,
            purposes=args.purpose,
            group_weights=APPLICATION_PROFILES.get(args.application_profile),
        )
    except (ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc


def _methods(names):
    requested = comparison_names(names)
    methods = default_methods(requested or None)
    if requested and (missing := set(requested) - {m.name for m in methods}):
        raise SystemExit(
            f"Install optional dependencies for: {', '.join(sorted(missing))}"
        )
    return methods


def cmd_list(args):
    print("Scenarios:")
    selected = SCENARIOS if args is None else _weights(args, discovery=True)
    for name in selected:
        cls = SCENARIOS[name]
        task = cls.task if isinstance(cls.task, str) else "configurable"
        groups = [g for g, ids in MECHANISM_GROUPS.items() if name in ids]
        print(f"  {name:22s} task={task:10s} group={', '.join(groups) or 'unassigned'}")
    print("\nMethods:")
    for name, factory in FACTORIES.items():
        avail = type(factory()).available()
        print(f"  {name:16s} available={avail}")


def cmd_generate(args):
    from fusionbench.store import save_dataset

    for sc in _scenarios(_weights(args), args.param):
        ds = sc.build(n_per_split=args.n, seed=args.seed)
        path = save_dataset(ds, args.out)
        print(f"wrote {path}")


def cmd_run(args):
    from fusionbench import __version__, summarize_suite
    from fusionbench.eval.reliability import reliability_summary
    from fusionbench.eval.report import save_report
    from fusionbench.eval.runner import run_suite

    weights = _weights(args)
    methods = _methods(args.methods)
    print(f"methods: {[m.name for m in methods]}")
    df = run_suite(
        _scenarios(weights, args.param),
        methods=methods,
        n_per_split=args.n,
        seed=args.seed,
    )
    out = save_report(df, args.out)
    rel = reliability_summary(df)
    rel.to_csv(out / "reliability.csv", index=False)
    summarize_suite(df, weights=weights).to_csv(out / "mechanisms.csv", index=False)
    manifest = {
        "fusionbench_version": __version__,
        "taxonomy": TAXONOMY_VERSION,
        "seed": args.seed,
        "n_per_split": args.n,
        "scenarios": list(weights),
        "scenario_weights": weights,
        "parameters": dict(args.param),
        "application_profile": args.application_profile,
        "methods": [m.name for m in methods],
    }
    (out / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(rel.to_string(index=False))
    print(f"\nreport written to {out}")


def cmd_evaluate(args):
    from fusionbench.eval.report import save_report
    from fusionbench.eval.runner import evaluate_dataset
    from fusionbench.store import load_dataset

    ds = load_dataset(args.dataset)
    methods = _methods(args.methods)
    results = evaluate_dataset(ds, methods)
    out = save_report(results, args.out)
    print(f"report written to {out}")
    failures = results.loc[results.error != "", ["method", "error"]]
    if not failures.empty:
        raise SystemExit(failures.to_string(index=False))


def cmd_sweep(args):
    from fusionbench.eval.sweep import (
        competence_sweep,
        optimism_sweep,
        reliability_sweep,
        save_sweep,
        sweep_table,
    )

    methods = _methods(args.methods)
    print(f"methods: {[m.name for m in methods]}")
    if args.kind == "reliability":
        sc = SCENARIOS[args.scenario]()
        fit_ns = args.fit_ns or [50_000, 5_000, 1_000, 300, 100]
        noises = args.noises or [0.0]
        df = reliability_sweep(
            sc,
            methods,
            fit_ns=fit_ns,
            noises=noises,
            n_per_split=args.n,
            seed=args.seed,
        )
        save_sweep(df, "fit_n", args.out, title=f"{sc.name}: reliability sweep")
        print(
            sweep_table(df[df.label_noise == noises[0]], "fit_n").round(3).to_string()
        )
    elif args.kind == "optimism":
        sc = SCENARIOS[args.scenario]()
        deltas = args.deltas or [0.0, 0.1, 0.2, 0.4]
        df = optimism_sweep(
            sc, methods, deltas=deltas, n_per_split=args.n, seed=args.seed
        )
        save_sweep(
            df, "optimism", args.out, title=f"{sc.name}: validation-optimism sweep"
        )
        print(sweep_table(df, "optimism").round(3).to_string())
    else:
        cls = SCENARIOS[args.scenario]
        accs = args.accuracies or [0.6, 0.7, 0.8, 0.9, 0.95]
        df = competence_sweep(
            cls, methods, accuracies=accs, n_per_split=args.n, seed=args.seed
        )
        save_sweep(df, "accuracy", args.out, title=f"{args.scenario}: competence sweep")
        print(sweep_table(df, "accuracy").round(3).to_string())
    print(f"\nsweep written to {args.out}")


def add_transfer_options(parser):
    from fusionbench.transfer import PRESETS

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=PRESETS, help="packaged transfer grid")
    source.add_argument("--config", help="path to a versioned transfer YAML file")
    parser.add_argument(
        "--scenarios", nargs="+", help="registered scenario IDs to include"
    )
    parser.add_argument("--methods", nargs="+", help="registered method IDs")
    parser.add_argument(
        "--dev-noise",
        nargs="+",
        type=float,
        help="development label flip probabilities",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        help="provider_swap, log_jitter, canonical_shift; clean always included",
    )
    parser.add_argument(
        "--severities", nargs="+", type=float, help="positive stress strengths up to 1"
    )
    parser.add_argument("--fit-rows", type=int)
    parser.add_argument(
        "--validation-rows",
        type=int,
        help="independent source rows for method selection (default: 192)",
    )
    parser.add_argument("--test-rows", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument(
        "--out", help="directory for reports and resolved configuration"
    )


def cmd_transfer_grid(args):
    from pathlib import Path

    import yaml

    from fusionbench.transfer import load_preset, run_transfer

    try:
        if args.config:
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError("Transfer config must be a YAML mapping")
        else:
            cfg = load_preset(args.preset or "smoke")
        if args.scenarios:
            choices = {
                **load_preset("standard")["scenarios"],
                **cfg.get("scenarios", {}),
            }
            missing = set(args.scenarios) - choices.keys()
            if missing:
                raise ValueError(
                    f"Supply a YAML scenario seed_slot for: {', '.join(sorted(missing))}"
                )
            if len(args.scenarios) != len(set(args.scenarios)):
                raise ValueError("Scenario IDs must be distinct")
            cfg["scenarios"] = {name: choices[name] for name in args.scenarios}
        for field in (
            "methods",
            "dev_noise",
            "modes",
            "severities",
            "fit_rows",
            "validation_rows",
            "test_rows",
            "repeats",
            "base_seed",
        ):
            if (value := getattr(args, field)) is not None:
                cfg[field] = value
        if "clean" in (args.modes or []):
            cfg["modes"] = [name for name in args.modes if name != "clean"]
        out = run_transfer(cfg, args.out or "results/transfer-grid")
    except (ValueError, OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Transfer benchmark: {exc}") from exc
    print(f"Transfer benchmark written to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fusionbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    from fusionbench.profiles import add_profile_parser

    add_profile_parser(sub)

    catalog = sub.add_parser(
        "list", help="list generators and installed fusion methods"
    )
    _filters(catalog)
    catalog.set_defaults(func=cmd_list)

    g = sub.add_parser("generate")
    g.add_argument("--out", default="data")
    g.add_argument("--n", type=int, default=50_000)
    g.add_argument("--seed", type=int, default=0)
    _filters(g)
    g.add_argument(
        "--param",
        action="append",
        type=_parameter,
        default=[],
        metavar="NAME=JSON",
        help="set a generator parameter on each selected scenario (repeatable)",
    )
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run")
    r.add_argument("--out", default="results")
    r.add_argument("--n", type=int, default=50_000)
    r.add_argument("--seed", type=int, default=0)
    _filters(r)
    r.add_argument(
        "--param",
        action="append",
        type=_parameter,
        default=[],
        metavar="NAME=JSON",
        help="set a generator parameter on each selected scenario (repeatable)",
    )
    r.add_argument(
        "--methods",
        nargs="*",
        choices=FACTORIES,
        help="fusion methods; OKAPI comparisons include the three reference methods",
    )
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="compare fusion methods on saved predictions")
    e.add_argument("--dataset", required=True, help="directory containing meta.json")
    e.add_argument("--methods", nargs="+", choices=FACTORIES, default=DEFAULT_METHODS)
    e.add_argument("--out", default="results/external")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("sweep")
    s.add_argument(
        "--kind",
        choices=["reliability", "competence", "optimism"],
        default="reliability",
    )
    s.add_argument("--scenario", required=True)
    s.add_argument("--out", default="results/sweep")
    s.add_argument("--n", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--methods",
        nargs="*",
        choices=FACTORIES,
        help="fusion methods; OKAPI comparisons include the three reference methods",
    )
    s.add_argument("--fit-ns", type=int, nargs="*", dest="fit_ns")
    s.add_argument("--noises", type=float, nargs="*")
    s.add_argument("--accuracies", type=float, nargs="*")
    s.add_argument("--deltas", type=float, nargs="*")
    s.set_defaults(func=cmd_sweep)

    grid = sub.add_parser(
        "transfer-grid", aliases=["transfer"], help="run the paired transfer benchmark"
    )
    add_transfer_options(grid)
    grid.set_defaults(func=cmd_transfer_grid)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
