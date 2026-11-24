"""Command-line interface: list / generate / run."""

from __future__ import annotations

import argparse
import json

from fusionbench.methods.registry import CORE, FACTORIES, default_methods
from fusionbench.scenarios.registry import SCENARIOS, STANDARD_SCENARIOS


def _comparison_names(names):
    """Include the three reference methods in each explicit OKAPI comparison."""
    if not names or not any(name.startswith("okapi") for name in names):
        return names
    names = list(dict.fromkeys(names))
    for name in ("simple_average", "top_n"):
        if name not in names:
            names.append(name)
    if not {"stacking_lr", "stacking_logit", "stacking_clr"}.intersection(names):
        names.append("stacking_lr")
    return names


def _parameter(text):
    name, separator, value = text.partition("=")
    if not separator or not name.isidentifier():
        raise argparse.ArgumentTypeError("Use --param name=JSON, for example --param accuracy=0.8")
    try:
        return name, json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON value for parameter {name!r}") from exc


def _scenarios(names, parameters=()):
    from fusionbench import get_scenario

    params = dict(parameters)
    if len(params) != len(parameters):
        raise SystemExit("Specify each scenario parameter once")
    try:
        return [get_scenario(name, **params) for name in (names or STANDARD_SCENARIOS)]
    except (ValueError, TypeError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


def cmd_list(args):
    print("Scenarios:")
    for name, cls in SCENARIOS.items():
        task = cls.task if isinstance(cls.task, str) else "configurable"
        print(f"  {name:22s} task={task:10s} winners={cls.expected_winners}")
    print("\nMethods:")
    for name, factory in FACTORIES.items():
        avail = type(factory()).available()
        print(f"  {name:16s} available={avail}")


def cmd_generate(args):
    from fusionbench.store import save_dataset
    for sc in _scenarios(args.scenarios, args.param):
        ds = sc.build(n_per_split=args.n, seed=args.seed)
        path = save_dataset(ds, args.out)
        print(f"wrote {path}")


def cmd_run(args):
    from fusionbench.eval.reliability import reliability_summary
    from fusionbench.eval.report import save_report
    from fusionbench.eval.runner import run_suite

    names = _comparison_names(args.methods)
    methods = default_methods(names or None)
    if names and (missing := set(names) - {m.name for m in methods}):
        raise SystemExit(f"Install optional dependencies for: {', '.join(sorted(missing))}")
    print(f"methods: {[m.name for m in methods]}")
    df = run_suite(_scenarios(args.scenarios, args.param), methods=methods,
                   n_per_split=args.n, seed=args.seed)
    out = save_report(df, args.out)
    rel = reliability_summary(df)
    rel.to_csv(out / "reliability.csv", index=False)
    print(rel.to_string(index=False))
    print(f"\nreport written to {out}")


def cmd_evaluate(args):
    from fusionbench.eval.report import save_report
    from fusionbench.eval.runner import evaluate_dataset
    from fusionbench.store import load_dataset

    ds = load_dataset(args.dataset)
    names = _comparison_names(args.methods)
    methods = default_methods(names)
    available = {m.name for m in methods}
    missing = set(names) - available
    if missing:
        raise SystemExit(f"Install optional dependencies for: {', '.join(sorted(missing))}")
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

    methods = default_methods(_comparison_names(args.methods) or None)
    print(f"methods: {[m.name for m in methods]}")
    if args.kind == "reliability":
        sc = SCENARIOS[args.scenario]()
        fit_ns = args.fit_ns or [50_000, 5_000, 1_000, 300, 100]
        noises = args.noises or [0.0]
        df = reliability_sweep(sc, methods, fit_ns=fit_ns, noises=noises,
                               n_per_split=args.n, seed=args.seed)
        save_sweep(df, "fit_n", args.out, title=f"{sc.name}: reliability sweep")
        print(sweep_table(df[df.label_noise == noises[0]], "fit_n").round(3).to_string())
    elif args.kind == "optimism":
        sc = SCENARIOS[args.scenario]()
        deltas = args.deltas or [0.0, 0.1, 0.2, 0.4]
        df = optimism_sweep(sc, methods, deltas=deltas, n_per_split=args.n, seed=args.seed)
        save_sweep(df, "optimism", args.out, title=f"{sc.name}: validation-optimism sweep")
        print(sweep_table(df, "optimism").round(3).to_string())
    else:
        cls = SCENARIOS[args.scenario]
        accs = args.accuracies or [0.6, 0.7, 0.8, 0.9, 0.95]
        df = competence_sweep(cls, methods, accuracies=accs, n_per_split=args.n,
                              seed=args.seed)
        save_sweep(df, "accuracy", args.out, title=f"{args.scenario}: competence sweep")
        print(sweep_table(df, "accuracy").round(3).to_string())
    print(f"\nsweep written to {args.out}")


def cmd_stability(args):
    from fusionbench.eval.search_stability import run_search_stability

    options = {key: value for key, value in vars(args).items() if key not in {"cmd", "func"}}
    run_search_stability(**options)


def cmd_mechanism_transfer(args):
    from fusionbench.eval.mechanism_transfer import run_study

    run_study(args.out, args.seeds or list(range(8)),
              rescue_rates=args.rescue_rates or [0., 0.03125, 0.5], followup=args.followup)


def add_transfer_options(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=("smoke", "standard", "standard-v1", "standard-v2", "standard-v3"), help="packaged transfer grid")
    source.add_argument("--config", help="path to a versioned transfer YAML file")
    parser.add_argument("--scenarios", nargs="+", help="registered scenario IDs to include")
    parser.add_argument("--methods", nargs="+", help="registered method IDs")
    parser.add_argument("--dev-noise", nargs="+", type=float, help="development label flip probabilities")
    parser.add_argument("--modes", nargs="+", help="provider_swap, log_jitter, canonical_shift; clean always included")
    parser.add_argument("--severities", nargs="+", type=float, help="positive stress strengths up to 1")
    parser.add_argument("--fit-rows", type=int)
    parser.add_argument("--validation-rows", type=int, help="independent source rows for method selection (default: 192)")
    parser.add_argument("--test-rows", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--out", help="directory for reports and resolved configuration")


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
            choices = {**load_preset("standard")["scenarios"], **cfg.get("scenarios", {})}
            missing = set(args.scenarios)-choices.keys()
            if missing:
                raise ValueError(f"Supply a YAML scenario seed_slot for: {', '.join(sorted(missing))}")
            if len(args.scenarios) != len(set(args.scenarios)):
                raise ValueError("Scenario IDs must be distinct")
            cfg["scenarios"] = {name: choices[name] for name in args.scenarios}
        for field in ("methods", "dev_noise", "modes", "severities", "fit_rows", "validation_rows", "test_rows",
                      "repeats", "base_seed"):
            if (value := getattr(args, field)) is not None:
                cfg[field] = value
        if "clean" in (args.modes or []):
            cfg["modes"] = [name for name in args.modes if name != "clean"]
        out = run_transfer(cfg, args.out or "results/transfer-grid")
    except (ValueError, OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Transfer benchmark: {exc}") from exc
    print(f"Transfer benchmark written to {out}")


def cmd_transfer(args):
    grid = any(getattr(args, field) is not None for field in
               ("preset", "config", "methods", "dev_noise", "modes", "severities",
                "fit_rows", "validation_rows", "test_rows", "repeats", "base_seed", "scenarios"))
    legacy = args.seeds is not None or args.rescue_rates is not None or args.followup
    if grid and legacy:
        raise SystemExit("Use transfer-grid options or the MLP study's --seeds/--rescue-rates/--followup separately")
    if legacy or (args.out is not None and not grid):
        if args.out is None:
            raise SystemExit("The MLP mechanism study requires --out")
        return cmd_mechanism_transfer(args)
    return cmd_transfer_grid(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fusionbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    from fusionbench.profiles import add_profile_parser
    add_profile_parser(sub)

    sub.add_parser("list").set_defaults(func=cmd_list)

    g = sub.add_parser("generate")
    g.add_argument("--out", default="data")
    g.add_argument("--n", type=int, default=50_000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--scenarios", nargs="*")
    g.add_argument("--param", action="append", type=_parameter, default=[],
                   metavar="NAME=JSON", help="set a generator parameter on each selected scenario (repeatable)")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run")
    r.add_argument("--out", default="results")
    r.add_argument("--n", type=int, default=50_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--scenarios", nargs="*")
    r.add_argument("--param", action="append", type=_parameter, default=[],
                   metavar="NAME=JSON", help="set a generator parameter on each selected scenario (repeatable)")
    r.add_argument("--methods", nargs="*")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="compare fusion methods on saved predictions")
    e.add_argument("--dataset", required=True, help="directory containing meta.json")
    e.add_argument("--methods", nargs="+", choices=FACTORIES, default=CORE)
    e.add_argument("--out", default="results/external")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("sweep")
    s.add_argument("--kind", choices=["reliability", "competence", "optimism"],
                   default="reliability")
    s.add_argument("--scenario", required=True)
    s.add_argument("--out", default="results/sweep")
    s.add_argument("--n", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--methods", nargs="*")
    s.add_argument("--fit-ns", type=int, nargs="*", dest="fit_ns")
    s.add_argument("--noises", type=float, nargs="*")
    s.add_argument("--accuracies", type=float, nargs="*")
    s.add_argument("--deltas", type=float, nargs="*")
    s.set_defaults(func=cmd_sweep)

    from fusionbench.eval.search_stability import add_stability_arguments

    stability = sub.add_parser("stability", help="cross dataset seeds with GP seeds and compare tree selectors")
    add_stability_arguments(stability)
    stability.set_defaults(func=cmd_stability)

    grid = sub.add_parser("transfer-grid", help="run paired development noise and deployment stresses")
    add_transfer_options(grid)
    grid.set_defaults(func=cmd_transfer_grid)

    transfer = sub.add_parser("transfer", help="use --preset smoke or --config FILE for the transfer grid",
                              description="Use --preset smoke/standard or --config FILE for the transfer grid. "
                                          "The legacy --out DIR alone runs the MLP mechanism study.")
    add_transfer_options(transfer)
    transfer.add_argument("--seeds", nargs="+", type=int, help="legacy MLP study seeds")
    transfer.add_argument("--rescue-rates", nargs="+", type=float, help="legacy MLP rescue rates")
    transfer.add_argument("--followup", action="store_true", help="legacy MLP algebraic training losses")
    transfer.set_defaults(func=cmd_transfer)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
