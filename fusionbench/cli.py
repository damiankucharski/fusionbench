"""Command-line interface: list / generate / run."""

from __future__ import annotations

import argparse

from fusionbench.methods.registry import CORE, FACTORIES, default_methods
from fusionbench.scenarios.registry import SCENARIOS, default_suite


def _comparison_names(names):
    """Include the three reference methods in each explicit OKAPI comparison."""
    if not names or not {"okapi", "okapi_shrunk"}.intersection(names):
        return names
    names = list(dict.fromkeys(names))
    for name in ("simple_average", "top_n"):
        if name not in names:
            names.append(name)
    if not {"stacking_lr", "stacking_logit", "stacking_clr"}.intersection(names):
        names.append("stacking_lr")
    return names


def _scenarios(names):
    if not names:
        return default_suite()
    return [SCENARIOS[n]() for n in names]


def cmd_list(args):
    print("Scenarios:")
    for name, cls in SCENARIOS.items():
        inst = cls()
        print(f"  {name:22s} task={inst.task:10s} winners={inst.expected_winners}")
    print("\nMethods:")
    for name, factory in FACTORIES.items():
        avail = type(factory()).available()
        print(f"  {name:16s} available={avail}")


def cmd_generate(args):
    from fusionbench.store import save_dataset
    for sc in _scenarios(args.scenarios):
        ds = sc.build(n_per_split=args.n, seed=args.seed)
        path = save_dataset(ds, args.out)
        print(f"wrote {path}")


def cmd_run(args):
    from fusionbench.eval.reliability import reliability_summary
    from fusionbench.eval.report import save_report
    from fusionbench.eval.runner import run_suite

    methods = default_methods(_comparison_names(args.methods) or None)
    print(f"methods: {[m.name for m in methods]}")
    df = run_suite(_scenarios(args.scenarios), methods=methods,
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


def cmd_transfer(args):
    from fusionbench.eval.mechanism_transfer import run_study

    run_study(args.out, args.seeds, rescue_rates=args.rescue_rates, followup=args.followup)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fusionbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    g = sub.add_parser("generate")
    g.add_argument("--out", default="data")
    g.add_argument("--n", type=int, default=50_000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--scenarios", nargs="*")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run")
    r.add_argument("--out", default="results")
    r.add_argument("--n", type=int, default=50_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--scenarios", nargs="*")
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

    transfer = sub.add_parser("transfer", help="train one MLP using symmetries of a fitted ensemble")
    transfer.add_argument("--out", required=True)
    transfer.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    transfer.add_argument("--rescue-rates", nargs="+", type=float, default=[0., 0.03125, 0.5])
    transfer.add_argument("--followup", action="store_true", help="compare algebraic training losses")
    transfer.set_defaults(func=cmd_transfer)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
