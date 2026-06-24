"""Command-line interface: list / generate / run."""

from __future__ import annotations

import argparse

from fusionbench.methods.registry import FACTORIES, default_methods
from fusionbench.scenarios.registry import SCENARIOS, default_suite


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

    methods = default_methods(args.methods or None)
    print(f"methods: {[m.name for m in methods]}")
    df = run_suite(_scenarios(args.scenarios), methods=methods,
                   n_per_split=args.n, seed=args.seed)
    out = save_report(df, args.out)
    rel = reliability_summary(df)
    rel.to_csv(out / "reliability.csv", index=False)
    print(rel.to_string(index=False))
    print(f"\nreport written to {out}")


def cmd_sweep(args):
    from fusionbench.eval.sweep import (
        competence_sweep,
        reliability_sweep,
        save_sweep,
        sweep_table,
    )

    methods = default_methods(args.methods or None)
    print(f"methods: {[m.name for m in methods]}")
    if args.kind == "reliability":
        sc = SCENARIOS[args.scenario]()
        fit_ns = args.fit_ns or [50_000, 5_000, 1_000, 300, 100]
        noises = args.noises or [0.0]
        df = reliability_sweep(sc, methods, fit_ns=fit_ns, noises=noises,
                               n_per_split=args.n, seed=args.seed)
        save_sweep(df, "fit_n", args.out, title=f"{sc.name}: reliability sweep")
        print(sweep_table(df[df.label_noise == noises[0]], "fit_n").round(3).to_string())
    else:
        cls = SCENARIOS[args.scenario]
        accs = args.accuracies or [0.6, 0.7, 0.8, 0.9, 0.95]
        df = competence_sweep(cls, methods, accuracies=accs, n_per_split=args.n,
                              seed=args.seed)
        save_sweep(df, "accuracy", args.out, title=f"{args.scenario}: competence sweep")
        print(sweep_table(df, "accuracy").round(3).to_string())
    print(f"\nsweep written to {args.out}")


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

    s = sub.add_parser("sweep")
    s.add_argument("--kind", choices=["reliability", "competence"], default="reliability")
    s.add_argument("--scenario", required=True)
    s.add_argument("--out", default="results/sweep")
    s.add_argument("--n", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--methods", nargs="*")
    s.add_argument("--fit-ns", type=int, nargs="*", dest="fit_ns")
    s.add_argument("--noises", type=float, nargs="*")
    s.add_argument("--accuracies", type=float, nargs="*")
    s.set_defaults(func=cmd_sweep)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
