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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
