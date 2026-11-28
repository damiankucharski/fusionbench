"""Enumerate the frozen binary KL rule and verify every winning comparison."""

import argparse
import csv
import io
import math
import runpy
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = runpy.run_path(str(HERE.parent / "verify.py"))
PROGRAMS = ORIGINAL["PROGRAMS"]
DEN = 2**56


def lower_root_bracket(active, correct):
    """Return dyadic numerators bracketing the lower Bernoulli KL root."""
    if correct == 0:
        return 0, 0
    missed = active - correct
    # On [0, v], exp(active*KL(v || j/DEN)) >= 160 iff the
    # integer on the left is at least the integer on the right below.
    left = correct**correct * missed**missed * DEN**active
    factor = 160 * active**active
    lo, hi = 0, (correct * DEN + active - 1) // active
    while hi - lo > 1:
        mid = (lo + hi) // 2
        right = factor * mid**correct * (DEN - mid)**missed
        if left >= right:
            lo = mid
        else:
            hi = mid
    return lo, hi


def double_root(active, correct):
    """Independent floating-point inversion of the log equation for auditing."""
    if correct == 0:
        return 0.
    target = math.log(160) / active
    if correct == active:
        return math.exp(-target)
    v = correct / active
    lo, hi = 0., v
    for _ in range(64):
        p = (lo + hi) / 2
        kl = v * math.log(v / p) + (1 - v) * math.log((1 - v) / (1 - p))
        if kl >= target:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def score_bounds(active, correct):
    lo, hi = lower_root_bracket(active, correct)
    approximate = double_root(active, correct)
    assert lo / DEN - 2e-14 <= approximate <= hi / DEN + 2e-14
    # Drop the common positive factor 1.6 / (10*N*DEN).
    return active * (10 * lo - 7 * DEN), active * (10 * hi - 7 * DEN)


def kl_decisions(per_context, old):
    table = {(0, 0): (0, 0)}
    for active in (per_context, 2 * per_context):
        for correct in range(active + 1):
            table[active, correct] = score_bounds(active, correct)
    decisions = {}
    for (k1, k2), (raw, _) in old.items():
        agreement = []
        for program in PROGRAMS:
            active = correct = 0
            for q, k in zip(program, (k1, k2), strict=True):
                if q != 5:
                    active += per_context
                    correct += k if q == 9 else per_context - k
            agreement.append((active, correct))
        best = max(range(len(PROGRAMS)), key=lambda i: table[agreement[i]][0])
        # Symmetric programs with identical counts have exactly the same
        # root. Confirm strict separation from every other score interval.
        for other in set(agreement) - {agreement[best]}:
            assert table[agreement[best]][0] > table[other][1]
        decisions[k1, k2] = (raw, best)
    return decisions


def render():
    with (HERE.parent / "results.csv").open(encoding="utf-8") as file:
        historical = {(int(r["n"]), r["law"], r["rule"]): r for r in csv.DictReader(file)}
    out = io.StringIO()
    fields = tuple(next(iter(historical.values())))
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for n in ORIGINAL["SIZES"]:
        half = n // 2
        old = ORIGINAL["policy_table"](half)
        decisions = kl_decisions(half, old)
        for name, law in ORIGINAL["LAWS"].items():
            raw, kl = ORIGINAL["summarize"](half, law, decisions)
            raw_row = historical[n, name, "raw"]
            assert tuple(raw_row[f] for f in fields[-4:]) == tuple(format(float(v), ".12g") for v in raw)
            writer.writerow(raw_row)
            writer.writerow({**historical[n, name, "certified"], "rule": "hoeffding"})
            writer.writerow({**raw_row, "rule": "kl", **{f: format(float(v), ".12g") for f, v in zip(fields[-4:], kl, strict=True)}})
            if n == 16 and name == "strong":
                assert kl[-1] == Fraction(9, 10)**16
            if n == 16 and name == "null":
                assert kl[1] == Fraction(4, 2**16)
    return out.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write results.csv instead of verifying it")
    args = parser.parse_args()
    data = render()
    destination = HERE / "results.csv"
    if args.write:
        destination.write_text(data, encoding="utf-8")
    elif destination.read_text(encoding="utf-8") != data:
        raise SystemExit("results.csv differs from enumeration")
    print(data, end="")
