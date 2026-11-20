"""Controlled reconstructions of routing and operator-ablation pilot designs.

Each run draws independent fit and test sets. These experiments use simulated
expert scores. The routing student sees those scores at inference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr

from fusionbench.rules import eval_rule
from fusionbench.scenarios import calibrated as cal


def _detector_logodds(label, accuracy, rng):
    # Same binary Gaussian observation model as cal.binary_calibrated.
    # Keep the likelihood ratio before clipping for the analytic audit.
    strength = cal.acc_to_strength(accuracy)
    return 2 * strength * ((2 * label.astype(float) - 1) * strength + rng.normal(size=len(label)))


def _routing_draw(n, rng, rare_prior, selector_accuracy=.92, sampling_prior=None):
    rare = rng.random(n) < (rare_prior if sampling_prior is None else sampling_prior)
    y = rng.integers(2, size=n)
    a = expit(_detector_logodds(np.where(rare, rng.integers(2, size=n), y), .90, rng))
    b = expit(_detector_logodds(np.where(rare, y, rng.integers(2, size=n)), .90, rng))
    posterior = expit(_detector_logodds(rare, selector_accuracy, rng) + logit(rare_prior))
    teacher = (1 - posterior) * a + posterior * b
    return a, b, posterior, rare, y, teacher


def _student_inputs(draw, visible):
    a, b, selector = draw[:3]
    a, b = 2 * a - 1, 2 * b - 1
    columns = [np.ones(len(a)), a, b]
    if visible:
        columns.extend([selector, a * selector, b * selector])
    return np.column_stack(columns)


def _fit_student(x, target):
    """Logistic student with an L2 penalty on all non-intercept terms."""
    penalty = 0.001

    def objective(weights):
        logits = x @ weights
        loss = np.mean(np.logaddexp(0, logits) - target * logits)
        gradient = x.T @ (expit(logits) - target) / len(target)
        loss += penalty * (weights[1:] @ weights[1:]) / 2
        gradient[1:] += penalty * weights[1:]
        return loss, gradient

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-8, "ftol": 1e-12})
    if not fit.success:
        raise RuntimeError(f"Routing fit failed: {fit.message}")
    return fit.x


def _routing(seed, fit_sizes, n_test, fit_prior):
    # Nested fit sets and a common independent test make all contrasts paired.
    prior = 0.10
    test = _routing_draw(n_test, np.random.default_rng([seed, 1]), prior)
    train = _routing_draw(max(fit_sizes), np.random.default_rng([seed, 2]), prior, sampling_prior=fit_prior)
    a, b, _, rare, y, teacher = test
    reference = (a > 0.5) == y
    ensemble = (teacher > 0.5) == y
    rows = []
    for size in fit_sizes:
        fit_y = train[4][:size]
        for visible in (False, True):
            train_x = _student_inputs(train, visible)[:size]
            test_x = _student_inputs(test, visible)
            # Accuracy-optimal decision for the exact information available.
            available_teacher = teacher if visible else (1 - prior) * a + prior * b
            predictions = {"oracle_for_visible_inputs": available_teacher > 0.5}
            targets = {"labels": fit_y, "distill": train[5][:size]}
            for name, target in targets.items():
                predictions[name] = test_x @ _fit_student(train_x, target) > 0
            for name, prediction in predictions.items():
                correct = prediction == y
                for cohort, mask in (("all", np.ones(n_test, bool)), ("rare", rare)):
                    repair = ~reference & ensemble & mask
                    rows.append(dict(seed=seed, n_fit=size, fit_rare_prior=fit_prior, fit_rare=int(train[3][:size].sum()),
                                     routing="visible" if visible else "hidden", supervision=name,
                                     cohort=cohort, n=int(mask.sum()), teacher_repairs=int(repair.sum()),
                                     retained=int(np.sum(correct & repair)),
                                     student_correct=int(correct[mask].sum()),
                                     reference_correct=int(reference[mask].sum()),
                                     teacher_correct=int(ensemble[mask].sum()),
                                     lost_reference=int(np.sum(reference & ~correct & mask))))
    return rows


def _flip_expectations(draw, prior, selector_accuracy, visible):
    """Conditional probabilities of teacher repair and reference correctness.

    A hidden-selector policy may independently decide which reference decisions
    to flip using only (a,b). When a and b disagree the teacher flips precisely
    when its posterior on the rare domain exceeds (.5-a)/(b-a). Integrate the
    Gaussian selector likelihood separately in each latent domain.
    """
    a, b, _, _, _, teacher = draw
    reference = a > .5
    if visible:
        reference_correct = np.where(reference, teacher, 1 - teacher)
        repaired = ((teacher > .5) != reference) * (1 - reference_correct)
        return repaired, reference_correct
    a_correct = np.where(reference, a, 1 - a)
    b_correct = np.where(reference, b, 1 - b)
    reference_correct = (1 - prior) * a_correct + prior * b_correct
    repaired = np.zeros_like(a)
    disagree = (b > .5) != reference
    cut = (.5 - a[disagree]) / (b[disagree] - a[disagree])
    strength = cal.acc_to_strength(selector_accuracy)
    evidence_cut = (logit(cut) - logit(prior)) / (2 * strength)
    flip_common = ndtr(-strength - evidence_cut)
    flip_rare = ndtr(strength - evidence_cut)
    repaired[disagree] = ((1 - prior) * (1 - a_correct[disagree]) * flip_common
                          + prior * (1 - b_correct[disagree]) * flip_rare)
    return repaired, reference_correct


def _frontier(seed, n_test, n_calibration):
    rows = []
    for prior in (.10, .50):
        for selector_accuracy in (.75, .92):
            calibration = _routing_draw(n_calibration, np.random.default_rng([seed, 4]), prior, selector_accuracy)
            test = _routing_draw(n_test, np.random.default_rng([seed, 5]), prior, selector_accuracy)
            a, _, _, _, y, teacher = test
            ref_correct = (a > .5) == y
            ens_correct = (teacher > .5) == y
            repair = ~ref_correct & ens_correct
            confident = np.maximum(a, 1 - a) >= .95
            # Calibrate a common budget from the teacher's expected collateral.
            _, ref_cost = _flip_expectations(calibration, prior, selector_accuracy, True)
            teacher_flip = (calibration[0] > .5) != (calibration[5] > .5)
            reference_budget = np.mean(teacher_flip * ref_cost)
            for visible in (False, True):
                expected_repair, expected_cost = _flip_expectations(calibration, prior, selector_accuracy, visible)
                ratio = expected_repair / expected_cost
                active = np.flatnonzero(ratio > 0)
                ordered = active[np.argsort(-ratio[active], kind="stable")]
                cumulative = np.cumsum(expected_cost[ordered]) / len(ratio)
                test_repair, test_cost = _flip_expectations(test, prior, selector_accuracy, visible)
                budgets = [(f"teacher_x{factor:g}", factor * reference_budget) for factor in (.25, 1., 2.)]
                budgets.append(("fixed", .01 if prior == .1 else .05))
                for policy, budget in budgets:
                    count = np.searchsorted(cumulative, budget, side="right")
                    threshold = np.inf if count == 0 else ratio[ordered[count - 1]]
                    # For a saturated budget, all positive-benefit flips are allowed.
                    if count == len(ordered):
                        threshold = 0.0
                    flip = (test_repair > 0) & (test_repair >= threshold * test_cost)
                    rows.append(dict(seed=seed, prior=prior, selector_accuracy=selector_accuracy,
                                     routing="visible" if visible else "hidden", budget_policy=policy, n_calibration=n_calibration,
                                     target_loss_fraction=budget, n=n_test, teacher_repairs=int(repair.sum()),
                                     retained=int(np.sum(repair & flip)), lost_reference=int(np.sum(ref_correct & flip)),
                                     predicted_repairs=float(test_repair[flip].sum()),
                                     predicted_losses=float(test_cost[flip].sum()),
                                     teacher_losses=int(np.sum(ref_correct & ~ens_correct)),
                                     repairs_confident_reference=int(np.sum(repair & confident))))
    return rows


def _rule(op, *args):
    return {"op": op, "args": list(args)}


def _repair_veto(seed, n_test):
    rng = np.random.default_rng([seed, 3])
    core = rng.random(n_test) < 0.35
    rescue = rng.random(n_test) < 0.12
    safe = rng.random(n_test) < 0.75
    y = (core | rescue) & safe
    # Scores are noisy sensors of their named condition; thresholds below use
    # the detector's equal-prior convention, including for the rare channel.
    scores = np.stack([cal.binary_calibrated(c, acc, rng)
                       for c, acc in ((core, .80), (core, .80), (rescue, .88), (safe, .90))], axis=1)
    scores = scores[:, :, None]
    c1, c2, r, s = ({"model": i} for i in range(4))
    consensus = _rule("mean", c1, c2)
    rescue_step = _rule("max", consensus, r)
    full = _rule("min", rescue_step, s)
    correct = (eval_rule(scores, full).ravel() > .5) == y
    rows = []
    for op, bypass in (("max_rescue", _rule("min", consensus, s)), ("min_safe", rescue_step)):
        reduced = (eval_rule(scores, bypass).ravel() > .5) == y
        for group, mask in (("all", np.ones(n_test, bool)), ("rescue_only_safe", ~core & rescue & safe),
                            ("unsafe_presence", (core | rescue) & ~safe)):
            repaired = int(np.sum(mask & ~reduced & correct))
            harmed = int(np.sum(mask & reduced & ~correct))
            assert repaired - harmed == (int(correct[mask].sum()) - int(reduced[mask].sum()))
            rows.append(dict(seed=seed, operator=op, cohort=group, n=int(mask.sum()),
                             repaired=repaired, harmed=harmed, full_correct=int(correct[mask].sum()),
                             bypass_correct=int(reduced[mask].sum())))
    return rows


def run_reconstruction(out, seeds=tuple(range(10)), fit_sizes=(64, 512), n_test=8192,
                       n_calibration=32768, frontier_only=False):
    """Write seed-level numerator/denominator records for both pilots."""
    if not seeds or not fit_sizes or min(fit_sizes) < 1 or min(n_test, n_calibration) < 1:
        raise ValueError("Pass at least one seed and positive sample sizes")
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    for filename, fn in (("routing.csv", lambda s: _routing(s, fit_sizes, n_test, .10)),
                         ("routing_low_coverage.csv", lambda s: _routing(s, fit_sizes, n_test, .025)),
                         ("repair_veto.csv", lambda s: _repair_veto(s, n_test)),
                         ("frontier.csv", lambda s: _frontier(s, n_test, n_calibration))):
        if frontier_only and filename != "frontier.csv":
            continue
        rows = [row for seed in seeds for row in fn(seed)]
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--fit-sizes", nargs="+", type=int, default=[64, 512])
    parser.add_argument("--n-test", type=int, default=8192)
    parser.add_argument("--n-calibration", type=int, default=32768)
    parser.add_argument("--frontier-only", action="store_true")
    run_reconstruction(**vars(parser.parse_args()))
