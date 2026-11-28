"""Native program search with a paired uncertainty deduction on the training rows."""

from numbers import Real

import numpy as np

from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI


def row_brier(predictions, labels):
    """Full-class Brier per example, accepting native positive-only binary scores."""
    p = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if p.shape[1] == 1:
        return 2 * np.square(p[:, 0] - labels)
    return np.square(p - np.eye(p.shape[1])[labels]).sum(axis=1)


class SearchGuardedNativeOKAPI(RegularizedNativePatternOKAPI):
    """Deduct z times the paired standard error of gain over the arithmetic mean.

    GP discovery, feature screening and both head fits use this training score.
    Each fit split has its own arithmetic-mean reference on the identical rows.
    The GP validation champion and three-family terminal choice use ordinary
    unpenalized Brier skill. Existing L2 head regularization is retained. At
    z=0 the estimator follows the historical regularized search path exactly.
    """

    name = "okapi_search_guard"

    def __init__(self, seed=0, *, split_seed=None, z=1.0):
        super().__init__(seed=seed, split_seed=split_seed)
        if (
            isinstance(z, (bool, np.bool_))
            or not isinstance(z, Real)
            or not np.isfinite(z)
            or z < 0
        ):
            raise ValueError("z must be a finite nonnegative real number")
        self.z = float(z)

    @classmethod
    def available(cls):
        if not super().available():
            return False
        try:
            from okapi.paired_fitness import PairedGainFitness  # noqa: F401
        except ImportError:
            return False
        return True

    def _training_fitness(self, probabilities, labels):
        if self.z == 0:
            return None
        from okapi.paired_fitness import PairedGainFitness

        baseline = probabilities.mean(axis=1)
        baseline_skill = 1.0 - 0.5 * row_brier(baseline, labels).mean()
        criterion = PairedGainFitness(baseline, labels, row_brier, z=self.z)

        def score(prediction, training_labels):
            # The additive baseline and half-unit conversion preserve the
            # incumbent's GP rounding and relative scale of the L2 penalty.
            return float(baseline_skill + 0.5 * criterion(prediction, training_labels))

        return score

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.counts_["search_guard_z"] = self.z
        self.counts_["training_fitness"] = (
            "brier_skill_minus_half_z_paired_brier_gain_se"
        )
        return self
