import hashlib

import numpy as np
import pytest
from r1 import fit_variants

from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI


@pytest.mark.parametrize("classes", [2, 5])
def test_full_archive_parity_and_nested_discovery_menus(classes):
    pytest.importorskip("okapi.pattern_features")
    rng = np.random.default_rng(4)
    x = rng.dirichlet(np.ones(classes), size=(95, 6))
    labels = rng.integers(classes, size=len(x))
    fitted, summary = fit_variants(x, labels)
    stock = RegularizedNativePatternOKAPI(seed=0, split_seed=0).fit(
        x, labels, task="binary" if classes == 2 else "multiclass", n_classes=classes
    )
    assert stock.counts_["gp_path_digest"] == summary["native_path"]
    np.testing.assert_array_equal(stock.predict(x), fitted["regularized"].predict(x))
    assert summary["choices"]["gp_top1"]["gp_discovery_rank"] == 1
    assert summary["choices"]["gp_top8"]["gp_discovery_rank"] <= 8
    for name, model in fitted.items():
        ids = stock.validation_indices_
        val_pred = model.validation_predict(x[ids])
        val_brier = ((val_pred - np.eye(classes)[labels[ids]]) ** 2).sum(axis=1).mean()
        assert val_brier == model.validation_brier_
        assert summary["choices"][name]["validation_brier"] == val_brier
    # A full-archive cap reuses validation results across the initial and final
    # checkpoints. Top-1 and Top-8 reproduce the independent full replay above.
    for cap, name in ((1, "gp_top1"), (8, "gp_top8"), (10_000, "regularized")):
        public = RegularizedNativePatternOKAPI(
            seed=0, split_seed=0, gp_validation_top_k=cap
        ).fit(
            x,
            labels,
            task="binary" if classes == 2 else "multiclass",
            n_classes=classes,
        )
        original = summary["choices"][name]
        assert public.counts_["gp_path_digest"] == summary["native_path"]
        assert public.chosen_ == original["family"]
        assert public.counts_["gp_final_eligible"] == original["eligible_gp"]
        calls = public.counts_["score_calls"]["gp_validation"]
        assert calls <= min(2 * cap, summary["archive_size"])
        if cap >= summary["archive_size"]:
            assert calls == summary["archive_size"]
        digest = hashlib.sha256(
            repr(public.trees_["gp"].structural_key()).encode()
        ).hexdigest()
        assert digest == original["gp_hash"]
        np.testing.assert_array_equal(public.predict(x), fitted[name].predict(x))
