"""Matched native pattern search with two optional pooling candidates."""

from fusionbench.eval.suite_point_gp import _full
from fusionbench.methods.native_pool_candidates import PoolCandidateNativePatternOKAPI


def fit_augmented(x, y, seed=0):
    x = _full(x)
    task = "binary" if x.shape[-1] == 2 else "multiclass"
    model = PoolCandidateNativePatternOKAPI(
        seed=seed, split_seed=0, gp_validation_top_k=8
    )
    return model.fit(x, y, task=task, n_classes=x.shape[-1])
