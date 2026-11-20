"""Check indistinguishability using the entire sign distribution."""
import itertools

import numpy as np
import pytest

pytest.importorskip("torch")
from fusionbench.eval.reference_copy import REFERENCE, STRONG, WEAK, alias_parameters


def test_weaker_primary_copy_matches_opposite_clean_law():
    copy, residual = alias_parameters()
    assert 0 < copy < 1 and 0 < residual < 1
    def sensor(sign, y, r):
        return (1 + sign*y*r)/2
    for a,b,h in itertools.product((-1,1), repeat=3):
        clean = sum(.5*sensor(a,y,STRONG)*sensor(b,y,WEAK)*sensor(h,y,REFERENCE) for y in (-1,1))
        copied = sum(.5*sensor(a,y,WEAK)*sensor(b,y,STRONG)*
                     (copy*(h==a) + (1-copy)*sensor(h,y,residual)) for y in (-1,1))
        np.testing.assert_allclose(clean,copied,rtol=1e-14,atol=1e-15)
