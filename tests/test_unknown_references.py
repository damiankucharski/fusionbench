import numpy as np
import pytest

pytest.importorskip("torch")
from fusionbench.eval.unknown_references import four_bit_law


def test_two_unknown_references_cannot_break_the_observable_alias():
    a, b = four_bit_law(0), four_bit_law(1)
    assert len(a) == 16
    assert np.all(a>0)
    np.testing.assert_allclose(a.sum(),1.)
    np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-15)
