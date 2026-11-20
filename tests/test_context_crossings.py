"""Implications of exchanging equal-product reliability regimes."""
import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fusionbench.eval.context_crossings import FusionNet, HELD_OUT, SOURCE, conditional_risk


def test_held_regimes_have_identical_unlabeled_primary_law():
    assert not set(HELD_OUT).intersection(SOURCE)
    x = np.array(list(itertools.product((-1.,1.), repeat=2)))
    laws = [sum(.5*np.prod((1+y*x*np.asarray(r))/2,axis=1) for y in (-1,1)) for r in HELD_OUT]
    np.testing.assert_allclose(*laws)
    strengths = np.array(HELD_OUT)
    probs = 1/(1+np.exp(-2*np.arctanh(strengths)@x.T))
    _, acc = conditional_risk(probs,x,strengths)
    np.testing.assert_allclose(acc, .925)


def test_tied_model_respects_exchange_and_ignores_absent_support():
    torch.manual_seed(100)
    net = FusionNet(tied=True)
    support = torch.randn(4,32,3)
    query = torch.randn(4,16,2)
    available = torch.ones(4,1)
    torch.testing.assert_close(net(support,available,query),
                               net(support[..., [1,0,2]],available,query.flip([-1])))
    torch.testing.assert_close(net(support,0*available,query),net(-support,0*available,query))
