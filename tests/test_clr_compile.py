import numpy as np
import pytest
from scipy.special import softmax

from fusionbench.methods.clr import StackingCLR


@pytest.mark.parametrize("classes,n_present",[(2,2),(3,3),(4,2)])
def test_compiled_softmax_matches_native_stack_on_unseen_features(classes,n_present):
    rng=np.random.default_rng(5)
    w=rng.normal(size=(2,classes,5))
    b=rng.normal(size=(2,classes))
    def predict(x):
        return softmax(np.einsum('id,kcd->ikc',x,w)+b,axis=2)
    x=rng.normal(size=(80,5))
    labels=rng.integers(n_present,size=len(x))
    if n_present < classes:
        labels=labels*2
    model=StackingCLR(multiplicities=[2.,1.]).fit(predict(x),labels,task='multiclass',n_classes=classes)
    held=rng.normal(size=(40,5))
    np.testing.assert_allclose(model.compile(w,b).predict(held),model.predict(predict(held)),atol=1e-12)


def test_multiplicity_recovers_the_penalty_geometry_of_true_copies():
    rng=np.random.default_rng(8)
    p=softmax(rng.normal(size=(120,2,3)),axis=-1)
    y=rng.integers(3,size=len(p))
    duplicate=StackingCLR().fit(p[:,[0,0,1]],y,task='multiclass',n_classes=3)
    weighted=StackingCLR(multiplicities=[2.,1.]).fit(p,y,task='multiclass',n_classes=3)
    np.testing.assert_allclose(duplicate.predict(p[:,[0,0,1]]),weighted.predict(p),atol=5e-6)
