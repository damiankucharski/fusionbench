import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("okapi.node")
from fusionbench.eval.intervention_witness import replay_tree
from okapi.globals import set_backend, set_postprocessing_function


def test_perturbation_changes_only_confidence_gated_records():
    set_backend("numpy")
    set_postprocessing_function(lambda x:x)
    gate=np.array([[True,False,True,False]])
    private=np.array([[1,1,-1,-1]])
    primary=np.array([[1,-1,1,-1]])
    base,changed=replay_tree(private,primary,gate)
    np.testing.assert_array_equal(base,np.where(gate,primary,private))
    np.testing.assert_array_equal(changed,np.where(gate,-primary,private))
    np.testing.assert_array_equal(base!=changed,gate)
