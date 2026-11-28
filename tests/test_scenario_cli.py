import numpy as np
import pytest

from fusionbench.cli import main


def test_generate_cli_matches_public_api(tmp_path):
    from fusionbench import generate_dataset, load_dataset

    main(["generate", "--scenarios", "committee", "--n", "30", "--seed", "18",
          "--param", "k=3", "--param", "accuracy=0.81", "--out", str(tmp_path)])
    path, = tmp_path.iterdir()
    generated = generate_dataset("committee", n_per_split=30, seed=18, k=3, accuracy=0.81)
    saved = load_dataset(path)
    assert generated.meta == saved.meta
    assert saved.meta["scenario_id"] == "committee"
    np.testing.assert_array_equal(saved.test_preds, generated.test_preds.astype(np.float32))
    np.testing.assert_array_equal(saved.test_y, generated.test_y)


@pytest.mark.parametrize("extra", [
    ["--param", "accuracy=bad"],
    ["--param", "k=2", "--param", "k=3"],
    ["--param", "misspelled=2"],
])
def test_bad_cli_parameters_fail_before_writing(tmp_path, extra):
    with pytest.raises(SystemExit):
        main(["generate", "--scenarios", "committee", "--n", "8", "--out", str(tmp_path), *extra])
    assert not list(tmp_path.iterdir())


def test_discovery_can_list_extensions_with_required_parameters(monkeypatch, capsys):
    from fusionbench import register_scenario
    from fusionbench.cli import _scenarios, cmd_list
    from fusionbench.scenarios.committee import Committee
    from fusionbench.scenarios.registry import SCENARIOS, default_suite

    class CustomCommittee(Committee):
        def __init__(self, required_k):
            super().__init__(k=required_k)

    monkeypatch.setattr("fusionbench.cli.FACTORIES", {})
    register_scenario("custom_committee", CustomCommittee)
    try:
        cmd_list(None)
        assert "custom_committee" in capsys.readouterr().out
        assert len(default_suite()) == len(_scenarios(None)) == 30
        assert _scenarios(["custom_committee"], [("required_k", 3)])[0].k == 3
    finally:
        del SCENARIOS["custom_committee"]


def test_explicit_okapi_request_requires_installed_method(monkeypatch, tmp_path):
    from fusionbench.methods.registry import FACTORIES

    monkeypatch.setattr(FACTORIES["okapi_native_pattern_l2"], "available", classmethod(lambda cls: False))
    with pytest.raises(SystemExit, match="okapi_native_pattern_l2"):
        main(["run", "--methods", "okapi_native_pattern_l2", "--out", str(tmp_path)])
    assert not list(tmp_path.iterdir())
