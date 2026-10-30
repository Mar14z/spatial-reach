from touch.grasp.grippable import grippable


def test_workspace_yes():
    r = grippable(0.12, 0.0, 10.0)
    assert r["verdict"] == "yes"


def test_out_of_workspace():
    assert grippable(0.20, 0.0, 5.0)["reason"] == "out_of_workspace"
    assert grippable(0.12, 0.6, 5.0)["reason"] == "az_out_of_range"


def test_no_error_estimate_is_maybe():
    r = grippable(0.12, 0.0, None)
    assert r["verdict"] == "maybe"
    assert r["reason"] == "no_error_estimate"


def test_error_bands():
    assert grippable(0.12, 0.0, 15.0)["verdict"] == "yes"
    assert grippable(0.12, 0.0, 15.1)["verdict"] == "maybe"
    assert grippable(0.12, 0.0, 20.0)["verdict"] == "maybe"
    assert grippable(0.12, 0.0, 20.1)["verdict"] == "no"
    assert grippable(0.12, 0.0, 20.1)["reason"] == "err_over_cliff"
