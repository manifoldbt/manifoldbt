"""The tick layer is Pro+, and Pro is not enough.

Separate from test_ticks.py on purpose: that module skips itself entirely when
the layer is locked, and THIS is the test that must run precisely then - under
a plain Pro licence, every entry point of `manifoldbt.ticks` refuses,
uniformly, before touching any file.
"""
import pytest

import manifoldbt as bt

pytestmark = pytest.mark.skipif(
    bt.license_info()[0] != "Pro",
    reason="needs a plain Pro licence to show that Pro is not enough",
)


def _locked():
    try:
        bt.ticks.tape_info(__file__)
    except PermissionError:
        return True
    except Exception:
        return False
    return False


@pytest.mark.skipif(not _locked(), reason="a Pro+ grant is present; nothing to lock")
def test_every_entry_point_refuses_without_pro_plus(tmp_path):
    calls = [
        lambda: bt.ticks.generate_tape(str(tmp_path / "t.csv"), n_ticks=10),
        lambda: bt.ticks.tape_info("nope.csv"),
        lambda: bt.ticks.tape_to_bars("nope.csv", 60),
        lambda: bt.ticks.run_orderflow("nope.csv"),
        lambda: bt.ticks.sweep_orderflow_thr("nope.csv", [0.1]),
        lambda: bt.ticks.run_market_maker("nope.csv"),
        lambda: bt.ticks.simulate_bracket("nope.csv", "long", 1.0, 0.9, 1.1),
        lambda: bt.ticks.simulate_brackets(
            "nope.csv", sides=["long"], entries=[1.0], sls=[0.9], tps=[1.1]),
        lambda: bt.ticks.run_strategy("nope.csv", lambda *a: 0.0),
    ]
    for call in calls:
        with pytest.raises(PermissionError, match="Pro\+"):
            call()
    # The refusal happens before any I/O: no file was created by generate_tape.
    assert not (tmp_path / "t.csv").exists()
