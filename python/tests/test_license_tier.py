"""license_info() names the tier actually granted.

A Researcher licence used to report "Pro", so the one call a user makes to
check their tier could not tell the two apart. These hold on any machine,
whatever licence it carries: the tier is read against the gates themselves.
"""
import manifoldbt as bt
from manifoldbt import _grant_couvre


def test_the_tier_is_one_of_the_three():
    tier, email = bt.license_info()
    assert tier in ("Community", "Pro", "Researcher")
    if tier == "Community":
        assert email is None


def test_an_open_tick_layer_reports_researcher():
    # The tick layer opens on the Researcher grant alone: no Pro licence and
    # no a-la-carte grant carries it. When it is open, the tier has to say so.
    if _grant_couvre("ticks"):
        assert bt.license_info()[0] == "Researcher"
    else:
        assert bt.license_info()[0] in ("Community", "Pro")


def test_the_banner_names_the_tier():
    # The banner swallows every exception and falls back to "Community": a
    # name it cannot resolve at import used to print Community on a paid
    # licence. It runs once, at import, so it is read from a fresh process.
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import manifoldbt"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert bt.license_info()[0] in out, out


def test_pro_gates_open_for_every_paid_tier():
    # Researcher carries all of Pro: a Pro gate that read the tier string
    # would have shut on the higher tier.
    assert bt._is_pro() == (bt.license_info()[0] in ("Pro", "Researcher"))
