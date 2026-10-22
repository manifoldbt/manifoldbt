"""A run whose curves come back daily says so when it runs.

Community output is capped to one row per day. The notice used to be printed
only at interpreter exit, which a Jupyter kernel never reaches, and the default
``output_resolution=None`` on sub-daily bars was capped by the engine without
any notice at all. Both cases now warn at run time, once per process; the
metrics, computed on every bar, are not what changes.

The tier is forced rather than read, so the suite asserts the Community
behaviour on a Pro machine too.
"""
import os
import warnings

import pytest

import manifoldbt as bt
from manifoldbt.helpers import Interval


@pytest.fixture
def community(monkeypatch):
    monkeypatch.setattr(bt, "_is_pro", lambda: False)
    monkeypatch.setattr(bt, "_daily_output_noticed", False)
    monkeypatch.setattr(bt, "_pro_warnings", [])


def _config(**kw):
    return bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=86_400_000_000_000, **kw
    )


def _notices(record):
    return [w for w in record if issubclass(w.category, UserWarning)
            and "one row per day" in str(w.message)]


def test_an_explicit_sub_daily_output_is_capped_and_announced(community):
    cfg = _config(bar_interval=Interval.minutes(1), output_resolution=Interval.hours(1))
    with pytest.warns(UserWarning, match="one row per day"):
        capped = bt._cap_output_resolution(cfg)
    assert capped.output_resolution is None
    assert cfg.output_resolution == Interval.hours(1)  # the caller's config is untouched


@pytest.mark.parametrize("kw", [
    {"bar_interval": Interval.hours(1)},
    {"bar_interval": None},  # the default bar is one minute
    {"bar_interval": Interval.days(1), "resample_to": Interval.hours(4)},
])
def test_the_default_output_on_sub_daily_bars_is_announced(community, kw):
    cfg = _config(**kw)
    with pytest.warns(UserWarning, match="one row per day"):
        out = bt._cap_output_resolution(cfg)
    assert out.output_resolution is None


@pytest.mark.parametrize("kw", [
    {"bar_interval": Interval.days(1)},
    {"bar_interval": Interval.days(1), "output_resolution": Interval.days(1)},
    # Per-event output is a shape the engine answers by name, not a resolution.
    {"bar_interval": Interval.trades()},
    {"bar_interval": Interval.trades(), "output_resolution": Interval.trades()},
])
def test_nothing_is_announced_when_nothing_is_capped(community, kw):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        bt._cap_output_resolution(_config(**kw))
    assert _notices(record) == []


def test_pro_is_not_warned(monkeypatch):
    monkeypatch.setattr(bt, "_is_pro", lambda: True)
    monkeypatch.setattr(bt, "_daily_output_noticed", False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        bt._cap_output_resolution(_config(bar_interval=Interval.hours(1)))
        bt._cap_output_resolution(
            _config(bar_interval=Interval.minutes(1), output_resolution=Interval.minutes(5)))
    assert _notices(record) == []


def test_the_notice_comes_once_per_process(community):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            bt._cap_output_resolution(_config(bar_interval=Interval.hours(1)))
        bt._cap_output_resolution(
            _config(bar_interval=Interval.minutes(1), output_resolution=Interval.minutes(5)))
    assert len(_notices(record)) == 1
    # The exit summary still lists it, once.
    assert bt._pro_warnings == ["output_resolution capped to daily"]


def test_bt_run_points_the_notice_at_the_caller(community, tmp_path):
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    rows = 3 * 1440
    start = pd.Timestamp("2021-03-01", tz="UTC")
    close = 100.0 + np.cumsum(np.random.default_rng(3).normal(0.0, 0.05, rows))
    frame = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=rows, freq="1min"),
        "open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
        "volume": np.full(rows, 1_000.0),
    })
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        frame, symbol="DLY", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=int(start.value),
        time_range_end=int(start.value) + rows * 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000.0,
    )
    strategy = bt.Strategy.create("hold").signal("on", bt.lit(1.0)).size(bt.lit(1.0))
    with pytest.warns(UserWarning, match="one row per day") as record:
        bt.run(strategy, config, store)
    notice = _notices(record)[0]
    assert os.path.normcase(notice.filename) == os.path.normcase(__file__)
