"""``mbt.monte_carlo``: the engine's Monte Carlo, as a public function.

The engine has resampled finished backtests natively for a long time (return,
block and trade bootstraps, the return and block ones on the GPU too), but
Python only reached it through ``py_run_monte_carlo``, a raw binding that takes
a JSON string and was taken out of ``__all__`` for that reason. These tests pin
the public wrapper: it returns exactly what the binding returns, it reads the
run's own capital, it refuses what it cannot do by name, and it says when a
``device="cuda"`` request ran on the CPU instead of letting the engine fall
back without a word.
"""
import json
import warnings

import pytest

import manifoldbt as bt
from manifoldbt.exceptions import LicenseError
from manifoldbt.expr import col, lit, when
from manifoldbt.helpers import Interval

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

N = 400


def _frame(seed):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=N, freq="1h", tz="UTC")
    close = np.maximum(100.0 + np.cumsum(rng.normal(0.0, 0.5, N)), 1.0)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": [10.0] * N,
    })


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mc")
    store = bt.import_dataframe(
        _frame(3), symbol="AAA", symbol_id=1, interval="1h",
        data_root=str(tmp / "data"), metadata_db=str(tmp / "meta.sqlite"),
    )
    config = bt.BacktestConfig(
        universe=["AAA"],
        time_range_start=0,
        time_range_end=4_102_444_800_000_000_000,
        bar_interval=Interval.hours(1),
        initial_capital=25_000.0,
        warmup_bars=0,
        fees=bt.FeeConfig.zero(),
    )
    # In and out of the market, so the trade bootstrap has round trips to shuffle.
    pos = when(col("close") > lit(100.0), lit(1.0), lit(0.0))
    strategy = bt.Strategy.create("mc").signal("position", pos).size(pos)
    return bt.run(strategy, config, store)


def _raw(result, **config):
    return bt.py_run_monte_carlo(result.raw, json.dumps(config))


def test_is_public_and_the_binding_is_not():
    assert "monte_carlo" in bt.__all__
    assert "py_run_monte_carlo" not in bt.__all__


def test_returns_what_the_binding_returns(result):
    mc = bt.monte_carlo(result, n_paths=200, method="block", block_size=10, seed=7)
    raw = _raw(
        result,
        n_paths=200,
        method={"type": "block_bootstrap", "block_size": 10},
        rng_seed=7,
        initial_capital=25_000.0,
        store_paths=False,
        device="cpu",
    )
    assert mc == raw
    assert mc["n_paths"] == 200
    for key in ("final_return", "max_drawdown"):
        assert {"percentiles", "cvar", "mean", "std", "min", "max"} <= set(mc[key])
    assert 0.0 <= mc["prob_of_ruin"] <= 1.0


def test_one_seed_gives_the_same_paths(result):
    a = bt.monte_carlo(result, n_paths=100, seed=11)
    b = bt.monte_carlo(result, n_paths=100, seed=11)
    c = bt.monte_carlo(result, n_paths=100, seed=12)
    assert a == b
    assert a["final_return"] != c["final_return"]


@pytest.mark.parametrize("short, full", [("return", "return_bootstrap"),
                                         ("block", "block_bootstrap"),
                                         ("trade", "trade_bootstrap")])
def test_short_and_full_method_names_agree(result, short, full):
    assert bt.monte_carlo(result, n_paths=50, method=short, seed=1) == bt.monte_carlo(
        result, n_paths=50, method=full, seed=1
    )


def test_capital_defaults_to_the_runs_own(result):
    # The return and block bootstraps resample returns, where capital is only a
    # scale; the trade bootstrap rebuilds the curve from P&L in money, so it is
    # the one that tells which capital was used.
    mc = lambda **kw: bt.monte_carlo(result, n_paths=50, seed=3, method="trade", **kw)  # noqa: E731
    assert mc() == mc(initial_capital=25_000.0)
    assert mc() != mc(initial_capital=50_000.0)


def test_accepts_the_raw_result(result):
    assert bt.monte_carlo(result.raw, n_paths=50, seed=5) == bt.monte_carlo(
        result, n_paths=50, seed=5
    )


def test_stored_paths_come_back_as_a_matrix(result):
    mc = bt.monte_carlo(result, n_paths=30, method="return", seed=2, store_paths=True)
    paths = mc["paths"]
    assert isinstance(paths, np.ndarray)
    assert paths.shape == (30, mc["paths_n_steps"])
    np.testing.assert_allclose(paths[:, 0], 25_000.0)


def test_risk_aggregates_are_passed_through(result):
    mc = bt.monte_carlo(result, n_paths=100, seed=4, cvar_levels=[0.9],
                        dd_thresholds=[0.01, 0.5], confidence_levels=[0.5])
    assert [lvl for lvl, _ in mc["final_return"]["cvar"]] == [0.9]
    assert [lvl for lvl, _ in mc["dd_exceedance"]] == [0.01, 0.5]
    assert [lvl for lvl, _ in mc["max_drawdown"]["percentiles"]] == [0.5]


@pytest.mark.parametrize("kw, needle", [
    ({"method": "shuffle"}, "unknown method"),
    ({"device": "tpu"}, "unknown device"),
    ({"n_paths": 0}, "n_paths"),
    ({"n_paths": 10.5}, "n_paths"),
    ({"method": "block", "block_size": 0}, "block_size"),
])
def test_refuses_what_it_cannot_do_by_name(result, kw, needle):
    with pytest.raises(ValueError, match=needle):
        bt.monte_carlo(result, **kw)


def test_rejects_something_that_is_not_a_backtest():
    with pytest.raises(TypeError, match="backtest result"):
        bt.monte_carlo(object())


def test_above_1000_paths_asks_the_licence_flag(result, monkeypatch):
    asked = []

    def grant(feature):
        asked.append(feature)
        return False

    monkeypatch.setattr(bt, "_grant_couvre", grant)
    monkeypatch.setattr(bt, "py_run_monte_carlo", lambda *_: pytest.fail("reached the engine"))
    with pytest.raises(LicenseError, match="runs up to 1,000"):
        bt.monte_carlo(result, n_paths=1001)
    assert asked == ["monte_carlo_unlimited"]


def test_1000_paths_never_consults_the_licence(result, monkeypatch):
    monkeypatch.setattr(bt, "_grant_couvre", lambda _f: pytest.fail("licence consulted"))
    assert bt.monte_carlo(result, n_paths=1000, seed=1)["n_paths"] == 1000


def test_cuda_without_the_gpu_grant_is_a_licence_error(result, monkeypatch):
    monkeypatch.setattr(bt, "_grant_couvre", lambda _f: False)
    monkeypatch.setattr(bt, "py_run_monte_carlo", lambda *_: pytest.fail("reached the engine"))
    with pytest.raises(LicenseError, match="Researcher"):
        bt.monte_carlo(result, n_paths=10, device="cuda")


@pytest.mark.parametrize("kw, why", [
    ({"method": "trade"}, "trade bootstrap"),
    ({"method": "block", "store_paths": True}, "store_paths"),
])
def test_cuda_on_a_cpu_only_case_says_so(result, monkeypatch, kw, why):
    sent = []
    monkeypatch.setattr(bt, "_grant_couvre", lambda _f: True)
    monkeypatch.setattr(
        bt, "py_run_monte_carlo",
        lambda _raw, cfg: sent.append(json.loads(cfg)) or {"paths": None},
    )
    with pytest.warns(UserWarning, match=why):
        bt.monte_carlo(result, n_paths=10, device="cuda", **kw)
    assert sent[0]["device"] == "cuda"


def test_cuda_on_a_gpu_case_does_not_warn(result, monkeypatch):
    monkeypatch.setattr(bt, "_grant_couvre", lambda _f: True)
    monkeypatch.setattr(bt, "py_run_monte_carlo", lambda _raw, _cfg: {"paths": None})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bt.monte_carlo(result, n_paths=10, device="cuda", method="block")
