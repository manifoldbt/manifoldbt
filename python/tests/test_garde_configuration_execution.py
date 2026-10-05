"""Configurations refused instead of producing a wrong number, or a panic.

Two guards, both on the Python side because that is the last place that still
knows what the user wrote.

**At ``signal_delay = 0``, a fill cannot land on a point of the bar it decided
on.** Decide on bar ``t``, execute on bar ``t`` is sound at the close: the fill
price *is* the information the decision was made on, and it is what
``signal_delay = 0`` exists for. It stops being sound as soon as the fill lands
somewhere else inside that same bar, because reaching that point would have
meant trading before the close that produced the signal. Measured on a
driftless random walk, "long when close > open" returns +1913% under
``AtOpen``, +361% under ``AtVwap``, ``MidPrice`` and ``custom("low")``, against
+5% under the defaults. The engine said nothing.

``custom(<a signal the strategy defines>)`` is exempt, and deliberately: there
the level is one the DSL computed, and the fill only books if the bar actually
traded through it. That is the band case ``test_exec_price_signal`` pins, and
it measures like the close on this bench. What such a level owes to its own bar
is the resting-order question, carried by the guard next door.

**``ExecutionConfig`` values reach ``f64::clamp`` unchecked.** The core clamps
the desired position to ``(-max_units, max_units)`` derived from
``max_position_pct``. Rust's ``f64::clamp`` asserts ``min <= max``, so a
negative percentage panics, and a non-finite one makes both bounds NaN and
panics too. A Rust panic crosses PyO3 as ``PanicException``, which inherits
from ``BaseException`` and walks straight through a user's
``except Exception``.

The resting-order guard next to these two already refused ``signal_delay = 0``
for limit entries; these tests also pin that it still does, and that the
default configuration keeps working.
"""
import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt.exceptions import ConfigError
from manifoldbt.expr import col, lit, when
from manifoldbt.helpers import ExecutionPrice, Slippage

N_BARS = 2000
CAPITAL = 10_000.0


def _bars(seed: int = 11) -> pd.DataFrame:
    """A driftless random walk whose open is the previous close."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0015, N_BARS)))
    open_ = np.empty(N_BARS)
    open_[0], open_[1:] = 100.0, close[:-1]
    ts = pd.date_range("2024-01-01", periods=N_BARS, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": open_,
        "high": np.maximum(open_, close) * 1.0002,
        "low": np.minimum(open_, close) * 0.9998,
        "close": close, "volume": 1000.0,
    })


_DF = _bars()
_END_NS = int(_DF["timestamp"].iloc[-1].value) + 86_400_000_000_000


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("garde_config")
    return bt.import_dataframe(
        _DF, symbol="TEST", symbol_id=1, interval="1m",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )


def _config(**execution) -> bt.BacktestConfig:
    return bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=_END_NS,
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(**execution),
        slippage=Slippage.none(),
    )


def _reads_its_own_bar() -> bt.Strategy:
    """Long when the current bar is up: it can only pay if the bar is known."""
    pos = when(col("close") > col("open"), lit(1.0), lit(0.0))
    return bt.Strategy.create("lit_sa_barre").signal("pos", pos).size(col("pos"))


# ---------------------------------------------------------------- the price --

@pytest.mark.parametrize("price", ["AtOpen", "AtVwap", "MidPrice", "NextBarOpen"])
def test_un_prix_intrabarre_sans_delai_est_refuse(store, price):
    with pytest.raises(ConfigError, match="needs signal_delay >= 1"):
        bt.run(_reads_its_own_bar(),
               _config(signal_delay=0, execution_price=price), store)


@pytest.mark.parametrize("colonne", ["low", "high", "open", "vwap"])
def test_une_colonne_de_barre_en_custom_sans_delai_est_refusee(store, colonne):
    """`custom("low")` at delay 0 hands the strategy its own bar's extreme."""
    with pytest.raises(ConfigError, match="needs signal_delay >= 1"):
        bt.run(_reads_its_own_bar(),
               _config(signal_delay=0,
                       execution_price=ExecutionPrice.custom(colonne)), store)


def test_un_niveau_calcule_par_la_strategie_reste_permis(store):
    """`custom(<a signal>)` is exempt: the level is one the DSL chose.

    The fill only books if the bar actually traded through it, which is the
    band case `test_exec_price_signal` pins. On this bench it measures like the
    close, not like `custom("low")`. What that level owes to its own bar is the
    resting-order question, carried by the guard next door.
    """
    pos = when(col("close") > col("open"), lit(1.0), lit(0.0))
    strategy = (bt.Strategy.create("bande")
                .signal("exec_level", col("close") * lit(1.0005))
                .signal("pos", pos).size(col("pos")))
    result = bt.run(strategy,
                    _config(signal_delay=0,
                            execution_price=ExecutionPrice.custom("exec_level")),
                    store)
    assert abs(result.metrics["total_return"]) < 0.5


def test_un_nom_inconnu_garde_le_message_du_moteur(store):
    """Neither a bar column nor a signal: the engine names it, not this guard."""
    with pytest.raises(Exception, match="neither a bar column nor a signal"):
        bt.run(_reads_its_own_bar(),
               _config(signal_delay=0,
                       execution_price=ExecutionPrice.custom("nexistepas")), store)


def test_le_message_nomme_le_prix_et_la_sortie(store):
    """A refusal has to say what to write instead, not just that it refused.

    `ConfigError` and not `ValueError`: `run()` funnels every `ValueError`
    through `_classify_error`, which buckets by keyword. This message contains
    "signal", so it would surface as a `StrategyError` when the execution
    config is what is wrong. Raising the right class crosses untouched.
    """
    with pytest.raises(ConfigError) as e:
        bt.run(_reads_its_own_bar(),
               _config(signal_delay=0, execution_price="AtOpen"), store)
    message = str(e.value)
    assert "AtOpen" in message
    assert "signal_delay=1" in message
    assert "AtClose" in message


@pytest.mark.parametrize("price", ["AtClose", "NextBarClose"])
def test_la_cloture_reste_permise_sans_delai(store, price):
    """These two fill at the close that produced the signal: nothing to refuse.

    This is the case the guard must not break: it is what `signal_delay = 0`
    is for, and the returns stay in the range a driftless walk allows.
    """
    result = bt.run(_reads_its_own_bar(),
                    _config(signal_delay=0, execution_price=price), store)
    assert abs(result.metrics["total_return"]) < 0.5


@pytest.mark.parametrize("price", ["AtOpen", "AtVwap", "MidPrice"])
def test_un_delai_dune_barre_rend_le_prix_licite(store, price):
    """With a delay the fill follows the signal, so every price is allowed."""
    result = bt.run(_reads_its_own_bar(),
                    _config(signal_delay=1, execution_price=price), store)
    assert abs(result.metrics["total_return"]) < 0.5


# ------------------------------------------------------------- the config ---

@pytest.mark.parametrize("valeur", [-1.0, 0.0, float("nan"), float("inf")])
def test_max_position_pct_hors_domaine_est_refuse(valeur):
    """Refused where the field still has a name, not in `f64::clamp`."""
    with pytest.raises(ValueError, match="max_position_pct"):
        bt.ExecutionConfig(max_position_pct=valeur)


@pytest.mark.parametrize("valeur", [-1, 1.5, True])
def test_signal_delay_hors_domaine_est_refuse(valeur):
    with pytest.raises(ValueError, match="signal_delay"):
        bt.ExecutionConfig(signal_delay=valeur)


def test_les_valeurs_licites_passent():
    assert bt.ExecutionConfig().max_position_pct == 1.0
    assert bt.ExecutionConfig(max_position_pct=0.5).max_position_pct == 0.5
    assert bt.ExecutionConfig(signal_delay=0).signal_delay == 0
    assert bt.ExecutionConfig(signal_delay=3).signal_delay == 3


def test_le_defaut_du_produit_tourne_toujours(store):
    """`AtClose` with `signal_delay = 0` is the shipped default, and is sound."""
    result = bt.run(_reads_its_own_bar(), _config(), store)
    assert abs(result.metrics["total_return"]) < 0.5


# ------------------------------------------ the guard that already existed --

def test_un_ordre_au_repos_sans_delai_reste_refuse(store):
    """The resting-order guard predates these two and must keep firing.

    It reads the orders off the strategy, which is where `entry_price()` puts
    them, and it raises a bare `ValueError` that `run()` then reclassifies by
    keyword: "signal_delay" matches "signal", so it arrives as a
    `StrategyError`. Pinned here as it stands, not as it should be; changing
    the class of an error users already catch is its own change.
    """
    pos = when(col("close") > col("open"), lit(1.0), lit(0.0))
    strategy = (bt.Strategy.create("au_repos").signal("pos", pos)
                .size(col("pos"))
                .limit_entry(offset_bps=5.0))
    with pytest.raises(bt.BacktesterError, match="signal_delay"):
        bt.run(strategy, _config(signal_delay=0), store)
