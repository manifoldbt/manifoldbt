"""``BacktestConfig`` refuse un reglage qu'il n'a pas.

Une dataclass ordinaire accepte n'importe quel attribut : ``cfg.tick_size =
0.1`` (un reglage d'``ExecutionConfig``) etait stocke et jamais lu.
"""
import pytest

import manifoldbt as bt


# ---------------------------------------------------------------------------
# BacktestConfig refuse un reglage qu'il n'a pas
# ---------------------------------------------------------------------------


def test_tick_size_sur_backtestconfig_est_refuse_et_renvoie_a_execution():
    cfg = bt.BacktestConfig()
    with pytest.raises(AttributeError) as err:
        cfg.tick_size = 0.1
    msg = str(err.value)
    assert "'tick_size'" in msg and "no effect" in msg
    assert "cfg.execution.tick_size" in msg
    assert not hasattr(cfg, "tick_size")


def test_un_reglage_des_frais_renvoie_a_fees():
    cfg = bt.BacktestConfig()
    with pytest.raises(AttributeError, match=r"cfg\.fees\.taker_fee_bps"):
        cfg.taker_fee_bps = 4.0


def test_une_faute_de_frappe_propose_le_bon_nom():
    cfg = bt.BacktestConfig()
    with pytest.raises(AttributeError, match="warmup_bars"):
        cfg.warmup_bar = 10


def test_les_vrais_reglages_restent_modifiables():
    cfg = bt.BacktestConfig()
    cfg.warmup_bars = 5
    cfg.initial_capital = 2.0
    cfg.execution.tick_size = 0.1
    cfg._prive = 1  # un nom prive reste libre
    assert cfg.warmup_bars == 5 and cfg.initial_capital == 2.0
    assert cfg.to_json_dict()["execution"]["tick_size"] == 0.1


def test_copie_et_replace_marchent_toujours():
    import copy
    import dataclasses
    import pickle

    cfg = bt.BacktestConfig(warmup_bars=3)
    assert copy.deepcopy(cfg).warmup_bars == 3
    assert copy.copy(cfg).warmup_bars == 3
    assert dataclasses.replace(cfg, warmup_bars=4).warmup_bars == 4
    assert pickle.loads(pickle.dumps(cfg)).warmup_bars == 3
