"""Doc <-> code signature contract.

These assertions encode the public signatures and helper outputs that the
online documentation and the interactive notebook rely on. They are cheap,
IO-free, and Pro-free, and exist to catch *doc drift*: if a documented kwarg,
preset, or helper shape changes in the code, a doc snippet silently breaks.

This guards, among others:
  * ``plot.monte_carlo`` exposing ``n_simulations`` (NOT ``n_paths``) -- the
    notebook bug where ``n_paths=`` raised TypeError.
  * ``Slippage.volume_impact`` emitting ``impact_coeff``/``exponent`` -- the
    notebook bug where ``{"coefficient": ...}`` failed Rust deserialization.
  * ``DataStore`` accepting ``mega``/``arrow_dir`` -- the doc signature that
    omitted them.
"""
import inspect

import manifoldbt as bt


def test_monte_carlo_uses_n_simulations_not_n_paths():
    params = inspect.signature(bt.plot.monte_carlo).parameters
    assert "n_simulations" in params
    assert "n_paths" not in params  # the notebook snippet bug


def test_slippage_helper_shapes_match_serde():
    # Keys must match the Rust SlippageConfig serde variants exactly.
    assert bt.Slippage.volume_impact(0.1) == {
        "VolumeImpact": {"impact_coeff": 0.1, "exponent": 1.5}
    }
    assert bt.Slippage.fixed_bps(2.0) == {"FixedBps": {"bps": 2.0}}


def test_interval_helper_shapes():
    assert bt.Interval.seconds(1) == {"Seconds": 1}
    assert bt.Interval.minutes(1) == {"Minutes": 1}
    assert bt.Interval.hours(12) == {"Hours": 12}
    assert bt.Interval.days(1) == {"Days": 1}


def test_fee_presets_match_documented_values():
    # Documented under #configuration > FeeConfig Presets.
    perps = bt.FeeConfig.binance_perps()
    assert (perps.maker_fee_bps, perps.taker_fee_bps) == (2.0, 5.0)
    spot = bt.FeeConfig.binance_spot()
    assert (spot.maker_fee_bps, spot.taker_fee_bps) == (10.0, 10.0)


def test_datastore_accepts_mega_and_arrow_dir_kwargs(tmp_path):
    # The real signature is (data_root, metadata_db, dataset, mega, arrow_dir).
    # We only assert the kwargs are *accepted* (no TypeError for unknown kwarg);
    # any runtime/IO error from opening an empty dir is fine for this contract.
    for kw in ("mega", "arrow_dir"):
        try:
            bt.DataStore(str(tmp_path), dataset="bars_1m", **{kw: str(tmp_path)})
        except TypeError as exc:  # unexpected keyword argument -> contract broken
            raise AssertionError(f"DataStore rejected kwarg {kw!r}: {exc}")
        except Exception:
            pass  # non-TypeError (e.g. cannot open store) -> kwarg was accepted
