"""What a multi-output indicator does when it is handed over whole.

`macd`, `bollinger_bands`, `keltner_channels`, `donchian_channels` and `vortex`
each return several series. Passed as one -- as a signal, in a comparison, as
position sizing -- they used to fail inside the engine with
``'tuple' object has no attribute '_param_meta'`` or a bare
``'>' not supported between instances of 'tuple' and 'int'``. Neither names the
call that produced the tuple, so neither tells the caller what to do.

These tests pin the two halves of the fix: the tuple still unpacks and indexes
exactly as before, and every misuse of it names the call, its members and the
unpacking.
"""
import json

import pytest

import manifoldbt as bt
from manifoldbt.expr import MultiExpr
from manifoldbt.indicators import (
    bollinger_bands,
    close,
    donchian_channels,
    keltner_channels,
    macd,
    vortex,
)

# Every call that returns several series, with the members it returns in order.
# The names are the ones the reference documentation uses.
MULTI = [
    (lambda: macd(close), "macd", ("macd_line", "signal_line", "histogram")),
    (lambda: bollinger_bands(close), "bollinger_bands", ("upper", "middle", "lower")),
    (lambda: keltner_channels(20), "keltner_channels", ("upper", "middle", "lower")),
    (lambda: donchian_channels(20), "donchian_channels", ("upper", "middle", "lower")),
    (lambda: vortex(14), "vortex", ("vi_plus", "vi_minus")),
]

IDS = [name for _, name, _ in MULTI]


@pytest.mark.parametrize("build,name,members", MULTI, ids=IDS)
def test_still_unpacks_and_indexes_like_a_tuple(build, name, members):
    """The tuple contract is unchanged: this is what callers already write."""
    value = build()
    assert isinstance(value, tuple)
    assert len(value) == len(members)

    unpacked = list(value)
    assert [v.to_json() for v in unpacked] == [value[i].to_json() for i in range(len(value))]


def test_serialization_of_a_member_is_untouched():
    """Picking a member yields the same node it always did."""
    macd_line, signal_line, histogram = macd(close, 12, 26, 9)
    assert macd_line.to_json() == close.macd_line(12, 26).to_json()
    assert signal_line.to_json() == close.macd_signal(12, 26, 9).to_json()
    assert histogram.to_json() == close.macd_hist(12, 26, 9).to_json()


@pytest.mark.parametrize("build,name,members", MULTI, ids=IDS)
def test_a_whole_tuple_as_a_signal_names_the_call(build, name, members):
    with pytest.raises(TypeError) as err:
        bt.Strategy(name="s", signals={"x": build()}).to_json()

    message = str(err.value)
    assert f"{name}() returns {len(members)} series" in message
    assert ", ".join(members) in message
    assert "signal 'x'" in message


@pytest.mark.parametrize("build,name,members", MULTI, ids=IDS)
def test_comparing_a_whole_tuple_names_the_call(build, name, members):
    with pytest.raises(TypeError) as err:
        bt.when(build() > 0, bt.lit(1.0), bt.lit(0.0))

    message = str(err.value)
    assert f"{name}() returns {len(members)} series" in message
    assert "`>` needs a single series on each side" in message


def test_position_sizing_names_the_call():
    with pytest.raises(TypeError, match=r"bollinger_bands\(\) returns 3 series"):
        bt.Strategy(name="s", signals={"a": close}, position_sizing=bollinger_bands(close))


def test_a_whole_tuple_as_a_when_condition_names_the_call():
    """The condition is checked like the branches; it used to sail through to JSON."""
    with pytest.raises(TypeError, match=r"vortex\(\) returns 2 series"):
        bt.when(vortex(14), bt.lit(1.0), bt.lit(0.0))


def test_a_method_of_one_series_names_the_call():
    with pytest.raises(TypeError, match=r"`\.rolling_mean` belongs to one series"):
        macd(close).rolling_mean(5)


def test_arithmetic_on_a_whole_tuple_is_refused_rather_than_concatenating():
    """`+` on a plain tuple concatenates and `*` repeats -- silently, and wrongly."""
    with pytest.raises(TypeError, match=r"`\+` needs a single series"):
        macd(close) + bollinger_bands(close)
    with pytest.raises(TypeError, match=r"`\*` needs a single series"):
        macd(close) * 2


def test_equality_against_a_number_is_refused_but_tuple_equality_is_not():
    """Only the comparison that cannot mean anything is taken away."""
    with pytest.raises(TypeError, match=r"`==` needs a single series"):
        macd(close) == 0

    bands = bollinger_bands(close)
    assert bands != ("upper",)


@pytest.mark.parametrize("build,name,members", MULTI, ids=IDS)
def test_no_engine_private_name_reaches_the_caller(build, name, members):
    """The reported symptom: an AttributeError on `_param_meta`, a name the
    caller never wrote and cannot act on."""
    with pytest.raises(TypeError) as err:
        bt.Strategy(name="s", signals={"x": build()}).to_json()
    assert "_param_meta" not in str(err.value)

    with pytest.raises(TypeError) as err:
        bt.Strategy(name="s", signals={"a": close}, position_sizing=build()).to_json()
    assert "_param_meta" not in str(err.value)


def test_a_signal_written_as_text_says_expressions_are_built():
    """Expressions are not parsed from strings; say so instead of failing on
    `'str' object has no attribute '_param_meta'`."""
    with pytest.raises(TypeError) as err:
        bt.Strategy(name="s", signals={"m": "macd(close)"}).to_json()

    message = str(err.value)
    assert "'macd(close)'" in message
    assert "not parsed from text" in message
    assert "_param_meta" not in message


def test_a_tuple_dropped_into_signals_after_construction_is_still_caught():
    """`signals` is a public dict; the check cannot live only in __init__."""
    strategy = bt.Strategy(name="s", signals={"a": close})
    strategy.signals["m"] = macd(close)
    with pytest.raises(TypeError, match=r"macd\(\) returns 3 series"):
        strategy.to_json()


def test_repr_says_what_the_call_returns():
    assert repr(macd(close)) == "macd(...) -> (macd_line, signal_line, histogram)"


def test_a_plain_tuple_of_expressions_is_refused_too():
    """Nothing hangs on MultiExpr: any tuple of expressions gets an answer."""
    with pytest.raises(TypeError, match="Unpack it and pass the one you meant"):
        bt.Strategy(name="s", signals={"x": (close, close)}).to_json()


def test_multiexpr_survives_a_round_trip_through_a_working_strategy():
    """The whole point is that correct code is untouched."""
    upper, middle, lower = bollinger_bands(close, 20, 2.0)
    strategy = (
        bt.Strategy.create("bands")
        .signal("upper", upper)
        .signal("lower", lower)
        .size(bt.when(close > bt.col("upper"), bt.lit(1.0), bt.lit(0.0)))
    )
    parsed = json.loads(strategy.to_json())
    assert set(parsed["signals"]) == {"upper", "lower"}
    assert isinstance(MultiExpr((upper, middle, lower)), tuple)
