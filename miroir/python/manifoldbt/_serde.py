"""Internal helpers to produce JSON matching Rust serde externally-tagged enums."""
from __future__ import annotations

import math
from typing import Any


def scalar_value_to_json(value: Any) -> Any:
    """Convert a Python value to a Rust ScalarValue JSON representation.

    Rust serde format (externally tagged):
      None  -> "Null"
      bool  -> {"Bool": v}
      int   -> {"Int64": v}
      float -> {"Float64": v}
      str   -> {"Utf8": v}
    """
    if value is None:
        return "Null"
    if isinstance(value, bool):
        return {"Bool": value}
    if isinstance(value, int):
        return {"Int64": value}
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return {"Float64": value}
    if isinstance(value, str):
        return {"Utf8": value}
    raise TypeError(f"Cannot convert {type(value).__name__} to ScalarValue")


def scalar_value_from_json(value: Any) -> Any:
    """Inverse of :func:`scalar_value_to_json`.

    The engine reports the parameters a run actually executed with (the
    ``parameters`` map of a result's ``manifest``) in this externally-tagged
    form. Anything that is not a ScalarValue encoding is returned unchanged.
    """
    if value == "Null":
        return None
    if value == "NaN":
        return math.nan
    if isinstance(value, dict) and len(value) == 1:
        (tag, inner), = value.items()
        if tag in ("Bool", "Int64", "Float64", "Utf8"):
            return inner
    return value
