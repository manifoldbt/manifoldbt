"""Stochastic simulation via SDE expression DSL.

Define stochastic differential equations as string expressions and simulate
price paths at native Rust speed with Rayon parallelism.

Example
-------
>>> import manifoldbt as mbt
>>> model = mbt.StochasticModel(
...     drift="mu",
...     diffusion="sqrt(h)",
...     jump_intensity="lambda",
...     jump_size="normal(mu_j, sigma_j)",
...     state_vars={"h": 1e-4},
...     state_update={"h": "omega + alpha * (ret - mu) ** 2 + beta * h"},
...     params={"mu": 0.08, "omega": 1e-6, "alpha": 0.1, "beta": 0.85,
...             "lambda": 5.0, "mu_j": -0.02, "sigma_j": 0.04},
... )
>>> result = mbt.run_stochastic(model, s0=100, n_paths=10000, n_steps=252, dt=1/252)

Presets
-------
>>> result = mbt.run_stochastic("gbm", s0=100, n_paths=10000, n_steps=252, dt=1/252,
...                              params={"mu": 0.05, "sigma": 0.2})
"""

from typing import Any, Dict, List, Optional


class StochasticModel:
    """Define a stochastic differential equation model via string expressions.

    The SDE has the form::

        dS = drift(S,t,state) * S * dt
           + diffusion(S,t,state) * S * dW
           + jump_size * dN(jump_intensity)

    where state variables (like GARCH variance ``h``) are updated after each
    price step according to ``state_update`` expressions.

    Parameters
    ----------
    drift : str
        Drift expression (μ). E.g. ``"mu"`` or ``"mu - 0.5 * h"``.
    diffusion : str
        Diffusion expression (σ). E.g. ``"sigma"`` or ``"sqrt(h)"``.
    jump_intensity : str, optional
        Jump intensity expression (λ). E.g. ``"lambda"``.
    jump_size : str, optional
        Jump size expression. E.g. ``"normal(mu_j, sigma_j)"``.
    state_vars : dict, optional
        State variable names and initial values. ``S`` is implicit (index 0).
    state_update : dict, optional
        Update expressions for state variables.
        E.g. ``{"h": "omega + alpha * (ret - mu) ** 2 + beta * h"}``.
    params : dict, optional
        Model parameter names and values.
    name : str
        Model name (default ``"custom"``).

    Available identifiers in expressions
    -------------------------------------
    - Any key in ``params`` → parameter value
    - Any key in ``state_vars`` → current state value
    - ``S`` → current price
    - ``ret`` → log-return from previous step
    - ``dt`` → time step size
    - ``t`` → current simulation time
    - ``step`` → current step index

    Available functions
    -------------------
    ``sqrt``, ``abs``, ``log``, ``exp``, ``floor``, ``max``, ``min``, ``pow``,
    ``normal(mu, sigma)``, ``uniform(lo, hi)``, ``randn()``,
    ``if(cond, then, else)``

    Operators: ``+``, ``-``, ``*``, ``/``, ``**``, ``>``, ``<``, ``>=``,
    ``<=``, ``==``, ``&&``, ``||``, ``!``, ternary ``cond ? a : b``
    """

    def __init__(
        self,
        *,
        drift: str,
        diffusion: str,
        jump_intensity: Optional[str] = None,
        jump_size: Optional[str] = None,
        state_vars: Optional[Dict[str, float]] = None,
        state_update: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, float]] = None,
        name: str = "custom",
    ):
        self.name = name
        self.drift = drift
        self.diffusion = diffusion
        self.jump_intensity = jump_intensity
        self.jump_size = jump_size
        self.state_vars = state_vars or {}
        self.state_update = state_update or {}
        self.params = params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dict suitable for JSON encoding."""
        d: Dict[str, Any] = {
            "name": self.name,
            "drift": self.drift,
            "diffusion": self.diffusion,
            "params": dict(self.params),
        }
        if self.jump_intensity is not None:
            d["jump_intensity"] = self.jump_intensity
        if self.jump_size is not None:
            d["jump_size"] = self.jump_size
        if self.state_vars:
            d["state_vars"] = dict(self.state_vars)
        if self.state_update:
            d["state_update"] = dict(self.state_update)
        return d

    def __repr__(self) -> str:
        parts = [f"StochasticModel(name={self.name!r}"]
        parts.append(f"drift={self.drift!r}")
        parts.append(f"diffusion={self.diffusion!r}")
        if self.jump_intensity:
            parts.append(f"jump_intensity={self.jump_intensity!r}")
        if self.state_vars:
            parts.append(f"state_vars={self.state_vars!r}")
        return ", ".join(parts) + ")"


# Preset shortcuts
GBM = StochasticModel(
    name="gbm",
    drift="mu",
    diffusion="sigma",
    params={"mu": 0.05, "sigma": 0.2},
)

HESTON = StochasticModel(
    name="heston",
    drift="mu",
    diffusion="sqrt(v)",
    state_vars={"v": 0.04},
    state_update={
        "v": "max(v + kappa * (theta - v) * dt + xi * sqrt(v) * sqrt(dt) * randn(), 0.0)"
    },
    params={"mu": 0.05, "kappa": 2.0, "theta": 0.04, "xi": 0.3},
)

MERTON = StochasticModel(
    name="merton",
    drift="mu",
    diffusion="sigma",
    jump_intensity="lambda",
    jump_size="normal(mu_j, sigma_j)",
    params={"mu": 0.05, "sigma": 0.2, "lambda": 1.0, "mu_j": -0.05, "sigma_j": 0.08},
)

GARCH_JD = StochasticModel(
    name="garch_jd",
    drift="mu",
    diffusion="sqrt(h)",
    jump_intensity="lambda",
    jump_size="normal(mu_j, sigma_j)",
    state_vars={"h": 1e-4},
    state_update={"h": "omega + alpha * (ret - mu) ** 2 + beta * h"},
    params={
        "mu": 0.08,
        "omega": 1e-6,
        "alpha": 0.1,
        "beta": 0.85,
        "lambda": 5.0,
        "mu_j": -0.02,
        "sigma_j": 0.04,
    },
)
