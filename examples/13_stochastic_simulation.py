"""Stochastic Simulation -- synthetic price paths via SDE expression DSL.

Demonstrates:
  - Built-in presets (GBM, Heston, Merton, GARCH-JD)
  - Custom SDE model via string expressions
  - Stochastic fan chart visualization
  - CUDA GPU acceleration (device="cuda")
  - All expressions compile to native Rust — full Rayon / CUDA parallelism

Usage:
    python examples/13_stochastic_simulation.py
"""
import time
import manifoldbt as mbt

N = 10_000_000  # 10M paths
DEVICE = "cuda"  # "cpu" or "cuda"

if __name__ == "__main__":
    # ── 1. Geometric Brownian Motion (preset) ───────────────────────────────
    print(f"1. GBM preset ({N:,} paths, 252 steps) [{DEVICE}]")
    t0 = time.perf_counter()
    result = mbt.run_stochastic(
        "gbm",
        s0=100.0,
        n_paths=N,
        n_steps=252,
        dt=1 / 252,
        params={"mu": 0.05, "sigma": 0.20},
        seed=42,
        device=DEVICE,
    )
    elapsed = time.perf_counter() - t0
    print(f"   Mean final price: {result['final_price']['mean']:.2f}")
    print(f"   Median max DD:    {result['max_drawdown']['percentiles'][3][1]:.2%}")
    print(f"   Elapsed: {elapsed:.3f}s\n")

    # ── 2. Heston stochastic volatility (preset) ────────────────────────────
    print(f"2. Heston preset ({N:,} paths) [{DEVICE}]")
    t0 = time.perf_counter()
    result = mbt.run_stochastic(
        "heston",
        s0=100.0,
        n_paths=N,
        n_steps=252,
        dt=1 / 252,
        params={"mu": 0.05, "kappa": 2.0, "theta": 0.04, "xi": 0.3},
        seed=42,
        device=DEVICE,
    )
    elapsed = time.perf_counter() - t0
    print(f"   Mean final price: {result['final_price']['mean']:.2f}")
    print(f"   Ann. vol (mean):  {result['annualized_vol']['mean']:.2%}")
    print(f"   Elapsed: {elapsed:.3f}s\n")

    # ── 3. Merton Jump Diffusion (preset) ───────────────────────────────────
    print(f"3. Merton Jump Diffusion ({N:,} paths) [{DEVICE}]")
    t0 = time.perf_counter()
    result = mbt.run_stochastic(
        "merton",
        s0=100.0,
        n_paths=N,
        n_steps=252,
        dt=1 / 252,
        params={
            "mu": 0.05,
            "sigma": 0.20,
            "lambda": 1.0,     # 1 jump/year on average
            "mu_j": -0.05,     # mean jump = -5%
            "sigma_j": 0.08,   # jump vol = 8%
        },
        seed=42,
        device=DEVICE,
    )
    elapsed = time.perf_counter() - t0
    print(f"   Mean final price: {result['final_price']['mean']:.2f}")
    print(f"   Elapsed: {elapsed:.3f}s\n")

    # ── 4. Custom GARCH(1,1) Jump Diffusion ─────────────────────────────────
    print(f"4. Custom GARCH(1,1) Jump Diffusion ({N:,} paths) [{DEVICE}]")
    model = mbt.StochasticModel(
        name="my_garch_jd",
        drift="mu",
        diffusion="sqrt(h)",
        jump_intensity="lambda",
        jump_size="normal(mu_j, sigma_j)",
        state_vars={"h": 1e-4},
        state_update={"h": "omega + alpha * (ret - mu) ** 2 + beta * h"},
        params={
            "mu": 0.08,
            "omega": 1e-6,
            "alpha": 0.10,
            "beta": 0.85,
            "lambda": 5.0,
            "mu_j": -0.02,
            "sigma_j": 0.04,
        },
    )
    t0 = time.perf_counter()
    result = mbt.run_stochastic(
        model,
        s0=100.0,
        n_paths=N,
        n_steps=252,
        dt=1 / 252,
        seed=42,
        device=DEVICE,
    )
    elapsed = time.perf_counter() - t0
    print(f"   Mean final price: {result['final_price']['mean']:.2f}")
    print(f"   Max DD (P5):      {result['max_drawdown']['percentiles'][0][1]:.2%}")
    print(f"   Elapsed: {elapsed:.3f}s\n")

    # ── 5. Custom mean-reverting model (CPU — store_paths needs RAM) ────────
    N_PLOT = 10_000
    print(f"5. Custom mean-reverting model ({N_PLOT:,} paths) [cpu, store_paths]")
    mean_rev = mbt.StochasticModel(
        name="mean_reverting",
        # Drift pulls price back toward 100
        drift="kappa * (log(100.0) - log(S))",
        diffusion="sigma",
        params={"kappa": 2.0, "sigma": 0.25},
    )
    t0 = time.perf_counter()
    result = mbt.run_stochastic(
        mean_rev,
        s0=80.0,   # start below mean
        n_paths=N_PLOT,
        n_steps=252,
        dt=1 / 252,
        seed=42,
        store_paths=True,
        device="cpu",
    )
    elapsed = time.perf_counter() - t0
    print(f"   Mean final price: {result['final_price']['mean']:.2f} (target: 100)")
    print(f"   Elapsed: {elapsed:.3f}s\n")

    # ── 6. Fan chart visualization ──────────────────────────────────────────
    print("6. Plotting fan chart...")
    mbt.plot.stochastic_paths(
        result,
        title=f"Mean-reverting model (S0=80, target=100, {N_PLOT:,} paths)",
    )
