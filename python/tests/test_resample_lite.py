"""`resample_to` sur un balayage allege: l'index du dernier element du groupe.

POURQUOI CE FICHIER. `compute_resample_groups` rend des paires `(start, end)`
ou `end` est l'index INCLUSIF du dernier element du seau, et le consommateur
canonique lit l'horodatage du seau comme `src.value(end)`. Quatre endroits de
l'orchestrateur construisaient a cote un vecteur parallele d'horodatages en
indexant `end - 1`. Consequences, les deux invisibles depuis un test qui ne
resample pas:

  1. le vecteur contredisait d'une barre native les barres qu'il accompagne;
  2. sur un premier seau d'UNE SEULE barre, `end` vaut 0 et `end - 1`
     sous-deborde `usize`: PanicException, "attempt to subtract with overflow".

Aucun test Python n'exercait `resample_to`, ce qui est exactement pourquoi ces
deux defauts ont survecu. Celui-ci le fait, et il commence les donnees A
L'INTERIEUR d'un seau, ce qui est la seule facon d'obtenir le premier groupe
incomplet.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, param, when  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.indicators import close as close_px, sma  # noqa: E402

CAPITAL = 100_000.0


def _barres(rows, depart, seed=11):
    """Marche aleatoire a la minute, demarrant a `depart`."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 3e-4, rows)))
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(depart, periods=rows, freq="1min", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.0002,
            "low": np.minimum(open_, close) * 0.9998,
            "close": close,
            "volume": np.full(rows, 1_000.0),
        }
    )


def _magasin(df, tmp_path):
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    return bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )


def _config(df, resample_to=None):
    last_ns = int(df["timestamp"].iloc[-1].value)
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 86_400_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=0, execution_price="AtClose", max_position_pct=1.0,
            allow_short=False, position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )
    cfg.resample_to = resample_to
    return cfg


def _strategie():
    return (
        bt.Strategy.create("croisement")
        .signal("fast", sma(close_px, param("fast")))
        .signal("slow", sma(close_px, param("slow")))
        .size(when(col("fast") > col("slow"), lit(1.0), lit(0.0)))
    )


# Le premier seau ne contient qu'une barre quand les donnees commencent une
# minute avant sa fin: 00:04 pour un seau de 5 minutes, 00:14 pour 15.
@pytest.mark.parametrize(
    "depart, minutes",
    [
        ("2021-03-01 00:04", 5),
        ("2021-03-01 00:14", 15),
        ("2021-03-01 00:59", 60),
        ("2021-03-01 00:00", 5),   # seau de tete complet: le temoin
    ],
)
def test_un_premier_seau_d_une_barre_ne_fait_pas_paniquer(tmp_path, depart, minutes):
    df = _barres(900, depart)
    store = _magasin(df, tmp_path)
    combos = bt.run_sweep_lite(
        _strategie(),
        {"fast": [5, 10], "slow": [30]},
        _config(df, resample_to=Interval.minutes(minutes)),
        store,
    )
    assert len(combos) == 2, combos
    for c in combos:
        assert np.isfinite(c.final_equity), c.final_equity


def test_le_balayage_allege_resample_comme_le_run(tmp_path):
    """Un resample ne doit pas separer les deux chemins: le meme jeu de
    parametres, resample de la meme facon, rend la meme equite finale.

    C'est ce test qui voit le decalage d'une barre, parce que le run complet
    lit l'horodatage du seau par `resample_batch` (donc `end`) tandis que le
    chemin allege le reconstruisait a cote (donc `end - 1`)."""
    df = _barres(1_500, "2021-03-01 00:04")
    store = _magasin(df, tmp_path)
    cfg = _config(df, resample_to=Interval.minutes(5))

    fixe = (
        bt.Strategy.create("fixe")
        .signal("fast", sma(close_px, 5))
        .signal("slow", sma(close_px, 30))
        .size(when(col("fast") > col("slow"), lit(1.0), lit(0.0)))
    )
    plein = bt.run(fixe, cfg, store)
    combos = bt.run_sweep_lite(_strategie(), {"fast": [5], "slow": [30]}, cfg, store)

    assert len(combos) == 1
    assert combos[0].final_equity == pytest.approx(
        plein.equity_df()["equity"].iloc[-1], rel=1e-9
    ), (
        f"allege {combos[0].final_equity:,.4f} contre plein "
        f"{plein.equity_df()['equity'].iloc[-1]:,.4f}"
    )


def test_un_compte_est_juge_pareil_des_deux_cotes_sous_resample(tmp_path):
    """Le verdict d'un compte lit les frontieres de seance, donc il lit les
    horodatages: c'est le test qui voit le decalage d'une barre.

    Les fonctions de balayage reconstruisaient les horodatages grossiers a
    cote, avec `end - 1`, tandis que le run les recevait de `resample_batch`,
    avec `end`. Les deux chemins attribuaient donc les memes marques a des
    seances differentes, et un compte mourait a une barre differente selon
    qu'on lancait un backtest ou un balayage a une seule combinaison.
    """
    df = _barres(4_320, "2021-03-01")   # trois jours a la minute
    store = _magasin(df, tmp_path)
    cfg = _config(df, resample_to=Interval.minutes(15))
    cfg.account_rules = bt.AccountRules(
        phases=[bt.AccountPhase(max_total_loss=0.02, max_daily_loss=0.005)],
        sessions=bt.account_sessions("2021-03-01", "2021-03-05", tz="UTC"),
    )

    fixe = (
        bt.Strategy.create("fixe")
        .signal("fast", sma(close_px, 5))
        .signal("slow", sma(close_px, 30))
        .size(when(col("fast") > col("slow"), lit(1.0), lit(0.0)))
    )
    plein = bt.run(fixe, cfg, store).account
    lite = bt.run_sweep_lite(_strategie(), {"fast": [5], "slow": [30]}, cfg, store)[0].account

    assert plein is not None, "le run doit rendre une issue"
    assert lite is not None, "le balayage aussi"
    for champ in ("outcome", "phase_index", "timestamp", "trading_days"):
        assert plein[champ] == lite[champ], (
            f"{champ}: run {plein[champ]!r} contre balayage {lite[champ]!r}"
        )
    for champ in ("equity", "floor"):
        assert plein[champ] == pytest.approx(lite[champ], rel=1e-12), (
            f"{champ}: run {plein[champ]} contre balayage {lite[champ]}"
        )


@pytest.mark.parametrize("bar_interval, resample", [
    ("1h", None),          # bar_interval plus grossier que le magasin
    ("1m", "1h"),          # resample_to plus grossier que le magasin
])
def test_une_infraction_interne_a_une_barre_grossiere_est_vue_des_deux_cotes(
    tmp_path, bar_interval, resample
):
    """Le cas qui separait les deux chemins, et que mon premier test de parite
    n'exercait pas.

    `bt.run` marque le compte a chaque barre NATIVE, y compris quand les
    signaux tournent sur une grille grossiere. Le chemin allege
    pre-echantillonnait, donc il ne marquait qu'une fois par barre grossiere:
    une infraction qui naît et se referme A L'INTERIEUR d'une barre grossiere
    etait vue par l'un et invisible a l'autre, dans un seul sens, celui qui
    SURESTIME le taux de reussite. Mesure avant correction, sur cette serie:
    run FAILED_MAX_LOSS a 78 000, balayage PASSED a 112 000.

    Le chemin allege delegue maintenant au noyau complet quand des regles de
    compte sont posees et que la grille simulee est plus grossiere, exactement
    comme il le fait deja pour un ordre de sortie, qui se resout lui aussi
    intra-barre.
    """
    n = 2_880
    px = np.full(n, 100.0)
    px[200:211] = 78.0       # -22 % pendant onze minutes, dans l'heure 03:00
    px[211:] = 100.0
    px[1_500:] = 112.0       # puis au-dessus de la cible
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": px, "high": px, "low": px, "close": px,
        "volume": np.full(n, 1_000.0),
    })
    store = _magasin(df, tmp_path)

    inter = Interval.hours(1) if bar_interval == "1h" else Interval.minutes(1)
    cfg = _config(df, resample_to=Interval.hours(1) if resample else None)
    cfg.bar_interval = inter
    cfg.account_rules = bt.AccountRules(
        phases=[bt.AccountPhase(profit_target=0.02, max_total_loss=0.10)],
        sessions=bt.account_sessions("2024-01-01", "2024-01-03", tz="UTC"),
    )

    tenir = bt.Strategy.create("tenir").signal("px", close_px).size(bt.lit(1.0))
    plein = bt.run(tenir, cfg, store).account
    lite = bt.run_sweep_lite(tenir, {}, cfg, store)[0].account

    assert plein is not None and lite is not None, (plein, lite)
    assert plein["outcome"] == "FAILED_MAX_LOSS", (
        f"le plongeon perce le plancher, le run doit le voir: {plein}"
    )
    for champ in ("outcome", "timestamp", "phase_index"):
        assert plein[champ] == lite[champ], (
            f"{champ}: run {plein[champ]!r} contre balayage {lite[champ]!r}"
        )
    assert plein["equity"] == pytest.approx(lite["equity"], rel=1e-12)
