"""Un compte de prop firm se ferme PENDANT le run, vu depuis Python.

Juger les limites d'une firme apres coup, sur une courbe d'equite finie, donne
la bonne premiere infraction et se trompe sur tout le reste : la courbe a
continue a negocier une vie que le compte n'avait plus. La serie de ce test
plonge sous le plancher puis se reprend bien au-dessus du depart, ce qui separe
les deux lectures en un seul chiffre.

Le test verifie aussi ce que l'appelant doit fournir : la frontiere de seance.
Le moteur ne porte aucune base de fuseaux, donc `account_sessions` la construit
avec `zoneinfo`, et le changement d'heure s'y voit.
"""
import os
import tempfile

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import manifoldbt as bt
from manifoldbt.indicators import close

N_BARS = 60
FOND = 25


def _prix():
    return np.array(
        [
            100.0 - 12.0 * i / FOND if i <= FOND
            else 88.0 + 32.0 * (i - FOND) / (N_BARS - 1 - FOND)
            for i in range(N_BARS)
        ]
    )


@pytest.fixture(scope="module")
def magasin(tmp_path_factory):
    px = _prix()
    ts = pd.date_range("2024-01-01", periods=N_BARS, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": px,
            "high": px * 1.0005,
            "low": px * 0.9995,
            "close": px,
            "volume": np.full(N_BARS, 1000.0),
        }
    )
    root = tmp_path_factory.mktemp("propfirm")
    return bt.import_dataframe(
        df, symbol="ZC", symbol_id=1, interval="1m", asset_class="equity",
        exchange="TEST", data_root=os.path.join(str(root), "data"),
        metadata_db=os.path.join(str(root), "metadata.sqlite"),
    )


def _config(regles):
    start, end = bt.time_range("2024-01-01", "2024-01-04")
    cfg = bt.BacktestConfig(
        universe=[1], time_range_start=start, time_range_end=end,
        initial_capital=100_000.0, provider="TEST",
        bar_interval=bt.Interval.hours(1), symbol_names={"ZC": 1},
    )
    cfg.warmup_bars = 0
    cfg.account_rules = regles
    return cfg


def _strategie():
    return (
        bt.Strategy.create("tenir")
        .signal("px", close)
        .size(bt.lit(0.9))
    )


def test_un_compte_ferme_ne_negocie_pas_la_reprise(magasin):
    libre = bt.run(_strategie(), _config(None), magasin)
    assert libre.account is None, "aucune regle, aucune issue"
    fin_libre = libre.equity_df()["equity"].iloc[-1]

    regles = bt.AccountRules(
        phases=[bt.AccountPhase(max_total_loss=0.10)],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
    )
    ferme = bt.run(_strategie(), _config(regles), magasin)
    compte = ferme.account
    assert compte is not None, "la regle doit rendre un verdict"
    assert compte["outcome"] == "FAILED_MAX_LOSS", compte
    assert compte["reason"], "le verdict dit pourquoi, en clair"

    fin_ferme = ferme.equity_df()["equity"].iloc[-1]
    assert fin_ferme < fin_libre - 10_000, (
        f"le compte ferme ne remonte pas: {fin_ferme:,.0f} contre {fin_libre:,.0f} "
        "sans regle. Un verdict post-hoc aurait lu la reprise."
    )


def test_les_seances_suivent_le_changement_d_heure():
    """Un decalage fixe serait faux deux fois par an: minuit a Prague vaut
    23:00 UTC en hiver et 22:00 UTC en ete."""
    s = bt.account_sessions("2024-03-29", "2024-04-02", tz="Europe/Prague")
    heures = [pd.Timestamp(t, unit="ns", tz="UTC").hour for t in s]
    assert heures[:3] == [23, 23, 23], heures
    assert heures[3:] == [22, 22], heures


def test_un_balayage_rend_un_verdict_par_combinaison(magasin):
    """Ce qui fait d'un balayage un taux de reussite : une issue par
    combinaison, jugee par la meme regle qu'un run simple.

    La parite stricte entre la boucle allegee et la boucle complete se teste
    cote Rust (`propfirm_account.rs`), ou les deux tournent sur la config
    IDENTIQUE. Ici c'est impossible : balayer un parametre le CHANGE, et
    `stop_loss` attache un ordre de sortie que le run simple n'a pas. Comparer
    les deux equites ici comparerait deux strategies differentes.
    """
    regles = bt.AccountRules(
        phases=[bt.AccountPhase(max_total_loss=0.10)],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
    )
    combos = bt.run_sweep_lite(
        _strategie(), {"stop_loss": [0.5, 0.6]}, _config(regles), magasin
    )
    assert combos, "au moins une combinaison"
    for c in combos:
        assert c.account is not None, "chaque combinaison rend un verdict"
        assert c.account["outcome"] in {
            "PASSED", "FAILED_DAILY_LOSS", "FAILED_MAX_LOSS", "RAN_OUT_OF_TIME"
        }, c.account
        assert c.account["reason"]

    # et c'est exactement ce qui rend un taux de reussite calculable
    taux = sum(1 for c in combos if c.account["outcome"] == "PASSED") / len(combos)
    assert 0.0 <= taux <= 1.0


def test_un_programme_en_deux_phases_repart_sur_un_compte_neuf(magasin):
    """Le 2-Step est DEUX phases, et la firme donne un compte neuf entre les
    deux. Une Verification ne demarre donc pas deja au-dessus de sa cible parce
    que le Challenge a fini a +10 %. Confondre les deux transforme un 2-Step en
    1-Step et gonfle tous les taux de reussite qu'on en tire."""
    # une serie qui monte: le Challenge passe, la Verification doit encore
    # gagner son propre +5 % a partir de la
    deux_phases = bt.AccountRules(
        phases=[
            bt.AccountPhase(name="Challenge", profit_target=0.02,
                            max_total_loss=0.50),
            bt.AccountPhase(name="Verification", profit_target=0.50,
                            max_total_loss=0.50),
        ],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
    )
    res = bt.run(_strategie(), _config(deux_phases), magasin)
    # la serie monte de 18 % en tout: assez pour le Challenge a +2 %, pas assez
    # pour une Verification a +50 % SUR LE SOLDE NEUF
    assert res.account is None or res.account["outcome"] != "PASSED", (
        "la Verification ne doit pas passer sur le profit du Challenge: "
        f"{res.account}"
    )

    # la meme Verification, a une cible qu'un solde neuf peut atteindre
    deux_phases.phases[1].profit_target = 0.02
    res2 = bt.run(_strategie(), _config(deux_phases), magasin)
    assert res2.account is not None and res2.account["outcome"] == "PASSED", res2.account
    assert res2.account["phase"] == "Verification", res2.account
    assert res2.account["phase_index"] == 1


def test_une_seance_qui_porte_tout_le_profit_bloque_le_paiement(magasin):
    """La regle de coherence ne FERME pas le compte: elle bloque le retrait.
    Le verdict la rapporte a cote de l'issue, jamais comme une infraction."""
    regles = bt.AccountRules(
        phases=[bt.AccountPhase(profit_target=0.02, max_total_loss=0.50)],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
        consistency=0.10,   # tres strict: une seance ne peut porter que 10 %
    )
    res = bt.run(_strategie(), _config(regles), magasin)
    assert res.account is not None
    assert res.account["outcome"] == "PASSED", "le compte n'est PAS ferme"
    assert res.account["consistency_ok"] is False, (
        "mais une seance a porte trop de profit, donc pas de paiement"
    )

    # sans cap, le meme run garde son paiement
    regles.consistency = None
    res2 = bt.run(_strategie(), _config(regles), magasin)
    assert res2.account["consistency_ok"] is True


def test_un_compte_ferme_previent_que_les_metriques_mentent(magasin):
    """Un compte ferme arrete de trader, donc la courbe est plate ensuite et
    `metrics` decrit une vie que le compte n'a pas eue. Sur un run mesure,
    `total_return` annoncait +10,39 % pendant que l'issue etait
    FAILED_DAILY_LOSS. Le moteur doit le DIRE, pas laisser lire."""
    regles = bt.AccountRules(
        phases=[bt.AccountPhase(max_total_loss=0.10)],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
    )
    res = bt.run(_strategie(), _config(regles), magasin)
    assert res.account is not None and res.account["outcome"] != "PASSED"
    joint = " ".join(res.warnings).lower()
    assert "frozen equity" in joint and "account" in joint, res.warnings

    # sans regle, aucun avertissement de ce genre
    libre = bt.run(_strategie(), _config(None), magasin)
    assert not any("frozen equity" in w for w in libre.warnings)


def test_un_compte_ferme_ne_laisse_aucun_fill_orphelin(magasin):
    """Les trois sorties doivent dire la MEME chose apres le verdict.

    Un verdict de cloture de seance se resout a l'arrivee d'une marque de la
    seance suivante. Tant qu'il etait rendu en FIN de barre, cette barre avait
    deja ete valorisee et avait deja pris ses remplissages: il fallait defaire
    l'equite apres coup, et le journal des trades gardait un achat que la
    courbe ne payait pas, frais compris. Mesure sur un run: un BUY de 10,98
    unites a 101,39 avec 0,55 de frais, a une barre ou `positions_df` disait
    zero et `equity_df` etait gelee.

    La frontiere de seance se lisant sur l'horodatage seul, le verdict tombe
    maintenant en TETE de barre et il n'y a plus rien a defaire.
    """
    regles = bt.AccountRules(
        phases=[bt.AccountPhase(profit_target=0.01, max_total_loss=0.50)],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
    )
    # Une taille qui BOUGE a chaque barre, donc un remplissage a chaque barre:
    # une strategie a taille constante ne remplit qu'une fois et l'assertion
    # sur le journal serait vide.
    mouvante = (
        bt.Strategy.create("mouvante")
        .signal("px", close)
        .size(close / bt.lit(150.0))
    )
    res = bt.run(mouvante, _config(regles), magasin)
    compte = res.account
    assert compte is not None, "la serie doit resoudre le compte"
    assert len(res.trades_df()) > 10, (
        f"le journal doit porter des remplissages tout du long, il en a "
        f"{len(res.trades_df())}"
    )

    # `equity_df` rend des horodatages naifs et `trades_df` des horodatages
    # localises: on compare en nanosecondes, ce qui marche des deux cotes.
    fin_ns = int(compte["timestamp"])
    ns = lambda col: pd.to_datetime(col, utc=True).astype("int64")

    trades = res.trades_df()
    apres = trades[ns(trades["execution_timestamp"]) > fin_ns]
    assert apres.empty, (
        f"un compte ferme ne prend plus de remplissage, or le journal en porte "
        f"{len(apres)} apres le verdict, pour {apres['fees'].sum():.6f} de frais que "
        f"la courbe ne paie pas:\n{apres}"
    )

    eq = res.equity_df()
    gelee = eq[ns(eq["timestamp"]) > fin_ns]
    assert len(gelee), "la serie doit continuer apres le verdict"
    assert gelee["equity"].nunique() == 1, (
        f"la courbe doit etre plate apres le verdict, elle prend "
        f"{gelee['equity'].nunique()} valeurs"
    )
    assert gelee["equity"].iloc[0] == pytest.approx(compte["equity"], rel=1e-12), (
        f"et plate SUR l'equite du verdict: {gelee['equity'].iloc[0]:,.6f} "
        f"contre {compte['equity']:,.6f}"
    )

    pos = res.positions_df()
    plates = pos[ns(pos["timestamp"]) > fin_ns]
    assert (plates["position"].abs() < 1e-12).all(), (
        f"un compte ferme ne tient aucune position:\n{plates.head()}"
    )
    assert (plates["capital"] - plates["equity"]).abs().max() < 1e-9, (
        "et un compte plat a son capital pour equite"
    )


def test_un_balayage_rapporte_ses_avertissements(magasin):
    """Une mise en garde que personne ne peut lire ne sert a rien.

    La boucle allegee collectait ses avertissements et le resultat les jetait:
    la destructuration de `SimLiteResult` les abandonnait et `BatchResultLite`
    n'avait pas le champ. Donc une liste de seances qui ne recouvre pas le run
    etait dite a voix haute sur un backtest simple et muette sur toute une
    grille, ce qui est exactement l'inverse de ce dont on a besoin: c'est sur
    un balayage qu'on lit un taux de reussite.
    """
    hors_periode = bt.AccountRules(
        phases=[bt.AccountPhase(max_total_loss=0.10)],
        # le magasin couvre janvier 2024; ces ouvertures visent 2022
        sessions=bt.account_sessions("2022-01-01", "2022-01-04", tz="UTC"),
    )
    combos = bt.run_sweep_lite(
        _strategie(), {"stop_loss": [0.5]}, _config(hors_periode), magasin
    )
    assert combos, "au moins une combinaison"
    joint = " ".join(combos[0].warnings)
    assert "session start" in joint, (
        f"le balayage doit dire que la liste ne recouvre pas la periode; "
        f"il dit: {combos[0].warnings}"
    )

    # et une liste qui recouvre ne raconte rien de ce genre
    bonnes = bt.AccountRules(
        phases=[bt.AccountPhase(max_total_loss=0.10)],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
    )
    propres = bt.run_sweep_lite(
        _strategie(), {"stop_loss": [0.5]}, _config(bonnes), magasin
    )
    assert not any("session start" in w for w in propres[0].warnings), (
        propres[0].warnings
    )


@pytest.mark.parametrize("champ", ["max_total_loss", "max_daily_loss"])
@pytest.mark.parametrize("valeur", [1e-17, 5e-324, 0.0])
def test_une_limite_de_perte_doit_etre_representable_sous_le_solde(
    magasin, champ, valeur
):
    """`1e-17` de perte maximale tuait un compte qui n'avait RIEN perdu.

    `initial - initial * pct` rend `initial` lui-meme des que le produit passe
    sous le demi-ULP de l'equite, donc le plancher se pose exactement SUR le
    solde et la comparaison `<=` ferme le compte a sa premiere marque. Mesure
    avant correction, prix constant et aucune position: plancher affiche a
    100000,0000000000 et FAILED_DAILY_LOSS des la premiere barre. Le seuil reel
    est ~7,3e-17 quel que soit le capital.
    """
    phase = bt.AccountPhase(max_total_loss=0.10)
    setattr(phase, champ, valeur)
    regles = bt.AccountRules(
        phases=[phase],
        sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
    )
    with pytest.raises(Exception) as capture:
        bt.run(_strategie(), _config(regles), magasin)
    assert champ in str(capture.value), (
        f"{champ}={valeur} doit se refuser en NOMMANT le champ, le refus dit: "
        f"{capture.value}"
    )


def test_une_limite_de_perte_realiste_reste_acceptee(magasin):
    """La borne ne doit pas mordre sur ce qu'une firme publie vraiment."""
    for total, journalier in ((0.10, 0.05), (0.04, None), (1.0, 1.0), (1e-9, 1e-9)):
        regles = bt.AccountRules(
            phases=[bt.AccountPhase(max_total_loss=total,
                                    max_daily_loss=journalier)],
            sessions=bt.account_sessions("2024-01-01", "2024-01-04", tz="UTC"),
        )
        res = bt.run(_strategie(), _config(regles), magasin)
        assert res is not None, (total, journalier)
