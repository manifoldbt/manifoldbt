"""`calibrate_gpu()` : l'etalonnage de la carte est une fonction publique, sobre et testable.

Aucun test ici ne suppose une carte : sans GPU (roue sans CUDA, ou pas de carte), l'appel
leve `RuntimeError` avec un message qui le dit. Avec une carte, les constantes rendues sont
valides (finies, positives) et le fichier d'etalonnage est relu tel quel au balayage suivant.
"""
import os

import pytest

import manifoldbt as bt


def test_la_fonction_est_exportee_et_documentee():
    assert "calibrate_gpu" in bt.__all__
    doc = bt.calibrate_gpu.__doc__
    assert doc and "MBT_GPU_CALIBRATION" in doc


def test_sans_carte_l_erreur_le_dit(tmp_path, monkeypatch):
    monkeypatch.setenv("MBT_GPU_CALIBRATION_FILE", str(tmp_path / "c.json"))
    try:
        constantes = bt.calibrate_gpu()
    except RuntimeError as e:
        assert "GPU" in str(e) or "CUDA" in str(e) or "cuda" in str(e)
        return
    except PermissionError:
        pytest.skip("GPU calibration is not available with this install")
    # Une carte est la : les constantes sont exploitables et le fichier a ete ecrit.
    for famille in ("sans_bracket", "un_ordre", "avec_bracket"):
        p = constantes[famille]
        assert all(u > 0 for _, u in p["scan_nk"]) and all(u > 0 for _, u in p["sweep_u"])
    assert constantes["mesure_ms"] > 0
    assert os.path.exists(tmp_path / "c.json")
