"""Rendre visibles les bibliotheques CUDA installees par pip.

L'extra ``manifoldbt[gpu]`` installe ``nvidia-cuda-nvrtc-cu12``, qui depose
``libnvrtc.so.12`` / ``nvrtc64_120_0.dll`` dans ``site-packages/nvidia/``.
Ce dossier n'est ni dans le ``PATH`` (Windows) ni dans le chemin de recherche du
chargeur dynamique (Linux). Le coeur Rust charge NVRTC par son NOM, via
``libloading``, donc sans ce coup de pouce il ne trouve rien et le chemin GPU
echoue alors que la bibliotheque est bel et bien installee.

C'est le meme probleme que PyTorch resout a son import, et par les memes moyens:
``os.add_dll_directory`` sous Windows, un pre-chargement ``RTLD_GLOBAL`` sous
Linux (une bibliotheque deja chargee sous son SONAME satisfait un ``dlopen``
ulterieur qui la demande par ce nom).

Sans effet quand l'extra n'est pas installe, ou sur une roue sans CUDA (macOS,
ARM, musl): les dossiers n'existent pas, tout est ignore. Aucune exception ne
remonte, un echec ici ne doit jamais empecher un import.
"""
import os
import sys
from pathlib import Path

# Sous-dossiers de site-packages/nvidia/ qui portent des bibliotheques utiles au
# moteur. NVRTC compile les noyaux au runtime; le pilote lui-meme (libcuda) vient
# de l'installation systeme, jamais de pip.
_COMPOSANTS = ("cuda_nvrtc", "cuda_runtime")

_fait = False


def _dossiers_candidats():
    """Les dossiers de bibliotheques des paquets nvidia-*, s'ils existent."""
    vus = set()
    for base in sys.path:
        if not base:
            continue
        racine = Path(base) / "nvidia"
        if racine in vus or not racine.is_dir():
            continue
        vus.add(racine)
        for composant in _COMPOSANTS:
            for feuille in ("bin", "lib"):
                d = racine / composant / feuille
                if d.is_dir():
                    yield d


def rendre_visible():
    """Idempotent, silencieux, sans effet quand aucune lib pip n'est presente."""
    global _fait
    if _fait:
        return
    _fait = True

    for d in _dossiers_candidats():
        try:
            if sys.platform == "win32":
                # add_dll_directory n'agit que sur les chargements ulterieurs,
                # d'ou l'appel a l'import et non au premier usage du GPU.
                os.add_dll_directory(str(d))
            else:
                import ctypes

                for lib in sorted(d.glob("libnvrtc.so*")):
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    break
        except Exception:
            # Un dossier illisible, une DLL incompatible, une plateforme
            # exotique: rien de tout cela ne justifie de casser l'import du
            # paquet. Le chemin GPU rendra une erreur claire s'il ne trouve
            # pas sa bibliotheque.
            continue
