"""Shared plumbing for the examples. Not an example itself.

Two things every `Data: shared store` example needs, in one place instead of
twenty copies:

  `open_store()`      opens the `data/` + `metadata/` pair those files read,
                      and says how to populate it when it is missing rather
                      than failing on an Arrow path the reader cannot act on.

  `plots_available()` answers whether the optional plotting extra is here.
                      Charts are an extra (`pip install manifoldbt[plot]`), so
                      an example that ends on a chart must still be able to
                      finish on the plain install the README leads with.

Imported as a bare module name (`from _bootstrap import open_store`), which
works because Python puts the script's own directory first on `sys.path` --
i.e. exactly when the file is run the documented way, `python examples/NN.py`.
"""
import importlib
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_ROOT = os.path.join(_ROOT, "data")
METADATA_DB = os.path.join(_ROOT, "metadata", "metadata.sqlite")
ARROW_DIR = os.path.join(DATA_ROOT, "mega")

_NO_STORE = f"""
No shared data store at {DATA_ROOT}

The examples marked `Data: shared store` read real market data, which is not
in the repository -- it has to be downloaded once. From the repository root:

    python examples/setup_data.py

That ingests the symbols these examples name, at 1h, and takes a few minutes.
The synthetic and self-contained examples (13, 18, 21, 22, 23, 24, 25) need
none of it.
"""


def open_store():
    """Open the store the `Data: shared store` examples read.

    Exits with the setup command when the store has not been populated yet.
    """
    import manifoldbt as mbt

    if not os.path.isdir(ARROW_DIR):
        raise SystemExit(_NO_STORE)

    return mbt.DataStore(
        data_root=DATA_ROOT,
        metadata_db=METADATA_DB,
        arrow_dir=ARROW_DIR,
    )


def plots_available():
    """True when charts can be drawn; otherwise prints why and returns False.

    Guards the chart at the end of an example so that `pip install manifoldbt`
    -- the engine-only install -- still runs the file to completion.
    """
    try:
        importlib.import_module("manifoldbt.plot")
    except ImportError as exc:
        print(f"\n[chart skipped] {exc}", file=sys.stderr)
        return False
    return True
