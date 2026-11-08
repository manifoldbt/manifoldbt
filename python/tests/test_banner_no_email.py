"""The import banner names the version and the tier, never an address.

Scripts and notebooks often capture stdout into logs and files, so what the
banner prints on `import manifoldbt` ends up wherever that output goes.
`license_info()` still returns the address on request; only the unprompted
printing is covered here.
"""
import pathlib
import re
import subprocess
import sys

import manifoldbt as bt


def test_the_banner_does_not_print_the_email():
    # The banner runs once, at import: read it from a fresh process.
    out = subprocess.run(
        [sys.executable, "-c", "import manifoldbt"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "manifoldbt v" in out, out
    assert not re.search(r"\S+@\S+", out), out
    _, email = bt.license_info()
    if email:
        assert email not in out, out


def test_the_banner_code_does_not_read_the_email():
    # Holds on a machine without any licence, where the run above has no
    # address to look for: the function that prints the banner must not
    # format the second value returned by the licence lookup.
    src = pathlib.Path(bt.__file__).read_text(encoding="utf-8")
    start = src.index("def _print_banner")
    end = src.index("_print_banner()", start)
    body = src[start:end]
    assert "email" not in body, body
