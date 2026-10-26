"""The generated bars are the same on every machine.

The CPU benchmark runs on a GitHub runner and the GPU benchmark on a rented
card; the site joins their times on the digest of the data. That join only
holds if the same seed gives the same bits on every CPU. It did not while the
walk went through ``numpy.exp``: an AVX-512 runner, a runner without AVX-512
and Windows gave three digests for one seed. These tests pin the property, not
the machine: the walk is a sequential product, and the digest of the default
dataset is fixed for the numpy release in requirements-lock.txt.
"""
import numpy as np
import pytest

import data

# numpy==2.4.3 (requirements-lock.txt); identical on Linux and Windows.
PINNED = {20_000: "0d360424244ea1ee", 200_000: "41728ac4d4fb6c04"}


def test_the_walk_is_a_sequential_product_of_the_draws():
    rows = 5_000
    frame = data.make_ohlcv(rows)
    ret = np.random.default_rng(data.DEFAULT_SEED).normal(2e-7, 3e-4, size=rows)
    level = 1.0
    expected = []
    for r in ret:
        level *= 1.0 + r
        expected.append(100.0 * level)
    assert frame["close"].to_numpy().tobytes() == np.asarray(expected).tobytes()


def test_open_is_the_previous_close():
    frame = data.make_ohlcv(1_000)
    assert frame["open"].iloc[0] == 100.0
    assert (frame["open"].to_numpy()[1:] == frame["close"].to_numpy()[:-1]).all()


@pytest.mark.skipif(np.__version__ != "2.4.3", reason="digest pinned for numpy 2.4.3")
@pytest.mark.parametrize("rows, digest", sorted(PINNED.items()))
def test_the_digest_does_not_depend_on_the_machine(rows, digest):
    assert data.digest(data.make_ohlcv(rows)) == digest
