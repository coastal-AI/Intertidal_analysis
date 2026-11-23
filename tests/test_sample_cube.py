# -*- coding: utf-8 -*-
"""The notebook sample (``data/villaviciosa_sample.nc``) reproduces the product's elevations on its block.

Skipped when the sample is not there. The NDWI and the clear looks are computed as ``marea.extract`` computes them,
and checked against ``marea.extract`` run on the sample itself. Every intertidal pixel of the block is then inverted
with ``marea.invert_series`` on the tide delayed by the pixel's stored delay, searching elevations between the lowest
and the highest overpass level, with no quality rule, as in the product: ``z`` and ``sigma`` must equal the stored
layers bit for bit, NaN included.
"""
import os  # paths

import numpy as np  # arrays
import pytest  # the test runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
SAMPLE = os.path.join(ROOT, "data", "villaviciosa_sample.nc")  # the sample of the notebooks

pytestmark = pytest.mark.skipif(not os.path.exists(SAMPLE), reason="data/villaviciosa_sample.nc is not present")


@pytest.fixture(scope="module")
def sample():
    """The bands, the looks, the tide and the stored layers of the sample, as arrays."""
    import xarray as xr

    from pyintertidal import marea

    with xr.open_dataset(SAMPLE) as s:  # only the variables below are read
        g = s["B03"].values.astype(np.float32)  # green; the int16 no-value code is decoded to NaN
        n = s["B08"].values.astype(np.float32)  # near infrared
        scl = s["SCL"].values  # scene classification
        out = {
            "clear_stored": s["clear"].values == 1,
            "intertidal": s["intertidal"].values == 1,
            "level": s["tide_level"].values.astype(float),
            "delayed": {int(d): s["tide_level_delayed"].sel(delay=int(d)).values.astype(float)
                        for d in np.unique(s["delay_marea"].values[s["intertidal"].values == 1])},
            "delay": s["delay_marea"].values,
            "z": s["z_marea"].values,
            "sigma": s["sigma_marea"].values,
            "dates": np.array([str(np.datetime64(v, "D")) for v in s["t"].values]),
        }
    with np.errstate(invalid="ignore", divide="ignore"):  # exactly the arithmetic of marea.extract
        out["ndwi"] = np.where(g + n != 0, (g - n) / (g + n), np.nan)
    out["clear"] = np.isin(scl, marea.CLEAR)
    return out


def test_clear_looks_are_the_scl_rule(sample):
    assert np.array_equal(sample["clear"], sample["clear_stored"])  # SCL 4, 5, 6 or 7


def test_ndwi_and_looks_equal_marea_extract(sample):
    from pyintertidal import marea

    ex = marea.extract(SAMPLE)  # MAREA's own streaming extraction on the sample
    assert tuple(ex["shape"]) == sample["ndwi"].shape[1:]
    assert list(ex["dates"]) == list(sample["dates"])
    rows, cols = np.divmod(ex["keep"], ex["shape"][1])  # the pixels extract keeps on this block
    assert ex["keep"].size > 1000
    assert np.array_equal(ex["Y"], sample["ndwi"][:, rows, cols], equal_nan=True)
    assert np.array_equal(ex["C"], sample["clear"][:, rows, cols])


def test_stored_delays_reproduce_z_and_sigma(sample):
    from pyintertidal import marea

    rows, cols = np.nonzero(sample["intertidal"])  # the product's pixels in this block, row by row
    assert rows.size == 15064
    Y = np.nan_to_num(sample["ndwi"][:, rows, cols], nan=0.0).astype(np.float64)  # as marea.reconstruct does
    C = sample["clear"][:, rows, cols].astype(np.float64)
    delay = sample["delay"][rows, cols]
    lo, hi = float(sample["level"].min()), float(sample["level"].max())  # the range searched by the product
    z = np.full(rows.size, np.nan)
    sg = np.full(rows.size, np.nan)
    for d in np.unique(delay):  # one inversion per delay, on the tide read that many minutes earlier
        j = delay == d
        z[j], sg[j] = marea.invert_series(Y[:, j], C[:, j], sample["delayed"][int(d)], lo, hi)
    assert np.array_equal(z.astype(np.float32), sample["z"][rows, cols], equal_nan=True)
    assert np.array_equal(sg.astype(np.float32), sample["sigma"][rows, cols], equal_nan=True)
    assert np.isfinite(z).sum() == 14171  # pixels with an elevation in this block
