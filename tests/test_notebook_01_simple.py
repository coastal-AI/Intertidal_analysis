"""The n-tables notebooks (notebooks/validation/01_*.ipynb) are plain notebooks.

* both are valid notebooks and every code cell compiles;
* no code cell reads the RTK survey (none mentions it);
* the site comes from the Parameters cell, overridable by NB_SITE;
* the scoring function of the site notebook gives the right numbers on tiny synthetic cases;
* the diagnostic variants: z at NDWI = 0 is MAREA's own fit (its midpoint is marea.invert_series' z, bit for
  bit) read where the fitted curve crosses 0; the variants are scored but never enter the common pixels; every
  default variant has a run function.

Run:  python -m pytest tests/test_notebook_01_simple.py   (or python tests/test_notebook_01_simple.py)
"""
import os

import nbformat
import numpy as np
from scipy.special import ndtr, ndtri

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_NB = os.path.join(ROOT, "notebooks", "validation", "01_fixed_config_n_tables_site.ipynb")
SUMMARY_NB = os.path.join(ROOT, "notebooks", "validation", "01b_fixed_config_n_tables_summary.ipynb")


def code_cells(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def cell_with(text, path=SITE_NB):
    return next(s for s in code_cells(path) if text in s)


def scores_function():
    ns = {"np": np}
    exec(cell_with("def scores("), ns)
    return ns["scores"]


def invert_ndwi0_function():
    import sys
    sys.path.insert(0, ROOT)
    from pyintertidal import marea
    from pyintertidal.elevation import _fit_block

    ns = {"np": np, "marea": marea, "_fit_block": _fit_block, "ndtri": ndtri}
    exec(cell_with("def invert_ndwi0("), ns)
    return ns["invert_ndwi0"], marea


def test_valid_and_every_code_cell_compiles():
    for path in (SITE_NB, SUMMARY_NB):
        for k, src in enumerate(code_cells(path)):
            compile(src, f"{os.path.basename(path)} cell {k}", "exec")


def test_no_rtk_in_code():
    for path in (SITE_NB, SUMMARY_NB):
        for src in code_cells(path):
            assert "rtk" not in src.lower(), f"{os.path.basename(path)}: a code cell mentions the RTK"


def test_site_from_parameters_and_environment():
    first = code_cells(SITE_NB)[0]
    assert 'os.environ.get("NB_SITE"' in first
    for name in ("NB_NS", "NB_METHODS", "NB_VARIANTS", "NB_EXTRA", "NB_OUT", "NB_FIRST"):
        assert name in first
    assert '"results/fixedcfg_nb_diag"' in first, "the default output must not be the first run's folder"


def test_scores_offset_is_bias_not_centred_error():
    scores = scores_function()
    t = np.linspace(-1.0, 1.0, 50)
    s = scores(t + 0.1, t)
    assert "n" not in s, "the scores would overwrite the CSV row's n (images in the window)"
    assert s["n_px"] == 50 and s["coverage"] == 1.0
    assert np.isclose(s["rmse"], 0.1) and np.isclose(s["bias"], 0.1)
    assert np.isclose(s["crmse"], 0.0, atol=1e-12)
    assert np.isclose(s["slope"], 1.0) and np.isclose(s["r"], 1.0)


def test_scores_compressed_relief_and_coverage():
    scores = scores_function()
    t = np.linspace(-1.0, 1.0, 50)
    p = 0.5 * t
    p[:10] = np.nan                                  # 10 of 50 pixels without a value
    s = scores(p, t)
    ok = np.isfinite(p)
    assert s["n_px"] == 40 and np.isclose(s["coverage"], 0.8)
    assert np.isclose(s["slope"], 0.5) and np.isclose(s["r"], 1.0)
    assert np.isclose(s["bias"], np.median(p[ok] - t[ok]))
    assert np.isclose(s["rmse"], np.sqrt(np.mean((p[ok] - t[ok]) ** 2)))
    pc, tc = p[ok] - np.median(p[ok]), t[ok] - np.median(t[ok])
    assert np.isclose(s["crmse"], np.sqrt(np.mean((pc - tc) ** 2)))


def test_scores_below_30_pixels_give_counts_only():
    scores = scores_function()
    t = np.arange(20.0)
    s = scores(t, t)
    assert s["n_px"] == 20 and s["coverage"] == 1.0
    assert all(np.isnan(s[k]) for k in ("rmse", "bias", "crmse", "slope", "r"))


def synthetic_curves(P=9000, T=60, seed=3):
    """NDWI series a + b Phi((h - z)/sigma) + noise, a few pixels the guards reject and a few that never cross 0."""
    rng = np.random.default_rng(seed)
    h = np.sort(rng.uniform(-2.0, 2.0, T))[rng.permutation(T)]
    a, b = rng.uniform(-0.5, -0.1, P), rng.uniform(0.3, 1.0, P)
    a[:50], b[:50] = 0.05, 0.5                       # dry and wet NDWI both above 0: no crossing
    z, sg = rng.uniform(-1.5, 1.5, P), rng.uniform(0.05, 0.4, P)
    Y = a + b * ndtr((h[:, None] - z) / sg) + rng.normal(0.0, 0.05, (T, P))
    C = rng.random((T, P)) > 0.15
    C[:, 50:60] = False                              # no clear observation: rejected by the guards
    return Y.astype(np.float32), C, h


def test_ndwi0_midpoint_is_marea_bit_for_bit():
    invert_ndwi0, marea = invert_ndwi0_function()
    Y, C, h = synthetic_curves()
    for mb in (0, 3):
        z0, zmid = invert_ndwi0(Y, C, h, h.min(), h.max(), min_bracket=mb)
        zm, _ = marea.invert_series(Y, C, h, h.min(), h.max(), min_bracket=mb)
        assert np.array_equal(zmid, zm, equal_nan=True), f"min_bracket={mb}: not invert_series' fit"
        assert np.all(np.isnan(z0[np.isnan(zmid)])), "a rejected pixel got a value"
        assert np.isnan(z0[:50]).sum() >= 45, "pixels whose curve never crosses 0 must have no value"


def test_ndwi0_is_where_the_fitted_curve_crosses_zero():
    """Noise-free curves with z and sigma on the fit's grids: the fit is exact, so h0 = z + sigma Phi^-1(-a/b)."""
    invert_ndwi0, marea = invert_ndwi0_function()
    T, lo, hi = 80, -2.0, 2.0
    h = np.linspace(lo, hi, T)
    grid = np.linspace(lo, hi, 50)
    a = np.array([-0.3, -0.1, -0.45, 0.05, -0.6])
    b = np.array([0.8, 0.9, 0.5, 0.6, 0.5])           # 4th: a > 0; 5th: a + b < 0 (never crosses)
    z = grid[[20, 25, 30, 25, 25]]
    sg = np.array(marea.SG_GRID)[[2, 3, 1, 2, 2]]
    Y = a + b * ndtr((h[:, None] - z) / sg)
    z0, zmid = invert_ndwi0(Y, np.ones_like(Y, bool), h, lo, hi)
    assert np.allclose(zmid[:3], z[:3])
    assert np.allclose(z0[:3], z[:3] + sg[:3] * ndtri(-a[:3] / b[:3]), atol=1e-6)
    assert np.allclose(a[:3] + b[:3] * ndtr((z0[:3] - z[:3]) / sg[:3]), 0.0, atol=1e-6)
    assert np.isfinite(zmid[3]) and np.isnan(z0[3]) and np.isnan(z0[4])


def test_variants_scored_but_never_in_the_common_pixels():
    ns = {"np": np, "METHODS": ["A", "B"], "truth": np.linspace(-1.0, 1.0, 100)}
    exec(cell_with("def scores("), ns)
    exec(cell_with("def score_window("), ns)
    z = ns["truth"].copy()
    variant = z + 0.1
    variant[:60] = np.nan                            # would shrink the common pixels to 40 if it entered them
    est = {"A": z, "B": z + 0.2, "A_var": variant}
    rows = ns["score_window"]({"px": np.arange(100)}, [{"method": m} for m in ("A", "B", "A_var")], est)
    a, b, v = rows
    assert a["common_share"] == 1.0 and a["common_n_px"] == 100 and b["common_n_px"] == 100
    assert v["n_px"] == 40 and np.isclose(v["bias"], 0.1)
    assert not any(k.startswith("common") for k in v), "a variant entered the common pixels"


def test_every_default_variant_has_a_run_function():
    first, dispatch = code_cells(SITE_NB)[0], cell_with("DIAGNOSTICS = {")
    default = first.split('os.environ.get("NB_VARIANTS", "')[1].split('"')[0].split(",")
    assert default == ["MAREA_nodelay", "MAREA_ndwi0", "E4_delay", "E5c_delay", "DEA_delay"]
    for v in default:
        assert f'"{v}":' in dispatch, f"{v} has no run function"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
