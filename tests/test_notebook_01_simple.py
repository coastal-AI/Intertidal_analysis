"""The n-tables notebooks (notebooks/validation/01_*.ipynb) are plain notebooks.

* both are valid notebooks and every code cell compiles;
* no code cell reads the RTK survey (none mentions it);
* the site comes from the Parameters cell, overridable by NB_SITE;
* the scoring function of the site notebook gives the right numbers on tiny synthetic cases.

Run:  python -m pytest tests/test_notebook_01_simple.py   (or python tests/test_notebook_01_simple.py)
"""
import os

import nbformat
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_NB = os.path.join(ROOT, "notebooks", "validation", "01_fixed_config_n_tables_site.ipynb")
SUMMARY_NB = os.path.join(ROOT, "notebooks", "validation", "01b_fixed_config_n_tables_summary.ipynb")


def code_cells(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def scores_function():
    src = next(s for s in code_cells(SITE_NB) if "def scores(" in s)
    ns = {"np": np}
    exec(src, ns)
    return ns["scores"]


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
    for name in ("NB_NS", "NB_METHODS", "NB_EXTRA"):
        assert name in first


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
