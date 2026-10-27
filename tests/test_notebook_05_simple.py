# -*- coding: utf-8 -*-
"""The attribution notebook (notebooks/validation/05_error_attribution_site.ipynb) is a plain notebook.

* it is what notebooks/validation/_build_05_attribution.py builds: valid, without outputs, every code cell compiles;
* no cell mentions the field survey of the Spanish site;
* the site comes from the Parameters cell (NB_SITE); the windows are n = 60 and 50 (Ems: 50 and 40); the output
  folder is results/attribution_nb; the sections are numbered 0..8;
* notebook 01's construction is reused: its data cells and the functions named below are in notebook 05 verbatim;
* V0's fit (marea_fit): its midpoint is marea.invert_series' z and its NDWI = 0 crossing is notebook 01's
  invert_ndwi0, bit for bit;
* MAREA-B (invert_bernoulli) is the brute-force maximum of the probit likelihood on its grid, takes the median of the
  plateau under complete separation, and keeps its guards (8 clear looks, one wet and one dry);
* the error decomposition and the strata's MSE shares add up; the gallery picks the extremes;
* every default estimate has a run path.

The notebook is not executed here: it needs the server's data (run_05_on_server.sh).

Run:  python -m pytest -q tests/test_notebook_05_simple.py   (or python tests/test_notebook_05_simple.py)
"""
import ast
import importlib.util
import os
import sys

import nbformat
import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr, ndtri

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks", "validation")
NB = os.path.join(NB_DIR, "05_error_attribution_site.ipynb")
NB01 = os.path.join(NB_DIR, "01_fixed_config_n_tables_site.ipynb")
BUILDER = os.path.join(NB_DIR, "_build_05_attribution.py")
ENV = ("NB_SITE", "NB_NS", "NB_METHODS", "NB_EXTRA", "NB_OUT", "NB_NTABLES", "NB_FIRST", "NB_BED", "NB_ROW_YEARS")
sys.path.insert(0, ROOT)


def cells(path=NB):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    return nb.cells


def code_cells(path=NB):
    return [c.source for c in cells(path) if c.cell_type == "code"]


def function_source(name, path=NB):
    for src in code_cells(path):
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(src, node)
    raise LookupError(name)


def notebook_function(name, path=NB, **names):
    """A function defined in a code cell of a notebook, compiled alone with the given globals."""
    ns = dict(np=np, **names)
    exec(compile(function_source(name, path), f"{os.path.basename(path)}:{name}", "exec"), ns)
    return ns[name]


def parameters(env=None):
    """The Parameters cell and the site cell run with a clean environment (plus `env`)."""
    saved = {k: os.environ.pop(k) for k in ENV if k in os.environ}
    try:
        os.environ.update(env or {})
        ns = {}
        first, sites = code_cells()[:2]
        exec(compile(first, "parameters", "exec"), ns)
        exec(compile(sites, "sites", "exec"), ns)
        return ns
    finally:
        for k in ENV:
            os.environ.pop(k, None)
        os.environ.update(saved)


def marea_module():
    from pyintertidal import marea
    from pyintertidal.elevation import _fit_block
    return marea, _fit_block


def bernoulli():
    marea, _ = marea_module()
    return notebook_function("invert_bernoulli", marea=marea, log_ndtr=log_ndtr, B_MIN_LOOKS=8, B_TIE_TOL=1e-9)


# ── the notebook ────────────────────────────────────────────────────────────

def test_is_the_builders_notebook():
    spec = importlib.util.spec_from_file_location("_build_05_attribution", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert nbformat.writes(mod.build()) + "\n" == open(NB, encoding="utf-8").read(), \
        "the notebook differs from what _build_05_attribution.py builds: rebuild it"


def test_valid_clean_and_compiles():
    for k, c in enumerate(cells()):
        if c.cell_type == "code":
            assert not c.get("outputs") and c.get("execution_count") is None, "the notebook carries outputs"
            compile(c.source, f"cell {k}", "exec")


def test_only_the_vaklodingen_is_read():
    for c in cells():
        assert "rtk" not in c.source.lower(), "a cell mentions the field survey of the Spanish site"


def test_parameters_and_sections():
    assert 'os.environ.get("NB_SITE"' in code_cells()[0]
    ns = parameters()
    assert ns["SITE"] == "escalda" and ns["NS"] == [60, 50] and ns["ROW_YEARS"] == [2025]
    assert ns["OUT"] == "results/attribution_nb"
    assert ns["NTABLES"] == "results/fixedcfg_nb_diag" and ns["FIRST"] == "results/fixedcfg_nb"
    assert ns["METHODS"] == ["MAREA", "MAREA_ndwi0", "MAREA_B", "MAREA_B_nodelay", "MAREA_nodelay", "E4", "E4_delay",
                             "E5c", "DEA", "NIDEM", "E5a"]
    assert parameters({"NB_SITE": "wadden"})["NS"] == [60, 50]
    assert parameters({"NB_SITE": "ems"})["NS"] == [50, 40], "60 images are not reachable at the Ems"
    assert parameters({"NB_SITE": "ems", "NB_NS": "40"})["NS"] == [40]
    assert ns["MAX_PX"] == 200_000 and ns["SEED"] == 20261019 and ns["RANGE_PAD_M"] == 0.25
    assert ns["MAREA_MIN_BRACKET"] == 0
    headings = [c.source.splitlines()[0] for c in cells() if c.cell_type == "markdown" and c.source.startswith("## ")]
    numbers = [h.split("·")[0].strip("# ").strip() for h in headings if "·" in h]
    assert numbers == [str(k) for k in range(9)], f"sections not numbered 0..8: {numbers}"


def test_notebook_01_construction_is_reused_verbatim():
    mine = code_cells()
    for start in ("aoi = pit.sites.get(SITE)", "cube = SentinelCube(", "t0 = time.perf_counter()\nreference_map",
                  "boundary = make_boundary(", "found = sorted(glob.glob(", "ex = np.load(P[\"extract\"])",
                  "seeds = pit.geometry.mouth_seeds(", "import pyproj", "path, key = P[\"nodelay\"]",
                  "shifted = {tau:", "LS = None"):
        theirs = next(s for s in code_cells(NB01) if s.startswith(start))
        assert theirs in mine, f"notebook 01's cell '{start[:30]}' is not in notebook 05 verbatim"
    for name in ("marea_delays", "lags_text", "level_at", "make_obs", "scores", "vaklodingen", "run_marea_nodelay",
                 "read_frames", "run_nidem", "run_dea"):
        assert function_source(name) == function_source(name, NB01), f"{name} differs from notebook 01's"


def test_calls_the_method_functions_directly():
    src = "\n".join(code_cells())
    for call in ("marea.invert_series(", "te.m2a_rasch(", "fm.METHODS[mid][1](", "dea.run(", "wc.nidem(",
                 "make_boundary(", "pit.reference_and_clouds(", "sample_truth(", "_fit_block("):
        assert call in src, call


def test_every_default_estimate_has_a_run_path():
    dispatch = function_source("run_method")
    for mid in parameters()["METHODS"]:
        assert f'"{mid}"' in dispatch or mid in ("E4", "E5c", "E5a"), f"{mid} has no run path"
    assert "check_exact=True" in code_cells()[-1], "the last cell is the bit-for-bit check"


# ── V0's fit and V1 ─────────────────────────────────────────────────────────

def synthetic_curves(P=9000, T=60, seed=3):
    """NDWI series a + b Phi((h - z)/sigma) + noise, a few pixels the guards reject and a few that never cross 0."""
    rng = np.random.default_rng(seed)
    h = np.sort(rng.uniform(-2.0, 2.0, T))[rng.permutation(T)]
    a, b = rng.uniform(-0.5, -0.1, P), rng.uniform(0.3, 1.0, P)
    a[:50], b[:50] = 0.05, 0.5
    z, sg = rng.uniform(-1.5, 1.5, P), rng.uniform(0.05, 0.4, P)
    Y = a + b * ndtr((h[:, None] - z) / sg) + rng.normal(0.0, 0.05, (T, P))
    C = rng.random((T, P)) > 0.15
    C[:, 50:60] = False
    return Y.astype(np.float32), C, h


def test_marea_fit_is_invert_series_and_notebook_01s_crossing():
    marea, _fit_block = marea_module()
    fit = notebook_function("marea_fit", marea=marea, _fit_block=_fit_block, ndtri=ndtri)
    ndwi0 = notebook_function("invert_ndwi0", NB01, marea=marea, _fit_block=_fit_block, ndtri=ndtri)
    Y, C, h = synthetic_curves()
    for mb in (0, 3):
        out = fit(Y, C, h, h.min(), h.max(), min_bracket=mb)
        zm, sm = marea.invert_series(Y, C, h, h.min(), h.max(), min_bracket=mb)
        z0, zmid = ndwi0(Y, C, h, h.min(), h.max(), min_bracket=mb)
        assert np.array_equal(out["z_mid"], zm, equal_nan=True), "the midpoint is not invert_series' z"
        assert np.array_equal(out["z0"], z0, equal_nan=True), "the crossing is not notebook 01's"
        ok = np.isfinite(zm)
        assert np.array_equal(out["sigma"][ok], sm[ok]), "sigma is not invert_series' sigma"
        assert np.all(np.isfinite(out["a"])) and np.all(np.isfinite(out["b"]))


# ── V2 MAREA-B ──────────────────────────────────────────────────────────────

def brute_force(wet, clear, h, lo, hi, sg_grid, tie_tol=1e-9, min_looks=8):
    """(z, sigma) of MAREA-B for one pixel, by loops (independent of the notebook's code)."""
    wet = wet & clear
    if clear.sum() < min_looks or not wet.any() or not (clear & ~wet).any():
        return np.nan, np.nan
    grid = np.linspace(lo, hi, 50)
    ll = np.array([[sum(float(log_ndtr((h[k] - zk) / sk if wet[k] else -(h[k] - zk) / sk))
                        for k in range(len(h)) if clear[k]) for sk in sg_grid] for zk in grid])
    prof = ll.max(axis=1)
    tied = np.flatnonzero(prof >= prof.max() - tie_tol)
    z = float(np.median(grid[tied]))
    near = tied[np.argmin(np.abs(grid[tied] - z))]
    return z, float(sg_grid[int(np.argmax(ll[near]))])


def test_marea_b_is_the_brute_force_maximum():
    marea, _ = marea_module()
    inv = bernoulli()
    rng = np.random.default_rng(7)
    T, P = 40, 12
    h = rng.uniform(-1.8, 1.8, T)
    z_true, s_true = rng.uniform(-1, 1, P), rng.uniform(0.05, 0.5, P)
    wet = rng.random((T, P)) < ndtr((h[:, None] - z_true) / s_true)
    Y = np.where(wet, 0.3, -0.25).astype(np.float32)
    Y[rng.random((T, P)) < 0.05] = np.nan                # undefined NDWI: a dry look
    C = rng.random((T, P)) > 0.2
    z, sg, nt = inv(Y, C, h, h.min(), h.max(), chunk=5)  # several chunks
    for j in range(P):
        with np.errstate(invalid="ignore"):
            zb, sb = brute_force(Y[:, j] > 0, C[:, j], h, h.min(), h.max(), marea.SG_GRID)
        assert np.isclose(z[j], zb, rtol=0, atol=1e-12, equal_nan=True), (j, z[j], zb)
        assert np.isclose(sg[j], sb, rtol=0, atol=0, equal_nan=True), (j, sg[j], sb)


def test_marea_b_separation_takes_the_middle_of_the_plateau():
    inv = bernoulli()
    h = np.concatenate([np.linspace(-2.0, -0.3, 30), np.linspace(0.5, 2.0, 30)])
    Y = np.where(h > 0, 0.4, -0.3)[:, None]
    z, sg, nt = inv(Y, np.ones_like(Y, bool), h, h.min(), h.max())
    step = (h.max() - h.min()) / 49
    assert abs(z[0] - 0.1) <= step / 2 + 1e-12, z      # the gap is [-0.3, 0.5]
    assert nt[0] >= 3 and sg[0] == 0.03


def test_marea_b_guards():
    inv = bernoulli()
    h = np.linspace(-1.5, 1.5, 30)
    Y = np.where(h > 0, 0.4, -0.3)[:, None].repeat(5, axis=1)
    C = np.ones_like(Y, bool)
    C[:, 0] = np.arange(30) % 4 == 0                     # 8 clear looks, wet and dry: a value
    C[:, 1] = np.arange(30) % 5 == 0                     # 6 clear looks: none
    Y[:, 2] = 0.4                                        # every look wet: none
    Y[:, 3] = -0.3                                       # every look dry: none
    Y[:, 4] = np.where(h > 0, 0.4, np.nan)               # dry looks with an undefined NDWI: a value
    z, sg, nt = inv(Y, C, h, h.min(), h.max())
    assert np.isfinite(z[0]) and np.isnan(z[1]) and np.isnan(z[2]) and np.isnan(z[3]) and np.isfinite(z[4])
    assert np.all(nt[1:4] == 0) and np.all(np.isnan(sg[1:4]))


def test_marea_b_delay_shifts_the_levels():
    """A pixel whose water arrives tau later: read at h(t - tau) (here h shifted by +0.3 m) its z moves by 0.3 m."""
    inv = bernoulli()
    rng = np.random.default_rng(2)
    h = rng.uniform(-1.5, 1.5, 50)
    Y = np.where(h > 0.2, 0.4, -0.3)[:, None]
    lo, hi = -2.0, 2.0
    z0 = inv(Y, np.ones_like(Y, bool), h, lo, hi)[0][0]
    z1 = inv(Y, np.ones_like(Y, bool), h + 0.3, lo, hi)[0][0]
    assert abs((z1 - z0) - 0.3) <= (hi - lo) / 49 + 1e-12


# ── the analysis ────────────────────────────────────────────────────────────

def analysis(name):
    scores = notebook_function("scores")
    return notebook_function(name, scores=scores, pd=pd, Q_BINS=5)


def test_decomposition_adds_up():
    decompose = analysis("decompose")
    rng = np.random.default_rng(4)
    t = rng.uniform(-1, 1, 500)
    d = decompose(0.5 * t + 0.2, t)
    assert np.isclose(d["slope"], 0.5) and np.isclose(d["relief"], 0.5 * t.std()) and d["scatter"] < 1e-12
    assert np.isclose(d["mean_bias"], np.mean(0.2 - 0.5 * t))
    p = 0.8 * t - 0.1 + rng.normal(0, 0.2, 500)
    p[:20] = np.nan
    d = decompose(p, t)
    assert np.isclose(d["mse"], d["mean_bias"] ** 2 + d["relief"] ** 2 + d["scatter"] ** 2, rtol=1e-12)
    assert np.isclose(d["rmse"] ** 2, d["mse"]) and d["n_px"] == 480


def test_strata_shares_add_up_and_bins():
    rmse = notebook_function("rmse")
    bins_of = notebook_function("bins_of", pd=pd, Q_BINS=5)
    strata = notebook_function("strata", pd=pd, Q_BINS=5, rmse=rmse, bins_of=bins_of)
    rng = np.random.default_rng(9)
    n = 1000
    err = {"A": rng.normal(0.1, 0.3, n), "B": rng.normal(0, 0.2, n)}
    cerr = {k: e - np.median(e) for k, e in err.items()}
    x = rng.uniform(0, 30, n)
    rows = strata(err, cerr, x, {"A_B": ("A", "B")})
    assert len(rows) == 5 and sum(r["n_px"] for r in rows) == n
    assert np.isclose(sum(r["d_mse_share_A_B"] for r in rows), np.mean(err["A"] ** 2) - np.mean(err["B"] ** 2))
    assert np.isclose(sum(r["share"] * r["crms_A"] ** 2 for r in rows), np.mean(cerr["A"] ** 2))
    assert [lab for lab, _ in bins_of(np.array([0.0, 0, 5, 5, 9]))] == ["0", "5", "9"]
    cats = bins_of(np.array(["stable", "changing", "stable"]))
    assert [lab for lab, _ in cats] == ["changing", "stable"] and cats[1][1].sum() == 2


def test_gallery_picks_the_extremes():
    pick = notebook_function("pick_gallery")
    d = np.array([0.1, 0.9, -0.5, 0.3, 0.9, -0.8, 0.0])
    worst, best = pick(d, 3, 2)
    assert worst.tolist() == [1, 4, 3] and best.tolist() == [5, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
