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
* V10, MAREA-B with lapse rates (invert_bernoulli_lapse), is the brute-force maximum over z, sigma and e; with e = 0
  alone it is V2 bit for bit; a few spurious wet looks at low tide do not drag its z; V2's guards;
* V9 (sigma_check): the shares within sigma and 2 sigma and Spearman's correlation of |p - t| with sigma;
* NB_NS runs the confirmatory windows; the last check compares the windows notebook 01's run computed;
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
    assert ns["METHODS"] == ["MAREA", "MAREA_ndwi0", "MAREA_B", "MAREA_BL", "MAREA_B_nodelay", "MAREA_nodelay", "E4",
                             "E4_delay", "E5c", "DEA", "NIDEM", "E5a"]
    assert ns["B_LAPSES"] == (0.0, 0.02, 0.05, 0.10)
    assert parameters({"NB_SITE": "wadden"})["NS"] == [60, 50]
    assert parameters({"NB_SITE": "ems"})["NS"] == [50, 40], "60 images are not reachable at the Ems"
    assert parameters({"NB_SITE": "ems", "NB_NS": "40"})["NS"] == [40]
    for site in ("escalda", "wadden", "ems"):                # the confirmatory windows
        assert parameters({"NB_SITE": site, "NB_NS": "125,100,75"})["NS"] == [125, 100, 75]
    assert ns["MAX_PX"] == 200_000 and ns["SEED"] == 20261019 and ns["RANGE_PAD_M"] == 0.25
    assert ns["MAREA_MIN_BRACKET"] == 0
    headings = [c.source.splitlines()[0] for c in cells() if c.cell_type == "markdown" and c.source.startswith("## ")]
    numbers = [h.split("·")[0].strip("# ").strip() for h in headings if "·" in h]
    assert numbers == [str(k) for k in range(9)], f"sections not numbered 0..8: {numbers}"


def test_notebook_01_construction_is_reused_verbatim():
    mine = code_cells()
    for start in ("aoi = pit.sites.get(SITE)", "cube = SentinelCube(", "t0 = time.perf_counter()\nreference_map",
                  "boundary = make_boundary(", "class SplitBoundary", "found = sorted(glob.glob(",
                  "ex = np.load(P[\"extract\"])", "seeds = pit.geometry.mouth_seeds(", "import pyproj",
                  "path, key = P[\"nodelay\"]", "shifted = {tau:", "LS = None",
                  "if len(x_dates):\n    cut = boundary.cut", "if len(x_dates):\n    ref = side[\"reference_map\"]"):
        theirs = next(s for s in code_cells(NB01) if s.startswith(start))
        assert theirs in mine, f"notebook 01's cell '{start[:30]}' is not in notebook 05 verbatim"
    for name in ("marea_delays", "lags_text", "level_at", "make_obs", "scores", "vaklodingen", "run_marea_nodelay",
                 "read_frames", "zarr_frames", "run_nidem", "run_dea"):
        assert function_source(name) == function_source(name, NB01), f"{name} differs from notebook 01's"
    assert "holds no whole frames" not in function_source("run_window"), "NIDEM must run on the pre-2023 windows"


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


def bernoulli_lapse():
    marea, _ = marea_module()
    return notebook_function("invert_bernoulli_lapse", marea=marea, log_ndtr=log_ndtr, ndtr=ndtr, B_MIN_LOOKS=8,
                             B_TIE_TOL=1e-9, B_LAPSES=(0.0, 0.02, 0.05, 0.10))


def brute_force_lapse(wet, clear, h, lo, hi, sg_grid, lapses=(0.0, 0.02, 0.05, 0.10), tie_tol=1e-9, min_looks=8):
    """(z, sigma, e) of V10 for one pixel, by loops (independent of the notebook's code)."""
    wet = wet & clear
    if clear.sum() < min_looks or not wet.any() or not (clear & ~wet).any():
        return np.nan, np.nan, np.nan

    def log_p(u, e):
        return float(log_ndtr(u)) if e == 0 else float(np.log(e + (1 - 2 * e) * ndtr(u)))

    grid = np.linspace(lo, hi, 50)
    pairs = [(sk, e) for sk in sg_grid for e in lapses]
    ll = np.array([[sum(log_p((h[k] - zk) / sk if wet[k] else -(h[k] - zk) / sk, e) for k in range(len(h)) if clear[k])
                    for sk, e in pairs] for zk in grid])
    prof = ll.max(axis=1)
    tied = np.flatnonzero(prof >= prof.max() - tie_tol)
    z = float(np.median(grid[tied]))
    near = tied[np.argmin(np.abs(grid[tied] - z))]
    sk, e = pairs[int(np.argmax(ll[near]))]
    return z, float(sk), float(e)


def test_marea_bl_is_the_brute_force_maximum():
    marea, _ = marea_module()
    inv = bernoulli_lapse()
    rng = np.random.default_rng(8)
    T, P = 36, 9
    h = rng.uniform(-1.8, 1.8, T)
    z_true, s_true = rng.uniform(-1, 1, P), rng.uniform(0.05, 0.5, P)
    p_wet = 0.05 + 0.9 * ndtr((h[:, None] - z_true) / s_true)        # a 5 % lapse
    Y = np.where(rng.random((T, P)) < p_wet, 0.3, -0.25).astype(np.float32)
    Y[rng.random((T, P)) < 0.05] = np.nan
    C = rng.random((T, P)) > 0.2
    z, sg, e, nt = inv(Y, C, h, h.min(), h.max(), chunk=4)
    for j in range(P):
        with np.errstate(invalid="ignore"):
            zb, sb, eb = brute_force_lapse(Y[:, j] > 0, C[:, j], h, h.min(), h.max(), marea.SG_GRID)
        assert np.isclose(z[j], zb, rtol=0, atol=1e-12, equal_nan=True), (j, z[j], zb)
        assert np.isclose(sg[j], sb, rtol=0, atol=0, equal_nan=True) and np.isclose(e[j], eb, rtol=0, atol=0,
                                                                                     equal_nan=True), (j, sg[j], sb, e[j], eb)


def test_marea_bl_with_no_lapse_is_v2_bit_for_bit():
    inv, inv_bl = bernoulli(), bernoulli_lapse()
    rng = np.random.default_rng(12)
    T, P = 50, 9000                                      # several chunks of 4,000 pixels
    h = rng.uniform(-2, 2, T)
    Y = np.where(rng.random((T, P)) < ndtr((h[:, None] - rng.uniform(-1, 1, P)) / 0.3), 0.3, -0.2)
    C = rng.random((T, P)) > 0.15
    z2, s2, n2 = inv(Y, C, h, h.min(), h.max())
    z0, s0, e0, n0 = inv_bl(Y, C, h, h.min(), h.max(), lapses=(0.0,))
    assert np.array_equal(z0, z2, equal_nan=True) and np.array_equal(s0, s2, equal_nan=True)
    assert np.array_equal(n0, n2) and np.all(e0[np.isfinite(z0)] == 0)


def test_marea_bl_is_not_dragged_by_spurious_looks():
    """A pixel that floods at 0.4 m, with 3 wet looks at the lowest tides: V2 needs a wide sigma or a lower z to
    explain them, V10 puts them on its lapse rate and keeps z where the record switches."""
    inv, inv_bl = bernoulli(), bernoulli_lapse()
    h = np.linspace(-1.8, 1.8, 60)
    Y = np.where(h > 0.4, 0.4, -0.3)[:, None]
    Y[[0, 1, 2], 0] = 0.4                                 # spurious wet looks at the lowest levels
    C = np.ones_like(Y, bool)
    z2 = inv(Y, C, h, h.min(), h.max())[0][0]
    zl, sl, el, _ = (v[0] for v in inv_bl(Y, C, h, h.min(), h.max()))
    step = (h.max() - h.min()) / 49
    assert el > 0 and abs(zl - 0.4) <= step, (zl, el)
    assert abs(z2 - 0.4) > abs(zl - 0.4), (z2, zl)


def test_marea_bl_guards():
    inv_bl = bernoulli_lapse()
    h = np.linspace(-1.5, 1.5, 30)
    Y = np.where(h > 0, 0.4, -0.3)[:, None].repeat(4, axis=1)
    C = np.ones_like(Y, bool)
    C[:, 0] = np.arange(30) % 4 == 0                     # 8 clear looks: a value
    C[:, 1] = np.arange(30) % 5 == 0                     # 6 clear looks: none
    Y[:, 2], Y[:, 3] = 0.4, -0.3                         # every look wet / dry: none
    z, sg, e, nt = inv_bl(Y, C, h, h.min(), h.max())
    assert np.isfinite(z[0]) and np.isfinite(sg[0]) and np.isfinite(e[0])
    assert np.all(np.isnan(z[1:])) and np.all(np.isnan(e[1:])) and np.all(nt[1:] == 0)


def test_sigma_check_v9():
    from scipy.stats import spearmanr
    check = notebook_function("sigma_check", spearmanr=spearmanr)
    rng = np.random.default_rng(3)
    t = rng.uniform(-1, 1, 400)
    sigma = rng.choice([0.03, 0.1, 0.22, 0.45], 400)
    p = t + rng.normal(0, 1, 400) * sigma                # errors that scale with sigma
    p[:10] = np.nan
    sigma[10:15] = np.nan
    out = check(p, t, sigma)
    ok = np.isfinite(p) & np.isfinite(sigma)
    e = np.abs(p[ok] - t[ok])
    assert out["n_px"] == 385
    assert np.isclose(out["within_1sigma"], np.mean(e <= sigma[ok])) and np.isclose(out["within_2sigma"],
                                                                                   np.mean(e <= 2 * sigma[ok]))
    assert 0.55 < out["within_1sigma"] < 0.8 and 0.9 < out["within_2sigma"] <= 1.0
    assert out["spearman_rho"] > 0.4 and np.isclose(out["spearman_rho"], spearmanr(e, sigma[ok])[0])
    assert np.isnan(check(p[:20], t[:20], sigma[:20])["within_1sigma"]), "below 30 pixels: counts only"


def test_last_check_compares_the_windows_notebook_01_computed():
    import tempfile
    from collections import Counter

    def row(site, n, m, status="ok", rmse=0.2):
        return dict(site=site, n=n, method=m, status=status, reason="band lags 0/3 min", first_date="2021-03-01",
                    last_date="2025-12-30", n_F=500, n_px=480, coverage=0.96, rmse=rmse, bias=0.1, crmse=0.15, slope=0.9,
                    r=0.95, available=142)

    with tempfile.TemporaryDirectory() as tmp:
        nt, first, out = (os.path.join(tmp, d) for d in ("nt", "first", "out"))
        for d in (nt, first, out):
            os.makedirs(d)
        ref = [row("s", 125, m) for m in ("MAREA", "E4")] + [row("s", 100, m, "not reachable") for m in ("MAREA", "E4")]
        pd.DataFrame(ref).to_csv(os.path.join(nt, "s.csv"), index=False)
        new = [row("s", n, m) for n in (125, 100) for m in ("MAREA", "MAREA_BL", "E4")]
        pd.DataFrame(new).to_csv(os.path.join(out, "s_scores.csv"), index=False)
        ns = dict(pd=pd, np=np, os=os, Counter=Counter, NTABLES=nt, FIRST=first, OUT=out, SITE="s", NS=[125, 100],
                  METHODS=["MAREA", "MAREA_BL", "E4"], windows=[{"site": "s"}])
        exec(code_cells()[-1], ns)                       # n = 100 was not reachable in notebook 01's run: not compared
        pd.DataFrame([dict(r, rmse=0.3) if r["method"] == "E4" else r for r in new]).to_csv(
            os.path.join(out, "s_scores.csv"), index=False)
        try:
            exec(code_cells()[-1], ns)
            raise AssertionError("a changed score passed")
        except AssertionError as e:
            assert "rmse" in str(e), e


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
