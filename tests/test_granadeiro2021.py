"""Tests of pyintertidal.granadeiro2021 (Granadeiro et al. 2021, Remote Sens.
13:320) and of its R side (pyintertidal/r/granadeiro_fit.R).

* Eq. 1 against the co-author's R line (Map_InterSedim_Bijagos,
  DEM_based_intertidalmask_creation.R), evaluated in R;
* tide-table HW/LW marks on a synthetic tide with ripples and a plateau;
* a synthetic flat with known elevations (the logistic recovers them) and a
  known tidal-stage lag (the rising/ebbing search recovers it);
* the lean nplr path equals nplr() + getInflexion (values to 1e-12 and
  identical failures), on synthetic pixels and, when a v8 run exists, on the
  real Villaviciosa proof fixture;
* the closed-form major axis equals lmodel2's MA row to 1e-10.

Run:  python -X utf8 -m pytest tests/test_granadeiro2021.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyintertidal import granadeiro2021 as g21  # noqa: E402

FIXTURE = os.path.join(ROOT, "products_marea", "granadeiro2021", "granadeiro2021.npz")


def _have_r():
    try:
        g21.find_rscript()
        return True
    except FileNotFoundError:
        return False


needs_r = pytest.mark.skipif(not _have_r(), reason="Rscript not found")


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    if not _have_r():
        pytest.skip("Rscript not found")
    r = g21.RRunner(workers=4, tmpdir=str(tmp_path_factory.mktemp("g21r")))
    yield r
    r.close()
    assert not os.path.exists(r.root)


def synthetic_marks(days=40, seed=0):
    """Marks of a clean synthetic semidiurnal tide sampled every minute."""
    t = np.arange(0, days * 1440, 1.0)
    h = (1.4 * np.cos(2 * np.pi * t / 745.2) + 0.45 * np.cos(2 * np.pi * t / 720.0 + 0.7)
         + 0.12 * np.cos(2 * np.pi * t / 372.6 + 1.1))
    return t, h, g21.tide_marks(t, h)


# ─────────────────────────────────────────────────────────────────────────────
#  Eq. 1
# ─────────────────────────────────────────────────────────────────────────────

@needs_r
def test_eq1_matches_coauthor_r(runner):
    # the co-author's two worked examples (minutes of the day)
    t_sat = np.array([11 * 60 + 22, 19 * 60 + 17], float)
    t_lw = np.array([11 * 60 + 58, 18 * 60 + 32], float)
    t_hw = np.array([5 * 60 + 45, 24 * 60 + 54], float)
    h_lw = np.array([1.5, 1.0])
    h_hw = np.array([3.5, 4.2])
    rng = np.random.default_rng(1)
    n = 2000
    # random rising (TLW < T < THW) and ebbing (THW < T < TLW) brackets
    a = rng.uniform(0, 1e5, n)
    dur = rng.uniform(300, 450, n)
    rising = rng.random(n) < 0.5
    tl = np.where(rising, a, a + dur)
    th = np.where(rising, a + dur, a)
    ts = a + rng.uniform(0, 1, n) * dur
    hl, hh = rng.uniform(-3, 1, n), rng.uniform(1.2, 4, n)
    T_sat = np.concatenate([t_sat, ts])
    T_lw = np.concatenate([t_lw, tl])
    T_hw = np.concatenate([t_hw, th])
    H_lw = np.concatenate([h_lw, hl])
    H_hw = np.concatenate([h_hw, hh])
    h_r = g21.r_eq1(runner, T_sat, T_lw, T_hw, H_lw, H_hw)
    h_py = g21.eq1(T_sat, T_hw, H_hw, T_lw, H_lw)
    assert np.max(np.abs(h_r - h_py)) <= 1e-12
    # endpoints: the low- and high-water marks themselves
    assert np.allclose(g21.eq1(T_lw, T_hw, H_hw, T_lw, H_lw), H_lw, atol=1e-12)
    assert np.allclose(g21.eq1(T_hw, T_hw, H_hw, T_lw, H_lw), H_hw, atol=1e-12)


def test_water_height_bracket_and_direction():
    t, h, marks = synthetic_marks(10)
    ts = np.linspace(2000, 12000, 500)
    hs, br = g21.water_height(ts, marks)
    # bracket: previous mark <= t < next mark, and the direction of the pair
    k = br["k"]
    assert np.all(marks["t"][k] <= ts) and np.all(ts < marks["t"][k + 1])
    assert np.array_equal(br["rising"], ~marks["is_hw"][k])
    assert np.all(hs <= br["h_hw"] + 1e-12) and np.all(hs >= br["h_lw"] - 1e-12)
    # heights follow the true curve (the cosine between marks misses the
    # M4 / S2 asymmetry of this synthetic tide by up to ~0.3 m: Eq. 1 itself)
    err = np.abs(hs - np.interp(ts, t, h))
    assert np.max(err) < 0.4 and np.median(err) < 0.1
    # at the marks the interpolant returns the marks' heights
    hm, _ = g21.water_height(marks["t"][2:-2], marks)
    assert np.allclose(hm, marks["h"][2:-2], atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
#  HW / LW marks
# ─────────────────────────────────────────────────────────────────────────────

def test_tide_marks_synthetic_with_ripples_and_plateau():
    t, h_clean, clean = synthetic_marks(40)
    # the clean tide: one HW and one LW per half-cycle, ~12.42 h cycle
    assert clean["n_pairs_removed"] == 0
    assert clean["n_not_global_extreme"] == 0
    cyc = np.diff(clean["t"][clean["is_hw"]]) / 60
    assert 11.5 < np.median(cyc) < 13.5
    # add a ripple (6 cm, 30-min period) -> many spurious turning points,
    # and a plateau (constant stretch) around one high water
    h = h_clean + 0.06 * np.sin(2 * np.pi * t / 30.0)
    j = int(clean["sample_index"][np.flatnonzero(clean["is_hw"])[10]])
    h[j - 3:j + 4] = h[j - 3:j + 4].max() + 0.2
    m = g21.tide_marks(t, h)
    assert m["n_turning_points"] > 3 * clean["n_turning_points"]
    assert m["n_pairs_removed"] > 0
    # alternation, >= 3 h everywhere
    assert np.all(m["is_hw"][1:] != m["is_hw"][:-1])
    assert np.min(np.diff(m["t"])) >= 180.0
    # away from the ends of the finite series (a ripple crest in the first
    # partial half-cycle is a legitimate first mark; the v8 series is padded
    # by 2 days so such edge marks never bracket a scene), the marks are
    # those of the clean tide: same count and types, each the global
    # extreme of its half-cycle, within 1 h of the clean extreme
    inner = lambda mk: (mk["t"] > 800) & (mk["t"] < t[-1] - 800)  # noqa: E731
    mi, ci = inner(m), inner(clean)
    assert mi.sum() == ci.sum()
    assert np.array_equal(m["is_hw"][mi], clean["is_hw"][ci])
    assert np.max(np.abs(m["t"][mi] - clean["t"][ci])) < 60.0
    assert m["n_not_global_extreme"] == 0
    # the plateau gives ONE mark, timed at the plateau centre
    i = int(np.argmin(np.abs(m["t"] - t[j])))
    assert m["is_hw"][i] and m["t"][i] == t[j]
    assert m["n_plateau_turns"] >= 1


def test_tide_marks_rejects_unsorted():
    with pytest.raises(ValueError):
        g21.tide_marks(np.array([0.0, 2.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0, 1.0]))


# ─────────────────────────────────────────────────────────────────────────────
#  synthetic flat: known elevations and a known lag
# ─────────────────────────────────────────────────────────────────────────────

def _flat(marks, t_scene, z, lag_min, rng, noise=0.01):
    """NIR of pixels with elevations ``z``: land (0.25) when the local water
    (the reference tide lagged by ``lag_min``) is below the pixel, water
    (0.01) otherwise."""
    h_loc, _ = g21.water_height(t_scene - lag_min, marks)
    nir = np.where(h_loc[None, :] < z[:, None], 0.25, 0.01)
    return nir + rng.normal(0, noise, nir.shape)


@needs_r
def test_synthetic_flat_known_elevations(runner):
    rng = np.random.default_rng(2)
    t, h, marks = synthetic_marks(60, seed=2)
    t_scene = np.sort(rng.uniform(2000, 60 * 1440 - 2000, 90))
    h_scene, _ = g21.water_height(t_scene, marks)
    z = rng.uniform(np.percentile(h_scene, 15), np.percentile(h_scene, 85), 60)
    nir = _flat(marks, t_scene, z, 0.0, rng)
    nir[rng.random(nir.shape) < 0.05] = np.nan            # no data, dropped as NAs
    rho = g21.convert_to_prop(nir, axis=1)
    fit = g21.fit_logistic(runner, rho, x=h_scene, chunk=15)
    ok = fit[:, 0] == g21.FIT_OK
    assert ok.mean() > 0.9
    # the inflection lies between the scene heights that bracket z
    gap = np.array([np.min(np.abs(h_scene - zz)) for zz in z])
    err = np.abs(fit[ok, 1] - z[ok])
    assert np.median(err) < 0.1 and np.all(err < 0.15 + 2 * gap[ok])


@needs_r
def test_synthetic_flat_known_lag(runner):
    """A flat whose local tide runs 30 min late: the rising/ebbing search
    recovers +30 min (positive lag = later local tide, h = Eq1(t - lag))."""
    rng = np.random.default_rng(3)
    t, h, marks = synthetic_marks(80, seed=3)
    t_scene = np.sort(rng.uniform(2000, 80 * 1440 - 2000, 120))
    h0, br = g21.water_height(t_scene, marks)
    rising = br["rising"]
    z = rng.uniform(np.percentile(h0, 30), np.percentile(h0, 70), 16)
    nir = _flat(marks, t_scene, z, 30.0, rng, noise=0.005)
    rho = g21.convert_to_prop(nir, axis=1)
    H_lag = g21.lag_heights(t_scene, marks)
    st, inf = g21.lag_fits(runner, rho, H_lag, rising, chunk=4)
    lag, gap, idx = g21.choose_lags(st, inf)
    assert np.isfinite(lag).mean() > 0.8
    assert abs(np.nanmedian(lag) - 30.0) <= 10.0


def test_choose_lags_which_min_semantics():
    grid = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    st = np.zeros((3, 5, 2))
    inf = np.zeros((3, 5, 2))
    # pixel 0: ties at lags 1 and 3 -> the first (which.min)
    inf[0, :, 0] = [1.0, 0.2, 0.5, 0.2, 0.9]
    # pixel 1: a failed fit at the true minimum is skipped (NA in R)
    inf[1, :, 0] = [1.0, 0.8, 0.0, 0.3, 0.9]
    st[1, 2, 1] = g21.FIT_NLM_NO_ITER
    # pixel 2: every lag has a failed limb -> no lag
    st[2, :, 0] = g21.FIT_ERROR
    lag, gap, idx = g21.choose_lags(st, inf, grid)
    assert lag[0] == -5.0 and idx[0] == 1
    assert lag[1] == 5.0
    assert np.isnan(lag[2]) and idx[2] == -1


# ─────────────────────────────────────────────────────────────────────────────
#  lean nplr path == nplr()
# ─────────────────────────────────────────────────────────────────────────────

@needs_r
def test_lean_equals_nplr_synthetic(runner):
    rng = np.random.default_rng(4)
    T, n = 97, 300
    x = rng.uniform(-2.5, 1.5, T)
    z = rng.uniform(-2.8, 1.8, n)
    y = np.where(x[None, :] < z[:, None], 0.3, 0.02) + rng.normal(0, 0.03, (n, T))
    y[rng.random((n, T)) < 0.04] = np.nan
    y[:3] = 0.1                                   # constant -> failure
    y[3:6] = np.nan                               # no data -> failure
    y[6:9, 2:] = np.nan                           # two points only
    y[9:40] = rng.normal(0.1, 0.05, (31, T))      # noise pixels
    lean = g21.fit_logistic(runner, g21.convert_to_prop(y, axis=1), x=x, chunk=50)
    ref = g21.fit_logistic(runner, y, x=x, reference=True, prop=True, chunk=50)
    assert np.array_equal(lean[:, 0], ref[:, 0])
    assert (lean[:, 0] != 0).sum() >= 6
    ok = lean[:, 0] == 0
    assert np.max(np.abs(lean[ok, 1] - ref[ok, 1])) <= 1e-12
    # per-pixel heights (the final-DEM layout)
    X = x[None, :] + rng.normal(0, 0.1, (n, T))
    lean2 = g21.fit_logistic(runner, g21.convert_to_prop(y, axis=1), x_fn=lambda a, b: X[a:b], chunk=60)
    ref2 = g21.fit_logistic(runner, y, x_fn=lambda a, b: X[a:b], reference=True, prop=True, chunk=60)
    assert np.array_equal(lean2[:, 0], ref2[:, 0])
    ok2 = lean2[:, 0] == 0
    assert np.max(np.abs(lean2[ok2, 1] - ref2[ok2, 1])) <= 1e-12
    # lag layout (rising / ebbing subsets at every lag)
    H = x[None, :] + np.linspace(-0.3, 0.3, 5)[:, None]
    rising = rng.random(T) < 0.5
    rho = g21.convert_to_prop(y[:30], axis=1)
    s1, i1 = g21.lag_fits(runner, rho, H, rising, chunk=8)
    s2, i2 = g21.lag_fits(runner, rho, H, rising, reference=True, chunk=8)
    assert np.array_equal(s1, s2)
    both = (s1 == 0)
    assert np.max(np.abs(i1[both] - i2[both])) <= 1e-12


@needs_r
@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="no v8 Villaviciosa run yet")
def test_lean_equals_nplr_real_fixture(runner):
    z = np.load(FIXTURE)
    x = z["proof_fixture_x"]
    nir = z["proof_fixture_nir_cal"]
    rho = z["proof_fixture_rho"]
    assert np.array_equal(g21.convert_to_prop(nir, axis=1), rho, equal_nan=True)
    lean = g21.fit_logistic(runner, rho, x=x, chunk=75)
    ref = g21.fit_logistic(runner, nir, x=x, reference=True, prop=True, chunk=75)
    assert np.array_equal(lean[:, 0], ref[:, 0])
    ok = lean[:, 0] == 0
    assert np.max(np.abs(lean[ok, 1] - ref[ok, 1])) <= 1e-12
    # and both equal what the run stored
    assert np.array_equal(lean[:, :2], z["proof_fixture_lean"], equal_nan=True)
    assert np.array_equal(ref[:, :2], z["proof_fixture_nplr"], equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────────
#  major axis == lmodel2
# ─────────────────────────────────────────────────────────────────────────────

@needs_r
def test_major_axis_equals_lmodel2(runner):
    rng = np.random.default_rng(5)
    n = 30000
    x = np.concatenate([rng.normal(0.02, 0.01, n // 2), rng.normal(0.3, 0.05, n // 2)])
    x[rng.random(n) < 0.01] = np.nan
    cases = [0.01 + 1.05 * x + rng.normal(0, 0.005, n),
             -0.02 + 0.8 * x + rng.normal(0, 0.02, n),
             0.5 - 0.7 * x + rng.normal(0, 0.03, n),
             2.0 * x + rng.normal(0, 0.1, n)]
    cases[1][rng.random(n) < 0.02] = np.nan
    rows = g21.lmodel2_rows(runner, x, lambda k: cases[k], len(cases), jobs_per_process=2)
    for k, y in enumerate(cases):
        b0, b1, nn = g21.major_axis(x, y)
        assert nn == rows[k, 0]
        assert abs(b0 - rows[k, 3]) <= 1e-10 and abs(b1 - rows[k, 4]) <= 1e-10
        # MA differs from OLS and SMA (the right row is read)
        assert abs(b1 - rows[k, 2]) > 1e-6 and abs(b1 - rows[k, 6]) > 1e-6


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="no v8 Villaviciosa run yet")
def test_stored_calibration_equals_lmodel2():
    z = np.load(FIXTURE)
    ref = int(z["reference_pos"])
    for band in ("B03", "B08"):
        coef = np.delete(z[f"cal_{band}"], ref, axis=0)
        rows = z[f"lmodel2_{band}"]
        assert np.max(np.abs(coef[:, 0] - rows[:, 3])) <= 1e-10
        assert np.max(np.abs(coef[:, 1] - rows[:, 4])) <= 1e-10


# ─────────────────────────────────────────────────────────────────────────────
#  small pieces
# ─────────────────────────────────────────────────────────────────────────────

def test_scene_rule_thresholds():
    cloud = np.array([0.0999, 0.10, 0.05, 0.0])
    nodata = np.array([0.0, 0.0, 0.0201, 0.02])
    use = g21.scene_rule(cloud, nodata, np.ones(4, bool), np.ones(4, bool))
    assert use.tolist() == [True, False, False, True]
    scl = np.full((10, 10), 4, np.int16)
    scl[0, :9] = 9
    scl[1, 0] = -32768
    c, nd = g21.scene_frame_shares(scl, fill=-32768)
    assert c == 0.09 and nd == 0.01


def test_exposure_eq5():
    e = g21.exposure_eq5(np.array([1.05, 3.90, 2.475, 0.5, 4.5]), 3.90, 1.05, 12.40)
    assert e[0] == 0.0 and abs(e[1] - 12.40) < 1e-12 and abs(e[2] - 6.20) < 1e-12
    assert np.isnan(e[3]) and np.isnan(e[4])


def test_convert_to_prop_and_calibrate():
    y = np.array([[0.1, np.nan, 0.3, 0.2]])
    p = g21.convert_to_prop(y, axis=1)
    assert np.allclose(p, [[0.0, np.nan, 1.0, 0.5]], equal_nan=True)
    x = np.array([[1.0, 2.0], [3.0, 5.0]])
    c = np.array([[0.0, 1.0], [1.0, 2.0]])
    assert np.array_equal(g21.calibrate(x, c), [[1.0, 2.0], [1.0, 2.0]])


def test_reference_scene_and_calibration_pixels():
    rg = np.array([0.10, 0.20, 0.15, 0.30])
    rn = np.array([0.30, 0.28, 0.31, 0.50])
    i, score = g21.reference_scene(rg, rn)
    assert i == 2
    nir = np.array([0.01, 0.049, 0.05, 0.1, 0.2, 0.21, np.nan, 0.3])
    excl = np.zeros(8, bool)
    excl[7] = True
    idx, n_sea, n_land = g21.calibration_pixels(nir, excl)
    assert idx.tolist() == [0, 1, 5] and (n_sea, n_land) == (2, 1)
