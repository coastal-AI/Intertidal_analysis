"""The n-tables notebooks (notebooks/validation/01_*.ipynb) are plain notebooks.

* both are valid notebooks and every code cell compiles;
* no code cell reads the RTK survey (none mentions it);
* the site comes from the Parameters cell, overridable by NB_SITE;
* the scoring function of the site notebook gives the right numbers on tiny synthetic cases;
* the diagnostic variants: z at NDWI = 0 is MAREA's own fit (its midpoint is marea.invert_series' z, bit for
  bit) read where the fitted curve crosses 0; the variants are scored but never enter the common pixels; every
  default variant has a run function;
* the pre-2023 scenes: SplitBoundary reads the pre-2023 boundary before 2023-01-01 and production's from then on,
  instant by instant; NIDEM reads their whole frames from the Zarr stores into the cube's own NDWI and clear mask
  (a tiny synthetic cube and stores), and the frames check stops on a difference the 2023 gate did not accept;
* the last cell compares the rows of the windows the earlier run computed (the seconds and `available` aside),
  and a site without an earlier run has nothing to compare;
* the sites: the Dutch ones and Villaviciosa as before the Spanish sites were generalised, Ferrol as its
  production notebook (tide_boundary_comparison_ferrol.ipynb) runs it;
* the Spanish sites' truth: NB_LIDAR / NB_LIDAR_DATE (the earlier NB_VVA_LIDAR / NB_VVA_LIDAR_DATE still work),
  the date by default the gridding JSON's truth_date, pending truth without the LiDAR; the flight period of the
  tiles under F.

Run:  python -m pytest tests/test_notebook_01_simple.py   (or python tests/test_notebook_01_simple.py)
"""
import os
import sys
import tempfile
import types
from collections import Counter

import nbformat
import numpy as np
import pandas as pd
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


# ── the pre-2023 scenes: the boundary ─────────────────────────────────────────

class FakeBoundary:
    """levels = a constant plus the minutes since 2022-12-31 00:00, recording every call."""

    def __init__(self, name, offset):
        self.name, self.offset, self.calls = name, offset, []

    def levels(self, times):
        t = pd.DatetimeIndex(times)
        self.calls.append(t)
        return self.offset + (t - pd.Timestamp("2022-12-31")) / pd.Timedelta(minutes=1)


def split_boundary_class():
    ns = {"np": np, "pd": pd}
    exec(cell_with("class SplitBoundary"), ns)
    return ns["SplitBoundary"]


def test_split_boundary_reads_each_side_of_the_cut():
    pre, prod = FakeBoundary("pre", 1000.0), FakeBoundary("prod", 0.0)
    b = split_boundary_class()(pre, prod, "2023-01-01")
    assert b.name == "prod (before 2023-01-01: pre)"
    t = pd.DatetimeIndex(["2022-12-31 22:00", "2022-12-31 23:59", "2023-01-01 00:00", "2023-01-01 03:00"])
    h = b.levels(t)
    assert np.array_equal(h, [1000 + 1320, 1000 + 1439, 1440, 1620]), h
    assert len(pre.calls) == 1 and pre.calls[0].equals(t[:2]), "the pre-2023 boundary is read once, before the cut"
    assert len(prod.calls) == 1 and prod.calls[0].equals(t[2:]), "production's is read once, from the cut on"
    # an instant shifted across the cut is read from the boundary of its own instant
    shifted = pd.DatetimeIndex(["2023-01-01 01:00"]) - pd.Timedelta(minutes=90)
    assert b.levels(shifted)[0] == 1000 + 1410
    pre.calls.clear(), prod.calls.clear()
    assert np.array_equal(b.levels(t[2:]), prod.levels(t[2:])) and not pre.calls, "after the cut: production alone"


def test_every_level_is_read_through_one_boundary_object():
    """The pre-2023 cell replaces `boundary`, and the levels table, level_at, the bank and the 1-min series read it."""
    found = cell_with("found = sorted(glob.glob(")
    assert "lte.pre2023_boundary(SITE, boundary_prod, lte.PRODUCTION_GAUGE_CACHE," in found
    assert "boundary = SplitBoundary(boundary_pre, boundary_prod, P[\"period\"][0])" in found
    assert "boundary.levels(inst)" in cell_with("shifted = {tau:")
    assert "LevelSeries.from_boundary(boundary," in cell_with("LS = None")
    assert "LEV[int(tau)][w]" in cell_with("M2 = yaml.safe_load(")
    check = cell_with("if len(x_dates):\n    cut = boundary.cut")
    for text in ('side["levels_pre2023"]', "boundary_prod.levels(inst_p)", "np.array_equal(LEV[0][have], h_rec)",
                 "np.array_equal(LS.h[a:b], h)"):
        assert text in check, text


# ── the pre-2023 scenes: NIDEM's frames from the Zarr stores ─────────────────

FILL = -32768


def synthetic_scene(rng, H, W):
    """B03, B08, SCL as int16 digital numbers with fills, a zero sum and every SCL class."""
    g = rng.integers(1, 3000, (H, W)).astype(np.int16)
    n = rng.integers(1, 3000, (H, W)).astype(np.int16)
    s = rng.integers(0, 12, (H, W)).astype(np.int16)
    g[0, :3], n[1, :2], s[2, :4] = FILL, FILL, FILL
    g[3, 3], n[3, 3] = 0, 0                              # B03 + B08 = 0: NaN NDWI
    return g, n, s


def write_synthetic_cubes(tmp, H=12, W=17):
    """A legacy netCDF cube of 2023 (4 scenes) and Zarr stores of 2022 (3 scenes) and 2023 (the cube's scenes, the
    same digital numbers), on the same x / y. The legacy cube has a static no-data block (every band filled on every
    scene) where the stores have data; a reference npz holds it as `static_nodata`, as light_to_extraction writes it.
    Returns (cube, store root, {date: (B03, B08, SCL) as stored}, H, W, the static mask, the reference record)."""
    import xarray as xr
    sys.path.insert(0, ROOT)
    from experiments import light_to_extraction as lte
    from experiments import zarr_cubes as zc

    rng = np.random.default_rng(11)
    x = 500000.0 + 10.0 * np.arange(W) + 5.0
    y = 5700000.0 - 10.0 * np.arange(H) - 5.0
    scenes = {d: synthetic_scene(rng, H, W) for d in ("2022-03-01", "2022-07-15", "2022-12-31",
                                                       "2023-01-05", "2023-02-14", "2023-05-20", "2023-09-09")}
    static = np.zeros((H, W), bool)
    static[9:11, :6] = True                              # no data on every scene of the legacy cube

    def dataset(dates, legacy):
        t = pd.DatetimeIndex(dates)
        bands = {}
        for k, b in enumerate(("B03", "B08", "SCL")):
            v = np.stack([scenes[d][k] for d in dates])
            if legacy:
                v[:, static] = FILL
            bands[b] = (("t", "y", "x"), v)
        return xr.Dataset(bands, coords={"t": t, "y": y, "x": x})

    enc = {b: {"_FillValue": np.int16(FILL)} for b in ("B03", "B08", "SCL")}
    cube = os.path.join(tmp, "legacy_cube.nc")
    d23 = [d for d in scenes if d >= "2023"]
    dataset(d23, legacy=True).to_netcdf(cube, encoding=enc)
    root = os.path.join(tmp, "cubes_light")
    for year, dates in (("2022", [d for d in scenes if d < "2023"]), ("2023", d23)):
        ds = dataset(dates, legacy=False)
        ds.attrs = {"format": zc.FORMAT_ID, "complete": 1}
        ds.to_zarr(str(zc.store_path("site", year, root)), consolidated=True, zarr_format=2, encoding=enc)
    ref = os.path.join(tmp, "site_reference.npz")
    np.savez_compressed(ref, static_nodata=static)
    return cube, root, scenes, H, W, static, {"path": ref, "sha256": lte.sha256_file(ref)}


def same_bits(a, b):
    """The same NaN positions and, elsewhere, the same float32 bits (the frames check's rule)."""
    na, nb = np.isnan(a), np.isnan(b)
    return np.array_equal(na, nb) and np.array_equal(a[~na].view(np.uint32), b[~nb].view(np.uint32))


def expected_frame(g, n, s, static):
    """The NDWI and clear mask of a store's scene with the static block set to no data (the converter's rule)."""
    gf = np.where(g == FILL, np.nan, g).astype(np.float32)
    nf = np.where(n == FILL, np.nan, n).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(gf + nf != 0, (gf - nf) / (gf + nf), np.nan).astype(np.float32)
    clear = np.isin(s, (4, 5, 6, 7))
    ndwi[static], clear[static] = np.nan, False
    return ndwi, clear


def frames_namespace(cube, root, H, W, cube_dates, static=None):
    import xarray as xr
    sys.path.insert(0, ROOT)
    import pyintertidal as pit
    from pyintertidal import waterline_contour as wc
    from experiments import light_to_extraction as lte
    from experiments import zarr_cubes as zc

    ns = dict(np=np, pit=pit, wc=wc, zc=zc, xr=xr, os=os, Counter=Counter, SITE="site", H=H, W=W,
              P={"cache": cube}, cube=types.SimpleNamespace(dates=list(cube_dates)),
              lte=types.SimpleNamespace(LIGHT_ROOT=root, sha256_file=lte.sha256_file),
              keep=np.arange(40))                        # keep pixels: the first rows, outside the static block
    exec(cell_with("cube_pos = {d: i for i, d in enumerate(cube.dates)}"), ns)
    assert ns["STATIC"] is None, "no static mask before the pre-2023 scenes are read"
    ns["STATIC"] = static
    return ns


def test_nidem_frames_before_2023_come_from_the_zarr_stores():
    with tempfile.TemporaryDirectory() as tmp:
        cube, root, scenes, H, W, static, _ = write_synthetic_cubes(tmp)
        d23 = [d for d in scenes if d >= "2023"]
        ns = frames_namespace(cube, root, H, W, d23, static)
        dates = ["2022-07-15", "2022-12-31", "2023-01-05", "2022-03-01", "2023-09-09"]   # mixed, any order
        ndwi, clear = ns["read_frames"](dates)
        for k, d in enumerate(dates):
            e_ndwi, e_clear = expected_frame(*scenes[d], static)   # a cube scene has no data in the block anyway
            assert same_bits(ndwi[k], e_ndwi), d
            assert np.array_equal(clear[k], e_clear), d
        # the reader of the stores gives the cube's own arrays on the cube's scenes (here: the 2023 store)
        n2, c2 = np.empty((len(d23), H, W), np.float32), np.empty((len(d23), H, W), bool)
        ns["zarr_frames"](d23, n2, c2, list(range(len(d23))))
        n1, c1 = ns["read_frames"](d23)
        assert same_bits(n1, n2) and np.array_equal(c1, c2)
        # without the static rule the stores would add observations the cube never has
        ns["STATIC"] = None
        ns["zarr_frames"](d23, n2, c2, list(range(len(d23))))
        assert c2[:, static].any() and not c1[:, static].any()


def run_frames_check(ns, x_dates, s_dates, gate, reference):
    ns.update(x_dates=np.array(x_dates), S_dates=np.array(s_dates), side={"gate_2023": gate, "reference_map": reference})
    exec(cell_with('if len(x_dates):\n    ref = side["reference_map"]'), ns)


def test_frames_check_passes_on_identical_stores_and_stops_on_a_difference():
    import io
    from contextlib import redirect_stdout
    import zarr
    sys.path.insert(0, ROOT)
    from experiments import light_to_extraction as lte

    with tempfile.TemporaryDirectory() as tmp:
        cube, root, scenes, H, W, static, reference = write_synthetic_cubes(tmp)
        d23 = [d for d in scenes if d >= "2023"]
        ns = frames_namespace(cube, root, H, W, d23)
        gate = {"ok": True, "strict_ok": True, "store": {"path": "cubes_light/site/2023.zarr"}}
        out = io.StringIO()
        with redirect_stdout(out):
            run_frames_check(ns, ["2022-03-01", "2022-12-31"], ["2022-03-01", "2022-12-31"] + d23[1:], gate, reference)
        assert "bit for bit" in out.getvalue() and np.array_equal(ns["STATIC"], static), out.getvalue()
        # a reference without the static block: the stores' observations there are caught
        empty = os.path.join(tmp, "empty_reference.npz")
        np.savez_compressed(empty, static_nodata=np.zeros_like(static))
        try:
            with redirect_stdout(io.StringIO()):
                run_frames_check(ns, ["2022-03-01"], ["2022-03-01"] + d23, gate,
                                 {"path": empty, "sha256": lte.sha256_file(empty)})
            raise AssertionError("observations in the static block passed the check")
        except AssertionError as e:
            assert "no 2023 gate difference was accepted" in str(e), e
        try:                                             # a reference that is not the extraction's
            run_frames_check(ns, ["2022-03-01"], ["2022-03-01"] + d23, gate, dict(reference, sha256="0" * 64))
            raise AssertionError("another reference passed")
        except AssertionError as e:
            assert "is not the extraction's reference" in str(e), e
        # one B08 digital number changed in the 2023 store: the check stops, unless the gate's differences were accepted
        z = zarr.open_group(os.path.join(root, "site", "2023.zarr"), mode="r+")
        z["B08"][1, 5, 5] = int(z["B08"][1, 5, 5]) + 7
        zarr.consolidate_metadata(os.path.join(root, "site", "2023.zarr"))
        try:
            with redirect_stdout(io.StringIO()):
                run_frames_check(ns, ["2022-03-01"], ["2022-03-01"] + d23, gate, reference)
            raise AssertionError("a different frame passed the check")
        except AssertionError as e:
            assert "no 2023 gate difference was accepted" in str(e), e
        accepted = dict(gate, ok=True, strict_ok=False, accepted={"by": "the first author"})
        out = io.StringIO()
        with redirect_stdout(out):
            run_frames_check(ns, ["2022-03-01"], ["2022-03-01"] + d23, accepted, reference)
        assert "DIFFERENT" in out.getvalue() and "accepted" in out.getvalue(), out.getvalue()


def test_nidem_is_computed_on_every_window():
    src = cell_with("def run_window(")
    assert "holds no whole frames" not in src and "frames = read_frames(dates)" in src


# ── the last cell: the windows the earlier run computed ─────────────────────

def run_last_cell(first_rows, new_rows, ns_=(125, 100, 75, 60)):
    with tempfile.TemporaryDirectory() as tmp:
        first, out = os.path.join(tmp, "first"), os.path.join(tmp, "out")
        os.makedirs(first), os.makedirs(out)
        pd.DataFrame(first_rows).to_csv(os.path.join(first, "s.csv"), index=False)
        pd.DataFrame(new_rows).to_csv(os.path.join(out, "s.csv"), index=False)
        ns = dict(os=os, pd=pd, np=np, FIRST=first, OUT=out, SITE="s", METHODS=["MAREA", "E4"], NS=list(ns_))
        exec(code_cells(SITE_NB)[-1], ns)


def nb_row(n, method, status="ok", available=68, rmse=0.2, reason="band lags 0/1 min"):
    return dict(site="s", n=n, method=method, status=status, available=available, reason=reason, rmse=rmse,
                seconds=1.0, seconds_own=np.nan)


def test_last_cell_compares_only_the_windows_the_earlier_run_computed():
    first = [nb_row(75, m, "not reachable", reason="68 screened images", rmse=np.nan) for m in ("MAREA", "E4")] + \
            [nb_row(60, m) for m in ("MAREA", "E4")]
    new = [nb_row(75, m, available=142) for m in ("MAREA", "E4")] + [nb_row(60, m, available=142) for m in ("MAREA", "E4")]
    run_last_cell(first, new)                          # n = 75 is new; n = 60 is the same but for `available`
    bad = [dict(r, rmse=0.25) if r["n"] == 60 and r["method"] == "E4" else r for r in new]
    try:
        run_last_cell(first, bad)
        raise AssertionError("a changed score of a window the earlier run computed passed")
    except AssertionError as e:
        assert "rmse" in str(e) or "values are different" in str(e), e
    run_last_cell(first, new[:2], ns_=(125, 100, 75))  # the confirmatory windows only: nothing to compare


def test_last_cell_without_an_earlier_run():
    import io
    from contextlib import redirect_stdout
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out")
        os.makedirs(out)
        pd.DataFrame([nb_row(60, m) for m in ("MAREA", "E4")]).to_csv(os.path.join(out, "s.csv"), index=False)
        ns = dict(os=os, pd=pd, np=np, FIRST=os.path.join(tmp, "none"), OUT=out, SITE="s", METHODS=["MAREA", "E4"],
                  NS=[60])
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code_cells(SITE_NB)[-1], ns)
        assert "nothing to compare" in buf.getvalue() and "refitted MAREA's band lags" in buf.getvalue()


# ── the sites: the Dutch ones unchanged, Ferrol as its production notebook runs it ──────────────────────────

FERROL_NB = os.path.join(ROOT, "tide_boundary_comparison_ferrol.ipynb")
DUTCH_AND_VVA = {       # SITES of commit b5c9b26, before the Spanish sites were generalised: must not change
    "escalda": {"cache": "ndwi_cube_escalda_2023-2025_10m.nc", "extract": "products_escalda_10m/marea_extract.npz",
                "overpass": "products_escalda_10m/overpass_times.json", "marea": "products_escalda_10m/marea_eot20_g2",
                "nodelay": ("products_escalda_10m/comparison_eot20_g2/products.npz", "z_plain"),
                "period": ("2023-01-01", "2025-12-31"), "n_gauges": 2, "judges": ["trnz"], "mouth_side": "west",
                "truth_years": [2025]},
    "wadden": {"cache": "ndwi_cube_wadden_2023-2025_10m.nc", "extract": "products_wadden_10m/marea_extract.npz",
               "overpass": "products_wadden_10m/overpass_times.json", "marea": "products_wadden_10m/marea_eot20_g1",
               "nodelay": ("products_wadden_10m/comparison_eot20_g1/products.npz", "z_plain"),
               "period": ("2023-01-01", "2025-12-31"), "n_gauges": 1, "judges": ["harl"], "mouth_side": "north+west",
               "truth_years": [2025, 2024, 2023]},
    "ems": {"cache": "ndwi_cube_ems_2023-2025_10m.nc", "extract": "products_ems_10m/marea_extract.npz",
            "overpass": "products_ems_10m/overpass_times.json", "marea": "products_ems_10m/marea_eot20_g1",
            "nodelay": ("products_ems_10m/comparison_eot20_g1/products.npz", "z_plain"),
            "period": ("2023-01-01", "2025-12-31"), "n_gauges": 1, "judges": ["delf"], "mouth_side": "north+west",
            "truth_years": [2025]},
    "villaviciosa": {"cache": "ndwi_cube_villaviciosa_grande_10y.nc", "extract": "products_marea/marea_extract.npz",
                     "overpass": "products_marea/overpass_times.json", "marea": "products_marea/marea_eot20_g0_2023-2025",
                     "nodelay": ("products_marea/external_products.npz", "no_delay"),
                     "period": ("2016-01-01", "2025-12-31"), "n_gauges": None, "judges": [], "mouth_side": "any",
                     "truth_years": None},
}


def parameters(env=None, site="escalda"):
    """The namespace of the Parameters and SITES cells, run with the environment `env` (NB_* only)."""
    import io
    from contextlib import redirect_stdout
    saved = {k: v for k, v in os.environ.items() if k.startswith("NB_")}
    for k in saved:
        del os.environ[k]
    os.environ.update({"NB_SITE": site, **(env or {})})
    try:
        ns = {}
        with redirect_stdout(io.StringIO()):
            exec(code_cells(SITE_NB)[0], ns)
            exec(code_cells(SITE_NB)[1], ns)
        return ns
    finally:
        for k in [k for k in os.environ if k.startswith("NB_")]:
            del os.environ[k]
        os.environ.update(saved)


def test_the_other_sites_are_unchanged():
    ns = parameters()
    for site, p in DUTCH_AND_VVA.items():
        assert ns["SITES"][site] == p, site
    assert set(ns["SITES"]) == set(DUTCH_AND_VVA) | {"ferrol"}
    assert ns["SPANISH"] == ("villaviciosa", "ferrol")


def test_ferrol_is_its_production_notebook():
    """Ferrol's entry is tide_boundary_comparison_ferrol.ipynb's Parameters cell and its outputs."""
    import io
    from contextlib import redirect_stdout
    prod = {}
    with redirect_stdout(io.StringIO()):
        exec(code_cells(FERROL_NB)[0], prod)
    f = parameters(site="ferrol")["SITES"]["ferrol"]
    d, tag = prod["OUT_DIR"], prod["BOUNDARY_TAG"]
    assert prod["SITE"] == "ferrol" and prod["TIDE_MODEL"] == "EOT20" and prod["RESOLUTION_M"] == 10
    assert f == dict(cache=prod["CACHE"], extract=f"{d}/marea_extract.npz", overpass=f"{d}/overpass_times.json",
                     marea=f"{d}/marea_{tag}", nodelay=(f"{d}/comparison_{tag}/products.npz", "z_plain"),
                     period=tuple(prod["TIME_EXTENT"]), n_gauges=prod["N_GAUGES"], judges=prod["JUDGE_GAUGES"],
                     mouth_side=prod["MOUTH_SIDE"], truth_years=None)


def test_lidar_parameters_and_the_earlier_names():
    ns = parameters()
    assert ns["LIDAR"] is None and ns["LIDAR_DATE"] is None
    ns = parameters({"NB_VVA_LIDAR": "a.tif", "NB_VVA_LIDAR_DATE": "2017-01-01"}, "villaviciosa")
    assert (ns["LIDAR"], ns["LIDAR_DATE"]) == ("a.tif", "2017-01-01")      # the earlier names still work
    ns = parameters({"NB_LIDAR": "b.tif", "NB_VVA_LIDAR": "a.tif", "NB_LIDAR_DATE": "2024-06-10"}, "ferrol")
    assert (ns["LIDAR"], ns["LIDAR_DATE"]) == ("b.tif", "2024-06-10")       # the new names win


def rows_cell(site, lidar=None, date=None, out=None):
    import io
    import json
    from contextlib import redirect_stdout
    ns = dict(os=os, json=json, pd=pd, SITE=site, SPANISH=("villaviciosa", "ferrol"), LIDAR=lidar, LIDAR_DATE=date,
              NS=[60, 50], METHODS=["MAREA", "E4"], OUT=out, ONE_ROW=True,
              P={"truth_years": DUTCH_AND_VVA.get(site, {}).get("truth_years")})
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(cell_with("LIDAR_META = None\nif SITE in SPANISH:"), ns)
    return ns, buf.getvalue()


def test_the_truth_date_of_the_spanish_sites():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        tif = os.path.join(tmp, "ferrol_pnoa3_10m.tif")
        open(tif, "wb").close()
        meta = {"truth_date": "2024-06-21", "truth_date_rule": "the last flight day under F",
                "source": {"product": "PNOA-LiDAR 3rd coverage"}, "datum": {"vertical": "orthometric"}}
        json.dump(meta, open(tif[:-4] + ".json", "w"))
        ns, out = rows_cell("ferrol", tif)
        assert ns["ROWS"] == [dict(site="ferrol", year=0, end="2024-06-21")] and "from" in out
        assert ns["LIDAR_META"]["truth_date"] == "2024-06-21"
        ns, _ = rows_cell("ferrol", tif, "2024-07-01")                     # a date given explicitly wins
        assert ns["ROWS"][0]["end"] == "2024-07-01"
        try:                                                               # no LiDAR: pending truth, nothing scored
            rows_cell("villaviciosa", None, None, tmp)
            raise AssertionError("no SystemExit without the LiDAR")
        except SystemExit:
            pending = pd.read_csv(os.path.join(tmp, "villaviciosa.csv"))
            assert set(pending.status) == {"pending truth"} and len(pending) == 4
        ns, _ = rows_cell("wadden")                                        # a Dutch site: its own rule, no LiDAR
        assert ns["LIDAR_META"] is None and ns["ROWS"] == [dict(site="wadden", year=2025, end="2025-12-31")]


def test_one_row_merges_only_the_sites_truth_years():
    """NB_ONE_ROW puts the site's truth years in its one row: the Wadden's 2023-2025 cells together, Westerschelde's
    2023 and 2024 cells (truth years [2025]) left out as with one row per year; `survey` keeps each cell's year."""
    class Dataset(dict):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    W_ = 4
    cube = Dataset(x=types.SimpleNamespace(values=np.arange(W_, dtype=float)),
                   y=types.SimpleNamespace(values=np.arange(3, dtype=float)))
    identity = types.SimpleNamespace(transform=lambda x, y: (x, y))
    pyproj_ = types.SimpleNamespace(Transformer=types.SimpleNamespace(from_crs=lambda *a, **k: identity))
    surveyed = {2025: [1, 0, 0, 0, 1], 2024: [0, 1, 0, 0, 1], 2023: [0, 0, 1, 0, 0]}   # pixel k surveyed that year

    def sample_truth(year, xq, yq):
        return np.where(np.array(surveyed[year]) == 1, float(year) / 1000, np.nan), None

    for site, years, one_row, group in (("escalda", [2025], True, [2025, 2024, 2023, 0, 2025]),
                                        ("escalda", [2025], False, [2025, 2024, 2023, 0, 2025]),
                                        ("wadden", [2025, 2024, 2023], True, [2025, 2025, 2025, 0, 2025])):
        ns = dict(np=np, xr=types.SimpleNamespace(open_dataset=lambda path: cube), pyproj=pyproj_, SITE=site,
                  SPANISH=("villaviciosa", "ferrol"), ONE_ROW=one_row, P={"cache": "c.nc", "truth_years": years},
                  ROWS=[dict(site=site, year=2025, end="2025-12-31")], keep=np.array([0, 1, 2, 3, 5]), W=W_,
                  crs="EPSG:25831", vaklodingen=lambda year: year, sample_truth=sample_truth)
        exec(cell_with("truth, group = np.full(len(keep), np.nan), np.zeros(len(keep), int)"), ns)
        assert ns["group"].tolist() == group, (site, one_row, ns["group"].tolist())
        assert ns["survey"].tolist() == [2025, 2024, 2023, 0, 2025]          # the most recent survey wins
        assert np.allclose(ns["truth"], [2.025, 2.024, 2.023, np.nan, 2.025], equal_nan=True)


def test_the_flight_period_of_the_tiles_under_F():
    import io
    from contextlib import redirect_stdout
    from rasterio.transform import from_origin
    W_ = 915
    transform = from_origin(301020.0, 4824840.0, 10.0, 10.0)
    # two pixels in tile 303-4820 (x 303000-304000, y 4819000-4820000), one in 304-4820
    rc = [(500, 200), (583, 297), (500, 300)]
    keep = np.array([r * W_ + c for r, c in rc])
    tiles = [dict(key="303-4820", first_utc="2024-06-10T10:00:00Z", last_utc="2024-06-11T09:00:00Z"),
             dict(key="304-4820", first_utc="2024-06-10T10:00:00Z", last_utc="2024-07-02T10:00:00Z"),
             dict(key="310-4825", first_utc="2025-01-01T10:00:00Z", last_utc="2025-01-01T10:00:00Z")]
    meta = dict(tiles=tiles, intertidal=dict(with_value=2, pixels=3, share_with_value=2 / 3, in_missing_tiles=0))
    for F, end, last, verdict in (([0, 1], "2024-06-11", "2024-06-11", " is the last"),
                                  ([0, 1, 2], "2024-06-11", "2024-07-02", "is NOT")):
        ns = dict(LIDAR_META=meta, transform=transform, keep=keep, F=np.array(F), W=W_,
                  ROWS=[dict(site="s", year=0, end=end)])
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(cell_with("if LIDAR_META:   # the Spanish sites"), ns)
        assert f"flown 2024-06-10 → {last}" in buf.getvalue() and verdict in buf.getvalue(), buf.getvalue()
        assert ns["under"] == ({"303-4820"} if len(F) == 2 else {"303-4820", "304-4820"})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
