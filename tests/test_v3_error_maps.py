# -*- coding: utf-8 -*-
"""Validation v3's error maps and decomposition (experiments/v3/error_maps.py), on synthetic pixels.

* scores and decompose are notebook 05's (notebooks/validation/_build_05_attribution.py), bit for bit, the offset
  being |mean_bias|;
* the identity adds up, MSE = offset² + relief² + scatter², and each term is what it says (an independent OLS fit);
* the decomposition table: each method on its own pixels and on the pixels common to all, a method that was not
  computable narrowing nothing, the family and the labels;
* the rasters put each error at its pixel (row-major flat indices), float32, NaN where not scored and where p or t is
  missing; the inputs are checked;
* the limit shared by the maps: the 98th percentile of |p - t| over every method, rounded up to 0.05 m;
* the maps' figure: one map per method, the same limits and colour scale for every one, the error of each pixel at
  its place, the non-scored pixels light grey, the axes cropped to the scored pixels, the titles and their colours,
  one colour bar in metres, NaN handled;
* the decomposition's figure: sorted by MSE, each segment the term's square, the family highlighted;
* the GeoTIFFs round trip; the colours are pyintertidal/palette.py's; matplotlib's global settings are left alone.

Every figure is built with the Agg backend.

Run:  python -m pytest -q tests/test_v3_error_maps.py
"""
import ast
import importlib.util
import io
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(ROOT, "notebooks", "validation", "_build_05_attribution.py")
PALETTE = os.path.join(ROOT, "pyintertidal", "palette.py")
sys.path.insert(0, ROOT)

from experiments.v3 import error_maps as em  # noqa: E402

LABELS = {"MAREA": "MAREA", "MAREA_ndwi0": "MAREA-0", "MAREA_B": "MAREA-B", "E4": "LT2D", "E5c": "SE"}


def builder_function(name, **names):
    """A function defined in a code cell of notebook 05's builder, compiled alone with numpy and the given globals."""
    with open(BUILDER, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "code" and call.args
                and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
            src = call.args[0].value.strip("\n")
            try:
                body = ast.parse(src).body
            except SyntaxError:
                continue
            for node in body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    ns = dict(np=np, **names)
                    exec(compile(ast.get_source_segment(src, node), f"_build_05_attribution.py:{name}", "exec"), ns)
                    return ns[name]
    raise LookupError(f"{name} in {BUILDER}")


def site(H=40, W=60, seed=0):
    """A synthetic window: scored pixels inside a sub-box of an H x W grid (unsorted, not every pixel of the box),
    a smooth truth, and five methods: offset, compressed relief, scatter, a few missing values, and a method that
    was not computable (every value NaN)."""
    rng = np.random.default_rng(seed)
    corners = [(5, 8), (30, 49)]                                 # the box's corners are scored
    box = np.array([(i, j) for i in range(5, 31) for j in range(8, 50) if (i, j) not in corners])
    pick = np.r_[rng.choice(len(box), 698, replace=False)]
    rows, cols = np.r_[5, 30, box[pick, 0]], np.r_[8, 49, box[pick, 1]]
    px = rows * W + cols
    t = np.sin(rows / 7.0) + 0.02 * cols + rng.normal(0, 0.05, len(px))
    est = {"MAREA": t + 0.15 + rng.normal(0, 0.08, len(px)),
           "MAREA_ndwi0": t.mean() + 0.6 * (t - t.mean()) + rng.normal(0, 0.05, len(px)),
           "MAREA_B": t + rng.normal(0, 0.25, len(px)),
           "E4": t - 0.05 + rng.normal(0, 0.12, len(px)),
           "E5c": np.full(len(px), np.nan)}
    est["MAREA_B"][rng.random(len(px)) < 0.1] = np.nan
    return (H, W), px, t, est


def map_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def left_title(ax):
    return ax._left_title                     # set_title(loc="left"): matplotlib keeps it there


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ── scores and decompose ────────────────────────────────────────────────────

def test_scores_and_decompose_are_notebook_05s():
    nb_scores = builder_function("scores")
    nb_decompose = builder_function("decompose", scores=nb_scores)
    rng = np.random.default_rng(1)
    t = rng.uniform(-1.5, 2.0, 600)
    noisy = 0.8 * t + 0.1 + rng.normal(0, 0.2, 600)
    noisy[rng.random(600) < 0.1] = np.nan
    noisy[3] = np.inf
    t_gaps = t.copy()
    t_gaps[:15] = np.nan
    cases = [(noisy, t), (noisy, t_gaps), (0.5 * t + 0.2, t), (t[:25] + 0.1, t[:25]),
             (np.full(40, np.nan), t[:40]), (np.empty(0), np.empty(0))]
    for p, tt in cases:
        mine, ref = em.decompose(p, tt), nb_decompose(p, tt)
        assert set(mine) == set(ref) | {"offset"}
        for key, value in ref.items():
            np.testing.assert_equal(mine[key], value, err_msg=key)          # bit for bit, NaN equal to NaN
        np.testing.assert_equal(mine["offset"], abs(ref["mean_bias"]))
        assert em.scores(p, tt).keys() == nb_scores(p, tt).keys()
        for key, value in nb_scores(p, tt).items():
            np.testing.assert_equal(em.scores(p, tt)[key], value, err_msg=key)


def test_the_identity_adds_up():
    rng = np.random.default_rng(2)
    t = rng.uniform(-1, 2, 2000)
    p = 0.7 * t + 0.25 + rng.normal(0, 0.15, 2000)
    p[:100] = np.nan
    d = em.decompose(p, t)
    assert d["n_px"] == 1900
    assert np.isclose(d["mse"], d["offset"] ** 2 + d["relief"] ** 2 + d["scatter"] ** 2, rtol=1e-12, atol=0)
    assert np.isclose(d["rmse"] ** 2, d["mse"], rtol=1e-12, atol=0)
    ok = np.isfinite(p)
    beta, a = np.polyfit(t[ok], p[ok], 1)                                 # an independent OLS fit of p on t
    assert np.isclose(d["slope"], beta, rtol=1e-9)
    assert np.isclose(d["offset"], abs(np.mean(p[ok] - t[ok])), rtol=1e-12)
    assert np.isclose(d["relief"], abs(1 - beta) * np.std(t[ok]), rtol=1e-9)
    assert np.isclose(d["scatter"], np.std(p[ok] - (a + beta * t[ok])), rtol=1e-9)

    lin = em.decompose(0.5 * t + 0.2, t)                                  # no scatter: offset and relief only
    assert lin["scatter"] < 1e-12 and np.isclose(lin["relief"], 0.5 * t.std())
    assert np.isclose(lin["offset"], abs(np.mean(0.2 - 0.5 * t)))
    shift = em.decompose(t - 0.3, t)                                      # a pure offset
    assert np.isclose(shift["offset"], 0.3) and shift["relief"] < 1e-12 and shift["scatter"] < 1e-12
    assert np.isclose(shift["mean_bias"], -0.3) and np.isclose(shift["mse"], 0.09)
    few = em.decompose(t[:29] + 0.1, t[:29])                              # fewer than 30 pixels: no score
    assert few["n_px"] == 29 and all(np.isnan(few[k]) for k in ("rmse", "mse", "offset", "relief", "scatter"))


def test_decomposition_on_own_and_common_pixels():
    _, _, t, est = site()
    tab = em.decomposition(t, est, labels=LABELS)
    assert list(tab.columns[:4]) == ["method", "label", "family", "pixels"]
    assert len(tab) == 2 * len(est) and list(tab.method[::2]) == list(est)
    common = np.isfinite(t) & np.isfinite(est["MAREA_B"])                 # E5c, never computed, narrows nothing
    assert np.array_equal(em.common_pixels(t, est), common)
    for m, p in est.items():
        own = tab[(tab.method == m) & (tab.pixels == "own")].iloc[0]
        com = tab[(tab.method == m) & (tab.pixels == "common")].iloc[0]
        for key, value in em.decompose(p, t).items():
            np.testing.assert_equal(own[key], value, err_msg=f"{m} own {key}")
        for key, value in em.decompose(p[common], t[common]).items():
            np.testing.assert_equal(com[key], value, err_msg=f"{m} common {key}")
        assert own.label == LABELS[m] and bool(own.family) == (m in em.MAREA_FAMILY)
    assert (tab[tab.pixels == "own"].set_index("method").n_px.to_dict()
            == {m: int(np.isfinite(p).sum()) for m, p in est.items()})
    scored = tab[(tab.pixels == "common") & (tab.method != "E5c")]
    assert (scored.n_px == common.sum()).all() and (scored.coverage == 1).all()
    assert tab.set_index(["method", "pixels"]).loc[("E5c", "own"), "n_px"] == 0
    none = em.decomposition(t, est, labels=None, family="E4")             # one id, not its letters
    assert list(none.label[::2]) == list(est) and list(none.family[::2]) == [m == "E4" for m in est]
    assert list(em.decomposition(t, {}).columns) == list(tab.columns)


# ── the rasters ─────────────────────────────────────────────────────────────

def test_rasters_put_each_error_at_its_pixel():
    rng = np.random.default_rng(3)
    H, W = 7, 11
    px = rng.choice(H * W, 30, replace=False)                             # unsorted flat indices
    t = rng.normal(0, 1, 30)
    p = t + rng.normal(0, 0.3, 30)
    p[4], t[7], p[9] = np.nan, np.nan, np.inf
    r = em.error_rasters((H, W), px, t, {"A": p, "B": t + 0.5})
    assert set(r) == {"A", "B"} and r["A"].shape == (H, W) and r["A"].dtype == np.float32
    for k, flat in enumerate(px):
        i, j = divmod(int(flat), W)                                       # row-major: pixel (i, j) is i * W + j
        if k in (4, 7, 9):
            assert np.isnan(r["A"][i, j])                                 # no estimate, no truth, not finite
        else:
            assert r["A"][i, j] == np.float32(p[k] - t[k])
        assert np.isnan(r["B"][i, j]) if k == 7 else np.isclose(r["B"][i, j], 0.5)
    outside = np.ones(H * W, bool)
    outside[px] = False
    assert np.isnan(r["A"].ravel()[outside]).all() and np.isnan(r["B"].ravel()[outside]).all()
    assert np.isfinite(r["A"]).sum() == 27


def test_inputs_are_checked():
    t = np.zeros(3)
    with pytest.raises(ValueError, match="twice"):
        em.error_rasters((2, 2), [0, 1, 1], t, {"A": t})
    with pytest.raises(ValueError, match="outside"):
        em.error_rasters((2, 2), [0, 1, 4], t, {"A": t})
    with pytest.raises(ValueError, match="flat indices"):
        em.error_rasters((2, 2), np.array([True, False, True, False]), t, {"A": t})
    with pytest.raises(ValueError, match="A"):
        em.error_rasters((2, 2), [0, 1, 2], t, {"A": np.zeros(4)})
    with pytest.raises(ValueError, match="truth"):
        em.error_rasters((2, 2), [0, 1, 2], np.zeros(2), {"A": t})
    with pytest.raises(ValueError, match="nothing to map"):
        em.figure_error_maps((2, 2), [0, 1, 2], t, {})


# ── the shared limit ────────────────────────────────────────────────────────

def test_the_limit_is_the_98th_percentile_rounded_up():
    t = np.zeros(50)
    a = np.arange(50) / 100.0                       # |p - t| = 0.00 ... 0.49
    b = -(np.arange(50) / 100.0 + 0.5)              # and 0.50 ... 0.99, below the truth
    pooled = np.abs(np.r_[a, b])
    assert np.isclose(np.percentile(pooled, 98), 0.9702)
    assert em.error_limit(t, {"A": a, "B": b}) == 1.0
    assert em.error_limit(t, {"A": a}) == math.ceil(np.percentile(a, 98) / 0.05) * 0.05 == 0.5
    assert em.error_limit(t, {"A": np.full(50, 0.3)}) == 0.3          # on the 0.05 grid: kept
    assert em.error_limit(t, {"A": np.full(50, 0.3001)}) == 0.35      # rounded up, never down
    assert em.error_limit(t, {"A": np.full(50, -0.01)}) == 0.05
    assert em.error_limit(t, {"A": t}) == 0.05                        # no error at all: the smallest limit
    assert em.error_limit(t, {"A": np.full(50, np.nan)}) == 0.05      # no value
    with_nan = {"A": a, "B": b, "C": np.full(50, np.nan)}            # a method without values changes nothing
    assert em.error_limit(t, with_nan) == em.error_limit(t, {"A": a, "B": b})
    assert em.error_limit(np.r_[t, np.nan], {"A": np.r_[a, 9.0]}) == 0.5   # nor a pixel without truth


# ── the maps' figure ────────────────────────────────────────────────────────

def test_maps_share_the_limits_and_put_each_error_at_its_pixel():
    shape, px, t, est = site()
    H, W = shape
    fig = em.figure_error_maps(shape, px, t, est, labels=LABELS, title="synthetic, n = 100")
    fig.savefig(io.BytesIO(), format="png")
    L = em.error_limit(t, est)
    rasters = em.error_rasters(shape, px, t, est)
    rows, cols = px // W, px % W
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    axes = map_axes(fig)
    assert len(axes) == len(est)
    for ax, (m, p) in zip(axes, est.items()):
        im = ax.images[0]
        assert im.get_clim() == (-L, L)                                   # the same limits, centred on 0
        assert im.cmap.name == "RdBu_r"
        assert np.allclose(im.cmap.get_bad(), to_rgba(em.LIGHT_GREY))     # the pixels not scored
        shown = np.ma.filled(im.get_array().astype(float), np.nan)
        np.testing.assert_array_equal(shown, rasters[m][r0:r1 + 1, c0:c1 + 1])
        assert ax.get_xlim() == (c0 - 0.5, c1 + 0.5) and ax.get_ylim() == (r1 + 0.5, r0 - 0.5)
        title = left_title(ax)
        s = em.scores(p, t)
        first, second = title.get_text().split("\n")
        assert first == LABELS[m]
        if s["n_px"]:
            assert second == f"RMSE {s['rmse']:.3f} · bias {s['bias']:+.3f} · cRMSE {s['crmse']:.3f} m"
        else:
            assert second == "no value"
        assert to_hex(title.get_color()).upper() == (em.ACCENT if m in em.MAREA_FAMILY else em.INK)
    bars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(bars) == 1 and "(m)" in bars[0].get_ylabel()
    assert np.allclose(bars[0].get_ylim(), (-L, L))
    assert fig.get_suptitle() == "synthetic, n = 100"


def test_maps_nan_handling_and_options():
    shape, px, t, est = site(seed=5)
    est["E4"][:-20] = np.nan                                              # 20 values: shown, not scored
    t[:10] = np.nan                                                       # no truth: grey in every map
    fig = em.figure_error_maps(shape, px, t, est, labels=LABELS, family=("E4",), limit=0.4, ncols=2)
    fig.savefig(io.BytesIO(), format="png")
    axes = map_axes(fig)
    assert len(axes) == len(est) and sum(not ax.axison for ax in fig.axes if not ax.images) == 1   # 3 x 2, one empty
    W = shape[1]
    r0, c0 = (px // W).min(), (px % W).min()
    for ax, m in zip(axes, est):
        im = ax.images[0]
        assert im.get_clim() == (-0.4, 0.4)
        shown = np.ma.filled(im.get_array().astype(float), np.nan)
        assert np.isnan(shown[px[:10] // W - r0, px[:10] % W - c0]).all()
        colour = to_hex(left_title(ax).get_color()).upper()
        assert colour == (em.ACCENT if m == "E4" else em.INK)
    titles = {m: left_title(ax).get_text() for ax, m in zip(axes, est)}
    assert titles["E4"].endswith("RMSE n/a · bias n/a · cRMSE n/a m")
    assert titles["E5c"].endswith("no value")
    assert np.isnan(np.ma.filled(axes[-1].images[0].get_array().astype(float), np.nan)).all()
    with pytest.raises(ValueError, match="positive"):
        em.figure_error_maps(shape, px, t, est, limit=0)


# ── the decomposition's figure ──────────────────────────────────────────────

def test_decomposition_figure():
    _, _, t, est = site()
    tab = em.decomposition(t, est, labels=LABELS)
    for pixels in ("common", "own"):
        fig = em.figure_decomposition(tab, pixels=pixels, title="synthetic, n = 100")
        fig.savefig(io.BytesIO(), format="png")
        ax = fig.axes[0]
        d = tab[(tab.pixels == pixels) & np.isfinite(tab.mse)].sort_values("mse")
        assert len(d) == len(est) - 1                                     # E5c has no score: left out
        assert [x.get_text() for x in ax.get_yticklabels()] == list(d.label)   # the smallest MSE first ...
        assert ax.get_ylim() == (len(d) - 0.5, -0.5)                      # ... at the top
        assert len(ax.containers) == 3
        left = np.zeros(len(d))
        for bars, (name, column, colour) in zip(ax.containers, em.COMPONENTS):
            assert bars.get_label() == name
            for i, rect in enumerate(bars.patches):
                assert np.isclose(rect.get_width(), d[column].iloc[i] ** 2, rtol=1e-12)
                assert np.isclose(rect.get_x(), left[i]) and np.isclose(rect.get_y() + rect.get_height() / 2, i)
                assert to_hex(rect.get_facecolor()).upper() == colour
            left += d[column].to_numpy() ** 2
        assert np.allclose(left, d.mse, rtol=1e-12)                       # each bar is the method's MSE
        assert [x.get_text() for x in ax.texts] == [f" {v:.3f}" for v in d.rmse]
        for tick, fam in zip(ax.get_yticklabels(), d.family):
            assert to_hex(tick.get_color()).upper() == (em.ACCENT if fam else em.INK)
            assert (tick.get_fontweight() == "bold") == bool(fam)
        assert [x.get_text() for x in ax.get_legend().get_texts()] == ["offset²", "relief²", "scatter²"]
        assert ax.get_title(loc="left").startswith("synthetic, n = 100 · ")
    fig, (a1, a2) = plt.subplots(1, 2)
    assert em.figure_decomposition(tab, ax=a2) is fig and len(a2.containers) == 3 and not a1.containers
    empty = em.figure_decomposition(tab[tab.method == "E5c"])            # no score at all: builds, says so
    empty.savefig(io.BytesIO(), format="png")
    assert "no method" in empty.axes[0].texts[0].get_text()


# ── the GeoTIFFs ────────────────────────────────────────────────────────────

def test_geotiffs_round_trip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    shape, px, t, est = site()
    rasters = em.error_rasters(shape, px, t, est)
    rasters["odd / id"] = rasters["MAREA"]
    transform, crs = from_origin(380000.0, 4820000.0, 10.0, 10.0), CRS.from_epsg(32630)
    folder = os.path.join(str(tmp_path), "maps", "site_n100")
    paths = em.write_error_geotiffs(rasters, transform, crs, folder)
    assert set(paths) == set(rasters)
    assert os.path.basename(paths["MAREA_ndwi0"]) == "MAREA_ndwi0_error.tif"
    assert os.path.basename(paths["odd / id"]) == "odd_id_error.tif"
    for m, path in paths.items():
        with rasterio.open(path) as src:
            assert (src.height, src.width, src.count) == shape + (1,) and src.dtypes == ("float32",)
            assert src.transform == transform and src.crs == crs and np.isnan(src.nodata)
            assert src.tags()["method"] == m and src.units == ("m",)
            np.testing.assert_array_equal(src.read(1), rasters[m])
    with pytest.raises(ValueError, match="same file"):
        em.write_error_geotiffs({"a/b": rasters["MAREA"], "a_b": rasters["MAREA"]}, transform, crs, folder)
    with pytest.raises(ValueError, match="2-D"):
        em.write_error_geotiffs({"A": np.zeros((2, 2, 2))}, transform, crs, folder)


# ── the constants and the style ─────────────────────────────────────────────

def test_constants_and_colours_are_the_palettes():
    spec = importlib.util.spec_from_file_location("palette", PALETTE)      # by its path, as the poster scripts do
    pal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pal)
    assert em.ACCENT == pal.ACCENT == "#B5523B"
    assert (em.INK, em.SECONDARY, em.LIGHT_GREY) == (pal.INK, pal.SECONDARY, pal.LIGHT_GREY)
    assert [c for _, _, c in em.COMPONENTS] == [pal.SAND, pal.DEEP_WATER, pal.SHALLOW_WATER]
    assert (em.MIN_PX, em.LIMIT_PERCENTILE, em.LIMIT_STEP, em.CMAP) == (30, 98, 0.05, "RdBu_r")
    assert em.MAREA_FAMILY[:4] == ("MAREA", "MAREA_ndwi0", "MAREA_B", "MAREA_BL") and "MAREA_Ss" in em.MAREA_FAMILY


def test_figures_leave_matplotlibs_settings_alone():
    before = dict(plt.rcParams)
    shape, px, t, est = site()
    em.figure_error_maps(shape, px, t, est).savefig(io.BytesIO(), format="png")
    em.figure_decomposition(em.decomposition(t, est)).savefig(io.BytesIO(), format="png")
    assert dict(plt.rcParams) == before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
