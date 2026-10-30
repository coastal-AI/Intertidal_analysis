# -*- coding: utf-8 -*-
"""The per-pixel error maps and the error decomposition of every method at one site and window, output 2 of the
design of 2026-10-29 (experiments/validation_v3_cantabrian_2026-10-29.md, section "Outputs").

Inputs
------
One window's scored pixels: the shape (H, W) of the site's 10 m grid; the flat indices of the scored pixels on it,
row-major (pixel (i, j) is i * W + j, numpy's ravel: the extraction's ``keep[F]``); the truth t at those pixels; and
{method: its estimate p at those pixels}, NaN where a method has no value. The labels of the methods name them in the
figures ({method: label}; a method without one is named by its key), and the methods of the MAREA family (MAREA,
MAREA-0, MAREA-B, MAREA-BL, MAREA-S; by default their ids in notebooks 01 and 05, MAREA_FAMILY) are drawn in the
accent colour. Every function reads the estimates in the order of the dict.

The decomposition
-----------------
Notebook 05's split (`decompose` in notebooks/validation/_build_05_attribution.py, section 3), matched to the last bit
(tests/test_v3_error_maps.py): with d = p - t and the OLS fit of p on t (slope beta, residual e, population moments),

    MSE = mean(d)^2 + ((1 - beta) sd(t))^2 + sd(e)^2

offset = |mean(d)|, relief = |1 - beta| sd(t) (the relief that a slope below 1 compresses), scatter = sd(e) (what the
linear fit leaves). With it come notebook 01's scores on the same pixels (`scores`: RMSE, bias = median(p - t),
centred RMSE, slope, r, the pixels and the coverage); neither scores fewer than MIN_PX = 30 pixels. `decomposition`
gives each method's split on its own pixels (the scored pixels where it has a value) and on the pixels common to all
(the scored pixels where every method has one). A method with no value at all, one that was not computable, does not
narrow the common pixels: notebook 05 leaves it out of its estimates.

The maps
--------
One map per method, a diverging scale (RdBu_r) centred on 0 from -L to +L, the same for every method of the window:
L is the 2nd-98th percentile stretch of |p - t|, taken over every method's scored pixels together. On a scale
centred on 0 only its upper end matters, so L is the 98th percentile of |p - t|, rounded up to a multiple of 0.05 m
(at least 0.05 m). `limit=` passes another L, for example one shared by the two windows of a site. The pixels not
scored are light grey, and so is a scored pixel where the method has no value: both are NaN in the error raster. One
colour bar, in metres. Each title gives the method's label and its RMSE, bias and centred RMSE on its own pixels (the
`own` rows of `decomposition`). The titles of the MAREA family are in the accent colour #B5523B. The axes are cropped
to the bounding box of the scored pixels, with row 0 at the top: north up on the sites' frames, whose transform has
a negative y step.

    error_rasters         {method: (H, W) float32 map of p - t}, NaN where the pixel is not scored or p or t is
                          missing
    scores, decompose     notebook 01's scores and notebook 05's split, on one set of pixels
    common_pixels         the scored pixels where every method with a value has one
    decomposition         decompose for every method, on its own pixels and on the common pixels: one row each
    error_limit           the colour limit L shared by every map of the window
    figure_error_maps     the maps, one per method
    figure_decomposition  stacked bars of offset², relief² and scatter² per method, sorted by MSE, from the table
    write_error_geotiffs  the error rasters as GeoTIFFs on the site's grid (float32, NaN nodata), one per method

In a notebook (keep: the extraction's flat indices; F: the scored pixels among them; truth and est[m] on F):

    from experiments.v3 import error_maps as em
    px = keep[F]
    tab = em.decomposition(truth, est, labels=LABELS)
    fig = em.figure_error_maps((H, W), px, truth, est, labels=LABELS, title=f"{site}, n = {n}: p - t")
    fig2 = em.figure_decomposition(tab, pixels="common", title=f"{site}, n = {n}")
    em.write_error_geotiffs(em.error_rasters((H, W), px, truth, est), transform, crs, f"{OUT}/{site}_n{n}")

The figure functions import matplotlib when called and leave its global settings alone; the colours are
pyintertidal/palette.py's, copied here by value (importing the package for a few constants would load all of it).
"""
import math
import os
import re

import numpy as np
import pandas as pd

MIN_PX = 30                   # notebook 01's scores: no score on fewer pixels
LIMIT_PERCENTILE = 98         # the maps' limit L: this percentile of |p - t| over every method's scored pixels,
LIMIT_STEP = 0.05             # rounded up to a multiple of this (m)
CMAP = "RdBu_r"               # the diverging scale of the maps, centred on 0
MAREA_FAMILY = ("MAREA", "MAREA_ndwi0", "MAREA_B", "MAREA_BL", "MAREA_S", "MAREA_Ss")
                              # MAREA, MAREA-0, MAREA-B, MAREA-BL and MAREA-S by their ids in notebooks 01 and 05:
                              # the design's MAREA-S is V15s (MAREA_Ss); V15 (MAREA_S, the band lags) is kept too

ACCENT = "#B5523B"            # pyintertidal/palette.py ACCENT: the MAREA family
INK = "#2B2B2B"               # INK: titles and labels
SECONDARY = "#6B6B6B"         # SECONDARY: ticks, frames, the RMSE next to each bar
LIGHT_GREY = "#B8B2A3"        # LIGHT_GREY: the pixels not scored
COMPONENTS = (("offset²", "offset", "#C9A66B"),      # SAND        (notebook 05's colours of the three terms)
              ("relief²", "relief", "#2F5D7C"),      # DEEP_WATER
              ("scatter²", "scatter", "#8DB3C7"))    # SHALLOW_WATER
PANEL_IN = 2.8                # the width of one map in the figure (inches): room for its title


# ── The inputs ──────────────────────────────────────────────────────────────

def _grid(shape, px):
    """H, W and the scored pixels' flat indices, checked: whole numbers, inside the grid, each pixel once."""
    H, W = (int(v) for v in shape)
    px = np.asarray(px)
    if px.size == 0:
        px = px.astype(np.int64)
    if px.ndim != 1 or not np.issubdtype(px.dtype, np.integer):
        raise ValueError(f"the scored pixels are a 1-D array of flat indices on the grid, not {px.dtype} {px.shape}")
    if len(px) and (px.min() < 0 or px.max() >= H * W):
        raise ValueError(f"a scored pixel lies outside the {H} x {W} grid")
    if len(np.unique(px)) < len(px):
        raise ValueError("a pixel is scored twice")
    return H, W, px.astype(np.int64)


def _values(truth, estimates, n=None):
    """The truth and the estimates as float arrays of the scored pixels, one value per pixel (n of them, by default
    as many as the truth holds)."""
    t = np.asarray(truth, float)
    if t.ndim != 1 or (n is not None and len(t) != n):
        raise ValueError(f"the truth is not one value per scored pixel: shape {t.shape}"
                         + (f" for {n} pixels" if n is not None else ""))
    est = {}
    for m, p in estimates.items():
        est[m] = np.asarray(p, float)
        if est[m].shape != t.shape:
            raise ValueError(f"{m}: shape {est[m].shape} for {len(t)} scored pixels")
    return t, est


def _family(family):
    return {family} if isinstance(family, str) else set(family)


def _error(p, t):
    """p - t, NaN wherever it is not finite."""
    with np.errstate(invalid="ignore"):
        d = p - t
    return np.where(np.isfinite(d), d, np.nan)


def error_rasters(shape, px, truth, estimates):
    """{method: (H, W) float32 map of p - t}: each method's error at its scored pixels; NaN where the pixel is not
    scored, and where the method or the truth has no value."""
    H, W, px = _grid(shape, px)
    t, est = _values(truth, estimates, len(px))
    out = {}
    for m, p in est.items():
        r = np.full(H * W, np.nan, np.float32)
        r[px] = _error(p, t)
        out[m] = r.reshape(H, W)
    return out


# ── The scores and the decomposition ────────────────────────────────────────

def scores(p, t):
    """Notebook 01's scores of p against t on the same pixels: RMSE, bias = median(p - t), centred RMSE (p and t each
    centred on its median), the OLS slope of p on t, r, the pixels scored (p and t finite) and their share, the
    coverage. NaN scores on fewer than MIN_PX pixels."""
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = np.isfinite(p) & np.isfinite(t)
    out = dict(n_px=int(ok.sum()), coverage=float(ok.mean()) if len(ok) else np.nan,
               rmse=np.nan, bias=np.nan, crmse=np.nan, slope=np.nan, r=np.nan)
    if ok.sum() >= MIN_PX:
        d = p[ok] - t[ok]
        pc, tc = p[ok] - np.median(p[ok]), t[ok] - np.median(t[ok])
        out.update(rmse=float(np.sqrt(np.mean(d ** 2))), bias=float(np.median(d)),
                   crmse=float(np.sqrt(np.mean((pc - tc) ** 2))), slope=float(np.polyfit(tc, pc, 1)[0]),
                   r=float(np.corrcoef(tc, pc)[0, 1]))
    return out


def decompose(p, t):
    """scores(p, t) plus notebook 05's additive split of the mean squared error on the same pixels,
    MSE = mean_bias² + relief² + scatter²: mean_bias = mean(p - t) (signed), offset = |mean_bias|,
    relief = |1 - beta| sd(t) and scatter = sd(e), with beta the slope and e the residual of the OLS fit of p on t
    (population moments). NaN on fewer than MIN_PX pixels."""
    out = dict(scores(p, t), mse=np.nan, mean_bias=np.nan, offset=np.nan, relief=np.nan, scatter=np.nan)
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() >= MIN_PX:
        d = p[ok] - t[ok]
        tc, pc = t[ok] - t[ok].mean(), p[ok] - p[ok].mean()
        beta = np.mean(tc * pc) / np.mean(tc * tc)
        out.update(mse=float(np.mean(d ** 2)), mean_bias=float(d.mean()), offset=float(abs(d.mean())),
                   relief=float(abs(1 - beta) * tc.std()), scatter=float((pc - beta * tc).std()))
    return out


def common_pixels(truth, estimates):
    """The scored pixels where the truth and every method have a value (bool, one per scored pixel). A method with no
    value at any pixel (not computable) does not narrow them."""
    t, est = _values(truth, estimates)
    common = np.isfinite(t)
    for p in est.values():
        ok = np.isfinite(p)
        if ok.any():
            common &= ok
    return common


def decomposition(truth, estimates, labels=None, family=MAREA_FAMILY):
    """decompose for every method, on its own pixels and on the pixels common to all, as notebook 05's table: one row
    per method and pixel set, the columns method, label, family (bool), pixels ('own' or 'common') and those of
    decompose. On the own pixels n_px counts the scored pixels where the method has a value and coverage is their
    share; on the common pixels n_px is the same for every method and coverage is 1."""
    t, est = _values(truth, estimates)
    labels, family = labels or {}, _family(family)
    common = common_pixels(t, est)
    rows = []
    for m, p in est.items():
        for pixels, keep in (("own", np.ones(len(t), bool)), ("common", common)):
            rows.append(dict(method=m, label=labels.get(m, m), family=m in family, pixels=pixels,
                             **decompose(p[keep], t[keep])))
    return pd.DataFrame(rows, columns=["method", "label", "family", "pixels", *decompose([], [])])


# ── The maps ────────────────────────────────────────────────────────────────

def error_limit(truth, estimates, q=LIMIT_PERCENTILE, step=LIMIT_STEP):
    """The limit L of the maps' colour scale, -L to +L for every method of the window: the q-th percentile of |p - t|
    over every method's scored pixels together, rounded up to a multiple of `step` m; `step` when no pixel has an
    error."""
    t, est = _values(truth, estimates)
    a = np.abs(np.concatenate([_error(p, t) for p in est.values()] or [np.empty(0)]))
    a = a[np.isfinite(a)]
    if not len(a):
        return float(step)
    v = float(np.percentile(a, q))
    return max(float(step), round(math.ceil(v / step - 1e-9) * step, 9))   # 1e-9: 0.30 stays 0.30, not 0.35


def _fmt(v, spec):
    return format(v, spec) if np.isfinite(v) else "n/a"


def _title(label, s):
    if not s["n_px"]:
        return f"{label}\nno value"
    return (f"{label}\nRMSE {_fmt(s['rmse'], '.3f')} · bias {_fmt(s['bias'], '+.3f')} · "
            f"cRMSE {_fmt(s['crmse'], '.3f')} m")


def figure_error_maps(shape, px, truth, estimates, labels=None, family=MAREA_FAMILY, limit=None, title=None,
                      ncols=None):
    """One map of p - t per method (the module docstring's maps), as a matplotlib figure. `limit`: the colour limit L
    (default error_limit of these estimates); `title`: the figure's title; `ncols`: the maps per row (default from the
    shape of the bounding box)."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    H, W, px = _grid(shape, px)
    t, est = _values(truth, estimates, len(px))
    if not est or not len(px):
        raise ValueError("nothing to map: no method or no scored pixel")
    labels, family = labels or {}, _family(family)
    rows, cols = px // W, px % W
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    bh, bw = r1 - r0 + 1, c1 - c0 + 1
    L = error_limit(t, est) if limit is None else float(limit)
    if not L > 0:
        raise ValueError(f"the colour limit must be positive, not {limit}")
    norm = Normalize(-L, L)
    cmap = colormaps[CMAP].with_extremes(bad=LIGHT_GREY)

    n, aspect = len(est), bw / bh
    if ncols is None:
        ncols = int(np.clip(round(math.sqrt(n * 1.5 / aspect)), 1, n))
        ncols = math.ceil(n / math.ceil(n / ncols))         # the same rows, filled evenly
    nrows = math.ceil(n / ncols)
    height = float(np.clip(PANEL_IN / aspect, 0.8, 4.5))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                             figsize=(ncols * PANEL_IN + 1.0, nrows * (height + 0.5) + 0.4 + 0.3 * bool(title)))
    extent = (c0 - 0.5, c1 + 0.5, r1 + 0.5, r0 - 0.5)       # grid columns and rows, row 0 at the top
    for ax, (m, p) in zip(axes.flat, est.items()):
        img = np.full((bh, bw), np.nan, np.float32)
        img[rows - r0, cols - c0] = _error(p, t)
        ax.imshow(img, cmap=cmap, norm=norm, extent=extent, interpolation="antialiased")
        ax.set_title(_title(labels.get(m, m), scores(p, t)), loc="left", fontsize=7.5, fontweight="bold",
                     color=ACCENT if m in family else INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(SECONDARY)
            spine.set_linewidth(0.5)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, extend="both", shrink=0.8, aspect=30)
    cb.set_label("p − t (m)", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=SECONDARY)
    cb.outline.set_edgecolor(SECONDARY)
    cb.outline.set_linewidth(0.5)
    fig.legend(handles=[Patch(facecolor=LIGHT_GREY, edgecolor="none", label="not scored, or no value")],
               loc="outside lower left", frameon=False, fontsize=7)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9, fontweight="bold", color=INK)
    return fig


def figure_decomposition(table, pixels="common", title=None, ax=None):
    """Stacked horizontal bars of offset², relief² and scatter² per method, from a `decomposition` table, on its
    'common' (default) or 'own' pixels: the methods sorted by MSE (the smallest at the top; a method without a score
    is left out), each bar's length its MSE and the number after it its RMSE (m), the labels of the MAREA family in
    the accent colour. `ax`: draw into these axes (default: a new figure). Returns the figure."""
    import matplotlib.pyplot as plt

    d = table[table["pixels"] == pixels]
    d = d[np.isfinite(d["mse"].to_numpy(float))].sort_values("mse", kind="stable")    # ties keep the table's order
    k = len(d)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 0.26 * max(k, 3) + 1.4), layout="constrained")
    else:
        fig = ax.figure
    y, left = np.arange(k), np.zeros(k)
    for name, column, colour in COMPONENTS:
        v = d[column].to_numpy(float) ** 2
        ax.barh(y, v, left=left, height=0.72, color=colour, edgecolor="white", linewidth=1, label=name)
        left += v
    for i, rmse in enumerate(d["rmse"].to_numpy(float)):
        ax.text(left[i], i, f" {rmse:.3f}", va="center", ha="left", fontsize=6.5, color=SECONDARY)
    ax.set_yticks(y, [str(v) for v in d["label"]])
    ax.tick_params(axis="y", labelsize=7.5, length=0)
    for tick, in_family in zip(ax.get_yticklabels(), d["family"]):
        tick.set_color(ACCENT if in_family else INK)
        tick.set_fontweight("bold" if in_family else "normal")
    if k:
        ax.set_ylim(k - 0.5, -0.5)                          # the smallest MSE at the top
        ax.set_xlim(0, float(left.max()) * 1.18 or 1.0)     # room for the RMSE after the longest bar
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=7, borderaxespad=0.3,
                  handlelength=1.2, columnspacing=1.2)
    else:
        ax.set_xticks([])
        ax.text(0.5, 0.5, f"no method has a score on the {pixels} pixels", transform=ax.transAxes, ha="center",
                va="center", fontsize=7.5, color=SECONDARY)
    ax.set_xlabel("MSE (m²)", fontsize=8, color=INK)
    ax.tick_params(axis="x", labelsize=7, colors=SECONDARY)
    ax.grid(axis="x", color=LIGHT_GREY, alpha=0.35, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SECONDARY)
        ax.spines[side].set_linewidth(0.6)
    if pixels == "common":
        where = f"common pixels · {int(d['n_px'].max()):,} px" if k else "common pixels"
    else:
        where = "each method on its own pixels"
    ax.set_title(f"{title} · {where}" if title else where, loc="left", fontsize=8.5, fontweight="bold", color=INK,
                 pad=16 if k else 6)                        # 16 pt: above the legend's row
    return fig


# ── The GeoTIFFs ────────────────────────────────────────────────────────────

def _stem(method):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method)).strip("._") or "method"


def write_error_geotiffs(rasters, transform, crs, folder):
    """Each error raster (error_rasters) as `<folder>/<method>_error.tif` on the site's grid (transform and crs as
    rasterio takes them): float32, NaN nodata, deflate, as pyintertidal.raster.write_geotiff writes them, with the
    method and the quantity (p - t, m) in the tags. A method id is made safe for a file name (characters other than
    letters, digits, '_', '.' and '-' become '_'). Returns {method: path}."""
    import rasterio

    stems = {m: _stem(m) for m in rasters}
    taken = {}
    for m, s in stems.items():
        if s.lower() in taken:
            raise ValueError(f"{taken[s.lower()]!r} and {m!r} would write the same file {s}_error.tif")
        taken[s.lower()] = m
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for m, r in rasters.items():
        r = np.asarray(r, np.float32)
        if r.ndim != 2:
            raise ValueError(f"{m}: a raster is 2-D, not {r.shape}")
        paths[m] = os.path.join(folder, f"{stems[m]}_error.tif")
        with rasterio.open(paths[m], "w", driver="GTiff", height=r.shape[0], width=r.shape[1], count=1,
                           dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate") as dst:
            dst.write(r, 1)
            dst.update_tags(method=str(m), quantity="p - t, the estimate minus the truth", units="m")
            dst.set_band_description(1, f"{m}: p - t (m)")
            dst.set_band_unit(1, "m")
    return paths
