# -*- coding: utf-8 -*-
"""CARTOGRASS 2026 poster, Methods: "A pixel is a tide gauge".

One real 10 m intertidal pixel near the mouth of the Ria de Villaviciosa,
2023-2025. Every look of it in MAREA's own 2023-2025 Villaviciosa record is
placed at the tide level of its image (EOT20, metres above mean sea level) and
coloured by its water index: above 0 teal (seen wet), 0 or below sand (seen
dry). A look follows the sheet's one rule (card 1 of the method flow chart,
"1 look = 1 cloud-free view"): a scene MAREA's record keeps (the whole frame at
most 5 % cloudy or, failing that, the transition zone at most 10 % cloudy:
97 scenes of 2023-2025, the record behind the pilot result) in which this
pixel's own Sentinel-2 cloud class is clear. Before 2026-10-19 the figure
plotted every overpass with the pixel clear (159 looks, no scene rule). The line is the
flooding curve MAREA fits to that record: its midpoint is the pixel's
elevation z and its width the uncertainty of z. The figure says the rule in
words and symbols on the empty side of the step: wet when the water is
above the ground, h > z.

Run from the repository root (a few seconds once the EOT20 levels are cached
in poster/tools/cg/cache/; the first run predicts them, peak about 0.65 GB):

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/fig_pixel_gauge.py
    ... fig_pixel_gauge.py --preset slot    # the earlier 240 x 88 mm slot
    ... fig_pixel_gauge.py --select         # reprint the pixel ranking (streams
                                            # the cube once, about 40 s, 0.7 GB)

Outputs, drawn at print size (type 21-24 pt, nothing under 20 pt):
    --preset half (default)  391 x 57.5 mm, the slot \\figPixW x \\figPixH of
                             poster/cartograss2026.tex (left half of the
                             Methods row, under the study sites; the panel's
                             title line says "one real pixel" and the footer
                             carries the Copernicus credit)
                             -> poster/figures/cg/pixel_gauge.{pdf,png}
    --preset strip           331 x 62 mm  -> pixel_gauge_strip.{pdf,png}
    --preset slot            240 x 88 mm  -> pixel_gauge_slot.{pdf,png}
    --preset wide            250 x 150 mm -> pixel_gauge_wide.{pdf,png}
The PDF is vector, Fira Sans embedded as TrueType through
b_common.use_poster_fonts (the face and the embedding of every other figure
of the poster); the PNG is a 300 dpi raster of the same page (pdftocairo).
A built-in check prints any two labels that overlap or any label that leaves
the figure.

Inputs (read only; nothing in pyintertidal/ or the products is modified):
    ndwi_cube_villaviciosa_total_2023-2025.nc
        the legacy 10 m openEO cube (B03, B08, SCL). Only one pixel's series
        is read, never a scene or the cube.
    products_marea/overpass_times.json
        the record instants (catalogue / product-name time), as in production.
    tide_models/EOT20
        the Villaviciosa production boundary, PyTMDBoundary("EOT20") at the
        site AOI centroid (experiments/benchmark/config.py, BOUNDARY), read at
        each overpass with no delay. Cached in poster/tools/cg/cache/ for
        every cube date; the record's dates are taken from that cache.
    results/server_10m/products_marea/marea_eot20_g0_2023-2025/record.json
        the 97 dates of that record (its "dates" list only: no elevation, delay
        or score of the record is read).

Reflectances are floored at 0 before the index is formed (the validation's
primary preprocessing, "floor0"): over water the L2A near-infrared can be
slightly negative (B08 down to -30 in cube units), which would push NDWI
above its physical maximum of 1. The fit does not change (z -0.366 m, sigma
0.06 m, with or without the floor); only the plotted index stops at 1.

The fit is the product's own: pyintertidal.elevation._fit_block on the marea
grid (50 levels between the record's lowest and highest tide, times
marea.SG_GRID), and the script asserts that marea.invert_series, with the
product's quality rule, returns the same elevation.

The pixel (row 400, column 708 of the cube; 43.5258 N, 5.3851 W) lies on the
open flat east of the main channel, 2.9 km from the mouth along the water (the
record's own along-channel distance of this pixel). It was picked
for legibility (``--select`` reprints the ranking): among the ~9,100
intertidal pixels with at least 120 clear looks, a water frequency of 0.3-0.7
and at least 25 looks on each side of the midpoint, it is in the top 20 by
agreement with "wet when the level is above z" (96.9 % of its 159 looks);
within that top 20 it was preferred for sitting on the open flat rather than
a channel edge, for its long record and for its sharp step. It illustrates
the measurement principle; it is not a statement about typical errors.

Near the mouth the tide's delay is negligible: re-fitted with the boundary
read a few minutes earlier, the elevation and the width do not change, so the
tide plotted here is also the corrected one for this pixel and the rule is
drawn as h > z (the method's general form, h(t - tau) > z, is the flow
chart's). The figure says so in words: "3 km from the mouth: delay tau ~ 0"
(tau is not zero by rule there, only negligible).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, "pyintertidal")):
    sys.exit("run this script from the repository root")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "poster", "tools", "cg"))

CUBE = "ndwi_cube_villaviciosa_total_2023-2025.nc"
OVERPASS = os.path.join("products_marea", "overpass_times.json")
RECORD = os.path.join("results", "server_10m", "products_marea", "marea_eot20_g0_2023-2025",
                      "record.json")
CACHE = os.path.join("poster", "tools", "cg", "cache",
                     "villaviciosa_eot20_overpass_2023-2025.csv")
OUT_DIR = os.path.join("poster", "figures", "cg")
MIKTEX = r"C:\Users\Jorge\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

ROW, COL = 400, 708            # the pixel, in the cube's (y, x) grid

# Poster palette (poster/preamble/colors.tex)
INK, SLATE, DEEP = "#16202A", "#4A5C6B", "#0B3B5A"
TEAL, SAND, MIST, RULE = "#1B7F86", "#C08A3E", "#EEF2F5", "#C9D3DA"
SAND_DARK = "#8A5F22"          # PSand darkened for small text on white (b2's)

MM = 1 / 25.4
PT = 25.4 / 72                 # mm per point


def log(msg):
    """Print and flush at once (progress of a long run)."""
    print(msg, flush=True)


def tint(hex_colour, share):
    """RGB tuple (0-1) of ``hex_colour`` mixed with white: ``share`` 1 is the colour, 0 white."""
    c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    return tuple((share * c + (1 - share) * 255) / 255)


# ─────────────────────────────────────────────────────────────────────────────
#  data
# ─────────────────────────────────────────────────────────────────────────────

def record_dates():
    """Every scene date of the cube, 'YYYY-MM-DD'."""
    import xarray as xr

    with xr.open_dataset(CUBE) as ds:
        return np.array([str(np.datetime64(v, "D")) for v in ds["t"].values])


def record_index(dates):
    """Positions, among the cube's dates, of the 97 scenes of MAREA's 2023-2025
    Villaviciosa record (its scene rule; only the date list is read)."""
    with open(RECORD, encoding="utf-8") as f:
        rec = json.load(f)["dates"]
    idx = np.flatnonzero(np.isin(dates, rec))
    if idx.size != len(rec):
        raise SystemExit(f"{len(rec) - idx.size} record date(s) missing from the cube")
    return idx


def boundary_levels(dates):
    """EOT20 at each overpass instant (the production boundary, no delay)."""
    with open(OVERPASS, encoding="utf-8") as f:
        ot = json.load(f)
    times = pd.DatetimeIndex(pd.to_datetime([ot[d] for d in dates]))
    # PyTMDBoundary returns its levels sorted by time
    assert times.is_monotonic_increasing and times.is_unique
    if os.path.exists(CACHE):
        c = pd.read_csv(CACHE)
        if list(c["date"]) == list(dates):
            return times, c["level_m"].to_numpy(float)
    from pyintertidal import sites
    from pyintertidal.boundary import PyTMDBoundary

    lat, lon = sites.get("villaviciosa").centroid      # (lat, lon)
    h = PyTMDBoundary("EOT20", lat, lon, directory="tide_models").levels(times)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.DataFrame({"date": dates,
                  "overpass_utc": times.strftime("%Y-%m-%d %H:%M:%S"),
                  "level_m": h}).to_csv(CACHE, index=False)
    log(f"EOT20 levels cached -> {CACHE}")
    return times, np.asarray(h, float)


def floor0(band):
    """Reflectance floored at 0 (NaN kept): the validation's "floor0"."""
    return np.where(np.isfinite(band), np.maximum(band, 0.0), band)


def pixel_series(row, col):
    """NDWI and the clear-sky flag of one pixel at every record date."""
    import xarray as xr

    from pyintertidal.marea import CLEAR

    with xr.open_dataset(CUBE) as ds:
        g = ds["B03"][:, row, col].values.astype(np.float64)
        n = ds["B08"][:, row, col].values.astype(np.float64)
        scl = ds["SCL"][:, row, col].values
    g, n = floor0(g), floor0(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(g + n != 0, (g - n) / (g + n), np.nan)
    clear = np.isin(scl, CLEAR) & np.isfinite(ndwi)
    return ndwi, clear


def fit_pixel(ndwi, clear, h):
    """The product's flooding-curve fit for one pixel."""
    from pyintertidal.elevation import _fit_block
    from pyintertidal.marea import QUALITY_MIN_BRACKET, SG_GRID, invert_series

    lo, hi = float(h.min()), float(h.max())
    Y = np.nan_to_num(ndwi, nan=0.0)[:, None]
    C = clear.astype(np.float64)[:, None]
    a, b, mu, sg, rmse, n = (float(v[0]) for v in
                             _fit_block(Y, C, h, np.linspace(lo, hi, 50), SG_GRID))
    z_inv, sg_inv = invert_series(Y, C, h, lo, hi,
                                  min_bracket=QUALITY_MIN_BRACKET)
    assert np.isfinite(z_inv[0]) and np.isclose(z_inv[0], mu), (z_inv, mu)
    assert np.isclose(sg_inv[0], sg)
    return dict(a=a, b=b, z=mu, sigma=sg, rmse=rmse, n=int(n))


def select(h):
    """Reprint the ranking the pixel was picked from (streams the cube once)."""
    import xarray as xr

    from pyintertidal.elevation import _fit_block
    from pyintertidal.marea import CLEAR, SG_GRID

    # the intertidal pixels of the 2016-2025 MAREA product, on the cube grid
    # (the product's frame starts 113 rows and 85 columns into this one)
    d = np.load(os.path.join("products_marea", "marea.npz"))
    rg, cg = np.divmod(d["keep"], int(d["shape"][1]))
    rt, ct = rg + 113, cg + 85
    with xr.open_dataset(CUBE) as ds:
        inside = (rt < ds.sizes["y"]) & (ct < ds.sizes["x"])
        rt, ct = rt[inside], ct[inside]
        r0, r1, c0, c1 = rt.min(), rt.max() + 1, ct.min(), ct.max() + 1
        T = ds.sizes["t"]
        Y = np.zeros((T, len(rt)), np.float32)
        C = np.zeros((T, len(rt)), bool)
        for i in range(T):
            g = floor0(ds["B03"][i, r0:r1, c0:c1].values[rt - r0, ct - c0].astype(np.float64))
            n = floor0(ds["B08"][i, r0:r1, c0:c1].values[rt - r0, ct - c0].astype(np.float64))
            s = ds["SCL"][i, r0:r1, c0:c1].values[rt - r0, ct - c0]
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(g + n != 0, (g - n) / (g + n), np.nan)
            C[i] = np.isin(s, CLEAR) & np.isfinite(y)
            Y[i] = np.nan_to_num(y, nan=0.0)
    lo, hi = float(h.min()), float(h.max())
    out = np.zeros((6, len(rt)))
    for j in range(0, len(rt), 4000):
        s = slice(j, min(j + 4000, len(rt)))
        out[:, s] = np.vstack(_fit_block(Y[:, s].astype(np.float64),
                                         C[:, s].astype(np.float64), h,
                                         np.linspace(lo, hi, 50), SG_GRID))
    a, b, mu, sg, rmse, N = out
    wet = C & (Y > 0)
    agree = (((h[:, None] > mu[None, :]) == wet) & C).sum(0) / np.maximum(C.sum(0), 1)
    below = (C & (h[:, None] < mu[None, :])).sum(0)
    above = (C & (h[:, None] > mu[None, :])).sum(0)
    wf = wet.sum(0) / np.maximum(C.sum(0), 1)
    ok = ((N >= 120) & (b > 0.15) & (b < 2.5) & (np.abs(a) < 2.5)
          & (wf > 0.3) & (wf < 0.7) & (below >= 25) & (above >= 25))
    idx = np.flatnonzero(ok)
    for j in idx[np.argsort(-agree[idx])][:20]:
        log(f"row {rt[j]} col {ct[j]}  looks {int(N[j])}  wf {wf[j]:.2f}  "
            f"z {mu[j]:+.2f} m  sigma {sg[j]:.2f} m  agreement {agree[j]:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
#  figure
# ─────────────────────────────────────────────────────────────────────────────

# Print sizes, mm; type in points at print size. The axes take the left part
# of the figure (``plot_w`` mm after a ``left`` margin for the index axis);
# the keys and the rule stand in a column to their right.
PRESETS = {
    "half": dict(stem="pixel_gauge", w=391.0, h=57.5, left=41.0, plot_w=227.0, bottom=20.5,
                 top=2.0, fs=22, fs_tick=21, fs_rule=24, dot=56, mid_dot=220, curve_lw=3.4,
                 ylim=(-0.47, 1.16), lead=9.4, ylab_flat=True),
    "strip": dict(stem="pixel_gauge_strip", w=331.0, h=62.0, left=26.0, plot_w=203.0, bottom=21.5,
                  top=2.5, fs=22, fs_tick=21, fs_rule=23, dot=60, mid_dot=230, curve_lw=3.4,
                  ylim=(-0.47, 1.16)),
    "slot": dict(stem="pixel_gauge_slot", w=240.0, h=88.0, left=23.0, plot_w=140.0, bottom=18.0,
                 top=2.5, fs=21, fs_tick=21, fs_rule=22, dot=60, mid_dot=200, curve_lw=3.2,
                 ylim=(-0.47, 1.18)),
    "wide": dict(stem="pixel_gauge_wide", w=250.0, h=150.0, left=25.0, plot_w=150.0,
                 bottom=20.0, top=4.0, fs=23, fs_tick=22, fs_rule=24, dot=90, mid_dot=250,
                 curve_lw=4.0, ylim=(-0.47, 1.18)),
}
X0, X1 = -2.35, 1.45            # tide axis, m above mean sea level


class Labels:
    """Every label set on the figure, for the overlap check."""

    def __init__(self, fig):
        self.fig = fig
        self.items = []

    def add(self, artist, name):
        """Register a text artist under ``name``; returns the artist."""
        self.items.append((artist, name))
        return artist

    def check(self):
        """Report labels that leave the figure or overlap; returns the number of overlaps."""
        r = self.fig.canvas.get_renderer()
        W = self.fig.get_figwidth() * self.fig.dpi
        H = self.fig.get_figheight() * self.fig.dpi
        boxes = []
        for a, name in self.items:
            if not a.get_text().strip():
                continue
            bb = a.get_window_extent(renderer=r)
            boxes.append((name, bb))
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
                print(f"  LEAVES FIGURE: {name}")
        n = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                if ox > 1 and oy > 1:
                    print(f"  OVERLAP: {boxes[i][0]!r} / {boxes[j][0]!r}")
                    n += 1
        print(f"  label check: {len(boxes)} labels, {n} overlap(s)")
        return n


def R(s):
    """A text run in the regular face, for ``runs``."""
    return (s, "normal", "normal")


def I(s):
    """A text run in italic (symbols such as h and z)."""
    return (s, "normal", "italic")


def B(s):
    """A text run in semibold."""
    return (s, "semibold", "normal")


def BI(s):
    """A text run in semibold italic."""
    return (s, "semibold", "italic")


def draw(p, h, ndwi, clear, fit, pdf_path):
    """Draw the figure at the preset ``p``: the pixel's looks against the mouth
    tide (wet and dry), its fitted flooding curve, the midpoint z and the width
    band, the keys and the rule; write the PDF at ``pdf_path`` and return the
    number of overlapping labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from scipy.special import erf, erfinv

    import b_common as bc

    bc.use_poster_fonts()
    wet = clear & (ndwi > 0)
    dry = clear & ~(ndwi > 0)
    z, a, b, sg = fit["z"], fit["a"], fit["b"], fit["sigma"]
    W, H, fs = p["w"], p["h"], p["fs"]
    L, PW, Bm, T = p["left"], p["plot_w"], p["bottom"], p["top"]
    PH = H - Bm - T

    fig = plt.figure(figsize=(W * MM, H * MM))
    r = fig.canvas.get_renderer()
    Tmm = mtransforms.Affine2D().scale(MM) + fig.dpi_scale_trans    # mm from bottom-left
    lab = Labels(fig)

    def width_mm(s, size, weight="normal", style="normal"):
        """Width (mm) of a text in this face, measured on the figure."""
        t = fig.text(0, 0, s, fontsize=size, fontweight=weight, fontstyle=style)
        w = t.get_window_extent(renderer=r).width / fig.dpi * 25.4
        t.remove()
        return w

    def runs(x, y, parts, size, colour=INK, ha="left", va="baseline", name=""):
        """Left-to-right runs [(text, weight, style)] at (x, y) mm from the bottom-left."""
        ws = [width_mm(s, size, w, st) for s, w, st in parts]
        x0 = {"left": x, "center": x - sum(ws) / 2, "right": x - sum(ws)}[ha]
        for (s, w, st), dw in zip(parts, ws):
            t = fig.text(x0, y, s, transform=Tmm, fontsize=size, fontweight=w, fontstyle=st,
                         color=colour, ha="left", va=va, zorder=8)
            lab.add(t, name or s)
            x0 += dw
        return x0

    ax = fig.add_axes([L / W, Bm / H, PW / W, PH / H])
    y0, y1 = p["ylim"]
    ax.set_xlim(X0, X1)
    ax.set_ylim(y0, y1)
    bc.style_axes(ax, p["fs_tick"])
    to_mm = ax.transData + Tmm.inverted()                            # data -> mm
    mmx = PW / (X1 - X0)

    # the width of the step: the uncertainty of z (z - sigma .. z + sigma)
    ax.add_patch(Rectangle((z - sg, y0), 2 * sg, y1 - y0, facecolor=tint(TEAL, 0.18),
                           edgecolor="none", zorder=0.5))
    # the wet/dry cut, a hairline
    ax.axhline(0.0, color=RULE, lw=1.6, zorder=1)
    # the looks: filled dots with a white ring so overlaps stay legible
    kw = dict(s=p["dot"], linewidths=0.9, edgecolors="white", zorder=3)
    ax.scatter(h[dry], ndwi[dry], color=SAND, **kw)
    ax.scatter(h[wet], ndwi[wet], color=TEAL, **kw)
    # the flooding curve the product fits, index = a + b * Phi((h - z) / sigma)
    hh = np.linspace(X0, X1, 800)
    curve = a + b * 0.5 * (1.0 + erf((hh - z) / (sg * np.sqrt(2.0))))
    ax.plot(hh, curve, color=DEEP, lw=p["curve_lw"], solid_capstyle="round", zorder=4)
    # its midpoint: the elevation
    ym = a + 0.5 * b
    ax.plot([z, z], [y0, ym], color=INK, lw=2.0, ls=(0, (4, 3)), zorder=2)
    ax.scatter([z], [ym], s=p["mid_dot"], color=INK, edgecolors="white", linewidths=2.2,
               zorder=6)

    # axes in plain words; tick labels in Fira (U+2212 is Fira's own minus);
    # the foot of the dashed line carries an italic z among the tide ticks
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_xticklabels(["−2", "−1", "0", "1"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    # the tide axis is named in runs (h in italic), centred under the axes. "Mouth
    # tide" (2026-10-19, round 3): the flow chart's legend, on the same row, defines
    # h as the tide at the mouth; the rule column's "3 km from the mouth: delay
    # tau ~ 0" then says why the mouth tide is this pixel's water
    runs(L + PW / 2, 2.1, [R("mouth tide when the image is taken, "), I("h"),
                           R(" (m above mean sea level)")], fs, ha="center", name="x label")
    if p.get("ylab_flat"):
        # a low strip has no room for a turned label: two flat lines left of the ticks
        ymid = Bm + PH / 2
        for txt, dy in (("water", 2.6), ("index", -5.2)):
            lab.add(fig.text(1.0, ymid + dy, txt, transform=Tmm, fontsize=fs, color=INK,
                             ha="left", va="baseline"), "y label " + txt)
    else:
        lab.add(ax.set_ylabel("water index", fontsize=fs, color=INK, labelpad=3), "y label")
    fig.canvas.draw()
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        lab.add(t, "tick " + t.get_text())
    zx, _ = to_mm.transform((z, y0))
    bb = ax.get_xticklabels()[0].get_window_extent(renderer=r)
    tick_base = bb.y0 / fig.dpi * 25.4 + 0.22 * p["fs_tick"] * PT     # tick-label baseline, mm
    fig.add_artist(Line2D([zx, zx], [Bm, Bm - 2.4], transform=Tmm, color=INK, lw=1.4))
    lab.add(fig.text(zx, tick_base, "z", transform=Tmm, fontsize=p["fs_tick"] + 1, color=INK,
                     fontstyle="italic", fontweight="semibold", ha="center", va="baseline"),
            "z tick")

    # --- the step, named on its empty (left) side, right-aligned short of it
    zxm, ymm = to_mm.transform((z, ym))
    xr_ = zx - sg * mmx - 4.2                       # right edge of the two labels, mm
    # width: a double arrow across the band at the top (named in the column)
    yw_mm = Bm + PH - 3.6
    inv = to_mm.inverted()
    ax.add_patch(FancyArrowPatch(tuple(inv.transform((zx - sg * mmx, yw_mm))),
                                 tuple(inv.transform((zx + sg * mmx, yw_mm))),
                                 arrowstyle="<|-|>,head_length=2.6,head_width=1.3",
                                 mutation_scale=1.0, color=INK, lw=1.3, shrinkA=0, shrinkB=0,
                                 zorder=8, clip_on=False))
    # the curve, named at its upper bend
    yc_mm = ymm + 0.5 * (Bm + PH - ymm) + 1.0
    q = 0.90                                        # the leader ends on the upper bend
    xq_mm, yq_mm = to_mm.transform((z + sg * np.sqrt(2.0) * erfinv(2.0 * q - 1.0), a + q * b))
    runs(xr_, yc_mm, [R("flooding curve")], fs, ha="right", va="center_baseline",
         name="flooding curve")
    fig.add_artist(Line2D([xr_ + 0.8, xq_mm - 0.9], [yc_mm, yq_mm + 0.5], transform=Tmm,
                          color=SLATE, lw=1.4, zorder=7))
    # the midpoint
    runs(xr_, ymm, [B("midpoint = elevation "), BI("z")], fs, ha="right", va="center_baseline",
         name="midpoint")
    fig.add_artist(Line2D([xr_ + 0.8, zxm - 2.6], [ymm, ymm], transform=Tmm, color=SLATE,
                          lw=1.4, zorder=7))

    # --- the column beside the plot: the keys (dots, the pale band), then the rule
    cx = L + PW + 7.0
    lead = p.get("lead", 1.30 * fs * PT)
    ytop = H - 2.2 - 0.72 * fs * PT
    keys = ((TEAL, "dot", [B("wet"), R(": index above 0")]),
            (SAND, "dot", [B("dry"), R(": index 0 or below")]),
            (tint(TEAL, 0.18), "band", [R("width = uncertainty of "), I("z")]))
    for k, (col, kind, parts) in enumerate(keys):
        yk = ytop - k * lead
        if kind == "dot":
            fig.add_artist(Line2D([cx + 1.8], [yk], transform=Tmm, marker="o", ms=8.5, mfc=col,
                                  mec="white", mew=1.2, ls="", zorder=8))
        else:
            fig.add_artist(Rectangle((cx + 0.2, yk - 2.6), 3.2, 5.2, transform=Tmm,
                                     facecolor=col, edgecolor="none", zorder=8))
        runs(cx + 5.2, yk, parts, fs, va="center_baseline", name=f"key {k}")
    # the rule, in the flow chart's symbols: 3 km from the mouth the delay is
    # negligible, so the water at the pixel is the tide at the mouth, h(t - tau) ~ h(t)
    yr = ytop - 3 * lead - 0.30 * lead
    runs(cx, yr, [B("wet when "), BI("h"), B(" > "), BI("z")], p["fs_rule"], colour=DEEP,
         va="center_baseline", name="rule 1")
    runs(cx, yr - 1.0 * lead, [R("3 km from the mouth: delay "), I("τ"), R(" ≈ 0")], fs,
         colour=SLATE, va="center_baseline", name="rule 2")

    fig.savefig(pdf_path)
    n = lab.check()
    right = max(a_.get_window_extent(renderer=r).x1 for a_, _ in lab.items) / fig.dpi * 25.4
    print(f"  rightmost label ends at {right:.1f} of {W:.0f} mm")
    plt.close(fig)
    return n


def to_png(pdf_path, dpi=300):
    """300 dpi PNG of the vector page, rasterised by poppler's cairo backend
    (MiKTeX's pdftocairo; its Splash backend, pdftoppm, leaves seams on some
    fill edges)."""
    stem = os.path.splitext(pdf_path)[0]
    exe = os.path.join(MIKTEX, "pdftocairo.exe")
    subprocess.run([exe, "-r", str(dpi), "-png", "-singlefile", pdf_path, stem],
                   check=True, stderr=subprocess.DEVNULL)
    return stem + ".png"


def main():
    """Read the pixel's record and its levels, fit its flooding curve and draw
    the figure at the chosen preset (``--select`` reprints the ranking the
    pixel was picked from instead)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--select", action="store_true",
                    help="reprint the ranking the pixel was picked from")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="half",
                    help="print size: half 391 x 57.5 mm (default), strip 331 x 62 mm, "
                         "slot 240 x 88 mm, wide 250 x 150 mm")
    args = ap.parse_args()
    p = PRESETS[args.preset]

    t0 = time.time()
    dates = record_dates()
    times, h = boundary_levels(dates)
    if args.select:
        select(h)
        return
    ndwi, clear = pixel_series(ROW, COL)
    rec = record_index(dates)
    log(f"MAREA's 2023-2025 record: {rec.size} of the cube's {len(dates)} scenes")
    dates, times, h, ndwi, clear = dates[rec], times[rec], h[rec], ndwi[rec], clear[rec]
    fit = fit_pixel(ndwi, clear, h)
    wet = clear & (ndwi > 0)
    agree = np.mean(((h > fit["z"]) == wet)[clear])
    log(f"pixel row {ROW} col {COL}: {len(dates)} record dates "
        f"({dates[0]} .. {dates[-1]}), {int(clear.sum())} clear looks "
        f"({int(wet.sum())} wet, {int((clear & ~wet).sum())} dry)")
    log(f"  level range of the record {h.min():+.2f} .. {h.max():+.2f} m")
    log(f"  fit: z {fit['z']:+.3f} m  sigma {fit['sigma']:.2f} m  a {fit['a']:+.3f}  "
        f"b {fit['b']:.3f}  rmse {fit['rmse']:.3f}  (n {fit['n']})")
    log(f"  looks that follow 'wet when the level is above z': {agree:.1%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, p["stem"] + ".pdf")
    draw(p, h, ndwi, clear, fit, pdf)
    png = to_png(pdf)
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = f", peak memory {getattr(mi, 'peak_wset', mi.rss) / 2**20:.0f} MB"
    except ImportError:
        peak = ""
    log(f"wrote {pdf} and {png} ({p['w']:.0f} x {p['h']:.0f} mm) "
        f"[{time.time() - t0:.0f} s{peak}]")


if __name__ == "__main__":
    main()
