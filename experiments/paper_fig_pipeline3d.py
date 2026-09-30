# -*- coding: utf-8 -*-
"""The pipeline as a 3-D workflow schematic (docs/paper/figures/fig_pipeline.png),
in the style of LogiTide2DEM's Fig. 2, on the Ría de Villaviciosa:

(a) six real Sentinel-2 scenes of the archive drawn as a skewed stack;
(b) the same stack read wet/dry (NDWI > 0, clouds masked);
(c) one pixel's record: wet/dry against the water level at the mouth at
    each visit, with the fitted transition Φ((h − z)/σ) and its z;
(d) the interior clock, our addition: the bands of distance from the mouth
    and the lag fitted per band with its bootstrap interval;
(e) the MAREA elevation map as a tile.

Everything is read from the cube and the products already on disk through
the helpers of ``experiments.paper_fig_pipeline`` / ``paper_fig_method``;
nothing is recomputed except the one-pixel fit of (c).

Run from the repository root:  python -m experiments.paper_fig_pipeline3d
Optional: MAREA_OVERPASS_CACHE=<json> points (c) at an overpass-time cache
when products_marea/overpass_times.json is not usable; ``nominal`` uses the
nominal overpass hour instead of querying the catalogue.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from scipy.stats import norm  # noqa: E402

import experiments.paper_fig_pipeline as P  # noqa: E402
from experiments.paper_fig_pipeline import (  # noqa: E402
    CUBE, EXTRACT, INK, MASK_TIF, NDWI_THRESHOLD, RED, TIDE, WF_TIF,
    composite, crop_box, marea_rasters, read, scene_ndwi)

OUT = "docs/paper/figures/fig_pipeline.png"
# clear scenes (≥ 95 % clear intertidal pixels, ≥ 99 % clear crop) spread
# over the archive, at different tide states; oldest at the back of the stack
DATES = ["2016-10-14", "2018-10-04", "2019-03-23", "2022-03-10", "2024-03-21", "2025-10-12"]
YEARS = ("2016", "2025")
NOMINAL_OVERPASS = "11:21"          # Sentinel-2 over Asturias, UTC, if no real times
SITE = "villaviciosa"
OVERPASS_JSON = "products_marea/overpass_times.json"
SITE_NAME = "Ría de Villaviciosa"

# the same figure on a comparison site: its cube, record and products
SITES = {
    "escalda": dict(name="Westerschelde", cube="ndwi_cube_escalda_2023-2025_20m.nc",
                    extract="products_escalda/marea_extract.npz",
                    marea="products_escalda/marea_eot20_g2/marea.npz",
                    wf="products_escalda/water_frequency.tif", mask="products_escalda/intertidal_mask.tif",
                    overpass="products_escalda/overpass_times.json", years=("2023", "2025"),
                    nominal="10:50",
                    # clear scenes (≥ 95 % clear record) at different tide states, oldest first
                    dates=["2023-02-14", "2023-06-14", "2023-09-07", "2024-03-08", "2025-02-23", "2025-08-12"]),
}


def configure(site):
    """Point every input at ``site`` (the module defaults are Villaviciosa)."""
    global CUBE, EXTRACT, MASK_TIF, WF_TIF, DATES, YEARS, NOMINAL_OVERPASS, SITE, OVERPASS_JSON, SITE_NAME
    if site == "villaviciosa":
        return
    S = SITES[site]
    SITE, SITE_NAME = site, S["name"]
    CUBE, EXTRACT, MASK_TIF, WF_TIF = S["cube"], S["extract"], S["mask"], S["wf"]
    DATES, YEARS, NOMINAL_OVERPASS, OVERPASS_JSON = S["dates"], S["years"], S["nominal"], S["overpass"]
    P.CUBE, P.EXTRACT, P.MAREA = S["cube"], S["extract"], S["marea"]     # the helpers read these

# the 3-D look: every tile is squashed vertically and sheared to the right
SQUASH, SKEW = 0.70, 0.40
FIG_W, FIG_H = 7.2, 4.0             # inches; the data units of every axes are inches
MUD = "#b08968"                     # a visit read dry (as in fig_method)
LAND, SEA = "#f0f0f0", "#dde8ee"    # tile backdrops
BAND_COLOURS = ["#7fbbd8", "#2a6f97", "#0d2f45"]   # mouth → interior, light → dark
CLOUD = "#c9c9c9"

fp = dict(color=INK)


# ── data ────────────────────────────────────────────────────────────────
def overpass_times(dates, bbox):
    """{date: datetime} for the visits, from the first usable source:
    MAREA_OVERPASS_CACHE, the repository cache, the STAC catalogue; the
    nominal overpass hour is the last resort."""
    import pandas as pd

    def load(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
            return d if isinstance(d, dict) and len(d) > 100 else None
        except (OSError, ValueError):
            return None

    wanted = os.environ.get("MAREA_OVERPASS_CACHE", "")
    if wanted == "nominal":                # offline layout runs
        print(f"  overpass times: the nominal {NOMINAL_OVERPASS} UTC (MAREA_OVERPASS_CACHE=nominal)")
        return {d: pd.Timestamp(f"{d} {NOMINAL_OVERPASS}") for d in dates}
    for path in (wanted, OVERPASS_JSON):
        if path and (times := load(path)) is not None:
            print(f"  overpass times from {path}")
            return {k: pd.Timestamp(v) for k, v in times.items()}
    try:
        import pyintertidal as pit
        pit.net.use_system_certificates()
        times = pit.overpass.get_overpass_times(bbox, (f"{YEARS[0]}-01-01", f"{YEARS[1]}-12-31"), verbose=False)
        print("  overpass times from the STAC catalogue")
        return {k: pd.Timestamp(v) for k, v in times.items()}
    except Exception as exc:          # offline: the nominal hour, a few minutes off at most
        print(f"  overpass times unavailable ({exc!r}); using the nominal {NOMINAL_OVERPASS} UTC")
        return {d: pd.Timestamp(f"{d} {NOMINAL_OVERPASS}") for d in dates}


def pixel_record():
    """One Villaviciosa pixel: the level at the mouth at each clear visit,
    its NDWI, and the fitted transition (z, sigma). ``paper_fig_method
    .one_pixel`` when its cache is in place, otherwise the same selection
    with the overpass times resolved here."""
    if SITE == "villaviciosa" and json_ok(OVERPASS_JSON):
        from experiments.paper_fig_method import one_pixel
        return one_pixel()
    import pandas as pd
    from pyintertidal import marea
    from pyintertidal.boundary import PyTMDBoundary
    d = np.load(EXTRACT, allow_pickle=True)
    Y, C, keep = d["Y"], d["C"].astype(bool), d["keep"]
    dates, bbox = np.array([str(x) for x in d["dates"]]), d["bbox"]
    bb = bbox.item() if bbox.dtype == object else \
        {"west": float(bbox[0]), "south": float(bbox[1]), "east": float(bbox[2]), "north": float(bbox[3])}
    bb = {k: bb[k] for k in ("west", "south", "east", "north")}
    times = overpass_times(dates, bb)
    has = np.array([x in times for x in dates])
    t = pd.DatetimeIndex([times[x] for x in dates[has]])
    t = t.tz_localize(None) if t.tz is not None else t
    eot = PyTMDBoundary("EOT20", 0.5 * (bb["south"] + bb["north"]), 0.5 * (bb["west"] + bb["east"]),
                        directory="tide_models")
    h = np.asarray(eot.levels(t), float)
    Yh, Ch = Y[has], C[has]
    rng = np.random.default_rng(0)                     # the same pixel as one_pixel()
    cand = np.sort(rng.choice(Yh.shape[1], size=min(2000, Yh.shape[1]), replace=False))
    n_clear = Ch[:, cand].sum(0)
    z_all, s_all = marea.invert_series(np.nan_to_num(Yh[:, cand], nan=0.0).astype(np.float64),
                                       Ch[:, cand].astype(np.float64), h)
    ok = (np.isfinite(z_all) & (n_clear >= 150)
          & (z_all > np.nanpercentile(h, 25)) & (z_all < np.nanpercentile(h, 75)))
    pick = np.flatnonzero(ok)
    jj = int(pick[len(pick) // 2]); j = int(cand[jj])
    m = Ch[:, j] & np.isfinite(h)
    y, hh = Yh[m, j].astype(float), h[m]
    z, s = float(z_all[jj]), float(s_all[jj])
    phi = norm.cdf((hh - z) / s)
    a, b = np.linalg.lstsq(np.column_stack([np.ones_like(phi), phi]), y, rcond=None)[0]
    return dict(h=hh, ndwi=y, z=z, sigma=s, a=a, b=b, n=int(m.sum()), pixel=int(keep[j]))


def json_ok(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
        return isinstance(d, dict) and len(d) > 100
    except (OSError, ValueError):
        return False


def to_rgba(rgb, fade=1.0):
    """RGB → opaque RGBA, blended towards white by (1 − fade) for the back
    layers of a stack (alpha would let the tiles behind show through)."""
    out = np.ones(rgb.shape[:2] + (4,))
    out[..., :3] = 1.0 - fade * (1.0 - np.clip(rgb, 0, 1))
    return out


def paint(classes, palette, fade=1.0):
    """Class raster (int, −1 = backdrop) → RGBA with one colour per class."""
    out = np.ones(classes.shape + (4,))
    for k, c in palette.items():
        out[classes == k, :3] = colors.to_rgb(c)
    out[..., :3] = 1.0 - fade * (1.0 - out[..., :3])
    return out


# ── drawing ─────────────────────────────────────────────────────────────
def tile_transform(shape, x0, y0, width):
    """Affine placing an image of ``shape`` as a sheared tile of the given
    width (inches) whose bottom-left corner is at (x0, y0) inches."""
    H, W = shape[:2]
    scale = width / (W + SKEW * SQUASH * H)
    tr = Affine2D().scale(scale, scale * SQUASH).skew(np.arctan(SKEW), 0).translate(x0, y0)
    return tr, scale * SQUASH * H


def draw_tile(ax, rgba, x0, y0, width, zorder=1, lw=0.5):
    tr, h = tile_transform(rgba.shape, x0, y0, width)
    H, W = rgba.shape[:2]
    ax.imshow(rgba, extent=(0, W, 0, H), transform=tr + ax.transData,
              interpolation="bilinear", zorder=zorder)
    ax.add_patch(Polygon([[0, 0], [W, 0], [W, H], [0, H]], closed=True, fill=False,
                         ec=INK, lw=lw, transform=tr + ax.transData, zorder=zorder + 0.5))
    return h


def draw_stack(ax, tiles, x0, y_bottom, width, step, labels=None):
    """Tiles from back (top) to front (bottom); returns the tile height."""
    n = len(tiles)
    h = None
    for i, rgba in enumerate(tiles):
        y = y_bottom + (n - 1 - i) * step
        h = draw_tile(ax, rgba, x0, y, width, zorder=i + 1)
        if labels is not None and labels[i]:      # by the bottom-left corner, the exposed part of every layer
            ax.text(x0 - 0.04, y + 0.1, labels[i], ha="right", va="center", fontsize=6, **fp)
    return h


def inch_axes(fig, x0, y0, w, h):
    """An axes whose data coordinates are inches of the figure."""
    ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return ax


def plot_axes(fig, x0, y0, w, h):
    ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6); ax.spines[side].set_color(INK)
    ax.tick_params(labelsize=6, length=2, width=0.5, colors=INK, pad=1.5)
    return ax


def title(fig, x, y, letter, main, sub):
    return [fig.text(x / FIG_W, y / FIG_H, f"({letter}) ", fontsize=7, fontweight="bold", ha="left", va="top", **fp),
            fig.text((x + 0.19) / FIG_W, y / FIG_H, main, fontsize=7, ha="left", va="top", **fp),
            fig.text(x / FIG_W, (y - 0.145) / FIG_H, sub, fontsize=6, ha="left", va="top", color="0.35")]


def arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0 / FIG_W, y / FIG_H), (x1 / FIG_W, y / FIG_H),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, color=INK, lw=0.9, shrinkA=0, shrinkB=0))


def main():
    wf, transform, _ = read(WF_TIF)
    mask = read(MASK_TIF)[0] > 0
    shape = wf.shape
    z, bands, m = marea_rasters(shape)
    rs, cs = crop_box(mask)
    sea = wf[rs, cs] > 0.95
    backdrop = np.where(sea, 1, -1)                     # 1 sea, −1 land

    # (a), (b): the scenes and their wet/dry reading, cropped at once
    scenes, readings = [], []
    n = len(DATES)
    for i, date in enumerate(DATES):
        fade = 0.55 + 0.45 * i / (n - 1)
        rgb, _, _, cshape = composite(CUBE, date)
        assert cshape == shape, (cshape, shape)
        scenes.append(to_rgba(rgb[rs, cs], fade))
        ndwi, clear = scene_ndwi(date)
        cls = np.where(clear[rs, cs], (ndwi[rs, cs] > NDWI_THRESHOLD).astype(int), 2)
        readings.append(paint(cls, {0: "white", 1: TIDE, 2: CLOUD}, fade))
        print(f"  {date}: wet {np.mean(cls[mask[rs, cs]] == 1):.2f} of the intertidal, "
              f"cloud {np.mean(cls == 2):.3f} of the crop")
        del rgb, ndwi, clear

    # (d), (e): bands and elevation tiles
    nb = len(m["tau_usado_min"])
    if nb > len(BAND_COLOURS):                          # more bands than the three preset colours
        BAND_COLOURS[:] = [colors.to_hex(cm.Blues(0.35 + 0.6 * k / max(nb - 1, 1))) for k in range(nb)]
    band_crop = np.where(np.isfinite(bands[rs, cs]), np.nan_to_num(bands[rs, cs], nan=-1), backdrop - 10)
    band_tile = paint(band_crop.astype(int), {**{k: BAND_COLOURS[k] for k in range(nb)},
                                              -11: LAND, -9: SEA, -1: "0.6"})
    zc = z[rs, cs]; rec = np.isfinite(zc)
    vmin, vmax = np.nanpercentile(z[np.isfinite(z)], [2, 98])
    norm_z = colors.Normalize(vmin, vmax)
    z_tile = paint(np.where(rec, -2, backdrop - 10).astype(int), {-11: LAND, -9: SEA})
    z_tile[rec] = cm.viridis(norm_z(zc[rec]))

    # (c): one pixel
    print("one pixel's record ...")
    px = pixel_record()
    wet = px["ndwi"] > NDWI_THRESHOLD
    print(f"  pixel {px['pixel']}: {px['n']} clear visits, z {px['z']:+.2f} m, sigma {px['sigma']:.2f} m, "
          f"{wet.sum()} read wet")

    # ── layout (inches) ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
    y_top, y_bot = 3.35, 0.22           # content band below the titles
    y_mid = 0.5 * (y_top + y_bot)
    xa, wa = 0.30, 1.15                  # (a)
    xb, wb = 1.68, 1.15                  # (b)
    xc, wc = 3.30, 0.92                  # (c) plot
    xd, wd = 4.50, 1.22                  # (d)
    xe, we = 5.95, 1.22                  # (e)
    T = 3.92
    texts = []
    texts += title(fig, 0.05, T, "a", "Multi-temporal images", f"Sentinel-2, {YEARS[0]}–{YEARS[1]}")
    texts += title(fig, xb - 0.08, T, "b", "Wet/dry classification", f"NDWI > {NDWI_THRESHOLD:g}, clouds masked")
    texts += title(fig, xc - 0.24, T, "c", "One pixel's record", "wet/dry vs. level at the mouth")
    texts += title(fig, xd - 0.04, T, "d", "Interior clock", "one lag per band of distance")
    texts += title(fig, xe - 0.22, T, "e", "Intertidal topography", "z per pixel, corrected clock")

    # (a) and (b): the stacks
    _, th = tile_transform(scenes[0].shape, 0, 0, wa)
    step = (y_top - y_bot - th) / (n - 1)
    ax = inch_axes(fig, 0, 0, FIG_W, FIG_H)          # one canvas for every tile
    draw_stack(ax, scenes, xa, y_bot, wa, step, labels=[d[:4] for d in DATES])
    draw_stack(ax, readings, xb, y_bot, wb, step, labels=["t = 1"] + [""] * (n - 2) + ["t = n"])

    # (c): the record
    axc = plot_axes(fig, xc, y_mid - 0.55, wc, 1.1)
    rng = np.random.default_rng(1)
    jit = rng.uniform(-0.05, 0.05, px["h"].size)
    axc.plot(px["h"][~wet], 0 + jit[~wet], "o", ms=2.2, mew=0, color=MUD, alpha=0.75)
    axc.plot(px["h"][wet], 1 + jit[wet], "o", ms=2.2, mew=0, color=TIDE, alpha=0.75)
    hx = np.linspace(px["h"].min() - 0.15, px["h"].max() + 0.15, 300)
    axc.plot(hx, norm.cdf((hx - px["z"]) / px["sigma"]), "-", color=INK, lw=1.1)
    axc.axvline(px["z"], color=RED, lw=0.8, ls=(0, (3, 2)))
    axc.text(px["z"] + 0.07, 0.25, "z", color=RED, fontsize=7, ha="left", va="center", style="italic")
    axc.set_yticks([0, 1]); axc.set_yticklabels(["dry", "wet"])
    axc.set_ylim(-0.18, 1.18)
    axc.set_xticks([t for t in (-3, -2, -1, 0, 1, 2) if hx.min() <= t <= hx.max()])
    axc.set_xlabel("level at the mouth h (m)", fontsize=6.2, labelpad=1.5, **fp)
    axc.text(0.03, 0.80, f"{px['n']} visits", transform=axc.transAxes, fontsize=5.5, ha="left",
             va="center", color="0.35")
    axc.text(0.03, 0.62, "Φ((h − z)/σ)", transform=axc.transAxes, fontsize=6, ha="left", va="center", **fp)

    # (d): the bands tile and the lag per band
    _, td = tile_transform(band_tile.shape, 0, 0, wd)
    draw_tile(ax, band_tile, xd, y_top - td, wd)
    axd = plot_axes(fig, xd + 0.30, y_bot + 0.28, wd - 0.30, y_top - td - y_bot - 0.70)
    centres = np.array([float(np.nanmedian(m["s_km"][m["band"] == k])) for k in range(nb)])
    tau, lo, hi = m["tau_usado_min"], m["tau_lo_min"], m["tau_hi_min"]
    axd.axhline(0, color="0.75", lw=0.5, zorder=0)
    for k in range(nb):
        axd.errorbar(centres[k], tau[k], yerr=[[max(tau[k] - lo[k], 0)], [max(hi[k] - tau[k], 0)]],
                     fmt="o", ms=3.2, mew=0.5, mec=INK, mfc=BAND_COLOURS[k], ecolor=INK, elinewidth=0.6,
                     capsize=1.8, capthick=0.6)
    axd.set_xlim(0, float(np.nanmax(m["s_km"])) + 0.5)
    axd.set_xlabel("distance from the mouth (km)", fontsize=6.2, labelpad=1.5, **fp)
    axd.set_ylabel("lag τ (min)", fontsize=6.2, labelpad=1.5, **fp)

    # (e): the elevation tile and its colour bar
    _, te = tile_transform(z_tile.shape, 0, 0, we)
    draw_tile(ax, z_tile, xe, y_mid - te / 2 + 0.15, we)
    cax = fig.add_axes([(xe + 0.2) / FIG_W, (y_mid - te / 2 - 0.22) / FIG_H, (we - 0.3) / FIG_W, 0.07 / FIG_H])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm_z, cmap="viridis"), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.5, colors=INK, pad=1)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(INK)
    cb.set_label("elevation (m)", fontsize=6.2, labelpad=1, **fp)

    # the flow
    arrow(fig, xa + wa + 0.03, xb - 0.03, y_mid)
    arrow(fig, xb + wb + 0.03, xc - 0.27, y_mid)
    arrow(fig, xc + wc + 0.05, xd - 0.04, y_mid)
    arrow(fig, xd + wd + 0.03, xe - 0.03, y_mid)

    # no text may run past the page: measure the titles on the renderer
    fig.canvas.draw()
    right = max(t.get_window_extent().x1 for t in texts) / fig.dpi
    if right > FIG_W - 0.02:
        print(f"WARNING: a title runs {right - FIG_W:+.2f} in past the right edge")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"-> {OUT}  ({FIG_W} x {FIG_H} in, scenes {', '.join(DATES)}; {nb} bands, "
          f"lags {np.round(tau, 1).tolist()} min)")


if __name__ == "__main__":
    import sys
    configure(sys.argv[1] if len(sys.argv) > 1 else "villaviciosa")
    main()
