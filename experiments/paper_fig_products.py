# -*- coding: utf-8 -*-
"""The products of the Ría de Villaviciosa, for the paper: the elevation
map with zoom windows, the same windows in the national LiDAR at its
native 5 m, and the hydroperiod.

Everything that is not a product is a light, blurred grey rendering of
the clear low-water scene, so the products carry the colour. The MAREA
map is shown on the notebook's intertidal mask (water-frequency window
0.05-0.95 inside the study polygon), the same mask the LiDAR windows use.

(a) MAREA elevation on the intertidal zone, whole site, with the two
    zoom windows marked;
(b), (d) the two windows in the MAREA map;
(c), (e) the same windows in the IGN LiDAR DTM read at its own 5-m grid
    (no resampling), same colour scale, datum offset removed;
(f) the hydroperiod computed from map (a) and the tide series.

Colour code as in the pipeline notebook: viridis for elevation (metres
above the tide-model datum), Blues for the fraction of time submerged.
Writes docs/paper/figures/fig_products.png.
Run from the repository root:  python -m experiments.paper_fig_products
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.windows import from_bounds

matplotlib.use("Agg")

from experiments.paper_fig_pipeline import (EXTRACT, LIDAR, MASK_TIF, WF_TIF, crop_box, low_water_scene,  # noqa: E402
                                            marea_rasters, read, tide_series)
from experiments.paper_style import INK, clean, grey_background, scale_bar  # noqa: E402

CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"
OUT = "docs/paper/figures/fig_products_villaviciosa.png"
WIN = 200                     # zoom window, pixels of the 10-m grid (2 km)


def zoom_windows(mask, win=WIN, n=2):
    """The ``n`` non-overlapping windows with the most intertidal pixels."""
    H, W = mask.shape
    cs = mask.astype(np.int32).cumsum(0).cumsum(1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    best, taken = [], []
    step = 20
    cand = []
    for r in range(0, H - win, step):
        for c in range(0, W - win, step):
            s = cs[r + win, c + win] - cs[r, c + win] - cs[r + win, c] + cs[r, c]
            cand.append((int(s), r, c))
    cand.sort(reverse=True)
    for s, r, c in cand:
        if all(abs(r - r2) >= win or abs(c - c2) >= win for r2, c2 in taken):
            taken.append((r, c)); best.append((r, r + win, c, c + win))
            if len(best) == n:
                break
    return best


def rgba(arr, cmap, vmin, vmax):
    """A product as RGBA, transparent where undefined, so that it can be
    resampled with anti-aliasing without bleeding into its surroundings."""
    norm = matplotlib.colors.Normalize(vmin, vmax)
    out = plt.get_cmap(cmap)(norm(np.nan_to_num(arr, nan=vmin)))
    out[..., 3] = np.isfinite(arr).astype(float)
    return out


def north(ax, shape):
    x = shape[1] * 0.06
    ax.annotate("N", xy=(x, shape[0] * 0.04), xytext=(x, shape[0] * 0.12), color=INK, fontsize=7,
                fontweight="bold", ha="center", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))


def main():
    from pyintertidal.hydroperiod import hydroperiod
    import pyintertidal as pit
    pit.net.use_system_certificates()

    wf, transform, crs = read(WF_TIF)
    mask = read(MASK_TIF)[0] > 0                       # the notebook's intertidal zone
    shape = wf.shape
    px_m = abs(transform.a)
    z, _, _ = marea_rasters(shape)
    z = np.where(mask, z, np.nan)                      # the map on the notebook's zone only
    date, _, _ = low_water_scene(EXTRACT)
    back = grey_background(CUBE, date)
    d = np.load(EXTRACT, allow_pickle=True); bbox = d["bbox"]
    bb = {"west": float(bbox[0]), "south": float(bbox[1]), "east": float(bbox[2]), "north": float(bbox[3])}
    _, h_h, _ = tide_series(bb)
    hydro = hydroperiod(z, h_h)
    rec = np.isfinite(z)
    vmin, vmax = np.nanpercentile(z[rec], [2, 98])
    wins = zoom_windows(mask)
    rs, cs = crop_box(mask, margin=35)                 # the ría fills the panel
    crop = (slice(rs.start, rs.stop), slice(cs.start, cs.stop))
    cshape = (rs.stop - rs.start, cs.stop - cs.start)

    # the LiDAR at its own grid, per window: bounds of the window in the
    # cube CRS -> the LiDAR CRS -> a rasterio window read at 5 m
    src = rasterio.open(LIDAR)
    to_lidar = Transformer.from_crs(crs, src.crs, always_xy=True)
    lidar_on_grid = pit.reproject_to_grid(LIDAR, transform, crs, shape)     # only for the datum offset
    offset = float(np.nanmedian(lidar_on_grid[rec]) - np.nanmedian(z[rec]))

    fig = plt.figure(figsize=(7.2, 3.45), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.45, 1, 1, 1.45], height_ratios=[1, 1])
    # (a) the whole site
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(back[crop], interpolation="antialiased")
    ax.imshow(rgba(z[crop], "viridis", vmin, vmax), interpolation="antialiased")
    for k, (r0, r1, c0, c1) in enumerate(wins):
        ax.add_patch(Rectangle((c0 - cs.start, r0 - rs.start), c1 - c0, r1 - r0, fill=False, ec=INK, lw=0.9))
        ax.text(c0 - cs.start + 6, r0 - rs.start + 24, "bd"[k], color=INK, fontsize=7, fontweight="bold")
    scale_bar(ax, cshape, px_m, km=2.0, colour="white", halo=INK)
    north(ax, cshape)
    clean(ax)
    ax.set_title(f"(a) MAREA elevation, intertidal zone\n    (low-water scene of {date} in grey)", loc="left", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=matplotlib.colors.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, shrink=0.8, location="bottom")
    cb.set_label("elevation (m, tide-model datum)", fontsize=6); cb.ax.tick_params(labelsize=6)
    # zoom windows: MAREA (row 0) and native LiDAR (row 1)
    labels = [("(b)", "(c)"), ("(d)", "(e)")]
    for k, (r0, r1, c0, c1) in enumerate(wins):
        axm = fig.add_subplot(gs[0, 1 + k])
        axm.imshow(back[r0:r1, c0:c1], interpolation="antialiased")
        axm.imshow(np.ma.masked_invalid(z[r0:r1, c0:c1]), cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        axm.set_title(f"{labels[k][0]} MAREA, window {'bd'[k]}", loc="left", fontsize=7)
        clean(axm)
        # the LiDAR at 5 m over the same ground
        xs = transform.c + np.array([c0, c1]) * transform.a
        ys = transform.f + np.array([r0, r1]) * transform.e
        lx, ly = to_lidar.transform(xs, ys)
        win = from_bounds(min(lx), min(ly), max(lx), max(ly), src.transform)
        lid = src.read(1, window=win).astype(float)
        lid[lid < -100] = np.nan
        lid = lid - offset
        # shown on the intertidal zone only, like the MAREA map (the mask,
        # 10 m, is repeated to the LiDAR's 5-m cells of the window)
        m10 = mask[r0:r1, c0:c1]
        fy, fx = lid.shape[0] / m10.shape[0], lid.shape[1] / m10.shape[1]
        iy = np.clip((np.arange(lid.shape[0]) / fy).astype(int), 0, m10.shape[0] - 1)
        ix = np.clip((np.arange(lid.shape[1]) / fx).astype(int), 0, m10.shape[1] - 1)
        lid = np.where(m10[np.ix_(iy, ix)], lid, np.nan)
        axl = fig.add_subplot(gs[1, 1 + k])
        axl.imshow(back[r0:r1, c0:c1], interpolation="antialiased", extent=(0, c1 - c0, r1 - r0, 0))
        axl.imshow(np.ma.masked_invalid(lid), cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=(0, c1 - c0, r1 - r0, 0))
        axl.set_title(f"{labels[k][1]} LiDAR 5 m, window {'bd'[k]}", loc="left", fontsize=7)
        clean(axl)
        if k == 0:
            scale_bar(axm, (r1 - r0, c1 - c0), px_m, km=0.5, colour="white", halo=INK, at=(0.05, 0.93))
    src.close()
    # (f) hydroperiod, whole site
    ax = fig.add_subplot(gs[:, 3])
    ax.imshow(back[crop], interpolation="antialiased")
    ax.imshow(rgba(hydro[crop], "Blues", 0, 1), interpolation="antialiased")
    scale_bar(ax, cshape, px_m, km=2.0, colour="white", halo=INK)
    clean(ax)
    ax.set_title("(f) hydroperiod from (a)", loc="left", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, shrink=0.8, location="bottom")
    cb.set_label("fraction of time submerged", fontsize=6); cb.ax.tick_params(labelsize=6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"-> {OUT}  (scene {date}; windows {wins}; LiDAR offset {offset:+.2f} m; z range {vmin:.2f}..{vmax:.2f}; "
          f"{int(rec.sum()):,} MAREA px on the notebook's zone)")


if __name__ == "__main__":
    main()
