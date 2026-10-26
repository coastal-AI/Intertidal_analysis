# -*- coding: utf-8 -*-
"""The products of a Dutch site (default: the Wadden Sea behind the Vlie),
for the paper: the MAREA elevation map with zoom windows, the same windows
in the Rijkswaterstaat Vaklodingen (the official bathymetry/topography
survey, one value per 20-m cell, NAP datum) and the hydroperiod.

Everything that is not a product is a light grey rendering of the clear
low-water scene of the cube, so the products carry the colour. The MAREA
map is shown on the notebook's intertidal mask (water-frequency window
0.05-0.95 inside the study box), the mask the Vaklodingen windows use.

(a) MAREA elevation on the intertidal zone, whole site, windows marked;
(b), (d) the two windows in the MAREA map (20-m grid, the site's cube);
(c), (e) the same windows in the Vaklodingen, cell for cell (nearest 20-m
    survey cell, no interpolation), same colour scale, datum offset removed;
(f) the hydroperiod computed from map (a) and the EOT20 series.

Inputs: products_<site>/comparison_<tag>/products.npz (the notebook's
products), products_<site>/intertidal_mask.tif (rebuild_site_masks.py),
products_<site>/marea_extract.npz (the record, for the low-water scene),
data_v4/truth/vaklodingen/*.nc (v0_vaklodingen_fetch.py).
Writes docs/paper/figures/fig_products.png.
Run from the repository root:  python -m experiments.paper_fig_products_nl [site]
"""
from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import xarray as xr
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

from experiments.paper_fig_pipeline import crop_box, low_water_scene, tide_series  # noqa: E402
from experiments.paper_fig_products import rgba, zoom_windows  # noqa: E402
from experiments.paper_style import INK, clean, grey_background, scale_bar  # noqa: E402

SITES = {
    "wadden": dict(name="Wadden Sea (Vlie)", cube="ndwi_cube_wadden_2023-2025_20m.nc",
                   products="products_wadden/comparison_eot20_g1/products.npz"),
    "ems": dict(name="Ems-Dollard", cube="ndwi_cube_ems_2023-2025_20m.nc",
                products="products_ems/comparison_eot20_g1/products.npz"),
    "escalda": dict(name="Westerschelde", cube="ndwi_cube_escalda_2023-2025_20m.nc",
                    products="products_escalda/comparison_eot20_g2/products.npz"),
}
OUT = "docs/paper/figures/fig_products.png"
WIN_M = 5000                  # zoom window side, metres


def grid_of(cube_path):
    from pyintertidal.cube import grid_from_dataset
    ds = xr.open_dataset(cube_path)
    transform, crs = grid_from_dataset(ds)
    shape = (ds.sizes["y"], ds.sizes["x"])
    ds.close()
    return transform, crs, shape


def vaklodingen_native(cells, transform, crs, r0, r1, c0, c1):
    """The Vaklodingen map itself over the window: its native 20-m grid in
    RD New (EPSG:28992) covering the window's footprint, merged over the
    sheets (latest survey per cell), no resampling. Returns the block, the
    survey year per cell and the block's extent in RD metres."""
    from experiments import v1_vaklodingen_validate as v1
    to_rd = pyproj.Transformer.from_crs(crs, "EPSG:28992", always_xy=True)
    xs = transform.c + np.array([c0, c1, c0, c1]) * transform.a
    ys = transform.f + np.array([r0, r0, r1, r1]) * transform.e
    rx, ry = to_rd.transform(xs, ys)
    x_ref, y_ref = next(iter(cells.values()))[:2]           # the sheets share one lattice
    dx = abs(float(np.median(np.diff(x_ref)))); dy = abs(float(np.median(np.diff(y_ref))))
    gx = x_ref[0] + dx * np.arange(np.floor((rx.min() - x_ref[0]) / dx), np.ceil((rx.max() - x_ref[0]) / dx) + 1)
    gy = y_ref[0] + dy * np.arange(np.floor((ry.min() - y_ref[0]) / dy), np.ceil((ry.max() - y_ref[0]) / dy) + 1)
    gy = gy[::-1]                                             # north up
    X, Y = np.meshgrid(gx, gy)
    v, yr = v1.sample_truth(cells, X.ravel(), Y.ravel())
    extent = (gx[0] - dx / 2, gx[-1] + dx / 2, gy[-1] - dy / 2, gy[0] + dy / 2)
    return v.reshape(X.shape), yr.reshape(X.shape), extent


def main(site="wadden"):
    from pyintertidal.hydroperiod import hydroperiod
    import pyintertidal as pit
    pit.net.use_system_certificates()
    S = SITES[site]
    transform, crs, shape = grid_of(S["cube"])
    px_m = abs(transform.a)
    P = np.load(S["products"])
    keep = P["keep"]
    z = np.full(shape[0] * shape[1], np.nan, np.float32); z[keep] = P["z_marea"]; z = z.reshape(shape)
    mask = rasterio.open(f"products_{site}/intertidal_mask.tif").read(1) > 0
    z = np.where(mask, z, np.nan)                         # the map on the notebook's zone only
    extract = f"products_{site}/marea_extract.npz"
    date, wet_frac, clear_frac = low_water_scene(extract)
    back = grey_background(S["cube"], date, sigma=0.5)
    d = np.load(extract, allow_pickle=True)
    bb = d["bbox"].item() if d["bbox"].dtype == object else \
        {"west": float(d["bbox"][0]), "south": float(d["bbox"][1]), "east": float(d["bbox"][2]), "north": float(d["bbox"][3])}
    _, h_h, _ = tide_series(bb)
    hydro = hydroperiod(z, h_h)
    rec = np.isfinite(z)
    vmin, vmax = np.nanpercentile(z[rec], [2, 98])
    win = int(round(WIN_M / px_m))
    wins = zoom_windows(mask, win=win)
    rs, cs = crop_box(mask, margin=int(600 / px_m))
    crop = (slice(rs.start, rs.stop), slice(cs.start, cs.stop))
    cshape = (rs.stop - rs.start, cs.stop - cs.start)
    # the Vaklodingen on the whole record, for the datum offset (median-centred, as scored)
    rr_all, cc_all = keep // shape[1], keep % shape[1]
    to_rd = pyproj.Transformer.from_crs(crs, "EPSG:28992", always_xy=True)
    from experiments import v1_vaklodingen_validate as v1
    cells = v1.truth_grid(site)
    xq, yq = to_rd.transform(transform.c + (cc_all + 0.5) * transform.a, transform.f + (rr_all + 0.5) * transform.e)
    truth_rec, _ = v1.sample_truth(cells, xq, yq)
    both = np.isfinite(truth_rec) & np.isfinite(z.ravel()[keep])
    offset = float(np.nanmedian(truth_rec[both]) - np.nanmedian(z.ravel()[keep][both]))

    portrait = cshape[0] >= cshape[1]
    fig = plt.figure(figsize=(7.2, 3.6 if portrait else 3.2), dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.45, 1, 1, 1.45], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(back[crop], interpolation="antialiased")
    ax.imshow(rgba(z[crop], "viridis", vmin, vmax), interpolation="antialiased")
    for k, (r0, r1, c0, c1) in enumerate(wins):
        ax.add_patch(Rectangle((c0 - cs.start, r0 - rs.start), c1 - c0, r1 - r0, fill=False, ec=INK, lw=0.9))
        ax.text(c0 - cs.start + 6, r0 - rs.start + 24, "bd"[k], color=INK, fontsize=7, fontweight="bold")
    scale_bar(ax, cshape, px_m, km=5.0, colour="white", halo=INK)
    clean(ax)
    ax.set_title(f"(a) MAREA elevation, {S['name']}\n    (low-water scene of {date} in grey)", loc="left", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=matplotlib.colors.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, shrink=0.8, location="bottom")
    cb.set_label("elevation (m, tide-model datum)", fontsize=6); cb.ax.tick_params(labelsize=6)
    labels = [("(b)", "(c)"), ("(d)", "(e)")]
    years = []
    for k, (r0, r1, c0, c1) in enumerate(wins):
        axm = fig.add_subplot(gs[0, 1 + k])
        axm.imshow(back[r0:r1, c0:c1], interpolation="antialiased")
        axm.imshow(np.ma.masked_invalid(z[r0:r1, c0:c1]), cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        axm.set_title(f"{labels[k][0]} MAREA, window {'bd'[k]}", loc="left", fontsize=7)
        clean(axm)
        if k == 0:
            scale_bar(axm, (r1 - r0, c1 - c0), px_m, km=1.0, colour="white", halo=INK, at=(0.05, 0.93))
        # the Vaklodingen as delivered: its own 20-m grid over the window's
        # footprint, every surveyed cell, same colour scale, datum offset removed
        vak, yr, ext = vaklodingen_native(cells, transform, crs, r0, r1, c0, c1)
        vak = vak - offset
        years.append((int(np.nanmin(np.where(yr > 0, yr, np.nan))), int(np.nanmax(np.where(yr > 0, yr, np.nan)))))
        axl = fig.add_subplot(gs[1, 1 + k])
        axl.set_facecolor("0.82")                              # cells without a survey
        axl.imshow(np.ma.masked_invalid(vak), cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=ext)
        axl.set_xlim(ext[0], ext[1]); axl.set_ylim(ext[2], ext[3]); axl.set_aspect("equal")
        axl.set_title(f"{labels[k][1]} Vaklodingen, window {'bd'[k]}", loc="left", fontsize=7)
        clean(axl)
    ax = fig.add_subplot(gs[:, 3])
    ax.imshow(back[crop], interpolation="antialiased")
    ax.imshow(rgba(hydro[crop], "Blues", 0, 1), interpolation="antialiased")
    scale_bar(ax, cshape, px_m, km=5.0, colour="white", halo=INK)
    clean(ax)
    ax.set_title("(f) hydroperiod from (a)", loc="left", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, shrink=0.8, location="bottom")
    cb.set_label("fraction of time submerged", fontsize=6); cb.ax.tick_params(labelsize=6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=400)
    print(f"-> {OUT}  ({site}: scene {date}, {wet_frac:.0%} wet, {clear_frac:.0%} clear; windows {wins}; "
          f"Vaklodingen offset {offset:+.2f} m, survey years {years}; z range {vmin:.2f}..{vmax:.2f}; "
          f"{int(rec.sum()):,} MAREA px on the notebook's zone)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wadden")
