# -*- coding: utf-8 -*-
"""On-site validation of the elevation: the RTK-GNSS survey of Villaviciosa
— the paper figure, assembled from what the topography notebook wrote.

The computation lives in sections 9a/9b of
``intertidal_topography_villaviciosa.ipynb``: the two maps on the
2023-2025 record (MAREA, our inversion with the lag every pixel received;
Granadeiro, their final map), scored on the development split of the
survey (``pyintertidal.rtk.load_rtk``) and written to
``products_marea/rtk_onsite_metrics.csv``; the maps themselves are saved
in ``products_marea/external_products.npz``. This script reads those
products, scores the same points the same way (a check that must
reproduce the CSV) and draws:

(a), (b) each map on the RTK elevation (OLS line, 1:1 line);
(c) the walked route on the low-water scene, the straight stretch in white;
(d) the profile along that stretch: RTK points joined in walking order and
    the two maps at the pixels the route crosses.

Writes results/v5_rtk_onsite.json and docs/paper/figures/fig_rtk_onsite.png.
Run from the repository root:  python -m experiments.v5_rtk_onsite
"""
from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

INK = "#14313c"; TIDE = "#2a6f97"; RED = "#c1121f"; MAG = "#ff5da2"; YEL = "#ffd166"
PRODUCTS = "products_marea/external_products.npz"
NOTEBOOK_CSV = "products_marea/rtk_onsite_metrics.csv"
MIN_CHORD_M = 120.0        # a stretch must be at least this long ...
MIN_STRAIGHT = 0.92        # ... and its chord at least this share of the walked path


def load_maps():
    e = np.load(PRODUCTS)
    return {"MAREA": e["MAREA"].astype(float), "Granadeiro": e["Granadeiro"].astype(float)}


def load_survey():
    """Development points of the survey on the product grid (the reserved
    blocks stay sealed), and the grid itself."""
    import rasterio
    from pyintertidal import rtk
    dev = rtk.load_rtk(rtk.DEFAULT_CSV, rtk.DEFAULT_GRID)
    with rasterio.open(rtk.DEFAULT_GRID) as s:
        grid = dict(transform=s.transform, crs=s.crs, shape=s.shape)
    print(f"{len(dev['elev'])} development RTK points ({dev.get('n_reserved_hidden', '?')} reserved, sealed)")
    return dev, grid


def ols(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[m] - np.median(x[m]), y[m] - np.median(y[m])
    b, a = np.polyfit(xc, yc, 1)
    r = float(np.corrcoef(xc, yc)[0, 1])
    rmse = float(np.sqrt(np.mean((yc - xc) ** 2)))
    bias = float(np.median(y[m] - x[m]))
    return {"n": int(m.sum()), "slope": float(b), "intercept": float(a), "r": r,
            "rmse_m": rmse, "bias_m": bias}, xc, yc


def straight_stretch(grid, maps):
    """The straightest long stretch of the walked route, development
    blocks only: RTK points in walking order with their distance along
    the chord, and each map's value at the pixel of every point."""
    import pandas as pd
    import pyproj
    from pyintertidal import rtk
    df = pd.read_csv(rtk.DEFAULT_CSV)
    df = df[df["Solution status"] == "FIX"].reset_index(drop=True)
    lon, lat, hgt = (df[c].to_numpy(float) for c in ("Longitude", "Latitude", "Ellipsoidal height"))
    fwd = pyproj.Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    x, y = fwd.transform(lon, lat)
    col, row = ~grid["transform"] * (x, y)
    row, col = np.floor(row).astype(int), np.floor(col).astype(int)
    _, is_res = rtk._assign_blocks(x, y)            # the notebook's own block split
    idx_dev = np.flatnonzero(~is_res)
    best = None
    n = len(idx_dev)
    for i in range(n):
        for j in range(i + 20, n):
            a, b = idx_dev[i], idx_dev[j]
            if b - a != j - i:                         # a reserved point in between: not consecutive
                break
            seg = slice(a, b + 1)
            chord = float(np.hypot(x[b] - x[a], y[b] - y[a]))
            path = float(np.sum(np.hypot(np.diff(x[seg]), np.diff(y[seg]))))
            if chord >= MIN_CHORD_M and path > 0 and chord / path >= MIN_STRAIGHT:
                if best is None or chord > best[0]:
                    best = (chord, a, b, chord / path)
    if best is None:
        raise SystemExit("no straight stretch of the development route found; relax MIN_CHORD_M / MIN_STRAIGHT")
    chord, a, b, straight = best
    covered = np.array([any(np.isfinite(v[r_, c_]) for v in maps.values())
                        if 0 <= r_ < grid["shape"][0] and 0 <= c_ < grid["shape"][1] else False
                        for r_, c_ in zip(row, col)])
    inside = np.flatnonzero(covered[a:b + 1])
    a, b = a + int(inside.min()), a + int(inside.max())
    chord = float(np.hypot(x[b] - x[a], y[b] - y[a]))
    seg = slice(a, b + 1)
    ux, uy = (x[b] - x[a]) / chord, (y[b] - y[a]) / chord
    dist = (x[seg] - x[a]) * ux + (y[seg] - y[a]) * uy
    vals = {name: v[row[seg], col[seg]] for name, v in maps.items()}
    print(f"straight stretch: points {a}–{b} of the route, chord {chord:.0f} m, "
          f"chord/path {straight:.2f}, {b - a + 1} RTK points")
    return dict(dist=dist, rtk=hgt[seg], maps=vals, chord_m=chord, straightness=straight,
                first=int(a), last=int(b), rows=row[seg], cols=col[seg], rows_all=row[~is_res], cols_all=col[~is_res])


def main():
    import pandas as pd
    maps = load_maps()
    dev, grid = load_survey()
    y = np.asarray(dev["elev"], float)
    rr, cc = np.asarray(dev["row"], int), np.asarray(dev["col"], int)
    results, fits = {}, {}
    for name, v in maps.items():
        results[name], *fits[name] = ols(y, v[rr, cc])
        r = results[name]
        print(f"{name:12s} n={r['n']:4d}  RMSE {r['rmse_m']:.3f} m  slope {r['slope']:.3f}  r {r['r']:.3f}  bias {r['bias_m']:+.2f}")
    if os.path.exists(NOTEBOOK_CSV):                    # must agree with the notebook's own scores
        nb = pd.read_csv(NOTEBOOK_CSV).set_index("product")
        for name in maps:
            if name in nb.index:
                d = abs(float(nb.loc[name, "rmse_m"]) - results[name]["rmse_m"])
                print(f"   notebook {name}: RMSE {float(nb.loc[name, 'rmse_m']):.3f} (difference {d:.4f} m)")
    stretch = straight_stretch(grid, maps)

    fig = plt.figure(figsize=(7.2, 3.4), dpi=200)      # page budget
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.72], width_ratios=[1.0, 1.0, 0.82, 1.0])
    colours = {"MAREA": RED, "Granadeiro": TIDE}
    for j, name in enumerate(maps):
        ax = fig.add_subplot(gs[0, j])
        xc, yc = fits[name]
        r = results[name]
        ax.plot(xc, yc, ".", ms=3, color=colours[name], alpha=0.7)
        lim = [min(xc.min(), yc.min()) - 0.1, max(xc.max(), yc.max()) + 0.1]
        ax.plot(lim, lim, "-", color="0.75", lw=0.8)
        xx = np.array(lim); ax.plot(xx, r["intercept"] + r["slope"] * xx, "-", color=INK, lw=1.2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_title(f"({'ab'[j]}) {name}", loc="left", fontsize=9)
        ax.set_xlabel("RTK elevation (m, centred)", fontsize=8)
        if j == 0:
            ax.set_ylabel("map elevation (m, centred)", fontsize=8)
        ax.text(0.03, 0.97, f"n = {r['n']}\nRMSE {r['rmse_m']:.3f} m\nslope {r['slope']:.2f}\nr {r['r']:.2f}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, :])
    d, rtk_h = stretch["dist"], stretch["rtk"]
    ax.plot(d, rtk_h - np.median(rtk_h), "o-", ms=3, lw=1.0, color=MAG, label="RTK-GNSS, walking order", zorder=5)
    for name, v in stretch["maps"].items():
        ok = np.isfinite(v)
        if ok.any():
            ax.plot(d[ok], v[ok] - np.nanmedian(v), "s-", ms=2.5, lw=1.2, color=colours[name], label=name)
    ax.set_xlabel("distance along the stretch (m)", fontsize=8)
    ax.set_ylabel("elevation (m, centred)", fontsize=8)
    ax.set_title(f"(e) the stretch of (d): a straight {stretch['chord_m']:.0f} m of the surveyed route", loc="left", fontsize=9)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))                      # headroom for the legend
    ax.legend(fontsize=6.5, ncol=3, loc="upper center", framealpha=0.9); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
    # (c) the ría with the survey, (d) a zoom on the survey with the stretch
    try:
        import matplotlib.patheffects as pe
        import rasterio
        from matplotlib.patches import Rectangle
        from experiments.paper_fig_pipeline import CUBE, EXTRACT, MASK_TIF, composite, crop_box, low_water_scene
        from experiments.paper_style import grey_background, outline
        halo = [pe.withStroke(linewidth=1.5, foreground="white")]
        date, _, _ = low_water_scene(EXTRACT)
        canvas = grey_background(CUBE, date)                   # grey: the survey and the zone carry the colour
        mask = rasterio.open(MASK_TIF).read(1) > 0
        rs, cs = crop_box(mask, margin=30)                     # the whole ría, as in the other figures
        rr_, cc_ = stretch["rows"], stretch["cols"]
        ra, ca = stretch["rows_all"], stretch["cols_all"]
        px = abs(grid["transform"].a)                          # metres per pixel
        # the zoom window: square, around the whole survey
        pad = 22
        side = int(max(ra.max() - ra.min(), ca.max() - ca.min()) + 2 * pad)
        zr0 = int(max(min((ra.min() + ra.max()) / 2 - side / 2, canvas.shape[0] - side), 0))
        zc0 = int(max(min((ca.min() + ca.max()) / 2 - side / 2, canvas.shape[1] - side), 0))
        zr1, zc1 = zr0 + side, zc0 + side
        # (c) the ría
        axl = fig.add_subplot(gs[0, 2])
        axl.imshow(canvas[rs, cs], interpolation="antialiased")
        outline(axl, mask[rs, cs])                             # thin dark line: the intertidal zone
        axl.plot(ca - cs.start, ra - rs.start, ".", ms=1.5, color=MAG, alpha=0.9)
        axl.add_patch(Rectangle((zc0 - cs.start, zr0 - rs.start), side, side, fill=False, ec=INK, lw=1.0))
        axl.text(zc1 - cs.start + 3, zr0 - rs.start + side / 2, "(d)", color=INK, fontsize=7, va="center",
                 fontweight="bold", path_effects=halo)
        km = 1000 / px
        H_ = rs.stop - rs.start
        axl.plot([8, 8 + km], [H_ - 12, H_ - 12], "-", color=INK, lw=2)
        axl.text(8 + km / 2, H_ - 16, "1 km", color=INK, ha="center", fontsize=6, path_effects=halo)
        axl.set_xticks([]); axl.set_yticks([])
        axl.set_title("(c) the ría", loc="left", fontsize=9)
        # (d) the survey and the stretch
        axz = fig.add_subplot(gs[0, 3])
        axz.imshow(canvas[zr0:zr1, zc0:zc1], interpolation="nearest")
        outline(axz, mask[zr0:zr1, zc0:zc1], lw=0.6)
        axz.plot(ca - zc0, ra - zr0, ".", ms=3, color=MAG, alpha=0.9)
        axz.plot(cc_ - zc0, rr_ - zr0, "-", color=INK, lw=2.0,
                 path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
        axz.plot(cc_[0] - zc0, rr_[0] - zr0, "o", ms=4.5, color=INK, mec="white")
        axz.text(cc_[0] - zc0 + 3, rr_[0] - zr0, "0 m", color=INK, fontsize=6, va="center", path_effects=halo)
        axz.text(cc_[-1] - zc0 + 3, rr_[-1] - zr0, f"{stretch['chord_m']:.0f} m", color=INK, fontsize=6,
                 va="center", path_effects=halo)
        m200 = 200 / px
        axz.plot([6, 6 + m200], [side - 8, side - 8], "-", color=INK, lw=2)
        axz.text(6 + m200 / 2, side - 11, "200 m", color=INK, ha="center", fontsize=6, path_effects=halo)
        axz.set_xticks([]); axz.set_yticks([])
        for sp in axz.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(1.0)
        axz.set_title("(d) the survey (pink)", loc="left", fontsize=9)
    except Exception as exc:                                   # the map is a courtesy, not a result
        print(f"[location panels skipped: {exc}]")
    fig.tight_layout()
    os.makedirs("docs/paper/figures", exist_ok=True)
    fig.savefig("docs/paper/figures/fig_rtk_onsite.png", dpi=200)
    os.makedirs("results", exist_ok=True)
    json.dump({"n_dev": int(len(y)), "rows": results,
               "stretch": {k: v for k, v in stretch.items() if k in ("chord_m", "straightness", "first", "last")}},
              open("results/v5_rtk_onsite.json", "w"), indent=1)
    print("-> results/v5_rtk_onsite.json, docs/paper/figures/fig_rtk_onsite.png")


if __name__ == "__main__":
    main()
