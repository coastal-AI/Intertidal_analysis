"""W1 — judge the waterline-stack DEM against the raw RTK points, and draw
what it is made of.

Same protocol as the stage-2 test (2026-09-14): development blocks only
(the reserved 35 % of 150-m blocks, seed 20260817, stays sealed),
median-centred axes (ellipsoidal RTK vs tide-datum DEMs), RMSE and slope,
and a common-subset table so nobody wins on easier pixels.

Figures (in _render_qc/):
  waterline_example.png   one clear scene's NDWI with its sub-pixel contour
  waterline_stack.png     every stacked waterline coloured by its tide level
  waterline_dems.png      HSR 10 m vs waterline 2.5 m, same window, shaded
  waterline_validation.png residuals of the three DEMs against the points

Run:  python -m experiments.w1_waterline_validate
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "_render_qc"
TIDE = "#2a6f97"; MUD = "#b08968"; RED = "#c1121f"; INK = "#14313c"


def main(dem_path="products_villaviciosa/waterline_dem_2p5m.tif",
         npz_path="products_villaviciosa/waterline_stack.npz",
         extra=None):
    import pandas as pd
    import pyproj
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from skimage.measure import find_contours

    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("data/villaviciosa_rtk_gnss.csv")
    df = df[df["Solution status"] == "FIX"]
    lon = df["Longitude"].to_numpy(float)
    lat = df["Latitude"].to_numpy(float)
    hz = df["Ellipsoidal height"].to_numpy(float)

    def sample(path):
        with rasterio.open(path) as s:
            fwd = pyproj.Transformer.from_crs("EPSG:4326", s.crs,
                                              always_xy=True)
            x, y = fwd.transform(lon, lat)
            v = np.array([q[0] for q in s.sample(zip(x, y))], float)
            if s.nodata is not None:
                v[v == s.nodata] = np.nan
            rows, cols = rasterio.transform.rowcol(s.transform, x, y)
            return v, np.asarray(rows), np.asarray(cols), s.transform

    z10, r10, c10, tr10 = sample("products_villaviciosa/hsr_eot20_2023-2025.tif")
    z25, _, _, _ = sample("products_villaviciosa/hsr_dem_fine.tif")
    zwl, _, _, _ = sample(dem_path)
    zx = sample(extra)[0] if extra else None

    # the sealed split, from the campo pixels (c1's formula)
    campo = np.load("products_villaviciosa/campo.npz", allow_pickle=True)
    rr, cc = campo["row"], campo["col"]
    east = tr10.c + (cc + 0.5) * tr10.a
    north = tr10.f + (rr + 0.5) * tr10.e
    block = (((east - east.min()) // 150.0).astype(int) * 1000
             + ((north - north.min()) // 150.0).astype(int))
    blocks = np.unique(block)
    rng = np.random.default_rng(20260817)
    hold = rng.choice(blocks, max(2, int(round(0.35 * len(blocks)))),
                      replace=False)
    hold_px = set(zip(rr[np.isin(block, hold)], cc[np.isin(block, hold)]))
    dev = np.array([(r, c) not in hold_px for r, c in zip(r10, c10)])
    print(f"development points: {int(dev.sum())} / {len(dev)} "
          f"(reserved untouched)")

    preds = {"HSR 10 m (mu)": z10, "HSR 2.5 m (stage 2)": z25,
             "waterline stack 2.5 m": zwl}
    if zx is not None:
        preds["waterline stack 10 m"] = zx
    rows = []
    for name, p in preds.items():
        m = dev & np.isfinite(p)
        pc = p[m] - np.median(p[m]); t = hz[m] - np.median(hz[m])
        rows.append((name, np.sqrt(np.mean((pc - t) ** 2)),
                     np.polyfit(t, pc, 1)[0], int(m.sum())))
    print(f"\n{'DEV BLOCKS':26s} {'RMSE':>7s} {'slope':>7s} {'n':>5s}")
    for name, r, s, n in rows:
        print(f"{name:26s} {r:7.3f} {s:7.3f} {n:5d}")
    common = dev
    for p in preds.values():
        common &= np.isfinite(p)
    print(f"\nCOMMON SUBSET ({int(common.sum())} points)")
    t = hz[common] - np.median(hz[common])
    res = {}
    for name, p in preds.items():
        pc = p[common] - np.median(p[common])
        res[name] = pc - t
        print(f"{name:26s} RMSE {np.sqrt(np.mean((pc - t) ** 2)):.3f}  "
              f"slope {np.polyfit(t, pc, 1)[0]:.3f}")

    # ── figures ───────────────────────────────────────────────────────────
    st = np.load(npz_path, allow_pickle=True)
    P = st["points"]
    r0, r1 = campo["row"].min() - 40, campo["row"].max() + 40
    c0, c1 = campo["col"].min() - 40, campo["col"].max() + 40

    # example scene and its contour
    ex = st["example_ndwi"]
    if ex.ndim == 2:
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        win = ex[r0:r1, c0:c1]
        ax.imshow(win, cmap="RdYlBu", vmin=-0.6, vmax=0.8)
        for path in find_contours(np.where(np.isfinite(win), win, -1e6), 0.0):
            if len(path) >= 20:
                ax.plot(path[:, 1], path[:, 0], "-", color="k", lw=1.2)
        ax.set_title(f"one clear scene ({st['example_date']}, tide "
                     f"{float(st['example_h']):+.2f} m): NDWI and its "
                     "sub-pixel waterline", fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{OUT}/waterline_example.png", bbox_inches="tight")
        plt.close(fig)

    # the stack, coloured by tide level
    m = (P[:, 0] >= r0) & (P[:, 0] < r1) & (P[:, 1] >= c0) & (P[:, 1] < c1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    sc = ax.scatter(P[m, 1] - c0, P[m, 0] - r0, c=P[m, 2], s=0.4,
                    cmap="terrain", vmin=-2, vmax=2, linewidths=0)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="tide level of the waterline (m)")
    ax.set_ylim(r1 - r0, 0); ax.set_xlim(0, c1 - c0)
    ax.set_aspect("equal")
    ax.set_title(f"{len(st['used'])} waterlines stacked "
                 f"({int(m.sum()):,} vertices in this window)",
                 fontsize=10.5, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/waterline_stack.png", bbox_inches="tight")
    plt.close(fig)

    # the two DEMs, same window, shaded
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        d10 = np.ma.filled(s.read(1, masked=True).astype(float), np.nan)
    with rasterio.open(dem_path) as s:
        dwl = np.ma.filled(s.read(1, masked=True).astype(float), np.nan)
    ls = LightSource(azdeg=315, altdeg=45)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.2), dpi=150)
    for ax, d, k, ttl in ((axs[0], d10[r0:r1, c0:c1], 1, "HSR, 10 m"),
                          (axs[1], dwl[r0 * 4:r1 * 4, c0 * 4:c1 * 4], 4,
                           "waterline stack, 2.5 m")):
        filled = np.where(np.isfinite(d), d, np.nanmedian(d))
        rgb = ls.shade(filled, cmap=plt.get_cmap("terrain"), vmin=-2, vmax=2,
                       vert_exag=8 / k, blend_mode="soft")
        rgb[~np.isfinite(d)] = 1.0
        ax.imshow(rgb, interpolation="antialiased")
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same window, same colour scale (-2 to +2 m), shaded",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/waterline_dems.png", bbox_inches="tight")
    plt.close(fig)

    # validation residuals
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    bins = np.linspace(-0.8, 0.8, 41)
    for (name, r), col in zip(res.items(), (TIDE, MUD, RED, "#1b7837")):
        ax.hist(r, bins=bins, histtype="step", lw=2, color=col,
                label=f"{name}: RMSE {np.sqrt(np.mean(r ** 2)):.3f} m")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("DEM minus RTK (m), median-centred, development blocks")
    ax.set_ylabel("points"); ax.legend(fontsize=9); ax.grid(alpha=0.2)
    ax.set_title(f"the judge: {int(common.sum())} raw RTK points, three DEMs",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/waterline_validation.png", bbox_inches="tight")
    print(f"\n-> figures in {OUT}/")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a + [None] * (3 - len(a)))[:3]) if a else main()
