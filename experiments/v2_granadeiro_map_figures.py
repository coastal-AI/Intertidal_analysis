"""V2 — redraw the map figures of the comparison from the saved products.

Two figures per site, the same the notebook draws, with two display fixes:

* Granadeiro's elevation was fitted on 1 pixel in k (px_stride) on the large
  flats: a lattice with (k*k - 1) empty pixels per block reads as the
  hillshade background. Each fitted pixel is painted over its k x k block,
  for display only and inside the intertidal record, and drawn without
  hillshade; the titles say so.
* On a site where MAREA applied no clock (every band at tau = 0) the
  "applied MAREA - plain" panel is identically zero. It is drawn flat and
  labelled as such, so a blank panel reads as a result, not as a failure.

Run:  python -m experiments.v2_granadeiro_map_figures [site ...]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SITES = {"escalda": ("products_escalda/comparison_eot20_g2/products.npz", "ndwi_cube_escalda_2023-2025_20m.nc"),
         "wadden": ("products_wadden/comparison_eot20_g1/products.npz", "ndwi_cube_wadden_2023-2025_20m.nc"),
         "ems": ("products_ems/comparison_eot20_g1/products.npz", "ndwi_cube_ems_2023-2025_20m.nc"),
         "ferrol": ("products_ferrol/comparison_eot20_g1/products.npz", "ndwi_cube_ferrol_2023-2025_10m.nc")}
OUT = "docs/figures/tide_comparison"


def main(sites):
    import pyproj
    import xarray as xr
    from affine import Affine
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pyintertidal as pit
    from pyintertidal import viz

    for site in sites:
        prod, cube = SITES[site]
        if not os.path.exists(prod):
            print(f"[{site}] no products"); continue
        P = np.load(prod)
        keep, H, W = P["keep"], *[int(v) for v in P["shape"]]
        ds = xr.open_dataset(cube)
        crs = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref"))
        xs, ys = ds["x"].values, ds["y"].values
        ds.close()
        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        transform = Affine(dx, 0.0, float(xs[0]) - dx / 2, 0.0, dy, float(ys[0]) - dy / 2)
        aoi = pit.sites.get(site)
        rr, cc = keep // W, keep % W
        rec = np.zeros((H, W), bool); rec[rr, cc] = True

        def to_raster(vals):
            r = np.full(H * W, np.nan, np.float32); r[keep] = vals; return r.reshape(H, W)

        z_plain, z_marea, z_fit = (P["z_plain"].astype(float), P["z_marea"].astype(float),
                                   P["z_marea_fit"].astype(float))

        # ── figure 1: without vs with MAREA ─────────────────────────────
        dem_plain, dem_marea, dem_fit = to_raster(z_plain), to_raster(z_marea), to_raster(z_fit)
        d_app = dem_marea - dem_plain
        flat = np.nanmax(np.abs(d_app)) < 1e-6 if np.isfinite(d_app).any() else True
        fig, axes = plt.subplots(2, 2, figsize=(19, 13))
        viz.plot_dem(dem_plain, transform, crs, aoi,
                     title="Without adjustment: boundary read at the overpass time", ax=axes[0, 0])
        viz.plot_dem(dem_marea, transform, crs, aoi,
                     title="With MAREA clocks (the applied product)", ax=axes[0, 1])
        if flat:
            viz.plot_map(d_app, transform, crs, aoi, cmap="RdBu_r", vmin=-0.3, vmax=0.3, robust=False,
                         hillshade=False, cbar_label="Δz (m)", ax=axes[1, 0],
                         title="applied MAREA − plain: identically 0\n(no band cleared the 10-min threshold, or the profile was vetoed)")
            axes[1, 0].text(0.5, 0.5, "same map\nΔz = 0 everywhere", transform=axes[1, 0].transAxes,
                            ha="center", va="center", fontsize=16, color="#555")
        else:
            viz.plot_map(d_app, transform, crs, aoi, cmap="RdBu_r", vmin=-0.3, vmax=0.3, robust=False,
                         hillshade=False, title="applied MAREA − plain (m)", cbar_label="Δz (m)", ax=axes[1, 0])
        viz.plot_map(dem_fit - dem_plain, transform, crs, aoi, cmap="RdBu_r", vmin=-0.3, vmax=0.3, robust=False,
                     hillshade=False, title="fitted clocks without the threshold − plain (m)\n(sensitivity, not a product)",
                     cbar_label="Δz (m)", ax=axes[1, 1])
        fig.tight_layout()
        fig.savefig(f"{OUT}/{site}_dem_without_vs_with.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ── figure 2: Granadeiro's maps ─────────────────────────────────
        z_gran, z_cross, lag = P["z_gran"].astype(float), P["z_cross"].astype(float), P["lag_all"].astype(float)
        fitted = np.isfinite(z_gran)
        stride = int(np.clip(round(1.0 / max(fitted.mean(), 1e-6)), 1, 4)) if fitted.mean() < 0.6 else 1

        def expand(vals):
            r = to_raster(vals)
            if stride == 1:
                return r
            out = np.full_like(r, np.nan)
            ok = np.isfinite(vals)
            for dr in range(stride):
                for dc in range(stride):
                    r2, c2 = np.clip(rr[ok] + dr, 0, H - 1), np.clip(cc[ok] + dc, 0, W - 1)
                    paint = rec[r2, c2] & ~np.isfinite(out[r2, c2])
                    out[r2[paint], c2[paint]] = vals[ok][paint]
            return out

        note = f"\n(fitted on 1 pixel in {stride}, expanded for display)" if stride > 1 else ""
        vmin, vmax = float(np.nanmin(dem_marea)), float(np.nanmax(dem_marea))
        fig, axes = plt.subplots(2, 2, figsize=(19, 13))
        viz.plot_map(to_raster(lag), transform, crs, aoi, cmap="magma", robust=True, hillshade=False,
                     title="Granadeiro lag map (min)\ntheir thin-plate spline over the ~1 000-pixel lag sample",
                     cbar_label="lag (min)", ax=axes[0, 0])
        viz.plot_map(expand(z_gran), transform, crs, aoi, cmap="viridis", robust=False, hillshade=False,
                     vmin=vmin, vmax=vmax, title="Granadeiro elevation: logistic inflection on NIR" + note,
                     cbar_label="Elevation (m above tide-model MSL)", ax=axes[0, 1])
        viz.plot_dem(to_raster(z_cross), transform, crs, aoi,
                     title="Crossed: Granadeiro's lag, our inversion", ax=axes[1, 0])
        viz.plot_map(expand(z_gran - z_marea), transform, crs, aoi, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                     robust=False, hillshade=False, title="Granadeiro − MAREA (m)" + note,
                     cbar_label="Δz (m)", ax=axes[1, 1])
        fig.tight_layout()
        fig.savefig(f"{OUT}/{site}_granadeiro_maps.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[{site}] stride {stride}, applied clock flat: {flat} -> {OUT}/{site}_*.png", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SITES))
