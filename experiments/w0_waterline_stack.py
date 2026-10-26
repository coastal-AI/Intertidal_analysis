"""W0 — sub-pixel waterline stacking (DEA/NIDEM-style) as REAL spatial
super-resolution, judged against the raw RTK points.

In every clear scene the waterline is not a pixel: the NDWI crosses the
water threshold BETWEEN pixel centres, and a contour of the index at that
threshold locates it with sub-pixel precision (skimage.find_contours,
linear interpolation between neighbours). Each contour carries a known
elevation — the tide at that overpass — so stacking hundreds of contours
from hundreds of tide levels builds a DEM whose horizontal resolution is
the contours', not the pixel's. This is the spatial information that the
statistical stage 2 of HSR lacked (measured 2026-09-14: the 2.5 m z_fine
was 7 cm WORSE than the 10 m mu against raw RTK).

Gate: the same test. On the development RTK blocks (reserved sealed), the
waterline DEM must beat HSR-10 m (0.219 m RMSE on 197 points) to count.

Run:  python -m experiments.w0_waterline_stack
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"
THRESHOLD = 0.0
SCALE = 4                       # 2.5 m output grid
MAX_BAD = 0.10                  # scene-level cloud rule (AOI-wide)
MIN_VERTICES = 20
OUT_DIR = "products_villaviciosa"


def main():
    import rasterio
    from rasterio.transform import Affine
    from scipy import ndimage
    from scipy.interpolate import griddata
    from skimage.measure import find_contours

    import pyintertidal as pit
    from pyintertidal import water as W
    from pyintertidal.cube import SentinelCube, open_cube
    pit.net.use_system_certificates()

    t0 = time.time()
    aoi = pit.sites.get("villaviciosa")
    cube = SentinelCube(aoi, ("2016-01-01", "2025-12-31"), water="ndwi",
                        cache_path=CUBE)
    transform, crs = cube.grid
    H, Wd = cube.shape
    dates = list(cube.dates)

    with rasterio.open(f"{OUT_DIR}/intertidal_mask.tif") as s:
        inter = s.read(1) > 0
    # tides at every overpass (mid-channel point, as the study does)
    with rasterio.open(f"{OUT_DIR}/reference_map.tif") as s:
        try:
            point = pit.water_centroid(s.read(1), s.transform, s.crs)
        except ValueError:
            lon, lat = aoi.centroid
            point = (lat, lon)
    tides = pit.TideService(model="EOT20", directory="./tide_models",
                            location=point)
    h_of = tides.heights_for(aoi, dates)

    # ── pass over the cube: one sub-pixel waterline per clear scene ────────
    pts = []                      # (row, col, z) in 10 m pixel units
    used, example = [], None
    ds, arrays, t_dim = open_cube(cube.cache_path, cube.bands)
    try:
        T = arrays["SCL"].sizes[t_dim]
        CH = 40
        for i0 in range(0, T, CH):
            sl = slice(i0, min(i0 + CH, T))
            g = np.asarray(arrays["B03"].isel({t_dim: sl}).values, np.float32)
            n = np.asarray(arrays["B08"].isel({t_dim: sl}).values, np.float32)
            scl = np.nan_to_num(np.asarray(arrays["SCL"].isel({t_dim: sl}).values,
                                           np.float32), nan=0).astype(np.int16)
            for k in range(g.shape[0]):
                i = i0 + k
                h = h_of.get(dates[i], np.nan)
                if not np.isfinite(h):
                    continue
                bad = np.isin(scl[k][inter], list(W.BAD_CLASSES)).mean()
                if bad > MAX_BAD:
                    continue
                clear = np.isin(scl[k], list(W.CLEAR_CLASSES))
                with np.errstate(invalid="ignore", divide="ignore"):
                    ndwi = (g[k] - n[k]) / (g[k] + n[k])
                ndwi = np.where(clear & np.isfinite(ndwi), ndwi, np.nan)
                finite = np.isfinite(ndwi)
                # contours must not be born at cloud edges: fill holes far
                # BELOW the level, then drop vertices touching a hole
                filled = np.where(finite, ndwi, -1e6)
                touch = ndimage.binary_dilation(~finite, iterations=1)
                n_pts = 0
                for path in find_contours(filled, THRESHOLD):
                    if len(path) < MIN_VERTICES:
                        continue
                    r, c = path[:, 0], path[:, 1]
                    ri = np.clip(np.round(r).astype(int), 0, H - 1)
                    ci = np.clip(np.round(c).astype(int), 0, Wd - 1)
                    ok = inter[ri, ci] & ~touch[ri, ci]
                    if ok.sum() < MIN_VERTICES:
                        continue
                    pts.append(np.column_stack([r[ok], c[ok],
                                                np.full(ok.sum(), h)]))
                    n_pts += int(ok.sum())
                if n_pts:
                    used.append((dates[i], float(h), n_pts))
                    if example is None and 0.35 < bad < 0.36 or (
                            example is None and abs(h + 0.3) < 0.15
                            and bad < 0.02):
                        example = (dates[i], ndwi.copy(), float(h))
            print(f"  scenes {min(i0 + CH, T)}/{T} · waterlines so far "
                  f"{len(used)}", flush=True)
    finally:
        ds.close()
    P = np.concatenate(pts)
    print(f"{len(used)} clear scenes -> {len(P):,} sub-pixel waterline "
          f"vertices, tide range [{P[:, 2].min():+.2f}, {P[:, 2].max():+.2f}] m "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ── stack -> DEM on the 2.5 m grid ───────────────────────────────────
    Hf, Wf = H * SCALE, Wd * SCALE
    rf = np.clip((P[:, 0] + 0.5) * SCALE, 0, Hf - 1e-3).astype(int)
    cf = np.clip((P[:, 1] + 0.5) * SCALE, 0, Wf - 1e-3).astype(int)
    # per fine cell: median of the waterline elevations that pass through
    lin = rf * Wf + cf
    order = np.argsort(lin)
    lin_s, z_s = lin[order], P[order, 2]
    uniq, start = np.unique(lin_s, return_index=True)
    med = np.array([np.median(z_s[a:b]) for a, b in
                    zip(start, list(start[1:]) + [len(z_s)])])
    cnt = np.diff(list(start) + [len(z_s)])
    cell_r, cell_c = uniq // Wf, uniq % Wf
    print(f"{len(uniq):,} fine cells touched by a waterline "
          f"(median {np.median(cnt):.0f} crossings each)", flush=True)

    inter_f = np.repeat(np.repeat(inter, SCALE, 0), SCALE, 1)
    tr, tc = np.where(inter_f)
    dem = np.full((Hf, Wf), np.nan, np.float32)
    # linear interpolation between waterlines (NIDEM's move), restricted to
    # the intertidal, in chunks to bound memory
    zi = griddata(np.column_stack([cell_r, cell_c]), med,
                  np.column_stack([tr, tc]), method="linear")
    dem[tr, tc] = zi
    # cells that carry a waterline keep their own median (no interpolation)
    dem[cell_r, cell_c] = med
    fine_tr = Affine(transform.a / SCALE, transform.b, transform.c,
                     transform.d, transform.e / SCALE, transform.f)
    out = f"{OUT_DIR}/waterline_dem_2p5m.tif"
    with rasterio.open(out, "w", driver="GTiff", height=Hf, width=Wf,
                       count=1, dtype="float32", transform=fine_tr, crs=crs,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(dem, 1)
    np.savez_compressed(f"{OUT_DIR}/waterline_stack.npz",
                        points=P.astype(np.float32),
                        used=np.array(used, dtype=object),
                        example_date=example[0] if example else "",
                        example_ndwi=example[1] if example else np.zeros(1),
                        example_h=example[2] if example else np.nan)
    print(f"-> {out} ({int(np.isfinite(dem).sum()):,} cells) "
          f"· {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
