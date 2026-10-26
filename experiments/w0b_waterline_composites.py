"""W0b — waterlines from TIDAL COMPOSITES (the NIDEM recipe), v1.

v0 contoured every scene at a fixed threshold and failed (RMSE 0.61 m,
slope 0.11 against raw RTK): wet mud and turbid water hover around NDWI 0,
so a single scene sprouts contours all over the flat, cloud holes grow
false shorelines, and waterlines of incompatible tides cross the same
cell (median spread 0.59 m). Bishop-Taylor et al. (NIDEM) avoid exactly
this by contouring MEDIAN COMPOSITES per tidal interval: averaging every
clear scene of one tide level cancels the speckle and leaves one clean,
sub-pixel shoreline per level.

Recipe here: the intertidal pixel series (marea_demo_extract.npz) are
binned into quantile tide intervals; each bin's per-pixel median NDWI is
placed back on the full frame, with permanent water (+1) and land (-1)
filled from the water-frequency product so contours can only be born on
the flat; find_contours at the threshold gives the shoreline of that
level; the stack is gridded at 2.5 m (and 10 m) inside the intertidal.

Run:  python -m experiments.w0b_waterline_composites [n_bins]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

THRESHOLD = 0.0
SCALE = 4
MIN_VERTICES = 30
MIN_OBS_BIN = 5
OUT_DIR = "products_villaviciosa"


def grid_stack(P, H, W, scale, inter, transform, crs, out):
    """Median per cell of the waterline levels, linear fill in between."""
    import rasterio
    from rasterio.transform import Affine
    from scipy.interpolate import griddata

    Hf, Wf = H * scale, W * scale
    rf = np.clip((P[:, 0] + 0.5) * scale, 0, Hf - 1e-3).astype(int)
    cf = np.clip((P[:, 1] + 0.5) * scale, 0, Wf - 1e-3).astype(int)
    lin = rf * Wf + cf
    order = np.argsort(lin)
    lin_s, z_s = lin[order], P[order, 2]
    uniq, start = np.unique(lin_s, return_index=True)
    ends = np.r_[start[1:], len(z_s)]
    med = np.array([np.median(z_s[a:b]) for a, b in zip(start, ends)])
    cr, cc = uniq // Wf, uniq % Wf
    inter_f = (np.repeat(np.repeat(inter, scale, 0), scale, 1)
               if scale > 1 else inter)
    tr, tc = np.where(inter_f)
    dem = np.full((Hf, Wf), np.nan, np.float32)
    dem[tr, tc] = griddata(np.column_stack([cr, cc]), med,
                           np.column_stack([tr, tc]), method="linear")
    dem[cr, cc] = med
    ftr = Affine(transform.a / scale, transform.b, transform.c,
                 transform.d, transform.e / scale, transform.f)
    with rasterio.open(out, "w", driver="GTiff", height=Hf, width=Wf,
                       count=1, dtype="float32", transform=ftr, crs=crs,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(dem, 1)
    return dem, len(uniq)


def main(n_bins=16):
    import rasterio
    from scipy import ndimage
    from skimage.measure import find_contours

    import pyintertidal as pit
    pit.net.use_system_certificates()

    t0 = time.time()
    z = np.load("marea_demo_extract.npz", allow_pickle=True)
    Y, C, keep = z["Y"], z["C"].astype(bool), z["keep"]
    H, W = (int(v) for v in z["shape"])
    dates = [str(d) for d in z["dates"]]

    with rasterio.open(f"{OUT_DIR}/water_frequency.tif") as s:
        wf = np.ma.filled(s.read(1, masked=True).astype(float), np.nan)
        transform, crs = s.transform, s.crs
    with rasterio.open(f"{OUT_DIR}/intertidal_mask.tif") as s:
        inter = s.read(1) > 0

    aoi = pit.sites.get("villaviciosa")
    lon_c, lat_c = aoi.centroid
    tides = pit.TideService(model="EOT20", directory="./tide_models",
                            location=(lat_c, lon_c))
    h_of = tides.heights_for(aoi, dates)
    h = np.array([h_of.get(d, np.nan) for d in dates])
    ok_t = np.isfinite(h)
    print(f"{int(ok_t.sum())} scenes with a tide level", flush=True)

    # ── tidal-interval composites ────────────────────────────────────────
    edges = np.quantile(h[ok_t], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    # the frame every composite is placed on: permanent water +1, land -1
    base = np.full(H * W, np.nan, np.float32)
    wfr = wf.ravel()
    base[wfr >= 0.90] = 1.0
    base[(wfr <= 0.10) | ~np.isfinite(wfr)] = -1.0

    pts, levels, example = [], [], None
    for b in range(n_bins):
        sel = ok_t & (h > edges[b]) & (h <= edges[b + 1])
        if sel.sum() < MIN_OBS_BIN:
            continue
        Yb = np.where(C[sel], Y[sel], np.nan).astype(np.float32)
        with np.errstate(all="ignore"):
            comp = np.nanmedian(Yb, axis=0)
        comp[np.sum(np.isfinite(Yb), axis=0) < MIN_OBS_BIN] = np.nan
        level = float(np.median(h[sel]))
        frame = base.copy()
        frame[keep] = comp
        frame = frame.reshape(H, W)
        finite = np.isfinite(frame)
        filled = np.where(finite, frame, -1e6)
        touch = ndimage.binary_dilation(~finite, iterations=2)
        n_pts = 0
        for path in find_contours(filled, THRESHOLD):
            if len(path) < MIN_VERTICES:
                continue
            r, c = path[:, 0], path[:, 1]
            ri = np.clip(np.round(r).astype(int), 0, H - 1)
            ci = np.clip(np.round(c).astype(int), 0, W - 1)
            good = inter[ri, ci] & ~touch[ri, ci]
            if good.sum() < MIN_VERTICES:
                continue
            pts.append(np.column_stack([r[good], c[good],
                                        np.full(good.sum(), level)]))
            n_pts += int(good.sum())
        levels.append((b, level, int(sel.sum()), n_pts))
        if example is None and abs(level + 0.4) < 0.25:
            example = (level, frame.copy())
        print(f"  bin {b:2d}: tide {level:+.2f} m · {int(sel.sum())} scenes "
              f"· {n_pts} shoreline vertices", flush=True)
    P = np.concatenate(pts)
    print(f"{len(P):,} vertices from {len(levels)} composites "
          f"({time.time() - t0:.0f} s)", flush=True)

    dem25, n25 = grid_stack(P, H, W, SCALE, inter, transform, crs,
                            f"{OUT_DIR}/waterline_v1_2p5m.tif")
    dem10, n10 = grid_stack(P, H, W, 1, inter, transform, crs,
                            f"{OUT_DIR}/waterline_v1_10m.tif")
    np.savez_compressed(
        f"{OUT_DIR}/waterline_stack_v1.npz", points=P.astype(np.float32),
        used=np.array(levels, dtype=object),
        example_date=f"composite {example[0]:+.2f} m" if example else "",
        example_ndwi=example[1] if example else np.zeros(1),
        example_h=example[0] if example else np.nan)
    print(f"-> waterline_v1_2p5m.tif ({n25:,} cells with a shoreline) and "
          f"waterline_v1_10m.tif ({n10:,}) · {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
