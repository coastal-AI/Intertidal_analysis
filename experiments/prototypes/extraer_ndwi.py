"""Pull the intertidal pixels' NDWI into memory, once.

Everything that follows — fitting the estuary's phase lag, separating flood
from ebb, testing for ponding — needs the same NDWI values fitted over and
over with a different tide axis each time. Streaming the cube per attempt
costs ten minutes; streaming it once and keeping the pixels that matter
costs ten minutes total and makes each later fit a few seconds.

Only pixels inside the intertidal mask are kept, subsampled if there are too
many, so the array stays small enough to hold comfortably.
"""
import os, sys, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import pyintertidal as pit
from pyintertidal.cube import open_cube
from pyintertidal import water as W

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CACHE = "ndwi_cube_villaviciosa_grande_10y.nc"
OUT = "products_villaviciosa"
MAX_PIX = 40000
ROW = 32


def main():
    aoi = pit.sites.get("villaviciosa")
    cube = pit.SentinelCube(aoi, ["2016-01-01", "2025-12-31"], water="ndwi",
                            cache_path=CACHE)
    transform, crs = cube.grid
    SH = cube.shape
    dates = cube.dates

    wf = rasterio.open(f"{OUT}/water_frequency.tif").read(1)
    with rasterio.open(f"{OUT}/reference_map.tif") as s:
        ref = s.read(1)
    poly = aoi.raster_mask(transform, crs, SH)
    inter = pit.intertidal_mask(wf, ref, 0.01, 0.99, aoi_mask=poly)

    # The field survey's own pixels must survive the subsampling: they are
    # the only independent check we have.
    fx = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
    fx = fx[fx["Solution status"] == "FIX"]
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(fx["Longitude"].values, fx["Latitude"].values)
    col = ((x - transform.c) / transform.a).astype(int)
    row = ((y - transform.f) / transform.e).astype(int)
    uniq, inv = np.unique(row * SH[1] + col, return_inverse=True)
    cnt = np.bincount(inv)
    gnss = np.bincount(inv, weights=fx["Ellipsoidal height"].values) / cnt
    gy_ = np.bincount(inv, weights=y) / cnt
    field_flat = uniq

    flat = np.flatnonzero(inter.ravel())
    keep = np.union1d(flat, field_flat[inter.ravel()[field_flat]])
    if keep.size > MAX_PIX:
        rng = np.random.default_rng(3)
        extra = np.setdiff1d(keep, field_flat)
        pick = rng.choice(extra, MAX_PIX - field_flat.size, replace=False)
        keep = np.union1d(pick, field_flat[inter.ravel()[field_flat]])
    keep = np.sort(keep)
    print(f"{inter.sum():,} px intermareales -> {keep.size:,} guardados "
          f"({np.isin(keep, field_flat).sum()} con medida de campo)", flush=True)

    kr, kc = keep // SH[1], keep % SH[1]

    ds, arrays, t_dim = open_cube(CACHE, ("B03", "B08", "SCL"))
    b03, b08, scl = arrays["B03"], arrays["B08"], arrays["SCL"]
    T = scl.sizes[t_dim]
    H = scl.sizes[scl.dims[1]]

    Y = np.full((T, keep.size), np.nan, np.float32)
    C = np.zeros((T, keep.size), bool)
    t0 = time.time()
    for y0 in range(0, H, ROW):
        y1 = min(y0 + ROW, H)
        sel = (kr >= y0) & (kr < y1)
        if not sel.any():
            continue
        sl = {scl.dims[1]: slice(y0, y1)}
        g = np.asarray(b03.isel(sl).values, np.float32)
        n = np.asarray(b08.isel(sl).values, np.float32)
        s = np.nan_to_num(np.asarray(scl.isel(sl).values),
                          nan=0.0).astype(np.int16)
        den = g + n
        with np.errstate(invalid="ignore", divide="ignore"):
            nd = np.where(den != 0, (g - n) / den, np.nan).astype(np.float32)
        ok = np.isin(s, list(W.CLEAR_CLASSES)) & np.isfinite(nd)
        rr_, cc_ = kr[sel] - y0, kc[sel]
        Y[:, sel] = nd[:, rr_, cc_]
        C[:, sel] = ok[:, rr_, cc_]
        done = y1 / H
        print(f"[extraer] filas {y1}/{H} ({100*done:4.1f} %)  "
              f"faltan ~{(time.time()-t0)*(1-done)/max(done,1e-9)/60:.1f} min",
              flush=True)
    ds.close()

    np.savez_compressed(
        os.path.join(SC, "ndwi_intermareal.npz"),
        Y=Y, C=C, keep=keep, dates=np.array(dates),
        field_flat=field_flat, gnss=gnss, gy=gy_, shape=np.array(SH))
    print(f"\nguardado: {Y.shape[0]} fechas x {Y.shape[1]:,} px, "
          f"{C.sum() / C.size * 100:.1f} % observaciones utiles", flush=True)


if __name__ == "__main__":
    main()
