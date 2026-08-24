"""Per-pixel NDWI series for the Scheldt flats — method B's development data.

Method B fits, per along-estuary band, a locally warped tide against every
pixel's NDWI series, with the pixel parameters profiled out. That needs the
per-pixel series in memory, and for the Scheldt only band-level counts were
ever extracted. One streaming pass fixes that.

The Scheldt is the DEVELOPMENT site because it is the only one with truth for
what B estimates: two gauges whose measured transfer (M2 gain 1.0713, lag
+22.8 min; EOT20 residual phase +10 min) can grade the inferred profile.

Memory arithmetic before writing anything: 73 764 intertidal pixels x 464
scenes as float32 NDWI plus a boolean mask is ~150 MB — fine on a machine
with 1.5 GB free, and far cheaper than holding any full grid.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import xarray as xr

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CUBE = "ndwi_cube_escalda_2023-2025_20m.nc"
CLEAR = (4, 5, 6, 7)
T_CHUNK = 40
MOUTH_LON, MOUTH_LAT = 3.597, 51.443


def main():
    ds = xr.open_dataset(CUBE)
    t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
    T = ds.sizes[t_dim]
    H, W = ds.sizes["y"], ds.sizes["x"]
    dates = np.array([str(np.datetime64(v, "D")) for v in ds[t_dim].values])

    # pass 1: the intertidal mask (same recipe as the validation run)
    wet_n = np.zeros((H, W), np.int32)
    clr_n = np.zeros((H, W), np.int32)
    for i0 in range(0, T, T_CHUNK):
        sl = {t_dim: slice(i0, min(i0 + T_CHUNK, T))}
        g = ds["B03"].isel(**sl).values.astype(np.float32)
        n = ds["B08"].isel(**sl).values.astype(np.float32)
        scl = ds["SCL"].isel(**sl).values
        clear = np.isin(scl, CLEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            wet = clear & ((g - n) / np.maximum(g + n, 1) > 0)
        wet_n += wet.sum(axis=0, dtype=np.int32)
        clr_n += clear.sum(axis=0, dtype=np.int32)
    wf = np.where(clr_n > 30, wet_n / np.maximum(clr_n, 1), np.nan)
    inter = np.isfinite(wf) & (wf > 0.10) & (wf < 0.90)
    keep = np.where(inter.ravel())[0]
    rows = keep // W
    cols = keep % W
    print(f"{len(keep):,} px intermareales", flush=True)

    # along-estuary coordinate of each kept pixel, km east of the mouth
    import pyproj
    crs_cube = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt")
                                   or ds["crs"].attrs.get("spatial_ref"))
    tr = pyproj.Transformer.from_crs(4326, crs_cube, always_xy=True)
    x_mouth = tr.transform(MOUTH_LON, MOUTH_LAT)[0]
    s_km = (ds["x"].values[cols] - x_mouth) / 1000.0

    # pass 2: the series
    P = len(keep)
    Y = np.full((T, P), np.nan, np.float32)
    C = np.zeros((T, P), bool)
    for i0 in range(0, T, T_CHUNK):
        i1 = min(i0 + T_CHUNK, T)
        sl = {t_dim: slice(i0, i1)}
        g = ds["B03"].isel(**sl).values.astype(np.float32)
        n = ds["B08"].isel(**sl).values.astype(np.float32)
        scl = ds["SCL"].isel(**sl).values
        clear = np.isin(scl, CLEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndwi = np.where(g + n != 0, (g - n) / (g + n), np.nan)
        Y[i0:i1] = ndwi[:, rows, cols]
        C[i0:i1] = clear[:, rows, cols]
        if (i0 // T_CHUNK) % 4 == 0:
            print(f"  escena {i0}/{T}", flush=True)

    good = C & np.isfinite(Y)
    print(f"observaciones utiles: {good.sum():,} ({100*good.mean():.0f} %)")
    np.savez_compressed(os.path.join(SC, "escalda_intermareal.npz"),
                        Y=Y, C=C, keep=keep, s_km=s_km.astype(np.float32),
                        dates=dates, shape=np.array([H, W]))
    print("guardado escalda_intermareal.npz")


if __name__ == "__main__":
    main()
