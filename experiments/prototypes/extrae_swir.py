"""Pull NDWI, MNDWI and red out of the new cube, onto the existing grid.

The new download carries B04 and B11 as well as B03 and B08, which is what
makes the double-index diagnostic possible: NDWI sees water through the near
infrared, MNDWI through the short-wave infrared. Their noise is largely
independent while the ground under them is identical, so fitting both and
differencing the two elevations cancels topography and leaves radiometry. The
red band then says whether that residue tracks turbidity.

The grid needed care. Asking for the site's own bounding box produced an
887x673 cube; converting the existing cube's UTM box to degrees produced
967x915, because a UTM rectangle is not a rectangle in latitude and taking
its corner extremes inflates it. Rather than spend a third download, note
that the 967x915 cube CONTAINS the 915x915 grid exactly — identical x, y
offset by 26 rows — and slice it. Verified here rather than assumed, because
a silent index mismatch of exactly this kind invalidated a morning's work.

Dates are matched against the existing npz rather than trusted to line up:
the two cubes were ordered separately and a shifted date vector would pair
each pixel with the wrong tide.
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
NEW = "swir_cube_villaviciosa_2023-2025.nc"
ROW_CHUNK = 64
CLEAR = (4, 5, 6, 7)          # SCL classes that are usable ground/water


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    keep, dates_old = d["keep"], np.array([str(s) for s in d["dates"]])
    SH = tuple(int(v) for v in d["shape"])

    ds = xr.open_dataset(NEW)
    t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
    nx, ny = ds["x"].values, ds["y"].values

    old = xr.open_dataset("ndwi_cube_villaviciosa_grande_10y.nc")
    ox, oy = old["x"].values, old["y"].values
    ix = int(np.argmin(np.abs(nx - ox[0])))
    iy = int(np.argmin(np.abs(ny - oy[0])))
    assert np.allclose(nx[ix:ix + len(ox)], ox, atol=0.51), "x no encaja"
    assert np.allclose(ny[iy:iy + len(oy)], oy, atol=0.51), "y no encaja"
    print(f"recorte verificado: filas {iy}:{iy+len(oy)}, "
          f"columnas {ix}:{ix+len(ox)}  ->  {(len(oy), len(ox))}")
    assert (len(oy), len(ox)) == SH, "el recorte no da la rejilla del npz"

    dates_new = np.array([str(np.datetime64(v, "D"))
                          for v in ds[t_dim].values])
    order = np.argsort(dates_new)
    dates_new = dates_new[order]
    common, i_old, i_new = np.intersect1d(dates_old, dates_new,
                                          return_indices=True)
    print(f"fechas: {len(dates_old)} en el npz, {len(dates_new)} en el cubo "
          f"nuevo, {len(common)} comunes")

    rows = keep // SH[1]
    cols = keep % SH[1]
    T, P = len(common), len(keep)
    out = {k: np.full((T, P), np.nan, np.float32)
           for k in ("ndwi", "mndwi", "red")}
    clear = np.zeros((T, P), bool)

    t_sel = order[i_new]
    print(f"extrayendo {T} x {P:,} en bloques de {ROW_CHUNK} filas",
          flush=True)
    for r0 in range(0, SH[0], ROW_CHUNK):
        r1 = min(r0 + ROW_CHUNK, SH[0])
        sel = (rows >= r0) & (rows < r1)
        if not sel.any():
            continue
        sl = dict(y=slice(iy + r0, iy + r1), x=slice(ix, ix + SH[1]))
        blk = {b: ds[b].isel(**sl).isel({t_dim: t_sel}).values.astype(np.float32)
               for b in ("B03", "B04", "B08", "B11")}
        scl = ds["SCL"].isel(**sl).isel({t_dim: t_sel}).values
        rr = rows[sel] - r0
        cc = cols[sel]
        g = blk["B03"][:, rr, cc]
        n = blk["B08"][:, rr, cc]
        s1 = blk["B11"][:, rr, cc]
        red = blk["B04"][:, rr, cc]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ndwi"][:, sel] = np.where(g + n != 0, (g - n) / (g + n),
                                           np.nan)
            out["mndwi"][:, sel] = np.where(g + s1 != 0, (g - s1) / (g + s1),
                                            np.nan)
        out["red"][:, sel] = red
        clear[:, sel] = np.isin(scl[:, rr, cc], CLEAR)
        if (r0 // ROW_CHUNK) % 4 == 0:
            print(f"  fila {r0}/{SH[0]}", flush=True)

    for k, v in out.items():
        good = np.isfinite(v) & clear
        print(f"{k:6s} observaciones utiles {good.sum():,} "
              f"({100*good.mean():.0f} %) · mediana {np.nanmedian(v[good]):+.3f}")

    np.savez_compressed(os.path.join(SC, "swir_intermareal.npz"),
                        ndwi=out["ndwi"], mndwi=out["mndwi"], red=out["red"],
                        clear=clear, dates=common, keep=keep,
                        shape=np.array(SH), i_old=i_old)
    print(f"\nguardado swir_intermareal.npz")


if __name__ == "__main__":
    main()
