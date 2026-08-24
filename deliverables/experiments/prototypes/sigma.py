"""Is sigma the sub-pixel relief it claims to be? The survey can answer.

The sigmoid fit reports two numbers per pixel. mu is the elevation, and every
paper that uses this estimator reports it. sigma is the width of the
transition, and in every published treatment it is a nuisance parameter — the
thing you fit so that mu comes out right.

This project reads it differently: if the wet fraction of a 10 m pixel is
Phi((h - mu)/sigma), then Phi is the cumulative distribution of the ground
INSIDE that pixel, mu is its median, and sigma is the standard deviation of
the elevations within it. Sub-pixel relief, as a measured product. The
literature review found no precedent for that reading, so it is the one part
of this pipeline that could be claimed.

A claim that has never been tested. Until now it could not be: the processed
survey file holds one entry per pixel. The raw RTK file does not — 361 fixed
points fall in 248 pixels, and 23 of those pixels contain three or more.
Inside such a pixel the spread of the surveyed heights IS the sub-pixel
relief, measured on the ground at 1.2 cm precision.

So the test is direct: does fitted sigma track measured spread?

Two honest limits, stated before the numbers:

  * the points inside a pixel lie along a walked track, not scattered over
    the full 100 m2. They therefore sample only part of the pixel, and the
    measured spread is a LOWER BOUND on the true one. Fitted sigma exceeding
    it is expected; falling below it is not.
  * with 23 pixels the power is small, so a null result here would not settle
    anything, while a clear correlation would be worth a great deal.

A permutation null accompanies the correlation, because 23 points can produce
a striking coefficient by luck.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rwt
from scipy import stats

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MIN_PTS = 3
N_PERM = 20000


def main():
    df = pd.read_csv("data/villaviciosa_rtk_gnss.csv")
    df = df[df["Solution status"] == "FIX"]
    lon = df["Longitude"].to_numpy(float)
    lat = df["Latitude"].to_numpy(float)
    h = df["Ellipsoidal height"].to_numpy(float)
    rms = df["Elevation RMS"].to_numpy(float)

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        tr, crs, SH = s.transform, s.crs, s.shape
    x, y = rwt("EPSG:4326", crs, lon, lat)
    col = np.floor((np.asarray(x) - tr.c) / tr.a).astype(int)
    row = np.floor((np.asarray(y) - tr.f) / tr.e).astype(int)
    ok = (row >= 0) & (row < SH[0]) & (col >= 0) & (col < SH[1])
    flat = row[ok] * SH[1] + col[ok]
    h, rms = h[ok], rms[ok]

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    keep = d["keep"]
    B = np.load(os.path.join(SC, "lamina_base.npz"))
    sg_fit, z_fit, good = B["sg"], B["z"], B["good"]
    pos = {int(k): i for i, k in enumerate(keep)}

    rows = []
    for k in np.unique(flat):
        v = h[flat == k]
        v = v[np.isfinite(v)]
        if len(v) < MIN_PTS:
            continue
        i = pos.get(int(k), -1)
        if i < 0 or not good[i]:
            continue
        rows.append((float(np.std(v, ddof=1)), float(sg_fit[i]),
                     float(np.median(v)), float(z_fit[i]), len(v),
                     float(np.ptp(v))))
    R = np.array(rows)
    print(f"precision del GNSS en cota: mediana {np.median(rms)*100:.1f} cm, "
          f"despreciable frente al relieve buscado\n")
    print(f"{len(R)} pixeles con {MIN_PTS}+ puntos dentro y ajuste util\n")

    meas, fit = R[:, 0], R[:, 1]
    print(f"{'':22s} {'mediana':>9s} {'p10':>8s} {'p90':>8s}")
    print("-" * 50)
    print(f"{'relieve MEDIDO':22s} {np.median(meas):9.3f} "
          f"{np.percentile(meas,10):8.3f} {np.percentile(meas,90):8.3f}  m")
    print(f"{'sigma AJUSTADA':22s} {np.median(fit):9.3f} "
          f"{np.percentile(fit,10):8.3f} {np.percentile(fit,90):8.3f}  m")
    print(f"\nrazon ajustada/medida: mediana {np.median(fit/np.maximum(meas,1e-3)):.2f}")
    print("se espera >1: los puntos van por una linea andada, no cubren")
    print("los 100 m2, asi que lo medido es una COTA INFERIOR del relieve real.")

    r_s = stats.spearmanr(meas, fit)
    r_p = stats.pearsonr(meas, fit)
    print(f"\nSpearman  {r_s.statistic:+.3f}   (p nominal {r_s.pvalue:.3g})")
    print(f"Pearson   {r_p.statistic:+.3f}")

    rng = np.random.default_rng(17)
    null = np.array([stats.spearmanr(meas, rng.permutation(fit)).statistic
                     for _ in range(N_PERM)])
    p_perm = float((np.abs(null) >= abs(r_s.statistic)).mean())
    print(f"\nnulo por permutacion ({N_PERM:,} barajas de la sigma ajustada)")
    print(f"  |correlacion| del nulo: p95 {np.percentile(np.abs(null),95):.3f}, "
          f"max {np.abs(null).max():.3f}")
    print(f"  p empirico = {p_perm:.4f}")

    # does the fit at least separate flat pixels from rough ones?
    lo, hi = meas <= np.median(meas), meas > np.median(meas)
    u = stats.mannwhitneyu(fit[hi], fit[lo], alternative="greater")
    print(f"\nsigma ajustada en los pixeles MAS rugosos frente a los mas llanos")
    print(f"  llanos  {np.median(fit[lo]):.3f} m   rugosos {np.median(fit[hi]):.3f} m"
          f"   p = {u.pvalue:.3g}")

    if p_perm < 0.05:
        print("\nSIGMA MIDE ALGO REAL. Es la pieza reclamable del metodo:")
        print("ningun trabajo publicado trata sigma como un producto fisico.")
    else:
        print("\nNO SE SOSTIENE con estos datos. Con 23 pixeles el poder es")
        print("corto, asi que esto no refuta la idea — pide mas puntos")
        print("agrupados dentro del mismo pixel.")

    json.dump({"n_pixels": len(R), "spearman": float(r_s.statistic),
               "p_perm": p_perm, "median_measured": float(np.median(meas)),
               "median_fitted": float(np.median(fit)),
               "mwu_p": float(u.pvalue)},
              open(os.path.join(SC, "sigma.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
