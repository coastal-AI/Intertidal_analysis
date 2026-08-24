"""Separating a real compression from the artefact of comparing point to pixel.

The satellite estimates the MEDIAN elevation of a 10 m pixel. The RTK measures
one point inside it. So the survey value is not the pixel's truth:

    x = z + e,     e = where in the pixel the surveyor happened to stand

and e sits in the PREDICTOR of the regression, which is exactly where noise
attenuates a slope. Writing the estimator's own behaviour as z_hat = a + c*z,

    slope observed = c * var(z) / (var(z) + var(e)) = c / (1 + rho)

with rho = var(e)/var(z). One regression cannot separate c from rho — they are
confounded, and a first attempt at this assumed c = 1 to solve for rho, which
is assuming the answer.

Averaging the points inside a pixel does separate them. With k points the
sampling term shrinks to var(e)/k while c is untouched, so

    slope(k) = c / (1 + rho/k)

and the RATIO between k = 1 and k = 2 depends on rho alone:

    slope(2)/slope(1) = (1 + rho) / (1 + rho/2)

That is measurable here: 77 pixels hold two or more RTK points. Solve for rho,
then recover c. Everything is computed on the SAME pixels — an earlier attempt
mixed a standard deviation taken over all 361 points with a slope taken over
the 77, and produced the impossible result that the method's own error was
exactly zero.

Restricted to pixels with EXACTLY two points, so k is not a mixture.
"""
import os
import sys
import json
import collections

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rwt

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
N_BOOT = 4000


def load():
    df = pd.read_csv("data/villaviciosa_rtk_gnss.csv")
    df = df[df["Solution status"] == "FIX"]
    lon = df["Longitude"].to_numpy(float)
    lat = df["Latitude"].to_numpy(float)
    h = df["Ellipsoidal height"].to_numpy(float)
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        tr, crs, SH = s.transform, s.crs, s.shape
        dem = s.read(1)
    x, y = rwt("EPSG:4326", crs, lon, lat)
    col = np.floor((np.asarray(x) - tr.c) / tr.a).astype(int)
    row = np.floor((np.asarray(y) - tr.f) / tr.e).astype(int)
    ok = (row >= 0) & (row < SH[0]) & (col >= 0) & (col < SH[1])
    flat = row[ok] * SH[1] + col[ok]
    return flat, h[ok], dem[row[ok], col[ok]]


def main():
    flat, h, sat = load()
    cnt = collections.Counter(flat.tolist())
    pairs = [k for k, v in cnt.items() if v == 2]
    groups = []
    for k in pairs:
        s = flat == k
        v, sv = h[s], sat[s][0]
        if np.isfinite(sv) and np.isfinite(v).all():
            groups.append((v[0], v[1], sv))
    G = np.array(groups)
    print(f"{len(G)} pixeles con EXACTAMENTE 2 puntos RTK dentro\n")

    def stats_from(idx):
        g = G[idx]
        pick = g[:, 0]                      # one point
        both = g[:, :2].mean(1)             # the pixel mean
        sv = g[:, 2]
        s1 = np.polyfit(pick, sv, 1)[0]
        s2 = np.polyfit(both, sv, 1)[0]
        # rho from the ratio: slope2/slope1 = (1+rho)/(1+rho/2)
        r = s2 / s1
        rho = 2.0 * (r - 1.0) / (2.0 - r) if r < 2 else np.nan
        c = s1 * (1.0 + rho)
        var_x = np.var(pick, ddof=1)
        var_e = var_x * rho / (1.0 + rho) if np.isfinite(rho) else np.nan
        # the two points also measure var(e) directly: their difference
        var_e_direct = np.var(g[:, 0] - g[:, 1], ddof=1) / 2.0
        res1 = sv - np.polyval(np.polyfit(pick, sv, 1), pick)
        res2 = sv - np.polyval(np.polyfit(both, sv, 1), both)
        return (s1, s2, rho, c, np.sqrt(max(var_e, 0)),
                np.sqrt(var_e_direct), np.std(res1, ddof=1),
                np.std(res2, ddof=1))

    base = stats_from(np.arange(len(G)))
    rng = np.random.default_rng(7)
    bs = np.array([stats_from(rng.integers(0, len(G), len(G)))
                   for _ in range(N_BOOT)])

    names = ["pendiente con 1 punto", "pendiente con la media de 2",
             "rho = var(e)/var(z)", "c = compresion VERDADERA",
             "sigma_e por atenuacion", "sigma_e por diferencia de pares",
             "dispersion residual con 1 punto",
             "dispersion residual con 2 puntos"]
    print(f"{'magnitud':34s} {'valor':>8s} {'IC 95%':>20s}")
    print("-" * 66)
    out = {}
    for i, nm in enumerate(names):
        col = bs[:, i]
        col = col[np.isfinite(col)]
        ci = np.percentile(col, [2.5, 97.5])
        print(f"{nm:34s} {base[i]:8.3f} [{ci[0]:8.3f},{ci[1]:8.3f}]")
        out[nm] = {"value": float(base[i]), "ci": ci.tolist()}

    print("\nLa clave son las dos ultimas filas de arriba: sigma_e sale por dos")
    print("caminos independientes. Uno es la ATENUACION de la pendiente al")
    print("promediar; el otro es la DIFERENCIA entre los dos puntos del mismo")
    print("pixel, que mide el muestreo sub-pixel sin tocar el satelite. Si")
    print("coinciden, el modelo describe los datos.")

    ci_c = np.percentile(bs[np.isfinite(bs[:, 3]), 3], [2.5, 97.5])
    print()
    print(f"COMPRESION VERDADERA c = {base[3]:.3f}  IC 95% "
          f"[{ci_c[0]:.3f}, {ci_c[1]:.3f}]")
    if ci_c[0] <= 1.0 <= ci_c[1]:
        print("El intervalo incluye 1: una vez descontado el muestreo")
        print("sub-pixel, NO QUEDA COMPRESION que explicar.")
    elif ci_c[1] < 1.0:
        print(f"Queda compresion real ({base[3]:.3f}), menor que el "
              f"{np.polyfit(G[:,0],G[:,2],1)[0]:.3f} aparente.")
    else:
        print("El intervalo esta por encima de 1, lo que no tiene lectura")
        print("fisica: probablemente 77 pares son pocos para este estimador.")

    json.dump({"n_pairs": int(len(G)),
               "results": out,
               "c_ci": ci_c.tolist()},
              open(os.path.join(SC, "atenuacion.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
