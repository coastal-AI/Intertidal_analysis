"""How accurate is the method really, once the survey's own noise is removed?

Every accuracy figure this project has quoted — 0.10, 0.13, 0.20, 0.21 m
against the RTK — is a sum of two things that were never separated:

    var(observed disagreement) = var(method) + var(sub-pixel sampling)

The second term exists because the survey measures a point while the estimate
describes a 10 m pixel, and today it was shown to be large enough to fabricate
the entire apparent relief compression.

Separating them needs no model and no regression. Inside a pixel holding k
survey points, averaging them divides the sampling term by k and leaves the
method term untouched:

    var(d_1) = s_m^2 + s_e^2
    var(d_2) = s_m^2 + s_e^2 / 2

so s_e^2 = 2 (var(d_1) - var(d_2)) and s_m^2 = 2 var(d_2) - var(d_1). Two
equations, two unknowns, from a difference that is measured rather than
assumed.

Two independent routes to s_e are computed and compared, because agreement
between them is the only evidence that the decomposition describes the data:

  * the AVERAGING route above, which involves the satellite;
  * the PAIR route, half the variance of the difference between two points in
    the same pixel, which never touches the satellite at all.

A caveat that bounds the answer rather than clouding it: the points inside a
pixel were taken seconds apart along a walked line, so they sample a fraction
of the 100 m2 and s_e comes out as a LOWER bound. The method's error is
therefore an UPPER bound — the true figure can only be better.
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
N_BOOT = 6000
PRODUCTS = {
    "HSR (EOT20)": "hsr_eot20_2023-2025.tif",
    "escalon DEA": "dea_eot20_2023-2025.tif",
    "HSR (ensemble)": "hsr_ensemble_2023-2025.tif",
}


def load_pixels():
    df = pd.read_csv("data/villaviciosa_rtk_gnss.csv")
    df = df[df["Solution status"] == "FIX"]
    lon = df["Longitude"].to_numpy(float)
    lat = df["Latitude"].to_numpy(float)
    h = df["Ellipsoidal height"].to_numpy(float)
    with rasterio.open("products_villaviciosa/" + PRODUCTS["HSR (EOT20)"]) as s:
        tr, crs, SH = s.transform, s.crs, s.shape
    x, y = rwt("EPSG:4326", crs, lon, lat)
    col = np.floor((np.asarray(x) - tr.c) / tr.a).astype(int)
    row = np.floor((np.asarray(y) - tr.f) / tr.e).astype(int)
    ok = (row >= 0) & (row < SH[0]) & (col >= 0) & (col < SH[1])
    return row[ok], col[ok], h[ok], SH


def decompose(d1, d2, k=2):
    """s_e and s_m from the same disagreement measured with 1 and k points."""
    v1 = np.var(d1 - np.median(d1), ddof=1)
    v2 = np.var(d2 - np.median(d2), ddof=1)
    se2 = (v1 - v2) * k / (k - 1.0)
    sm2 = v1 - se2
    return (np.sqrt(max(se2, 0.0)), np.sqrt(max(sm2, 0.0)),
            np.sqrt(v1), np.sqrt(v2))


def main():
    row, col, h, SH = load_pixels()
    flat = row * SH[1] + col
    cnt = collections.Counter(flat.tolist())
    pairs = np.array([k for k, v in cnt.items() if v == 2])
    print(f"{len(pairs)} pixeles con exactamente 2 puntos RTK dentro\n")

    out = {}
    for name, fn in PRODUCTS.items():
        p = "products_villaviciosa/" + fn
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as s:
            dem = s.read(1)
        rows = []
        for k in pairs:
            m = flat == k
            v = h[m]
            sv = dem[row[m][0], col[m][0]]
            if np.isfinite(sv) and np.isfinite(v).all():
                rows.append((v[0], v[1], sv))
        G = np.array(rows)
        if len(G) < 20:
            continue

        rng = np.random.default_rng(4)
        idx = rng.integers(0, len(G), (N_BOOT, len(G)))
        pick = rng.integers(0, 2, (N_BOOT, len(G)))
        est = []
        for b in range(N_BOOT):
            g = G[idx[b]]
            one = g[np.arange(len(g)), pick[b]]
            both = g[:, :2].mean(1)
            sv = g[:, 2]
            est.append(decompose(one - sv, both - sv))
        est = np.array(est)
        # pair route: half the variance of the within-pixel difference
        se_pair = np.array([np.sqrt(np.var(np.diff(G[i][:, :2], axis=1),
                                           ddof=1) / 2.0) for i in idx])

        base = decompose(G[:, 0] - G[:, 2], G[:, :2].mean(1) - G[:, 2])
        print(f"{name}  ({len(G)} pixeles)")
        print(f"  {'magnitud':38s} {'valor':>7s} {'IC 95%':>17s}")
        print(f"  {'-'*64}")
        labels = ["desacuerdo observado con 1 punto",
                  "desacuerdo observado con la media de 2",
                  "muestreo sub-pixel s_e  (via promediado)",
                  "ERROR DEL METODO s_m"]
        order = [2, 3, 0, 1]
        for lab, i in zip(labels, order):
            ci = np.percentile(est[:, i], [2.5, 97.5])
            print(f"  {lab:38s} {base[i]:7.3f} [{ci[0]:7.3f},{ci[1]:7.3f}]")
        ci = np.percentile(se_pair, [2.5, 97.5])
        print(f"  {'s_e por diferencia de pares (control)':38s} "
              f"{se_pair.mean():7.3f} [{ci[0]:7.3f},{ci[1]:7.3f}]")
        frac = 100 * (1 - base[1] ** 2 / base[2] ** 2)
        print(f"\n  del error que se reportaba, el {frac:.0f} % era la "
              f"referencia, no el metodo\n")
        out[name] = {"obs_1pt": float(base[2]), "obs_2pt": float(base[3]),
                     "s_e": float(base[0]), "s_m": float(base[1]),
                     "s_m_ci": np.percentile(est[:, 1], [2.5, 97.5]).tolist(),
                     "s_e_pair": float(se_pair.mean()), "n": int(len(G))}

    print("El s_e por promediado y el s_e por diferencia de pares son")
    print("independientes: uno pasa por el satelite y el otro no. Si coinciden,")
    print("la descomposicion describe los datos.")
    print()
    print("Y recuerda la cota: los dos puntos de cada par se tomaron con")
    print("segundos de diferencia, asi que infra-muestrean el pixel. s_e es")
    print("cota INFERIOR y por tanto s_m es cota SUPERIOR — el metodo solo")
    print("puede ser mejor que esto.")

    json.dump(out, open(os.path.join(SC, "varianzas.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
