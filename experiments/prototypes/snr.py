"""Is the compression a tide problem or a signal-to-noise problem?

Everything tidal has now failed. There is no phase lag (-8.8 min, i.e. none),
ponding would show the opposite sign to what we measure, and amplification
would need the wave to grow 2.5x in 700 m. So consider the explanation that
needs no estuary at all:

a noisy fit shrinks. If a pixel's NDWI barely separates wet from dry — weak
sigmoid amplitude b — its elevation is poorly constrained and the fitted mu
drifts toward the middle of the tide range. Averaged over many such pixels
that IS a slope below one, it appears in any estimator limited by the same
noise, and it has nothing to do with tides.

The prediction that distinguishes it: compression should track fit quality,
not position. So sort the field pixels by b and by fit residual, and see
whether the slope follows those instead of distance — and whether distance
survives once quality is in the model.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
from scipy import stats

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss, gy = d["field_flat"], d["gnss"], d["gy"]

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide = model_tides(x=[lon_c], y=[lat_c], time=times, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    # Recent epoch only, to match the products.
    yrs = np.array([int(s[:4]) for s in dates])
    sel = (yrs >= 2023) & np.isfinite(tide)
    print(f"{sel.sum()} fechas de 2023-2025", flush=True)

    Yv = np.nan_to_num(Y[sel], nan=0.0).astype(np.float64)
    Cv = C[sel].astype(np.float64)
    t = tide[sel]
    mu_grid = np.linspace(t.min(), t.max(), MU_POINTS)
    a, b, mu, sg, rmse, N = _fit_block(Yv, Cv, t, mu_grid, SG_GRID)
    print(f"ajuste hecho para {Y.shape[1]:,} px", flush=True)

    # Line up the field survey with the pixels we kept.
    pos = {int(k): i for i, k in enumerate(keep)}
    idx = np.array([pos.get(int(f), -1) for f in field_flat])
    have = idx >= 0
    idx, G, D = idx[have], gnss[have], gy[have]
    D = (D.max() - D) / 1000.0

    ok = (N[idx] >= 8) & (b[idx] > 0.15) & np.isfinite(mu[idx])
    idx, G, D = idx[ok], G[ok], D[ok]
    V, B, R, S = mu[idx], b[idx], rmse[idx], sg[idx]
    n = len(idx)
    print(f"{n} pixeles de campo con ajuste valido\n")

    print("correlaciones entre los candidatos")
    for nm, v in [("b (amplitud)", B), ("rmse", R), ("sigma", S)]:
        print(f"  distancia vs {nm:14s} r = {np.corrcoef(D, v)[0,1]:+.3f}")
    print()

    def slope(m):
        return float(np.polyfit(G[m], V[m], 1)[0])

    print("PENDIENTE PARTIENDO POR CADA CANDIDATO (mitades)")
    print(f"{'criterio':22s} {'mitad baja':>11s} {'mitad alta':>11s} "
          f"{'cambio':>8s}")
    print("-" * 56)
    for nm, v, lab in [("distancia rio arriba", D, "cerca/lejos"),
                       ("b (amplitud NDWI)", B, "b bajo/alto"),
                       ("rmse del ajuste", R, "bueno/malo"),
                       ("sigma (rugosidad)", S, "fino/grueso")]:
        o = np.argsort(v)
        lo, hi = o[:n // 2], o[n // 2:]
        print(f"{nm:22s} {slope(lo):11.3f} {slope(hi):11.3f} "
              f"{slope(hi) - slope(lo):+8.3f}")

    print("\nMODELO CONJUNTO  V = a + b1*G + c_dist*(G*D) + c_b*(G*B)")
    print("cual de los dos terminos de interaccion sobrevive al otro\n")
    Gc, Dc, Bc = G - G.mean(), D - D.mean(), B - B.mean()
    for lab, cols in [("solo distancia", [Gc, Dc, Gc * Dc]),
                      ("solo amplitud b", [Gc, Bc, Gc * Bc]),
                      ("ambos", [Gc, Dc, Bc, Gc * Dc, Gc * Bc])]:
        X = np.column_stack([np.ones(n)] + cols)
        beta, *_ = np.linalg.lstsq(X, V, rcond=None)
        resid = V - X @ beta
        s2 = (resid ** 2).sum() / (n - X.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        print(f"  {lab}")
        names = (["G", "D", "G*D"] if lab == "solo distancia" else
                 ["G", "B", "G*B"] if lab == "solo amplitud b" else
                 ["G", "D", "B", "G*D", "G*B"])
        for i, nm in enumerate(names, start=1):
            t_ = beta[i] / se[i]
            p = 2 * (1 - stats.t.cdf(abs(t_), n - X.shape[1]))
            star = " *" if p < 0.05 else ""
            print(f"    {nm:5s} {beta[i]:+8.3f}  t {t_:+6.2f}  p {p:.4f}{star}")
        print(f"    R2 = {1 - resid.var() / V.var():.3f}\n")

    json.dump({"n": n,
               "r_dist_b": float(np.corrcoef(D, B)[0, 1])},
              open(os.path.join(SC, "snr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
