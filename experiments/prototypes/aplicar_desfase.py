"""Apply the measured lag gradient and see whether anything improves.

Measured: the tide reaches the head of the ria about 7.3 min later per
kilometre, ~38 min over the 6.94 km axis, which matches the shallow-water
travel time for a 3 m mean depth. A single global lag averaged that to zero
and hid it.

So give each pixel the tide of its own position — h(t - tau(x)) — and check
three things against the flat-tide fit:

  * does the elevation agree better with the RTK survey;
  * does the compression slope move toward one;
  * does the upstream slope gradient, the anomaly that started all of this,
    shrink.

Honest limit, stated before the numbers: the survey spans a small part of the
axis, so the lag varies little across it and the RMSE has little room to
move. The gradient tests are the informative ones — the correction's real
benefit would be over the rest of the ria, where we have nothing to check it
against.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
SLOPE_MIN_PER_KM = 7.3
N_BANDS = 12
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(d["shape"])

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        transform = s.transform
    rows = keep // SH[1]
    north = transform.f + (rows + 0.5) * transform.e
    dist = (north.max() - north) / 1000.0

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    band_id = np.clip((dist / dist.max() * N_BANDS).astype(int), 0, N_BANDS - 1)
    centres = np.array([dist[band_id == i].mean() if (band_id == i).any()
                        else np.nan for i in range(N_BANDS)])
    # Centred so the correction is a pure gradient: the absolute lag folds in
    # EOT20's own phase error and is not ours to claim.
    lags = np.round((centres - np.nanmean(centres)) * SLOPE_MIN_PER_KM)

    offs = sorted(set(lags[np.isfinite(lags)].astype(int)) | {0})
    uniq_t = pd.DatetimeIndex(np.concatenate(
        [(base - pd.Timedelta(minutes=int(o))).values for o in offs])
    ).unique().sort_values()
    df = model_tides(x=[lon_c], y=[lat_c], time=uniq_t, model="EOT20",
                     directory="tide_models", crs="EPSG:4326",
                     extrapolate=True, cutoff=np.inf,
                     parallel=False).reset_index()
    lut = pd.Series(df["tide_height"].to_numpy(float),
                    index=pd.DatetimeIndex(df["time"])).groupby(level=0).first()

    def tide_at(shift):
        return lut.reindex(base - pd.Timedelta(minutes=int(shift))
                           ).to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    recent = yrs >= 2023
    print(f"{recent.sum()} fechas, {len(keep):,} px, {N_BANDS} bandas")
    print(f"correccion aplicada: {lags.min():+.0f} a {lags.max():+.0f} min\n",
          flush=True)

    def fit_with(lag_per_band):
        mu_all = np.full(Y.shape[1], np.nan)
        for i in range(N_BANDS):
            cols = np.flatnonzero(band_id == i)
            if not cols.size:
                continue
            t = tide_at(lag_per_band[i])
            m = recent & np.isfinite(t)
            grid = np.linspace(np.nanmin(t[m]), np.nanmax(t[m]), MU_POINTS)
            a, b, mu, sg, rmse, N = _fit_block(
                np.nan_to_num(Y[m][:, cols], nan=0.0).astype(np.float64),
                C[m][:, cols].astype(np.float64), t[m], grid, SG_GRID)
            mu_all[cols] = np.where((N >= 8) & (b > 0.15), mu, np.nan)
        return mu_all

    t0 = time.time()
    flat = fit_with(np.zeros(N_BANDS))
    grad = fit_with(lags)
    print(f"dos ajustes en {(time.time()-t0)/60:.1f} min\n", flush=True)

    pos = {int(k): i for i, k in enumerate(keep)}
    idx = np.array([pos.get(int(f), -1) for f in field_flat])
    have = idx >= 0
    idx, G = idx[have], gnss[have]
    Df = dist[idx]

    both = np.isfinite(flat[idx]) & np.isfinite(grad[idx])
    idx, G, Df = idx[both], G[both], Df[both]
    n = len(idx)
    print(f"{n} pixeles de campo comunes")
    print(f"  abarcan {Df.min():.2f}-{Df.max():.2f} km del eje  "
          f"=> la correccion varia solo "
          f"{(Df.max()-Df.min())*SLOPE_MIN_PER_KM:.0f} min entre ellos\n")

    rng = np.random.default_rng(2026)
    bidx = rng.integers(0, n, size=(20000, n))
    res = {}
    print(f"{'ajuste':22s} {'RMSE':>7s} {'r':>7s} {'pendiente':>10s}")
    print("-" * 50)
    for lab, arr in [("marea plana", flat), ("gradiente de fase", grad)]:
        v = arr[idx]
        dd = G - v
        dd = dd - np.median(dd)
        res[lab] = dd
        print(f"{lab:22s} {np.sqrt(np.mean(dd**2)):7.3f} "
              f"{np.corrcoef(G, v)[0,1]:7.3f} "
              f"{np.polyfit(G, v, 1)[0]:10.3f}")

    a_, b_ = res["marea plana"] ** 2, res["gradiente de fase"] ** 2
    obs = np.sqrt(b_.mean()) - np.sqrt(a_.mean())
    boot = np.sqrt(b_[bidx].mean(1)) - np.sqrt(a_[bidx].mean(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\ncambio de RMSE al corregir: {obs:+.4f} m  "
          f"IC 95% [{lo:+.4f},{hi:+.4f}]")
    print("  " + ("SIGNIFICATIVO" if lo * hi > 0 else "no significativo"))

    print("\nGRADIENTE DE COMPRESION (la anomalia que originó todo)")
    Gc, Dc = G - G.mean(), Df - Df.mean()
    X = np.column_stack([np.ones(n), Gc, Dc, Gc * Dc])
    for lab, arr in [("marea plana", flat), ("gradiente de fase", grad)]:
        v = arr[idx]
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        r = v - X @ beta
        s2 = (r ** 2).sum() / (n - 4)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        t_ = beta[3] / se[3]
        p = 2 * (1 - stats.t.cdf(abs(t_), n - 4))
        print(f"  {lab:22s} interaccion {beta[3]:+7.3f}  t {t_:+6.2f}  "
              f"p {p:.4f}")
    print("\nsi el termino se acerca a 0, la correccion de fase explica")
    print("parte de la anomalia; si no se mueve, no era la fase")

    json.dump({"n": n, "lags": lags.tolist(),
               "rmse_plana": float(np.sqrt(np.mean(res['marea plana']**2))),
               "rmse_grad": float(np.sqrt(np.mean(res['gradiente de fase']**2))),
               "delta": float(obs), "ic": [float(lo), float(hi)]},
              open(os.path.join(SC, "aplicar_desfase.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
