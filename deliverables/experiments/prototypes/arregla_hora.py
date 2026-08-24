"""Refit with the tide read at the real acquisition time.

Sentinel-2 crosses Villaviciosa at 11:21 UTC, not the 11:00 this project has
assumed for every scene. The offset is nearly constant — median +21.1 min,
5th to 95th percentile spanning only 19 minutes — so it is a systematic phase
error rather than scatter, and it costs 0.134 m of standard deviation in the
water level fed to every pixel fit, with excursions to 0.36 m.

That is not a rounding detail. The method's own disagreement with the survey
is about 0.20 m, and the sub-pixel sampling of the RTK accounts for 0.214 m of
it. A 0.134 m error in the independent variable sits in the same range as
everything this project has spent the day trying to explain, and unlike those
it is simply wrong and fixable.

The times come from the STAC catalogue through
pyintertidal.overpass.get_overpass_times, which exists for exactly this and
had not been wired into the analysis.

What is measured here: the same fit, the same pixels, the same survey, with
only the tide instants changed. Everything reported as an improvement is
therefore attributable to the correction and nothing else. The comparison is
paired over survey points so the two are not judged on different samples.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
N_BOOT = 4000


def fit(Y, C, tide):
    grid = np.linspace(tide.min(), tide.max(), MU_POINTS)
    P = Y.shape[1]
    z = np.full(P, np.nan)
    ok = np.zeros(P, bool)
    for j in range(0, P, 4000):
        s = slice(j, j + 4000)
        a, b, mu, sg, _, N = _fit_block(Y[:, s], C[:, s], tide, grid, SG_GRID)
        z[s] = mu
        ok[s] = (N >= MIN_OBS) & (b > 0.15)
    return np.where(ok, z, np.nan)


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    dates = np.array([str(s) for s in dates])

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    print(f"{len(times)} fechas con hora real del catalogo")

    yrs = np.array([int(s[:4]) for s in dates])
    have = np.array([(s in times) and (int(s[:4]) >= 2023) for s in dates])
    print(f"escenas 2023-2025 con hora real: {int(have.sum())} de "
          f"{int((yrs>=2023).sum())}")

    ds = dates[have]
    real = pd.DatetimeIndex(pd.to_datetime([times[s] for s in ds])
                            ).tz_localize(None)
    assumed = pd.DatetimeIndex(pd.to_datetime([f"{s} 11:00:00" for s in ds]))

    def tides_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    h_real, h_assumed = tides_at(real), tides_at(assumed)
    print(f"diferencia de marea: sd {np.std(h_assumed-h_real):.3f} m\n")

    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    ok = fi >= 0
    fi, y = fi[ok], gnss[ok]
    Yf = np.nan_to_num(Y[have][:, fi], nan=0.0).astype(np.float64)
    Cf = C[have][:, fi].astype(np.float64)

    v_old = fit(Yf, Cf, h_assumed)
    v_new = fit(Yf, Cf, h_real)

    m = np.isfinite(v_old) & np.isfinite(v_new) & np.isfinite(y)
    y, v_old, v_new = y[m], v_old[m], v_new[m]
    print(f"{len(y)} puntos de campo comparados en pareja\n")

    rng = np.random.default_rng(11)
    idx = rng.integers(0, len(y), (N_BOOT, len(y)))

    def stats(v):
        sl = np.array([np.polyfit(y[i], v[i], 1)[0] for i in idx])
        rm = np.array([np.sqrt(np.mean(
            ((y[i] - v[i]) - np.median(y[i] - v[i])) ** 2)) for i in idx])
        return sl, rm

    sl_o, rm_o = stats(v_old)
    sl_n, rm_n = stats(v_new)

    print(f"{'marea usada':28s} {'RMSE':>7s} {'IC 95%':>17s} "
          f"{'pend':>7s} {'IC 95%':>17s}")
    print("-" * 82)
    for lab, sl, rm in (("11:00 supuestas (lo de hoy)", sl_o, rm_o),
                        ("hora real del catalogo", sl_n, rm_n)):
        ci_r = np.percentile(rm, [2.5, 97.5])
        ci_s = np.percentile(sl, [2.5, 97.5])
        print(f"{lab:28s} {rm.mean():7.3f} [{ci_r[0]:6.3f},{ci_r[1]:6.3f}] "
              f"{sl.mean():7.3f} [{ci_s[0]:6.3f},{ci_s[1]:6.3f}]")

    d_rm = rm_o - rm_n
    d_sl = sl_n - sl_o
    ci_rm = np.percentile(d_rm, [2.5, 97.5])
    ci_sl = np.percentile(d_sl, [2.5, 97.5])
    print(f"\nmejora de RMSE     {d_rm.mean():+.3f} m  "
          f"IC 95% [{ci_rm[0]:+.3f}, {ci_rm[1]:+.3f}]")
    print(f"cambio de pendiente {d_sl.mean():+.3f}    "
          f"IC 95% [{ci_sl[0]:+.3f}, {ci_sl[1]:+.3f}]")
    print()
    if ci_rm[0] > 0:
        print("LA HORA IMPORTA: corregirla mejora el ajuste, y el intervalo no")
        print("cruza cero. Todos los numeros del proyecto estaban medidos con")
        print("una marea desplazada 21 minutos.")
    elif ci_rm[1] < 0:
        print("Corregir la hora EMPEORA, lo que no tiene lectura fisica y")
        print("apunta a un fallo en el emparejamiento de fechas.")
    else:
        print("La hora no cambia el resultado de forma medible con estos 195")
        print("puntos: el error de 0.134 m se promedia entre las 465 escenas")
        print("que entran en cada pixel.")

    json.dump({"n": int(len(y)),
               "rmse_old": float(rm_o.mean()), "rmse_new": float(rm_n.mean()),
               "slope_old": float(sl_o.mean()), "slope_new": float(sl_n.mean()),
               "d_rmse_ci": ci_rm.tolist(), "d_slope_ci": ci_sl.tolist()},
              open(os.path.join(SC, "arregla_hora.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(SC, "mareas_hora_real.npz"),
                        dates=ds, tide=h_real, tide_11h=h_assumed)


if __name__ == "__main__":
    main()
