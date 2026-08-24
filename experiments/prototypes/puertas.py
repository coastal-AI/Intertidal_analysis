"""The three gates. Opened in order; the survey is opened last and once.

Gate 1 asks whether gamma_t is physics or noise, using no ground truth at all.
Gate 3 asks whether gamma predicts a number that never entered the fit.
Gate 2 spends the held-out survey, and only if the first gate passed.

Why gamma_t must depend on the tide, not merely be non-zero
----------------------------------------------------------
A gamma that is the same on every date is a static bias field: it shifts
elevations by -gamma*s and nothing more. The relief COMPRESSION this project
measured needs gamma to change with the tide. Write gamma(h0) = g0 + g1*h0.
A pixel at distance s with true elevation z is submerged when

    h0 + (g0 + g1*h0) * s = z

so the level the fit reports, which is the h0 solving that, is

    z_hat = (z - g0*s) / (1 + g1*s)

— a shift AND a division. The division is the compression, and it gives a
prediction that is quantitative and independent:

    slope of z_hat against z, at distance s   =   1 / (1 + g1*s)

g1 comes from the fitted gammas; the slope was measured against the RTK
survey months of work ago and never entered any of this. If the two agree,
the method explains a measurement it was not fitted to. That is Gate 3.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

from pyintertidal.net import use_system_certificates
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
BLOCK_M = 150.0
HOLDOUT_FRACTION = 0.35
SEED = 20260817


def gate1(gamma, h0, dates):
    """Does gamma behave like an estuary, or like noise?"""
    use_system_certificates()
    from eo_tides.model import model_tides
    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    grid = np.vstack([
        model_tides(x=[lon_c], y=[lat_c],
                    time=base + pd.Timedelta(hours=float(h)), model="EOT20",
                    directory="tide_models", crs="EPSG:4326",
                    extrapolate=True, cutoff=np.inf,
                    parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)
        for h in np.arange(-18, 18.5, 3.0)])
    day_range = np.nanmax(grid, 0) - np.nanmin(grid, 0)

    print("PUERTA 1 — gamma contra la fisica, sin una sola etiqueta")
    print("-" * 62)
    out = {}
    for name, x, expect in (
            ("nivel de marea h0", h0,
             "gamma debe DEPENDER de h0: es lo que produce la compresion"),
            ("rango diario", day_range,
             "vivas y muertas no deben propagar igual")):
        ok = np.isfinite(x) & np.isfinite(gamma)
        r = stats.pearsonr(x[ok], gamma[ok])
        sl = np.polyfit(x[ok], gamma[ok], 1)[0]
        print(f"  {name:20s} r={r.statistic:+.3f}  p={r.pvalue:8.2g}  "
              f"pendiente {sl:+.3f} (m/km)/m")
        print(f"      {expect}")
        out[name] = {"r": float(r.statistic), "p": float(r.pvalue),
                     "slope": float(sl)}
    out["g1"] = out["nivel de marea h0"]["slope"]
    out["g0"] = float(np.polyfit(h0[np.isfinite(gamma)],
                                 gamma[np.isfinite(gamma)], 1)[1])
    print(f"\n  gamma(h0) = {out['g0']:+.3f} {out['g1']:+.3f}*h0   m/km")
    print(f"  gamma: mediana {np.median(gamma):+.3f}, "
          f"rango {gamma.min():+.3f} a {gamma.max():+.3f} m/km")
    return out


def gate3(g1, s_field_km, measured_slope):
    """The independent prediction: slope = 1 / (1 + g1*s)."""
    print("\nPUERTA 3 — prediccion independiente de la compresion")
    print("-" * 62)
    s_bar = float(np.median(s_field_km))
    pred = 1.0 / (1.0 + g1 * s_bar)
    print(f"  distancia mediana del GNSS      {s_bar:.3f} km")
    print(f"  g1 ajustado (sin ver el GNSS)   {g1:+.3f} (m/km)/m")
    print(f"  pendiente PREDICHA              {pred:.3f}")
    print(f"  pendiente MEDIDA                {measured_slope:.3f}")
    print(f"  diferencia                      {pred-measured_slope:+.3f}")
    # what g1 would have been needed
    need = (1.0 / measured_slope - 1.0) / max(s_bar, 1e-6)
    print(f"  g1 que haria falta para explicarla del todo: {need:+.3f}")
    return {"s_bar": s_bar, "g1": float(g1), "predicted": float(pred),
            "measured": float(measured_slope), "g1_needed": float(need)}


def gate2(z_new, z_base, good, keep, SH, field_flat, gnss, s_km):
    """The held-out survey. Opened once."""
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr_v, tr = s.read(1)[rr, cc], s.transform
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        dea_v = s.read(1)[rr, cc]

    east = tr.c + (cc + .5) * tr.a
    north = tr.f + (rr + .5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    blocks = np.unique(block)
    rng = np.random.default_rng(SEED)
    hold = np.isin(block, rng.choice(
        blocks, max(2, int(round(HOLDOUT_FRACTION * len(blocks)))),
        replace=False))

    print("\nPUERTA 2 — conjunto reservado del GNSS")
    print(f"  {len(y)} puntos · {len(blocks)} bloques de {BLOCK_M:.0f} m · "
          f"entrenamiento {int((~hold).sum())} / reservado {int(hold.sum())}")
    print("-" * 62)
    print(f"  {'metodo':26s} {'RMSE':>7s} {'pend':>7s} {'n':>5s}")

    rows = {}
    cands = [("HSR (producto)", hsr_v),
             ("escalon DEA (producto)", dea_v),
             ("base gamma=0 (este codigo)", np.where(good, z_base, np.nan)[fi]),
             ("lamina inclinada", np.where(good, z_new, np.nan)[fi])]
    for name, v in cands:
        m = np.isfinite(v) & np.isfinite(y)
        # datum from the TRAINING half only — in production there is no survey
        tr_m = m & ~hold
        if tr_m.sum() < 5:
            print(f"  {name:26s}   sin datos de entrenamiento")
            continue
        off = float(np.median(y[tr_m] - v[tr_m]))
        h = m & hold
        r = y[h] - (v[h] + off)
        sl = float(np.polyfit(y[h], v[h], 1)[0])
        rms = float(np.sqrt(np.mean(r ** 2)))
        print(f"  {name:26s} {rms:7.3f} {sl:7.3f} {int(h.sum()):5d}")
        rows[name] = {"rmse": rms, "slope": sl, "n": int(h.sum())}
    return rows


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    keep, field_flat, gnss = d["keep"], d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])
    L = np.load(os.path.join(SC, "lamina.npz"), allow_pickle=True)
    B = np.load(os.path.join(SC, "lamina_base.npz"))
    gamma, h0, dates = L["gamma"], L["h0"], L["dates"]
    good = L["good"] & B["good"]
    s_km = L["s_km"]

    out = {"gate1": gate1(gamma, h0, dates)}

    rr, cc = field_flat // SH[1], field_flat % SH[1]
    ch = np.load(os.path.join(SC, "canal.npz"))
    s_field = ch["geo"][rr, cc] / 1000.0
    s_field = s_field[np.isfinite(s_field)]

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr_map = s.read(1)
    hv, gv = hsr_map[rr, cc], gnss
    m = np.isfinite(hv) & np.isfinite(gv)
    measured_slope = float(np.polyfit(gv[m], hv[m], 1)[0])
    out["gate3"] = gate3(out["gate1"]["g1"], s_field, measured_slope)

    if abs(out["gate1"]["nivel de marea h0"]["r"]) < 0.15:
        print("\nPUERTA 1 NO SUPERADA: gamma no depende de la marea.")
        print("El conjunto reservado NO se abre.")
    else:
        out["gate2"] = gate2(L["z"], B["z"], good, keep, SH,
                             field_flat, gnss, s_km)

    json.dump(out, open(os.path.join(SC, "puertas.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
