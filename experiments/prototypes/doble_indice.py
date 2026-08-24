"""Two water indices on the same ground: what is left is optics.

The last physical hypothesis standing. NDWI detects water through the near
infrared, MNDWI through the short-wave infrared. Fit the same sigmoid against
the same tide with each, on the same pixels and the same scenes, and the
topography under them is identical by construction. Whatever separates the
two elevations,

    delta = z_MNDWI - z_NDWI

is radiometric: it cannot be terrain, because terrain cancels.

This is the diagnostic the extra bands were downloaded for, and it is the only
hypothesis about the residual error that has not yet been tested. Everything
else has failed its control today: a tilted water surface in three variants,
inverse-barometer surge, censoring, morphological drift between imagery and
survey, and finally the compression itself, which turned out to be the
artefact of comparing a point survey with a pixel median.

Three things are asked of delta, in order of how much they would be worth:

  1. is it bigger than two independent fits of the SAME truth would produce by
     chance? Two noisy estimates of one number differ; the null says by how
     much. Without that the exercise is meaningless.
  2. does it track TURBIDITY? Red reflectance on the wet observations is the
     standard proxy, and suspended sediment is what would plausibly move the
     two indices differently in an estuary carrying river load.
  3. does it predict where the satellite disagrees with the survey? That is
     the only question that matters for the product, and it is asked last and
     once.

A caveat recorded before the numbers: B11 is a 20 m band resampled to 10 m, so
MNDWI noise is correlated between neighbouring pixels while NDWI noise is not.
That inflates delta's spatial structure without any optics being involved,
which is precisely why claim 2 is tested against red reflectance rather than
against position.

Tides are taken at the REAL acquisition times from the STAC catalogue (11:21
UTC median, not the 11:00 assumed until today).
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
from scipy.special import erf

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
MIN_B = 0.15
N_NULL = 30
CHUNK = 4000


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def fit_all(Y, C, tide):
    grid = np.linspace(tide.min(), tide.max(), MU_POINTS)
    P = Y.shape[1]
    a = np.zeros(P); b = np.zeros(P); z = np.full(P, np.nan)
    sg = np.zeros(P); N = np.zeros(P)
    for j in range(0, P, CHUNK):
        s = slice(j, j + CHUNK)
        aa, bb, mu, ss, _, n = _fit_block(Y[:, s], C[:, s], tide, grid,
                                          SG_GRID)
        a[s], b[s], z[s], sg[s], N[s] = aa, bb, mu, ss, n
    ok = (N >= MIN_OBS) & (b > MIN_B) & np.isfinite(z)
    return a, b, np.where(ok, z, np.nan), sg, N, ok


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    S = np.load(os.path.join(SC, "swir_intermareal.npz"), allow_pickle=True)
    ndwi, mndwi, red = S["ndwi"], S["mndwi"], S["red"]
    clear, keep, dates = S["clear"], S["keep"], np.array(
        [str(s) for s in S["dates"]])
    SH = tuple(int(v) for v in S["shape"])

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2023-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t = pd.DatetimeIndex(pd.to_datetime([times[s] for s in dates[have]])
                         ).tz_localize(None)
    tide = model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)
    print(f"{int(have.sum())} escenas con hora real · marea "
          f"{tide.min():+.2f} a {tide.max():+.2f} m")

    Yn = np.nan_to_num(ndwi[have], nan=0.0).astype(np.float64)
    Ym = np.nan_to_num(mndwi[have], nan=0.0).astype(np.float64)
    Cn = (clear[have] & np.isfinite(ndwi[have])).astype(np.float64)
    Cm = (clear[have] & np.isfinite(mndwi[have])).astype(np.float64)
    Rd = red[have]

    an, bn, zn, sgn, Nn, okn = fit_all(Yn, Cn, tide)
    am, bm, zm, sgm, Nm, okm = fit_all(Ym, Cm, tide)
    both = okn & okm
    print(f"pixeles con ajuste util: NDWI {int(okn.sum()):,}, "
          f"MNDWI {int(okm.sum()):,}, ambos {int(both.sum()):,}\n")

    delta = zm - zn
    dg = delta[both]
    print(f"delta = z_MNDWI - z_NDWI")
    print(f"  mediana {np.median(dg):+.3f} m · sd {np.std(dg):.3f} m · "
          f"p5 {np.percentile(dg,5):+.3f} · p95 {np.percentile(dg,95):+.3f}")

    # ── 1. the null: two fits of the SAME truth ─────────────────────────
    rng = np.random.default_rng(3)
    sub = np.where(both)[0]
    sub = rng.choice(sub, min(8000, len(sub)), replace=False)
    z0 = zn[sub]
    resid_n = np.sqrt((np.where(Cn[:, sub] > 0,
                                (Yn[:, sub] - (an[None, sub] + bn[None, sub]
                                 * phi((tide[:, None] - zn[None, sub])
                                       / np.maximum(sgn[None, sub], 1e-3)))
                                 ), 0.0) ** 2).sum(0)
                      / np.maximum((Cn[:, sub] > 0).sum(0), 1))
    resid_m = np.sqrt((np.where(Cm[:, sub] > 0,
                                (Ym[:, sub] - (am[None, sub] + bm[None, sub]
                                 * phi((tide[:, None] - zm[None, sub])
                                       / np.maximum(sgm[None, sub], 1e-3)))
                                 ), 0.0) ** 2).sum(0)
                      / np.maximum((Cm[:, sub] > 0).sum(0), 1))
    print(f"\nnulo: dos ajustes de la MISMA cota, con el ruido y las "
          f"amplitudes reales de cada indice")
    sds = []
    for k in range(3):
        r = np.random.default_rng(50 + k)
        cn = an[None, sub] + bn[None, sub] * phi(
            (tide[:, None] - z0[None, :]) / np.maximum(sgn[None, sub], 1e-3))
        cm = am[None, sub] + bm[None, sub] * phi(
            (tide[:, None] - z0[None, :]) / np.maximum(sgm[None, sub], 1e-3))
        _, _, za, _, _, oa = fit_all(
            cn + r.normal(0, 1, cn.shape) * resid_n[None, :], Cn[:, sub], tide)
        _, _, zb, _, _, ob = fit_all(
            cm + r.normal(0, 1, cm.shape) * resid_m[None, :], Cm[:, sub], tide)
        g = oa & ob
        sds.append(float(np.std((zb - za)[g])))
        print(f"  replica {k+1}: sd del delta espurio {sds[-1]:.3f} m "
              f"({int(g.sum()):,} px)")
    sd_null = float(np.mean(sds))
    sd_real = float(np.std(delta[sub][np.isfinite(delta[sub])]))
    print(f"\n  delta real sd {sd_real:.3f} m  ·  nulo {sd_null:.3f} m  ·  "
          f"razon {sd_real/max(sd_null,1e-9):.2f}x")
    if sd_real < 1.3 * sd_null:
        print("  El desacuerdo entre indices NO supera al de dos ajustes")
        print("  independientes de la misma verdad: no hay sesgo optico que ver.")

    # ── 2. does delta track turbidity? ──────────────────────────────────
    wet = (Cn > 0) & (Yn > 0)
    red_wet = np.where(wet, Rd, np.nan)
    with np.errstate(invalid="ignore"):
        turb = np.nanmedian(red_wet, axis=0)
    g = both & np.isfinite(turb) & np.isfinite(delta)
    r = stats.spearmanr(turb[g], delta[g])
    print(f"\ndelta contra turbidez (rojo mediano en observaciones mojadas)")
    print(f"  Spearman {r.statistic:+.3f} sobre {int(g.sum()):,} px")
    print(f"  (p nominal no vale nada aqui: los pixeles vecinos no son")
    print(f"   independientes, y B11 viene remuestreado de 20 m)")

    # ── 3. does delta predict the disagreement with the survey? ─────────
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    ff, gn = d["field_flat"], d["gnss"]
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in ff])
    hv = fi >= 0
    fi, y, flat = fi[hv], gn[hv], ff[hv]
    rr, cc = flat // SH[1], flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]
    df, zf = delta[fi], zn[fi]
    m = np.isfinite(df) & np.isfinite(y) & np.isfinite(zf)
    err = y[m] - zf[m]
    err = err - np.median(err)
    rr2 = stats.spearmanr(np.abs(df[m]), np.abs(err))
    print(f"\ndelta contra el desacuerdo con el GNSS ({int(m.sum())} puntos)")
    print(f"  Spearman(|delta|, |error|) = {rr2.statistic:+.3f}  "
          f"p = {rr2.pvalue:.3g}")

    print()
    if sd_real >= 1.3 * sd_null and abs(rr2.statistic) > 0.2:
        print("HAY SESGO OPTICO Y PREDICE EL ERROR: merece modelarse.")
    elif sd_real >= 1.3 * sd_null:
        print("Hay desacuerdo optico real entre indices, pero NO predice")
        print("donde falla el satelite contra el terreno. Es ruido con")
        print("estructura, no una correccion aprovechable.")
    else:
        print("NO HAY SESGO OPTICO SEPARABLE: los dos indices discrepan lo")
        print("que discreparian dos ajustes cualesquiera de la misma verdad.")
        print("Se cierra la ultima hipotesis fisica en pie.")

    json.dump({"delta_sd": sd_real, "null_sd": sd_null,
               "spearman_turbidez": float(r.statistic),
               "spearman_error": float(rr2.statistic),
               "n_both": int(both.sum())},
              open(os.path.join(SC, "doble_indice.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(SC, "doble_indice.npz"),
                        delta=delta, z_ndwi=zn, z_mndwi=zm, turb=turb,
                        both=both, keep=keep)


if __name__ == "__main__":
    main()
