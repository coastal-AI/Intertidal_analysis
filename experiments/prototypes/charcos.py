"""Ponding: the one consequence of connectivity that needs no fitting at all.

Every published method models a pixel as wet iff the water level is above its
elevation. That makes wetness a function of level alone, so the response is
symmetric: whatever the pixel does on a rising tide it must undo on a falling
one. The sigmoid Phi((h-z)/sigma) has that symmetry built in and cannot
express anything else.

Connectivity breaks it. A pixel in a hollow floods LATE — the water has to
clear the surrounding rim first — and then does not drain at all, because
the same rim traps it. Floods late, stays wet. That is a hysteresis, and no
sigmoid can fit it: the fitted elevation of such a pixel is not wrong by a
little, it is meaningless.

So the test, and it uses no elevation estimate and no survey:

  1. label every scene rising or falling, from the tide an hour earlier
  2. at MATCHED water levels, measure how much wetter a pixel is on the
     falling limb than on the rising one — call that the ponding index
  3. ask whether the pixels with a high index are where HSR disagrees with
     the RTK survey

Step 3 is the payoff. If ponded pixels are ordinary, connectivity buys
nothing here and should be dropped. If they carry the errors, then a physical
quality flag falls out — and unlike everything else tried today it costs one
pass over data already in memory.

The survey is read once, at the end, and only to score.
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
N_LEVEL_BINS = 10
MIN_PER_LIMB = 3
NDWI_WET = 0.0


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])

    cache = os.path.join(SC, "mareas_villa.npz")
    tide = np.load(cache)["tide"]

    # rising or falling: the tide an hour before the overpass
    prev_cache = os.path.join(SC, "mareas_villa_prev.npz")
    if os.path.exists(prev_cache):
        tide_prev = np.load(prev_cache)["tide_prev"]
    else:
        use_system_certificates()
        from eo_tides.model import model_tides
        aoi = pit.sites.get("villaviciosa")
        lat_c, lon_c = aoi.centroid
        t_prev = pd.to_datetime([f"{s} 10:00:00" for s in dates])
        tide_prev = model_tides(
            x=[lon_c], y=[lat_c], time=t_prev, model="EOT20",
            directory="tide_models", crs="EPSG:4326", extrapolate=True,
            cutoff=np.inf, parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)
        np.savez_compressed(prev_cache, tide_prev=tide_prev)

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide) & np.isfinite(tide_prev)
    Y, C, tide, tide_prev = Y[ep], C[ep], tide[ep], tide_prev[ep]
    rising = tide > tide_prev
    print(f"{Y.shape[0]} escenas · {int(rising.sum())} con marea subiendo, "
          f"{int((~rising).sum())} bajando")

    wet = (C > 0) & (Y > NDWI_WET)
    obs = C > 0

    # ── ponding index, at matched water levels ───────────────────────────
    edges = np.quantile(tide, np.linspace(0, 1, N_LEVEL_BINS + 1))
    edges[0] -= 1e-9
    num = np.zeros(Y.shape[1])
    den = np.zeros(Y.shape[1])
    used = 0
    for i in range(N_LEVEL_BINS):
        inb = (tide > edges[i]) & (tide <= edges[i + 1])
        up, dn = inb & rising, inb & ~rising
        if up.sum() < MIN_PER_LIMB or dn.sum() < MIN_PER_LIMB:
            continue
        used += 1
        # within one narrow level bin the two limbs see the same water height,
        # so any difference is not a level effect
        f_up = wet[up].sum(0) / np.maximum(obs[up].sum(0), 1)
        f_dn = wet[dn].sum(0) / np.maximum(obs[dn].sum(0), 1)
        ok = (obs[up].sum(0) >= MIN_PER_LIMB) & (obs[dn].sum(0) >= MIN_PER_LIMB)
        num += np.where(ok, f_dn - f_up, 0.0)
        den += ok
    pond = np.where(den >= 3, num / np.maximum(den, 1), np.nan)
    print(f"{used} de {N_LEVEL_BINS} bins de nivel con ambas ramas · "
          f"{int(np.isfinite(pond).sum()):,} px con indice")
    print(f"indice de encharcamiento: mediana {np.nanmedian(pond):+.3f}, "
          f"p90 {np.nanpercentile(pond, 90):+.3f}, "
          f"p99 {np.nanpercentile(pond, 99):+.3f}")
    print("  (positivo = mas mojado bajando que subiendo = se queda charco)")

    # ── is the asymmetry real, or does the split manufacture it? ─────────
    rngs = np.random.default_rng(4)
    fakes = []
    for k in range(5):
        fake = rngs.permutation(rising)
        n2 = np.zeros(Y.shape[1]); d2 = np.zeros(Y.shape[1])
        for i in range(N_LEVEL_BINS):
            inb = (tide > edges[i]) & (tide <= edges[i + 1])
            up, dn = inb & fake, inb & ~fake
            if up.sum() < MIN_PER_LIMB or dn.sum() < MIN_PER_LIMB:
                continue
            f_up = wet[up].sum(0) / np.maximum(obs[up].sum(0), 1)
            f_dn = wet[dn].sum(0) / np.maximum(obs[dn].sum(0), 1)
            ok = ((obs[up].sum(0) >= MIN_PER_LIMB)
                  & (obs[dn].sum(0) >= MIN_PER_LIMB))
            n2 += np.where(ok, f_dn - f_up, 0.0); d2 += ok
        fakes.append(np.nanpercentile(
            np.where(d2 >= 3, n2 / np.maximum(d2, 1), np.nan), 99))
    print(f"nulo (subiendo/bajando barajado), p99: "
          f"{', '.join(f'{v:+.3f}' for v in fakes)}")

    # ── does it predict where HSR fails? the survey, read once ───────────
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]
    p_field = pond[fi]
    ok = np.isfinite(hsr) & np.isfinite(y) & np.isfinite(p_field)
    err = np.abs(y[ok] - hsr[ok] - np.median(y[ok] - hsr[ok]))
    p_ok = p_field[ok]
    print(f"\n{int(ok.sum())} puntos de campo con indice")
    r = stats.spearmanr(p_ok, err)
    print(f"Spearman(indice, |error de HSR|) = {r.statistic:+.3f}  "
          f"p = {r.pvalue:.3g}")

    hi = p_ok >= np.percentile(p_ok, 75)
    print(f"\n{'grupo':22s} {'n':>4s} {'|error| mediano':>16s} "
          f"{'RMSE':>7s}")
    print("-" * 54)
    for lab, m in (("encharcados (top 25%)", hi), ("resto", ~hi)):
        print(f"{lab:22s} {int(m.sum()):4d} {np.median(err[m]):16.3f} "
              f"{np.sqrt(np.mean(err[m]**2)):7.3f}")
    u = stats.mannwhitneyu(err[hi], err[~hi], alternative="greater")
    print(f"\nMann-Whitney (encharcados peores): p = {u.pvalue:.3g}")

    json.dump({"spearman": float(r.statistic), "p": float(r.pvalue),
               "mwu_p": float(u.pvalue),
               "pond_p99": float(np.nanpercentile(pond, 99)),
               "null_p99": [float(v) for v in fakes]},
              open(os.path.join(SC, "charcos.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(SC, "charcos.npz"), pond=pond, keep=keep)


if __name__ == "__main__":
    main()
