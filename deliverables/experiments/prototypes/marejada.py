"""The water level is not the tide. Recover the missing part from the weather.

Every published method for intertidal topography feeds its fit an ASTRONOMICAL
tide: EOT20, FES, GOT, an eo-tides ensemble. The astronomical tide is what the
moon and sun alone would produce. The sea also responds to the weather — it
rises when the air presses on it less, and a deep Atlantic low over the Bay of
Biscay can lift it half a metre above prediction.

That gap matters more than it looks, because of what was measured today. The
relief compresses because pixels near the top of the archive's water range
have nothing above them to pin down their transition: the survey shows the
underestimate growing by 0.189 m for every metre of missing headroom, at
t = -5.3, and it survives controlling for how often each pixel was seen. Going
from three years of archive to ten does NOT help, because the astronomical
maximum is fixed and three years already reaches it: +1.32 m against +1.33 m.

Surge is the one thing that can genuinely raise the ceiling. And it does two
separate jobs:

  1. it WIDENS the window — the scenes taken during storms saw water higher
     than any astronomical prediction, and those are exactly the scenes that
     carry information about the high pixels
  2. it FIXES MISLABELLED SCENES — a scene shot during a surge is currently
     filed under its astronomical level, so its wet pixels are attributed to a
     water level lower than the one that actually wetted them. That is not
     just wasted information, it is wrong information, and it biases every
     pixel in the scene.

The surge is approximated here by the inverse barometer, the standard
first-order relation: the sea rises about 1 cm for every hectopascal the mean
sea-level pressure falls below its long-run average. It ignores wind setup,
which in a shallow ria is real and would only add to the effect, so this is a
conservative estimate of what a proper water level would give.

The control that decides it: the same surge values SHUFFLED across dates. That
keeps the distribution, the magnitude and the widened window, and destroys
only the pairing between a scene and its own weather. If shuffled surge helps
as much as real surge, the gain is an artefact of adding spread to the tide
axis rather than of getting the water level right.
"""
import os
import sys
import json
import time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
PIX_CHUNK = 4000
CM_PER_HPA = 1.0          # inverse barometer
OVERPASS_HOUR = 11


def fetch_pressure(lat, lon, start, end):
    """Hourly mean sea-level pressure from the Open-Meteo archive.

    Public, key-free, read-only. Cached to disk so this runs once.
    """
    import urllib.request
    cache = os.path.join(SC, f"presion_{start}_{end}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat:.4f}&longitude={lon:.4f}"
           f"&start_date={start}&end_date={end}"
           "&hourly=pressure_msl&timezone=GMT")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = json.load(r)
    json.dump(data, open(cache, "w"))
    return data


def fit(Y, C, level):
    grid = np.linspace(level.min(), level.max(), MU_POINTS)
    P = Y.shape[1]
    z = np.full(P, np.nan); ok = np.zeros(P, bool)
    for j in range(0, P, PIX_CHUNK):
        s = slice(j, j + PIX_CHUNK)
        a, b, mu, sg, _, N = _fit_block(Y[:, s], C[:, s], level, grid, SG_GRID)
        z[s] = mu
        ok[s] = (N >= MIN_OBS) & (b > 0.15)
    return np.where(ok, z, np.nan)


def score(y, v, label, top):
    m = np.isfinite(y) & np.isfinite(v)
    r = y[m] - v[m]
    off = np.median(r)
    sl = float(np.polyfit(y[m], v[m], 1)[0])
    rms = float(np.sqrt(np.mean((r - off) ** 2)))
    print(f"{label:32s} {sl:7.3f} {rms:7.3f} {int(m.sum()):5d} {top:8.2f}")
    return {"slope": sl, "rmse": rms, "n": int(m.sum()), "top": float(top)}


def main():
    use_system_certificates()
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    tide_all = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    ds = pd.to_datetime(pd.Series([str(s) for s in dates]))
    start, end = ds.min().strftime("%Y-%m-%d"), ds.max().strftime("%Y-%m-%d")
    print(f"presion en {lat_c:.3f}, {lon_c:.3f} de {start} a {end}", flush=True)
    raw = fetch_pressure(lat_c, lon_c, start, end)

    ts = pd.to_datetime(raw["hourly"]["time"])
    pmsl = pd.Series(np.asarray(raw["hourly"]["pressure_msl"], dtype=float),
                     index=ts)
    p_ref = float(np.nanmean(pmsl))
    want = pd.to_datetime([f"{s} {OVERPASS_HOUR:02d}:00:00" for s in dates])
    p_at = pmsl.reindex(want).to_numpy()
    surge = -(p_at - p_ref) * CM_PER_HPA / 100.0
    print(f"presion media {p_ref:.1f} hPa · marejada estimada "
          f"{np.nanmin(surge):+.2f} a {np.nanmax(surge):+.2f} m "
          f"(sd {np.nanstd(surge):.3f})")

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide_all) & np.isfinite(surge)
    tide, surge = tide_all[ep], surge[ep]
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y = fi[have], gnss[have]
    Yf = np.nan_to_num(Y[ep][:, fi], nan=0.0).astype(np.float64)
    Cf = C[ep][:, fi].astype(np.float64)
    print(f"{int(ep.sum())} escenas · {len(fi)} px de campo\n")

    level = tide + surge
    print(f"techo de la ventana: {tide.max():+.3f} m solo marea  ->  "
          f"{level.max():+.3f} m con marejada  "
          f"({level.max()-tide.max():+.3f} m)")
    print(f"escenas cuyo nivel supera la marea maxima astronomica: "
          f"{int((level > tide.max()).sum())}\n")

    print(f"{'nivel usado':32s} {'pend':>7s} {'RMSE':>7s} {'n':>5s} "
          f"{'techo':>8s}")
    print("-" * 66)
    out = {}
    out["solo marea"] = score(y, fit(Yf, Cf, tide), "solo marea (EOT20)",
                              tide.max())
    out["marea + marejada"] = score(y, fit(Yf, Cf, level),
                                    "marea + marejada", level.max())

    # ── the control: same surge, wrong dates ─────────────────────────────
    print("\ncontrol — la misma marejada asignada a fechas equivocadas")
    print("-" * 66)
    sh = []
    for k in range(4):
        r = np.random.default_rng(30 + k)
        lv = tide + r.permutation(surge)
        s = score(y, fit(Yf, Cf, lv), f"  barajada {k+1}", lv.max())
        sh.append(s)
    out["barajada"] = sh

    real = out["marea + marejada"]
    base = out["solo marea"]
    sl_sh = np.array([s["slope"] for s in sh])
    rm_sh = np.array([s["rmse"] for s in sh])
    print(f"\n{'-'*66}")
    print(f"pendiente  base {base['slope']:.3f} · real {real['slope']:.3f} · "
          f"barajada {sl_sh.mean():.3f} (max {sl_sh.max():.3f})")
    print(f"RMSE       base {base['rmse']:.3f} · real {real['rmse']:.3f} · "
          f"barajada {rm_sh.mean():.3f} (min {rm_sh.min():.3f})")
    better = (real["slope"] > sl_sh.max()) and (real["rmse"] < rm_sh.min())
    print()
    if better:
        print("LA MAREJADA APORTA: y no por ensanchar el eje, porque la misma")
        print("marejada en fechas equivocadas no consigue lo mismo.")
    else:
        print("NO APORTA MAS QUE BARAJADA: lo que ayuda es el ensanchamiento")
        print("del eje, no acertar con el nivel de cada escena.")

    json.dump(out, open(os.path.join(SC, "marejada.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
