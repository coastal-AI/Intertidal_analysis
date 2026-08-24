"""Did the flat move? The confound nobody checked: two years between the
imagery and the survey.

The RTK campaign was walked on 14 August 2026. Every elevation it is compared
against was fitted from imagery spanning 2023 to 2025 — a mean gap of 2.1
years. Intertidal flats are not static over two years. They accrete where the
energy is low and scour where it is high, and both processes are ELEVATION
DEPENDENT, which is precisely the shape needed to produce a slope below one
without anything being wrong with the estimator at all.

That would explain what nothing else has. The compression is physical (the
null returns 1.019 and never reaches the observed 0.561), it is spatially
structured, and it is not predicted by channel distance, terrain slope,
roughness, sigma, amplitude, observation count, water frequency or wet
fraction. Morphological change is the one candidate consistent with all of
that: it is real ground movement, it varies from place to place with the local
sediment regime, and none of those covariates describes a sediment regime.

The prediction is sharp and the archive can test it today. If the flat is
drifting, then the OLDER the imagery, the worse the disagreement with a survey
walked in 2026 — monotonically. If the compression instead comes from the
estimator or the optics, the epoch should not matter.

Design, with the obvious confound controlled: the ten-year archive is cut into
three consecutive epochs with the SAME NUMBER OF SCENES, not the same number
of years. Scene count drives how well a sigmoid is resolved, so equal counts
are what make the three fits comparable; equal spans would not.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
from scipy import stats

from pyintertidal.elevation import _fit_block

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
N_EPOCHS = 3
SURVEY = pd.Timestamp("2026-08-14")
N_BOOT = 3000


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
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    tide_all = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]

    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y = fi[have], gnss[have]
    Yf = np.nan_to_num(Y[:, fi], nan=0.0).astype(np.float64)
    Cf = C[:, fi].astype(np.float64)

    ts = pd.to_datetime([str(s) for s in dates])
    fin = np.isfinite(tide_all)
    order = np.argsort(ts.values)
    order = order[fin[order]]
    chunks = np.array_split(order, N_EPOCHS)

    print(f"campana RTK: {SURVEY.date()}")
    print(f"archivo: {len(order)} escenas utiles de "
          f"{ts[order].min().date()} a {ts[order].max().date()}")
    print(f"cortado en {N_EPOCHS} epocas de IGUAL numero de escenas\n")

    print(f"{'epoca':22s} {'escenas':>8s} {'desfase':>9s} {'pend':>7s} "
          f"{'RMSE':>7s} {'n':>5s}")
    print("-" * 64)
    rows = []
    for k, idx in enumerate(chunks):
        t_mid = ts[idx].mean()
        gap = (SURVEY - t_mid).days / 365.25
        v = fit(Yf[idx], Cf[idx], tide_all[idx])
        m = np.isfinite(v) & np.isfinite(y)
        sl = float(np.polyfit(y[m], v[m], 1)[0])
        r = y[m] - v[m]
        rms = float(np.sqrt(np.mean((r - np.median(r)) ** 2)))
        lab = f"{ts[idx].min().date()}..{ts[idx].max().date()}"
        print(f"{lab:22s} {len(idx):8d} {gap:8.2f}a {sl:7.3f} {rms:7.3f} "
              f"{int(m.sum()):5d}")
        rows.append({"label": lab, "n_scenes": int(len(idx)),
                     "gap_years": float(gap), "slope": sl, "rmse": rms,
                     "n": int(m.sum()),
                     "tide_span": float(tide_all[idx].max()
                                        - tide_all[idx].min()),
                     "_v": v, "_m": m})

    print("\nla marea cubierta por cada epoca, que tambien podria explicar")
    print("diferencias, y por eso se mira:")
    for r in rows:
        print(f"  {r['label']:22s} rango de marea {r['tide_span']:.2f} m")

    gaps = np.array([r["gap_years"] for r in rows])
    sls = np.array([r["slope"] for r in rows])
    print(f"\nPREDICCION: si el fango se mueve, la pendiente EMPEORA con el "
          f"desfase")
    print(f"  desfases {np.round(gaps,2)} anos")
    print(f"  pendientes {np.round(sls,3)}")
    trend = np.polyfit(gaps, sls, 1)[0]
    print(f"  tendencia {trend:+.4f} de pendiente por ano de desfase")
    print(f"  (NEGATIVA = las imagenes viejas concuerdan peor, como predice)")

    # bootstrap over survey points, keeping the epochs paired
    rng = np.random.default_rng(5)
    common = rows[0]["_m"] & rows[1]["_m"] & rows[2]["_m"]
    idxs = np.where(common)[0]
    print(f"\nbootstrap emparejado sobre {len(idxs)} puntos comunes")
    tr_bs, sl_bs = [], []
    for _ in range(N_BOOT):
        i = rng.choice(idxs, len(idxs), replace=True)
        s3 = [np.polyfit(y[i], r["_v"][i], 1)[0] for r in rows]
        tr_bs.append(np.polyfit(gaps, s3, 1)[0])
        sl_bs.append(s3)
    tr_bs = np.array(tr_bs)
    sl_bs = np.array(sl_bs)
    ci = np.percentile(tr_bs, [2.5, 97.5])
    print(f"  tendencia {tr_bs.mean():+.4f} por ano  "
          f"IC 95% [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    for k, r in enumerate(rows):
        c = np.percentile(sl_bs[:, k], [2.5, 97.5])
        print(f"  {r['label']:22s} pendiente {sl_bs[:,k].mean():.3f} "
              f"[{c[0]:.3f}, {c[1]:.3f}]")

    print()
    if ci[1] < 0:
        rate = -tr_bs.mean()
        print("EL DESFASE IMPORTA: las imagenes mas viejas concuerdan peor con")
        print("el terreno de 2026, y el intervalo no cruza cero. La compresion")
        print("es en parte CAMBIO MORFOLOGICO, no error del metodo — y eso")
        print(f"implica que con {gaps.min():.1f} anos de desfase todavia queda")
        print(f"{rate*gaps.min():.3f} de pendiente perdida por esta via.")
    elif ci[0] > 0:
        print("AL REVES DE LO PREDICHO: las imagenes viejas concuerdan MEJOR.")
        print("Eso no lo explica el movimiento del fango.")
    else:
        print("EL DESFASE NO EXPLICA NADA: el intervalo cruza cero. La")
        print("compresion no viene de que la ria haya cambiado entre las")
        print("imagenes y la campana.")

    json.dump({"survey": str(SURVEY.date()),
               "epochs": [{k: v for k, v in r.items()
                           if not k.startswith("_")} for r in rows],
               "trend_per_year": float(tr_bs.mean()),
               "trend_ci": ci.tolist()},
              open(os.path.join(SC, "deriva.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
