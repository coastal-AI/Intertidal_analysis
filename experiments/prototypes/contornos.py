"""Where does the operator's accuracy actually come from — the transfer, or
the boundary?

The transfer operator improved on a global model by factors of two to four at
several stations, and it is worth knowing which of its parts did that, because
the answer decides whether an ensemble of ocean models would help.

Four boundaries, one transfer, the same stations:

  EOT20 solo            the global model at the point. What the pipeline does.
  ensemble de 3         EOT20 + GOT4.10 + GOT4.8 averaged. More models.
  operador sobre modelo the transfer, fed a MODEL as boundary.
  operador sobre marea  the transfer, fed a GAUGE as boundary. This is v4.

The distinction that matters is the last one. The operator's residual term
carries the boundary's non-tidal signal inland — storm surge, above all —
scaled by a coefficient measured between the two stations. A tide gauge
records surge. A global ocean model does not contain it at all: it predicts
the astronomical tide, so its residual is zero by construction, and averaging
five such models leaves it zero.

If that reading is right, the ensemble will land on top of the single models,
the operator fed a model will gain only what the tidal reshaping is worth, and
only the operator fed a gauge will show the large improvement. If it is wrong
— if the ensemble closes most of the gap — then the gain was in the tide after
all and a better ocean product would be the cheaper route.
"""
import glob
import json
import os
import sys

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
use_system_certificates()

from pyintertidal.boundary import (GaugeBoundary, PyTMDBoundary,
                                   EnsembleBoundary)
from pyintertidal.estuary import EstuaryTransfer

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MODELS = ["EOT20", "GOT4.10_nc", "GOT4.8_nc"]
CONST = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "M4", "MS4", "M6")

# outer, inner, km, (lat, lon) of the inner station
PAIRS = [
    ("Keehi Hbr", "hono", "keehi", 2.2, 21.318, -157.885),
    ("Danshuei", "tdan", "tdas", 2.0, 25.168, 121.428),
    ("Goteborg Hisingsbron", "goer", "gohi", 4.1, 57.712, 11.966),
    ("Vlissingen", "brsk", "vlis", 5.8, 51.443, 3.597),
    ("Port de Barcelona", "barc", "barc2", 2.9, 41.363, 2.187),
    ("Ferrol 2", "fer1", "fer2", 6.4, 43.476, -8.249),
]


def load(code):
    raw = []
    for h in sorted(glob.glob(os.path.join(SC, f"ioc_{code}_*.json"))):
        raw += json.load(open(h))
    if not raw:
        return None
    df = pd.DataFrame(raw)
    if "sensor" in df:
        df = df[df["sensor"] == df["sensor"].value_counts().idxmax()]
    out = (pd.DataFrame({"time": pd.to_datetime(df["stime"]),
                         "level_m": pd.to_numeric(df["slevel"],
                                                  errors="coerce")})
           .dropna().sort_values("time").drop_duplicates("time"))
    return out if len(out) > 500 else None


def rmse(pred, truth):
    m = np.isfinite(pred) & np.isfinite(truth)
    if m.sum() < 50:
        return np.nan
    d = pred[m] - truth[m]
    return float(np.sqrt(np.mean((d - np.median(d)) ** 2)))


def main():
    print(f"{'estacion':22s} {'EOT20':>8s} {'ensemble':>9s} "
          f"{'op+modelo':>10s} {'op+ensemble':>11s} {'op+marea':>9s}   "
          f"{'g_resid':>7s}")
    print("-" * 88)
    rows = []
    for label, oc, ic, km, lat, lon in PAIRS:
        o, i = load(oc), load(ic)
        if o is None or i is None:
            print(f"{label:22s} sin datos")
            continue
        lo = max(o.time.min(), i.time.min())
        hi = min(o.time.max(), i.time.max())
        split = lo + (hi - lo) * 0.65
        t = pd.date_range(split, hi, freq="1h")

        bo = GaugeBoundary(o.time, o.level_m)
        bi = GaugeBoundary(i.time, i.level_m)
        truth = bi.levels(t)

        T = EstuaryTransfer.from_gauges(
            GaugeBoundary(o.time[o.time < split], o.level_m[o.time < split]),
            GaugeBoundary(i.time[i.time < split], i.level_m[i.time < split]),
            km, constituents=CONST)

        # boundaries made of models, placed at the OUTER station
        olat, olon = None, None
        for lb, occ, icc, k, la, ln in PAIRS:
            if occ == oc and icc == ic:
                pass
        # the outer coordinates: take them from the station table used earlier
        OUTER = {"hono": (21.307, -157.867), "tdan": (25.183, 121.410),
                 "goer": (57.700, 11.900), "brsk": (51.401, 3.548),
                 "barc": (41.342, 2.166), "fer1": (43.463, -8.326)}
        olat, olon = OUTER[oc]

        mods = {}
        for m in MODELS:
            try:
                mods[m] = PyTMDBoundary(m, lat, lon)   # at the INNER point
            except Exception:
                pass
        r_single = rmse(mods["EOT20"].levels(t), truth) if "EOT20" in mods \
            else np.nan
        try:
            ens = EnsembleBoundary([PyTMDBoundary(m, lat, lon)
                                    for m in MODELS])
            r_ens = rmse(ens.levels(t), truth)
        except Exception:
            r_ens = np.nan

        # the transfer, fed a MODEL at the outer point (no surge to carry)
        try:
            bmod = PyTMDBoundary("EOT20", olat, olon)
            r_opm = rmse(T.levels(bmod, km, t, constituents=CONST,
                                  keep_residual=True), truth)
        except Exception:
            r_opm = np.nan
        # the transfer, fed the ENSEMBLE at the outer point
        try:
            bens = EnsembleBoundary([PyTMDBoundary(m, olat, olon)
                                     for m in MODELS])
            r_ope = rmse(T.levels(bens, km, t, constituents=CONST,
                                  keep_residual=True), truth)
        except Exception:
            r_ope = np.nan
        # the transfer, fed the GAUGE at the outer point — v4
        r_opg = rmse(T.levels(bo, km, t, constituents=CONST,
                              keep_residual=True), truth)

        print(f"{label:22s} {r_single:8.4f} {r_ens:9.4f} {r_opm:10.4f} "
              f"{r_ope:11.4f} {r_opg:9.4f}   {T.residual_gain:7.3f}")
        rows.append(dict(label=label, km=km, eot20=r_single, ensemble=r_ens,
                         op_modelo=r_opm, op_ensemble=r_ope, op_marea=r_opg,
                         g_resid=T.residual_gain))

    print("\nRMSE en metros contra el mareografo interior, ventana no vista.")
    if rows:
        A = {k: np.array([r[k] for r in rows], dtype=float)
             for k in ("eot20", "ensemble", "op_modelo", "op_ensemble",
                       "op_marea")}
        print(f"\n{'':22s} {'mediana':>9s}")
        for k, v in A.items():
            v = v[np.isfinite(v)]
            if len(v):
                print(f"{k:22s} {np.median(v):9.4f}")
        d_ens = np.nanmedian(A["eot20"] - A["ensemble"])
        d_opm = np.nanmedian(A["eot20"] - A["op_modelo"])
        d_opg = np.nanmedian(A["eot20"] - A["op_marea"])
        print(f"\nmejora sobre EOT20 solo:")
        print(f"  juntar 3 modelos en un ensemble   {d_ens:+.4f} m")
        print(f"  aplicar la transferencia a un modelo {d_opm:+.4f} m")
        print(f"  aplicar la transferencia al ENSEMBLE  "
              f"{np.nanmedian(A['eot20'] - A['op_ensemble']):+.4f} m")
        print(f"  aplicar la transferencia a un mareografo {d_opg:+.4f} m")
        print()
        if d_opg > 3 * max(abs(d_ens), 1e-9):
            print("LA GANANCIA ESTA EN EL MAREOGRAFO, no en el modelo. Un")
            print("ensemble de cinco modelos no traeria marejada, porque")
            print("ninguno la contiene: promediar ceros da cero.")
    json.dump(rows, open(os.path.join(SC, "contornos.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
