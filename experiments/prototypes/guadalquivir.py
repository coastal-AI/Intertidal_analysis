"""The Guadalquivir: measure what an estuary does to a tide, then predict it.

Villaviciosa has no tide gauge, so the estuarine transfer operator built today
had nothing to be judged against. It does not need a field campaign: an
estuary with a gauge at the mouth and another far upstream examines the
operator on its own. Feed it the outer record, ask for the inner one, and
compare against a series it never saw.

The Guadalquivir is the strongest case on the peninsula. Bonanza sits at the
mouth, Sevilla some 85 km up a tidal river that is narrow, shallow and
dredged, so the wave arriving at Sevilla is hours late and reshaped — damped
in some constituents, and asymmetric because the overtides grow. If a transfer
operator cannot be measured here it cannot be measured anywhere.

Both records come from the IOC Sea Level Monitoring facility, open and without
registration.

The discipline that matters: the operator is fitted on one stretch of record
and tested on a DIFFERENT one. Fitting and scoring on the same weeks would
measure nothing but the harmonic fit's own flexibility, and today has already
shown twice what happens when that distinction is skipped.

The comparison that decides it is not against zero but against doing nothing:
what would the error be if we assumed, as every method in this package
currently does, that the tide inside equals the tide outside?
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.boundary import GaugeBoundary, transfer_from_gauges
from pyintertidal.estuary import EstuaryTransfer, validate_against_gauge

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
BASE = "http://www.ioc-sealevelmonitoring.org/service.php"
STATIONS = {"bon2": "Bonanza (boca)", "sev2": "Sevilla (85 km rio arriba)"}
CHANNEL_KM = 85.0          # along the river, not straight line
FIT_DAYS = 60
TEST_DAYS = 30
# constituents a 85 km tidal river actually shows; the diurnals are tiny here
CONST = ("M2", "S2", "N2", "K1", "O1", "M4", "MS4", "M6")


def fetch_window(code, start, stop):
    url = (f"{BASE}?query=data&code={code}"
           f"&timestart={start}&timestop={stop}&format=json")
    req = urllib.request.Request(url, headers={"User-Agent": "pyintertidal"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def load(code, t0, t1):
    """Download in monthly chunks — the service caps the span per request."""
    cache = os.path.join(SC, f"ioc_{code}_{t0}_{t1}.json")
    if os.path.exists(cache):
        raw = json.load(open(cache))
    else:
        raw = []
        for a, b in zip(pd.date_range(t0, t1, freq="30D"),
                        pd.date_range(t0, t1, freq="30D")[1:].append(
                            pd.DatetimeIndex([pd.Timestamp(t1)]))):
            try:
                raw += fetch_window(code, a.strftime("%Y-%m-%dT%H:%M:%S"),
                                    b.strftime("%Y-%m-%dT%H:%M:%S"))
            except Exception as e:
                print(f"    tramo {a.date()}: {type(e).__name__}")
        json.dump(raw, open(cache, "w"))
    if not raw:
        return None
    df = pd.DataFrame(raw)
    if "stime" not in df or "slevel" not in df:
        print(f"    respuesta inesperada, columnas {list(df.columns)[:8]}")
        return None
    # several sensors per station; keep the one with most data
    if "sensor" in df:
        best = df["sensor"].value_counts().idxmax()
        print(f"    sensores: {dict(df['sensor'].value_counts())} -> {best}")
        df = df[df["sensor"] == best]
    out = pd.DataFrame({
        "time": pd.to_datetime(df["stime"]),
        "level_m": pd.to_numeric(df["slevel"], errors="coerce"),
    }).dropna().sort_values("time").drop_duplicates("time")
    return out


def main():
    t1 = pd.Timestamp("2025-12-31")
    t0 = t1 - pd.Timedelta(days=FIT_DAYS + TEST_DAYS + 5)
    recs = {}
    for code, label in STATIONS.items():
        print(f"descargando {code} — {label}", flush=True)
        df = load(code, t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
        if df is None or len(df) < 500:
            print(f"    sin datos utiles")
            continue
        dt = np.median(np.diff(df["time"].values)) / np.timedelta64(1, "m")
        print(f"    {len(df):,} registros de {df['time'].min().date()} a "
              f"{df['time'].max().date()}, cada {dt:.0f} min")
        recs[code] = df
    if len(recs) < 2:
        print("\nno hay dos series: no se puede medir la transferencia")
        return

    lo = max(r["time"].min() for r in recs.values())
    hi = min(r["time"].max() for r in recs.values())
    print(f"\nsolape: {lo.date()} a {hi.date()} ({(hi-lo).days} dias)")
    split = lo + (hi - lo) * 0.65
    print(f"ajuste hasta {split.date()}, prueba a partir de ahi\n")

    def boundary(code, a, b):
        d = recs[code]
        m = (d["time"] >= a) & (d["time"] < b)
        return GaugeBoundary(d["time"][m], d["level_m"][m], name=code)

    mouth_fit = boundary("bon2", lo, split)
    inner_fit = boundary("sev2", lo, split)

    tr = transfer_from_gauges(mouth_fit, inner_fit, CONST)
    print("LO QUE EL ESTUARIO LE HACE A LA MAREA (medido, 85 km)")
    print(f"{'const':6s} {'ganancia':>9s} {'±':>7s} {'retardo (h)':>12s}")
    print("-" * 40)
    for k in CONST:
        if k in tr:
            v = tr[k]
            print(f"{k:6s} {v['gain']:9.3f} {v['gain_sigma']:7.3f} "
                  f"{v['lag_hours']:12.2f}")
    print("\nganancia < 1 = el rio amortigua ese constituyente;")
    print("que M4 y M6 ganen respecto a M2 es la asimetria de la onda.")

    T = EstuaryTransfer.from_gauges(mouth_fit, inner_fit, CHANNEL_KM,
                                    name="Guadalquivir", constituents=CONST)
    print(f"\n{T}")

    # ── the test: a stretch of record the operator never saw ─────────────
    mouth_te = boundary("bon2", split, hi)
    inner_te = boundary("sev2", split, hi)
    times = inner_te.times
    v = validate_against_gauge(T, mouth_te, inner_te, CHANNEL_KM, times)
    print(f"\nPRUEBA EN PERIODO RESERVADO ({split.date()} a {hi.date()})")
    print(f"{'':16s} {'RMSE (m)':>10s} {'varianza explicada':>20s} {'n':>7s}")
    print("-" * 58)
    for k in ("sin_operador", "con_operador"):
        r = v.get(k)
        if r:
            print(f"{k:16s} {r['rmse_m']:10.3f} {r['var_explained']:20.3f} "
                  f"{r['n']:7d}")
    if "mejora_rmse_m" in v:
        print(f"\nmejora {v['mejora_rmse_m']:+.3f} m")
        base = v["sin_operador"]["rmse_m"]
        print(f"reduccion del error: {100*v['mejora_rmse_m']/base:.0f} %")
        print()
        if v["mejora_rmse_m"] > 0.05:
            print("EL OPERADOR FUNCIONA sobre datos que no vio. Es la primera")
            print("validacion independiente de esta pieza.")
        else:
            print("El operador no mejora de forma util: revisar antes de usarlo.")

    json.dump({"transfer": tr, "channel_km": CHANNEL_KM,
               "validation": v,
               "fit_period": [str(lo.date()), str(split.date())],
               "test_period": [str(split.date()), str(hi.date())]},
              open(os.path.join(SC, "guadalquivir.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
