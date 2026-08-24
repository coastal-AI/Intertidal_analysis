"""Does an estuarine transfer matter at the distances we actually map?

The Guadalquivir showed the operator works: 85 km up a tidal river it cut the
error by 76 %. But 85 km up a tidal river is not what this project maps.
Villaviciosa, Santona, Foz and Urdaibai are short coastal rias, a few
kilometres from the open sea, and there the wave has had almost no time to be
reshaped. Gains will sit near one and lags will be minutes rather than hours.

So the honest question is not "does the operator work" — it does — but "is
there anything left to correct at three to six kilometres". If M2 gains 1 % and
arrives three minutes late, the correction is a centimetre and plugging it
into the pipeline buys nothing. If it gains 5 % and arrives fifteen minutes
late, it matters.

Two pairs answer it, and the second is what makes the first believable:

* FERROL — two stations about six kilometres apart inside a short Atlantic
  ria. This is the regime the project actually works in.
* A CORUNA and LANGOSTEIRA — two stations on the OPEN COAST, eleven
  kilometres apart with no estuary between them. There is nothing to
  propagate, so the method must report a gain of one and a lag of zero.
  Whatever it reports instead is what it manufactures from noise, and the
  Ferrol numbers have to beat it.

Without that control a small gain at Ferrol would be uninterpretable: every
estimator returns something, and the only way to know whether a small number
means anything is to measure the same thing where the answer is known.
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
CONST = ("M2", "S2", "N2", "K1", "O1", "M4", "MS4")
DAYS = 95

PAIRS = [
    {"label": "Ferrol (ria corta, ~6 km)", "outer": "fer1", "inner": "fer2",
     "km": 6.4, "kind": "estuario"},
    {"label": "A Coruna - Langosteira (costa abierta, CONTROL)",
     "outer": "lang", "inner": "acor1", "km": 11.0, "kind": "control"},
]


def fetch_window(code, start, stop):
    url = (f"{BASE}?query=data&code={code}"
           f"&timestart={start}&timestop={stop}&format=json")
    req = urllib.request.Request(url, headers={"User-Agent": "pyintertidal"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def load(code, t0, t1):
    cache = os.path.join(SC, f"ioc_{code}_{t0}_{t1}.json")
    if os.path.exists(cache):
        raw = json.load(open(cache))
    else:
        raw = []
        edges = list(pd.date_range(t0, t1, freq="30D")) + [pd.Timestamp(t1)]
        for a, b in zip(edges, edges[1:]):
            try:
                raw += fetch_window(code, a.strftime("%Y-%m-%dT%H:%M:%S"),
                                    b.strftime("%Y-%m-%dT%H:%M:%S"))
            except Exception as e:
                print(f"    {a.date()}: {type(e).__name__}")
        json.dump(raw, open(cache, "w"))
    if not raw:
        return None
    df = pd.DataFrame(raw)
    if "stime" not in df or "slevel" not in df:
        return None
    if "sensor" in df:
        df = df[df["sensor"] == df["sensor"].value_counts().idxmax()]
    return (pd.DataFrame({"time": pd.to_datetime(df["stime"]),
                          "level_m": pd.to_numeric(df["slevel"],
                                                   errors="coerce")})
            .dropna().sort_values("time").drop_duplicates("time"))


def run(pair, t0, t1):
    print(f"\n{'='*66}\n{pair['label']}\n{'='*66}")
    recs = {}
    for role in ("outer", "inner"):
        code = pair[role]
        df = load(code, t0, t1)
        if df is None or len(df) < 500:
            print(f"  {code}: sin datos utiles")
            return None
        print(f"  {code}: {len(df):,} registros, "
              f"{df['time'].min().date()} a {df['time'].max().date()}")
        recs[role] = df

    lo = max(r["time"].min() for r in recs.values())
    hi = min(r["time"].max() for r in recs.values())
    if (hi - lo).days < 30:
        print(f"  solape de solo {(hi-lo).days} dias: insuficiente")
        return None
    split = lo + (hi - lo) * 0.65

    def B(role, a, b):
        d = recs[role]
        m = (d["time"] >= a) & (d["time"] < b)
        return GaugeBoundary(d["time"][m], d["level_m"][m], name=pair[role])

    tr = transfer_from_gauges(B("outer", lo, split), B("inner", lo, split),
                             CONST)
    print(f"\n  {'const':6s} {'ganancia':>9s} {'±':>7s} {'retardo (min)':>14s}")
    print(f"  {'-'*40}")
    for k in CONST:
        if k in tr:
            v = tr[k]
            print(f"  {k:6s} {v['gain']:9.4f} {v['gain_sigma']:7.4f} "
                  f"{v['lag_hours']*60:14.1f}")

    T = EstuaryTransfer.from_gauges(B("outer", lo, split),
                                    B("inner", lo, split),
                                    pair["km"], constituents=CONST)
    v = validate_against_gauge(T, B("outer", split, hi), B("inner", split, hi),
                               pair["km"])
    print(f"\n  prueba en {(hi-split).days} dias no vistos:")
    for k in ("sin_operador", "solo_armonicos", "con_operador"):
        r = v.get(k)
        if r:
            print(f"    {k:16s} RMSE {r['rmse_m']:.4f} m · "
                  f"varianza explicada {r['var_explained']:.4f}")
    if "mejora_del_operador_m" in v:
        print(f"    mejora atribuible al OPERADOR (contra solo armonicos): "
              f"{v['mejora_del_operador_m']:+.4f} m")
        print(f"    mejora contra la serie cruda (incluye filtrado): "
              f"{v.get('mejora_sobre_crudo_m', float('nan')):+.4f} m")
    return {"pair": pair, "transfer": tr, "validation": v,
            "days": int((hi - lo).days)}


def main():
    t1 = pd.Timestamp("2025-12-31")
    t0 = t1 - pd.Timedelta(days=DAYS)
    out = [run(p, t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
           for p in PAIRS]
    out = [o for o in out if o]

    print(f"\n\n{'='*66}\nLECTURA CONJUNTA\n{'='*66}")
    est = [o for o in out if o["pair"]["kind"] == "estuario"]
    ctl = [o for o in out if o["pair"]["kind"] == "control"]
    if est and ctl:
        for k in ("M2", "S2", "M4"):
            ge = [o["transfer"][k]["gain"] for o in est
                  if k in o["transfer"]]
            gc = [o["transfer"][k]["gain"] for o in ctl
                  if k in o["transfer"]]
            le = [o["transfer"][k]["lag_hours"] * 60 for o in est
                  if k in o["transfer"]]
            lc = [o["transfer"][k]["lag_hours"] * 60 for o in ctl
                  if k in o["transfer"]]
            if ge and gc:
                print(f"  {k}: ganancia estuario {ge[0]:.4f} · "
                      f"control {gc[0]:.4f}   |   retardo estuario "
                      f"{le[0]:+.1f} min · control {lc[0]:+.1f} min")
        me = est[0]["validation"].get("mejora_del_operador_m")
        mc = ctl[0]["validation"].get("mejora_del_operador_m")
        print(f"\n  mejora del operador: estuario {me:+.4f} m · "
              f"control {mc:+.4f} m")
        print()
        if me is not None and mc is not None and me > 3 * abs(mc) and me > 0.02:
            print("  A DISTANCIAS COSTERAS SI HAY ALGO QUE CORREGIR, y el")
            print("  control confirma que no es un artefacto del metodo.")
        elif me is not None and me < 0.02:
            print("  A DISTANCIAS COSTERAS LA CORRECCION ES DESPRECIABLE:")
            print("  menos de 2 cm. Enchufarlo al pipeline no compensa la")
            print("  complejidad, y la marea de la boca vale tal cual.")
        else:
            print("  El estuario no se separa limpiamente del control: con")
            print("  estos datos no se puede afirmar que haya efecto.")

    json.dump([{k: v for k, v in o.items() if k != "pair"} | {
        "label": o["pair"]["label"]} for o in out],
        open(os.path.join(SC, "pares_costeros.json"), "w"), indent=1,
        default=float)


if __name__ == "__main__":
    main()
