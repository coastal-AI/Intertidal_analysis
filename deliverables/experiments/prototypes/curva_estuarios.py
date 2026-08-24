"""From what length does an estuary start reshaping the tide enough to matter?

The practical question behind every decision about this operator. Two points
exist: 85 km up the Guadalquivir the transfer cuts the error by 76 %, and
6.4 km into the ria de Ferrol it buys 8.8 mm. Everything between is guesswork,
and the sites this project maps sit at the small end.

Eight pairs of gauges, spanning 2.7 to 135 km, measured identically:

ESTUARIES
  Ferrol            6 km   short Atlantic ria — the regime we map
  Scheldt          28 km   Vlissingen to Terneuzen, the textbook amplifier
  Guadalquivir     85 km   Bonanza to Sevilla, a tidal river
  Delaware        135 km   Lewes to Philadelphia

OPEN COAST — the control, at matched distances
  Honolulu          3 km
  Barcelona         3 km
  Sables-Nazaire   90 km
  Le Havre-Dieppe  90 km

Nothing propagates between two open-coast stations, so those must return a
gain of one and a lag near zero. Whatever they return instead is what this
measurement manufactures, and the estuaries have to clear it. Having the
control at MATCHED distances matters: a method that drifts with separation
would otherwise be mistaken for physics.

Reported are the gain and lag of M2 themselves, not the derived mu and tau,
because those depend on a distance along the channel that is only estimated.
Gain and lag are measured directly and are what a user needs anyway: how much
bigger, and how much later.

The honest score comes from ``mejora_del_operador_m`` — the transfer's gain
over a harmonic reconstruction with no transfer at all. Comparing against the
raw series instead credits the operator with filtering out surge and seiches,
which at Ferrol was 96 % of the apparent improvement.
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
CONST = ("M2", "S2", "N2", "K1", "O1", "M4")
DAYS = 95

PAIRS = [
    dict(label="Honolulu - Keehi", outer="hono", inner="keehi", km=2.7,
         kind="costa abierta"),
    dict(label="Barcelona - Port de Barcelona", outer="barc", inner="barc2",
         km=2.9, kind="costa abierta"),
    dict(label="Ferrol (ria corta)", outer="fer1", inner="fer2", km=6.4,
         kind="estuario"),
    dict(label="Escalda: Vlissingen - Terneuzen", outer="vlis", inner="trnz",
         km=28.0, kind="estuario"),
    dict(label="Guadalquivir: Bonanza - Sevilla", outer="bon2", inner="sev2",
         km=85.0, kind="estuario"),
    dict(label="Saint-Nazaire - Le Crouesty", outer="lsol", inner="sain",
         km=90.0, kind="costa abierta"),
    dict(label="Le Havre - Dieppe", outer="leha", inner="diep", km=90.0,
         kind="costa abierta"),
    dict(label="Delaware: Lewes - Filadelfia", outer="lede", inner="phpa",
         km=135.0, kind="estuario"),
]


def fetch_window(code, a, b):
    url = (f"{BASE}?query=data&code={code}"
           f"&timestart={a}&timestop={b}&format=json")
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
            except Exception:
                pass
        json.dump(raw, open(cache, "w"))
    if not raw:
        return None
    df = pd.DataFrame(raw)
    if "stime" not in df or "slevel" not in df:
        return None
    if "sensor" in df:
        df = df[df["sensor"] == df["sensor"].value_counts().idxmax()]
    out = (pd.DataFrame({"time": pd.to_datetime(df["stime"]),
                         "level_m": pd.to_numeric(df["slevel"],
                                                  errors="coerce")})
           .dropna().sort_values("time").drop_duplicates("time"))
    return out if len(out) > 500 else None


def run(p, t0, t1):
    recs = {}
    for role in ("outer", "inner"):
        df = load(p[role], t0, t1)
        if df is None:
            print(f"  {p['label']:34s} sin datos en {p[role]}", flush=True)
            return None
        recs[role] = df
    lo = max(r["time"].min() for r in recs.values())
    hi = min(r["time"].max() for r in recs.values())
    if (hi - lo).days < 40:
        print(f"  {p['label']:34s} solape {(hi-lo).days} d, insuficiente",
              flush=True)
        return None
    split = lo + (hi - lo) * 0.65

    def B(role, a, b):
        d = recs[role]
        m = (d["time"] >= a) & (d["time"] < b)
        return GaugeBoundary(d["time"][m], d["level_m"][m], name=p[role])

    tr = transfer_from_gauges(B("outer", lo, split), B("inner", lo, split),
                             CONST)
    if not tr or "M2" not in tr:
        print(f"  {p['label']:34s} no se pudo descomponer", flush=True)
        return None
    T = EstuaryTransfer.from_gauges(B("outer", lo, split),
                                    B("inner", lo, split), p["km"],
                                    constituents=CONST)
    v = validate_against_gauge(T, B("outer", split, hi),
                               B("inner", split, hi), p["km"])
    m2 = tr["M2"]
    row = dict(label=p["label"], km=p["km"], kind=p["kind"],
               gain_M2=m2["gain"], gain_sigma=m2["gain_sigma"],
               lag_M2_min=m2["lag_hours"] * 60.0,
               gain_M4=tr.get("M4", {}).get("gain", np.nan),
               mejora=v.get("mejora_del_operador_m", np.nan),
               solo_arm=v.get("solo_armonicos", {}).get("rmse_m", np.nan),
               con_op=v.get("con_operador", {}).get("rmse_m", np.nan),
               dias=int((hi - lo).days))
    print(f"  {p['label']:34s} ganancia M2 {row['gain_M2']:.4f} · "
          f"retardo {row['lag_M2_min']:+6.1f} min · "
          f"aporta {row['mejora']:+.4f} m", flush=True)
    return row


def main():
    t1 = pd.Timestamp("2025-12-31")
    t0 = t1 - pd.Timedelta(days=DAYS)
    print(f"midiendo {len(PAIRS)} parejas, {DAYS} dias hasta {t1.date()}\n")
    rows = [r for r in (run(p, t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
                        for p in PAIRS) if r]

    rows.sort(key=lambda r: r["km"])
    print(f"\n\n{'='*88}")
    print(f"{'pareja':34s} {'km':>6s} {'tipo':13s} {'gan M2':>8s} "
          f"{'retardo':>9s} {'aporta':>9s}")
    print("-" * 88)
    for r in rows:
        print(f"{r['label'][:34]:34s} {r['km']:6.1f} {r['kind']:13s} "
              f"{r['gain_M2']:8.4f} {r['lag_M2_min']:+8.1f}m "
              f"{r['mejora']:+9.4f}")

    est = [r for r in rows if r["kind"] == "estuario"]
    ctl = [r for r in rows if r["kind"] == "costa abierta"]
    print(f"\n{'='*88}")
    if ctl:
        g = np.array([abs(r["gain_M2"] - 1) for r in ctl])
        l = np.array([abs(r["lag_M2_min"]) for r in ctl])
        m = np.array([r["mejora"] for r in ctl])
        print(f"CONTROL (costa abierta, {len(ctl)} parejas de "
              f"{min(r['km'] for r in ctl):.0f} a "
              f"{max(r['km'] for r in ctl):.0f} km):")
        print(f"  |ganancia - 1| hasta {g.max():.4f} · "
              f"|retardo| hasta {l.max():.1f} min · "
              f"aporta hasta {m.max():+.4f} m")
        print("  esto es lo que el metodo devuelve cuando NO hay estuario.")
    if est and ctl:
        thr = float(np.max([r["mejora"] for r in ctl]))
        over = [r for r in est if r["mejora"] > max(3 * thr, 0.02)]
        print(f"\nESTUARIOS que superan claramente al control:")
        for r in sorted(est, key=lambda x: x["km"]):
            flag = "SI" if r in over else "no"
            print(f"  {r['km']:6.1f} km  {r['label'][:34]:34s} "
                  f"aporta {r['mejora']:+.4f} m   {flag}")
        if over:
            k = min(r["km"] for r in over)
            print(f"\n  El efecto empieza a importar hacia los {k:.0f} km.")
            print(f"  Por debajo, la marea de la boca vale tal cual.")
        else:
            print("\n  Ninguno supera al control: con estos datos no se puede")
            print("  afirmar que la transferencia importe a ninguna distancia.")

    json.dump(rows, open(os.path.join(SC, "curva_estuarios.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
