"""Is the estuarine transfer worth anything at the lengths we actually map?

One measurement said no — Ferrol, 6.4 km, 8.9 mm — but one point is not an
answer, and it is the point that decides whether this operator ever enters the
pipeline. So: every short pair of gauges worldwide that could plausibly sit in
an estuary, plus controls at matched distances, measured identically.

The estuaries were chosen because a river actually runs between the two
stations:

  Danhai - Danshuei      2.0 km   mouth of the Tamsui, Taiwan
  Goteborg (three pairs) 4-6 km   along the Gota alv, from the outer harbour
                                  to the city
  Mehuin - Queule        5.4 km   two Chilean river mouths
  Anping - Sicao         5.6 km   Tainan canal system
  Breskens - Vlissingen  5.8 km   the Scheldt mouth
  Ferrol                 6.4 km   short Atlantic ria — our own regime

The controls have open water between them and must return nothing:

  Honolulu - Keehi       2.7 km
  Barcelona - Port       2.9 km
  Cartagena - Murcia     2.9 km
  Algeciras - Gibraltar  4.4 km
  Zeebrugge platforms    4-6 km   North Sea measuring posts

Two numbers are reported per site, and the second is the one that matters:

  ``solo armonicos``   error when the boundary is rebuilt from constituents
                       and used as-is, which is what assuming "inside equals
                       outside" costs;
  ``con operador``     error once the transfer is applied.

Their difference is the operator's whole contribution. Scoring against the raw
series instead would credit it with filtering out surge and seiches: at Ferrol
that was 96 % of the apparent improvement, and it is exactly the mistake this
comparison exists to avoid.
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
    # --- short estuaries, a river between the two stations ---------------
    dict(l="Tamsui: Danhai - Danshuei", o="tdan", i="tdas", km=2.0, k="e"),
    dict(l="Gota alv: Eriksberg - Hisingsbron", o="goer", i="gohi", km=4.1,
         k="e"),
    dict(l="Gota alv: Torshamnen - Tangudden", o="goto", i="tang", km=4.8,
         k="e"),
    dict(l="Gota alv: Krossholmen - Tangudden", o="gokr", i="tang", km=6.1,
         k="e"),
    dict(l="Mehuin - Queule (Chile)", o="mehu", i="quel", km=5.4, k="e"),
    dict(l="Anping - Sicao (Taiwan)", o="tanp", i="tsic", km=5.6, k="e"),
    dict(l="Escalda: Breskens - Vlissingen", o="brsk", i="vlis", km=5.8,
         k="e"),
    dict(l="Ferrol", o="fer1", i="fer2", km=6.4, k="e"),
    # --- controls: open water between them --------------------------------
    dict(l="Honolulu - Keehi", o="hono", i="keehi", km=2.7, k="c"),
    dict(l="Barcelona - Port de Barcelona", o="barc", i="barc2", km=2.9,
         k="c"),
    dict(l="Cartagena - Murcia", o="carg", i="murc2", km=2.9, k="c"),
    dict(l="Algeciras - Gibraltar", o="alge", i="gibr", km=4.4, k="c"),
    dict(l="Blankenberge - Zeebrugge", o="blkw", i="zwdw", km=6.5, k="c"),
    dict(l="Goteborg: Torshamnen - Mavholmsbadan", o="goto", i="mavh",
         km=5.2, k="c"),
]
KIND = {"e": "estuario", "c": "control"}


def fetch(code, a, b):
    url = f"{BASE}?query=data&code={code}&timestart={a}&timestop={b}&format=json"
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
                raw += fetch(code, a.strftime("%Y-%m-%dT%H:%M:%S"),
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
    for role in ("o", "i"):
        df = load(p[role], t0, t1)
        if df is None:
            print(f"  {p['l']:38s} sin datos ({p[role]})", flush=True)
            return None
        recs[role] = df
    lo = max(r["time"].min() for r in recs.values())
    hi = min(r["time"].max() for r in recs.values())
    if (hi - lo).days < 40:
        print(f"  {p['l']:38s} solape {(hi-lo).days} d", flush=True)
        return None
    split = lo + (hi - lo) * 0.65

    def B(role, a, b):
        d = recs[role]
        m = (d["time"] >= a) & (d["time"] < b)
        return GaugeBoundary(d["time"][m], d["level_m"][m], name=p[role])

    tr = transfer_from_gauges(B("o", lo, split), B("i", lo, split), CONST)
    if not tr or "M2" not in tr:
        print(f"  {p['l']:38s} no descompone", flush=True)
        return None
    T = EstuaryTransfer.from_gauges(B("o", lo, split), B("i", lo, split),
                                    p["km"], constituents=CONST)
    v = validate_against_gauge(T, B("o", split, hi), B("i", split, hi),
                               p["km"])
    sa = v.get("solo_armonicos") or {}
    co = v.get("con_operador") or {}
    row = dict(label=p["l"], km=p["km"], kind=KIND[p["k"]],
               gain=tr["M2"]["gain"], sigma=tr["M2"]["gain_sigma"],
               lag_min=tr["M2"]["lag_hours"] * 60,
               err_sin=sa.get("rmse_m", np.nan),
               err_con=co.get("rmse_m", np.nan),
               aporta=v.get("mejora_del_operador_m", np.nan),
               amplitud=sa.get("sd_truth_m", np.nan),
               dias=int((hi - lo).days))
    print(f"  {p['l']:38s} gan {row['gain']:.4f} · "
          f"error {row['err_sin']:.4f} -> {row['err_con']:.4f} m", flush=True)
    return row


def main():
    t1 = pd.Timestamp("2025-12-31")
    t0 = t1 - pd.Timedelta(days=DAYS)
    print(f"midiendo {len(PAIRS)} parejas cortas\n")
    rows = [r for r in (run(p, t0.strftime("%Y-%m-%d"),
                            t1.strftime("%Y-%m-%d")) for p in PAIRS) if r]
    rows.sort(key=lambda r: (r["kind"], r["km"]))

    print(f"\n\n{'='*100}")
    print(f"{'sitio':38s} {'km':>5s} {'tipo':9s} {'gan M2':>7s} "
          f"{'retardo':>8s} {'ERROR sin':>10s} {'ERROR con':>10s} "
          f"{'aporta':>8s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['label'][:38]:38s} {r['km']:5.1f} {r['kind']:9s} "
              f"{r['gain']:7.4f} {r['lag_min']:+7.1f}m {r['err_sin']:10.4f} "
              f"{r['err_con']:10.4f} {r['aporta']:+8.4f}")

    est = [r for r in rows if r["kind"] == "estuario"]
    ctl = [r for r in rows if r["kind"] == "control"]
    print(f"\n{'='*100}")
    if ctl:
        a = np.array([r["aporta"] for r in ctl])
        print(f"CONTROLES ({len(ctl)}): aporta de {a.min():+.4f} a "
              f"{a.max():+.4f} m · mediana {np.median(a):+.4f}")
        print("  es lo que el operador devuelve cuando NO hay estuario.")
    if est:
        a = np.array([r["aporta"] for r in est])
        print(f"ESTUARIOS ({len(est)}): aporta de {a.min():+.4f} a "
              f"{a.max():+.4f} m · mediana {np.median(a):+.4f}")
    if est and ctl:
        thr = float(np.percentile([r["aporta"] for r in ctl], 95))
        win = [r for r in est if r["aporta"] > max(thr, 0.01)]
        print(f"\numbral (p95 de los controles, o 1 cm): {max(thr,0.01):+.4f} m")
        print(f"estuarios cortos que lo superan: {len(win)} de {len(est)}")
        for r in win:
            print(f"   {r['km']:5.1f} km  {r['label'][:38]:38s} "
                  f"{r['aporta']:+.4f} m")
        print()
        if not win:
            print("EL METODO NO VALE A ESTAS DISTANCIAS: en ninguna ria corta")
            print("del mundo la transferencia aporta mas que el ruido del")
            print("propio metodo. La marea de la boca sirve tal cual.")
        elif len(win) < len(est) / 2:
            print("VALE SOLO EN ALGUNOS: no basta con que sea corto, depende")
            print("del estuario. Habria que saber cual de antemano.")
        else:
            print("VALE TAMBIEN EN RIAS CORTAS, al contrario de lo que")
            print("sugeria el unico punto de Ferrol.")

    json.dump(rows, open(os.path.join(SC, "rias_cortas.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
