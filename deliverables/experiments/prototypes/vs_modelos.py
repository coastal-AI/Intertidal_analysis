"""Global tide models against real gauges, inside estuaries and out.

Every elevation this package produces inland is anchored to a global ocean
tide model queried at the site — EOT20 at the water centroid of each cell in
the north-coast campaign. Nobody had checked what that costs, because checking
needs a gauge and the mapped sites have none. But seventeen gauges are already
downloaded, several deep inside estuaries, and three models sit on disk.

Three questions:

1. how good is a global model AT THE COAST, the regime it was built for;
2. how much worse INSIDE an estuary, where its nearest ocean cell is
   kilometres away over land and it is extrapolating;
3. how does the boundary-plus-transfer operator compare — because its real
   competitor is not "no correction", it is "query a global model there".

For the third question every station gets an operator value, not only the
inner member of a hand-picked pair. Each station is predicted from the CLOSEST
other gauge with data, using a transfer measured between the two on earlier
weeks. A transfer runs in either direction, so an outer station is predicted
from its inner neighbour just as readily. That turns the comparison into the
practical one a user faces: a neighbouring gauge with a measured transfer, or
a global model at the point?

Datums differ between gauge and model, so the median offset is removed before
scoring — the offset says nothing about skill. What is compared is the shape
of the curve through time.

Caveat on coverage: EOT20, GOT4.10 and GOT4.8 are on disk; FES2022 and TPXO
are not, and both are usually better in shelf seas. This ranks what we have.
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

from pyintertidal.boundary import GaugeBoundary
from pyintertidal.estuary import EstuaryTransfer

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MODELS = ["EOT20", "GOT4.10_nc", "GOT4.8_nc"]
CONST = ("M2", "S2", "N2", "K1", "O1", "M4")

STATIONS = {
    "hono":  (21.307, -157.867, "Honolulu", "costa"),
    "keehi": (21.318, -157.885, "Keehi Hbr", "costa"),
    "barc":  (41.342, 2.166, "Barcelona", "costa"),
    "barc2": (41.363, 2.187, "Port de Barcelona", "puerto"),
    "fer1":  (43.463, -8.326, "Ferrol 1", "ria (boca)"),
    "fer2":  (43.476, -8.249, "Ferrol 2", "ria (6 km)"),
    "vlis":  (51.443, 3.597, "Vlissingen", "estuario (boca)"),
    "brsk":  (51.401, 3.548, "Breskens", "estuario (boca)"),
    "trnz":  (51.336, 3.820, "Terneuzen", "estuario (28 km)"),
    "bon2":  (36.802, -6.338, "Bonanza", "estuario (boca)"),
    "sev2":  (37.318, -6.008, "Sevilla", "rio mareal (85 km)"),
    "tdan":  (25.183, 121.410, "Danhai", "estuario (boca)"),
    "tdas":  (25.168, 121.428, "Danshuei", "estuario (2 km)"),
    "goer":  (57.700, 11.900, "Goteborg Eriksberg", "estuario"),
    "gohi":  (57.712, 11.966, "Goteborg Hisingsbron", "estuario"),
    "leha":  (49.482, 0.106, "Le Havre", "costa"),
    "diep":  (49.930, 1.085, "Dieppe", "costa"),
}


def load_cached(code):
    hits = sorted(glob.glob(os.path.join(SC, f"ioc_{code}_*.json")))
    raw = []
    for h in hits:
        raw += json.load(open(h))
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


def rmse(pred, truth):
    m = np.isfinite(pred) & np.isfinite(truth)
    if m.sum() < 100:
        return np.nan
    d = pred[m] - truth[m]
    return float(np.sqrt(np.mean((d - np.median(d)) ** 2)))


def hav(a, b):
    R = 6371.0
    p1, p2 = np.radians(a[0]), np.radians(b[0])
    h = (np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2)
         * np.sin(np.radians(b[1] - a[1]) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(h))


def main():
    from eo_tides.model import model_tides

    full = {c: load_cached(c) for c in STATIONS}
    full = {c: v for c, v in full.items() if v is not None}
    recs = {c: v.set_index("time")["level_m"].resample("1h").mean()
            for c, v in full.items()}
    print(f"{len(recs)} estaciones con registro\n")

    lo = max(s.index.min() for s in recs.values())
    hi = min(s.index.max() for s in recs.values())
    times = pd.date_range(max(lo, hi - pd.Timedelta(days=60)), hi, freq="1h")
    print(f"ventana de prueba: {times[0].date()} a {times[-1].date()}\n")

    codes = list(recs)
    lats = [STATIONS[c][0] for c in codes]
    lons = [STATIONS[c][1] for c in codes]
    preds = {}
    for m in MODELS:
        try:
            df = model_tides(x=lons, y=lats, time=times, model=m,
                             directory="tide_models", crs="EPSG:4326",
                             extrapolate=True, cutoff=np.inf,
                             parallel=False).reset_index()
            preds[m] = {c: df[(df["x"] == lons[i]) & (df["y"] == lats[i])]
                        .sort_values("time")["tide_height"].to_numpy(float)
                        for i, c in enumerate(codes)}
        except Exception as e:
            print(f"  {m}: fallo {type(e).__name__}")

    # ── the operator, for EVERY station, from its nearest neighbour ──────
    fit_end = times[0]
    op, src = {}, {}
    for c in codes:
        here = (STATIONS[c][0], STATIONS[c][1])
        near = sorted(((hav(here, (STATIONS[o][0], STATIONS[o][1])), o)
                       for o in codes if o != c),
                      key=lambda x: x[0])
        for km, other in near[:1]:
            if km < 0.3 or km > 150:
                continue
            try:
                o_full, i_full = full[other], full[c]
                mo = o_full["time"] < fit_end
                mi = i_full["time"] < fit_end
                if mo.sum() < 3000 or mi.sum() < 3000:
                    continue
                T = EstuaryTransfer.from_gauges(
                    GaugeBoundary(o_full["time"][mo], o_full["level_m"][mo]),
                    GaugeBoundary(i_full["time"][mi], i_full["level_m"][mi]),
                    max(km, 0.5), constituents=CONST)
                op[c] = T.levels(
                    GaugeBoundary(o_full["time"], o_full["level_m"]),
                    max(km, 0.5), times)
                src[c] = (other, km)
            except Exception as e:
                print(f"  operador {c}: {type(e).__name__}")

    hdr = " ".join(f"{m[:9]:>9s}" for m in preds)
    print(f"{'estacion':22s} {'posicion':20s} {hdr} {'OPERADOR':>9s}   "
          f"desde")
    print("-" * 108)
    rows = []
    for c in codes:
        truth = recs[c].reindex(times).to_numpy(float)
        row = {"code": c, "label": STATIONS[c][2], "pos": STATIONS[c][3]}
        cells = []
        for m in preds:
            r = rmse(preds[m][c], truth)
            row[m] = r
            cells.append(f"{r:9.3f}" if np.isfinite(r) else f"{'—':>9s}")
        if c in op:
            r = rmse(op[c], truth)
            row["operador"] = r
            row["desde"] = f"{STATIONS[src[c][0]][2]} ({src[c][1]:.1f} km)"
            cells.append(f"{r:9.3f}")
        else:
            row["desde"] = ""
            cells.append(f"{'—':>9s}")
        print(f"{STATIONS[c][2][:22]:22s} {STATIONS[c][3]:20s} "
              + " ".join(cells) + f"   {row['desde']}")
        rows.append(row)

    print("\nRMSE en metros contra el mareografo real, quitado el desfase de")
    print("datum. 'desde' es el mareografo usado como contorno del operador.")

    coast = [r for r in rows if r["pos"] in ("costa", "puerto")]
    inside = [r for r in rows if "km)" in r["pos"] or r["pos"] == "estuario"]
    for name, grp in (("EN COSTA", coast), ("DENTRO DE ESTUARIO", inside)):
        if not grp:
            continue
        print(f"\n{name} ({len(grp)} estaciones):")
        best = None
        for m in list(preds) + ["operador"]:
            v = np.array([r.get(m, np.nan) for r in grp], dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                med = float(np.median(v))
                print(f"  {m:12s} RMSE mediano {med:.3f} m ({len(v)} est.)")
                if best is None or med < best[1]:
                    best = (m, med)
        if best:
            print(f"  -> mejor: {best[0]} ({best[1]:.3f} m)")

    won = [r for r in rows if np.isfinite(r.get("operador", np.nan))
           and r["operador"] < min(r.get(m, np.inf) for m in preds)]
    print(f"\nel operador gana al mejor modelo global en {len(won)} de "
          f"{sum(1 for r in rows if 'operador' in r)} estaciones:")
    for r in won:
        b = min(r.get(m, np.inf) for m in preds)
        print(f"  {r['label'][:24]:24s} {r['pos']:20s} "
              f"{b:.3f} -> {r['operador']:.3f} m   (desde {r['desde']})")

    json.dump(rows, open(os.path.join(SC, "vs_modelos.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
