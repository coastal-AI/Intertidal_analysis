"""The Terneuzen table, repeated at ALL the studied tide gauges.

Per INNER station of every gauge pair the project has studied (the interior
point, where the interior-tide question lives), same recipe as p6: cached
IOC record, test window = last 35 %, 10-min grid, median-centred RMSE.

Rows per site: the three servable ocean models as-is, their plain mean, and
— cited from the sealed 2026-08-18 run (curva_estuarios), same gauges, 95-day
harmonic fit — the gauge-calibrated transfer operator (solo_armonicos vs
con_operador). The zero-instrument imagery operator only exists where an
image CUBE exists: the Scheldt (p6). That asymmetry is the point: everywhere
else, these are the yardsticks any future cube would be scored against.

Run:  python -m experiments.p7_tabla_mareografos   (~4 min)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
from pyintertidal import seal
from pyintertidal.gauges import load_cached_ioc as load_gauge

OUT = os.path.join("results", "p7_tabla_mareografos")
MODELS = ["EOT20", "GOT4.10_nc", "GOT4.8_nc"]
# INNER stations of the studied pairs + their historical label
SITES = [
    ("keehi", "Honolulu - Keehi (control, 2.7 km)", "Honolulu - Keehi"),
    ("barc2", "Barcelona - Port (control, 2.9 km)",
     "Barcelona - Port de Barcelona"),
    ("tdas", "Tamsui/Danshuei (estuario somero, 2.0 km)", None),
    ("fer2", "Ferrol interior (ria profunda, 6.4 km)", "Ferrol (ria corta)"),
    ("trnz", "Terneuzen (Escalda, 28 km)",
     "Escalda: Vlissingen - Terneuzen"),
    ("sev2", "Sevilla (Guadalquivir, 85 km)",
     "Guadalquivir: Bonanza - Sevilla"),
    ("diep", "Dieppe (costa abierta, 90 km)", "Le Havre - Dieppe"),
]


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    stations = {s["code"]: s for s in
                json.load(open("data_v4/gauges/ioc_stations.json"))}
    hist = {r["label"]: r for r in json.load(
        open("data_v4/prototypes/results_json/curva_estuarios.json"))}

    table = {}
    for code, label, hist_label in SITES:
        g = load_gauge(code)
        if g is None or len(g) < 2000:
            table[label] = {"error": "not enough cached data"}
            continue
        st = stations[code]
        lat, lon = float(st["Lat"]), float(st["Lon"])
        lo, hi = g["time"].min(), g["time"].max()
        split = lo + (hi - lo) * 0.65
        tt = pd.date_range(split, hi, freq="10min")
        x = g["time"].values.astype("int64").astype(float)
        truth = np.interp(tt.values.astype("int64").astype(float), x,
                          g["level_m"].to_numpy(float),
                          left=np.nan, right=np.nan)

        def rmse(pred):
            m = np.isfinite(pred) & np.isfinite(truth)
            d = pred[m] - truth[m]
            return float(np.sqrt(np.mean((d - np.median(d)) ** 2)))

        rows, series = {}, {}
        for mdl in MODELS:
            p = model_tides(x=[lon], y=[lat], time=tt, model=mdl,
                            directory="tide_models", crs="EPSG:4326",
                            extrapolate=True, cutoff=np.inf,
                            parallel=False).reset_index().sort_values(
                "time")["tide_height"].to_numpy(float)
            series[mdl] = p
            rows[mdl] = rmse(p)
        rows["ensemble_media3"] = rmse(
            np.mean([series[m] for m in MODELS], axis=0))
        if hist_label and hist_label in hist:
            rows["historico_solo_armonicos"] = hist[hist_label]["solo_arm"]
            rows["historico_con_operador_mareografos"] = \
                hist[hist_label]["con_op"]
        table[label] = {"code": code, "lat": lat, "lon": lon,
                        "ventana": [str(split), str(hi)], "rmse_m": rows}
        best = min((v for k, v in rows.items() if k in MODELS
                    or k == "ensemble_media3"))
        print(f"{label:46s} EOT20 {rows['EOT20']:.3f} · "
              f"GOT4.10 {rows['GOT4.10_nc']:.3f} · "
              f"ens {rows['ensemble_media3']:.3f} · best {best:.3f}",
              flush=True)

    os.makedirs(OUT, exist_ok=True)
    json.dump({"tabla": table,
               "nota": "imagery operator only where a cube exists (Scheldt, "
                       "p6); these are the bars for future cubes",
               "inputs_sha": {"curva_estuarios": seal._sha256(
                   "data_v4/prototypes/results_json/curva_estuarios.json")}},
              open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
