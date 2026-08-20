"""The final table: tide models vs methods, against the Terneuzen tide
gauge — the only external judge of LEVEL this project has.

Same recipe as the 2026-08-18 validation (escalda_valida prototype), so the
numbers are directly comparable: test window = last 35 % of the cached gauge
record, 10-min grid, median-centred RMSE (datum-free). Rows:

* the four OCEAN MODELS on disk, as-is (EOT20, GOT4.10c, GOT4.8, GOT5.6)
  and their plain-mean ensemble — what you get with no method at all;
* EOT20 + the v3 imagery adapter (tau=+5.2 min, historical number rerun);
* EOT20 + the NEW operator: tau from the M2a profile measured on the sealed
  Scheldt archive (results/p4_escalda), interpolated at Terneuzen's
  position on the same axis — zero instruments, gate-validated engine;
* the ceiling: best pure lag found by sweeping AGAINST the gauge (cheating,
  for reference);
* (cited, not recomputed) the gauge-calibrated harmonic transfer operator —
  needs two gauges, so it is the yardstick the zero-instrument rows chase.

Run:  python -m experiments.p6_comparativa_terneuzen   (~3 min)
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

OUT = os.path.join("results", "p6_comparativa")
INNER = (51.336, 3.820)              # Terneuzen (lat, lon)
S_INNER_KM = 15.63                   # position on the axis east of
                                     # Vlissingen
MODELS = ["EOT20", "GOT4.10_nc", "GOT4.8_nc"]   # the actually servable ones
# (GOT5.6 is half on disk and GOT4.10_SAL is the load tide, not the tide)
TAU_V3 = 5.2                         # the v3 adapter (historical, resealed)
TAU_CEIL_GRID = np.arange(-20.0, 31.0, 2.5)
# historical reference (needs TWO tide gauges; 2026-08-18):
GAUGE_OP = {"solo_armonicos": 0.4051, "con_operador": 0.3072}


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    # tau of the NEW operator at Terneuzen, from the sealed-archive profile
    p4 = json.load(open("results/p4_escalda/result.json", encoding="utf-8"))
    tau_new = float(np.interp(S_INNER_KM, p4["centros_km"],
                              p4["tau_m2a_min"]))

    g = load_gauge("trnz")
    lo, hi = g["time"].min(), g["time"].max()
    split = lo + (hi - lo) * 0.65
    tt = pd.date_range(split, hi, freq="10min")
    x = g["time"].values.astype("int64").astype(float)
    truth = np.interp(tt.values.astype("int64").astype(float), x,
                      g["level_m"].to_numpy(float), left=np.nan,
                      right=np.nan)

    def rmse(pred):
        m = np.isfinite(pred) & np.isfinite(truth)
        d = pred[m] - truth[m]
        return float(np.sqrt(np.mean((d - np.median(d)) ** 2)))

    def tide(model, tau_min=0.0):
        t = tt - pd.Timedelta(minutes=float(tau_min))
        return model_tides(x=[INNER[1]], y=[INNER[0]], time=t, model=model,
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    rows = {}
    series = {}
    for m in MODELS:
        series[m] = tide(m)
        rows[f"{m} tal cual"] = rmse(series[m])
        print(f"{m} tal cual: {rows[f'{m} tal cual']:.4f}", flush=True)
    rows["ensemble (media de los 3)"] = rmse(
        np.mean([series[m] for m in MODELS], axis=0))
    rows[f"EOT20 + adaptador v3 (tau={TAU_V3:.0f}m)"] = rmse(
        tide("EOT20", TAU_V3))
    rows[f"EOT20 + OPERADOR NUEVO M2a (tau={tau_new:.1f}m)"] = rmse(
        tide("EOT20", tau_new))
    sweep = {float(tv): rmse(tide("EOT20", tv)) for tv in TAU_CEIL_GRID}
    tau_best = min(sweep, key=sweep.get)
    rows[f"techo: mejor retardo contra el mareografo ({tau_best:+.1f}m)"] = \
        sweep[tau_best]

    result = {
        "ventana_test": [str(split), str(hi)],
        "tau_operador_nuevo_min": tau_new,
        "rmse_m": rows,
        "referencia_con_mareografos": GAUGE_OP,
        "barrido_techo": sweep,
        "inputs_sha": {
            "p4_escalda": seal._sha256("results/p4_escalda/result.json")},
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print(json.dumps(rows, indent=1))
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
