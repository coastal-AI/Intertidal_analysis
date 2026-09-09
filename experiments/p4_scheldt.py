"""Parte IV: external validation of the M2 engine on the Scheldt.

The only external truth available without new instruments: the Westerschelde
has tide gauges, and the imagery-only lag gradient was already measured both
ways on 2026-08-18 — gauges give ~0.9 min/km along the axis, and EOT20's
residual phase error at Terneuzen is +10 min. Here the NEW estimator (M2a,
phase-only, sigma-profiled — the one that won the M2 gate) runs on the
sealed Scheldt archive, blind, and its along-axis lag gradient is compared
with the gauge number. M2d (the literature baseline) runs alongside.

Pre-registered check (tolerance in-line with its rationale): the fitted
gradient must fall within 0.5 min/km of the gauges' 0.9 — the estimator's
own demonstrated noise (~4-5 min RMSE over an ~18 km axis => ~0.4 min/km on
a fitted slope) plus margin. Sign control: the mouth-anchored profile must
INCREASE inland, as the physics and the gauges say.

Run:  python -m experiments.p4_escalda      (~10 min)
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml

from pyintertidal.net import use_system_certificates
from pyintertidal import seal
from pyintertidal import tide_estimators as te
import pyintertidal as pit

CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "p4_escalda")
NPZ = "data_v4/store/escalda_intermareal.npz"
MOUTH = (51.443, 3.597)               # Vlissingen side, as in the prototype
BBOX = {"west": 3.55, "south": 51.33, "east": 3.85, "north": 51.46,
        "crs": "EPSG:4326"}
GAUGE_GRADIENT = 0.9                  # min/km, measured with tide gauges
TOL_GRADIENT = 0.5                    # min/km (demonstrated noise + margin)


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(NPZ, allow_pickle=True)
    Y, C = d["Y"], d["C"]
    dates = np.array([str(s) for s in d["dates"]])
    s_km = d["s_km"].astype(float)
    times = pit.overpass.get_overpass_times(
        BBOX, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    Y = np.nan_to_num(Y[have], nan=0.0).astype(np.float32)
    Cb = C[have] > 0
    wet = Cb & (Y > 0)
    print(f"Scheldt: {Y.shape[0]} scenes x {Y.shape[1]:,} px", flush=True)

    lat_c, lon_c = MOUTH

    def tide_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    # the Scheldt's lags are small (±10 min): a tighter bank around zero
    bank_taus = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0]
    bank = te.ShiftBank(bank_taus, [
        tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
    rising = (tide_at(t_real + pd.Timedelta(minutes=30))
              - tide_at(t_real - pd.Timedelta(minutes=30))) > 0
    print(f"bank ready ({time.time()-t0:.0f} s)", flush=True)

    nb = CFG2["n_bandas"]
    edges, centers, band_of = te.make_bands(s_km, nb)
    tau_grid = np.arange(-20.0, 30.0 + 5.0, 5.0)

    r2a = te.m2a_rasch(wet, Cb, bank, band_of, centers, nb,
                       sigma0=CFG2["sigma0_m"],
                       sg_grid=tuple(CFG2["sigma_perfil_m"]),
                       tau_bounds=(-30.0, 45.0),
                       z_points=CFG2["z_puntos"],
                       rng=np.random.default_rng(CFG2["seed"] + 7),
                       max_px_band=CFG2["max_px_banda"]["m2a"])
    tau_d = te.m2d_flood_ebb(Y.astype(np.float64), Cb.astype(np.float64),
                             bank, rising, band_of, nb, tau_grid,
                             rng=np.random.default_rng(CFG2["seed"] + 8),
                             max_px_band=CFG2["max_px_banda"]["m2d"])
    tau_a = np.asarray(r2a["tau"], float)
    print(f"tau_a={np.round(tau_a, 1)}  tau_d={np.round(tau_d, 1)} "
          f"({time.time()-t0:.0f} s)", flush=True)

    # gradient of the mouth-anchored profile along the axis
    grad_a = float(np.polyfit(centers, tau_a, 1)[0])
    fin = np.isfinite(tau_d)
    grad_d = float(np.polyfit(centers[fin], tau_d[fin], 1)[0]) \
        if fin.sum() >= 3 else float("nan")
    ok_grad = abs(grad_a - GAUGE_GRADIENT) <= TOL_GRADIENT
    ok_sign = tau_a[-1] > tau_a[1]

    result = {
        "centros_km": centers.tolist(),
        "tau_m2a_min": tau_a.tolist(),
        "tau_m2d_min": np.asarray(tau_d, float).tolist(),
        "gradiente_m2a_min_km": grad_a,
        "gradiente_m2d_min_km": grad_d,
        "gradiente_mareografos_min_km": GAUGE_GRADIENT,
        "tolerancia_min_km": TOL_GRADIENT,
        "validacion": {"gradiente_ok": bool(ok_grad),
                       "signo_ok": bool(ok_sign),
                       "PASA": bool(ok_grad and ok_sign)},
        "n_escenas": int(Y.shape[0]), "n_px": int(Y.shape[1]),
        "inputs_sha": {"escalda": seal._sha256(NPZ)},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(centers, tau_a, "o-", color="C0",
            label=f"M2a (gradient {grad_a:.2f} min/km)")
    ax.plot(centers, tau_d, "x--", color="C3",
            label=f"M2d (gradient {grad_d:.2f})")
    xs = np.array([centers[0], centers[-1]])
    ax.plot(xs, tau_a[0] + GAUGE_GRADIENT * (xs - centers[0]), "k:",
            label="tide gauges: 0.9 min/km")
    ax.set_xlabel("s along the axis (km)")
    ax.set_ylabel("lag vs EOT20 (min)")
    ax.legend(fontsize=8)
    ax.set_title(f"Scheldt: external validation — "
                 f"{'PASS' if result['validacion']['PASA'] else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print(json.dumps(result["validacion"], indent=1), flush=True)


if __name__ == "__main__":
    main()
