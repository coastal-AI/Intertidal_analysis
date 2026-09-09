"""P10 — linear lag (P9) vs free bands (M2a) across the external-truth AOIs.

For each site with a float-precision survey (Tagus, Vadehavet, Aveiro),
three clock models are scored on elevation against the SAME truth, on the
SAME pixels, with the SAME recipe as the stored p5 runs:

  uniforme   one EOT20 level per scene            (stored in z_scores.npz)
  operador   free per-band clocks, M2a, gated     (stored in z_scores.npz)
  lineal     tau(s) = gamma * (s - s0), gamma fitted here by the same
             profiled likelihood sweep as experiments/p9 (subsampled), then
             elevations re-inverted with the linear clocks (full pixels)

Also reported per site: the NLL ladder (uniform / linear / free) on the
subsampled population, so the likelihood picture and the external-truth
picture can be read side by side. The free-band taus are NOT refitted —
they are read from the stored p5 result, so the comparison uses exactly
the clocks that produced the stored z_op.

Run:  python -m experiments.p10_linear_vs_free_aoi     (~45 min, 3 sites)
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
from pyintertidal.marea import invert_series
import pyintertidal as pit
from experiments.p4_site import extract

CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "p10_linear_vs_free")
SITES = {
    "tejo": "ndwi_cube_tejo_2019_2021.nc",
    "vadehavet": "ndwi_cube_vadehavet_2019_2021.nc",
    "aveiro": "ndwi_cube_aveiro_2023-2025.nc",
}
# widened twice (8 -> 16 -> 32) after the ML optimum pinned at the grid
# edge: a capped grid silently regularises gamma and flatters the linear
# model (phase log, P10) — the cap is NOT a tunable
GAMMA_GRID = np.arange(-2.0, 32.0 + 0.5, 0.5)
SUB_PX_BAND = 2000


def nll_of_taus(taus_min, wet, clear, bank, cols_by_band, z_grid):
    total, npx_sum = 0.0, 0
    for k, cols in enumerate(cols_by_band):
        if len(cols) < 50:
            continue
        h = bank.at(float(taus_min[k]))
        nll, npx = te._band_nll(wet[:, cols], clear[:, cols], h,
                                z_grid, CFG2["sigma0_m"],
                                sg_grid=tuple(CFG2["sigma_perfil_m"]))
        total += nll * npx
        npx_sum += npx
    return total / max(npx_sum, 1)


def score(z, z_ref, com):
    e = z[com] - z_ref[com]
    e = e - np.median(e)
    return {"rmse_centrado": float(np.sqrt(np.mean(e ** 2))),
            "pendiente": float(np.polyfit(z_ref[com], z[com], 1)[0]),
            "pearson": float(np.corrcoef(z_ref[com], z[com])[0, 1]),
            "n": int(com.sum())}


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    only = os.environ.get("SITES")
    sites = {k: v for k, v in SITES.items()
             if only is None or k in only.split(",")}
    out_all = {}
    for site, cube in sites.items():
        ts = time.time()
        Z = np.load(f"results/p5_{site}/z_scores.npz")
        p5 = json.load(open(f"results/p5_{site}/result.json",
                            encoding="utf-8"))
        tau_free = np.asarray(p5["tau_m2a_min"], float)

        Y, C, keep, dates, SH, sea, inter, bbox = extract(cube)
        assert np.array_equal(keep, Z["keep"]), f"{site}: keep mismatch"
        times = pit.overpass.get_overpass_times(
            bbox, ("2016-01-01", "2025-12-31"), verbose=False)
        have = np.array([s in times for s in dates])
        t_real = pd.DatetimeIndex(pd.to_datetime(
            [times[s] for s in dates[have]])).tz_localize(None)
        Y = np.nan_to_num(Y[have], nan=0.0).astype(np.float64)
        C = (C[have] > 0).astype(np.float64)
        wet = (C > 0) & (Y > 0)
        lat_c = 0.5 * (bbox["south"] + bbox["north"])
        lon_c = 0.5 * (bbox["west"] + bbox["east"])

        def tide_at(t):
            return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                               directory="tide_models", crs="EPSG:4326",
                               extrapolate=True, cutoff=np.inf,
                               parallel=False).reset_index().sort_values(
                "time")["tide_height"].to_numpy(float)

        bank_taus = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0,
                     75.0, 90.0, 105.0, 120.0]
        bank = te.ShiftBank(bank_taus, [
            tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
        h0 = bank.at(0.0)

        s_km = Z["s_km"].astype(float)
        nb = CFG2["n_bandas"]
        edges, centers, band_of = te.make_bands(s_km, nb)
        assert len(tau_free) == nb, f"{site}: stored taus != {nb} bands"

        # gamma sweep on a subsample, same likelihood as p9
        rng = np.random.default_rng(CFG2["seed"] + 77)
        cols_sub = []
        for k in range(nb):
            cand = np.where(band_of == k)[0]
            cols_sub.append(np.sort(rng.choice(
                cand, min(len(cand), SUB_PX_BAND), replace=False)))
        z_grid = np.linspace(h0.min() - 0.3, h0.max() + 0.3,
                             CFG2["z_puntos"])
        s_rel = centers - centers[0]
        curve = np.array([nll_of_taus(g * s_rel, wet, C > 0, bank,
                                      cols_sub, z_grid)
                          for g in GAMMA_GRID])
        g_hat = float(GAMMA_GRID[int(np.argmin(curve))])
        tau_lin = g_hat * s_rel
        ladder = {
            "uniforme": float(curve[np.argmin(np.abs(GAMMA_GRID))]),
            "lineal": float(curve.min()),
            "bandas_libres": float(nll_of_taus(tau_free, wet, C > 0, bank,
                                               cols_sub, z_grid)),
        }
        print(f"[{site}] gamma={g_hat:+.1f} min/km · tau_lin="
              f"{np.round(tau_lin, 1)} · ladder {ladder} "
              f"({time.time()-ts:.0f} s)", flush=True)

        # invert the linear variant on ALL pixels, band by band
        lo, hi = float(h0.min()), float(h0.max())
        z_lin = np.full(len(keep), np.nan)
        for k in range(nb):
            cols = np.where(band_of == k)[0]
            if not len(cols):
                continue
            h_k = bank.at(float(tau_lin[k])) if tau_lin[k] else h0
            z_lin[cols], _ = invert_series(Y[:, cols], C[:, cols], h_k,
                                           lo, hi)

        z_ref, z_uni, z_op = Z["z_ref"], Z["z_uni"], Z["z_op"]
        com = (np.isfinite(z_ref) & np.isfinite(z_uni) & np.isfinite(z_op)
               & np.isfinite(z_lin) & np.isfinite(s_km))
        res = {"uniforme": score(z_uni, z_ref, com),
               "lineal": score(z_lin, z_ref, com),
               "operador": score(z_op, z_ref, com)}
        out_all[site] = {"gamma_min_per_km": g_hat,
                         "tau_lineal_min": tau_lin.tolist(),
                         "tau_libres_min": tau_free.tolist(),
                         "centros_km": centers.tolist(),
                         "nll_ladder": ladder, **res}
        for name in ("uniforme", "lineal", "operador"):
            v = res[name]
            print(f"   {name:9s} RMSE={v['rmse_centrado']:.3f} "
                  f"slope={v['pendiente']:.3f} r={v['pearson']:.3f} "
                  f"n={v['n']:,}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    prev = (json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))["sitios"]
        if os.path.exists(os.path.join(OUT, "result.json")) else {})
    prev.update(out_all)
    json.dump({"sitios": prev,
               "inputs_sha": {s: seal._sha256(c)
                              for s, c in SITES.items()}},
              open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
