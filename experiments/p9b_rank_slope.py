"""P9b — gamma the simple way: rank correlation per band + a straight line.

The user asked for a determination of gamma (lag per km) that does not go
through the profiled likelihood. This is it, and it is the direct heir of
the ORIGINAL August prototype (retardo_imagen.py), whose synthetic
calibration measured bias -0.2 +/- 3.9 min:

  Step A  per band k, the wet fraction f_k(t) — the share of the band's
          clear pixels that are wet in scene t — is a unitless tide gauge.
          Slide the clock: tau_k = argmax_tau Spearman(f_k, h(t - tau)).
          Rank correlation is invariant to gain and datum by construction,
          so the affine theorem is respected for free.
  Step B  gamma = ordinary least-squares slope of tau_k against the
          geodesic distance s_k (weighted by band size).

Same pixels, bands and null replicas as m2_real/p9, so all three gamma
routes (likelihood sweep, this, and the free-band regression) are directly
comparable. Runs in ~2 min: nothing is profiled.

Run:  python -m experiments.p9b_rank_slope
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
from scipy.stats import spearmanr

from pyintertidal.net import use_system_certificates
from pyintertidal import simulator, seal
from pyintertidal import tide_estimators as te
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "p9b_rank_slope")
N_NULL = 20                       # cheap now: no profiling
TAU_GRID = np.arange(-45.0, 120.0 + 5.0, 5.0)


def band_lags(wet, clear, bank, cols_by_band, min_cover=50):
    """tau per band by sliding-clock rank correlation of the wet fraction."""
    taus, rhos = [], []
    for cols in cols_by_band:
        W = wet[:, cols]
        C = clear[:, cols]
        n = C.sum(axis=1)
        ok = n >= min_cover
        f = np.where(ok, W.sum(axis=1) / np.maximum(n, 1), np.nan)
        scores = []
        for tv in TAU_GRID:
            h = bank.at(float(tv))
            m = ok & np.isfinite(f)
            scores.append(spearmanr(f[m], h[m]).statistic)
        scores = np.asarray(scores)
        i = int(np.nanargmax(scores))
        # parabolic refinement around the argmax: sub-grid tau resolution
        taus.append(te._parabolic_argmax(scores, TAU_GRID))
        rhos.append(float(scores[i]))
    return np.asarray(taus), np.asarray(rhos)


def slope(taus, centers, weights):
    """Weighted OLS slope of tau against s, anchored nowhere (free line)."""
    w = np.asarray(weights, float)
    s = np.asarray(centers, float)
    t = np.asarray(taus, float)
    sw, tw = np.average(s, weights=w), np.average(t, weights=w)
    return float(np.sum(w * (s - sw) * (t - tw)) / np.sum(w * (s - sw) ** 2))


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(CFG["datos"]["store"], allow_pickle=True)
    dates = np.array([str(s) for s in d["dates"]])
    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([(s in times) and (int(s[:4]) >= CFG["epoca_min"])
                     for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)

    def tide_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    bank_taus = [float(v) for v in CFG["taus_banco_min"]]
    bank = te.ShiftBank(bank_taus, [
        tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
    h0 = bank.at(0.0)
    rising = (tide_at(t_real + pd.Timedelta(minutes=30))
              - tide_at(t_real - pd.Timedelta(minutes=30))) > 0

    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG["epoca_min"],
                              sd_level=CFG["sd_nivel_m"])
    geo = np.load(CFG["datos"]["geometria"])
    s_all = geo["s_keep"].astype(float) / 1000.0
    s_tpl = s_all[tpl["idx"]]
    nb = CFG["n_bandas"]
    edges, centers, band_all = te.make_bands(
        np.where(np.isfinite(s_tpl), s_tpl, np.nan), nb)
    # identical seed and per-band cap as m2_real/p9/p9c: every gamma route
    # scores the very same pixel subsample, so the routes are comparable
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for k in range(nb):
        cand = np.where(band_all == k)[0]
        rows.append(rng.choice(cand, min(len(cand), 2000), replace=False))
    rows = np.sort(np.concatenate(rows))
    band_of = band_all[rows]
    cols_store = tpl["idx"][rows]
    cols_by_band = [np.where(band_of == k)[0] for k in range(nb)]
    n_by_band = [len(c) for c in cols_by_band]

    Yr = np.nan_to_num(d["Y"][have][:, cols_store], nan=0.0).astype(np.float32)
    Cr = d["C"][have][:, cols_store] > 0
    wet_r = Cr & (Yr > 0)

    taus, rhos = band_lags(wet_r, Cr, bank, cols_by_band)
    g_hat = slope(taus, centers, n_by_band)
    print(f"REAL: tau per band {np.round(taus, 1)} · rho {np.round(rhos, 3)}"
          f" · gamma = {g_hat:+.2f} min/km  ({time.time()-t0:.0f} s)",
          flush=True)

    z_true = tpl["z"][rows]
    g_nulls = []
    for i in range(N_NULL):
        Y0, C0 = simulator.simulate(tpl, z_true, h0,
                                    seed=CFG["seed"] + 100 + i,
                                    rising=rising, s_km=s_tpl[rows],
                                    C_real=Cr, template_rows=rows)
        w0 = C0 & (Y0 > 0)
        t_n, _ = band_lags(w0, C0, bank, cols_by_band)
        g_nulls.append(slope(t_n, centers, n_by_band))
    g_nulls = np.asarray(g_nulls)
    outside = bool(g_hat < g_nulls.min() or g_hat > g_nulls.max())
    print(f"nulls: {np.round(g_nulls, 2)} -> outside: {outside}  "
          f"({time.time()-t0:.0f} s)", flush=True)

    c_ms = (1000.0 / (60.0 * g_hat)) if g_hat > 0 else None
    result = {
        "metodo": "Spearman(f_band, h(t-tau)) per band + weighted OLS slope",
        "tau_por_banda_min": taus.tolist(),
        "rho_por_banda": rhos.tolist(),
        "centros_km": centers.tolist(),
        "gamma_min_per_km": g_hat,
        "gamma_nulos": g_nulls.tolist(),
        "fuera_del_nulo": outside,
        "celeridad_m_s": c_ms,
        "profundidad_efectiva_m": (c_ms ** 2 / 9.81) if c_ms else None,
        "n_px": int(len(rows)), "n_escenas": int(Yr.shape[0]),
        "n_nulos": N_NULL, "seed": CFG["seed"],
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    ax.plot(centers, taus, "o", color="#1a76b3", ms=10,
            label="tau per band (rank correlation)")
    ss = np.linspace(centers[0], centers[-1], 50)
    tw = np.average(taus, weights=n_by_band)
    sw = np.average(centers, weights=n_by_band)
    ax.plot(ss, tw + g_hat * (ss - sw), "-", color="#e8a013", lw=2.5,
            label=f"OLS: gamma = {g_hat:+.2f} min/km")
    for g0 in g_nulls:
        ax.plot(ss, tw + g0 * (ss - sw), "-", color="#9aa39c",
                lw=0.8, alpha=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("geodesic distance from the mouth s (km)")
    ax.set_ylabel("tau (min)")
    ax.set_title("gamma without likelihood: slide-the-clock correlation "
                 "+ a straight line\n(grey: 20 uniform-tide null slopes)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"))
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
