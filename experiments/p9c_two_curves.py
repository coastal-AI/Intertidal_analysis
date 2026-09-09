"""P9c — gamma with NO optimisation at all: two medians, a subtraction,
a division.

The simplest estimator of the interior lag that exists, requested by the
user after finding both the likelihood sweep (p9) and the sliding-clock
correlation (p9b) too elaborate. The physics in one sentence: if a zone's
water arrives late, the zone looks TOO DRY while the mouth tide is rising
and TOO WET while it is falling — at the same mouth level. So, per band:

  h_up   = median mouth tide of RISING scenes with the band half-flooded
           (wet fraction between 0.25 and 0.75)
  h_down = the same for FALLING scenes
  tau    = (h_up - h_down) / (2 * v)        with v = median |dh/dt|

The gap between the two half-flooded levels, in metres, divided by how
fast the tide moves, is the lag in minutes. No sweep, no argmax, no
correlation — arithmetic. gamma is then the school-level OLS slope of tau
against the geodesic distance of the band centres (and, even simpler, the
end-to-end slope is reported too).

Known limitation, declared: this conflates propagation lag with ponding
hysteresis (both open the rising/falling gap). It is the honest first
rung of the ladder, not the final word — the free-band likelihood (M2a)
remains the adopted estimator.

Same pixels, bands and null seeds as m2_real/p9/p9b: all gamma routes are
directly comparable. Runs in ~2 min.

Run:  python -m experiments.p9c_two_curves
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
from pyintertidal import simulator, seal
from pyintertidal import tide_estimators as te
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "p9c_two_curves")
N_NULL = 20


def band_lags_two_curves(wet, clear, h0, rate, rising, cols_by_band,
                         f_lo=0.25, f_hi=0.75, min_cover=50):
    """tau per band from the rising-vs-falling half-flooded levels."""
    taus, gaps, n_used = [], [], []
    v = float(np.nanmedian(np.abs(rate)))          # m per minute
    for cols in cols_by_band:
        W = wet[:, cols]
        C = clear[:, cols]
        n = C.sum(axis=1)
        ok = n >= min_cover
        f = np.where(ok, W.sum(axis=1) / np.maximum(n, 1), np.nan)
        half = ok & (f > f_lo) & (f < f_hi)
        up = half & rising
        down = half & ~rising
        if up.sum() < 8 or down.sum() < 8:
            taus.append(np.nan); gaps.append(np.nan)
            n_used.append(int(half.sum()))
            continue
        h_up = float(np.median(h0[up]))
        h_down = float(np.median(h0[down]))
        gap = h_up - h_down
        # the lag displaces the two curves by +/- v*tau in opposite
        # directions, so the observed gap is 2*v*tau
        taus.append(gap / (2.0 * v))
        gaps.append(gap)
        n_used.append(int(half.sum()))
    return np.asarray(taus), np.asarray(gaps), n_used, v


def ols_slope(taus, centers, weights):
    m = np.isfinite(taus)
    w = np.asarray(weights, float)[m]
    s = np.asarray(centers, float)[m]
    t = np.asarray(taus, float)[m]
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

    h0 = tide_at(t_real)
    h_p = tide_at(t_real + pd.Timedelta(minutes=30))
    h_m = tide_at(t_real - pd.Timedelta(minutes=30))
    rate = (h_p - h_m) / 60.0                      # m per minute, signed
    rising = rate > 0

    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG["epoca_min"],
                              sd_level=CFG["sd_nivel_m"])
    geo = np.load(CFG["datos"]["geometria"])
    s_all = geo["s_keep"].astype(float) / 1000.0
    s_tpl = s_all[tpl["idx"]]
    nb = CFG["n_bandas"]
    edges, centers, band_all = te.make_bands(
        np.where(np.isfinite(s_tpl), s_tpl, np.nan), nb)
    # identical seed and per-band cap as m2_real/p9/p9b: every gamma route
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

    taus, gaps, n_used, v = band_lags_two_curves(
        wet_r, Cr, h0, rate, rising, cols_by_band)
    g_hat = ols_slope(taus, centers, n_by_band)
    g_ends = float((taus[-1] - taus[0]) / (centers[-1] - centers[0])) \
        if np.isfinite(taus[[0, -1]]).all() else None
    print(f"REAL: tau {np.round(taus, 1)} min · gaps {np.round(gaps, 3)} m "
          f"· v={v*100:.2f} cm/min · gamma OLS {g_hat:+.2f} "
          f"(end-to-end {g_ends:+.2f}) min/km  ({time.time()-t0:.0f} s)",
          flush=True)

    z_true = tpl["z"][rows]
    g_nulls = []
    for i in range(N_NULL):
        Y0, C0 = simulator.simulate(tpl, z_true, h0,
                                    seed=CFG["seed"] + 100 + i,
                                    rising=rising, s_km=s_tpl[rows],
                                    C_real=Cr, template_rows=rows)
        w0 = C0 & (Y0 > 0)
        t_n, _, _, _ = band_lags_two_curves(w0, C0, h0, rate, rising,
                                            cols_by_band)
        g_nulls.append(ols_slope(t_n, centers, n_by_band))
    g_nulls = np.asarray(g_nulls)
    outside = bool(g_hat < np.nanmin(g_nulls) or g_hat > np.nanmax(g_nulls))
    print(f"nulls: {np.round(g_nulls, 2)} -> outside: {outside}", flush=True)

    result = {
        "metodo": "half-flooded level gap rising-vs-falling; "
                  "tau = gap/(2*v); gamma = weighted OLS slope",
        "tau_por_banda_min": taus.tolist(),
        "gap_por_banda_m": gaps.tolist(),
        "v_mediana_m_por_min": v,
        "centros_km": centers.tolist(),
        "gamma_min_per_km": g_hat,
        "gamma_puntas_min_per_km": g_ends,
        "gamma_nulos": g_nulls.tolist(),
        "fuera_del_nulo": outside,
        "n_px": int(len(rows)), "n_escenas": int(Yr.shape[0]),
        "n_nulos": N_NULL, "seed": CFG["seed"],
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
