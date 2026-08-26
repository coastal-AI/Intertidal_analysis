"""P9 — the one-parameter tide adapter: tau(s) = gamma * (s - s0).

The free-band MAREA estimator (M2a) spends five free clocks; this baseline
spends ONE: a linear dependence of the lag on the along-water distance to
the mouth, anchored like the band model (tau = 0 at the mouth band). It is
the simplest adaptation model that can generate a lag from geometry, and
the natural first rung of a piece-by-piece validation ladder:

    uniform (gamma = 0)  ->  linear (this)  ->  free bands (M2a)

Everything else is held identical to experiments/m2_real.py: same store,
same template pixels, same band assignment and subsampling seed, same
profiled Bernoulli NLL (z and sigma grids from configs/m2.yaml), same
uniform-tide null replicas (same seeds). The verdict is therefore directly
comparable: NLL(gamma) on the same axis as the band model's NLL, and a
gamma null band that says what slope the machinery invents on its own.

Physics bonus: gamma is a wave celerity in disguise — c = 1/gamma, and the
effective depth follows from c = sqrt(g*h_eff).

Run:  python -m experiments.p9_parametric_lag     (~20-40 min)
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
OUT = os.path.join("results", "p9_parametric_lag")
N_NULL = 5
GAMMA_GRID = np.arange(-2.0, 8.0 + 0.25, 0.25)     # min per km


def nll_of_taus(taus_min, wet, clear, bank, cols_by_band, z_grid):
    """The exact objective of m2a_rasch, evaluated at a FIXED tau vector."""
    total, npx_sum = 0.0, 0
    for k, cols in enumerate(cols_by_band):
        if len(cols) < 50:
            continue
        h = bank.at(float(taus_min[k]))
        nll, npx = te._band_nll(wet[:, cols], clear[:, cols], h,
                                z_grid, CFG["sigma0_m"],
                                sg_grid=tuple(CFG["sigma_perfil_m"]))
        total += nll * npx
        npx_sum += npx
    return total / max(npx_sum, 1)


def gamma_sweep(wet, clear, bank, cols_by_band, z_grid, centers):
    s_rel = centers - centers[0]
    curve = np.array([nll_of_taus(g * s_rel, wet, clear, bank,
                                  cols_by_band, z_grid)
                      for g in GAMMA_GRID])
    i = int(np.argmin(curve))
    return float(GAMMA_GRID[i]), curve


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

    # identical pixel population to m2_real: template + same subsampling
    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG["epoca_min"],
                              sd_level=CFG["sd_nivel_m"])
    geo = np.load(CFG["datos"]["geometria"])
    s_all = geo["s_keep"].astype(float) / 1000.0
    s_tpl = s_all[tpl["idx"]]
    nb = CFG["n_bandas"]
    edges, centers, band_all = te.make_bands(
        np.where(np.isfinite(s_tpl), s_tpl, np.nan), nb)
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for k in range(nb):
        cand = np.where(band_all == k)[0]
        rows.append(rng.choice(cand, min(len(cand), 2000), replace=False))
    rows = np.sort(np.concatenate(rows))
    band_of = band_all[rows]
    cols_store = tpl["idx"][rows]
    cols_by_band = [np.where(band_of == k)[0] for k in range(nb)]

    Yr = np.nan_to_num(d["Y"][have][:, cols_store], nan=0.0).astype(np.float32)
    Cr = d["C"][have][:, cols_store] > 0
    wet_r = Cr & (Yr > 0)
    z_grid = np.linspace(h0.min() - 0.3, h0.max() + 0.3, CFG["z_puntos"])
    print(f"population: {Yr.shape[0]} scenes x {Yr.shape[1]:,} px",
          flush=True)

    # ── real archive: the three rungs of the ladder ──────────────────────
    g_hat, curve = gamma_sweep(wet_r, Cr, bank, cols_by_band, z_grid,
                               centers)
    nll_gamma = float(curve.min())
    nll_uniform = float(curve[np.argmin(np.abs(GAMMA_GRID))])
    m2 = json.load(open("results/m2_real/result.json", encoding="utf-8"))
    tau_free = np.array(m2["veredicto"]["m2a_tau"]["real"], float)
    nll_free = nll_of_taus(tau_free, wet_r, Cr, bank, cols_by_band, z_grid)
    print(f"REAL: gamma={g_hat:+.2f} min/km · "
          f"NLL uniform {nll_uniform:.5f} / linear {nll_gamma:.5f} / "
          f"free bands {nll_free:.5f}  ({time.time()-t0:.0f} s)", flush=True)

    # ── the same sweep on the uniform-tide nulls (same seeds as m2_real) ─
    z_true = tpl["z"][rows]
    g_nulls = []
    for i in range(N_NULL):
        Y0, C0 = simulator.simulate(tpl, z_true, h0,
                                    seed=CFG["seed"] + 100 + i,
                                    rising=rising, s_km=s_tpl[rows],
                                    C_real=Cr, template_rows=rows)
        w0 = C0 & (Y0 > 0)
        g0, _ = gamma_sweep(w0, C0, bank, cols_by_band, z_grid, centers)
        g_nulls.append(g0)
        print(f"null {i+1}/{N_NULL}: gamma={g0:+.2f}  "
              f"({time.time()-t0:.0f} s)", flush=True)

    outside = bool(g_hat < min(g_nulls) or g_hat > max(g_nulls))
    # celerity reading: gamma [min/km] -> c [m/s] -> effective depth [m]
    c_ms = (1000.0 / (60.0 * g_hat)) if g_hat > 0 else None
    h_eff = (c_ms ** 2 / 9.81) if c_ms else None

    result = {
        "modelo": "tau(s) = gamma * (s - s0), anclado en la banda bocana",
        "gamma_min_per_km": g_hat,
        "gamma_grid": GAMMA_GRID.tolist(),
        "nll_curve": curve.tolist(),
        "nll": {"uniforme": nll_uniform, "lineal": nll_gamma,
                "bandas_libres": float(nll_free)},
        "gamma_nulos": g_nulls,
        "fuera_del_nulo": outside,
        "celeridad_m_s": c_ms, "profundidad_efectiva_m": h_eff,
        "centros_km": centers.tolist(),
        "tau_lineal_min": (g_hat * (centers - centers[0])).tolist(),
        "tau_bandas_libres_min": tau_free.tolist(),
        "n_px": int(len(rows)), "n_escenas": int(Yr.shape[0]),
        "seed": CFG["seed"],
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
    ax[0].plot(GAMMA_GRID, curve, "-", color="#1a76b3", lw=2.5)
    ax[0].axvline(g_hat, color="#e8a013", lw=2, ls="--",
                  label=f"gamma = {g_hat:+.2f} min/km")
    for g0 in g_nulls:
        ax[0].axvline(g0, color="#9aa39c", lw=1, alpha=0.8)
    ax[0].set_xlabel("gamma  (min/km)")
    ax[0].set_ylabel("profiled NLL")
    ax[0].set_title("the one-parameter bowl (grey: null replicas)")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.25)
    ax[1].plot(centers, tau_free, "o", color="#1a76b3", ms=9,
               label="free bands (M2a)")
    ax[1].plot(centers, g_hat * (centers - centers[0]), "-",
               color="#e8a013", lw=2.5,
               label=f"linear: {g_hat:+.2f} min/km")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("s from the mouth (km)")
    ax[1].set_ylabel("tau (min)")
    ax[1].set_title("what one parameter captures — and what it misses")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"))
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
