"""M2 identifiability gate: plant a KNOWN interior tide, demand its recovery.

The tribunal question of phase M2 (plan v4): if the interior tide really is

    h(s, t) = alpha(s) * h_mouth(t - tau(s)),   alpha: 1.0 -> 1.3,
                                                tau:   0  -> 40 min,

can the estimators read it back FROM WET/DRY IMAGERY ALONE — with the real
temporal sampling (the real overpass instants of 2023-2025), the real cloud
masks of the very pixels sampled, the archive's own per-pixel (a, b, sigma,
noise) resampled whole, and the measured scene-level tide error re-injected?

Gate criteria v2 (configs/m2.yaml; pre-registered in PHASE_LOG after the v1
run came back RED and the diagnosis landed on the project's own affine
theorem — binary data cannot see gain):
  * PHASE: the best new estimator (M2a/b/c) beats the literature baseline
    M2d (Granadeiro flood/ebb discrepancy) in RMSE against the planted lag
    profile (unchanged from v1);
  * AMPLITUDE: the machinery must EXPOSE the degeneracy, not invent a
    number — NLL(alpha) with (z, sigma) profiled per pixel must be flat
    within tolerance. Amplitude estimation lives where the OOS judge on
    CONTINUOUS NDWI already validated it (metodo b2 / plan v3).
If the gate fails IN SIMULATION the plan stops and reports (spec: pivot).

Truth targets are the per-band MEANS of the planted profiles over the very
pixels sampled — the estimand a band-wise estimator actually addresses.

Run:  python -m experiments.m2_gate_sim     (~15 min, model calls + fits)
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
from pyintertidal.simulator import Mechanism
from pyintertidal import tide_estimators as te
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "m2_gate_sim")


class PlantedTransfer(Mechanism):
    """The truth being planted: per-pixel gain and lag over the shift bank.

    Receives the jittered boundary (tide + scene-level error) and applies the
    SAME warp the estimators assume — the gate tests recovery, not model
    mismatch (mismatch robustness is a separate, later question).
    """

    name = "planted_transfer"

    def __init__(self, bank, tide0, alpha_px, tau_px):
        self.bank = bank
        self.tide0 = np.asarray(tide0, float)
        self.alpha = np.asarray(alpha_px, float)
        # quantise lags to 2 min so equal-lag pixels share one series;
        # 2 min << the 15 min bank step already being interpolated
        self.tau = np.round(np.asarray(tau_px, float) / 2.0) * 2.0

    def level(self, h, s_km, rising):
        jit = h - self.tide0                       # the scene-level error
        T, P = len(h), len(self.alpha)
        out = np.empty((T, P))
        for tv in np.unique(self.tau):
            cols = self.tau == tv
            out[:, cols] = (self.bank.at(tv) + jit)[:, None] \
                * self.alpha[cols][None, :]
        return out


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    # ── real overpass instants and the shift bank ────────────────────────
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
    print(f"{int(have.sum())} scenes with real overpass hour "
          f"({dates[have][0]}..{dates[have][-1]})", flush=True)

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
    # cross-check against the sealed real-hour series (same model, same hours)
    hr = np.load(CFG["datos"]["mareas_hora_real"])
    ds_hr = np.array([str(s) for s in hr["dates"]])
    common = np.isin(dates[have], ds_hr)
    ref = hr["tide"][np.searchsorted(ds_hr, dates[have][common])]
    dmax = float(np.max(np.abs(h0[common] - ref)))
    assert dmax < 0.02, f"bank vs sealed series differ by {dmax:.3f} m"
    rising = (tide_at(t_real + pd.Timedelta(minutes=30))
              - tide_at(t_real - pd.Timedelta(minutes=30))) > 0
    print(f"bank of {len(bank_taus)} shifts ready · "
          f"{int(rising.sum())} rising / {int((~rising).sum())} falling "
          f"({time.time()-t0:.0f} s)", flush=True)

    # ── template + geometry ──────────────────────────────────────────────
    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG["epoca_min"],
                              sd_level=CFG["sd_nivel_m"])
    geo = np.load(CFG["datos"]["geometria"])
    s_km_all = geo["s_keep"].astype(float)[tpl["idx"]] / 1000.0

    fin = np.isfinite(s_km_all)
    nb = CFG["n_bandas"]
    edges, centers, band_all = te.make_bands(np.where(fin, s_km_all, np.nan),
                                             nb)
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for k in range(nb):
        cand = np.where(band_all == k)[0]
        take = min(len(cand), 2000)
        rows.append(rng.choice(cand, take, replace=False))
    rows = np.sort(np.concatenate(rows))
    s_px = s_km_all[rows]
    band_of = band_all[rows]
    z_true = tpl["z"][rows]
    print(f"{len(rows):,} px planted in {nb} bands · centres "
          f"{np.round(centers, 2)} km", flush=True)

    # ── the planted truth ────────────────────────────────────────────────
    a_lo, a_hi = CFG["plantado"]["alpha_boca_a_fondo"]
    t_lo, t_hi = CFG["plantado"]["tau_boca_a_fondo_min"]
    s_ref = float(np.nanmax(s_px))
    frac = np.clip(s_px / s_ref, 0, 1)
    alpha_px = a_lo + (a_hi - a_lo) * frac
    tau_px = t_lo + (t_hi - t_lo) * frac
    alpha_true = np.array([alpha_px[band_of == k].mean() for k in range(nb)])
    tau_true = np.array([tau_px[band_of == k].mean() for k in range(nb)])

    mech = PlantedTransfer(bank, h0, alpha_px, tau_px)
    C_real = (d["C"][have][:, tpl["idx"][rows]] > 0)
    Y, C = simulator.simulate(tpl, z_true, h0, seed=CFG["seed"],
                              mechanisms=(mech,), rising=rising,
                              s_km=s_px, C_real=C_real, template_rows=rows)
    wet = C & (Y > 0)
    print(f"synthetic archive {Y.shape[0]}x{Y.shape[1]:,} ready "
          f"({time.time()-t0:.0f} s)", flush=True)

    # ── the four estimators, blind to the truth ──────────────────────────
    tg = CFG["tau_malla_min"]
    tau_grid = np.arange(tg["desde"], tg["hasta"] + tg["paso"], tg["paso"])
    mx = CFG["max_px_banda"]

    est = {}
    print("M2d (Granadeiro, the bar)...", flush=True)
    est["m2d"] = {"tau": te.m2d_flood_ebb(
        Y.astype(np.float64), C.astype(np.float64), bank, rising, band_of,
        nb, tau_grid, rng=np.random.default_rng(CFG["seed"] + 1),
        max_px_band=mx["m2d"])}
    print(f"  tau_d = {np.round(est['m2d']['tau'], 1)} "
          f"({time.time()-t0:.0f} s)", flush=True)

    print("M2b (per-pixel concordance)...", flush=True)
    est["m2b"] = {"tau": te.m2b_concordance(
        wet, C, bank, band_of, nb, tau_grid,
        rng=np.random.default_rng(CFG["seed"] + 2), max_px_band=mx["m2b"])}
    print(f"  tau_b = {np.round(est['m2b']['tau'], 1)}", flush=True)

    print("M2c (rank-based waterline)...", flush=True)
    est["m2c"] = {"tau": te.m2c_waterline(
        wet, C, bank, band_of, nb, tau_grid,
        min_cover=CFG["m2c_min_cobertura"])}
    print(f"  tau_c = {np.round(est['m2c']['tau'], 1)}", flush=True)

    print("M2a (Rasch/IRT, phase-only, profiled sigma)...", flush=True)
    sg_prof = tuple(CFG["sigma_perfil_m"])
    r2a = te.m2a_rasch(wet, C, bank, band_of, centers, nb,
                       sigma0=CFG["sigma0_m"], sg_grid=sg_prof,
                       alpha_bounds=tuple(CFG["alpha_limites"]),
                       tau_bounds=tuple(CFG["tau_limites_min"]),
                       z_points=CFG["z_puntos"],
                       rng=np.random.default_rng(CFG["seed"] + 3),
                       max_px_band=mx["m2a"], verbose=True)
    est["m2a"] = {"tau": r2a["tau"], "nll": r2a["nll"],
                  "converged": r2a["converged"]}
    print(f"  ({time.time()-t0:.0f} s)", flush=True)

    # amplitude: demonstrate the affine flatness on the uppermost band, at
    # the fitted lag — the degeneracy meter of the v2 gate
    a_grid = np.asarray(CFG["puerta"]["alpha_malla_perfil"], float)
    cols_up = np.where(band_of == nb - 1)[0][:mx["m2a"]]
    h0v = bank.at(0.0)
    z_grid = np.linspace(h0v.min() - 0.3, h0v.max() + 0.3, CFG["z_puntos"])
    prof = te.alpha_nll_profile(wet[:, cols_up], C[:, cols_up], bank,
                                float(r2a["tau"][-1]), a_grid, z_grid,
                                sg_prof)
    nll_range = float(prof.max() - prof.min())
    print(f"  NLL(alpha) profile, upper band: range {nll_range:.5f}",
          flush=True)

    # ── verdict against the planted truth ────────────────────────────────
    # the mouth band is the anchor (alpha=1, tau=0 by construction), so the
    # informative comparison is bands 1..nb-1
    inner = slice(1, nb)

    def phase_rmse(tau_hat):
        e = np.asarray(tau_hat, float)[inner] - tau_true[inner]
        e = e[np.isfinite(e)]
        return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")

    rmse = {k: phase_rmse(v["tau"]) for k, v in est.items()}
    best_new = min(("m2a", "m2b", "m2c"), key=lambda k: rmse[k])
    ok_alpha = bool(nll_range < CFG["puerta"]["max_recorrido_nll_alpha"])
    ok_phase = bool(rmse[best_new] < rmse["m2d"])

    result = {
        "version_puerta": 2,
        "centros_km": centers.tolist(),
        "alpha_true": alpha_true.tolist(),
        "tau_true_min": tau_true.tolist(),
        "estimadores": {k: {kk: (np.asarray(vv).tolist()
                                 if isinstance(vv, np.ndarray) else vv)
                            for kk, vv in v.items()}
                        for k, v in est.items()},
        "perfil_nll_alpha": {"malla": a_grid.tolist(),
                             "nll": prof.tolist(),
                             "recorrido": nll_range},
        "rmse_fase_min": rmse,
        "mejor_nuevo": best_new,
        "puerta": {"alpha_plano_ok": ok_alpha, "fase_ok": ok_phase,
                   "PASA": ok_alpha and ok_phase},
        "n_px": int(len(rows)), "n_escenas": int(Y.shape[0]),
        "seed": CFG["seed"],
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    # ── the storytelling figure: planted vs recovered ────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(centers, tau_true, "k-", lw=2, label="planted")
    style = {"m2a": ("C0", "o", "M2a Rasch"), "m2b": ("C2", "s", "M2b copulas"),
             "m2c": ("C4", "^", "M2c waterline"),
             "m2d": ("C3", "x", "M2d Granadeiro (bar)")}
    for k, (c, m, lab) in style.items():
        ax[0].plot(centers, est[k]["tau"], m + "-", color=c, alpha=0.8,
                   label=f"{lab} (RMSE {rmse[k]:.1f} min)")
    ax[0].set_xlabel("s from the mouth (km)")
    ax[0].set_ylabel("lag τ (min)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("phase: planted vs recovered")
    ax[1].plot(a_grid, prof, "o-", color="C0")
    ax[1].axvline(alpha_true[-1], color="k", lw=1, ls="--",
                  label=f"planted α ({alpha_true[-1]:.2f})")
    ax[1].set_xlabel("imposed gain α")
    ax[1].set_ylabel("mean NLL (upper band, z and σ profiled)")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"the affine theorem, visible: range "
                    f"{nll_range:.4f} (threshold "
                    f"{CFG['puerta']['max_recorrido_nll_alpha']})")
    fig.suptitle(f"Gate M2 — {'GREEN' if result['puerta']['PASA'] else 'RED'}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)

    print(json.dumps({k: result[k] for k in
                      ("rmse_fase_min", "perfil_nll_alpha", "puerta")},
                     indent=1), flush=True)
    print(f"GATE M2 (simulation): "
          f"{'GREEN' if result['puerta']['PASA'] else 'RED — STOP'}",
          flush=True)


if __name__ == "__main__":
    main()
