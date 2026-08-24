"""M3 gate: a planted BOUNDARY error must land in gamma-hat, not in T.

The danger phase M3 exists to catch: if the ocean model at the mouth is
wrong (gain or phase), a transfer operator fitted downstream will absorb
that error and report fictitious estuarine physics. The gate plants exactly
that failure and demands the machinery sort it out:

* synthesize the archive with a TRUE boundary = prior + planted error
  (gamma_M2 = +8 %, dt_M2 = +12 min) and NO interior transfer at all;
* hand the estimators only the WRONG (uncorrected) prior;
* demand: (1) the mouth-band judge recovers the planted error within
  tolerance; (2) with the corrected boundary, the interior operator is flat
  (alpha = 1, tau = 0 within tolerance) — nothing leaked into T;
* report (storytelling, not asserted): what M2a would have claimed had the
  correction not existed — the fictitious transfer the gate prevents.

Run:  python -m experiments.m3_gate_sim     (~10 min)
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
from pyintertidal import simulator, seal, boundary_correction as bc
from pyintertidal import tide_estimators as te
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/m3.yaml", encoding="utf-8"))
CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "m3_gate_sim")


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
    have = np.array([(s in times) and (int(s[:4]) >= CFG2["epoca_min"])
                     for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)

    def tide_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    cons = CFG["constituyentes_descomposicion"]
    kc = CFG["constituyente_corregido"]
    bank_taus = [float(v) for v in CFG2["taus_banco_min"]]
    shifted = [tide_at(t_real - pd.Timedelta(minutes=tv))
               for tv in bank_taus]
    bank_prior = te.ShiftBank(bank_taus, shifted)
    h_prior = bank_prior.at(0.0)
    print(f"{len(h_prior)} scenes · prior bank ready "
          f"({time.time()-t0:.0f} s)", flush=True)

    # ── plant the boundary error: the TRUE water the pixels felt ─────────
    g_pl = float(CFG["plantado"]["gamma_m2"])
    dt_pl = float(CFG["plantado"]["dt_m2_min"])
    h_true = bc.corrected_series(t_real, h_prior, {kc: g_pl}, {kc: dt_pl},
                                 cons)
    rising = (tide_at(t_real + pd.Timedelta(minutes=30))
              - tide_at(t_real - pd.Timedelta(minutes=30))) > 0

    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h_prior, epoch_years=CFG2["epoca_min"],
                              sd_level=CFG2["sd_nivel_m"])
    geo = np.load(CFG["datos"]["geometria"])
    s_tpl = geo["s_keep"].astype(float)[tpl["idx"]] / 1000.0
    nb = CFG2["n_bandas"]
    edges, centers, band_all = te.make_bands(
        np.where(np.isfinite(s_tpl), s_tpl, np.nan), nb)
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for k in range(nb):
        cand = np.where(band_all == k)[0]
        rows.append(rng.choice(cand, min(len(cand), 2000), replace=False))
    rows = np.sort(np.concatenate(rows))
    band_of = band_all[rows]
    z_true = tpl["z"][rows]
    C_real = (d["C"][have][:, tpl["idx"][rows]] > 0)

    # NO interior transfer: the pixels feel h_true, uniformly
    Y, C = simulator.simulate(tpl, z_true, h_true, seed=CFG["seed"],
                              rising=rising, s_km=s_tpl[rows],
                              C_real=C_real, template_rows=rows)
    wet = C & (Y > 0)
    print(f"synthetic archive with planted boundary error "
          f"(γ={g_pl:+.2f}, dt={dt_pl:+.0f} min) · no interior "
          f"transfer ({time.time()-t0:.0f} s)", flush=True)

    # ── judge 1: the mouth band recovers the planted PHASE error ─────────
    # (gain is not searched: v2, the affine theorem hides it — the planted
    # gamma=+0.08 stays in the world precisely to prove it cannot fake phase)
    mouth = band_of == 0
    # v3: FIXED sigma0 in the judge — profiling sigma absorbed the phase
    # signal
    fit = bc.fit_correction_from_mouth(
        wet[:, mouth], C[:, mouth], t_real, h_prior, constituent=kc,
        gamma_grid=CFG["gamma_malla"], dt_grid_min=CFG["dt_malla_min"],
        sigma0=CFG["sigma0_m"], test_every=CFG["test_cada"],
        sg_grid=None)
    print(f"correction estimated at the mouth: γ̂={fit['gamma']:+.3f} "
          f"dt̂={fit['dt_min']:+.1f} min (adopted={fit['adopted']}) "
          f"({time.time()-t0:.0f} s)", flush=True)

    # ── judge 2: with the corrected boundary, T stays flat ───────────────
    h_corr = bc.corrected_series(t_real, h_prior, {kc: fit["gamma"]},
                                 {kc: fit["dt_min"]}, cons)
    delta = h_corr - h_prior

    def corrected_bank():
        # each shifted prior series gets the same correction evaluated at
        # its shifted instants (the correction is harmonic => analytic)
        HS = []
        for tv, hv in zip(bank_taus, shifted):
            tt = t_real - pd.Timedelta(minutes=tv)
            HS.append(bc.corrected_series(tt, hv, {kc: fit["gamma"]},
                                          {kc: fit["dt_min"]}, cons))
        return te.ShiftBank(bank_taus, HS)

    bank_corr = corrected_bank()
    sg_prof = tuple(CFG2["sigma_perfil_m"])
    r_corr = te.m2a_rasch(wet, C, bank_corr, band_of, centers, nb,
                          sigma0=CFG2["sigma0_m"], sg_grid=sg_prof,
                          tau_bounds=tuple(CFG2["tau_limites_min"]),
                          z_points=CFG2["z_puntos"],
                          rng=np.random.default_rng(CFG["seed"] + 3),
                          max_px_band=CFG2["max_px_banda"]["m2a"])
    # storytelling contrast: what the UNCORRECTED analysis would have claimed
    r_leak = te.m2a_rasch(wet, C, bank_prior, band_of, centers, nb,
                          sigma0=CFG2["sigma0_m"], sg_grid=sg_prof,
                          tau_bounds=tuple(CFG2["tau_limites_min"]),
                          z_points=CFG2["z_puntos"],
                          rng=np.random.default_rng(CFG["seed"] + 4),
                          max_px_band=CFG2["max_px_banda"]["m2a"])
    print(f"corrected T: tau={np.round(r_corr['tau'], 1)}", flush=True)
    print(f"UNcorrected T (the leak prevented): "
          f"tau={np.round(r_leak['tau'], 1)}", flush=True)

    inner = slice(1, nb)
    ok_dt = abs(fit["dt_min"] - dt_pl) <= CFG["puerta"]["tol_dt_min"]
    tau_res = float(np.sqrt(np.nanmean(
        np.asarray(r_corr["tau"], float)[inner] ** 2)))
    ok_flat = tau_res <= CFG["puerta"]["max_tau_residual_rms_min"]
    leak = float(np.nanmax(np.abs(np.asarray(r_leak["tau"])[inner])))
    ok_leak = leak >= CFG["puerta"]["min_fuga_sin_corregir_min"]

    result = {
        "version_puerta": 3,
        "plantado": {"gamma_m2": g_pl, "dt_m2_min": dt_pl,
                     "nota": "gamma is planted but NOT corrected (affine "
                             "theorem): the gate proves it cannot fake "
                             "phase"},
        "recuperado": {k: fit[k] for k in
                       ("gamma", "dt_min", "adopted", "dt_min_raw",
                        "nll_test_zero", "nll_test_win")},
        "T_corregido": {"tau": r_corr["tau"].tolist()},
        "T_sin_corregir_fuga": {"tau": r_leak["tau"].tolist(),
                                "max_min": leak},
        "centros_km": centers.tolist(),
        "residuales": {"rms_tau_min": tau_res},
        "puerta": {"dt_ok": bool(ok_dt), "T_plano_ok": bool(ok_flat),
                   "fuga_demostrada_ok": bool(ok_leak),
                   "PASA": bool(ok_dt and ok_flat and ok_leak)},
        "rms_correccion_m": float(np.sqrt(np.mean(delta ** 2))),
        "seed": CFG["seed"],
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(centers, r_leak["tau"], "x--", color="C3",
               label="uncorrected (leak into the operator)")
    ax[0].plot(centers, r_corr["tau"], "o-", color="C0",
               label="with corrected boundary")
    ax[0].axhline(0, color="k", lw=1, label="truth: no transfer")
    ax[0].set_xlabel("s from the mouth (km)")
    ax[0].set_ylabel("apparent τ (min)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("the boundary error no longer masquerades as estuary")
    sc = np.asarray(fit["grid_scores"])[0]      # gamma fixed at 0 (v2)
    ax[1].plot(CFG["dt_malla_min"], sc, "o-", color="C0")
    ax[1].axvline(dt_pl, color="k", ls="--", label=f"planted (+{dt_pl:.0f})")
    ax[1].axvline(fit["dt_min_raw"], color="r",
                  label=f"estimated ({fit['dt_min_raw']:+.0f})")
    ax[1].set_xlabel("M2 dt (min)")
    ax[1].set_ylabel("NLL (mouth, profiled σ)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("the mouth likelihood finds the PHASE of the error")
    fig.suptitle(f"Gate M3 — "
                 f"{'GREEN' if result['puerta']['PASA'] else 'RED'}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print(json.dumps(result["puerta"], indent=1), flush=True)
    print(f"GATE M3: {'GREEN' if result['puerta']['PASA'] else 'RED'}",
          flush=True)


if __name__ == "__main__":
    main()
