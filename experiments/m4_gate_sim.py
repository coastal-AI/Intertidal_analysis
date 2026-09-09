"""M4 gate: the operator T must BITE (recover planted damage) and CONTRACT.

Rebuilds the exact M2-gate synthetic archive (same seed, same config — the
planted truth alpha 1->1.3, tau 0->40 min with real sampling and clouds),
measures the transfer with M2a as the operator would be built in production,
and then asks the three questions that make an operator real:

1. BENEFIT — inverting elevations per band with the operator's level series
   must recover at least half of the planted damage (RMSE of z-hat vs z-true,
   operator vs uniform boundary) in the interior bands;
2. NO HARM — at the mouth, where transfer is identity by construction, the
   operator must not degrade the inversion beyond refit noise;
3. CONTRACTION — feeding M2a the operator's own level series must return
   identity (alpha=1, tau=0 within the M3 leak tolerances): a fixed point,
   not a runaway.

Run:  python -m experiments.m4_gate_sim     (~15 min)
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
from pyintertidal.marea import invert_series
from pyintertidal.operator import InteriorTide
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/m4.yaml", encoding="utf-8"))
CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "m4_gate_sim")


def fit_z(Y, C, h, mu_points, min_obs, min_b):
    """Thin wrapper over the canonical pyintertidal.marea.invert_series."""
    z, _ = invert_series(Y, C, h, mu_points=mu_points, min_obs=min_obs,
                         min_b=min_b)
    return z


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides
    from experiments.m2_gate_sim import PlantedTransfer

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

    bank_taus = [float(v) for v in CFG2["taus_banco_min"]]
    bank = te.ShiftBank(bank_taus, [
        tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
    h0 = bank.at(0.0)
    rising = (tide_at(t_real + pd.Timedelta(minutes=30))
              - tide_at(t_real - pd.Timedelta(minutes=30))) > 0
    print(f"bank ready ({time.time()-t0:.0f} s)", flush=True)

    # ── the exact M2-gate synthetic archive (same seed => same world) ────
    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG2["epoca_min"],
                              sd_level=CFG2["sd_nivel_m"])
    geo = np.load(CFG["datos"]["geometria"])
    s_km_all = geo["s_keep"].astype(float)[tpl["idx"]] / 1000.0
    nb = CFG2["n_bandas"]
    edges, centers, band_all = te.make_bands(
        np.where(np.isfinite(s_km_all), s_km_all, np.nan), nb)
    rng = np.random.default_rng(CFG2["seed"])
    rows = []
    for k in range(nb):
        cand = np.where(band_all == k)[0]
        rows.append(rng.choice(cand, min(len(cand), 2000), replace=False))
    rows = np.sort(np.concatenate(rows))
    s_px = s_km_all[rows]
    band_of = band_all[rows]
    z_true = tpl["z"][rows]
    # phase-only world (configs/m4.yaml): T v1 is judged on what it claims
    # to correct — lags; per-band gain is affine-blind in binary (gate M2 v2)
    t_lo, t_hi = CFG["plantado"]["tau_boca_a_fondo_min"]
    frac = np.clip(s_px / float(np.nanmax(s_px)), 0, 1)
    mech = PlantedTransfer(bank, h0, np.ones_like(s_px),
                           t_lo + (t_hi - t_lo) * frac)
    C_real = (d["C"][have][:, tpl["idx"][rows]] > 0)
    Y, C = simulator.simulate(tpl, z_true, h0, seed=CFG2["seed"],
                              mechanisms=(mech,), rising=rising,
                              s_km=s_px, C_real=C_real, template_rows=rows)
    wet = C & (Y > 0)
    print(f"M2-gate world rebuilt ({time.time()-t0:.0f} s)",
          flush=True)

    # ── build T as production would: M2a measures the transfer ───────────
    sg_prof = tuple(CFG2["sigma_perfil_m"])
    r2a = te.m2a_rasch(wet, C, bank, band_of, centers, nb,
                       sigma0=CFG2["sigma0_m"], sg_grid=sg_prof,
                       tau_bounds=tuple(CFG2["tau_limites_min"]),
                       z_points=CFG2["z_puntos"],
                       rng=np.random.default_rng(CFG2["seed"] + 3),
                       max_px_band=CFG2["max_px_banda"]["m2a"])
    T = InteriorTide(bank, centers, r2a["alpha"], r2a["tau"],
                     meta={"boundary": "EOT20 (the gate's bank)",
                           "correction": "none (world without boundary "
                                         "error)",
                           "source": "experiments/m4_gate_sim"})
    print(T.describe(), flush=True)

    # ── 1+2: elevation inversion, operator vs uniform boundary ───────────
    Y64, C64 = Y.astype(np.float64), C.astype(np.float64)
    rmse_uni, rmse_op = [], []
    for k in range(nb):
        cols = band_of == k
        zt = z_true[cols]
        z_uni = fit_z(Y64[:, cols], C64[:, cols], h0,
                      CFG["mu_puntos"], CFG["min_obs"], CFG["min_b"])
        h_k = T.level(float(centers[k]))
        z_op = fit_z(Y64[:, cols], C64[:, cols], h_k,
                     CFG["mu_puntos"], CFG["min_obs"], CFG["min_b"])
        both = np.isfinite(z_uni) & np.isfinite(z_op)
        rmse_uni.append(float(np.sqrt(np.mean((z_uni[both] - zt[both]) ** 2))))
        rmse_op.append(float(np.sqrt(np.mean((z_op[both] - zt[both]) ** 2))))
        print(f"  band {k} (s={centers[k]:.2f} km, n={int(both.sum())}): "
              f"uniform RMSE {rmse_uni[-1]:.3f} -> operator "
              f"{rmse_op[-1]:.3f} m", flush=True)

    # the oracle: inversion with the TRUE planted level (the damage floor)
    rmse_oracle = []
    for k in range(nb):
        cols = np.where(band_of == k)[0]
        h_true_k = mech.level(h0, None, None)[:, cols[0]]
        z_or = fit_z(Y64[:, cols], C64[:, cols], h_true_k,
                     CFG["mu_puntos"], CFG["min_obs"], CFG["min_b"])
        both = np.isfinite(z_or)
        rmse_oracle.append(float(np.sqrt(np.mean(
            (z_or[both] - z_true[cols][both]) ** 2))))

    # ── 3: contraction — M2a on the operator's own levels ────────────────
    banks_fix = []
    for k in range(nb):
        HS = [T.alpha_at(centers[k]) * bank.at(T.tau_at(centers[k]) + tv)
              for tv in bank_taus]
        banks_fix.append(te.ShiftBank(bank_taus, HS))
    r_fix = te.m2a_rasch(wet, C, banks_fix[0], band_of, centers, nb,
                         sigma0=CFG2["sigma0_m"], sg_grid=sg_prof,
                         tau_bounds=(-60.0, 60.0),
                         z_points=CFG2["z_puntos"],
                         rng=np.random.default_rng(CFG2["seed"] + 5),
                         max_px_band=CFG2["max_px_banda"]["m2a"],
                         bank_by_band=banks_fix)

    inner = slice(1, nb)
    dam = np.asarray(rmse_uni[1:]) - np.asarray(rmse_oracle[1:])
    rec = np.asarray(rmse_uni[1:]) - np.asarray(rmse_op[1:])
    frac_rec = float(rec.sum() / max(dam.sum(), 1e-9))
    ok_bite = frac_rec >= CFG["puerta"]["min_fraccion_dano_recuperado"]
    ok_mouth = (rmse_op[0] - rmse_uni[0]
                ) <= CFG["puerta"]["max_degradacion_boca_m"]
    tau_res = float(np.nanmax(np.abs(np.asarray(r_fix["tau"])[inner])))
    ok_fix = tau_res <= CFG["puerta"]["max_tau_residual_min"]

    result = {
        "centros_km": centers.tolist(),
        "operador": {"alpha": r2a["alpha"].tolist(),
                     "tau_min": r2a["tau"].tolist()},
        "rmse_z": {"uniforme": rmse_uni, "operador": rmse_op,
                   "oraculo_nivel_verdadero": rmse_oracle},
        "fraccion_dano_recuperado": frac_rec,
        "contraccion": {"tau": r_fix["tau"].tolist(),
                        "max_tau_residual_min": tau_res},
        "puerta": {"muerde_ok": bool(ok_bite),
                   "boca_ok": bool(ok_mouth),
                   "contraccion_ok": bool(ok_fix),
                   "PASA": bool(ok_bite and ok_mouth and ok_fix)},
        "seed": CFG2["seed"],
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(centers, rmse_uni, "x--", color="C3",
               label="uniform boundary (the damage)")
    ax[0].plot(centers, rmse_op, "o-", color="C0", label="with operator T")
    ax[0].plot(centers, rmse_oracle, ":", color="k",
               label="oracle (true level)")
    ax[0].set_xlabel("s from the mouth (km)")
    ax[0].set_ylabel("elevation RMSE (m)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"benefit: {100*frac_rec:.0f} % of damage recovered")
    ax[1].plot(centers, r_fix["tau"], "o-", color="C0", label="residual τ")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].axhspan(-CFG["puerta"]["max_tau_residual_min"],
                  CFG["puerta"]["max_tau_residual_min"],
                  color="C0", alpha=0.08, label="tolerance")
    ax[1].set_xlabel("s from the mouth (km)")
    ax[1].set_ylabel("residual τ (min)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("contraction: the operator is a fixed point")
    fig.suptitle(f"Gate M4 — "
                 f"{'GREEN' if result['puerta']['PASA'] else 'RED'}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print(json.dumps({k: result[k] for k in
                      ("fraccion_dano_recuperado", "puerta")}, indent=1),
          flush=True)
    print(f"GATE M4: {'GREEN' if result['puerta']['PASA'] else 'RED'}",
          flush=True)


if __name__ == "__main__":
    main()
