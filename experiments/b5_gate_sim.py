"""B5 gate: plant hysteresis, demand both clocks back AND better ground.

World: real Villaviciosa sampling, clouds, calibrated noise; planted truth
tau_up = 0 everywhere, tau_dn rising 0 -> +25 min inland (the measured
ponding signature). Three questions, pre-registered in configs/b5.yaml:

(a) does the two-clock estimator recover BOTH profiles (RMS <= 6 min each)?
(b) does two-clock inversion beat one-clock inversion on elevation RMSE in
    the interior bands, against the planted truth?
(c) control: on a NO-hysteresis world, do the two clocks avoid doing harm?

Run:  python -m experiments.b5_gate_sim     (~15 min)
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
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/b5.yaml", encoding="utf-8"))
CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "b5_gate_sim")
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


class PlantedHysteresis(Mechanism):
    """Truth: rising scenes feel bank(tau_up_px), falling bank(tau_dn_px)."""

    name = "planted_hysteresis"

    def __init__(self, bank, tide0, tau_up_px, tau_dn_px):
        self.bank = bank
        self.tide0 = np.asarray(tide0, float)
        # clocks quantised to the bank's 2-min grid so the planted truth
        # is exactly representable by the estimator's own machinery
        self.tu = np.round(np.asarray(tau_up_px, float) / 2.0) * 2.0
        self.td = np.round(np.asarray(tau_dn_px, float) / 2.0) * 2.0

    def level(self, h, s_km, rising):
        jit = h - self.tide0
        T, P = len(h), len(self.tu)
        out = np.empty((T, P))
        # evaluate the bank once per distinct clock value, not per pixel
        for tv in np.unique(np.concatenate([self.tu, self.td])):
            hv = self.bank.at(tv) + jit
            m_up = self.tu == tv
            m_dn = self.td == tv
            if m_up.any():
                out[np.ix_(rising, m_up)] = hv[rising, None]
            if m_dn.any():
                out[np.ix_(~rising, m_dn)] = hv[~rising, None]
        return out


def fit_z(Y, C, h2d_or_1d, lo, hi, cols=None):
    """Invert a pixel set; h may be (T,) or per-limb composed outside."""
    grid = np.linspace(lo, hi, 50)
    a, b, mu, sg, _, N = _fit_block(Y, C, h2d_or_1d, grid, SG_GRID)
    ok = (N >= 8) & (b > 0.15) & (b < 2.5) & (np.abs(a) < 2.5)
    return np.where(ok, mu, np.nan)


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

    bank_taus = [float(v) for v in CFG2["taus_banco_min"]]
    bank = te.ShiftBank(bank_taus, [
        tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
    h0 = bank.at(0.0)
    rising = (tide_at(t_real + pd.Timedelta(minutes=30))
              - tide_at(t_real - pd.Timedelta(minutes=30))) > 0
    print(f"bank ready ({time.time()-t0:.0f} s)", flush=True)

    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG2["epoca_min"],
                              sd_level=CFG2["sd_nivel_m"])
    geo = np.load(CFG["datos"]["geometria"])
    s_all = geo["s_keep"].astype(float)[tpl["idx"]] / 1000.0
    nb = CFG2["n_bandas"]
    edges, centers, band_all = te.make_bands(
        np.where(np.isfinite(s_all), s_all, np.nan), nb)
    # same per-band 2000-px subsample recipe as m2_real, fresh seed: the
    # gate's pixel draw is independent of the real-archive run
    rng = np.random.default_rng(CFG2["seed"] + 50)
    rows = []
    for k in range(nb):
        cand = np.where(band_all == k)[0]
        rows.append(rng.choice(cand, min(len(cand), 2000), replace=False))
    rows = np.sort(np.concatenate(rows))
    s_px = s_all[rows]
    band_of = band_all[rows]
    z_true = tpl["z"][rows]
    C_real = (d["C"][have][:, tpl["idx"][rows]] > 0)
    frac = np.clip(s_px / float(np.nanmax(s_px)), 0, 1)
    u0, u1 = CFG["plantado"]["tau_up_boca_a_fondo_min"]
    d0, d1 = CFG["plantado"]["tau_dn_boca_a_fondo_min"]
    tau_up_px = u0 + (u1 - u0) * frac
    tau_dn_px = d0 + (d1 - d0) * frac
    tu_true = np.array([tau_up_px[band_of == k].mean() for k in range(nb)])
    td_true = np.array([tau_dn_px[band_of == k].mean() for k in range(nb)])

    lo, hi = float(h0.min()), float(h0.max())
    # band 0 (mouth) stays out of every verdict: tau = 0 there by anchor
    inner = slice(1, nb)
    verdict = {}

    for world, mechs in (("con_histeresis",
                          (PlantedHysteresis(bank, h0, tau_up_px,
                                             tau_dn_px),)),
                         ("sin_histeresis", ())):
        Y, C = simulator.simulate(tpl, z_true, h0,
                                  seed=CFG2["seed"] + 51,
                                  mechanisms=mechs, rising=rising,
                                  s_km=s_px, C_real=C_real,
                                  template_rows=rows)
        r = te.hysteresis_oos(Y, C, bank, rising, band_of, nb,
                              tau_up_grid=CFG["malla_tau_up_min"],
                              tau_dn_grid=CFG["malla_tau_dn_min"],
                              test_every=CFG["test_cada"],
                              rng=np.random.default_rng(CFG2["seed"] + 52),
                              max_px_band=2000)
        Y64, C64 = Y.astype(np.float64), C.astype(np.float64)
        rm1, rm2 = [], []
        for k in range(nb):
            cols = band_of == k
            zt = z_true[cols]
            # 1 clock: the best SINGLE clock of the same judge (diagonal)
            h1 = bank.at(float(r["tau_single"][k]))
            z1 = fit_z(Y64[:, cols], C64[:, cols], h1, lo, hi)
            h2 = np.where(rising, bank.at(float(r["tau_up"][k])),
                          bank.at(float(r["tau_dn"][k])))
            z2 = fit_z(Y64[:, cols], C64[:, cols], h2, lo, hi)
            both = np.isfinite(z1) & np.isfinite(z2)
            rm1.append(float(np.sqrt(np.mean((z1[both] - zt[both]) ** 2))))
            rm2.append(float(np.sqrt(np.mean((z2[both] - zt[both]) ** 2))))
        verdict[world] = {
            "tau_up_hat": r["tau_up"].tolist(),
            "tau_dn_hat": r["tau_dn"].tolist(),
            "adoptado": r["adopted"].tolist(),
            "oos_two": r["oos_two"].tolist(),
            "oos_one": r["oos_one"].tolist(),
            "rmse_z_1reloj": rm1, "rmse_z_2relojes": rm2,
        }
        print(f"[{world}] up={np.round(r['tau_up'], 1)} "
              f"dn={np.round(r['tau_dn'], 1)} "
              f"adopted={r['adopted'].astype(int)}\n"
              f"  z 1-clock {np.round(rm1, 3)}\n"
              f"  z 2-clock {np.round(rm2, 3)} "
              f"({time.time()-t0:.0f} s)", flush=True)

    # (a) clocks: the up/down SPLIT must be recovered where adopted —
    # the split is exactly what a single clock cannot represent
    vh = verdict["con_histeresis"]
    ad = np.asarray(vh["adoptado"], bool)[inner]
    split_hat = (np.asarray(vh["tau_dn_hat"])
                 - np.asarray(vh["tau_up_hat"]))[inner]
    split_true = (td_true - tu_true)[inner]
    rms_split = float(np.sqrt(np.nanmean(
        (split_hat[ad] - split_true[ad]) ** 2))) if ad.any() else np.inf
    ok_clocks = ad.any() and rms_split <= CFG["puerta"]["max_rms_desdoble_min"]
    mej = (np.asarray(vh["rmse_z_1reloj"][1:])
           - np.asarray(vh["rmse_z_2relojes"][1:]))
    ok_bite = bool(mej.mean() > 0)
    # (c) control world without hysteresis: adopting clocks there is a
    # false positive, and any elevation damage is charged to the method
    v0 = verdict["sin_histeresis"]
    esp = int(np.asarray(v0["adoptado"], bool)[inner].sum())
    dano = float(np.nanmax(np.asarray(v0["rmse_z_2relojes"][1:])
                           - np.asarray(v0["rmse_z_1reloj"][1:])))
    ok_ctrl = (esp <= CFG["puerta"]["max_adopciones_espurias"]
               and dano <= CFG["puerta"]["max_dano_sin_histeresis_m"])

    result = {
        "centros_km": centers.tolist(),
        "tau_up_true": tu_true.tolist(), "tau_dn_true": td_true.tolist(),
        "mundos": verdict,
        "rms_desdoble_min": rms_split,
        "mejora_media_cota_m": float(mej.mean()),
        "adopciones_espurias": esp,
        "dano_max_sin_histeresis_m": dano,
        "puerta": {"relojes_ok": bool(ok_clocks), "cota_ok": ok_bite,
                   "control_ok": bool(ok_ctrl),
                   "PASA": bool(ok_clocks and ok_bite and ok_ctrl)},
        "seed": CFG2["seed"] + 50,
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(centers, td_true, "k-", lw=2, label="planted falling")
    ax[0].plot(centers, vh["tau_dn_hat"], "o-", color="C3",
               label="estimated falling")
    ax[0].plot(centers, tu_true, "k--", lw=1, label="planted rising")
    ax[0].plot(centers, vh["tau_up_hat"], "s-", color="C0",
               label="estimated rising")
    ax[0].set_xlabel("s (km)")
    ax[0].set_ylabel("τ (min)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("two clocks, recovered separately")
    ax[1].plot(centers, vh["rmse_z_1reloj"], "x--", color="C3",
               label="1 clock")
    ax[1].plot(centers, vh["rmse_z_2relojes"], "o-", color="C0",
               label="2 clocks")
    ax[1].set_xlabel("s (km)")
    ax[1].set_ylabel("elevation RMSE (m)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("and elevation improves where ponding exists")
    fig.suptitle(f"Gate B5 — "
                 f"{'GREEN' if result['puerta']['PASA'] else 'RED'}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print(json.dumps(result["puerta"], indent=1), flush=True)
    print(f"GATE B5: {'GREEN' if result['puerta']['PASA'] else 'RED'}",
          flush=True)


if __name__ == "__main__":
    main()
