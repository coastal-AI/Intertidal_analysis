"""M2 on the REAL archive: is there interior-tide signal, judged vs a null.

After the identifiability gate, the same four estimators run on the real
Villaviciosa archive. The verdict is never a naive p-value (R3): the yardstick
is a NULL BAND built by simulating the archive with a UNIFORM tide (all
mechanisms off — alpha=1, tau=0 everywhere, same pixels, same clouds, same
scene-level jitter) and running the identical estimators on each replica.
Whatever tau/alpha profile the null replicas fabricate is what the machinery
invents on its own; the real archive only carries signal where it leaves that
band.

Prior expectation (declared, not tuned): the 2026-08-18 rank-based prototype
measured lags rising to ~+74 min in the upper estuary, and the flood/ebb
asymmetry suggested ponding. Consistency with that is a check, not a target.

Run:  python -m experiments.m2_real      (~30-40 min: 1 real + N null runs)
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
OUT = os.path.join("results", "m2_real")
N_NULL = 5      # replicas of the uniform-tide null; 5 x 4 estimators bounds
                # the band well enough for a per-band sign verdict and keeps
                # the run under an hour on this machine (7.4 GB RAM)


def run_all(wet, C, Y, bank, rising, band_of, centers, nb, tau_grid, seed):
    mx = CFG["max_px_banda"]
    out = {}
    out["m2d"] = te.m2d_flood_ebb(
        Y.astype(np.float64), C.astype(np.float64), bank, rising, band_of,
        nb, tau_grid, rng=np.random.default_rng(seed + 1),
        max_px_band=mx["m2d"]).tolist()
    out["m2b"] = te.m2b_concordance(
        wet, C, bank, band_of, nb, tau_grid,
        rng=np.random.default_rng(seed + 2),
        max_px_band=mx["m2b"]).tolist()
    out["m2c"] = te.m2c_waterline(
        wet, C, bank, band_of, nb, tau_grid,
        min_cover=CFG["m2c_min_cobertura"]).tolist()
    r = te.m2a_rasch(wet, C, bank, band_of, centers, nb,
                     sigma0=CFG["sigma0_m"],
                     sg_grid=tuple(CFG["sigma_perfil_m"]),
                     alpha_bounds=tuple(CFG["alpha_limites"]),
                     tau_bounds=tuple(CFG["tau_limites_min"]),
                     z_points=CFG["z_puntos"],
                     rng=np.random.default_rng(seed + 3),
                     max_px_band=mx["m2a"])
    out["m2a_tau"] = r["tau"].tolist()
    return out


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

    nb = CFG["n_bandas"]
    tg = CFG["tau_malla_min"]
    tau_grid = np.arange(tg["desde"], tg["hasta"] + tg["paso"], tg["paso"])

    # ── the REAL archive, on the SAME pixel population as the null ───────
    # (the template's good-fit pixels: both runs must see identical pixels,
    # clouds and bands or the comparison is not matched)
    s_tpl = s_all[tpl["idx"]]
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

    Yr = np.nan_to_num(d["Y"][have][:, cols_store], nan=0.0).astype(np.float32)
    Cr = d["C"][have][:, cols_store] > 0
    wet_r = Cr & (Yr > 0)
    print(f"real archive: {Yr.shape[0]} scenes x {Yr.shape[1]:,} px · "
          f"centres {np.round(centers, 2)} km", flush=True)

    real = run_all(wet_r, Cr, Yr, bank, rising, band_of, centers, nb,
                   tau_grid, CFG["seed"])
    print(f"REAL: tau_a={np.round(real['m2a_tau'], 1)}\n"
          f"      tau_b={np.round(real['m2b'], 1)} "
          f"tau_c={np.round(real['m2c'], 1)} "
          f"tau_d={np.round(real['m2d'], 1)} "
          f"({time.time()-t0:.0f} s)", flush=True)

    # ── the uniform-tide null band (R3) ──────────────────────────────────
    z_true = tpl["z"][rows]
    nulls = []
    for i in range(N_NULL):
        Y0, C0 = simulator.simulate(tpl, z_true, h0,
                                    seed=CFG["seed"] + 100 + i,
                                    rising=rising, s_km=s_tpl[rows],
                                    C_real=Cr, template_rows=rows)
        w0 = C0 & (Y0 > 0)
        nulls.append(run_all(w0, C0, Y0, bank, rising, band_of, centers,
                             nb, tau_grid, CFG["seed"] + 100 + i))
        print(f"null {i+1}/{N_NULL}: "
              f"tau_a={np.round(nulls[-1]['m2a_tau'], 1)} "
              f"({time.time()-t0:.0f} s)", flush=True)

    # per band and estimator: the null band is [min, max] over replicas
    def band_of_nulls(key):
        arr = np.array([n[key] for n in nulls], float)  # (N, nb)
        return np.nanmin(arr, 0).tolist(), np.nanmax(arr, 0).tolist()

    verdict = {}
    for key, rv in (("m2a_tau", real["m2a_tau"]),
                    ("m2b", real["m2b"]), ("m2c", real["m2c"]),
                    ("m2d", real["m2d"])):
        lo, hi = band_of_nulls(key)
        outside = [bool(np.isfinite(v) and (v < l or v > h))
                   for v, l, h in zip(rv, lo, hi)]
        verdict[key] = {"real": rv, "nulo_min": lo, "nulo_max": hi,
                        "fuera_del_nulo": outside}

    result = {
        "centros_km": centers.tolist(),
        "veredicto": verdict,
        "n_nulos": N_NULL,
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
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, c, lab in (("m2a_tau", "C0", "M2a"), ("m2b", "C2", "M2b"),
                        ("m2c", "C4", "M2c"), ("m2d", "C3", "M2d")):
        v = verdict[key]
        ax[0].plot(centers, v["real"], "o-", color=c, label=lab)
        ax[0].fill_between(centers, v["nulo_min"], v["nulo_max"],
                           color=c, alpha=0.12)
    ax[0].set_xlabel("s from the mouth (km)")
    ax[0].set_ylabel("lag τ (min)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("real archive (shaded bands = uniform null)")
    # right panel: how many estimators agree outside the null, per band —
    # the plain-sight summary of where the interior tide is real
    counts = np.sum([verdict[k]["fuera_del_nulo"]
                     for k in ("m2a_tau", "m2b", "m2c", "m2d")], axis=0)
    ax[1].bar(centers, counts, width=0.35, color="C0")
    ax[1].set_xlabel("s from the mouth (km)")
    ax[1].set_ylabel("estimators outside the null (of 4)")
    ax[1].set_ylim(0, 4.2)
    ax[1].set_title("agreement between estimators")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
