"""B7: per-pixel uncertainty for the DEM, calibrated in the digital twin.

No published intertidal DEM ships a per-pixel confidence interval. This one
does, and the interval is EARNED, not assumed: plant the template's own
elevations in the calibrated simulator (real clouds, real noise, real level
jitter), refit with the standard estimator, and tabulate the robust spread
of the recovery error by (headroom-to-tide-window, n_obs) — the two axes
the measured bias curve moves along. Each real pixel then carries the
sigma_z of its cell. The bias itself is NOT corrected (R4: bias-curve
inversion stays dead); this is uncertainty reporting, never correction.

Gate (configs/b7.yaml): against dev RTK, the 68 % interval must cover
68 +/- 10 % of real errors once the measured point-vs-pixel sampling term
(0.214 m) is added in quadrature — the framework's own decomposition,
closing the loop.

Run:  python -m experiments.b7_incertidumbre     (~12 min)
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
from pyintertidal import simulator, seal, rtk
from pyintertidal.marea import invert_series
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/b7.yaml", encoding="utf-8"))
CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "b7_incertidumbre")


def refit(Y, C, h):
    """Thin wrapper over the canonical pyintertidal.marea.invert_series."""
    z, _ = invert_series(Y, C, h)
    return z


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
    h0 = model_tides(x=[lon_c], y=[lat_c], time=t_real, model="EOT20",
                     directory="tide_models", crs="EPSG:4326",
                     extrapolate=True, cutoff=np.inf,
                     parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG2["epoca_min"],
                              sd_level=CFG2["sd_nivel_m"])
    P = tpl["n_template"]
    z_true = tpl["z"]
    C_real = (d["C"][have][:, tpl["idx"]] > 0)
    n_obs = C_real.sum(axis=0)
    head = np.minimum(z_true - h0.min(), h0.max() - z_true)

    errs = []
    for i in range(CFG["n_replicas"]):
        Y, C = simulator.simulate(tpl, z_true, h0,
                                  seed=20260819 + i, C_real=C_real,
                                  template_rows=np.arange(P))
        z_hat = refit(Y, C, h0)
        errs.append(z_hat - z_true)
        print(f"replicate {i+1}/{CFG['n_replicas']} "
              f"({time.time()-t0:.0f} s)", flush=True)
    E = np.concatenate(errs)
    HH = np.tile(head, CFG["n_replicas"])
    NN = np.tile(n_obs, CFG["n_replicas"])

    bh = np.asarray(CFG["bins_holgura"], float)
    bn = np.asarray(CFG["bins_nobs"], float)
    ih = np.clip(np.digitize(HH, bh) - 1, 0, len(bh) - 2)
    io = np.clip(np.digitize(NN, bn) - 1, 0, len(bn) - 2)
    sig = np.full((len(bh) - 1, len(bn) - 1), np.nan)
    n_cell = np.zeros_like(sig, int)
    for a in range(sig.shape[0]):
        for b in range(sig.shape[1]):
            e = E[(ih == a) & (io == b)]
            e = e[np.isfinite(e)]
            n_cell[a, b] = len(e)
            if len(e) >= 100:
                sig[a, b] = (1.4826 * np.median(np.abs(e - np.median(e)))
                             if CFG["robusto"] else float(np.std(e)))
    print("sigma_z table (headroom x n_obs):", flush=True)
    print(np.round(sig, 3), flush=True)

    # ── attach to the shipped product ────────────────────────────────────
    B = np.load(CFG["datos"]["bathy"])
    z_prod = B["z_operador"]
    keep = B["keep"]
    head_p = np.minimum(z_prod - h0.min(), h0.max() - z_prod)
    nob_p = (d["C"][have] > 0).sum(axis=0)
    with np.errstate(invalid="ignore"):
        ihp = np.clip(np.digitize(np.nan_to_num(head_p, nan=-99), bh) - 1,
                      0, len(bh) - 2)
        iop = np.clip(np.digitize(nob_p, bn) - 1, 0, len(bn) - 2)
    sigma_px = np.where(np.isfinite(z_prod), sig[ihp, iop],
                        np.nan).astype(np.float32)

    # ── gate: coverage on dev RTK ────────────────────────────────────────
    SH = tuple(int(x) for x in d["shape"])
    rk = rtk.load_rtk()
    flat_rtk = rk["row"].astype(np.int64) * SH[1] + rk["col"]
    pp = np.searchsorted(keep, flat_rtk)
    valid = (pp < len(keep)) & (keep[np.minimum(pp, len(keep) - 1)]
                                 == flat_rtk)
    zi = np.where(valid, z_prod[np.minimum(pp, len(z_prod) - 1)], np.nan)
    si = np.where(valid, sigma_px[np.minimum(pp, len(sigma_px) - 1)],
                  np.nan)
    # v2: the point-vs-pixel sampling term is SPATIAL — the relief inside
    # the pixel (sigma_topo, layer B2) is exactly the expected deviation of
    # a point from the median of its pixel
    st_all = np.load(CFG["datos"]["sigma_decomp"])["sg_topo"]
    se = np.where(valid, st_all[np.minimum(pp, len(st_all) - 1)], np.nan)
    ok = np.isfinite(zi) & np.isfinite(si) & np.isfinite(se)
    e = zi[ok] - rk["elev"][ok]
    e = e - np.median(e)                      # datum, as always
    s_tot = np.sqrt(si[ok] ** 2 + se[ok] ** 2)
    cover = float(np.mean(np.abs(e) <= s_tot))
    obj = CFG["puerta"]["cobertura_objetivo"]
    tol = CFG["puerta"]["tolerancia_cobertura"]
    ok_gate = abs(cover - obj) <= tol
    print(f"68% coverage on dev RTK: {100*cover:.0f} % (target "
          f"{100*obj:.0f} ± {100*tol:.0f}) · n={int(ok.sum())}", flush=True)

    result = {
        "tabla_sigma_z": sig.tolist(), "n_por_celda": n_cell.tolist(),
        "bins_holgura": bh.tolist(), "bins_nobs": bn.tolist(),
        "cobertura_dev": cover, "n_rtk": int(ok.sum()),
        "sigma_z_mediana_producto": float(np.nanmedian(sigma_px)),
        "puerta": {"cobertura_ok": bool(ok_gate), "PASA": bool(ok_gate)},
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "sigma_z.npz"),
                        sigma_z=sigma_px, keep=keep, shape=d["shape"],
                        tabla=sig, bins_holgura=bh, bins_nobs=bn)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    img = np.full(SH[0] * SH[1], np.nan, np.float32)
    img.ravel()[keep] = sigma_px
    im = ax[0].imshow(img.reshape(SH), cmap="magma", vmax=0.6)
    ax[0].set_title("per-pixel σ_z (m) — calibrated in the twin")
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    im2 = ax[1].imshow(sig, aspect="auto", cmap="viridis")
    ax[1].set_xticks(range(len(bn) - 1))
    ax[1].set_xticklabels([f"{int(a)}-{int(b)}" for a, b in zip(bn, bn[1:])])
    ax[1].set_yticks(range(len(bh) - 1))
    ax[1].set_yticklabels([f"{a:g}..{b:g}" for a, b in zip(bh, bh[1:])])
    ax[1].set_xlabel("n_obs")
    ax[1].set_ylabel("headroom (m)")
    ax[1].set_title(f"σ_z table · dev coverage {100*cover:.0f} %")
    plt.colorbar(im2, ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print(f"GATE B7: {'GREEN' if ok_gate else 'RED'}", flush=True)


if __name__ == "__main__":
    main()
