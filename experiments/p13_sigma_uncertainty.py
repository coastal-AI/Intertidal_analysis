"""P13 — sigma put to work: a-posteriori per-pixel elevation uncertainty.

The user's plan (announced to the tutor): give the fitted transition width
sigma_p a physical explanation and an applicability. Both in one formula —
the Cramer-Rao bound of the sigmoid fit, evaluated a posteriori from each
pixel's own fit:

    sigma_z  =  s_e * sigma_p / ( b_p * sqrt( sum_t phi_t^2 ) )

    s_e      residual NDWI noise of THIS pixel (from its own fit)
    sigma_p  its transition width (relief + level error, the blur)
    b_p      its wet-dry contrast (the signal)
    phi_t    the Gaussian bell at each observation's margin — only
             observations near the crossing carry elevation information,
             so sum(phi^2) is the pixel's effective crossing count
             (the coverage-density link, made explicit)

Physical reading: elevation precision = noise-to-contrast ratio, times the
blur, divided by how often the waterline was actually caught crossing.

VALIDATION (the point of the exercise): against Aveiro LiDAR, the
normalised error u = (z - z_ref)/sigma_z must behave like N(0,1) if the
bars tell the truth: coverage(|u|<1) ~ 68%, coverage(|u|<1.96) ~ 95%,
and sigma_z must RANK the real errors (binned |e| rising with sigma_z).
Declared caveats: the bound omits the tide-model systematic and any
point-vs-pixel yardstick term, so mild under-coverage is expected — the
twin-calibrated product sigma_z (B7) over-covers (83% at 68%); the two
routes should bracket the truth.

Run:  python -m experiments.p13_sigma_uncertainty     (~15 min)
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
from scipy.stats import norm

from pyintertidal.net import use_system_certificates
from pyintertidal import seal
from pyintertidal.elevation import _fit_block
from pyintertidal.marea import SG_GRID
import pyintertidal as pit
from experiments.p4_site import extract

OUT = os.path.join("results", "p13_sigma_uncertainty")
CUBE = "ndwi_cube_aveiro_2023-2025.nc"
CHUNK = 4000


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    p11 = json.load(open("results/p11_ndwi_depth/result.json",
                         encoding="utf-8"))
    datum = p11["datum_m"]
    Z = np.load("results/p5_aveiro/z_scores.npz")
    z_ref = Z["z_ref"] - datum

    Y, C, keep, dates, SH, sea, inter, bbox = extract(CUBE)
    assert np.array_equal(keep, Z["keep"])
    times = pit.overpass.get_overpass_times(
        bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    Y = np.nan_to_num(Y[have], nan=0.0).astype(np.float64)
    Cb = (C[have] > 0)
    C64 = Cb.astype(np.float64)
    lat_c = 0.5 * (bbox["south"] + bbox["north"])
    lon_c = 0.5 * (bbox["west"] + bbox["east"])
    h = model_tides(x=[lon_c], y=[lat_c], time=t_real, model="EOT20",
                    directory="tide_models", crs="EPSG:4326",
                    extrapolate=True, cutoff=np.inf,
                    parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)
    T, P = Y.shape
    lo, hi = float(h.min()), float(h.max())
    grid = np.linspace(lo, hi, 50)
    print(f"{T} scenes x {P:,} px ({time.time()-t0:.0f} s)", flush=True)

    a_all = np.full(P, np.nan)
    b_all = np.full(P, np.nan)
    z_all = np.full(P, np.nan)
    sg_all = np.full(P, np.nan)
    se_all = np.full(P, np.nan)
    sphi2 = np.full(P, np.nan)
    nobs = Cb.sum(axis=0)

    for j0 in range(0, P, CHUNK):
        s = slice(j0, min(j0 + CHUNK, P))
        a, b, mu, sg, _, N = _fit_block(Y[:, s], C64[:, s], h, grid,
                                        SG_GRID)
        ok = (N >= 8) & (b > 0.15) & (b < 2.5) & (np.abs(a) < 2.5)
        a_all[s] = np.where(ok, a, np.nan)
        b_all[s] = np.where(ok, b, np.nan)
        z_all[s] = np.where(ok, mu, np.nan)
        sg_all[s] = np.where(ok, sg, np.nan)
        # residual noise and effective crossing count, per pixel
        arg = (h[:, None] - mu[None, :]) / sg[None, :]
        pred = a[None, :] + b[None, :] * norm.cdf(arg)
        res = np.where(Cb[:, s], Y[:, s] - pred, np.nan)
        se_all[s] = np.where(ok, np.nanstd(res, axis=0), np.nan)
        phi = norm.pdf(arg)
        sphi2[s] = np.where(ok, np.nansum(
            np.where(Cb[:, s], phi ** 2, 0.0), axis=0), np.nan)
        if j0 % 40000 == 0:
            print(f"  {j0}/{P} ({time.time()-t0:.0f} s)", flush=True)

    # the Cramer-Rao bar: noise-to-contrast ratio, times the blur, divided
    # by the effective number of waterline crossings actually observed
    sigma_z = se_all * sg_all / (b_all * np.sqrt(np.maximum(sphi2, 1e-9)))
    valid = (np.isfinite(sigma_z) & np.isfinite(z_all)
             & np.isfinite(z_ref) & (sphi2 > 0.5))
    e = z_all[valid] - z_ref[valid]
    e = e - np.median(e)      # datum removed: the bar claims spread, not offset
    u = e / sigma_z[valid]
    cov68 = float((np.abs(u) < 1.0).mean())
    cov95 = float((np.abs(u) < 1.96).mean())
    from scipy.stats import spearmanr
    rank = float(spearmanr(np.abs(e), sigma_z[valid]).statistic)
    print(f"\nsigma_z median {np.nanmedian(sigma_z[valid]):.3f} m · "
          f"|e| median {np.median(np.abs(e)):.3f} m")
    print(f"coverage: |u|<1 -> {cov68*100:.1f}% (target 68) · "
          f"|u|<1.96 -> {cov95*100:.1f}% (target 95)")
    print(f"does sigma_z RANK the real errors? Spearman(|e|, sigma_z) = "
          f"{rank:.3f} on {int(valid.sum()):,} px", flush=True)

    # binned reliability curve
    qs = np.nanquantile(sigma_z[valid], np.linspace(0, 1, 11))
    bin_sz, bin_e = [], []
    for loq, hiq in zip(qs, qs[1:]):
        m = (sigma_z[valid] >= loq) & (sigma_z[valid] < hiq)
        if m.sum() < 200:
            continue
        bin_sz.append(float(np.median(sigma_z[valid][m])))
        bin_e.append(float(np.sqrt(np.mean(e[m] ** 2))))

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "sigma_z.npz"),
                        keep=keep, sigma_z=sigma_z.astype(np.float32),
                        z=z_all.astype(np.float32),
                        se=se_all.astype(np.float32),
                        sphi2=sphi2.astype(np.float32))
    json.dump({
        "formula": "sigma_z = s_e * sigma_p / (b_p * sqrt(sum phi^2))",
        "n_px": int(valid.sum()),
        "sigma_z_mediana_m": float(np.nanmedian(sigma_z[valid])),
        "abs_error_mediana_m": float(np.median(np.abs(e))),
        "cobertura_68": cov68, "cobertura_95": cov95,
        "spearman_abs_e_vs_sigma_z": rank,
        "fiabilidad_binned": {"sigma_z_m": bin_sz, "rmse_m": bin_e},
        "inputs_sha": {"cube": seal._sha256(CUBE)},
        "duracion_s": round(time.time() - t0, 1),
    }, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    ax[0].hist(np.clip(u, -5, 5), bins=100, density=True, color="#7ec8e3",
               alpha=0.8)
    xx = np.linspace(-5, 5, 200)
    ax[0].plot(xx, norm.pdf(xx), "r-", lw=2, label="N(0,1)")
    ax[0].set_xlabel("normalised error  u = (z − LiDAR)/σ_z")
    ax[0].set_title(f"do the bars tell the truth?\n"
                    f"|u|<1: {cov68*100:.0f}% (target 68) · "
                    f"|u|<1.96: {cov95*100:.0f}% (target 95)")
    ax[0].legend()
    ax[1].plot(bin_sz, bin_e, "o-", color="#1a76b3", lw=2.5, ms=9)
    lim = max(max(bin_sz), max(bin_e)) * 1.05
    ax[1].plot([0, lim], [0, lim], "r--", lw=1.5, label="perfect calibration")
    ax[1].set_xlabel("predicted σ_z (binned median, m)")
    ax[1].set_ylabel("actual RMSE in bin (m)")
    ax[1].set_title(f"does σ_z rank the errors? Spearman {rank:.2f}")
    ax[1].legend()
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"))
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
