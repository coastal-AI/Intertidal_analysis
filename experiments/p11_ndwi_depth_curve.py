"""P11 — is the CONTINUOUS NDWI a depth gauge? The master curve and the
blind-band test.

Motivation (Santander, 2026-08-27): the lowest central banks were never
seen exposed at the sun-synchronous hour (a measured 21 cm sampling
floor), so wet/dry methods cannot map them. But those banks sit under
shallow clear water — if NDWI rises with water depth in a usable way,
every SUBMERGED observation becomes a sounding, and the intertidal pixels
whose elevation the sigmoid already measures become free calibration
targets (self-calibrated optical bathymetry, no in-situ points).

Site: Aveiro (float LiDAR truth for 333k px, cube on disk, same epoch).

Four questions, four products:
1. MASTER CURVE: pooled median NDWI vs instantaneous depth
   d = tide(t) - z_LiDAR, from -2 m (exposed) to +4 m (submerged):
   shape, monotonicity, saturation depth, spread.
2. THRESHOLD PLAY: exposure-frequency elevation at thresholds
   {-0.1, 0, 0.1, 0.2, 0.3} scored against LiDAR -> bias per threshold
   (what a threshold choice costs in centimetres).
3. EXPOSED BRANCH: does exposed-mud NDWI track elevation (drainage)?
4. BLIND-BAND TEST: pixels never seen dry (the Santander analogue).
   Invert the master curve (built WITHOUT them) per observation:
   d_hat = curve^-1(NDWI); z_hat = median_t [tide(t) - d_hat(t)].
   Validate z_hat against LiDAR on that band alone.

Run:  python -m experiments.p11_ndwi_depth_curve      (~15 min)
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

from pyintertidal.net import use_system_certificates
from pyintertidal import seal
import pyintertidal as pit
from experiments.p4_site import extract

OUT = os.path.join("results", "p11_ndwi_depth")
CUBE = "ndwi_cube_aveiro_2023-2025.nc"
DEPTH_BINS = np.arange(-2.0, 4.0 + 0.1, 0.1)
THRESHOLDS = (-0.1, 0.0, 0.1, 0.2, 0.3)


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    Z = np.load("results/p5_aveiro/z_scores.npz")
    z_ref = Z["z_ref"]

    Y, C, keep, dates, SH, sea, inter, bbox = extract(CUBE)
    assert np.array_equal(keep, Z["keep"])
    times = pit.overpass.get_overpass_times(
        bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    Y = Y[have].astype(np.float32)
    C = (C[have] > 0)
    lat_c = 0.5 * (bbox["south"] + bbox["north"])
    lon_c = 0.5 * (bbox["west"] + bbox["east"])
    h = model_tides(x=[lon_c], y=[lat_c], time=t_real, model="EOT20",
                    directory="tide_models", crs="EPSG:4326",
                    extrapolate=True, cutoff=np.inf,
                    parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)
    T, P = Y.shape
    print(f"{T} scenes x {P:,} px  ({time.time()-t0:.0f} s)", flush=True)

    # z_ref to the tide frame: remove the constant datum offset using the
    # median offset of the well-measured uniform inversion (affine theorem:
    # the constant is not knowable; the SHAPE of the curve is what we study)
    z_uni = Z["z_uni"]
    ok = np.isfinite(z_ref) & np.isfinite(z_uni)
    datum = float(np.median(z_ref[ok] - z_uni[ok]))
    z_t = z_ref - datum              # LiDAR elevation in tide-model frame
    print(f"datum offset removed: {datum:+.2f} m", flush=True)

    fin = np.isfinite(Y) & C & np.isfinite(z_t)[None, :]
    ever_dry = ((Y <= 0) & fin).sum(axis=0)
    use_px = np.isfinite(z_t)

    # ── 1. master curve (built ONLY from sometimes-exposed pixels) ──────
    calib_px = use_px & (ever_dry >= 3)
    blind_px = use_px & (ever_dry == 0)
    print(f"calibration px {calib_px.sum():,} · blind px {blind_px.sum():,}",
          flush=True)
    D = h[:, None] - z_t[None, :]
    med, q25, q75, cnt = [], [], [], []
    m_cal = fin & calib_px[None, :]
    for lo, hi in zip(DEPTH_BINS, DEPTH_BINS[1:]):
        m = m_cal & (D >= lo) & (D < hi)
        v = Y[m]
        if v.size < 200:
            med.append(np.nan); q25.append(np.nan); q75.append(np.nan)
            cnt.append(int(v.size)); continue
        med.append(float(np.median(v)))
        q25.append(float(np.percentile(v, 25)))
        q75.append(float(np.percentile(v, 75)))
        cnt.append(int(v.size))
    med = np.asarray(med)
    centers = 0.5 * (DEPTH_BINS[:-1] + DEPTH_BINS[1:])
    # saturation: first depth after which the curve stops rising materially
    rising = np.where(np.isfinite(med))[0]
    d_sat = None
    for i in rising:
        if centers[i] > 0.3 and np.isfinite(med[i]):
            later = med[(centers > centers[i] + 0.5) & np.isfinite(med)]
            if later.size and (later.max() - med[i]) < 0.02:
                d_sat = float(centers[i]); break
    print(f"master curve built; saturation ~{d_sat} m", flush=True)

    # ── 2. threshold play: frequency-based elevation per threshold ───────
    # elevation estimate = tide quantile at the pixel's exposure frequency
    thr_out = {}
    h_sorted = np.sort(h)
    for thr in THRESHOLDS:
        wet = (Y > thr) & fin
        nobs = fin.sum(axis=0)
        f = np.where(nobs >= 30, wet.sum(axis=0) / np.maximum(nobs, 1),
                     np.nan)
        # z_est = the tide height exceeded with probability f
        z_est = np.interp(1 - f, np.linspace(0, 1, len(h_sorted)), h_sorted)
        m = use_px & calib_px & np.isfinite(f) & (f > 0.05) & (f < 0.95)
        e = z_est[m] - z_t[m]
        e = e - np.median(e)
        thr_out[str(thr)] = {
            "rmse_centrado": float(np.sqrt(np.mean(e ** 2))),
            "pendiente": float(np.polyfit(z_t[m], z_est[m], 1)[0]),
            "n": int(m.sum())}
        print(f"  thr {thr:+.1f}: RMSE {thr_out[str(thr)]['rmse_centrado']:.3f} "
              f"slope {thr_out[str(thr)]['pendiente']:.3f}", flush=True)

    # ── 3. exposed branch: NDWI of exposed mud vs elevation ─────────────
    exp_m = fin & (D < -0.15) & calib_px[None, :]
    px_med = np.full(P, np.nan)
    counts = exp_m.sum(axis=0)
    for j in np.where(calib_px & (counts >= 10))[0]:
        px_med[j] = np.median(Y[exp_m[:, j], j])
    mm = np.isfinite(px_med) & np.isfinite(z_t)
    r_exposed = float(np.corrcoef(z_t[mm], px_med[mm])[0, 1])
    print(f"exposed-mud NDWI vs elevation: r = {r_exposed:.3f} "
          f"({mm.sum():,} px)", flush=True)

    # ── 4. blind-band test ──────────────────────────────────────────────
    # invert the master curve on its monotone rising part
    mono = np.isfinite(med) & (centers > -0.3) & \
        (centers < (d_sat if d_sat else 2.5))
    xc, yc = centers[mono], med[mono]
    order = np.argsort(yc)
    yc_s, xc_s = yc[order], xc[order]
    z_hat = np.full(P, np.nan)
    blind_idx = np.where(blind_px)[0]
    for j in blind_idx:
        m = fin[:, j] & (Y[:, j] > yc_s[0]) & (Y[:, j] < yc_s[-1])
        if m.sum() < 10:
            continue
        d_est = np.interp(Y[m, j], yc_s, xc_s)
        z_hat[j] = float(np.median(h[m] - d_est))
    mb = np.isfinite(z_hat) & np.isfinite(z_t)
    e = z_hat[mb] - z_t[mb]
    bias = float(np.median(e))
    e = e - bias
    blind = {"n": int(mb.sum()),
             "rmse_centrado": float(np.sqrt(np.mean(e ** 2))),
             "pendiente": float(np.polyfit(z_t[mb], z_hat[mb], 1)[0])
             if mb.sum() > 100 else None,
             "sesgo_m": bias}
    print(f"BLIND BAND: n={blind['n']:,} RMSE={blind['rmse_centrado']:.3f} "
          f"slope={blind['pendiente']} bias={bias:+.2f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    json.dump({
        "curva": {"d_m": centers.tolist(), "ndwi_mediana": med.tolist(),
                  "q25": q25, "q75": q75, "n": cnt},
        "saturacion_m": d_sat,
        "umbrales": thr_out,
        "r_emergido": r_exposed,
        "banda_ciega": blind,
        "datum_m": datum,
        "n_px_calibracion": int(calib_px.sum()),
        "n_px_ciegos": int(blind_px.sum()),
        "inputs_sha": {"cube": seal._sha256(CUBE)},
        "duracion_s": round(time.time() - t0, 1),
    }, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    ax[0].fill_between(centers, q25, q75, color="#7ec8e3", alpha=0.4)
    ax[0].plot(centers, med, "-", color="#1a76b3", lw=2.5)
    ax[0].axvline(0, color="k", lw=0.8)
    if d_sat:
        ax[0].axvline(d_sat, color="#c94a3a", ls="--", lw=2,
                      label=f"saturation ~{d_sat:.1f} m")
        ax[0].legend(fontsize=9)
    ax[0].set_xlabel("instantaneous depth  d = tide − z  (m)")
    ax[0].set_ylabel("NDWI (median, IQR)")
    ax[0].set_title("the master curve: NDWI vs depth (Aveiro, LiDAR truth)")
    ax[0].grid(alpha=0.25)
    if mb.sum() > 100:
        hb = ax[1].hexbin(z_t[mb], z_hat[mb] - bias, gridsize=45,
                          cmap="viridis", mincnt=3)
        lims = [np.percentile(z_t[mb], 1), np.percentile(z_t[mb], 99)]
        ax[1].plot(lims, lims, "r--", lw=1.5)
        ax[1].set_xlabel("LiDAR z (tide frame, m)")
        ax[1].set_ylabel("z from inverted NDWI curve (m)")
        ax[1].set_title(f"blind band (never seen dry): "
                        f"RMSE {blind['rmse_centrado']:.2f} m, "
                        f"slope {blind['pendiente']:.2f}")
        fig.colorbar(hb, ax=ax[1], label="px")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"))
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
