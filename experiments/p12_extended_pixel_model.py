"""P12 — the six-parameter pixel: drainage + mixing + attenuation, per pixel.

The P11 master curve showed the pixel's full response has four regimes,
and P11b showed why only per-pixel fits survive the bottom-albedo spread:
normalisation requires seeing the pixel in its own reference states. This
experiment extends the per-pixel model from the 4-parameter sigmoid to a
6-parameter three-regime curve, blended by the mixing fraction itself:

    f      = Phi((h - z) / sigma)                      (mixing fraction)
    dry(h) = a + (m0 - a) * exp(-(z - h)+ / L)         (drainage moisture)
    wet(h) = w_inf - (w_inf - w0) * exp(-(h - z)+ * k) (water attenuation)
    NDWI(h) = (1 - f) * dry(h) + f * wet(h)

Per-pixel parameters: a (drained level), z (elevation), sigma (transition
width), L (drainage length, m), w0 (just-submerged level), k (attenuation
rate, 1/m). Global site constants, read from the P11 pooled curve: m0
(saturated bare sediment, NDWI at d->0-) and w_inf (the shallow-water
asymptote at the curve's turning point).

PRE-REGISTERED GATE (decided before running):
  the extended model becomes a stage-1 candidate ONLY if
  (i)  its out-of-sample NDWI residual (fit on the first 65% of scenes,
       scored on the last 35%) improves over the 4-parameter fit for a
       majority of pixels, AND
  (ii) its elevation z does not degrade against LiDAR (centred RMSE and
       slope no worse than the 4-parameter fit on the same pixels).
  L and k maps are reported as exploratory products regardless, with a
  qualitative sanity check: k should localise in/near channels
  (turbidity), L should be spatially coherent (sediment provinces).

Site: Aveiro (LiDAR truth). Run:
  python -m experiments.p12_extended_pixel_model     (~20-30 min)
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
from scipy.optimize import least_squares

from pyintertidal.net import use_system_certificates
from pyintertidal import seal
from pyintertidal.marea import invert_series
import pyintertidal as pit
from experiments.p4_site import extract

OUT = os.path.join("results", "p12_extended_pixel")
CUBE = "ndwi_cube_aveiro_2023-2025.nc"
N_PX = 20000
MIN_OBS, MIN_DRY, MIN_WET = 100, 12, 12
TRAIN_FRAC = 0.65
SEED = 20260827


def model6(theta, h, m0, w_inf):
    a, z, sg, L, w0, k = theta
    f = norm.cdf((h - z) / sg)
    dry = a + (m0 - a) * np.exp(-np.clip(z - h, 0, None) / L)
    wet = w_inf - (w_inf - w0) * np.exp(-np.clip(h - z, 0, None) * k)
    return (1 - f) * dry + f * wet


def model4(theta, h):
    a, b, z, sg = theta
    return a + b * norm.cdf((h - z) / sg)


def fit_px(y, h, m0, w_inf, lo, hi):
    """Both models on the same observations; returns thetas or None."""
    # both models share the same robust starting guesses and the same z
    # bounds: the comparison is between shapes, not initialisations
    a0 = float(np.percentile(y, 10))
    w0_0 = float(np.percentile(y, 90))
    z0 = float(np.median(h))
    th4, th6 = None, None
    try:
        r4 = least_squares(
            lambda t: model4(t, h) - y,
            x0=[a0, max(w0_0 - a0, 0.2), z0, 0.25],
            bounds=([-0.8, 0.05, lo, 0.03], [0.5, 1.6, hi, 0.9]),
            max_nfev=200)
        th4 = r4.x
    except Exception:
        pass
    try:
        r6 = least_squares(
            lambda t: model6(t, h, m0, w_inf) - y,
            x0=[a0, z0, 0.25, 0.4, w0_0, 1.0],
            bounds=([-0.8, lo, 0.03, 0.05, -0.3, 0.05],
                    [0.5, hi, 0.9, 3.0, 1.0, 12.0]),
            max_nfev=300)
        th6 = r6.x
    except Exception:
        pass
    return th4, th6


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    p11 = json.load(open("results/p11_ndwi_depth/result.json",
                         encoding="utf-8"))
    dc = np.asarray(p11["curva"]["d_m"], float)
    mc = np.asarray(p11["curva"]["ndwi_mediana"], float)
    m0 = float(mc[np.argmin(np.abs(dc - (-0.05)))])
    w_inf = float(np.nanmax(mc))
    datum = p11["datum_m"]
    print(f"globals from P11: m0={m0:+.3f}  w_inf={w_inf:+.3f}")

    Z = np.load("results/p5_aveiro/z_scores.npz")
    z_ref = Z["z_ref"] - datum

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
    lo, hi = float(h.min()) - 0.3, float(h.max()) + 0.3
    # chronological split, not random: the OOS judge only ever sees
    # scenes strictly later than everything the fit saw
    n_train = int(TRAIN_FRAC * T)
    tr, te = slice(0, n_train), slice(n_train, T)
    print(f"{T} scenes ({n_train} train / {T-n_train} test) x {P:,} px "
          f"({time.time()-t0:.0f} s)", flush=True)

    fin = np.isfinite(Y) & C
    n_obs = fin.sum(axis=0)
    n_dry = (fin & (Y <= 0)).sum(axis=0)
    n_wet = (fin & (Y > 0)).sum(axis=0)
    elig = (np.isfinite(z_ref) & (n_obs >= MIN_OBS)
            & (n_dry >= MIN_DRY) & (n_wet >= MIN_WET))
    rng = np.random.default_rng(SEED)
    px = rng.choice(np.where(elig)[0], min(N_PX, int(elig.sum())),
                    replace=False)
    print(f"eligible {elig.sum():,} -> fitting {len(px):,} px", flush=True)

    res = {k: np.full(len(px), np.nan) for k in
           ("z4", "z6", "L", "k", "w0", "oos4", "oos6")}
    for i, j in enumerate(px):
        m_tr = fin[tr, j]
        m_te = fin[te, j]
        if m_tr.sum() < 60 or m_te.sum() < 25:
            continue
        y_tr = Y[tr, j][m_tr].astype(float)
        h_tr = h[tr][m_tr]
        y_te = Y[te, j][m_te].astype(float)
        h_te = h[te][m_te]
        th4, th6 = fit_px(y_tr, h_tr, m0, w_inf, lo, hi)
        if th4 is not None:
            res["z4"][i] = th4[2]
            res["oos4"][i] = float(np.sqrt(np.mean(
                (model4(th4, h_te) - y_te) ** 2)))
        if th6 is not None:
            res["z6"][i] = th6[1]
            res["L"][i] = th6[3]
            res["w0"][i] = th6[4]
            res["k"][i] = th6[5]
            res["oos6"][i] = float(np.sqrt(np.mean(
                (model6(th6, h_te, m0, w_inf) - y_te) ** 2)))
        if i % 2000 == 0:
            print(f"  {i}/{len(px)} ({time.time()-t0:.0f} s)", flush=True)

    both = np.isfinite(res["oos4"]) & np.isfinite(res["oos6"])
    frac_better = float((res["oos6"][both] < res["oos4"][both]).mean())
    oos4_med = float(np.median(res["oos4"][both]))
    oos6_med = float(np.median(res["oos6"][both]))

    zt = z_ref[px]
    def score_z(z):
        m = both & np.isfinite(z) & np.isfinite(zt)
        e = z[m] - zt[m]
        e = e - np.median(e)
        return {"rmse_centrado": float(np.sqrt(np.mean(e ** 2))),
                "pendiente": float(np.polyfit(zt[m], z[m], 1)[0]),
                "n": int(m.sum())}
    s4, s6 = score_z(res["z4"]), score_z(res["z6"])

    # pre-registered gate (see docstring): adoption needs BOTH conditions
    gate_i = frac_better > 0.5
    gate_ii = (s6["rmse_centrado"] <= s4["rmse_centrado"] + 0.005
               and s6["pendiente"] >= s4["pendiente"] - 0.01)
    print(f"\nOOS NDWI RMSE: 4-param {oos4_med:.4f} vs 6-param "
          f"{oos6_med:.4f} · 6-param better on {frac_better*100:.1f}% of px")
    print(f"z vs LiDAR: 4-param {s4['rmse_centrado']:.3f}/"
          f"{s4['pendiente']:.3f} · 6-param {s6['rmse_centrado']:.3f}/"
          f"{s6['pendiente']:.3f}")
    print(f"GATE: (i) OOS majority: {gate_i} · (ii) z not degraded: "
          f"{gate_ii} -> {'CANDIDATE' if gate_i and gate_ii else 'NOT ADOPTED'}",
          flush=True)
    print(f"L median {np.nanmedian(res['L']):.2f} m · "
          f"k median {np.nanmedian(res['k']):.2f} 1/m")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "fits.npz"),
                        px=px, **{k: v for k, v in res.items()})
    json.dump({
        "globals": {"m0": m0, "w_inf": w_inf},
        "oos_ndwi_rmse": {"p4": oos4_med, "p6": oos6_med,
                          "frac_p6_better": frac_better},
        "z_vs_lidar": {"p4": s4, "p6": s6},
        "gate": {"oos_majority": bool(gate_i),
                 "z_not_degraded": bool(gate_ii),
                 "PASA": bool(gate_i and gate_ii)},
        "L_median_m": float(np.nanmedian(res["L"])),
        "k_median_1_m": float(np.nanmedian(res["k"])),
        "n_px": int(both.sum()),
        "inputs_sha": {"cube": seal._sha256(CUBE)},
        "duracion_s": round(time.time() - t0, 1),
    }, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
