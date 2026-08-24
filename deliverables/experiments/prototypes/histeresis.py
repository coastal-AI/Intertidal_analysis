"""Is the upstream lag propagation or ponding? The two limbs decide.

The lag profile is real — three sign-consistent asymmetries said so — but a
constant time shift failed as a correction, and one explanation fits that
failure exactly: HYSTERESIS. A pond behind a sill fills late on the flood and
then does not drain on the ebb. A time shift cannot represent that, because a
shift moves both limbs of the tide together while ponding separates them.

The two mechanisms make opposite predictions, and the archive can tell them
apart with machinery already built:

    pure propagation   the wave arrives tau late, floods tau late and ebbs
                       tau late: THE SAME lag on both limbs;
    ponding            water arrives late but leaves much later, or never:
                       the ebb lag is MUCH larger than the flood lag.

So: split the scenes by the tide's direction at the overpass instant, and
estimate the lag separately on each limb, per channel band.

Splitting by limb weakens the estimator, and that has to be measured rather
than hoped away. With both limbs mixed, a shift moves rising and falling
scenes in opposite directions, which is what makes the rank order sensitive;
within one limb the sensitivity comes only from the spread of dh/dt across
scenes. The synthetic calibration is therefore run PER LIMB, on that limb's
own scenes, and every lag is reported with the precision the calibration
demonstrates — not with a nominal one.

If the ebb lag comes out far above the flood lag upstream, three independent
lines point at the same unmodelled physics: the 101 RTK points never seen wet
in nine years, the connectivity fill threshold (a 0.30 m pit that floods at
1.50 m), and this.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
from scipy import stats

from pyintertidal.net import use_system_certificates
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
TAU_GRID = np.arange(-90, 121, 5)     # ebb lag may exceed 90: leave headroom
MIN_CLEAR = 0.30
N_BOOT = 300
N_SYNTH = 40
N_BANDS = 6


def wet_fraction(Y, C):
    obs = C > 0
    wet = obs & (Y > 0)
    n = obs.sum(axis=1)
    return (np.where(n > 0, wet.sum(axis=1) / np.maximum(n, 1), np.nan),
            n / Y.shape[1])


def lag_for(frac, h_cols):
    rho = np.array([stats.spearmanr(frac, h).statistic for h in h_cols])
    j = int(np.nanargmax(rho))
    if 0 < j < len(TAU_GRID) - 1:
        y0, y1, y2 = rho[j - 1], rho[j], rho[j + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        return float(TAU_GRID[j] + np.clip(off, -1, 1) * 5.0), rho
    return float(TAU_GRID[j]), rho


def estimate(frac, h_cols, rng, boot=N_BOOT):
    tau, rho = lag_for(frac, h_cols)
    n = len(frac)
    taus = []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        t, _ = lag_for(frac[i], h_cols[:, i])
        taus.append(t)
    return tau, np.percentile(taus, [2.5, 97.5]), np.array(taus)


def calibrate(h0, r_obs, h_cols, n_syn=N_SYNTH):
    """Precision of the estimator on THIS limb's scenes, with no lag."""
    sd_h = np.std(h0)
    sd_n = sd_h * np.sqrt(max(1.0 / max(r_obs, 0.05) ** 2 - 1.0, 1e-6))
    out = []
    for k in range(n_syn):
        rng = np.random.default_rng(400 + k)
        f_syn = h0 + rng.normal(0, sd_n, len(h0))
        t, _ = lag_for(f_syn, h_cols)
        out.append(t)
    out = np.array(out)
    return float(out.mean()), float(out.std())


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    dates = np.array([str(s) for s in dates])
    ch = np.load(os.path.join(SC, "canal.npz"))
    s_km = ch["s_keep"].astype(float) / 1000.0

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_all = pd.DatetimeIndex(
        pd.to_datetime([times[s] for s in dates[have]])).tz_localize(None)
    Yh, Ch = Y[have], C[have]

    frac_g, cover = wet_fraction(np.nan_to_num(Yh, nan=-9.0), Ch)
    ok = (cover >= MIN_CLEAR) & np.isfinite(frac_g)
    t_ok = t_all[ok]
    print(f"{int(ok.sum())} escenas utiles", flush=True)

    def tide(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    # limb: sign of dh/dt at the overpass
    dh = (tide(t_ok + pd.Timedelta(minutes=30))
          - tide(t_ok - pd.Timedelta(minutes=30)))
    rising = dh > 0
    print(f"  subiendo {int(rising.sum())} · bajando {int((~rising).sum())}",
          flush=True)

    print("calculando h(t - tau) para el barrido…", flush=True)
    h_by_tau = np.array([tide(t_ok - pd.Timedelta(minutes=float(tv)))
                         for tv in TAU_GRID])
    h0 = h_by_tau[np.searchsorted(TAU_GRID, 0)]

    qs = np.nanquantile(s_km, np.linspace(0, 1, N_BANDS + 1))
    qs[0] -= 1e-9
    regions = [("toda la ria", np.ones(len(s_km), bool))]
    for a, b in list(zip(qs, qs[1:]))[-2:]:      # the two inner bands
        regions.append((f"banda {a:.2f}-{b:.2f} km",
                        (s_km > a) & (s_km <= b) & np.isfinite(s_km)))

    rng = np.random.default_rng(11)
    print(f"\n{'region':22s} {'rama':9s} {'n esc':>6s} {'tau (min)':>10s} "
          f"{'IC 95%':>18s} {'precision':>10s}")
    print("-" * 82)
    out = []
    for rname, sel_px in regions:
        fb, _ = wet_fraction(np.nan_to_num(Yh[:, sel_px], nan=-9.0),
                             Ch[:, sel_px])
        fb = fb[ok]
        res = {"region": rname}
        boots = {}
        for lname, sel_sc in (("subiendo", rising), ("bajando", ~rising)):
            m = sel_sc & np.isfinite(fb)
            if m.sum() < 60:
                print(f"{rname:22s} {lname:9s} {int(m.sum()):6d}   "
                      f"insuficiente")
                continue
            tau, ci, bt = estimate(fb[m], h_by_tau[:, m], rng)
            _, rho = lag_for(fb[m], h_by_tau[:, m])
            r_obs = rho[np.searchsorted(TAU_GRID, 0)]
            bias0, sd0 = calibrate(h0[m], r_obs, h_by_tau[:, m])
            print(f"{rname:22s} {lname:9s} {int(m.sum()):6d} {tau:+10.1f} "
                  f"[{ci[0]:+7.1f},{ci[1]:+7.1f}] {sd0:8.1f} min",
                  flush=True)
            res[lname] = {"tau": tau, "ci": ci.tolist(),
                          "synth_sd": sd0, "synth_bias": bias0,
                          "n": int(m.sum())}
            boots[lname] = bt
        if "subiendo" in res and "bajando" in res:
            db = boots["bajando"][:min(len(boots["bajando"]),
                                       len(boots["subiendo"]))] \
                - boots["subiendo"][:min(len(boots["bajando"]),
                                         len(boots["subiendo"]))]
            dci = np.percentile(db, [2.5, 97.5])
            dtau = res["bajando"]["tau"] - res["subiendo"]["tau"]
            sd_comb = np.hypot(res["bajando"]["synth_sd"],
                               res["subiendo"]["synth_sd"])
            res["dif"] = {"tau": dtau, "ci": dci.tolist(),
                          "synth_sd": float(sd_comb)}
            sig = "SI" if (dci[0] > 0 and dtau > 2 * sd_comb) else "no"
            print(f"{'':22s} {'DIF b-s':9s} {'':6s} {dtau:+10.1f} "
                  f"[{dci[0]:+7.1f},{dci[1]:+7.1f}] {sd_comb:8.1f} min  "
                  f"histeresis: {sig}")
        out.append(res)
        print()

    print("Lectura: si la DIFERENCIA bajante-subida es positiva, mayor que")
    print("2x su precision y con IC sobre cero, el agua se va mas tarde de")
    print("lo que llega — encharcamiento, no propagacion. Si la diferencia")
    print("es ~0 con retardo alto en ambas ramas, es propagacion pura.")

    json.dump(out, open(os.path.join(SC, "histeresis.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
