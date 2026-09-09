"""Elevation RMSE per AOI against EXTERNAL truth (EMODnet), with and
without the operator.

Usage:  SITE=tejo python -m experiments.p5_rmse_aoi
        SITE=vadehavet python -m experiments.p5_rmse_aoi

The two sites with float-precision surveys OF THE FLATS themselves (not boat
soundings stopping at the low-water line): the inner Tagus cell
(IB_Tagus_2020_64, 22.6 m grid) and the Danish Wadden cell
(IB_Danske_Vadden_2020_64, 16.5 m). For each:

1. extract the intertidal per-pixel NDWI series from the cube (2019-2021,
   matching the 2020 survey date);
2. invert elevations the standard way with EOT20 at the REAL overpass hour,
   one level per scene (``uniforme`` — what the literature does);
3. measure the interior phase profile with M2a (mouth-anchored at the
   seaward border of the cell) and, wherever |tau| > 5 min (the M2a noise
   level), invert again with the per-band shifted clock (``operador``);
4. score both against the survey: SLOPE (scale-free) and MEDIAN-CENTRED
   RMSE — the two frames differ by a datum constant (LAT vs MSL) that a
   satellite method cannot know (affine theorem), so the honest comparables
   are shape and dispersion, exactly as with the RTK diagnostic.

Caveat stated up front: the Tagus cell sits ~30 km INSIDE the estuary, so
the "mouth" anchor is the cell's seaward edge, and the correction is only
the RELATIVE lag across the cell — the common lag of the whole cell against
EOT20 is invisible from inside and lands in the datum constant.

Writes results/p5_<site>/{result.json, figure.png}.
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
from pyintertidal import geometry, seal
from pyintertidal import tide_estimators as te
from pyintertidal.marea import invert_series
import pyintertidal as pit
from experiments.p4_site import extract

CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
TAU_APLICA_MIN = 5.0     # the demonstrated noise level of M2a (gate M2):
                         # below it, the "lag" cannot be told from zero and
                         # the operator stays at identity

SITES = {
    "tejo": {"cube": "ndwi_cube_tejo_2019_2021.nc", "pixel_m": 10.0,
             "ref": "bathy_tagus/1528_IB_Tagus_2020_64.nc"},
    "vadehavet": {"cube": "ndwi_cube_vadehavet_2019_2021.nc",
                  "pixel_m": 10.0,
                  "ref": "bathy_candidatos/1528_IB_Danske_Vadden_2020_64.nc"},
    "aveiro": {"cube": "ndwi_cube_aveiro_2023-2025.nc", "pixel_m": 10.0,
               "ref": "bathy_candidatos/590_HR_Lidar_Norte.nc"},
}


def invert(Y, C, h, lo, hi):
    """Thin wrapper over the canonical pyintertidal.marea.invert_series."""
    z, _ = invert_series(Y, C, h, lo, hi)
    return z


def main():
    t0 = time.time()
    site = os.environ["SITE"]
    cfg = SITES[site]
    out_dir = os.path.join("results", f"p5_{site}")
    os.makedirs(out_dir, exist_ok=True)
    use_system_certificates()
    from eo_tides.model import model_tides
    import xarray as xr

    print(f"[{site}] extracting {cfg['cube']}...", flush=True)
    Y, C, keep, dates, SH, sea, inter, bbox = extract(cfg["cube"], out_dir)
    H, W = SH
    seeds = geometry.mouth_seeds(sea)
    s_m = geometry.along_distance(sea | inter, seeds, cfg["pixel_m"])
    s_km = s_m.ravel()[keep] / 1000.0
    print(f"  {len(keep):,} px · s up to {np.nanmax(s_km):.1f} km "
          f"({time.time()-t0:.0f} s)", flush=True)

    times = pit.overpass.get_overpass_times(
        bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    Y, C = np.nan_to_num(Y[have], nan=0.0).astype(np.float64), \
        (C[have] > 0).astype(np.float64)
    wet = (C > 0) & (Y > 0)
    lat_c = 0.5 * (bbox["south"] + bbox["north"])
    lon_c = 0.5 * (bbox["west"] + bbox["east"])

    def tide_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    bank_taus = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0,
                 75.0, 90.0, 105.0, 120.0]   # the Tagus saturated the edge
                                             # at +45
    bank = te.ShiftBank(bank_taus, [
        tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
    h0 = bank.at(0.0)
    print(f"  bank ready, {int(have.sum())} scenes "
          f"({time.time()-t0:.0f} s)", flush=True)

    # ── bands: per arm in branched lagoons ───────────────────────────────
    # A single tau(s) profile forces the same clock onto arms with distinct
    # physics (measured in Aveiro: -61 mm in the transit band). Beyond the
    # central hub the lagoon splits into connected components = arms,
    # detected automatically; each arm carries its own bands of s.
    from scipy import ndimage as ndi
    nb = CFG2["n_bandas"]
    PER_ARM = os.environ.get("PER_ARM", "0") == "1"
    # arm mode (PER_ARM=1): tried in Aveiro 2026-08-20 — per-arm clocks
    # plausible (up to +75 min in Ovar) but NO global improvement at 462
    # scenes (0.594->0.597): splitting into 8 groups leaves each clock
    # noisy, the same power limitation as B5. Documented; OFF by default.
    if not PER_ARM:
        edges, centers, band_of = te.make_bands(s_km, nb)
    else:
      s_map = np.full(H * W, np.nan, np.float32)
      s_map[keep] = s_km
      s_map = s_map.reshape(H, W)
      hub_km = float(np.nanquantile(s_km, 0.25))
      lab, nlab = ndi.label(np.isfinite(s_map) & (s_map > hub_km),
                          structure=np.ones((3, 3)))
      arm_of = lab.ravel()[keep]
      sizes = np.bincount(arm_of, minlength=nlab + 1)
      arms = [a for a in range(1, nlab + 1) if sizes[a] >= 20000]
      band_of = np.zeros(len(s_km), int)      # 0 = hub/mouth (anchor, tau=0)
      centers = [0.0]                          # anchor centre set at the end
      nxt = 1
      per_arm = max(2, (nb - 1) // max(len(arms), 1))
      for a in arms:
        m = arm_of == a
        qs = np.nanquantile(s_km[m], np.linspace(0, 1, per_arm + 1))
        qs[0] -= 1e-9
        for lo_, hi_ in zip(qs, qs[1:]):
            mm = m & (s_km > lo_) & (s_km <= hi_)
            if mm.sum() < 2000:
                continue
            band_of[mm] = nxt
            centers.append(float(np.nanmedian(s_km[mm])))
            nxt += 1
      centers[0] = float(np.nanmedian(s_km[band_of == 0]))
      centers = np.asarray(centers)
      nb = nxt
      print(f"  arms detected: {len(arms)} (hub at {hub_km:.1f} km) -> "
          f"{nb} groups (anchor included)", flush=True)
    r2a = te.m2a_rasch(wet, C > 0, bank, band_of, centers, nb,
                       sigma0=CFG2["sigma0_m"],
                       sg_grid=tuple(CFG2["sigma_perfil_m"]),
                       tau_bounds=(-45.0, 120.0), z_points=CFG2["z_puntos"],
                       rng=np.random.default_rng(CFG2["seed"] + 21),
                       max_px_band=CFG2["max_px_banda"]["m2a"])
    tau = np.asarray(r2a["tau"], float)
    tau_used = np.where(np.abs(tau) > TAU_APLICA_MIN, tau, 0.0)
    print(f"  tau_a={np.round(tau, 1)} -> applied "
          f"{np.round(tau_used, 1)}", flush=True)

    lo, hi = float(h0.min()), float(h0.max())
    z_uni = invert(Y, C, h0, lo, hi)
    z_op = z_uni.copy()
    for k in np.where(tau_used != 0)[0]:
        cols = np.where(band_of == k)[0]
        z_op[cols] = invert(Y[:, cols], C[:, cols], bank.at(tau_used[k]),
                            lo, hi)
    print(f"  inversions done ({time.time()-t0:.0f} s)", flush=True)

    # ── sample the survey at each pixel ──────────────────────────────────
    import pyproj
    ds = xr.open_dataset(cfg["cube"])
    crs_cube = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt")
                                   or ds["crs"].attrs.get("spatial_ref"))
    tr = pyproj.Transformer.from_crs(crs_cube, 4326, always_xy=True)
    rows, cols_px = keep // W, keep % W
    lon_px, lat_px = tr.transform(ds["x"].values[cols_px],
                                  ds["y"].values[rows])
    ref = xr.open_dataset(cfg["ref"])
    z_ref = ref["elevation"].interp(
        lon=xr.DataArray(lon_px), lat=xr.DataArray(lat_px),
        method="nearest").values

    both = np.isfinite(z_ref) & np.isfinite(z_uni) & np.isfinite(z_op)
    # hydraulically CONNECTED domain: an enclosure with no navigable path to
    # the mouth (salt pans, gated creeks) has s = NaN — it wets without
    # obeying the tide and must not score a tidal method
    connected = both & np.isfinite(s_km)
    res = {}
    for dom_name, dom in (("todos", both), ("conectados", connected)):
        res[dom_name] = {}
        for name, zz in (("uniforme", z_uni), ("operador", z_op)):
            e = zz[dom] - z_ref[dom]
            res[dom_name][name] = {
                "n": int(dom.sum()),
                "pendiente": float(np.polyfit(z_ref[dom], zz[dom], 1)[0]),
                "rmse_centrado": float(np.sqrt(np.mean(
                    (e - np.median(e)) ** 2))),
                "offset_datum_mediano": float(np.median(e)),
            }
            r = res[dom_name][name]
            print(f"  [{dom_name}/{name}] n={r['n']:,} "
                  f"slope={r['pendiente']:.3f} "
                  f"centred RMSE={r['rmse_centrado']:.3f} m", flush=True)
    np.savez_compressed(os.path.join(out_dir, "z_scores.npz"),
                        z_uni=z_uni.astype(np.float32),
                        z_op=z_op.astype(np.float32),
                        z_ref=z_ref.astype(np.float32),
                        s_km=s_km.astype(np.float32), keep=keep,
                        shape=np.array(SH))
    res = res["conectados"] | {"todos": res["todos"]}

    result = {
        "sitio": site, "n_escenas": int(have.sum()),
        "s_max_km": float(np.nanmax(s_km)),
        "centros_km": centers.tolist(),
        "tau_m2a_min": tau.tolist(), "tau_aplicado_min": tau_used.tolist(),
        "contra_levantamiento": res,
        "inputs_sha": {"cube": seal._sha256(cfg["cube"]),
                       "ref": seal._sha256(cfg["ref"])},
        "duracion_s": round(time.time() - t0, 1),
    }
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"),
              indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    hb = ax[0].hexbin(z_ref[both], z_op[both] -
                      res["operador"]["offset_datum_mediano"],
                      gridsize=60, cmap="viridis", mincnt=1)
    lim = [np.nanpercentile(z_ref[both], 1), np.nanpercentile(z_ref[both], 99)]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set_xlabel("EMODnet survey (m)")
    ax[0].set_ylabel("satellite (operator, datum aligned)")
    ax[0].set_title(f"{site}: slope "
                    f"{res['operador']['pendiente']:.2f}, RMSE "
                    f"{res['operador']['rmse_centrado']:.2f} m")
    plt.colorbar(hb, ax=ax[0], shrink=0.8)
    ax[1].plot(centers, tau, "o-", color="C0", label="τ M2a")
    ax[1].plot(centers, tau_used, "s--", color="C2", label="τ applied")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("s from the seaward edge (km)")
    ax[1].set_ylabel("lag (min)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure.png"), dpi=130)
    print(f"written {out_dir} ({time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
