"""Part IV, generic site: interior phase profile from an on-disk cube.

Usage:  SITE=stmalo python -m experiments.p4_sitio
        SITE=sheerness python -m experiments.p4_sitio

For each site this does, streaming (the cubes are 1-2 GB, the machine has
7.4 GB): (1) intertidal mask and per-pixel NDWI series from the sealed-cube
recipe; (2) the M1 geometry — permanent-water mask, mouth = sea touching the
image border, geodesic distance s through water-occupiable ground (no
hand-drawn axis; same code as Villaviciosa); (3) M2a (the gate-winning
estimator) and M2d (the literature baseline) on quantile bands of s.

These sites have no gauge PAIR, so there is no external gradient truth like
the Scheldt's: the output is EXPLORATORY — the phase profile each estuary's
archive carries, with the mouth as anchor. Scoring against the local gauge
(Sheerness has one) is a later, separate step.

Writes results/p4_<site>/{series.npz, result.json, figure.png}.
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
from pyintertidal.marea import extract as _marea_extract
import pyintertidal as pit

CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))

SITES = {
    "stmalo": {"cube": "ndwi_cube_stmalo_2023-2025_20m.nc", "pixel_m": 20.0},
    "sheerness": {"cube": "ndwi_cube_sheerness_2023-2025_20m.nc",
                  "pixel_m": 20.0},
}


def extract(cube, out_npz=None):
    """Streaming intertidal extraction + M1 geometry from a raw-band cube.

    Thin wrapper over the canonical :func:`pyintertidal.marea.extract`,
    kept because experiments.p5_rmse_aoi imports this exact tuple shape:
    (Y, C, keep, dates, SH, sea, inter, bbox).
    """
    d = _marea_extract(cube)
    print(f"  {len(d['keep']):,} intertidal px, "
          f"{int(d['sea'].sum()):,} sea px", flush=True)
    return (d["Y"], d["C"], d["keep"], d["dates"], tuple(d["shape"]),
            d["sea"], d["inter"], d["bbox"])


def main():
    t0 = time.time()
    site = os.environ["SITE"]
    cfg = SITES[site]
    out_dir = os.path.join("results", f"p4_{site}")
    os.makedirs(out_dir, exist_ok=True)
    use_system_certificates()
    from eo_tides.model import model_tides

    print(f"[{site}] extracting {cfg['cube']}...", flush=True)
    Y, C, keep, dates, SH, sea, inter, bbox = extract(cfg["cube"],
                                                      out_dir)
    H, W = SH
    # M1 geometry: mouth = sea at the border; s through water-occupiable px
    seeds = geometry.mouth_seeds(sea)
    s_m = geometry.along_distance(sea | inter, seeds, cfg["pixel_m"])
    s_km = s_m.ravel()[keep] / 1000.0
    print(f"  s up to {np.nanmax(s_km):.1f} km "
          f"({time.time()-t0:.0f} s)", flush=True)

    times = pit.overpass.get_overpass_times(
        bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    Y, C = Y[have], C[have]
    wet = C & (Y > 0)
    lat_c = 0.5 * (bbox["south"] + bbox["north"])
    lon_c = 0.5 * (bbox["west"] + bbox["east"])

    def tide_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    bank_taus = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0]
    bank = te.ShiftBank(bank_taus, [
        tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
    rising = (tide_at(t_real + pd.Timedelta(minutes=30))
              - tide_at(t_real - pd.Timedelta(minutes=30))) > 0
    print(f"  bank ready, {int(have.sum())} scenes "
          f"({time.time()-t0:.0f} s)", flush=True)

    nb = CFG2["n_bandas"]
    edges, centers, band_of = te.make_bands(s_km, nb)
    tau_grid = np.arange(-20.0, 45.0 + 5.0, 5.0)
    r2a = te.m2a_rasch(wet, C, bank, band_of, centers, nb,
                       sigma0=CFG2["sigma0_m"],
                       sg_grid=tuple(CFG2["sigma_perfil_m"]),
                       tau_bounds=(-30.0, 60.0), z_points=CFG2["z_puntos"],
                       rng=np.random.default_rng(CFG2["seed"] + 11),
                       max_px_band=CFG2["max_px_banda"]["m2a"])
    tau_d = te.m2d_flood_ebb(Y.astype(np.float64), C.astype(np.float64),
                             bank, rising, band_of, nb, tau_grid,
                             rng=np.random.default_rng(CFG2["seed"] + 12),
                             max_px_band=CFG2["max_px_banda"]["m2d"])
    tau_a = np.asarray(r2a["tau"], float)
    print(f"  tau_a={np.round(tau_a, 1)}  tau_d={np.round(tau_d, 1)}",
          flush=True)

    np.savez_compressed(os.path.join(out_dir, "series.npz"),
                        keep=keep, s_km=s_km.astype(np.float32),
                        shape=np.array(SH), dates=dates[have])
    result = {
        "sitio": site, "bbox": bbox,
        "n_escenas": int(have.sum()), "n_px": int(len(keep)),
        "s_max_km": float(np.nanmax(s_km)),
        "centros_km": centers.tolist(),
        "tau_m2a_min": tau_a.tolist(),
        "tau_m2d_min": np.asarray(tau_d, float).tolist(),
        "nota": "EXPLORATORY: without a gauge pair there is no external "
                "truth for the gradient; anchored at the opening to the sea",
        "inputs_sha": {"cube": seal._sha256(cfg["cube"])},
        "duracion_s": round(time.time() - t0, 1),
    }
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"),
              indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    s_show = np.full(H * W, np.nan, np.float32)
    s_show[keep] = s_km
    im = ax[0].imshow(s_show.reshape(H, W), cmap="viridis")
    ax[0].set_title(f"{site}: s from the opening (km)")
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].plot(centers, tau_a, "o-", color="C0", label="M2a")
    ax[1].plot(centers, tau_d, "x--", color="C3", label="M2d")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("s (km)")
    ax[1].set_ylabel("lag vs EOT20 (min)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("interior phase profile (exploratory)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure.png"), dpi=130)
    print(f"written {out_dir} ({time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
