# -*- coding: utf-8 -*-
"""Diagnostic: MAREA's lag fitted on the pixels around a gauge vs on its band.

The lag of MAREA is one number per 3-km band of geodesic distance from the
mouth, constrained to grow inland. At the Wadden it overshoots the gauge
(121 vs 83 min at the Sentinel-2 instants). This scans the SAME profiled
Bernoulli likelihood (tide_estimators._band_nll, sigma profiled on the
archive atoms) over a single lag for pixel sets of growing radius around the
inner gauge, and for the gauge's band, with and without the transition-zone
scene rule. No truth is used: the gauge enters only as a position.

usage: python -m experiments.v11_local_lag [site]     (wadden by default)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pyintertidal as pit  # noqa: E402
from pyintertidal import marea, tide_estimators as te  # noqa: E402
from pyintertidal.boundary import make_boundary  # noqa: E402

SITES = {"wadden": dict(pdir="products_wadden", n_g=1, judge="harl", mouth="north+west", px=20.0),
         "escalda": dict(pdir="products_escalda", n_g=2, judge="trnz", mouth="west", px=20.0),
         "ems": dict(pdir="products_ems", n_g=1, judge="delf", mouth="north+west", px=20.0)}
RADII_KM = (1.0, 2.0, 3.0, 5.0)
TAUS = np.arange(-30, 181, 5.0)
MAX_PX = 1200            # the per-band subsample of m2a_rasch
SEED = 20261006
P = ("2023-01-01", "2025-12-31")


def main(site):
    import rasterio
    import pyproj
    import yaml
    cfg = SITES[site]
    pit.net.use_system_certificates()
    m2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
    sg_grid = tuple(m2["sigma_perfil_m"])
    ex = np.load(f"{cfg['pdir']}/marea_extract.npz", allow_pickle=True)
    keep, (H, W) = ex["keep"], tuple(int(v) for v in ex["shape"])
    dates = np.array([str(d) for d in ex["dates"]])
    over = json.load(open(f"{cfg['pdir']}/overpass_times.json"))
    cloud = json.load(open(f"{cfg['pdir']}/cloud_pct.json"))
    usable = set(pit.filter_dates(cloud, 0.10))
    in_p = np.array([(P[0] <= d <= P[1]) and d in over for d in dates])
    with rasterio.open(f"{cfg['pdir']}/reference_map.tif") as r:
        tr, crs = r.transform, r.crs
    st = {s["Code"]: s for s in json.load(open("data_v4/gauges/ioc_stations.json", encoding="utf-8"))}
    g = st[cfg["judge"]]
    gx, gy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(float(g["Lon"]), float(g["Lat"]))
    rows, cols = keep // W, keep % W
    xs, ys = tr.c + (cols + 0.5) * tr.a, tr.f + (rows + 0.5) * tr.e
    dist_km = np.hypot(xs - gx, ys - gy) / 1000.0
    s_km = ex["s_km"].astype(float)
    edges, centers, band_of = te.make_bands_fixed(np.where(np.isfinite(s_km), s_km, np.nan), 3.0, 500)
    near = np.argsort(dist_km)[:200]
    k_g = int(np.bincount(band_of[near][band_of[near] >= 0]).argmax())
    print(f"{site}: {len(keep):,} px; gauge {cfg['judge']} at s = {np.nanmedian(s_km[near]):.1f} km, band {k_g} "
          f"({edges[k_g]:.1f}-{edges[k_g + 1]:.1f} km); nearest intertidal px {dist_km.min():.2f} km", flush=True)

    bnd = make_boundary(pit.sites.get(site), tide_model="EOT20", n_gauges=cfg["n_g"], period=P,
                        exclude=[cfg["judge"]], verbose=False)
    rng = np.random.default_rng(SEED)
    out = []
    for rec_name, sel in (("all scenes", in_p), ("transition-zone rule", in_p & np.isin(dates, list(usable)))):
        t_real = pd.DatetimeIndex(pd.to_datetime([over[d] for d in dates[sel]])).tz_localize(None)
        bank = te.ShiftBank(marea.BANK_TAUS, [np.asarray(bnd.levels(t_real - pd.Timedelta(minutes=tv)), float)
                                              for tv in marea.BANK_TAUS])
        h0 = bank.at(0.0)
        z_grid = np.linspace(h0.min() - 0.3, h0.max() + 0.3, m2["z_puntos"])
        sets = {f"band {k_g} (MAREA's)": np.flatnonzero(band_of == k_g)}
        for rk in RADII_KM:
            sets[f"within {rk:g} km of the gauge"] = np.flatnonzero(dist_km <= rk)
        for name, idx in sets.items():
            if len(idx) < 50:
                print(f"  [{rec_name}] {name}: {len(idx)} px, skipped")
                continue
            if len(idx) > MAX_PX:
                idx = np.sort(rng.choice(idx, MAX_PX, replace=False))
            Y = np.nan_to_num(ex["Y"][sel][:, idx], nan=0.0)
            C = ex["C"][sel][:, idx].astype(bool)
            wet = C & (Y > 0)
            nll = np.array([te._band_nll(wet, C, bank.at(float(t)), z_grid, m2["sigma0_m"], sg_grid=sg_grid)[0]
                            for t in TAUS])
            k = int(np.argmin(nll))
            # lags whose NLL is within 0.1 % of the minimum: the flatness of the profile
            flat = TAUS[nll <= nll[k] * 1.001]
            out.append({"record": rec_name, "scenes": int(sel.sum()), "pixels": name, "n_px": len(idx),
                        "tau_min": float(TAUS[k]), "flat_lo": float(flat.min()), "flat_hi": float(flat.max()),
                        "nll_curvature": float(np.ptp(nll))})
            print(f"  [{rec_name}, {sel.sum()} scenes] {name:28s} n={len(idx):5d}  lag {TAUS[k]:+5.0f} min  "
                  f"(NLL within 0.1 %: [{flat.min():+.0f}, {flat.max():+.0f}])", flush=True)
    pd.DataFrame(out).to_csv(f"results/scene_rule_study_2026-09-30/local_lag_{site}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wadden")
