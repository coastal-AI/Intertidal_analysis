"""C2/C3 — what the boundary level does to the elevations.

The SAME pixels, the SAME scenes (the 2023-2025 epoch of the Villaviciosa
extraction), the SAME inversion (`marea.invert_series`); only the water
level assigned to each overpass changes. Judged on the development RTK
blocks (reserved sealed), median-centred, RMSE + slope, common subset.

Variants of the boundary level h(t):
  EOT20                 the ocean model at the site (the pipeline default)
  gauge Gijon           the nearest tide gauge (IOC gij2), demeaned
  consensus of N gauges Gijon + Santander (+ Bilbao) (+ Ferrol): the mean
                        of the demeaned records — the "consensus" question
  attenuation baseline  EOT20 carried inland by EstuaryTransfer.from_geometry
                        (gain and lag growing with distance s) — UNVALIDATED
                        operator, labelled as such, per band of s
  MAREA band clocks     EOT20 read on each band's estimated clock (the
                        product's tau_usado)

Run:  python -m experiments.c2_boundary_variants
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

EPOCH = (2023, 2025)
GAUGE_SETS = [["gij2"], ["gij2", "san2"], ["gij2", "san2", "bil3"],
              ["gij2", "san2", "bil3", "fer2"]]


def main():
    import pandas as pd
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pyintertidal as pit
    from pyintertidal import marea, overpass
    from pyintertidal import tide_estimators as te
    from pyintertidal.boundary import (EnsembleBoundary, GaugeBoundary,
                                       PyTMDBoundary)
    from pyintertidal.estuary import EstuaryTransfer
    from pyintertidal.gauges import load_cached_ioc
    pit.net.use_system_certificates()

    z = np.load("marea_demo_extract.npz", allow_pickle=True)
    Y, C, keep, s_km = z["Y"], z["C"].astype(bool), z["keep"], z["s_km"]
    H, W = (int(v) for v in z["shape"])
    dates, bbox = [str(d) for d in z["dates"]], z["bbox"]
    yr = np.array([int(d[:4]) for d in dates])
    ep = (yr >= EPOCH[0]) & (yr <= EPOCH[1])

    bb = {"west": float(bbox[0]), "south": float(bbox[1]),
          "east": float(bbox[2]), "north": float(bbox[3]), "crs": "EPSG:4326"}
    times = overpass.get_overpass_times(bb, ("2016-01-01", "2025-12-31"),
                                        verbose=False)
    have = ep & np.array([d in times for d in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[d] for d, h in zip(dates, have) if h])).tz_localize(None)
    Ye, Ce = np.nan_to_num(Y[have], nan=0.0).astype(np.float64), C[have].astype(np.float64)
    print(f"{int(have.sum())} epoch scenes with an exact overpass time")

    lat_c = 0.5 * (bb["south"] + bb["north"])
    lon_c = 0.5 * (bb["west"] + bb["east"])
    eot = PyTMDBoundary("EOT20", lat_c, lon_c, directory="tide_models")

    variants = {}
    variants["EOT20 (model)"] = np.asarray(eot.levels(t_real), float)

    # ── gauges, demeaned over their own record ────────────────────────────
    gauges = {}
    for code in ["gij2", "san2", "bil3", "fer2"]:
        df = load_cached_ioc(code)
        if df is None or len(df) < 1000:
            print(f"  gauge {code}: no usable record")
            continue
        df = df[(df["time"] >= f"{EPOCH[0]}-01-01")
                & (df["time"] <= f"{EPOCH[1]}-12-31")]
        lev = df["level_m"].to_numpy(float)
        lev = lev - np.nanmedian(lev)
        gauges[code] = GaugeBoundary(df["time"], lev, name=code)
        cov = np.isfinite(gauges[code].levels(t_real)).mean()
        print(f"  gauge {code}: {len(df):,} samples, covers "
              f"{cov:.0%} of the epoch overpasses")
    for gs in GAUGE_SETS:
        members = [gauges[c] for c in gs if c in gauges]
        if len(members) != len(gs):
            continue
        b = members[0] if len(members) == 1 else EnsembleBoundary(members)
        name = ("gauge " + gs[0] if len(gs) == 1
                else f"consensus of {len(gs)} gauges")
        variants[name] = np.asarray(b.levels(t_real), float)

    # ── attenuation-with-distance baseline (from geometry, UNVALIDATED) ──
    edges_, centers, band_of = te.make_bands(s_km, 6)
    # from_geometry with guessed convergence/depth amplified the tide 2.7x
    # (measured here: RMSE 0.756 m) - the class says UNVALIDATED and means
    # it. The defensible attenuation-with-distance baseline is the operator
    # MEASURED between two gauges on a comparable deep Cantabrian ria
    # (Ferrol, 6.4 km: M2 gain +1.8 %, lag +4.2 min): 0.28 %/km, 0.66 min/km,
    # applied to every constituent alike.
    from pyintertidal.estuary import PERIODS_H
    mu_km = float(np.log(1.018) / 6.4)
    tau_h_km = float(4.2 / 60.0 / 6.4)
    tr = EstuaryTransfer({k: mu_km for k in PERIODS_H},
                         {k: tau_h_km for k in PERIODS_H},
                         name="ferrol-calibrated")
    h_att = np.full((len(t_real), len(keep)), np.nan)
    for k, sc in enumerate(centers):
        cols = np.flatnonzero(band_of == k)
        if not len(cols) or not np.isfinite(sc):
            continue
        lev = np.asarray(tr.levels(eot, max(sc - float(np.nanmin(s_km)), 0.0),
                                   t_real, keep_residual=False), float)
        h_att[:, cols] = lev[:, None]
    variants["attenuation baseline (Ferrol-calibrated)"] = (h_att, band_of)

    # ── MAREA band clocks from the shipped product ────────────────────────
    res_path = "products_villaviciosa_marea/result.json"
    if os.path.exists(res_path):
        res = json.load(open(res_path))
        tau_used = np.asarray(res["tau_usado_min"], float)
        cen = np.asarray(res["centros_km"], float)
        h_m = np.full((len(t_real), len(keep)), np.nan)
        # bands as the product made them (same make_bands on the same s)
        _, _, band_prod = te.make_bands(s_km, len(cen))
        for k in range(len(cen)):
            cols = np.flatnonzero(band_prod == k)
            if not len(cols):
                continue
            lev = np.asarray(eot.levels(
                t_real - pd.Timedelta(minutes=float(tau_used[k]))), float)
            h_m[:, cols] = lev[:, None]
        variants["MAREA band clocks"] = (h_m, band_prod)
        print(f"  MAREA clocks applied: {np.round(tau_used, 0)} min")

    # ── invert every variant on the same record ──────────────────────────
    campo = np.load("products_villaviciosa/campo.npz", allow_pickle=True)
    rr, cc, gnss = campo["row"], campo["col"], campo["gnss"]
    flat = rr * W + cc
    pos = {int(k): i for i, k in enumerate(keep)}
    ki = np.array([pos.get(int(f), -1) for f in flat])
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        tr10 = s.transform
    east = tr10.c + (cc + 0.5) * tr10.a
    north = tr10.f + (rr + 0.5) * tr10.e
    block = (((east - east.min()) // 150.0).astype(int) * 1000
             + ((north - north.min()) // 150.0).astype(int))
    blocks = np.unique(block)
    rng = np.random.default_rng(20260817)
    hold = rng.choice(blocks, max(2, int(round(0.35 * len(blocks)))),
                      replace=False)
    dev = ~np.isin(block, hold) & (ki >= 0)
    print(f"{int(dev.sum())} development RTK pixels (reserved sealed)")

    results = {}
    preds = {}
    for name, h in variants.items():
        if isinstance(h, tuple):
            h2, bands = h
            zc = np.full(len(keep), np.nan)
            for k in np.unique(bands[bands >= 0]):
                cols = np.flatnonzero(bands == k)
                hk = h2[:, cols[0]]
                ok = np.isfinite(hk)
                if ok.sum() < 8:
                    continue
                zk, _ = marea.invert_series(Ye[ok][:, cols], Ce[ok][:, cols],
                                            hk[ok])
                zc[cols] = zk
        else:
            ok = np.isfinite(h)
            zc, _ = marea.invert_series(Ye[ok], Ce[ok], h[ok])
        preds[name] = zc[ki[dev]]
        p = preds[name]; t = gnss[dev]
        m = np.isfinite(p)
        pc, tc = p[m] - np.median(p[m]), t[m] - np.median(t[m])
        results[name] = {"rmse_m": float(np.sqrt(np.mean((pc - tc) ** 2))),
                         "slope": float(np.polyfit(tc, pc, 1)[0]),
                         "n": int(m.sum())}
        print(f"{name:34s} RMSE {results[name]['rmse_m']:.3f}  "
              f"slope {results[name]['slope']:.3f}  n {m.sum()}")

    common = np.ones(int(dev.sum()), bool)
    for p in preds.values():
        common &= np.isfinite(p)
    t = gnss[dev][common]; tc = t - np.median(t)
    print(f"\nCOMMON SUBSET ({int(common.sum())} pixels)")
    common_rows = {}
    for name, p in preds.items():
        pc = p[common] - np.median(p[common])
        common_rows[name] = {"rmse_m": float(np.sqrt(np.mean((pc - tc) ** 2))),
                             "slope": float(np.polyfit(tc, pc, 1)[0])}
        print(f"{name:34s} RMSE {common_rows[name]['rmse_m']:.3f}  "
              f"slope {common_rows[name]['slope']:.3f}")

    os.makedirs("results", exist_ok=True)
    # per-pixel predictions for a paired block bootstrap of the differences
    np.savez_compressed("results/c2_preds.npz", gnss=gnss[dev],
                        block=block[dev], names=np.array(list(preds)),
                        preds=np.vstack([preds[n] for n in preds]))
    json.dump({"epoch": EPOCH, "n_scenes": int(have.sum()),
               "dev_pixels": int(dev.sum()), "rows": results,
               "common": common_rows},
              open("results/c2_boundary_variants.json", "w"), indent=1)

    names = list(common_rows)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=140)
    axs[0].barh(names, [common_rows[n]["rmse_m"] for n in names],
                color="#2a6f97")
    axs[0].set_xlabel("RMSE vs RTK (m), common subset"); axs[0].invert_yaxis()
    axs[1].barh(names, [common_rows[n]["slope"] for n in names],
                color="#b08968")
    axs[1].axvline(1, color="k", lw=0.8, ls=":")
    axs[1].set_xlabel("slope vs RTK (1 = no compression)"); axs[1].invert_yaxis()
    axs[1].set_yticklabels([])
    for ax in axs:
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("Same pixels, same scenes, same inversion — only the "
                 "boundary level changes", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    os.makedirs("_render_qc", exist_ok=True)
    fig.savefig("_render_qc/c2_boundary_variants.png", bbox_inches="tight")
    print("-> results/c2_boundary_variants.json, "
          "_render_qc/c2_boundary_variants.png")


if __name__ == "__main__":
    main()
