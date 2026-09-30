"""Does the retired bracketing method generalise? Score it where truth is.

The dictionary (pure per-pixel bracketing: midpoint of [max tide seen dry,
min tide seen wet], NDWI threshold 0.1 — replicated faithfully from
intertidal/bathymetry.py) beat plain HSR on Villaviciosa's RTK with
statistical separation. The user's counter-question is the right one: is
that a mouth-zone fluke, or does it hold in other regions?

For each site with an external survey (Tagus, Vadehavet, Aveiro): extract
the series, compute the bracket elevation, and score it against the SAME
truth, on the SAME pixels, with the SAME recipe as the stored p5 runs —
paired with the stored uniform-HSR and MAREA elevations (z_scores.npz,
same deterministic extraction, so pixel alignment is exact).

Run:  python -m experiments.p8_bracket_regions      (~30 min, all sites)
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
from pyintertidal.marea import extract, build_bank
import pyintertidal as pit

SITES = {
    "tejo": "ndwi_cube_tejo_2019_2021.nc",
    "vadehavet": "ndwi_cube_vadehavet_2019_2021.nc",
    "aveiro": "ndwi_cube_aveiro_2023-2025.nc",
}
NDWI_THR = 0.1          # the legacy bracketing threshold, replicated


def main():
    use_system_certificates()
    out_all = {}
    for site, cube in SITES.items():
        t0 = time.time()
        zs_path = f"results/p5_{site}/z_scores.npz"
        if not os.path.exists(zs_path):
            print(f"[{site}] no stored p5 scores, skipping", flush=True)
            continue
        Z = np.load(zs_path)
        ex = extract(cube)
        assert np.array_equal(ex["keep"], Z["keep"]), \
            f"{site}: extraction no longer aligns with stored scores"
        times = pit.overpass.get_overpass_times(
            ex["bbox"], ("2016-01-01", "2025-12-31"), verbose=False)
        have = np.array([s in times for s in ex["dates"]])
        t_real = pd.DatetimeIndex(pd.to_datetime(
            [times[s] for s in ex["dates"][have]])).tz_localize(None)
        lat_c = 0.5 * (ex["bbox"]["south"] + ex["bbox"]["north"])
        lon_c = 0.5 * (ex["bbox"]["west"] + ex["bbox"]["east"])
        bank, _ = build_bank(lat_c, lon_c, t_real)
        h0 = bank.at(0.0)

        Y = ex["Y"][have]
        C = ex["C"][have]
        fin = np.isfinite(Y)
        wet = C & fin & (Y > NDWI_THR)
        dry = C & fin & (Y <= NDWI_THR)
        h2 = h0[:, None]
        # bracket: the highest tide ever seen dry bounds z from below, the
        # lowest tide ever seen wet bounds it from above; +/-inf marks
        # pixels never observed on one side (excluded just after)
        zmin = np.where(dry, h2, -np.inf).max(axis=0)
        zmax = np.where(wet, h2, np.inf).min(axis=0)
        ok_b = np.isfinite(zmin) & np.isfinite(zmax) \
            & (zmin > -1e30) & (zmax < 1e30)
        z_dic = np.where(ok_b, 0.5 * (zmin + zmax), np.nan)

        z_ref, z_uni, z_op = Z["z_ref"], Z["z_uni"], Z["z_op"]
        s_km = Z["s_km"]
        com = (np.isfinite(z_ref) & np.isfinite(z_uni) & np.isfinite(z_op)
               & np.isfinite(z_dic) & np.isfinite(s_km))
        res = {}
        for name, zz in (("bracket", z_dic), ("uniforme", z_uni),
                         ("operador", z_op)):
            e = zz[com] - z_ref[com]
            e = e - np.median(e)
            res[name] = {
                "rmse_centrado": float(np.sqrt(np.mean(e ** 2))),
                "pendiente": float(np.polyfit(z_ref[com], zz[com], 1)[0]),
                "pearson": float(np.corrcoef(z_ref[com], zz[com])[0, 1]),
            }
        out_all[site] = {"n_px_comunes": int(com.sum()), **res}
        print(f"[{site}] n={int(com.sum()):,} "
              f"({time.time()-t0:.0f} s)", flush=True)
        for k, v in res.items():
            print(f"   {k:9s} RMSE={v['rmse_centrado']:.3f} "
                  f"pendiente={v['pendiente']:.3f} r={v['pearson']:.3f}",
                  flush=True)

    os.makedirs("results/p8_bracket", exist_ok=True)
    json.dump({"sitios": out_all, "umbral_ndwi": NDWI_THR,
               "inputs_sha": {s: seal._sha256(c)
                              for s, c in SITES.items()
                              if os.path.exists(c)}},
              open("results/p8_bracket/result.json", "w"), indent=1)
    print("written results/p8_bracket", flush=True)


if __name__ == "__main__":
    main()
