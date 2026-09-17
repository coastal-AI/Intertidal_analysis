"""E0 — the slow half of the Scheldt comparison notebook, run once.

`tide_boundary_comparison_escalda.ipynb` compares MAREA and Granadeiro on the
Westerschelde under one boundary (model / N gauges / consensus). Two of its
inputs take hours and belong in a background job, not in a notebook cell:

  products_escalda/marea_extract.npz       the NDWI wet/dry record of the
                                           intertidal pixels (marea.extract)
  products_escalda/granadeiro_extract.npz  NIR at the same pixels + the
                                           sea/land calibration pixels
  products_escalda/granadeiro_products_2023-2025_<tag>.npz
                                           Granadeiro et al. 2021, faithful
                                           recipe, under the SAME boundary
                                           the notebook uses

The boundary is the notebook's: tide model + N nearest gauges, with the
inner gauge (Terneuzen) held out as the judge.

Run:  python -m experiments.e0_escalda_prep [site] [tide_model|none] [n_gauges|none]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SITES = {
    # cube, pixel size, sea edge(s) of the box, held-out judge gauge(s)
    "escalda": dict(cube="ndwi_cube_escalda_2023-2025_20m.nc", pixel_m=20.0,
                    mouth_side="west", judge=["trnz"]),
    "ems": dict(cube="ndwi_cube_ems_2023-2025_20m.nc", pixel_m=20.0,
                mouth_side="north+west", judge=["delf"], px_stride=4),
    "wadden": dict(cube="ndwi_cube_wadden_2023-2025_20m.nc", pixel_m=20.0,
                   mouth_side="north+west", judge=["harl"], px_stride=4),
    "ferrol": dict(cube="ndwi_cube_ferrol_2023-2025_10m.nc", pixel_m=10.0,
                   mouth_side="west", judge=["fer2"]),
}


def boundary_tag(tide_model, n_gauges):
    return f"{(tide_model or 'nomodel').lower()}_g{n_gauges or 0}"


CLOUD_RULE = ("intertidal", 0.20)   # their frame rule leaves 11 rising
                                     # scenes here; declared substitution


def main(site="escalda", tide_model="EOT20", n_gauges=2, n_lag_px=1000):
    import pyintertidal as pit
    from pyintertidal import geometry, marea
    from pyintertidal.boundary import make_boundary
    from experiments import c0_granadeiro_extract as c0
    from experiments import sota_granadeiro as sg
    pit.net.use_system_certificates()
    cfg = SITES[site]
    CUBE, PIXEL_M, MOUTH_SIDE, JUDGE = (cfg["cube"], cfg["pixel_m"],
                                        cfg["mouth_side"], cfg["judge"])
    OUT = f"products_{site}"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    base = os.path.join(OUT, "marea_extract.npz")
    if not os.path.exists(base):
        ex = marea.extract(CUBE)
        seeds = geometry.mouth_seeds(ex["sea"], side=MOUTH_SIDE)
        s_m = geometry.along_distance(ex["sea"] | ex["inter"], seeds, PIXEL_M)
        s_km = (s_m.ravel()[ex["keep"]] / 1000.0).astype(np.float32)
        np.savez_compressed(base, Y=ex["Y"], C=ex["C"], keep=ex["keep"],
                            dates=ex["dates"], shape=np.array(ex["shape"]),
                            sea=ex["sea"], inter=ex["inter"], s_km=s_km,
                            bbox=np.array(ex["bbox"], dtype=object),
                            crs_wkt=np.array(ex["crs_wkt"]))
        print(f"-> {base}: {len(ex['keep']):,} px, {len(ex['dates'])} scenes "
              f"({time.time() - t0:.0f} s)", flush=True)
        del ex
    else:
        print(f"{base} exists", flush=True)

    gran = os.path.join(OUT, "granadeiro_extract.npz")
    if not os.path.exists(gran):
        c0.main(site, base=base, out=gran, cube=CUBE)
    else:
        print(f"{gran} exists", flush=True)

    tag = boundary_tag(tide_model, n_gauges)
    prod = os.path.join(OUT, f"granadeiro_products_2023-2025_{tag}.npz")
    if not os.path.exists(prod):
        aoi = pit.sites.get(site)
        b = make_boundary(aoi, tide_model=tide_model, n_gauges=n_gauges,
                          exclude=JUDGE)
        print(f"boundary: {b.name}", flush=True)
        sg.main(site, n_lag_px=n_lag_px, years=(2023, 2025), boundary=b,
                tag=tag, base=base, gran=gran, out_dir=OUT,
                cloud_rule=CLOUD_RULE, px_stride=cfg.get("px_stride", 1))
    else:
        print(f"{prod} exists", flush=True)
    print(f"E0 done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    site = a[0] if a else "escalda"
    tm = (None if len(a) > 1 and a[1].lower() == "none"
          else (a[1] if len(a) > 1 else "EOT20"))
    ng = (None if len(a) > 2 and a[2].lower() == "none"
          else (int(a[2]) if len(a) > 2 else 2))
    main(site, tm, ng)
