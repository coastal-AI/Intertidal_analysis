"""E1 — cubes for the estuaries that carry two or more IOC tide gauges.

The Scheldt comparison needs company: a method that measures the interior
clock must be judged wherever an inner gauge exists. Active IOC pairs with
real intertidal flats between them, 2023-2025:

  ems       Borkum (bork, mouth) -> Delfzijl (delf, 28 km up the Ems); the
            Dollard flats. Judge: delf.
  wadden    Terschelling Noordzee (ters, sea side) -> Harlingen (harl,
            behind the Vlie flats, 28 km). Judge: harl.
  ferrol    Ferrol1 (fer1, outer ria) -> Ferrol2 (fer2, inner, 6.4 km); a
            Cantabrian-type ria like Villaviciosa, small flats. Judge: fer2.

Run:  python -m experiments.e1_download_gauge_sites [site ...]
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PERIOD = ("2023-01-01", "2025-12-31")
SITES = {
    "ems": dict(west=6.80, south=53.22, east=7.30, north=53.50, res=20),
    "wadden": dict(west=5.15, south=53.15, east=5.55, north=53.45, res=20),
    "ferrol": dict(west=-8.36, south=43.44, east=-8.15, north=43.52, res=10),
}


def main(keys):
    import truststore
    truststore.inject_into_ssl()
    from pyintertidal.aoi import AOI
    from pyintertidal.cube import SentinelCube
    from pyintertidal.scenes import connect

    conn = connect()
    for k in keys:
        s = SITES[k]
        aoi = AOI.from_bbox(west=s["west"], south=s["south"], east=s["east"],
                            north=s["north"], name=k)
        path = f"ndwi_cube_{k}_2023-2025_{s['res']}m.nc"
        print(f"[{k}] {aoi.area_km2:.0f} km2 @ {s['res']} m -> {path}", flush=True)
        t0 = time.time()
        cube = SentinelCube(aoi, PERIOD, water="ndwi", resolution=s["res"],
                            cache_path=path)
        cube.ensure(conn)
        print(f"[{k}] done in {(time.time() - t0) / 60:.0f} min: "
              f"{len(cube.dates)} scenes, shape {cube.shape}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SITES))
