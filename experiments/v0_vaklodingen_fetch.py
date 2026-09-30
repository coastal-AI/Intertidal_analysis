"""V0 — fetch the Rijkswaterstaat Vaklodingen sheets that cover our Dutch
sites (Westerschelde, Wadden/Vlie, Ems-Dollard).

Vaklodingen: the yearly bathymetry + intertidal topography of the Dutch
estuaries on a 20 m RD grid (EPSG:28992), NAP datum, served by Deltares
OpenEarth (https://opendap.deltares.nl/thredds/.../vaklodingen/). The
catalogue file lists every map sheet with its RD extent; a sheet is fetched
when it intersects a site's box. Files land in data_v4/truth/vaklodingen/.

Run:  python -m experiments.v0_vaklodingen_fetch [site ...]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

BASE = "https://opendap.deltares.nl/thredds/fileServer/opendap/rijkswaterstaat/vaklodingen/"
OUT = "data_v4/truth/vaklodingen"
BOXES = {"escalda": (3.55, 51.33, 3.85, 51.46),
         "wadden": (5.15, 53.15, 5.55, 53.45),
         "ems": (6.80, 53.22, 7.30, 53.50)}


def sheets_for(site):
    import pyproj
    import xarray as xr

    cat = xr.open_dataset("data_v4/truth/vaklodingen_catalog.nc")
    urls = [u.decode() if isinstance(u, bytes) else str(u) for u in cat["urlPath"].values]
    px, py = cat["projectionCoverage_x"].values, cat["projectionCoverage_y"].values
    tc = cat["timeCoverage"].values
    w, s, e, n = BOXES[site]
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    xs, ys = tr.transform([w, e, w, e], [s, s, n, n])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    out = []
    for u, (ax0, ax1), (ay0, ay1), (t0, t1) in zip(urls, px, py, tc):
        if ax1 < x0 or ax0 > x1 or ay1 < y0 or ay0 > y1:
            continue
        name = u.rsplit("/", 1)[-1]
        last = (np.datetime64("1970-01-01") + np.timedelta64(int(t1), "D")) if np.isfinite(t1) else None
        out.append((name, (ax0, ax1, ay0, ay1), last))
    return out


def main(sites):
    import truststore
    truststore.inject_into_ssl()
    import requests

    os.makedirs(OUT, exist_ok=True)
    for site in sites:
        sheets = sheets_for(site)
        print(f"[{site}] {len(sheets)} sheets intersect the box", flush=True)
        for name, (ax0, ax1, ay0, ay1), last in sheets:
            path = os.path.join(OUT, name)
            if os.path.exists(path) and os.path.getsize(path) > 1e6:
                print(f"  {name}: cached (last survey {str(last)[:10]})", flush=True)
                continue
            t0 = time.time()
            for attempt in range(4):
                try:
                    r = requests.get(BASE + name, timeout=600, stream=True)
                    r.raise_for_status()
                    with open(path + ".part", "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    os.replace(path + ".part", path)
                    print(f"  {name}: {os.path.getsize(path) / 1e6:.0f} MB in "
                          f"{time.time() - t0:.0f} s (last survey {str(last)[:10]}; "
                          f"x {ax0 / 1e3:.0f}-{ax1 / 1e3:.0f} km, y {ay0 / 1e3:.0f}-{ay1 / 1e3:.0f} km)",
                          flush=True)
                    break
                except Exception as exc:
                    print(f"  {name}: attempt {attempt + 1} failed ({exc})", flush=True)
                    time.sleep(30)
    print("V0 done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(BOXES))
