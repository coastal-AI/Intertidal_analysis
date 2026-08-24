"""Do the candidate surveys actually cover the flats, like the Tagus did?

Same test as the Tagus: a survey useful to us must have values ABOVE chart
datum, in float, actually measured. One that stops at 0 m is a boat survey of
the channel and validates nothing.
"""
import os, sys, zipfile, io
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

from pyintertidal.net import use_system_certificates
use_system_certificates()

import numpy as np
import requests
import xarray as xr

WMS = "https://ows.emodnet-bathymetry.eu/wms"
CAND = [("Aveiro (PT)", -8.70, 40.65), ("Ria Formosa (PT)", -7.90, 37.00),
        ("Nordfriesland (DE)", 8.60, 54.55), ("Jade (DE)", 8.15, 53.60),
        ("Wadden (DK)", 8.45, 55.30), ("Venecia (IT)", 12.35, 45.40),
        ("Sado (PT)", -8.85, 38.50)]
OUT = "bathy_candidatos"
os.makedirs(OUT, exist_ok=True)


def urls(lon, lat, d=0.20):
    p = {"service": "WMS", "version": "1.3.0", "request": "GetFeatureInfo",
         "layers": "emodnet:hr_bathymetry_area",
         "query_layers": "emodnet:hr_bathymetry_area", "crs": "EPSG:4326",
         "info_format": "application/json", "width": 101, "height": 101,
         "i": 50, "j": 50, "feature_count": 4,
         "bbox": f"{lat-d},{lon-d},{lat+d},{lon+d}"}
    r = requests.get(WMS, params=p, timeout=70)
    out = []
    for f in r.json().get("features", []):
        pr = f["properties"]
        out.append((pr.get("identifier"), pr.get("resolution"),
                    pr.get("download_url")))
    return out


print(f"{'sitio':20s} {'producto':30s} {'MB':>6s} {'res_m':>6s} "
      f"{'rango (m)':>18s} {'>0 m':>7s}")
print("-" * 96)
for name, lon, lat in CAND:
    for ident, res, url in urls(lon, lat)[:1]:
        if not url:
            print(f"{name:20s} {str(ident)[:30]:30s}  sin URL")
            continue
        zp = os.path.join(OUT, os.path.basename(url))
        try:
            if not os.path.exists(zp):
                r = requests.get(url, timeout=900)
                r.raise_for_status()
                open(zp, "wb").write(r.content)
            mb = os.path.getsize(zp) / 1e6
            with zipfile.ZipFile(zp) as z:
                nc = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc:
                    print(f"{name:20s} {str(ident)[:30]:30s} {mb:6.1f}  sin .nc")
                    continue
                z.extract(nc[0], OUT)
            ds = xr.open_dataset(os.path.join(OUT, nc[0]))
            zz = ds.elevation.values.astype(float)
            lo_, la_ = ds.lon.values, ds.lat.values
            px = abs(lo_[1] - lo_[0]) * 111320 * np.cos(np.radians(lat))
            v = zz[np.isfinite(zz)]
            above = 100 * np.mean(v > 0)
            print(f"{name:20s} {str(ident)[:30]:30s} {mb:6.1f} {px:6.1f} "
                  f"{v.min():+8.2f}..{v.max():+8.2f} {above:6.0f}%")
            ds.close()
        except Exception as exc:
            print(f"{name:20s} {str(ident)[:30]:30s}  ERROR "
                  f"{type(exc).__name__}: {str(exc)[:35]}")
