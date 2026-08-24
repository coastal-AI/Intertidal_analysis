"""Screen candidate estuaries by how usable the IGN LiDAR is over the flats.

A survey flown at high water returns the WATER SURFACE, which is flat: a
large contiguous area sitting at one single value, with no gradient. A survey
flown at low water returns the BED, which slopes. That difference is visible
in the DEM alone, with no satellite data at all, so a site can be screened in
seconds before committing to a 3 GB cube download.
"""
import io, os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

from pyintertidal.net import use_system_certificates
use_system_certificates()

import numpy as np
import rasterio
import requests

WCS = "https://servicios.idee.es/wcs-inspire/mdt"

SITES = {
    "Villaviciosa":      (-5.455, 43.470, -5.375, 43.548),
    "Santoña":           (-3.500, 43.400, -3.420, 43.460),
    "Bahía Santander":   (-3.850, 43.400, -3.750, 43.470),
    "Urdaibai":          (-2.710, 43.350, -2.650, 43.420),
    "Ría del Eo":        (-7.060, 43.490, -7.000, 43.560),
    "Foz":               (-7.290, 43.520, -7.210, 43.590),
    "Ortigueira":        (-7.890, 43.640, -7.780, 43.720),
    "Bahía de Cádiz":    (-6.250, 36.450, -6.120, 36.580),
    "Odiel (Huelva)":    (-6.980, 37.150, -6.850, 37.280),
    "Ría de Arousa":     (-8.870, 42.500, -8.760, 42.600),
}


def fetch(bbox, timeout=180):
    w, s, e, n = bbox
    p = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
         ("coverageId", "Elevacion4258_5"), ("format", "image/tiff"),
         ("subset", f"Lat({s},{n})"), ("subset", f"Long({w},{e})")]
    r = requests.get(WCS, params=p, timeout=timeout)
    if r.status_code != 200 or len(r.content) < 2000:
        raise RuntimeError(f"HTTP {r.status_code}")
    with rasterio.open(io.BytesIO(r.content)) as src:
        return src.read(1).astype(float), src.dtypes[0]


def score(z):
    """Diagnostics of the low-lying band, where the intertidal must live."""
    band = z[(z > -5) & (z < 5)]
    if band.size < 5000:
        return None
    vals, counts = np.unique(band, return_counts=True)
    flat = counts.max() / band.size              # share at ONE single value
    # Local gradient: a real bed slopes, a water surface does not.
    low = np.where((z > -5) & (z < 5), z, np.nan)
    gy, gx = np.gradient(np.nan_to_num(low, nan=0.0))
    m = np.isfinite(low)
    grad = float(np.nanmedian(np.hypot(gy, gx)[m])) if m.any() else np.nan
    return dict(n=band.size, distinct=vals.size, flat=flat, grad=grad,
                lo=float(band.min()), hi=float(band.max()))


print(f"{'sitio':18s} {'dtype':7s} {'px bajos':>9s} {'valores':>8s} "
      f"{'% en 1 valor':>12s} {'grad':>7s}")
print("-" * 70)
rows = []
for name, bbox in SITES.items():
    try:
        z, dt = fetch(bbox)
        s = score(z)
        if s is None:
            print(f"{name:18s} {dt:7s}  (sin zona baja suficiente)")
            continue
        rows.append((name, s))
        print(f"{name:18s} {dt:7s} {s['n']:9,d} {s['distinct']:8d} "
              f"{100 * s['flat']:11.1f}% {s['grad']:7.3f}")
    except Exception as exc:
        print(f"{name:18s} ERROR {type(exc).__name__}: {str(exc)[:40]}")

if rows:
    print("\nMenos plano = mas probable que el LiDAR midiera el FONDO:")
    for name, s in sorted(rows, key=lambda r: r[1]["flat"]):
        verdict = ("lamina de agua" if s["flat"] > 0.35 else
                   "dudoso" if s["flat"] > 0.15 else "parece fondo")
        print(f"  {name:18s} {100 * s['flat']:5.1f}% en un solo valor, "
              f"gradiente {s['grad']:.3f}  -> {verdict}")
