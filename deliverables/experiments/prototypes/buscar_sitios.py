"""Which European intertidal systems have a high-resolution survey?

One site is an anecdote. The Tagus said HSR beats the published baseline by
10 % on common pixels; that only becomes a result if it holds somewhere else,
ideally in a different tidal regime.
"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

from pyintertidal.net import use_system_certificates
use_system_certificates()

import requests

WMS = "https://ows.emodnet-bathymetry.eu/wms"

SITES = [
    # Iberia
    ("Tejo (PT)", -9.00, 38.78), ("Sado (PT)", -8.85, 38.50),
    ("Ria de Aveiro (PT)", -8.70, 40.65), ("Ria Formosa (PT)", -7.90, 37.00),
    ("Cadiz (ES)", -6.20, 36.52), ("Odiel (ES)", -6.93, 37.22),
    ("Guadalquivir (ES)", -6.35, 36.85),
    # France
    ("Arcachon (FR)", -1.15, 44.68), ("Gironde (FR)", -0.95, 45.35),
    ("Loire (FR)", -2.10, 47.25), ("Mont-St-Michel (FR)", -1.55, 48.62),
    ("Baie de Somme (FR)", 1.55, 50.20), ("Seine (FR)", 0.25, 49.44),
    # UK
    ("Wash (UK)", 0.30, 52.90), ("Humber (UK)", -0.10, 53.60),
    ("Thames (UK)", 0.70, 51.48), ("Severn (UK)", -2.95, 51.50),
    ("Morecambe (UK)", -3.00, 54.10), ("Solway (UK)", -3.40, 54.90),
    ("Dee (UK)", -3.15, 53.30),
    # Wadden Sea
    ("Texel (NL)", 4.90, 53.05), ("Terschelling (NL)", 5.35, 53.38),
    ("Ameland (NL)", 5.75, 53.44), ("Eems-Dollard (NL/DE)", 6.90, 53.35),
    ("Jade (DE)", 8.15, 53.60), ("Weser (DE)", 8.35, 53.75),
    ("Elbe (DE)", 8.75, 53.90), ("Nordfriesland (DE)", 8.60, 54.55),
    ("Wadden (DK)", 8.45, 55.30),
    # Others
    ("Venecia (IT)", 12.35, 45.40), ("Po (IT)", 12.45, 44.85),
]


def probe(lon, lat, d=0.20, n=8):
    p = {"service": "WMS", "version": "1.3.0", "request": "GetFeatureInfo",
         "layers": "emodnet:hr_bathymetry_area",
         "query_layers": "emodnet:hr_bathymetry_area", "crs": "EPSG:4326",
         "info_format": "application/json", "width": 101, "height": 101,
         "i": 50, "j": 50, "feature_count": n,
         "bbox": f"{lat-d},{lon-d},{lat+d},{lon+d}"}
    try:
        r = requests.get(WMS, params=p, timeout=70)
        return r.json().get("features", []) if r.status_code == 200 else []
    except Exception:
        return []


found = []
print(f"{'sitio':22s} {'levantamientos':>14s}   detalle")
print("-" * 78)
for name, lon, lat in SITES:
    fs = probe(lon, lat)
    if not fs:
        print(f"{name:22s} {'-':>14s}")
        continue
    seen, rows = set(), []
    for f in fs:
        pr = f["properties"]
        key = (pr.get("identifier"), pr.get("resolution"))
        if key in seen:
            continue
        seen.add(key)
        rows.append((pr.get("identifier"), pr.get("resolution"),
                     pr.get("release"), pr.get("download_url")))
    print(f"{name:22s} {len(rows):>14d}   "
          + "; ".join(f"{i[:26]} (1/{r}', {y})" for i, r, y, _ in rows[:2]))
    for i, r, y, u in rows:
        found.append((name, i, r, y, u))

print(f"\n{len(found)} levantamientos en {len({f[0] for f in found})} sitios")
print("\nLos mas finos (mayor denominador = mejor resolucion):")
for name, i, r, y, u in sorted(found, key=lambda x: -(x[2] or 0))[:10]:
    m = 1852 / 60 / (r or 1) * 60 / 60      # 1/r arc-min -> metres approx
    print(f"  {name:22s} {str(i)[:32]:34s} 1/{r}' ~ {1852/(r or 1)/60*60/60:.0f} m  {y}")
