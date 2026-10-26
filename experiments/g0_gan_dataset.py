"""Export a TideGAN training archive from a cached pyintertidal cube.

Writes ``gan_archive_{site}/`` with (T, H, W) memmaps:

* ``ndwi.f16`` — the water index in [-1, 1], the channel MAREA estimates on;
* ``nir.f16``  — B08 reflectance mapped to [-1, 1] (DN/10000, clipped),
  the raw radiometry the index is made from;
* ``clear.u8`` — 1 where SCL calls the pixel clear sky;
* ``meta.json`` — dates, EOT20 tide heights, per-date clear fractions,
  the tide normalisation range, and provenance.

Physical channels instead of stretched 8-bit PNGs, on purpose: MAREA judges
a generated scene by where it puts the waterline in *index* space, so the
model must be trained and evaluated there, not in display space.

Run:  python -m experiments.g0_gan_dataset [site]     (default villaviciosa)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import rasterio

import pyintertidal as pit
from pyintertidal import water as W
from pyintertidal.cube import SentinelCube

KNOWN_CACHES = {"villaviciosa": "ndwi_cube_villaviciosa_grande_10y.nc",
                "santander": "ndwi_cube_santander_2023_2025.nc",
                "arousa": "ndwi_cube_arousa_2023-2025_20m.nc"}
SITE_NAMES = {"villaviciosa": "Villaviciosa", "santander": "Santander",
              "foz": "Foz"}          # TideGAN's site vocabulary


def main(site="villaviciosa"):
    aoi = pit.sites.get(site)
    cube = SentinelCube(aoi, ("2016-01-01", "2026-01-01"), water="ndwi",
                        cache_path=KNOWN_CACHES[site])
    if not cube.cached:
        raise SystemExit(f"cube {cube.cache_path} not on disk")
    dates = cube.dates
    T = len(dates)
    H, Wd = cube.shape
    print(f"{site}: {T} dates, grid {H} x {Wd}")

    # ── tides at the mid-channel point, like the study notebooks ──────────
    ref_path = os.path.join(f"products_{site}", "reference_map.tif")
    point = None
    if os.path.exists(ref_path):
        with rasterio.open(ref_path) as src:
            try:
                point = pit.water_centroid(src.read(1), src.transform,
                                           src.crs)
            except ValueError:
                # a reference map without stable water (it happens: the
                # 2026-08-12 Villaviciosa one) cannot give a mid-channel point
                point = None
    if point is None:
        lon, lat = aoi.centroid
        point = (lat, lon)
    tides = pit.TideService(model="EOT20", directory="./tide_models",
                            location=point)
    heights = tides.heights_for(aoi, dates)
    h = np.asarray([heights.get(d, np.nan) for d in dates], dtype=np.float32)
    ok = np.isfinite(h)
    print(f"tide heights: {ok.sum()}/{T} dates, "
          f"range [{np.nanmin(h):+.2f}, {np.nanmax(h):+.2f}] m")

    # ── stream the cube into the memmaps ──────────────────────────────────
    out = f"gan_archive_{site}"
    os.makedirs(out, exist_ok=True)
    ndwi_mm = np.memmap(os.path.join(out, "ndwi.f16"), dtype=np.float16,
                        mode="w+", shape=(T, H, Wd))
    nir_mm = np.memmap(os.path.join(out, "nir.f16"), dtype=np.float16,
                       mode="w+", shape=(T, H, Wd))
    clear_mm = np.memmap(os.path.join(out, "clear.u8"), dtype=np.uint8,
                         mode="w+", shape=(T, H, Wd))
    clear_frac = np.zeros(T, dtype=np.float32)

    done = 0
    for sl, blocks in cube.stream():
        ndwi = W.index_block(cube.water, blocks)
        ndwi = np.clip(np.nan_to_num(ndwi, nan=0.0), -1.0, 1.0)
        nir = np.asarray(blocks["B08"], dtype=np.float32)
        nir = np.clip(np.nan_to_num(nir, nan=0.0) / 10000.0, 0.0, 1.0)
        nir = nir * 2.0 - 1.0
        scl = np.nan_to_num(np.asarray(blocks["SCL"], dtype=np.float32),
                            nan=0.0).astype(np.int16)
        clear = np.isin(scl, list(W.CLEAR_CLASSES))
        ndwi_mm[sl] = ndwi.astype(np.float16)
        nir_mm[sl] = nir.astype(np.float16)
        clear_mm[sl] = clear.astype(np.uint8)
        clear_frac[sl] = clear.reshape(clear.shape[0], -1).mean(axis=1)
        done += ndwi.shape[0]
        print(f"  {done}/{T} dates", flush=True)
    ndwi_mm.flush(); nir_mm.flush(); clear_mm.flush()

    meta = {
        "site": SITE_NAMES.get(site, site.capitalize()),
        "site_key": site,
        "shape": [T, H, Wd],
        "dates": dates,
        "tides_m": [None if not np.isfinite(v) else round(float(v), 4)
                    for v in h],
        "tide_range_m": [round(float(np.nanmin(h)), 4),
                         round(float(np.nanmax(h)), 4)],
        "tide_model": "EOT20",
        "tide_point_latlon": [round(float(point[0]), 5),
                              round(float(point[1]), 5)],
        "clear_fraction": [round(float(v), 4) for v in clear_frac],
        "nir_normalisation": "clip(DN/10000, 0, 1) * 2 - 1",
        "source_cube": cube.cache_path,
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    gb = sum(os.path.getsize(os.path.join(out, n))
             for n in os.listdir(out)) / 1e9
    print(f"archive -> {out}/  ({gb:.1f} GB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "villaviciosa")
