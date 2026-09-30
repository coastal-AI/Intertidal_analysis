"""C0a — extract what Granadeiro's method consumes from a cached cube.

Granadeiro et al. 2021 (Remote Sens. 13, 320) fit their 4-parameter
logistic to the **NIR reflectance** of each intertidal pixel, after an
inter-calibration step: every scene is regressed (major-axis) against one
reference scene using ONLY stable pixels — open sea (NIR < 0.05) and land
(NIR > 0.2), never the intertidal — and corrected with those coefficients.

This pass streams the cube once and writes ``granadeiro_extract.npz``:

* ``NIR``   (T, P) float16 — NIR reflectance at the same ``keep`` pixels as
  ``marea_demo_extract.npz`` (so wet/dry, clear masks and s_km are shared);
* ``calib`` (T, K) float32 — NIR at K fixed sea/land calibration pixels;
* ``calib_is_land`` (K,) bool, ``ref_idx`` int — the reference scene
  (clearest, fully covered), chosen exactly as a canvas is;
* ``bad_frac`` (T,) — per-scene cloudy fraction, for their <10 % rule.

Run:  python -m experiments.c0_granadeiro_extract [site]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pyintertidal.cube import open_cube
from pyintertidal.water import BAD_CLASSES

CUBES = {"villaviciosa": "ndwi_cube_villaviciosa_grande_10y.nc",
         "escalda": "ndwi_cube_escalda_2023-2025_20m.nc"}


def main(site="villaviciosa", base="marea_demo_extract.npz",
         out="granadeiro_extract.npz", cube=None):
    """``base`` is any extraction with ``keep`` and ``shape`` (the MAREA
    extraction of the same cube); ``out`` names the NIR file."""
    src = np.load(base, allow_pickle=True)
    keep = src["keep"]
    H, W = (int(v) for v in src["shape"])
    rows, cols = keep // W, keep % W
    inter_mask = np.zeros(H * W, bool)
    inter_mask[keep] = True

    ds, arrays, t_dim = open_cube(cube or CUBES[site],
                                  ("B03", "B08", "SCL"))
    try:
        T = arrays["SCL"].sizes[t_dim]

        # pass 1 — cloud + coverage score, subsampled (canvas rule)
        bad = np.empty(T, np.float32)
        nodata = np.empty(T, np.float32)
        for i in range(T):
            tile = np.nan_to_num(
                arrays["SCL"].isel({t_dim: i}).values[::6, ::6], nan=0.0)
            bad[i] = np.isin(tile, list(BAD_CLASSES)).mean()
            nodata[i] = (tile == 0).mean()
        covered = nodata <= 0.02
        ref_idx = int(np.flatnonzero(covered)[
            np.argmin(bad[covered])]) if covered.any() else int(np.argmin(bad))
        print(f"reference scene: index {ref_idx} "
              f"(bad {bad[ref_idx]:.3f}, nodata {nodata[ref_idx]:.3f})",
              flush=True)

        # calibration pixels: sea (NIR<0.05) and land (NIR>0.2) on the
        # REFERENCE scene, excluding the intertidal — their Figure 2 rule
        nir_ref = np.asarray(arrays["B08"].isel({t_dim: ref_idx}).values,
                             np.float32).ravel() / 10000.0
        rng = np.random.default_rng(11)
        sea_pool = np.flatnonzero((nir_ref < 0.05) & (nir_ref > 0)
                                  & ~inter_mask)
        land_pool = np.flatnonzero((nir_ref > 0.2) & ~inter_mask)
        sea = rng.choice(sea_pool, min(4000, len(sea_pool)), replace=False)
        land = rng.choice(land_pool, min(4000, len(land_pool)),
                          replace=False)
        calib_idx = np.concatenate([sea, land])
        calib_is_land = np.zeros(len(calib_idx), bool)
        calib_is_land[len(sea):] = True
        crows, ccols = calib_idx // W, calib_idx % W
        print(f"calibration pixels: {len(sea)} sea + {len(land)} land",
              flush=True)

        # pass 2 — NIR at keep + calibration pixels, streamed
        NIR = np.full((T, len(keep)), np.nan, np.float16)
        calib = np.full((T, len(calib_idx)), np.nan, np.float32)
        CH = 40
        for i0 in range(0, T, CH):
            sl = slice(i0, min(i0 + CH, T))
            b8 = np.asarray(arrays["B08"].isel({t_dim: sl}).values,
                            np.float32) / 10000.0
            b8[b8 <= 0] = np.nan          # nodata DN 0 -> hole, not water
            NIR[sl] = b8[:, rows, cols].astype(np.float16)
            calib[sl] = b8[:, crows, ccols]
            print(f"  {min(i0 + CH, T)}/{T} scenes", flush=True)
    finally:
        ds.close()

    np.savez_compressed(
        out, NIR=NIR, calib=calib,
        calib_is_land=calib_is_land, ref_idx=np.array(ref_idx),
        bad_frac=bad, nodata_frac=nodata)
    gb = os.path.getsize(out) / 1e9
    print(f"-> {out} ({gb:.2f} GB)")


if __name__ == "__main__":
    main(*sys.argv[1:4]) if len(sys.argv) > 1 else main()
