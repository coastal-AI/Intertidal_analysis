# -*- coding: utf-8 -*-
"""The grid of the validation runs: one switch for the Dutch cubes.

Villaviciosa and Ferrol were always 10 m. The Dutch sites (Westerschelde,
Wadden, Ems) ran at 20 m until 2026-09-30, then at 10 m (author's
decision), so every site is on the same grid. Every script that reads the
Dutch cubes or products takes the names from here; the notebooks set the
same values in their first cell (RESOLUTION_M, CACHE, OUT_DIR).

The 20 m products stay in products_<site>/ (the local archive); the 10 m ones
go to products_<site>_10m/, so the two never mix.
"""

DUTCH_SITES = ("escalda", "wadden", "ems")
DUTCH_RES_M = 10
DUTCH_PERIOD = ("2023-01-01", "2025-12-31")

#: the thinned Vaklodingen subset keeps one pixel per THIN_M x THIN_M block,
#: against spatial autocorrelation (5 x 5 px at 20 m, 10 x 10 px at 10 m)
THIN_M = 100.0


def dutch_cube(site, res_m=DUTCH_RES_M):
    return f"ndwi_cube_{site}_2023-2025_{res_m}m.nc"


def dutch_products(site, res_m=DUTCH_RES_M):
    return f"products_{site}" if res_m == 20 else f"products_{site}_{res_m}m"


def thin_px(pixel_m):
    """Block side in pixels of the thinned subset for a grid of pixel_m."""
    return max(1, int(round(THIN_M / float(pixel_m))))


def thin_mask(rows_k, cols_k, pixel_m):
    """One pixel per THIN_M block: the first record pixel of each block."""
    import numpy as np
    k = thin_px(pixel_m)
    rows_k, cols_k = np.asarray(rows_k), np.asarray(cols_k)
    thin = np.zeros(len(rows_k), bool)
    thin[np.unique((rows_k // k) * 7919 + (cols_k // k), return_index=True)[1]] = True
    return thin
