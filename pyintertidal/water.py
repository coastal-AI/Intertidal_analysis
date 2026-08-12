"""
water.py — Water detection: one registry, four detectors, clear guidance
========================================================================

Every stage that needs to know "is this pixel wet in this scene?" calls this
registry, so changing the detector is ONE parameter (``water="ndwi"``)
everywhere — stability, frequency and elevation all follow automatically, and
no stage can silently disagree with another.

Which detector should I use?
----------------------------
``ndwi``  (default) — NDWI = (B03 − B08)/(B03 + B08). Both bands are native
    10 m, so this is the only index with TRUE 10 m resolution, and it won our
    LiDAR benchmark. Threshold ≈ 0.0 for turbid estuaries (Cantabrian rías),
    ≈ 0.1 for clear water.
``mndwi`` — (B03 − B11)/(B03 + B11). More sensitive under thin vegetation and
    very shallow water, but B11 is 20 m (halves real resolution) and it
    over-detects turbid water. Consider for heavily vegetated marshes.
``awei``  — 4(B03 − B11) − (0.25·B08 + 2.75·B12). Suppresses shadow/built-up
    confusion; 20 m bands. Rarely better than NDWI on open coasts.
``scl``   — ESA's categorical scene classification (class 6 = water,
    12 = flooded vegetation). 20 m, binary. Kept for comparison studies and
    as the CLOUD mask for every detector (that role never changes).

All detectors share the same cloud handling: an observation only counts when
its SCL class is in ``CLEAR_CLASSES`` — so swapping detectors is a fair A/B
comparison by construction.
"""

from __future__ import annotations

import numpy as np

#: Sentinel-2 L2A Scene Classification, with the colours used in figures.
#: Class 12 is a project extension: salt-marsh vegetation that SCL labels as
#: dry vegetation but which is actually flooded (see :mod:`pyintertidal.marsh`).
SCL_CLASSES = {
    0: ("No data", "#000000"),
    1: ("Saturated / defective", "#ff0000"),
    2: ("Dark area", "#2f2f2f"),
    3: ("Cloud shadow", "#643200"),
    4: ("Vegetation", "#00a000"),
    5: ("Bare soil", "#ffe65a"),
    6: ("Water", "#0000ff"),
    7: ("Unclassified", "#808080"),
    8: ("Cloud medium probability", "#c0c0c0"),
    9: ("Cloud high probability", "#ffffff"),
    10: ("Thin cirrus", "#64c8ff"),
    11: ("Snow / ice", "#ff96ff"),
    12: ("Flooded vegetation (marsh)", "#009999"),
}

#: Sentinel-2 L2A SCL classes treated as cloud/bad in every stage.
BAD_CLASSES = (3, 8, 9, 10)
#: SCL classes accepted as a usable ("clear") observation.
CLEAR_CLASSES = (4, 5, 6, 12)
#: SCL classes counted as water when ``water="scl"``.
SCL_WATER_CLASSES = (6, 12)

#: Bands each detector needs (SCL is always added for the cloud mask).
REQUIRED_BANDS = {
    "ndwi": ("B03", "B08"),
    "mndwi": ("B03", "B11"),
    "awei": ("B03", "B08", "B11", "B12"),
    "scl": (),
}


def bands_for(water):
    """Bands to download for a detector (cloud-mask SCL always included)."""
    if water not in REQUIRED_BANDS:
        raise ValueError(f"unknown water detector {water!r}; "
                         f"choose from {sorted(REQUIRED_BANDS)}")
    return tuple(REQUIRED_BANDS[water]) + ("SCL",)


def index_block(water, blocks):
    """Water index for a block dict ``{band: (t, y, x) array}``.

    Returns float32, or ``None`` for the categorical ``"scl"`` detector.
    """
    if water == "scl":
        return None
    g = np.asarray(blocks["B03"], np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        if water == "ndwi":
            n = np.asarray(blocks["B08"], np.float32)
            den = g + n
            return np.where(den != 0, (g - n) / den, np.nan).astype(np.float32)
        if water == "mndwi":
            s1 = np.asarray(blocks["B11"], np.float32)
            den = g + s1
            return np.where(den != 0, (g - s1) / den, np.nan).astype(np.float32)
        if water == "awei":
            n = np.asarray(blocks["B08"], np.float32)
            s1 = np.asarray(blocks["B11"], np.float32)
            s2 = np.asarray(blocks["B12"], np.float32)
            scale = 10000.0 if np.nanmax(g) > 1.5 else 1.0   # DN vs reflectance
            return (4 * (g - s1) / scale
                    - (0.25 * n + 2.75 * s2) / scale).astype(np.float32)
    raise ValueError(f"unknown water detector {water!r}")


def water_land_masks(water, blocks, threshold, clear_classes=CLEAR_CLASSES):
    """Boolean ``(wet, dry, clear)`` masks for one time-block.

    ``clear`` always comes from SCL (the cloud mask); wet/dry are the clear
    observations above/below the index threshold — or the SCL water/land
    classes for the categorical detector.
    """
    scl = np.nan_to_num(np.asarray(blocks["SCL"]), nan=0.0).astype(np.int16)
    clear = np.isin(scl, list(clear_classes))
    if water == "scl":
        wet = clear & np.isin(scl, list(SCL_WATER_CLASSES))
        dry = clear & np.isin(scl, [4, 5])
        return wet, dry, clear
    idx = index_block(water, blocks)
    finite = np.isfinite(idx)
    wet = clear & finite & (idx > threshold)
    dry = clear & finite & (idx <= threshold)
    return wet, dry, clear


def scene_quality(scl, bad_classes=BAD_CLASSES):
    """Quality summary of a single scene from its SCL band.

    Returns ``{"bad_fraction", "clear_fraction", "counts"}`` where ``counts``
    is the full class histogram. Use it to inspect why a particular date was
    accepted or rejected — the pipeline decides with ``bad_fraction``, but
    the histogram tells you whether the problem was cloud, shadow or snow.
    """
    scl = np.nan_to_num(np.asarray(scl), nan=0.0).astype(np.int16)
    total = max(scl.size, 1)
    counts = {int(c): int((scl == c).sum()) for c in np.unique(scl)}
    bad = float(np.isin(scl, list(bad_classes)).sum()) / total
    clear = float(np.isin(scl, list(CLEAR_CLASSES)).sum()) / total
    return {"bad_fraction": bad, "clear_fraction": clear, "counts": counts}


def otsu_threshold(index_samples, clip=(-0.4, 0.25)):
    """Otsu's split of a pooled sample of index values.

    Coastal water indices are bimodal — water high, land low — so the optimal
    cut is the split that best separates the two modes. The result is clipped
    to a physically sensible range: turbid estuaries legitimately push the
    optimum below the textbook 0.0, but a value far outside this range means
    the sample was not bimodal (all land, or all water) and should not be
    trusted.

    Feed it a representative sample — :func:`calibrate_threshold` builds one.
    """
    from skimage.filters import threshold_otsu

    vals = np.asarray(index_samples)
    vals = vals[np.isfinite(vals)]
    if vals.size < 100:
        return 0.0
    return float(np.clip(float(threshold_otsu(vals)), clip[0], clip[1]))


def calibrate_threshold(cube, n_scenes=12, max_pixels=400_000, chunk=None,
                        clip=(-0.4, 0.25), verbose=True):
    """Calibrate the water threshold from the cube's own clearest scenes.

    Sampling matters more than the statistic here. A scene picked at random
    may be all cloud, or cover only land at low tide, and its histogram is
    then not bimodal at all — Otsu on that sample returns nonsense. So this
    function:

    1. scans the archive cheaply (subsampled) to rank scenes by cloudiness,
    2. pools the clear-sky index values of the ``n_scenes`` clearest ones,
    3. returns the Otsu split of that pooled sample.

    A returned value pinned at either end of ``clip`` is a warning sign, not
    a result: it means even the clearest scenes did not show two clear modes,
    and you should set the threshold yourself.

    Returns ``(threshold, diagnostics)``.
    """
    from .cube import open_cube

    if cube.water == "scl":
        return 0.0, {"note": "categorical detector: no threshold needed"}

    dates = cube.dates
    subsample = 8            # spatial stride for the cheap ranking pass

    # Pass 1 — rank scenes by cloudiness. Only SCL is read, and spatially
    # subsampled: cloud fraction does not need full resolution, and reading
    # the reflectance bands here would triple the cost for nothing.
    ds, arrays, t_dim = open_cube(cube.cache_path, ("SCL",))
    try:
        scl_da = arrays["SCL"]
        cloudiness = []
        for i in range(scl_da.sizes[t_dim]):
            tile = np.nan_to_num(
                np.asarray(scl_da.isel({t_dim: i}).values[::subsample,
                                                          ::subsample]),
                nan=0.0).astype(np.int16)
            cloudiness.append((float(np.isin(tile, list(BAD_CLASSES)).mean()), i))
    finally:
        ds.close()
    cloudiness.sort()
    clearest = sorted(i for _, i in cloudiness[:int(n_scenes)])

    # Pass 2 — read only the chosen scenes, at full resolution.
    ds, arrays, t_dim = open_cube(cube.cache_path, cube.bands)
    samples, used = [], []
    try:
        per_scene = max_pixels // max(len(clearest), 1)
        for i in clearest:
            blocks = {b: np.asarray(arrays[b].isel({t_dim: i}).values)[None]
                      for b in cube.bands}
            index = index_block(cube.water, blocks)[0]
            scl = np.nan_to_num(blocks["SCL"][0], nan=0.0).astype(np.int16)
            clear = np.isin(scl, list(CLEAR_CLASSES))
            vals = index[clear & np.isfinite(index)]
            if vals.size:
                samples.append(vals[:: max(1, vals.size // per_scene)])
                used.append(dates[i])
    finally:
        ds.close()

    if not samples:
        return 0.0, {"note": "no clear-sky samples found"}
    pooled = np.concatenate(samples)
    threshold = otsu_threshold(pooled, clip=clip)
    saturated = threshold in clip
    info = {"threshold": threshold, "n_pixels": int(pooled.size),
            "n_scenes": len(used), "scenes": used,
            "cloudiest_sampled": round(max(c for c, _ in cloudiness[:len(clearest)]), 3),
            "clipped": saturated}
    if verbose:
        print(f"[water] Otsu on {pooled.size:,} clear pixels from "
              f"{len(used)} scenes → {threshold:+.3f}"
              + ("  (clipped — the sample was not clearly bimodal; "
                 "set the threshold manually)" if saturated else ""))
    return threshold, info
