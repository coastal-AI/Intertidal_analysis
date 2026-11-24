"""
marea.py — MAREA end to end: from the looks of a cube to the elevation map
==========================================================================

*Mudflat Altimetry via Remotely-Estimated tides from the Archive.*

The method in words, with its three equations: ``guide/METHOD.md``. For every
intertidal pixel ``p`` MAREA fits the flooding curve (METHOD Eq. 1) on the
tide that reaches the pixel (Eq. 2)::

    NDWI_pt = a_p + b_p * Phi((h_B(t - tau(s_p)) - z_p) / sigma_p)

where ``h_B`` is the water level at the mouth (a
:class:`pyintertidal.boundary.BoundaryProvider`, metres), ``tau(s)`` the delay
of the tide at the distance ``s`` from the mouth along the water (minutes,
never negative), ``z_p`` the elevation (metres, in the datum of ``h_B``) and
``sigma_p`` the width of the dry-to-wet transition (metres).

What each function does (the step numbers of METHOD.md, section 10):

* :func:`extract` — step 4: the looks (NDWI and a clear flag) of every
  intertidal pixel, and the permanent sea, in two streaming passes.
* :func:`reconstruct` — steps 7–10 and the hypsometry of step 11; it reads
  the tide (step 6) itself only without a ``boundary``: the mouth, the
  distance ``s``, the 3-km sections, one delay per section with a bootstrap
  interval, the delay of every pixel, and :func:`invert_series` per delay.
  Writes ``marea.npz``, ``result.json`` and ``figure.png``.
* :func:`invert_series` — step 10 alone: ``z`` and ``sigma`` of a block of
  pixels that share one level series ``h``.

Steps 1-3, 5 and 11 (download, overpass times, screening, record, emersion
time, GeoTIFFs) live elsewhere: ``SentinelCube.ensure``,
``overpass.get_overpass_times`` and ``scripts/marea_tools.py`` (``screen``,
``build_record``, ``emersion_time``, ``export_geotiffs``).
``scripts/quickstart.py`` runs them all in one call.

Words of the code, and the words of METHOD.md
---------------------------------------------
``band``, ``band_of``, ``band_km``
    a 3-km *section* along ``s`` (0 is the mouth; -1: the pixel has no ``s``).
``bank`` (:class:`~pyintertidal.tide_estimators.ShiftBank`)
    the mouth level read ``tau`` minutes earlier at every scene, for each
    ``tau`` of :data:`BANK_TAUS` (linear in between).
``keep``
    flat indices (``row * W + col``) of the intertidal pixels in the frame.
``Y``, ``C``
    (T, P) arrays: the NDWI of each look, and whether the look is clear.
``h0``
    the mouth level at the overpass instants (no delay).
``tau``, ``tau_used``, ``tau_px``
    section delays as fitted, as applied, and the delay of each pixel (min).
``sigma``, ``sg``
    the width of the flooding curve (m). The maps show it as the uncertainty
    of ``z``, but it is not a standard error (METHOD.md, section 8).

The keys of ``result.json`` are in Spanish, as in the published product
(``products_marea/result.json``):
``sitio`` site; ``estado`` status (``"ok"``; ``"sin_intermareal"``: fewer
than 3,000 intertidal pixels; ``"pocas_escenas"``: fewer than ``min_scenes``
scenes); ``n_escenas`` scenes in the record; ``n_px`` intertidal pixels;
``n_px_cota`` pixels with an elevation; ``centros_km`` section centres;
``tau_usado_min`` applied section delays; ``con_operador`` some pixel has a
non-zero delay; ``veto_fisico`` a negative section delay was clipped to 0;
``rango_marea_m`` [min, max] of ``h0``, the range searched for ``z``;
``hipsometria`` hypsometric integral and area; ``duracion_s`` run time (s).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import geometry, seal
from . import tide_estimators as te
from .elevation import _fit_block

#: Candidate widths sigma (m) of the elevation fit (step 10). The delay search
#: (step 8) profiles the six widths of configs/m2.yaml ``sigma_perfil_m``
#: instead. Same values as ``tide_estimators.SG_GRID``.
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
#: SCL classes of a clear look: 4 vegetation, 5 bare soil, 6 water,
#: 7 unclassified (METHOD.md, section 2). The screening
#: (``stability.reference_and_clouds``) uses ``water.CLEAR_CLASSES``
#: = (4, 5, 6, 12) instead.
CLEAR = (4, 5, 6, 7)
#: Scenes read per streaming block in :func:`extract` (memory only).
T_CHUNK = 40
TAU_DETECT_MIN = 0.0      # no detection threshold: the fitted lag of every
                          # band is applied as measured (a fixed cut-off
                          # would set small but real delays to 0). Negative
                          # lags are clipped to 0 by causality, band by band.
                          # With 0 the test |tau| > TAU_DETECT_MIN in
                          # reconstruct changes nothing: this constant, the
                          # default of reconstruct's tau_detect_min, has no
                          # effect in this edition.
#: Per-pixel quality rule (2026-10-06): an
#: elevation is kept only if it is INTERPOLATED from the pixel's own record,
#: i.e. at least this many clear observations were taken with the (lagged)
#: water level below it and as many above it. Fewer than 3 on a side cannot
#: tell a wet/dry transition from one misclassified or mislevelled scene
#: (non-tidal residual ~0.13 m). It removes elevations pinned to the edge of
#: the sampled range, such as the lowest inner flats of a long estuary when
#: the delay moves the record off low water. Applied to MAREA and no delay
#: alike.
QUALITY_MIN_BRACKET = 3


def extract(cube_path):
    """Step 4: the looks of every intertidal pixel of a cube, in two passes.

    The first pass counts, per pixel, the clear looks and the wet ones
    (NDWI > 0); the second keeps the NDWI series of the intertidal pixels.

    Parameters
    ----------
    cube_path : str
        netCDF cube with the variables ``B03`` and ``B08`` (reflectance, any
        scale) and ``SCL`` (classes 0-11) on the dimensions (time, y, x), and
        a ``crs`` grid-mapping variable with ``crs_wkt`` or ``spatial_ref``:
        an openEO netCDF, or ``data/villaviciosa_sample.nc``. The grid must be
        projected, in metres (UTM). Every date of the cube is used.

    Returns
    -------
    dict
        ``Y`` : (T, P) float32
            NDWI = (B03 - B08) / (B03 + B08) of each look; NaN where
            B03 + B08 = 0.
        ``C`` : (T, P) bool
            the look is clear: SCL in :data:`CLEAR` = (4, 5, 6, 7).
        ``keep`` : (P,) int
            flat indices (``row * W + col``) of the intertidal pixels.
        ``dates`` : (T,) str
            ``'YYYY-MM-DD'`` of each scene, in cube order.
        ``shape`` : (H, W)
        ``inter`` : (H, W) bool
            intertidal: more than 30 clear looks, wet (NDWI > 0) in more than
            10 % and less than 90 % of them.
        ``sea`` : (H, W) bool
            permanent water: more than 30 clear looks, wet in at least 90 %.
            Its pixels on the frame's edge are the mouth.
        ``bbox`` : dict
            ``west``/``south``/``east``/``north`` in degrees, from the pixel
            centres, and ``crs`` ``"EPSG:4326"``.
        ``crs_wkt`` : str
            the cube's coordinate reference system.

    Notes
    -----
    The mask depends on every scene of the cube: the same area extracted
    from a 3-year and a 10-year cube gives different pixels (on the sample
    block, 13,294 pixels from its 97 scenes against 15,064 from the
    product's 1,379). ``Y`` and ``C`` take about 5 bytes x T x P (1,379
    scenes x 32,000 pixels is about 220 MB).

    Pass the dict to :func:`reconstruct` as ``extraction``, after
    ``scripts/marea_tools.build_record`` has kept the record's scenes.

    Examples
    --------
    >>> ex = extract("data/villaviciosa_sample.nc")      # doctest: +SKIP
    >>> ex["Y"].shape                                    # doctest: +SKIP
    (97, 13294)
    """
    import xarray as xr

    ds = xr.open_dataset(cube_path)
    t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
    T = ds.sizes[t_dim]
    H, W = ds.sizes["y"], ds.sizes["x"]
    dates = np.array([str(np.datetime64(v, "D")) for v in ds[t_dim].values])

    wet_n = np.zeros((H, W), np.int32)
    clr_n = np.zeros((H, W), np.int32)
    for i0 in range(0, T, T_CHUNK):
        sl = {t_dim: slice(i0, min(i0 + T_CHUNK, T))}
        g = ds["B03"].isel(**sl).values.astype(np.float32)
        n = ds["B08"].isel(**sl).values.astype(np.float32)
        scl = ds["SCL"].isel(**sl).values
        clear = np.isin(scl, CLEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            wet = clear & ((g - n) / np.maximum(g + n, 1) > 0)
        wet_n += wet.sum(axis=0, dtype=np.int32)
        clr_n += clear.sum(axis=0, dtype=np.int32)
    wf = np.where(clr_n > 30, wet_n / np.maximum(clr_n, 1), np.nan)
    inter = np.isfinite(wf) & (wf > 0.10) & (wf < 0.90)
    sea = np.isfinite(wf) & (wf >= 0.90)
    keep = np.where(inter.ravel())[0]
    rows, cols = keep // W, keep % W

    Y = np.full((T, len(keep)), np.nan, np.float32)
    C = np.zeros((T, len(keep)), bool)
    for i0 in range(0, T, T_CHUNK):
        i1 = min(i0 + T_CHUNK, T)
        sl = {t_dim: slice(i0, i1)}
        g = ds["B03"].isel(**sl).values.astype(np.float32)
        n = ds["B08"].isel(**sl).values.astype(np.float32)
        scl = ds["SCL"].isel(**sl).values
        clear = np.isin(scl, CLEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndwi = np.where(g + n != 0, (g - n) / (g + n), np.nan)
        Y[i0:i1] = ndwi[:, rows, cols]
        C[i0:i1] = clear[:, rows, cols]

    import pyproj
    crs_cube = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt")
                                   or ds["crs"].attrs.get("spatial_ref"))
    tr = pyproj.Transformer.from_crs(crs_cube, 4326, always_xy=True)
    xs, ys = ds["x"].values, ds["y"].values
    lons, lats = tr.transform([xs.min(), xs.max()], [ys.min(), ys.max()])
    bbox = {"west": float(min(lons)), "south": float(min(lats)),
            "east": float(max(lons)), "north": float(max(lats)),
            "crs": "EPSG:4326"}
    return dict(Y=Y, C=C, keep=keep, dates=dates, shape=(H, W), sea=sea,
                inter=inter, bbox=bbox, crs_wkt=str(crs_cube.to_wkt()))


def invert_series(Y, C, h, lo=None, hi=None, mu_points=50, chunk=4000,
                  min_obs=8, min_b=0.15, max_b=2.5, max_a=2.5, min_bracket=0):
    """Step 10: elevation ``z`` and width ``sigma`` of pixels that share one level series.

    For every pixel, fits NDWI = a + b * Phi((h - z) / sigma) (METHOD Eq. 1)
    to its clear looks by least squares: ``z`` runs over ``mu_points`` levels
    evenly spaced from ``lo`` to ``hi``, ``sigma`` over :data:`SG_GRID`
    (0.03 to 0.65 m), and for each (z, sigma) pair ``a`` and ``b`` are solved
    in closed form (:func:`pyintertidal.elevation._fit_block`). The pair with
    the smallest sum of squared residuals is kept, so ``z`` is always one of
    the grid levels (a step of (hi - lo) / (mu_points - 1): 6.8 cm at
    Villaviciosa) and ``sigma`` one of the eight widths.

    Parameters
    ----------
    Y : (T, P) array
        NDWI of each look (scene t, pixel p). It must be finite wherever
        ``C`` is set: :func:`reconstruct` passes NaN as 0. A NaN on a clear
        look leaves that pixel without an elevation.
    C : (T, P) bool or float array
        1 for a clear look, 0 otherwise (float values act as weights).
    h : (T,) array
        The water level of each look, in metres: the mouth level already
        delayed by these pixels' delay. ``z`` comes out in its datum.
    lo, hi : float, optional
        Range of the ``z`` grid (default: the minimum and maximum of ``h``).
        :func:`reconstruct` passes the range of the UNDELAYED mouth level for
        every delay; pass the same to reproduce its ``z``.
    mu_points : int
        Number of ``z`` levels (50 in the product).
    chunk : int
        Pixels fitted per block (memory only).
    min_obs, min_b, max_b, max_a
        The guards: a pixel keeps its fit only with at least ``min_obs``
        clear looks, a dry-to-wet jump ``min_b < b < max_b`` and a dry level
        ``|a| < max_a``.
    min_bracket : int
        The quality rule (:data:`QUALITY_MIN_BRACKET` in the products): keep
        ``z`` only with at least this many clear looks whose level ``h`` is
        below it and as many above it. 0 (the default here) switches it
        off; :func:`reconstruct` uses 3 by default, and the Villaviciosa
        product was made with 0.

    Returns
    -------
    (z, sigma) : two (P,) float64 arrays
        Metres; NaN where a guard rejects the pixel.

    Examples
    --------
    One delay for every pixel, on the range of the undelayed level ``h0``:

    >>> z, sg = invert_series(Y, C, h_delayed, lo=h0.min(), hi=h0.max(),
    ...                       min_bracket=3)              # doctest: +SKIP

    Notebook 01 does exactly this on ``data/villaviciosa_sample.nc`` and
    reproduces the product's ``z`` and ``sigma`` bit for bit.
    """
    lo = float(np.min(h)) if lo is None else lo
    hi = float(np.max(h)) if hi is None else hi
    z = np.full(Y.shape[1], np.nan)
    sg_out = np.full(Y.shape[1], np.nan)
    grid = np.linspace(lo, hi, mu_points)
    for j in range(0, Y.shape[1], chunk):
        s = slice(j, min(j + chunk, Y.shape[1]))
        a, b, mu, sg, _, N = _fit_block(np.asarray(Y[:, s], np.float64),
                                        np.asarray(C[:, s], np.float64),
                                        h, grid, SG_GRID)
        ok = ((N >= min_obs) & (b > min_b) & (b < max_b)
              & (np.abs(a) < max_a))
        if min_bracket:
            clear = np.asarray(C[:, s]) > 0
            hh = np.asarray(h, float)[:, None]
            below = (clear & (hh < mu[None, :])).sum(axis=0)
            above = (clear & (hh > mu[None, :])).sum(axis=0)
            ok &= (below >= min_bracket) & (above >= min_bracket)
        z[s] = np.where(ok, mu, np.nan)
        sg_out[s] = np.where(ok, sg, np.nan)
    return z, sg_out


_invert = invert_series          # the name reconstruct calls below: the same function


def build_bank(lat, lon, times, bank_taus=None, model="EOT20",
               tide_dir="tide_models"):
    """The bank of delayed mouth levels from a tide model, in ONE model call.

    Used by :func:`reconstruct` only when it is given no ``boundary``.

    Every consumer used to make one ``model_tides`` call per shift (13+),
    each re-reading the constituent grids from disk — measured as ~80 % of
    a cell's wall time. All shifted instants go into a single prediction
    over the sorted unique times; each series is then a positional lookup.

    Parameters
    ----------
    lat, lon : float
        The point where the tide is predicted (degrees, WGS 84).
    times : pandas.DatetimeIndex
        The overpass instants, naive UTC.
    bank_taus : sequence of float, optional
        The delays (min) of the bank (default :data:`BANK_TAUS`).
    model, tide_dir : str
        The pyTMD model name and the folder of its files.

    Returns
    -------
    (bank, rising)
        ``bank``: a :class:`~pyintertidal.tide_estimators.ShiftBank` of the
        level (m) read each delay earlier; ``rising``: (T,) bool, the tide is
        rising at each instant (level 30 min later above the level 30 min
        earlier). :func:`reconstruct` does not use ``rising``.
    """
    import pandas as pd
    from .boundary import model_tides          # keeps sys.stdout intact (eo-tides/colorama)
    from . import tide_estimators as te

    bank_taus = list(BANK_TAUS if bank_taus is None else bank_taus)
    shifted = [times - pd.Timedelta(minutes=float(tv)) for tv in bank_taus]
    limb = [times - pd.Timedelta(minutes=30),
            times + pd.Timedelta(minutes=30)]
    all_t = pd.DatetimeIndex(np.unique(np.concatenate(
        [s.values for s in shifted + limb])))
    h_all = model_tides(x=[lon], y=[lat], time=all_t, model=model,
                        directory=tide_dir, crs="EPSG:4326",
                        extrapolate=True, cutoff=np.inf,
                        parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    def at(tt):
        pos = np.searchsorted(all_t.values, tt.values)
        return h_all[np.minimum(pos, len(h_all) - 1)]

    bank = te.ShiftBank(bank_taus, [at(s) for s in shifted])
    rising = (at(limb[1]) - at(limb[0])) > 0
    return bank, rising


#: The delays (min) at which the mouth level is precomputed for the delay
#: search (step 8). Between two of them the level is interpolated linearly;
#: outside -45 to 180 min it is clipped to the end value, so no section delay
#: beyond 180 min can be measured.
BANK_TAUS = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0,
             75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
# 180 min, not 120: in a long tidal basin the inner bands can reach a
# 120-min bound, and a lag pinned at the bound is not a measurement
#: Bounds (min) passed to ``tide_estimators.m2a_rasch``. With
#: ``monotone=True``, as reconstruct calls it, only the upper value is used,
#: and it bounds each section's increment over the previous section (0 to
#: 180 min), not the total; the bank above caps the total at 180 min.
TAU_BOUNDS = (-45.0, 180.0)


def reconstruct(cube_path, out_dir, name=None, n_bands=6, pixel_m=10.0,
                tau_detect_min=TAU_DETECT_MIN, model="EOT20",
                tide_dir="tide_models", region_mask=None, boundary=None,
                extraction=None, mouth_side="any", verbose=True,
                band_km=3.0, min_px_band=500, n_boot=30, min_scenes=80,
                overpass_times=None, quality_min_bracket=QUALITY_MIN_BRACKET):
    """MAREA steps 7–10 for one cube: the delay of every pixel and its elevation.

    It also makes the hypsometry of step 11, and reads the tide (step 6)
    itself only without a ``boundary``.

    Writes ``out_dir``/{marea.npz, result.json, figure.png} and returns the
    content of result.json as a dict, or None when the run is refused;
    result.json then holds the reason in ``estado``: ``"sin_intermareal"``
    with fewer than 3,000 intertidal pixels, ``"pocas_escenas"`` with fewer
    than ``min_scenes`` scenes.

    What it does, in order:

    1. the looks: ``extraction`` if given, else :func:`extract` on the whole
       cube — every scene, with no cloud screening and no epoch (the record
       of the published product comes from ``scripts/marea_tools.screen``
       and ``build_record``; ``scripts/quickstart.py`` chains them);
    2. the mouth (the permanent sea on the frame's edge ``mouth_side``) and
       the distance ``s`` of every pixel from it, through water and
       intertidal pixels only (with ``region_mask``, only the region's
       intertidal pixels). With no permanent sea on that edge there is
       no mouth, and the run goes on without distances or delays, with
       ``estado`` ``"ok"``: result.json then shows ``s_max_km`` null,
       ``centros_km`` [0.0] and ``con_operador`` false (see
       :func:`pyintertidal.geometry.mouth_seeds`);
    3. the scenes with an overpass time and their mouth level, read from
       ``boundary`` (else from the tide model ``model`` at the centre of the
       frame's box);
    4. sections of ``band_km`` along ``s``, thinner ones (``min_px_band``)
       merged; section 0, at the mouth, is pinned at 0 min and each further
       section adds a non-negative increment
       (``tide_estimators.m2a_rasch``, METHOD Eq. 3), with a 95 % interval
       from ``n_boot`` scene resamples. With 5,000 pixels or fewer having a
       distance ``s``, no delay is fitted and every pixel keeps the mouth
       tide. A single section (flats that lie, after thin sections are
       merged, within one ``band_km`` of the mouth: a short estuary, or a
       short branch passed as ``region_mask``) currently raises
       ``ValueError`` in the delay fit, which then has no delay to fit;
    5. the delay of every pixel: the section delays smoothed along ``s``
       (Gaussian kernel of ``band_km``), 0 at the mouth, never decreasing
       inland, rounded to the minute; pixels without ``s`` keep 0;
    6. for each distinct delay, :func:`invert_series` on the mouth level
       read that many minutes earlier, ``z`` searched within the minimum and
       maximum of the undelayed level, with the guards and the quality rule
       ``quality_min_bracket``;
    7. the hypsometric curve, with ``sigma`` as each pixel's jitter.

    Parameters
    ----------
    cube_path : str
        The cube (see :func:`extract`). When ``extraction`` is given it is not
        read for data: only its sha256 goes into result.json
        (``inputs_sha.cube``); notebook 01 passes the kit's record .npz here.
    out_dir : str
        Output folder, created if needed.
    name : str, optional
        Site name in the messages and in result.json (default: the cube's
        file name).
    n_bands : int
        Has no effect in this edition: the estuary's length sets the number
        of sections (``band_km``).
    pixel_m : float
        Pixel size (m) of the cube's projected grid, for ``s`` and the areas.
    tau_detect_min : float
        At the default 0 (:data:`TAU_DETECT_MIN`) it has no effect in this
        edition: every fitted section delay is applied. Leave it there. A
        positive value sets every section delay whose size is at most that
        many minutes to 0.
    model, tide_dir : str
        The pyTMD tide model and the folder of its files, used only when
        ``boundary`` is None.
    region_mask : (H, W) bool array, optional
        Restricts the INTERTIDAL pixels to one region. The mouth anchor still
        comes from the whole frame's permanent sea, but the geodesic distance
        ``s`` runs only through the permanent sea and the region's own
        intertidal pixels: flats outside the mask are not crossed, so a
        region joined to the mouth only through them gets longer distances,
        or none (then no delay is fitted, and result.json shows
        ``s_max_km`` null). Keep in the mask the flats that join the region
        to the mouth. This is how a branched system is analysed honestly:
        bands of ``s`` over a whole ría put different branches — different
        clocks — into the same band, and their lags average away. Banding
        one branch at a time removes the cancellation without touching the
        estimator.
    boundary : BoundaryProvider, optional
        The water level at the mouth: anything whose ``levels(times)``
        returns metres for naive-UTC instants — ``PyTMDBoundary`` (a tide
        model), ``GaugeBoundary`` (a record), ``make_boundary`` (model,
        nearest gauges, or their consensus) or ``marea_tools.TableBoundary``.
        ``z`` comes out in its datum. Recommended: without it the model is
        read at the centre of the frame's box, which may lie inland.
    extraction : dict, optional
        The dict :func:`extract` returns for this cube, usually restricted to
        the record by ``marea_tools.build_record``. Saves a second pass over
        the cube.
    mouth_side : str
        Which frame edge(s) count as the sea: ``"any"``, or ``"north"``,
        ``"south"``, ``"east"``, ``"west"`` joined by ``"+"`` (see
        :func:`pyintertidal.geometry.mouth_seeds`). ``"any"`` suits a box
        that only touches the sea on one side.
    verbose : bool
        Print the progress.
    band_km : float
        Length of a section and bandwidth of the smoothing (km).
    min_px_band : int
        A section with fewer intertidal pixels joins the next one.
    n_boot : int
        Scene resamples for the 95 % interval of each section's delay (0: no
        interval, several times faster).
    min_scenes : int
        The record must hold at least this many scenes with an overpass
        time, or the run is refused. The default 80 is the historical floor;
        a record restricted by a scene rule (e.g. the transition-zone cloud
        screening) can be shorter: the Villaviciosa product used 40.
    overpass_times : dict, optional
        ``{'YYYY-MM-DD': UTC timestamp}`` of the acquisitions, e.g. the
        site's cached overpass_times.json; the keys are date strings as in
        ``extraction['dates']``. When given, it replaces the live STAC
        query, so the record's instants are exactly the ones the caller uses
        elsewhere and no network is needed. Without it the query covers
        2016-01-01 to 2025-12-31 only: later scenes are left out.
    quality_min_bracket : int
        The quality rule of :func:`invert_series` (see
        :data:`QUALITY_MIN_BRACKET`; 0 disables it). The Villaviciosa
        product was made with 0: 29,728 elevations, against 28,631 with 3.

    Returns
    -------
    dict or None
        The content of result.json (see the module docstring for its keys),
        or None when the run is refused.

    Notes
    -----
    ``marea.npz`` holds, per intertidal pixel (P = len(keep)): ``z`` and
    ``sigma`` (float32, m; NaN: no elevation), ``s_km`` (float32), ``band``
    (int16; -1: no distance; all 0 when no delay was fitted) and
    ``tau_px_min`` (float32, min); per section:
    ``tau_min`` (fitted), ``tau_usado_min`` (applied), ``tau_lo_min`` and
    ``tau_hi_min`` (the 95 % interval) and ``tau_boot_min`` (n_boot x
    sections); ``keep`` and ``shape`` to put the per-pixel arrays back on the
    frame (``img = np.full(H * W, np.nan); img[keep] = z``, then
    ``img.reshape(H, W)``); and the hypsometric curve ``hyp_z``, ``hyp_area``,
    ``hyp_lo``, ``hyp_hi``.

    The settings of the delay search come from ``configs/m2.yaml``, read from
    the source tree: run it from a checkout of the repository.

    Examples
    --------
    A cube and a tide-gauge series (naive-UTC times, metres):

    >>> from pyintertidal import marea
    >>> from pyintertidal.boundary import GaugeBoundary
    >>> tide = GaugeBoundary(gauge_times, gauge_levels_m, name="gauge")   # doctest: +SKIP
    >>> res = marea.reconstruct("cubes/site.nc", "runs/site", boundary=tide,
    ...                         overpass_times=times, min_scenes=40)      # doctest: +SKIP

    The published Villaviciosa product, offline, from the replication kit:
    notebook 01, section 4, or
    ``python -X utf8 scripts/reproduce_villaviciosa.py --from-kit replication``.
    """
    import pandas as pd
    from .net import use_system_certificates
    from . import overpass

    t0 = time.time()
    name = name or os.path.splitext(os.path.basename(cube_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    use_system_certificates()

    ex = dict(extraction) if extraction is not None else extract(cube_path)
    H, W = ex["shape"]
    keep, sea, inter = ex["keep"], ex["sea"], ex["inter"]
    if region_mask is not None:
        rm = np.asarray(region_mask, bool)
        sel = rm.ravel()[keep]
        keep = keep[sel]
        ex["Y"] = ex["Y"][:, sel]
        ex["C"] = ex["C"][:, sel]
        inter = inter & rm
        if verbose:
            print(f"[{name}] region mask keeps {len(keep):,} of "
                  f"{sel.size:,} intertidal px", flush=True)
    if verbose:
        print(f"[{name}] {len(keep):,} intertidal px", flush=True)
    if len(keep) < 3000:
        json.dump({"sitio": name, "estado": "sin_intermareal",
                   "n_px": int(len(keep))},
                  open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        return None

    # geometry (step 7): the mouth and the distance s along the water. When no
    # permanent sea touches the chosen edge, mouth_seeds raises, s is NaN
    # everywhere and no delay is fitted (every pixel keeps the mouth tide)
    try:
        seeds = geometry.mouth_seeds(sea, side=mouth_side)
        s_m = geometry.along_distance(sea | inter, seeds, pixel_m)
        s_km = s_m.ravel()[keep] / 1000.0
    except ValueError:
        s_km = np.full(len(keep), np.nan)

    # the acquisition time of every scene (UTC); a scene without one leaves
    # the record. The STAC fallback covers 2016-2025 only
    if overpass_times is not None:
        times = {str(k): pd.Timestamp(v).to_pydatetime()
                 for k, v in dict(overpass_times).items()}
        overpass_source = "given"
    else:
        times = overpass.get_overpass_times(
            ex["bbox"], ("2016-01-01", "2025-12-31"), verbose=False)
        overpass_source = "stac"
    have = np.array([s in times for s in ex["dates"]])
    if have.sum() < min_scenes:
        json.dump({"sitio": name, "estado": "pocas_escenas",
                   "n_escenas": int(have.sum()),
                   "min_scenes": int(min_scenes)},
                  open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        return None
    # the record's instants as naive UTC (a time zone is dropped, not converted:
    # pass UTC), its looks with NaN NDWI as 0, and the wet looks (clear, NDWI > 0)
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in ex["dates"][have]])).tz_localize(None)
    Y = np.nan_to_num(ex["Y"][have], nan=0.0).astype(np.float64)
    C = (ex["C"][have]).astype(np.float64)
    wet = (C > 0) & (Y > 0)
    lat_c = 0.5 * (ex["bbox"]["south"] + ex["bbox"]["north"])
    lon_c = 0.5 * (ex["bbox"]["west"] + ex["bbox"]["east"])

    # the tide at the mouth (step 6) as a bank: the level read tau minutes
    # earlier at every scene, for each tau of BANK_TAUS (linear in between);
    # h0 is the level at the overpass instants themselves
    if boundary is not None:
        # any BoundaryProvider (model, N gauges, consensus): the bank is
        # its level read tau minutes earlier, for the standard tau grid
        bank = te.ShiftBank(BANK_TAUS, [
            np.asarray(boundary.levels(t_real - pd.Timedelta(minutes=tv)),
                       float) for tv in BANK_TAUS])
    else:
        bank, _rising = build_bank(lat_c, lon_c, t_real, model=model,
                                   tide_dir=tide_dir)
    h0 = bank.at(0.0)

    # interior tide (only with a usable mouth anchor): steps 7-8, one delay
    # per section (METHOD Eq. 3), fitted only when more than 5,000 pixels
    # have a distance s
    if np.isfinite(s_km).sum() > 5000:
        # bands of fixed length along the mouth distance (``n_bands`` has
        # no effect in this edition; the estuary's length sets the count),
        # monotone lag profile, scene-bootstrap interval.
        # The settings come from configs/m2.yaml, read from the source tree:
        # seed, sigma0_m (unused once sigma_perfil_m is given), sigma_perfil_m
        # (six widths profiled per pixel), z_puntos (60 elevations over the
        # mouth range +-0.3 m) and max_px_banda.m2a (1,200 seeded pixels per
        # section). Its other keys are not read here.
        edges, centers, band_of = te.make_bands_fixed(s_km, band_km, min_px_band)
        nb = len(centers)
        import yaml
        cfg2 = yaml.safe_load(open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "m2.yaml"), encoding="utf-8"))
        fit_kw = dict(sigma0=cfg2["sigma0_m"],
                      sg_grid=tuple(cfg2["sigma_perfil_m"]),
                      tau_bounds=TAU_BOUNDS, z_points=cfg2["z_puntos"],
                      max_px_band=cfg2["max_px_banda"]["m2a"], monotone=True)
        r2a = te.m2a_rasch(wet, C > 0, bank, band_of, centers, nb,
                           rng=np.random.default_rng(cfg2["seed"] + 31), **fit_kw)
        tau = np.asarray(r2a["tau"], float)
        if n_boot:
            # the 95 % interval: refit on n_boot resamples of the scenes, with
            # 600 pixels per section (half the point fit's 1,200, for speed)
            t_b = time.time()
            boot = te.bootstrap_lags(wet, C > 0, bank, band_of, centers, nb,
                                     n_boot=n_boot, seed=cfg2["seed"],
                                     **{**fit_kw, "max_px_band": 600})
            tau_lo = np.nanpercentile(boot, 2.5, axis=0)
            tau_hi = np.nanpercentile(boot, 97.5, axis=0)
            if verbose:
                print(f"[{name}] bootstrap of {n_boot} scene resamples "
                      f"[{time.time() - t_b:.0f} s]", flush=True)
        else:
            boot = None
            tau_lo = tau_hi = np.full(nb, np.nan)
    else:
        # no mouth anchor: one section, delay 0, every pixel in section 0
        # (also the pixels without a distance)
        centers = np.array([0.0])
        band_of = np.zeros(len(keep), int)
        tau = np.array([0.0])
        boot = None
        tau_lo = tau_hi = np.array([np.nan])
    # With tau_detect_min = 0 (the default) this line changes nothing.
    tau_used = np.where(np.abs(tau) > tau_detect_min, tau, 0.0)
    # Causality. The interior lag is a propagation delay: non-negative. A
    # band whose fitted lag is negative (the interior running AHEAD of the
    # ocean) is clipped to 0, band by band; the other bands keep their
    # measured lag. (Until 2026-09-23 a negative band above the detection
    # threshold vetoed the whole profile; with the threshold gone that rule
    # would zero every profile with one noisy band, so it was replaced.)
    # m2a_rasch with monotone=True, as called above, returns lags >= 0 by
    # construction, so this clip (and veto_fisico) only acts for a fit with
    # monotone=False.
    neg = tau_used < 0
    veto = bool(neg.any())
    tau_used = np.where(neg, 0.0, tau_used)
    if verbose:
        print(f"[{name}] tau={np.round(tau, 1)} -> "
              f"applied {np.round(tau_used, 1)}"
              + (f"  [causality: {int(neg.sum())} negative band(s) clipped to 0]"
                 if veto else ""), flush=True)

    # The lag every pixel receives: the band profile smoothed along the
    # mouth distance (Gaussian kernel, bandwidth = band length), anchored at
    # 0 at the mouth, rounded to the minute. Pixels without a mouth
    # distance keep the boundary's clock.
    if len(centers) > 1:
        tau_px = te.smooth_lag_profile(centers, tau_used, s_km, bandwidth_km=band_km)
    else:
        tau_px = np.zeros(len(keep))
    tau_px = np.where(np.isfinite(tau_px), np.round(tau_px), 0.0)
    if verbose:
        print(f"[{name}] per-pixel lag: median {np.median(tau_px):.0f} min, "
              f"max {tau_px.max():.0f} min, {int((tau_px > 0).sum()):,} px shifted", flush=True)

    # step 10: the elevation and width of every pixel on its own delayed tide,
    # one invert_series call per distinct delay. The z grid spans the
    # UNDELAYED mouth level, the same for every delay, so every z is one of
    # the same 50 levels
    lo, hi = float(h0.min()), float(h0.max())
    z = np.full(len(keep), np.nan)
    sg = np.full(len(keep), np.nan)
    for tv in np.unique(tau_px):
        cols = np.where(tau_px == tv)[0]
        if not len(cols):
            continue
        if not tv:
            h_k = h0
        elif boundary is not None:
            # the exact shifted level, not the bank's linear interpolation
            # between 15-min shifts (a few cm at most, but the product and
            # a notebook re-inverting the same record should agree exactly)
            h_k = np.asarray(boundary.levels(
                t_real - pd.Timedelta(minutes=float(tv))), float)
        else:
            h_k = bank.at(float(tv))
        z[cols], sg[cols] = _invert(Y[:, cols], C[:, cols], h_k, lo, hi,
                                    min_bracket=quality_min_bracket)

    # hypsometry with uncertainty (sigma as the per-pixel jitter): each z is
    # jittered by its own width, 30 draws, 5-95 % envelope
    from .hypsometry import curve_with_uncertainty
    hyp = curve_with_uncertainty(z, pixel_m=pixel_m, sigma_z=sg)

    np.savez_compressed(
        os.path.join(out_dir, "marea.npz"),
        z=z.astype(np.float32), sigma=sg.astype(np.float32),
        s_km=s_km.astype(np.float32), band=band_of.astype(np.int16),
        keep=keep, shape=np.array([H, W]),
        tau_min=tau, tau_usado_min=tau_used,
        tau_lo_min=np.asarray(tau_lo, float), tau_hi_min=np.asarray(tau_hi, float),
        tau_px_min=tau_px.astype(np.float32),
        band_km=np.array([band_km]),
        **({"tau_boot_min": boot} if boot is not None else {}),
        hyp_z=hyp["z"], hyp_area=hyp["area_km2"],
        hyp_lo=hyp["area_lo"], hyp_hi=hyp["area_hi"])
    result = {
        "sitio": name, "estado": "ok",
        "n_escenas": int(have.sum()), "n_px": int(len(keep)),
        "record_dates": [str(s) for s in ex["dates"][have]],
        "min_scenes": int(min_scenes), "overpass_source": overpass_source,
        "n_px_cota": int(np.isfinite(z).sum()),
        "s_max_km": (float(np.nanmax(s_km))
                     if np.isfinite(s_km).any() else None),
        "band_km": float(band_km), "min_px_band": int(min_px_band),
        "quality_min_bracket": int(quality_min_bracket),
        "centros_km": np.asarray(centers).tolist(),
        "tau_min": np.asarray(tau).tolist(),
        "tau_usado_min": np.asarray(tau_used).tolist(),
        "tau_lo_min": np.asarray(tau_lo, float).tolist(),
        "tau_hi_min": np.asarray(tau_hi, float).tolist(),
        "n_boot": int(n_boot),
        "tau_px_median_min": float(np.median(tau_px)),
        "tau_px_max_min": float(tau_px.max()),
        "con_operador": bool((tau_px != 0).any()),
        "veto_fisico": veto,
        "hipsometria": {"integral": hyp["integral"],
                        "area_total_km2": hyp["area_total_km2"]},
        "rango_marea_m": [lo, hi],
        "inputs_sha": {"cube": seal._sha256(cube_path)},
        "duracion_s": round(time.time() - t0, 1),
    }
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"),
              indent=1)

    import matplotlib
    # headless campaigns need Agg, but inside a notebook this call would
    # kill the inline backend for every later cell — switch only if no
    # interactive backend is already live
    if "inline" not in matplotlib.get_backend():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    img = np.full(H * W, np.nan, np.float32)
    img[keep] = z
    im = ax[0].imshow(img.reshape(H, W), cmap="viridis")
    ax[0].set_title(f"{name}: cota MAREA (m)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].plot(centers, tau, "o-", color="C0", label="medido")
    ax[1].plot(centers, tau_used, "s--", color="C2", label="applied")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("s (km)"); ax[1].set_ylabel("tau (min)")
    ax[1].legend(fontsize=8); ax[1].set_title("marea interior")
    ax[2].fill_betweenx(hyp["z"], hyp["area_lo"], hyp["area_hi"],
                        alpha=0.25, color="C0")
    ax[2].plot(hyp["area_km2"], hyp["z"], color="C0")
    ax[2].set_xlabel("area acumulada (km2)"); ax[2].set_ylabel("cota (m)")
    ax[2].set_title(f"hipsometria (HI={hyp['integral']:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure.png"), dpi=130)
    plt.close(fig)
    if verbose:
        print(f"[{name}] listo en {time.time()-t0:.0f} s -> {out_dir}",
              flush=True)
    return result
