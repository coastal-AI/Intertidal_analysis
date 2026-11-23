"""
pyintertidal — MAREA and the tools around it (CARTOGRASS 2026 edition)
======================================================================

*MAREA: Mudflat Altimetry via Remotely-Estimated tides from the Archive.*

Maps the elevation of tidal flats at 10 m from Sentinel-2 openEO data cubes,
with the width of each pixel's wet/dry transition (sigma) and its emersion
time. The delay of the tide inside the estuary is read from the images
themselves.

This edition holds the code that the method, its scripts and its notebooks
use, the general tools around it (hypsometry, terrain, export, ...) and a few
alternative delay estimators kept for comparison (marked as such in
:mod:`~pyintertidal.tide_estimators`).

Where to start
--------------
* Reproduce the published Villaviciosa map offline (about a minute)::

      python -X utf8 scripts/reproduce_villaviciosa.py --from-kit replication

* Map a new estuary from a cube and a tide series: ``scripts/quickstart.py``
  (function ``map_estuary``).
* Download and map a new ría through openEO:
  ``notebooks/03_new_ria_openeo.ipynb``.
* The method in words: ``guide/METHOD.md``; the products: ``guide/DATA.md``.

Words and units
---------------
* **look** — one clear view of a pixel (Sentinel-2 SCL class 4 to 7) and its
  NDWI; a look is wet when NDWI > 0.
* **record** — the screened scenes of the product's epoch that have an
  acquisition time and a tide level at the mouth.
* **section** (called **band** in the code: ``band_km``, ``band_of``, the
  ``band`` layer) — a stretch of about 3 km along the water from the mouth;
  each section has one delay.
* **delay** ``tau`` — minutes by which the tide inside is later than at the
  mouth (positive = later).
* **sigma** — the width (m) of a pixel's dry-to-wet transition; not an
  accuracy measured against a survey.
* Times are UTC; heights are metres on the scale of the tide at the mouth
  (EOT20: about mean sea level); grids are the cube's UTM grid, rows north to
  south.

Design principles
-----------------
1. **No magic.** Each function does ONE scientific step, takes explicit
   parameters and returns plain arrays, so a notebook shows the whole method.
2. **One download, then streaming.** Imagery is fetched once per area as a
   cached datacube; every product streams that cube in bounded memory.
3. **Polygons first.** Study areas are polygons; bounding boxes are an
   internal download detail.

The MAREA path: the 11 steps of ``scripts/marea_tools.STEPS``
-------------------------------------------------------------
* (1) download the cube — ``scenes.connect``, ``cube.SentinelCube(...).ensure``
* (2) acquisition times — ``overpass.get_overpass_times`` (live) or
  ``overpass.load_overpass`` (a cached file)
* (3) screen the scenes — ``stability.reference_and_clouds``,
  ``stability.filter_dates`` (``scripts/marea_tools.screen`` calls both)
* (4) the looks and the intertidal mask — ``marea.extract``
* (5) the record — ``scripts/marea_tools.build_record``
* (6) the tide at the mouth — :mod:`~pyintertidal.boundary`
* (7)-(10) and the hypsometry of (11): the mouth, the sections, their delays,
  every pixel's delay and elevation — ``marea.reconstruct`` (it calls
  geometry, tide_estimators, elevation, hypsometry and seal; it reads the
  tide (6) itself only without a ``boundary``)
* (11) emersion time and GeoTIFFs — ``scripts/marea_tools.emersion_time``,
  ``scripts/marea_tools.export_geotiffs``

The modules
-----------
=====================================  =======================================  =========
module                                 what it does                             MAREA
=====================================  =======================================  =========
:mod:`~pyintertidal.sites`             named study areas                        tide point
:mod:`~pyintertidal.aoi`               areas of interest: polygons, masks       step 1
:mod:`~pyintertidal.scenes`            openEO connection, single scenes         step 1
:mod:`~pyintertidal.cube`              the cached datacube and its streaming    steps 1, 3
:mod:`~pyintertidal.overpass`          the acquisition time of every scene      step 2
:mod:`~pyintertidal.water`             water detection (NDWI and SCL)           step 3
:mod:`~pyintertidal.stability`         reference map and the cloud filter       step 3
:mod:`~pyintertidal.frequency`         water frequency, intertidal window       tool
:mod:`~pyintertidal.boundary`          the water level at the mouth             step 6
:mod:`~pyintertidal.gauges`            tide-gauge records                       step 6
:mod:`~pyintertidal.coverage`          how well the record samples the tide     report
:mod:`~pyintertidal.geometry`          the mouth, the distance along the water  step 7
:mod:`~pyintertidal.tide_estimators`   the delay of each section                steps 7-9
:mod:`~pyintertidal.elevation`         the per-pixel fit                        step 10
:mod:`~pyintertidal.marea`             MAREA end to end                         steps 4, 7–10 and the hypsometry of 11
:mod:`~pyintertidal.hypsometry`        area-elevation curve                     step 11
:mod:`~pyintertidal.hydroperiod`       time under water, emersion time          step 11
:mod:`~pyintertidal.terrain`           slope, hillshade, contours               tool
:mod:`~pyintertidal.export`            Cloud-Optimized GeoTIFF and STAC         tool
:mod:`~pyintertidal.raster`            GeoTIFF input and output                 step 11
:mod:`~pyintertidal.net`               TLS certificates on managed networks     (automatic)
:mod:`~pyintertidal.seal`              SHA-256 of the inputs (provenance)       step 11
=====================================  =======================================  =========

"tool" = not called by MAREA; there for maps, figures and other methods.

Quick start (offline, from the repository root)
-----------------------------------------------
Steps 6-11 of the published Villaviciosa product from the replication kit:
no account, no download, no tide model, about 20 seconds::

    import sys
    sys.path.insert(0, "scripts")                    # scripts/marea_tools.py
    import marea_tools as mt
    from pyintertidal import marea

    rec = "replication/villaviciosa_record_2023-2025.npz"
    res = marea.reconstruct(
        rec, "runs/quickstart", name="Villaviciosa",
        extraction=mt.load_extraction(rec),          # the looks of the record
        overpass_times=mt.load_overpass("products_marea/overpass_times.json"),
        boundary=mt.TableBoundary("replication/villaviciosa_eot20_levels.npz"),
        min_scenes=40, quality_min_bracket=0,        # the product's settings
        n_boot=0)                                    # skip the 30 bootstrap fits
    print(res["tau_usado_min"])                      # the delay of each section (min)

``runs/quickstart`` then holds marea.npz, result.json and figure.png; the
arrays equal those of ``products_marea/marea.npz`` (the delay intervals,
which need the bootstrap, are not computed).

Importing the package sets a legacy Windows console to replace the
characters it cannot print instead of raising (see ``_allow_unicode_output``).
"""

__version__ = "0.1.0"


# ── console set-up (runs once, on import) ───────────────────────────────────
def _allow_unicode_output():
    """Stop a legacy Windows console from killing a run over a print.

    The progress messages use arrows and middle dots. On Windows a plain
    ``python script.py`` gets a cp1252 stdout, which cannot encode them, and
    the process dies inside ``print`` — hours into a fit, with a traceback
    that points at the formatting rather than at anything real. Notebooks are
    UTF-8 and never see this. Degrading unencodable characters is strictly
    better than losing the computation.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower()
        if enc.replace("-", "") in ("utf8", "utf16", "utf32"):
            continue
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:                      # not a reconfigurable stream
            pass


_allow_unicode_output()

# ── names at the top level ──────────────────────────────────────────────────
# The areas, the cube, raster IO, the screening (step 3) and the older
# HSR/step elevation methods. MAREA itself is used through its module:
# ``from pyintertidal import marea`` (marea.extract, marea.reconstruct).
from .aoi import AOI, as_aoi, parse_dms
from .cube import SentinelCube, open_cube, auto_chunk
from .raster import write_geotiff, read_band, read_rgb, read_scl
from .stability import reference_and_clouds, filter_dates, date_recovery_gain
from .frequency import water_frequency, multiotsu_window, intertidal_mask
from .elevation import (fit_hsr, fit_step, epochs, ElevationResult,
                        extract_intertidal_ndwi)

# ── modules (import as namespaces, they are used by name) ───────────────────
from . import sites
from . import scenes
from . import water
from . import overpass
from . import coverage
from . import boundary
from . import gauges
from . import geometry
from . import tide_estimators
from . import marea
from . import hypsometry
from . import hydroperiod
from . import terrain
from . import export
from . import seal
from . import net

__all__ = [
    "AOI", "as_aoi", "parse_dms",
    "SentinelCube", "open_cube", "auto_chunk",
    "write_geotiff", "read_band", "read_rgb", "read_scl",
    "reference_and_clouds", "filter_dates", "date_recovery_gain",
    "water_frequency", "multiotsu_window", "intertidal_mask",
    "fit_hsr", "fit_step", "epochs", "ElevationResult",
    "extract_intertidal_ndwi",
    "sites", "scenes", "water", "overpass", "coverage", "boundary", "gauges",
    "geometry", "tide_estimators", "marea", "hypsometry", "hydroperiod",
    "terrain", "export", "seal", "net",
]
