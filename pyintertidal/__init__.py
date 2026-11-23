"""
pyintertidal — MAREA and the tools around it (CARTOGRASS 2026 edition)
======================================================================

*MAREA: Mudflat Altimetry via Remotely-Estimated tides from the Archive.*

Maps intertidal elevation, its uncertainty and emersion time at 10 m from
Sentinel-2 openEO data cubes, with the delay of the tide inside the estuary
read from the images themselves.

This edition holds the part of the library that the method and its example
notebooks use. The modules used only for validation, the experimental
variants and the superseded methods are not part of it.

Design principles
-----------------
1. **No magic.** Each function does ONE scientific step, takes explicit
   parameters and returns plain arrays, so a notebook shows the whole method.
2. **One download, then streaming.** Imagery is fetched once per area as a
   cached datacube; every product streams that cube in bounded memory.
3. **Polygons first.** Study areas are polygons; bounding boxes are an
   internal download detail.

The pipeline, module by module
------------------------------
=====================================  =======================================
:mod:`~pyintertidal.sites`             named study areas
:mod:`~pyintertidal.aoi`               areas of interest: polygons, masks
:mod:`~pyintertidal.scenes`            openEO connection, single scenes
:mod:`~pyintertidal.cube`              the cached datacube and its streaming
:mod:`~pyintertidal.overpass`          the acquisition time of every scene
:mod:`~pyintertidal.water`             water detection (NDWI and SCL)
:mod:`~pyintertidal.stability`         reference map and the cloud filter
:mod:`~pyintertidal.frequency`         water frequency and the intertidal mask
:mod:`~pyintertidal.boundary`          the water level at the mouth
:mod:`~pyintertidal.gauges`            tide-gauge records
:mod:`~pyintertidal.coverage`          how well the archive samples the tide
:mod:`~pyintertidal.geometry`          mouth and distance along the water
:mod:`~pyintertidal.tide_estimators`   the delay of each 3-km section
:mod:`~pyintertidal.elevation`         the per-pixel fit
:mod:`~pyintertidal.marea`             MAREA end to end
:mod:`~pyintertidal.hypsometry`        area-elevation curve
:mod:`~pyintertidal.hydroperiod`       emersion time
:mod:`~pyintertidal.terrain`           slope, hillshade, contours
:mod:`~pyintertidal.export`            Cloud-Optimized GeoTIFF and STAC
:mod:`~pyintertidal.raster`            GeoTIFF input and output
:mod:`~pyintertidal.net`               TLS certificates on managed networks
:mod:`~pyintertidal.seal`              SHA-256 of the inputs (provenance)
=====================================  =======================================

Quick start
-----------
>>> import pyintertidal as pit
>>> from pyintertidal import marea
>>> result = marea.reconstruct(cube_path, out_dir, boundary=boundary,
...                            extraction=record, overpass_times=times)
"""

__version__ = "0.1.0"


# ── core objects ────────────────────────────────────────────────────────────
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
