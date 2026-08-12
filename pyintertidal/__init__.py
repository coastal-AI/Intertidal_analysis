"""
pyintertidal — a transparent toolkit for intertidal-zone science
================================================================

Maps the intertidal zone from satellite imagery: where it is, how often it
floods, how high it stands, how long it is submerged, and how it changes.
Built around Sentinel-2 (via openEO / Copernicus) and global tide models.

Design principles
-----------------
1. **No magic.** There is no "run everything" function. Each function does
   ONE scientific step, takes explicit parameters and returns plain arrays,
   so a notebook shows the whole method in its cells — nothing important
   happens where the reader cannot see it.
2. **One download, then streaming.** Imagery is fetched once per area as a
   cached datacube; every product streams that cube in bounded memory. Long
   archives and large areas cost time, never RAM, and contiguous tiles run
   in parallel.
3. **Polygons first.** Study areas are polygons. Bounding boxes are an
   internal download detail; products and figures are clipped to the polygon
   so neighbouring tiles mosaic without seams.
4. **Show the data.** :mod:`pyintertidal.explain` draws the cube and every
   operation on it, in the visual language of the openEO documentation, and
   records the process graph that produced a result.
5. **One implementation per idea.** Where the project once had several
   versions of the same algorithm, the core keeps the one that measured best
   and :mod:`pyintertidal.legacy` keeps the superseded ones runnable for the
   published comparisons.

The pipeline, module by module
------------------------------
============================  ==============================================
:mod:`~pyintertidal.sites`    named study areas
:mod:`~pyintertidal.aoi`      areas of interest: polygons, masks, tiling
:mod:`~pyintertidal.scenes`   connect, list dates, download single images
:mod:`~pyintertidal.cube`     the cached datacube and its streaming
:mod:`~pyintertidal.water`    water detection (NDWI · MNDWI · AWEI · SCL)
:mod:`~pyintertidal.stability` reference map and the cloud filter
:mod:`~pyintertidal.frequency` water frequency and the intertidal window
:mod:`~pyintertidal.tides`    overpass times and tide heights
:mod:`~pyintertidal.coverage` is this site mappable? sampling diagnostics
:mod:`~pyintertidal.elevation` HSR and step elevation, with uncertainty
:mod:`~pyintertidal.terrain`  slope, aspect, hillshade, contours, roughness
:mod:`~pyintertidal.hydroperiod` submergence time and ecological zonation
:mod:`~pyintertidal.hypsometry` area–elevation signature of the estuary
:mod:`~pyintertidal.morphodynamics` change between epochs, sediment budgets
:mod:`~pyintertidal.shoreline` waterlines and shoreline change
:mod:`~pyintertidal.marsh`    flooded vegetation (salt marsh)
:mod:`~pyintertidal.validation` LiDAR references, metrics, field transects
:mod:`~pyintertidal.gauges`   tide-gauge comparison
:mod:`~pyintertidal.viz`      composable maps and profiles
:mod:`~pyintertidal.explain`  datacube diagrams and provenance graphs
:mod:`~pyintertidal.mosaic`   stitch tiles into one map
:mod:`~pyintertidal.export`   COG and STAC outputs
:mod:`~pyintertidal.report`   a self-contained HTML report per site
:mod:`~pyintertidal.legacy`   superseded methods, kept for the paper
============================  ==============================================

Quick start
-----------
>>> import pyintertidal as pit
>>> aoi = pit.sites.get("villaviciosa")
>>> cube = pit.SentinelCube(aoi, ("2016-01-01", "2025-12-31")).ensure(conn)
>>> pit.explain.describe_cube(cube)
>>> reference, clouds = pit.reference_and_clouds(cube, threshold=0.0)
>>> usable = pit.filter_dates(clouds, cloud_threshold=0.10)
>>> wf = pit.water_frequency(cube, dates=usable, threshold=0.0, min_obs=15)
"""

__version__ = "0.1.0"

# ── core objects ────────────────────────────────────────────────────────────
from .aoi import AOI, as_aoi, parse_dms
from .cube import SentinelCube, open_cube, auto_chunk
from .raster import write_geotiff, read_band, read_rgb, read_scl
from .stability import reference_and_clouds, filter_dates, date_recovery_gain
from .frequency import water_frequency, multiotsu_window, intertidal_mask
from .tides import TideService, water_centroid
from .elevation import fit_hsr, fit_step, epochs, ElevationResult
from .validation import (download_mdt_ign, reproject_to_grid, compare_dems,
                         transect_profile)

# ── modules (import as namespaces, they are used by name) ───────────────────
from . import aoi as aoi_module          # noqa: F401  (kept for completeness)
from . import sites
from . import scenes
from . import water
from . import terrain
from . import coverage
from . import hydroperiod
from . import hypsometry
from . import morphodynamics
from . import shoreline
from . import marsh
from . import gauges
from . import validation
from . import viz
from . import explain
from . import mosaic
from . import export
from . import report

__all__ = [
    "AOI", "as_aoi", "parse_dms",
    "SentinelCube", "open_cube", "auto_chunk",
    "write_geotiff", "read_band", "read_rgb", "read_scl",
    "reference_and_clouds", "filter_dates", "date_recovery_gain",
    "water_frequency", "multiotsu_window", "intertidal_mask",
    "TideService", "water_centroid",
    "fit_hsr", "fit_step", "epochs", "ElevationResult",
    "download_mdt_ign", "reproject_to_grid", "compare_dems",
    "transect_profile",
    "sites", "scenes", "water", "terrain", "coverage", "hydroperiod",
    "hypsometry", "morphodynamics", "shoreline", "marsh", "gauges",
    "validation", "viz", "explain", "mosaic", "export", "report",
]
