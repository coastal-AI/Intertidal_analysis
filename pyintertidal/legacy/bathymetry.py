"""
bathymetry.py — Server-side elevation methods (the comparison benchmark)
========================================================================

The four elevation estimators that the current
:mod:`pyintertidal.elevation` supersedes, kept runnable because the paper
compares against them. All of them execute INSIDE the OpenEO backend via the
verbatim UDFs in :mod:`pyintertidal.legacy.udf`; the class below only builds
the process graph, submits the job and assembles the result.

The methods, and how they scored against IGN 5 m LiDAR at Villaviciosa
(10 years, datum bias removed):

``optimized``  RMSE 1.14 m — bracketing + energy optimisation + diffusion
``pixels``     RMSE 3.35 m — pure bracketing (midpoint of the tide bracket)
``isolines``   RMSE 3.39 m — waterline interpolation
``step``       RMSE 0.85 m — DEA-style dry→wet crossing (also available as a
               local, cloud-free implementation in
               :func:`pyintertidal.elevation.fit_step`, which is what new
               work should use)
``siq``        not benchmarked — Soft Inundation Quantile, an experiment
               that inverts the tide CDF from a soft inundation frequency

For reference, the current default (:func:`pyintertidal.elevation.fit_hsr`)
scores RMSE 0.76 m and needs no cloud job at all.

Why keep server-side versions at all? They are the only way to reproduce the
published numbers, and they exercise a genuinely different execution model
(compute next to the data) that is worth benchmarking against local
streaming.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from ..aoi import as_aoi
from ..raster import write_geotiff
from ..terrain import slope, aspect, hillshade, contours
from .udf import BATHYMETRY_UDF_SCL, BATHYMETRY_UDF_NDWI

#: Elevation estimators available in the UDFs.
METHODS = ("optimized", "pixels", "isolines", "step", "siq")


@dataclass
class BathymetryResult:
    """Bands returned by the server-side reconstruction, plus derivatives."""

    elevation: np.ndarray
    confidence: np.ndarray
    residual: np.ndarray
    observation_count: np.ndarray
    transform: object
    crs: object
    method: str
    mask: np.ndarray | None = None
    slope: np.ndarray | None = None
    aspect: np.ndarray | None = None
    hillshade: np.ndarray | None = None
    contours: object | None = None

    def clip_to(self, mask):
        """Restrict the DEM (and derivatives) to a mask — normally the
        intertidal mask, so elevation is never reported outside it."""
        m = np.asarray(mask, dtype=bool)
        keep = np.zeros(self.elevation.shape, bool)
        h = min(m.shape[0], keep.shape[0])
        w = min(m.shape[1], keep.shape[1])
        keep[:h, :w] = m[:h, :w]
        for name in ("elevation", "confidence", "residual", "slope", "aspect",
                     "hillshade"):
            arr = getattr(self, name, None)
            if arr is not None and arr.shape == keep.shape:
                setattr(self, name, np.where(keep, arr, np.nan).astype(np.float32))
        self.mask = keep if self.mask is None else (self.mask & keep)
        return self

    def save(self, out_dir="."):
        """Write elevation, confidence and observation count as GeoTIFFs."""
        os.makedirs(out_dir, exist_ok=True)
        for name in ("elevation", "confidence", "residual", "observation_count"):
            write_geotiff(os.path.join(out_dir, f"{self.method}_{name}.tif"),
                          getattr(self, name), self.transform, self.crs,
                          "float32", np.nan)
        return out_dir


class BathymetryReconstructor:
    """Runs the legacy server-side elevation methods on OpenEO.

    Parameters
    ----------
    connection : openeo connection
        Authenticated backend connection.
    water_source : {"ndwi", "scl"}
        Which UDF to run. ``"ndwi"`` detects water with the 10 m index (and
        is the only source supporting ``step``/``siq``); ``"scl"`` is the
        original 20 m classification route.
    """

    def __init__(self, connection, water_source="ndwi"):
        if water_source not in ("ndwi", "scl"):
            raise ValueError("water_source must be 'ndwi' or 'scl'")
        self.connection = connection
        self.water_source = water_source

    # ── graph ───────────────────────────────────────────────────────────────
    def _build_cube(self, aoi, time_extent, tide_heights, valid_dates,
                    method, ndwi_threshold):
        """Assemble the OpenEO graph for one reconstruction.

        Uses ``apply_dimension`` rather than ``reduce_dimension`` because the
        UDF must see the whole time axis and REPLACE it with a 4-element
        ``bands`` dimension (elevation, confidence, residual, observations).
        """
        import openeo

        aoi = as_aoi(aoi)
        bands = ["SCL"] if self.water_source == "scl" else ["B03", "B08", "SCL"]
        cube = self.connection.load_collection(
            "SENTINEL2_L2A", spatial_extent=aoi.bbox,
            temporal_extent=list(time_extent), bands=bands, max_cloud_cover=100,
        )
        if self.water_source == "ndwi":
            try:
                cube = cube.resample_spatial(resolution=10, method="near")
            except Exception:
                pass

        # Pre-filtering to the usable dates is what keeps a decade-long job
        # inside the backend's memory limits (1379 → ~300 scenes).
        if valid_dates:
            try:
                cube = cube.filter_labels(
                    dimension="t", condition=lambda x: x.isin(list(valid_dates)))
            except Exception:
                pass

        context = {"tide_heights": tide_heights, "method": method,
                   "valid_dates": list(valid_dates) if valid_dates else None}
        if self.water_source == "ndwi":
            context["ndwi_threshold"] = float(ndwi_threshold)
        udf = openeo.UDF(
            code=(BATHYMETRY_UDF_NDWI if self.water_source == "ndwi"
                  else BATHYMETRY_UDF_SCL),
            runtime="Python", context=context,
        )
        return cube.apply_dimension(dimension="t", target_dimension="bands",
                                    process=udf)

    def _run(self, cube, out_path, title, verbose=True):
        """Submit the batch job with memory settings that survive 10 years.

        The UDF holds the whole temporal stack in RAM per executor, so the
        Python-side allowance has to be raised explicitly — the default kills
        long ranges with an out-of-memory error.
        """
        job = cube.save_result(format="GTiff").create_job(
            title=title,
            job_options={"executor-memory": "4G",
                         "executor-memoryOverhead": "2G",
                         "python-memory": "6G"},
        )
        if verbose:
            print(f"[legacy] running '{title}' on the backend…")
        job.start_and_wait()
        assets = job.get_results().get_assets()
        if not assets:
            raise RuntimeError(f"job '{title}' returned no assets")
        os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
        assets[0].download(str(out_path))
        return out_path

    # ── public API ──────────────────────────────────────────────────────────
    def reconstruct(self, aoi, time_extent, tide_heights, valid_dates=None,
                    method="optimized", out_dir="bathymetry_legacy",
                    ndwi_threshold=0.1, force=False, min_confidence=0.0,
                    derivatives=True, verbose=True):
        """Run one legacy method and return a :class:`BathymetryResult`.

        Parameters
        ----------
        method : str
            One of :data:`METHODS`.
        min_confidence : float
            Pixels at or below this confidence are treated as fill (NaN):
            the UDF writes a value everywhere, but only bracketed pixels
            carry information.
        derivatives : bool
            Also compute slope/aspect/hillshade/contours
            (:mod:`pyintertidal.terrain`).
        """
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        if method in ("step", "siq") and self.water_source != "ndwi":
            raise ValueError(f"method {method!r} requires water_source='ndwi'")

        out_dir = Path(out_dir)
        dem_path = out_dir / f"bathymetry_{method}.tif"
        if force or not dem_path.exists():
            cube = self._build_cube(aoi, time_extent, tide_heights,
                                    valid_dates, method, ndwi_threshold)
            self._run(cube, dem_path, f"bathymetry_{method}", verbose=verbose)
        elif verbose:
            print(f"[legacy] reusing {dem_path}")

        with rasterio.open(dem_path) as src:
            elevation = src.read(1).astype(np.float32)
            confidence = src.read(2).astype(np.float32)
            residual = src.read(3).astype(np.float32)
            observations = src.read(4).astype(np.float32)
            transform, crs = src.transform, src.crs

        mask = (np.isfinite(elevation) & np.isfinite(confidence)
                & (confidence > min_confidence))
        elevation = np.where(mask, elevation, np.nan).astype(np.float32)

        result = BathymetryResult(
            elevation=elevation, confidence=confidence, residual=residual,
            observation_count=observations, transform=transform, crs=crs,
            method=method, mask=mask,
        )
        if derivatives:
            px = abs(transform.a)
            result.slope = slope(elevation, px)
            result.aspect = aspect(elevation, px)
            result.hillshade = hillshade(elevation, px, vertical_exaggeration=10)
            try:
                result.contours = contours(elevation, transform, crs,
                                           interval=0.5)
            except Exception:
                result.contours = None
        return result
