"""
tides.py — Tide heights for every satellite scene
=================================================

Elevation-from-satellite methods need the water level at the exact moment
each image was taken. Two moving parts, both explicit parameters:

1. **When was each scene taken?** Overpass times from the Copernicus STAC
   catalogue (free metadata query).
2. **How high was the tide?** A global tide model evaluated with pyTMD.
   ``model`` is selectable (``"GOT4.10"`` default; FES2014 / TPXO9 variants
   work when their data files are present in ``directory``). The prediction
   point is user-given or the AOI centroid (``location="auto"``).

Documented approximation: ONE prediction point serves the whole AOI. Tidal
phase and amplitude shift along long estuaries, adding a site-dependent
vertical error; with a local tide gauge available, pass its coordinates as
``location``. :func:`water_centroid` offers a better default than the plain
polygon centroid for estuaries whose centroid falls on land.
"""

from __future__ import annotations

import numpy as np

from .aoi import as_aoi
from .overpass import get_overpass_times
from .tidemodels import PyTMDTideModel, CopernicusTideModel


class TideService:
    """Tide heights at satellite overpass times for one AOI.

    Parameters
    ----------
    model : str
        Tide model name understood by pyTMD (default ``"GOT4.10"``).
    directory : str
        Folder holding the model's data files (downloaded once, reused).
    location : (lat, lon) | "auto"
        Prediction point; ``"auto"`` uses the AOI centroid.
    """

    def __init__(self, model="GOT4.10", directory="./tide_models",
                 location="auto", backend="pytmd"):
        self.model = model
        self.directory = directory
        self.location = location
        self.backend = backend
        self._model = None

    def _tide_model(self, verbose=True):
        """Instantiate (once) the configured backend."""
        if self._model is None:
            if self.backend == "cmems":
                self._model = CopernicusTideModel()
            else:
                self._model = PyTMDTideModel(
                    model_name=self.model, directory=self.directory,
                    box_size=2.0, resolution=0.05, verbose=verbose)
        return self._model

    def _point(self, aoi):
        if self.location == "auto" or self.location is None:
            return as_aoi(aoi).centroid
        return tuple(self.location)

    def heights_for(self, aoi, dates, verbose=True):
        """Tide height (m, model MSL datum) at each date's overpass time.

        Returns ``{date: height}`` for the dates with a resolvable overpass
        time and a finite prediction.
        """
        aoi = as_aoi(aoi)
        if not dates:
            return {}
        overpass = get_overpass_times(aoi.bbox, [min(dates), max(dates)])
        lat, lon = self._point(aoi)
        model = self._tide_model(verbose=verbose)
        dated = [d for d in dates if d in overpass]
        heights = model.get_tide_heights_batch(lat, lon,
                                               [overpass[d] for d in dated])
        out = {d: float(h) for d, h in zip(dated, heights)
               if h is not None and np.isfinite(h)}
        if verbose:
            hs = np.array(list(out.values()))
            print(f"[tides] {self.model} @ ({lat:.3f}, {lon:.3f}) → "
                  f"{len(out)} dates, range [{hs.min():.2f}, {hs.max():.2f}] m")
        return out

    def series(self, aoi, start, end, every_hours=1, verbose=False):
        """Continuous tide series between two dates.

        Feeds two things: the tidal-sampling diagnostics of
        :mod:`pyintertidal.coverage` (did the satellite sample the whole tidal
        range?) and the hydroperiod computation of
        :mod:`pyintertidal.hydroperiod` (how long is each pixel submerged?).

        Returns ``(times, heights)`` as a list of timestamps and a float array.
        """
        import pandas as pd

        aoi = as_aoi(aoi)
        lat, lon = self._point(aoi)
        times = list(pd.date_range(start, end, freq=f"{every_hours}h"))
        model = self._tide_model(verbose=verbose)
        heights = model.get_tide_heights_batch(lat, lon, times)
        return times, np.asarray(
            [np.nan if h is None else h for h in heights], dtype=float)


def water_centroid(reference_map, transform, crs, water_class=1):
    """A tide-prediction point that is guaranteed to be ON water.

    The polygon centroid of a curved estuary often falls on land, where tide
    models return NaN and the nearest-ocean-cell search has to wander. This
    picks the pixel deepest inside the stable-water class (the maximum of the
    distance transform, i.e. the middle of the main channel) and returns it
    as ``(lat, lon)``.

    Parameters
    ----------
    reference_map : 2-D array
        Output of :func:`pyintertidal.stability.reference_and_clouds`.
    transform, crs
        Grid georeferencing of that map (``cube.grid``).
    water_class : int
        Class value standing for stable water (1 by default).

    Returns
    -------
    (lat, lon) — ready to pass as ``TideService(location=...)``.
    """
    from scipy.ndimage import distance_transform_edt
    from pyproj import Transformer
    import rasterio

    water = np.asarray(reference_map) == water_class
    if not water.any():
        raise ValueError("the reference map has no stable-water pixels")
    dist = distance_transform_edt(water)
    row, col = np.unravel_index(int(np.argmax(dist)), dist.shape)
    x, y = rasterio.transform.xy(transform, row, col)
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(x, y)
    return (float(lat), float(lon))
