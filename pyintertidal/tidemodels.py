"""
tidemodels.py — Global tide models (pyTMD and CMEMS backends)
=============================================================

Low-level tide predictors. Most users go through
:class:`pyintertidal.tides.TideService`, which wraps these and pairs them
with satellite overpass times; use these classes directly when you want raw
tide predictions at arbitrary points and times.

Two backends
------------
:class:`PyTMDTideModel`
    Harmonic global tide models via pyTMD. ``EOT20`` (1/8°) is the default:
    sixteen times finer than GOT4.10, free of credentials, and the model DEA
    Intertidal itself defaults to. It does not self-download — see the
    instructions in the error this class raises when it is missing.
    ``GOT4.10`` (NASA GSFC) fetches itself on first use and needs no setup,
    which makes it convenient, but its 0.5° grid puts the nearest ocean cell
    34 km from the Ría de Villaviciosa against EOT20's 13 km — check your own
    site with :func:`pyintertidal.tidecheck.nearest_ocean_km`.
    ``FES2014``/``FES2022`` (AVISO/CNES) are
    higher resolution but need AVISO credentials in the environment
    variables ``PYTMD_FES_USER`` / ``PYTMD_FES_PASSWORD``
    (register at https://www.aviso.altimetry.fr/en/data/data-access.html).
:class:`CopernicusTideModel`
    Sea-surface height from the CMEMS Iberia–Biscay–Ireland reanalysis. This
    is *modelled sea level*, not a harmonic tide: it includes storm surge and
    other non-astronomical signals, which makes it a useful cross-check
    against the harmonic prediction (and against tide gauges, see
    :mod:`pyintertidal.gauges`).

Land-mask handling (why the code looks the way it does)
-------------------------------------------------------
Estuary centroids frequently fall on a LAND cell of the tide grid, where the
model returns NaN. Both classes therefore search outward for the nearest
valid ocean cell and predict there. :meth:`PyTMDTideModel.get_tide_heights_batch`
does that search ONCE and then evaluates every timestamp in a single
vectorised call — roughly N times faster than looping, which matters when a
decade of imagery means hundreds of dates.
"""

from __future__ import annotations

import os

import numpy as np


class PyTMDTideModel:
    """Harmonic tide predictions from a global pyTMD model.

    Parameters
    ----------
    model_name : str
        One of :attr:`SUPPORTED_MODELS`.
    directory : str
        Cache folder for the model data files (downloaded once).
    box_size : float
        Half-width, in degrees, of the search box used to find a valid ocean
        cell around the requested point.
    resolution : float
        Sampling step, in degrees, of that search grid.
    """

    #: Models this wrapper can fetch, and whether they need credentials.
    SUPPORTED_MODELS = {
        "GOT4.10": {"provider": "GSFC", "requires_auth": False},
        "EOT20": {"provider": "SEANOE", "requires_auth": False},
        "FES2022": {"provider": "AVISO", "requires_auth": True},
        "FES2014": {"provider": "AVISO", "requires_auth": True},
    }

    #: Sub-folder each model unpacks into (used to detect an existing cache).
    _MODEL_DIRS = {"GOT4.10": "GOT4.10c", "EOT20": "EOT20",
                   "FES2022": "fes2022b", "FES2014": "fes2014"}

    def __init__(self, model_name="EOT20", directory="./tide_models",
                 box_size=0.4, resolution=0.05, verbose=True):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"unsupported model {model_name!r}; available: "
                             f"{', '.join(self.SUPPORTED_MODELS)}")
        self.model_name = model_name
        self.directory = directory
        self.box_size = box_size
        self.resolution = resolution
        self.verbose = verbose
        self.model_path = os.path.join(directory, model_name)
        self._ensure_model()

    # ── model data ──────────────────────────────────────────────────────────
    def _ensure_model(self):
        """Download the model data files unless they are already cached."""
        cached = False
        if os.path.isdir(self.directory):
            expected = self._MODEL_DIRS.get(self.model_name)
            cached = bool(expected and expected in os.listdir(self.directory))
        if cached:
            if self.verbose:
                print(f"[tides] model {self.model_name} already cached")
            return

        from pyTMD.datasets import fetch_gsfc_got, fetch_aviso_fes

        info = self.SUPPORTED_MODELS[self.model_name]
        if info["requires_auth"] and not (os.environ.get("PYTMD_FES_USER")
                                          and os.environ.get("PYTMD_FES_PASSWORD")):
            raise EnvironmentError(
                f"model {self.model_name} requires AVISO credentials; set the "
                f"environment variables PYTMD_FES_USER and PYTMD_FES_PASSWORD "
                f"(register at https://www.aviso.altimetry.fr/en/data/data-access.html)")
        if info["provider"] == "SEANOE":
            # EOT20 has no fetcher in pyTMD: it is published as one 2.3 GB zip
            # holding two inner zips. Unpack it by hand rather than pretend we
            # can download it here.
            raise FileNotFoundError(
                f"{self.model_name} is not cached in {self.directory!r}. "
                f"Download https://www.seanoe.org/data/00683/79489/data/85762.zip "
                f"(2.3 GB), extract the inner ocean_tides.zip, and place the "
                f"constituents in {self.directory}/EOT20/ocean_tides/")
        if self.verbose:
            print(f"[tides] downloading {self.model_name} ({info['provider']})…")
        if info["provider"] == "GSFC":
            fetch_gsfc_got(model=self.model_name, directory=self.directory,
                           format="netcdf", compressed=False)
        else:
            fetch_aviso_fes(model=self.model_name, directory=self.directory,
                            user=os.environ.get("PYTMD_FES_USER"),
                            password=os.environ.get("PYTMD_FES_PASSWORD"),
                            compressed=True)

    @property
    def _pytmd_name(self):
        """pyTMD's internal name for the model (GOT4.10 ships as netCDF)."""
        return f"{self.model_name}_nc" if self.model_name == "GOT4.10" \
            else self.model_name

    def _elevations(self, lons, lats, times):
        """Thin wrapper around ``pyTMD.compute.tide_elevations``.

        pyTMD's parameters are UPPER-case (``MODEL``, ``DIRECTORY``, ``EPSG``,
        ``TIME``). Lower-case aliases exist but do not map reliably: passing
        ``model=`` can leave ``MODEL`` as ``None``, and the failure surfaces as
        ``ValueError: Unlisted tide model None`` — which reads like a missing
        data file and sends you looking in the wrong place entirely.

        Returns a plain float array with NaN where the model has no water.
        pyTMD hands back a masked array here; callers that reached for
        ``.values`` were relying on a different return type and broke the
        moment the call was corrected. Normalising once, in this one place,
        is what stops that repeating.
        """
        import pyTMD
        out = pyTMD.compute.tide_elevations(
            lons, lats, times, MODEL=self._pytmd_name,
            DIRECTORY=self.directory, EPSG="4326", TIME="datetime",
        )
        if hasattr(out, "values"):          # a DataFrame/Series-like return
            out = out.values
        return np.ma.filled(np.asarray(out, dtype=float), np.nan)

    # ── predictions ─────────────────────────────────────────────────────────
    def get_tide_height(self, lat, lon, dt):
        """Tide height (m) at one datetime, at the nearest valid ocean cell.

        Raises ``ValueError`` when the whole search box is land — try a
        larger ``box_size`` or a finer model.
        """
        if lat is None or lon is None:
            raise ValueError("lat and lon are required")

        step = self.resolution
        lat_lines = np.arange(lat - self.box_size, lat + self.box_size + step, step)
        lon_lines = np.arange(lon - self.box_size, lon + self.box_size + step, step)
        lons_2d, lats_2d = np.meshgrid(lon_lines, lat_lines)
        flat_lats, flat_lons = lats_2d.ravel(), lons_2d.ravel()

        times = np.array([np.datetime64(dt)] * flat_lats.size, dtype="datetime64[ns]")
        heights = self._elevations(flat_lons, flat_lats, times)

        valid = ~np.isnan(heights)
        if not valid.any():
            raise ValueError("no valid tide data inside the search box; try a "
                             "larger box_size or a different model")
        dx = flat_lons[valid] - lon
        dy = flat_lats[valid] - lat
        nearest = np.flatnonzero(valid)[np.argmin(dx * dx + dy * dy)]
        return float(heights[nearest])

    def get_tide_heights_batch(self, lat, lon, datetimes):
        """Tide heights (m) for many datetimes at one point — vectorised.

        Two-step strategy, ~N times faster than looping:

        1. Find the nearest valid ocean cell ONCE, growing the search box
           (0.2° → 0.5° → 1.0° → ``box_size``) until a non-NaN cell appears.
        2. Evaluate every timestamp at that cell in a single pyTMD call.

        Returns a list of floats in the same order as ``datetimes``.
        """
        if not len(datetimes):
            return []

        best_lat = best_lon = None
        for probe in (0.2, 0.5, 1.0, self.box_size):
            step = self.resolution
            lat_lines = np.arange(lat - probe, lat + probe + step, step)
            lon_lines = np.arange(lon - probe, lon + probe + step, step)
            lons_2d, lats_2d = np.meshgrid(lon_lines, lat_lines)
            flat_lats, flat_lons = lats_2d.ravel(), lons_2d.ravel()
            t0 = np.array([np.datetime64(datetimes[0])] * flat_lats.size,
                          dtype="datetime64[ns]")
            heights = self._elevations(flat_lons, flat_lats, t0)
            valid = ~np.isnan(heights)
            if valid.any():
                dx = flat_lons[valid] - lon
                dy = flat_lats[valid] - lat
                idx = np.flatnonzero(valid)[np.argmin(dx * dx + dy * dy)]
                best_lat, best_lon = float(flat_lats[idx]), float(flat_lons[idx])
                break
        if best_lat is None:
            raise ValueError("no valid tide data inside the search box")

        times = np.array([np.datetime64(dt) for dt in datetimes],
                         dtype="datetime64[ns]")
        result = self._elevations(np.full(times.size, best_lon),
                                  np.full(times.size, best_lat), times)
        return [float(h) for h in result]

    def __repr__(self):
        return (f"PyTMDTideModel({self.model_name!r}, box_size={self.box_size}, "
                f"resolution={self.resolution})")


class CopernicusTideModel:
    """Sea-surface height from the CMEMS IBI reanalysis (0.027°, hourly).

    Unlike :class:`PyTMDTideModel` this is *total* modelled sea level (tide +
    surge + steric signals), which makes it a good independent cross-check
    but a different quantity from a harmonic prediction. Requires the
    ``copernicusmarine`` package and a CMEMS login (prompted on first use).
    """

    DATASET_ID = "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m"

    def __init__(self, box_size=0.1):
        import copernicusmarine          # lazy: heavy optional dependency
        copernicusmarine.login()
        self.box_size = box_size

    def _open(self, lat, lon, start, end):
        import copernicusmarine
        return copernicusmarine.open_dataset(
            dataset_id=self.DATASET_ID,
            minimum_longitude=lon - self.box_size,
            maximum_longitude=lon + self.box_size,
            minimum_latitude=lat - self.box_size,
            maximum_latitude=lat + self.box_size,
            start_datetime=start, end_datetime=end, variables=["zos"],
        )

    @staticmethod
    def _nearest_ocean_cell(ds, lat, lon):
        """Coordinates of the nearest cell with data (skips the land mask)."""
        first = ds["zos"].isel(time=0)
        valid = first.notnull().values
        lon_grid, lat_grid = np.meshgrid(ds["longitude"].values,
                                         ds["latitude"].values)
        vlat, vlon = lat_grid[valid], lon_grid[valid]
        if vlat.size == 0:
            raise ValueError(f"no valid ocean cells within {lat}, {lon}")
        idx = np.argmin((vlat - lat) ** 2 + (vlon - lon) ** 2)
        return float(vlat[idx]), float(vlon[idx])

    def get_tide_heights_batch(self, lat, lon, datetimes):
        """Sea level (m) for many datetimes: one download, one interpolation."""
        import pandas as pd

        if not len(datetimes):
            return []
        times = pd.to_datetime(datetimes)
        ds = self._open(lat, lon,
                        (times.min() - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S"),
                        (times.max() + pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S"))
        olat, olon = self._nearest_ocean_cell(ds, lat, lon)
        series = ds.sel(latitude=olat, longitude=olon).interp(time=times,
                                                              method="cubic")
        return [float(h) for h in series["zos"].values]

    def get_tide_height(self, lat, lon, dt):
        """Sea level (m) at a single datetime."""
        return self.get_tide_heights_batch(lat, lon, [dt])[0]

    def __repr__(self):
        return f"CopernicusTideModel(box_size={self.box_size})"
