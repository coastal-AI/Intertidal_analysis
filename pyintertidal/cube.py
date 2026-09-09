"""
cube.py — The Sentinel-2 datacube: download once, stream forever
================================================================

The single most important design rule of this package: **satellite data is
fetched ONCE per area** (one OpenEO batch job producing a netCDF cube cached
on disk) and every later product streams that cube in small time blocks.
Consequences a pipeline author should know:

* Re-running a notebook cell never re-downloads: if the cache file exists it
  is trusted and reused, whatever the cell.
* Memory is bounded by the block size (:func:`auto_chunk`), not by the AOI
  size or the number of years — that is what makes it safe to run many
  contiguous tiles in parallel on one machine.
* The bands in the cube depend only on the chosen water detector
  (:func:`pyintertidal.water.bands_for`), so cubes are interchangeable
  between experiments that use the same detector.

The building blocks are deliberately separate functions/objects — nothing is
hidden behind a single "do everything" call:

``SentinelCube``   the cached cube (metadata, ``ensure``, ``stream``)
``open_cube``      lazy access to any cached cube
``auto_chunk``     memory-safe block size (documented rule, overridable)

Writing products is :func:`pyintertidal.raster.write_geotiff` — re-exported
here for convenience, defined once there.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
import xarray as xr

from .aoi import as_aoi
from .raster import write_geotiff          # single definition, re-exported
from . import water as W


# ─────────────────────────────────────────────────────────────────────────────
#  netCDF primitives
# ─────────────────────────────────────────────────────────────────────────────

def grid_from_dataset(ds):
    """``(transform, crs)`` of an OpenEO netCDF dataset.

    The x/y coordinates are pixel CENTRES; the affine origin is therefore the
    upper-left pixel CORNER (x0 − dx/2, y0 − dy/2). The CRS is read from the
    WKT the Copernicus backend writes into the ``crs`` variable.
    """
    x = np.asarray(ds["x"].values, dtype="float64")
    y = np.asarray(ds["y"].values, dtype="float64")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])                    # usually negative (north-up)
    transform = rasterio.transform.from_origin(
        x[0] - dx / 2.0, y[0] - dy / 2.0, dx, -dy)

    crs = None
    for vname in ("crs", "spatial_ref"):
        if vname in ds.variables:
            attrs = ds[vname].attrs
            wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
            if wkt:
                crs = rasterio.crs.CRS.from_wkt(wkt)
                break
    return transform, crs


def open_cube(nc_path, bands=("B03", "B08", "SCL")):
    """Open a cached cube LAZILY — nothing is read into memory yet.

    Returns ``(ds, band_arrays, t_dim)`` where ``band_arrays`` maps band name
    to an xarray DataArray transposed to ``(t, y, x)``. Close ``ds`` when
    done. Handles both encodings the backend produces (bands as separate
    variables, or one variable with a ``bands`` dimension).
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    def _get(name):
        if name in ds.data_vars:
            da = ds[name]
        else:
            var = next(v for v in ds.data_vars if "bands" in ds[v].dims)
            da = ds[var].sel(bands=name)
        t_dim = "t" if "t" in da.dims else (
            "time" if "time" in da.dims else
            [d for d in da.dims if d not in ("y", "x", "bands")][0])
        ydim = "y" if "y" in da.dims else da.dims[-2]
        xdim = "x" if "x" in da.dims else da.dims[-1]
        return da.transpose(t_dim, ydim, xdim), t_dim

    arrays, t_dim = {}, None
    for b in bands:
        arrays[b], t_dim = _get(b)
    return ds, arrays, t_dim


def cube_dates(band_array, t_dim):
    """The cube's acquisition dates as ``'YYYY-MM-DD'`` strings."""
    import re
    out = []
    for v in np.asarray(band_array[t_dim].values):
        m = re.search(r"\d{4}-\d{2}-\d{2}", str(v))
        out.append(m.group(0) if m else str(v))
    return out


def auto_chunk(pixels, chunk=None, budget_bytes=400_000_000, cap=128,
               bytes_per_obs=20):
    """Time-block size that keeps one streaming pass inside ``budget_bytes``.

    ``bytes_per_obs`` is the true per-pixel-per-date footprint of the pass:
    ~20 for index-based water detection (two float32 reflectance bands, the
    index, SCL and boolean masks live simultaneously), ~3 for SCL-only
    passes. Pass an explicit ``chunk`` to override the rule entirely.
    """
    if chunk is not None:
        return max(1, int(chunk))
    return max(4, min(cap, int(budget_bytes / max(pixels * bytes_per_obs, 1))))


# ─────────────────────────────────────────────────────────────────────────────
#  The cached cube object
# ─────────────────────────────────────────────────────────────────────────────

class SentinelCube:
    """A cached Sentinel-2 L2A cube for one AOI, streamed block by block.

    Parameters
    ----------
    aoi : AOI | polygon | bbox dict
        Study area; the DOWNLOAD uses its bounding box, products are clipped
        to the polygon later by the analysis/visualization layers.
    time_extent : (start, end)
        ISO dates.
    water : str
        Water detector name — decides which bands are downloaded.
    cache_path : str, optional
        NetCDF location (default ``<water>_cube_<name>.nc`` in the working
        directory). An existing file is trusted and reused, never rewritten.
    resolution : int
        Target grid resolution in metres (10 = native for B03/B08; SCL is
        upsampled and only ever used as a cloud mask).
    extra_bands : sequence of str, optional
        Bands to download IN ADDITION to the ones the detector needs, without
        changing which detector is used. Two reasons this exists rather than
        inventing a new detector: a second water index computed on the same
        scenes separates optical bias from topography, since two indices with
        independent noise should agree on elevation and their disagreement is
        radiometric; and the red band diagnoses turbidity, which is the
        obvious suspect behind such a disagreement. Both are diagnostics, not
        detectors, so ``water`` stays what it was and the index logic in
        :mod:`pyintertidal.water` is untouched.
    """

    def __init__(self, aoi, time_extent, water="ndwi", cache_path=None,
                 resolution=10, max_cloud_cover=100, extra_bands=()):
        self.aoi = as_aoi(aoi)
        self.time_extent = list(time_extent)
        self.water = water
        base = W.bands_for(water)
        # SCL must stay last: readers index it by name, but several places
        # assume the reflectance bands come first when reporting.
        extra = tuple(b for b in extra_bands if b not in base)
        self.bands = tuple(b for b in base if b != "SCL") + extra + ("SCL",)
        self.resolution = int(resolution)
        self.max_cloud_cover = max_cloud_cover
        self.cache_path = cache_path or (
            f"{water}_cube_{self.aoi.name.lower().replace(' ', '_')}.nc")
        self._dates = None
        self._grid = None
        self._shape = None
        #: openEO job id of the last download, so a campaign can log it and a
        #: killed run leaves a traceable job rather than an orphan.
        self.last_job_id = None

    # ── acquisition ─────────────────────────────────────────────────────────
    @property
    def cached(self):
        """True when a COMPLETE netCDF cache exists on disk.

        Existence is not enough. A download interrupted mid-write — the
        laptop sleeps, the network drops — leaves a file of the right name
        and plausible size whose HDF5 end-of-file marker points past its
        last byte. Every reader then fails somewhere unpredictable, hours
        later and far from the cause. Opening it here turns that into an
        immediate, obvious re-download.
        """
        if not os.path.exists(self.cache_path):
            return False
        try:
            import xarray as xr

            with xr.open_dataset(self.cache_path) as ds:
                return len(ds.data_vars) > 0
        except Exception:
            print(f"[cube] {self.cache_path} exists but is unreadable "
                  f"(truncated download?) — it will be fetched again")
            return False

    def ensure(self, connection=None, verbose=True):
        """Download the cube if — and only if — it is not cached yet.

        One OpenEO batch job: ``load_collection`` → resample to the target
        grid → ``save_result(netCDF)`` → download. Returns ``self``.
        """
        if self.cached:
            if verbose:
                print(f"[cube] using cache {self.cache_path}")
            return self
        if connection is None:
            raise ValueError(f"cube {self.cache_path!r} is not cached and no "
                             f"OpenEO connection was provided")
        if verbose:
            print(f"[cube] downloading {self.bands} for {self.aoi.name} "
                  f"({self.time_extent[0]} → {self.time_extent[1]})")
        cube = connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=self.aoi.bbox,
            temporal_extent=self.time_extent,
            bands=list(self.bands),
            max_cloud_cover=self.max_cloud_cover,
        )
        try:
            cube = cube.resample_spatial(resolution=self.resolution,
                                         method="near")
        except Exception:
            pass
        job = cube.save_result(format="netCDF").create_job(
            title=f"pyintertidal_cube_{self.aoi.name}")
        # Record the id BEFORE blocking on it. `start_and_wait` can run for an
        # hour; if the process dies in that window — a killed campaign, a
        # laptop that slept — the backend job carries on with nobody holding a
        # reference to it, and there is no way back to it from Python. With
        # the id written down it can be reattached or cancelled.
        self.last_job_id = getattr(job, "job_id", None)
        if verbose:
            print(f"[cube] openEO job {self.last_job_id} started for "
                  f"{self.aoi.name}")
        job.start_and_wait()
        assets = job.get_results().get_assets()
        if not assets:
            raise RuntimeError("OpenEO job returned no assets")
        # Download beside the target, then rename. A rename is atomic on
        # every filesystem we care about, so the cache path either does not
        # exist or holds a complete file — never a half-written one that a
        # later run would happily treat as cached.
        tmp = f"{self.cache_path}.part"
        try:
            assets[0].download(tmp)
            with xr.open_dataset(tmp) as ds:            # refuse a bad file
                if not len(ds.data_vars):
                    raise RuntimeError("downloaded cube has no variables")
            os.replace(tmp, self.cache_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if verbose:
            print(f"[cube] cached -> {self.cache_path} "
                  f"({os.path.getsize(self.cache_path) / 1e6:.0f} MB)")
        return self

    # ── metadata (read once, then memoised) ─────────────────────────────────
    def _meta(self):
        if self._dates is None:
            ds, arrays, t_dim = open_cube(self.cache_path, self.bands)
            try:
                self._dates = cube_dates(arrays["SCL"], t_dim)
                self._grid = grid_from_dataset(ds)
                self._shape = tuple(arrays["SCL"].shape[1:])
            finally:
                ds.close()
        return self._dates, self._grid, self._shape

    @property
    def dates(self):
        """Acquisition dates ('YYYY-MM-DD') of every scene in the cube."""
        return self._meta()[0]

    @property
    def grid(self):
        """``(transform, crs)`` of the analysis grid."""
        return self._meta()[1]

    @property
    def shape(self):
        """``(rows, cols)`` of the analysis grid."""
        return self._meta()[2]

    # ── streaming ───────────────────────────────────────────────────────────
    def stream(self, chunk=None):
        """Yield ``(t_slice, {band: (t, y, x) array})``, memory-bounded.

        This is THE access pattern of the package: no caller ever
        materialises the full ``(T, H, W)`` cube in RAM.
        """
        ds, arrays, t_dim = open_cube(self.cache_path, self.bands)
        try:
            T = arrays["SCL"].sizes[t_dim]
            H, Wd = self.shape
            ch = auto_chunk(H * Wd, chunk)
            for a in range(0, T, ch):
                sl = slice(a, min(a + ch, T))
                blocks = {b: np.asarray(arrays[b].isel({t_dim: sl}).values)
                          for b in self.bands}
                yield sl, blocks
        finally:
            ds.close()

    def thumbnails(self, n_dates=6, size=18, spread=True):
        """Tiny downsampled previews of the cube, for the explain diagrams.

        Reads ``n_dates`` scenes — evenly spread across the archive by
        default, so the previews show the range of conditions rather than
        one cloudy week — and block-averages each one down to about
        ``size × size`` cells. Cheap by construction: a handful of small
        reads, never the whole cube.

        Returns ``({band: [array, …]}, [date, …])``.
        """
        dates = self.dates
        if not dates:
            return {}, []
        n = max(1, min(int(n_dates), len(dates)))
        if spread and len(dates) > n:
            idx = np.linspace(0, len(dates) - 1, n).round().astype(int).tolist()
        else:
            idx = list(range(n))

        H, Wd = self.shape
        step_y = max(1, H // int(size))
        step_x = max(1, Wd // int(size))

        ds, arrays, t_dim = open_cube(self.cache_path, self.bands)
        try:
            out = {b: [] for b in self.bands}
            for i in idx:
                for b in self.bands:
                    tile = np.asarray(
                        arrays[b].isel({t_dim: i}).values[::step_y, ::step_x],
                        dtype=np.float32)
                    tile = np.where(tile == 0, np.nan, tile)   # padding → hole
                    out[b].append(tile)
        finally:
            ds.close()
        return out, [dates[i] for i in idx]

    def __repr__(self):
        state = "cached" if self.cached else "NOT cached"
        return (f"SentinelCube({self.aoi.name!r}, {self.time_extent}, "
                f"water={self.water!r}, bands={self.bands}, {state})")
