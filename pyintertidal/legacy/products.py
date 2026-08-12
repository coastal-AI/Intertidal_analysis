"""
products.py — Server-side products kept for the comparison study
================================================================

One product lives here: the **multi-index water frequency**, which computes
the water frequency with SCL, NDWI, MNDWI and AWEI in a SINGLE backend job,
over identical dates and an identical cloud mask.

That last point is the reason it exists. Comparing water indices fairly means
changing exactly one thing — the wet/dry rule — while everything else (dates,
clouds, minimum observations) stays byte-identical. Running four separate
pipelines would leave room for a difference to sneak in; one job with four
outputs does not.

This is what produced the index comparison in the paper: NDWI won against
LiDAR (Spearman -0.624), followed by SCL (-0.586), AWEI (-0.584) and MNDWI
(-0.559), which is why :mod:`pyintertidal.water` defaults to NDWI.

For everyday work use :func:`pyintertidal.frequency.water_frequency`, which
streams the cached cube locally and needs no cloud job.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio

from ..aoi import as_aoi


WATER_FREQUENCY_MULTI_UDF = r"""
import numpy as np
import xarray


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:

    methods = context.get("methods", ["scl", "ndwi", "mndwi", "awei"])
    thresholds = context.get("thresholds", {})
    scl_water = context.get("scl_water_classes", [6, 12])
    clear_classes = context.get("clear_classes", [4, 5, 6, 12])
    valid_dates = context.get("valid_dates", None)
    min_obs = int(context.get("min_obs", 0))

    arr = cube.values.astype("float64")  # (t, bands, y, x)
    bn = [str(b) for b in cube.coords["bands"].values]

    def band(name):
        return arr[:, bn.index(name), :, :]

    green = band("B03"); red = band("B04"); nir = band("B08")
    swir1 = band("B11"); swir2 = band("B12"); scl = band("SCL")

    if valid_dates:
        tname = "t" if "t" in cube.dims else cube.dims[0]
        tstr = np.array([str(t)[:10] for t in np.asarray(cube.coords[tname].values)])
        keep = np.isin(tstr, list(valid_dates))
        if keep.any():
            green, red, nir, swir1, swir2, scl = (
                green[keep], red[keep], nir[keep], swir1[keep], swir2[keep], scl[keep]
            )

    scl_int = np.round(scl).astype(np.int16)
    clear = np.isin(scl_int, clear_classes)

    # Escala de reflectancia (AWEI es lineal -> necesita [0,1]; NDWI/MNDWI son
    # razones y no dependen de la escala).
    finite_g = green[np.isfinite(green)]
    scale = 10000.0 if (finite_g.size and np.nanmax(finite_g) > 1.5) else 1.0
    g, r, n = green / scale, red / scale, nir / scale
    s1, s2 = swir1 / scale, swir2 / scale

    with np.errstate(invalid="ignore", divide="ignore"):
        d_ndwi = g + n
        ndwi = np.where(d_ndwi != 0, (g - n) / d_ndwi, np.nan)
        d_mndwi = g + s1
        mndwi = np.where(d_mndwi != 0, (g - s1) / d_mndwi, np.nan)
        awei = 4.0 * (g - s1) - (0.25 * n + 2.75 * s2)

    clear_votes = np.sum(clear, axis=0).astype("float32")
    safe = np.where(clear_votes == 0, 1.0, clear_votes)
    threshold_obs = max(1, min_obs)

    def water_freq(water_mask):
        wv = np.sum(clear & water_mask, axis=0).astype("float32")
        wf = (wv / safe).astype("float32")
        wf[clear_votes < threshold_obs] = np.nan
        return wf

    out_bands, out_names = [], []
    for m in methods:
        if m == "scl":
            water = np.isin(scl_int, scl_water)
        elif m == "ndwi":
            water = np.isfinite(ndwi) & (ndwi > float(thresholds.get("ndwi", 0.0)))
        elif m == "mndwi":
            water = np.isfinite(mndwi) & (mndwi > float(thresholds.get("mndwi", 0.0)))
        elif m == "awei":
            water = np.isfinite(awei) & (awei > float(thresholds.get("awei", 0.0)))
        else:
            continue
        out_bands.append(water_freq(water))
        out_names.append("wf_" + m)

    stacked = np.stack(out_bands, axis=0)

    return xarray.DataArray(
        stacked,
        dims=["bands", "y", "x"],
        coords={"bands": out_names, "y": cube.coords["y"], "x": cube.coords["x"]},
    )
"""


def water_frequency_multi(connection, aoi, time_extent, valid_dates=None,
                          methods=("scl", "ndwi", "mndwi", "awei"),
                          thresholds=None, clear_classes=(4, 5, 6, 12),
                          scl_water_classes=(6, 12), min_obs=8,
                          max_cloud_cover=40, out_path="wf_multi.tif",
                          force=False, verbose=True):
    """Water frequency computed with several indices in one backend job.

    Parameters
    ----------
    methods : tuple
        Any subset of ``("scl", "ndwi", "mndwi", "awei")``. The output
        GeoTIFF has one band per method, IN THIS ORDER — band names are not
        always preserved by the backend, so the order is the contract.
    thresholds : dict, optional
        Per-index water threshold (default 0.0 for all three indices).
    valid_dates : list, optional
        Restrict to cloud-filtered dates, exactly as the local pipeline does.

    Returns
    -------
    (dict {method: array}, transform, crs)
    """
    import openeo

    methods = list(methods)
    thresholds = dict(thresholds or {"ndwi": 0.0, "mndwi": 0.0, "awei": 0.0})
    aoi = as_aoi(aoi)

    if (not force) and os.path.exists(out_path):
        with rasterio.open(out_path) as src:
            if src.count != len(methods):
                raise ValueError(
                    f"{out_path!r} has {src.count} bands but methods={methods}; "
                    f"pass force=True or match the file")
            arrays = {methods[i]: src.read(i + 1).astype(np.float32)
                      for i in range(src.count)}
            return arrays, src.transform, src.crs

    cube = connection.load_collection(
        "SENTINEL2_L2A", spatial_extent=aoi.bbox,
        temporal_extent=list(time_extent),
        bands=["B03", "B04", "B08", "B11", "B12", "SCL"],
        max_cloud_cover=max_cloud_cover,
    )
    if valid_dates:
        try:
            cube = cube.filter_labels(dimension="t",
                                      condition=lambda x: x.isin(list(valid_dates)))
        except Exception:
            pass          # the UDF filters internally as well

    udf = openeo.UDF(
        code=WATER_FREQUENCY_MULTI_UDF, runtime="Python",
        context={"methods": methods, "thresholds": thresholds,
                 "clear_classes": list(clear_classes),
                 "scl_water_classes": list(scl_water_classes),
                 "valid_dates": list(valid_dates) if valid_dates else None,
                 "min_obs": int(min_obs)},
    )
    job = (cube.reduce_dimension(dimension="t", reducer=udf)
              .save_result(format="GTiff")
              .create_job(title=f"wf_multi_{aoi.name}"))
    if verbose:
        print(f"[legacy] multi-index water frequency ({', '.join(methods)})…")
    job.start_and_wait()
    assets = job.get_results().get_assets()
    if not assets:
        raise RuntimeError("multi-index job returned no assets")
    assets[0].download(out_path)

    with rasterio.open(out_path) as src:
        arrays = {methods[i]: src.read(i + 1).astype(np.float32)
                  for i in range(src.count)}
        return arrays, src.transform, src.crs
