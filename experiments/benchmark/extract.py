# -*- coding: utf-8 -*-
"""Validation v2 — a small Zarr extract of a legacy netCDF cube, for local runs only.

Question
--------
How can the Zarr path of validation v2 (``inputs.ZarrSource``, the inputs
report, the v2 package) be exercised on the laptop before the real stores
exist? By writing a SMALL store in the exact format of
``experiments.zarr_cubes`` (one calendar year, a window of a few km, a few
dozen dates) whose B03, B08 and SCL are the real values of a legacy netCDF
cube, whose other bands are synthetic but plausible, and which says so in
its attributes. Nothing computed on an extract is a result: it is a
software check of the Zarr path.

Pre-registered rules (:data:`SYNTHETIC_RULES`)
-----------------------------------------------
* Window: rows [r0, r1) x columns [c0, c1) of the legacy 10 m grid (the
  pixel centres are copied), so the extract's grid is exactly a sub-grid of
  the legacy one; ``inputs.inputs_report`` compares it with that window. By
  default the window of the requested size that holds the most intertidal
  pixels of the production mask (``products_<site>/intertidal_mask.tif``)
  when it exists on the legacy grid, else the frame centre.
* Dates: the legacy dates of one calendar year, every k-th with
  k = max(1, n_year // n_dates), the first ``n_dates`` of them, so the
  extract spans the year.
* Real bands: B03, B08 and SCL exactly as stored in the legacy cube (int16,
  fill -32768; offset-harmonised values, so o_s = 0 for every scene).
* The 20 m lattice: the phase of the legacy SCL on the window
  (``inputs.measure_lattice20`` on up to 8 dates; the grid geometry when the
  data cannot tell). Every synthetic 20 m band is an exact 2 x 2 replica on
  it, like the delivered B8A, B11, B12, SCL and CLD.
* Synthetic bands (integer DN, deterministic, no random numbers; fill
  wherever a real input is fill):
  B02 = round(0.92 B03 + 20); B04 = round(0.85 B03 + 0.10 B08 - 30);
  B8A = round(2 x 2 block mean of B08) (fill if any pixel of the block is);
  B11 = round(0.75 B8A); B12 = round(0.60 B11);
  CLD = from the block's SCL: 9 -> 90, 8 -> 60, 10 -> 30, 3 -> 20, 0 or
  no data -> fill, any other class -> 2.
  Water (low B08) thus keeps low SWIR (MNDWI and NDWI agree in sign), and
  the cloud classes keep a high CLD.
* Attributes: ``extract`` (source cube, window, dates rule, lattice),
  ``real_bands``, ``synthetic_bands``, ``synthetic_rules`` and
  ``purpose = "local software checks only; never results"``; the store is
  written by ``download_cube_zarr.write_store`` (read back bit for bit).
* ``overpass_times.json`` beside the store: the catalogue times (product
  names, the datatake start) of the extract dates, from the production
  labels; the default scene instants of validation v2 (user decision of
  2026-10-14).
* After writing, the real bands are read back through ``inputs.ZarrSource``
  and must equal the legacy values bit for bit (else the extract is
  refused and deleted).

Memory: the window of the selected dates, three int16 bands at a time (a few
tens of MB); the legacy cube is read lazily (its chunks are 1 x 256 x 256).
This module has no import-time side effects.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import zarr_cubes as zc
from experiments.benchmark import config, inputs, obs

FILL = obs.FILL
EXTRACT_FORMAT = "validation-v2-extract/1"
REAL_BANDS = ("B03", "B08", "SCL")
SYNTHETIC_BANDS = ("B02", "B04", "B8A", "B11", "B12", "CLD")
#: the frozen rules of the synthetic bands (written into the store)
SYNTHETIC_RULES = {
    "B02": "round(0.92 B03 + 20)",
    "B04": "round(0.85 B03 + 0.10 B08 - 30)",
    "B8A": "round(2 x 2 block mean of B08 on the SCL lattice), replicated",
    "B11": "round(0.75 B8A)",
    "B12": "round(0.60 B11)",
    "CLD": "block SCL 9 -> 90, 8 -> 60, 10 -> 30, 3 -> 20, 0 or no data -> fill, else 2",
    "fill": "fill wherever a real input is fill (B03 or B08 for the spectral bands, SCL for CLD)",
}
CLD_OF_SCL = {9: 90, 8: 60, 10: 30, 3: 20}
CLD_DEFAULT = 2
DEFAULT_SIZE = (300, 400)
DEFAULT_DATES = 40
PURPOSE = "local software checks only; never results"


def log(msg):
    print(msg, flush=True)


class ExtractError(RuntimeError):
    """The extract could not be written or does not read back as written."""


def _clip16(a):
    return np.clip(np.rint(a), -32767, 32767).astype(np.int16)


def best_window(mask, size):
    """``(r0, c0)`` of the ``size`` window holding the most True pixels of
    ``mask`` (ties: the first in row-major order)."""
    h, w = int(size[0]), int(size[1])
    H, W = mask.shape
    if h > H or w > W:
        raise ValueError(f"window {size} larger than the frame {mask.shape}")
    s = np.zeros((H + 1, W + 1), np.int64)
    s[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    tot = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]
    k = int(np.argmax(tot))
    return k // tot.shape[1], k % tot.shape[1]


def _mask_on_grid(path, xs, ys):
    """A GeoTIFF mask on the legacy grid (same pixel centres), or None."""
    if path is None or not os.path.exists(path):
        return None
    try:
        g = inputs.grid_from_geotiff(path)
    except Exception:                         # an unreadable mask only loses the hint
        return None
    if len(g.xs) != len(xs) or len(g.ys) != len(ys) or \
            not (np.allclose(g.xs, xs, atol=1e-3) and np.allclose(g.ys, ys, atol=1e-3)):
        return None
    import rasterio
    with rasterio.open(path) as r:
        a = r.read(1)
    return np.asarray(a) > 0


def synthetic_bands(b03, b08, scl, lattice):
    """The synthetic bands of :data:`SYNTHETIC_RULES` from the real ones
    (int16 (T, H, W), fill :data:`FILL`). Returns ``{band: int16 array}``."""
    T, H, W = b03.shape
    bad = (b03 == FILL) | (b08 == FILL)
    f03 = b03.astype(np.float64)
    f08 = b08.astype(np.float64)
    out = {}
    b02 = _clip16(0.92 * f03 + 20.0)
    b04 = _clip16(0.85 * f03 + 0.10 * f08 - 30.0)
    b02[bad] = FILL
    b04[bad] = FILL
    out["B02"], out["B04"] = b02, b04
    n8 = np.where(b08 == FILL, np.nan, f08)
    m = obs.block_mean20(n8, lattice)                       # NaN if any pixel of a block is
    b8a_b = np.where(np.isfinite(m), np.rint(m), np.nan)
    b11_b = np.where(np.isfinite(b8a_b), np.rint(0.75 * b8a_b), np.nan)
    b12_b = np.where(np.isfinite(b11_b), np.rint(0.60 * b11_b), np.nan)
    for name, blk in (("B8A", b8a_b), ("B11", b11_b), ("B12", b12_b)):
        ok = np.isfinite(blk)
        v = _clip16(np.where(ok, blk, 0.0))
        v[~ok] = FILL                          # the fill is never clipped
        out[name] = obs.expand20(v, lattice, (H, W)).astype(np.int16)
    s_b = obs.block_take20(scl, lattice).astype(np.int64)
    cld = np.full(s_b.shape, CLD_DEFAULT, np.int64)
    for code, v in CLD_OF_SCL.items():
        cld[s_b == code] = v
    cld[(s_b == 0) | (s_b == FILL)] = FILL
    out["CLD"] = obs.expand20(cld.astype(np.int16), lattice, (H, W)).astype(np.int16)
    return out


def _labels(site, path=None):
    """The production catalogue times ``{date: iso}`` of a site."""
    p = path or config.site_config(site, 10).overpass
    p = config.resolve(p)
    if not p or not os.path.exists(p):
        raise ExtractError(f"{site}: no catalogue times ({p})")
    with open(p, encoding="utf-8") as f:
        return {str(k)[:10]: v for k, v in json.load(f).items()}


def legacy_extract(site, out_root, cube=None, labels=None, year=2023, n_dates=DEFAULT_DATES,
                   size=DEFAULT_SIZE, origin=None, mask=None, force=False):
    """Write ``<out_root>/<site>/<year>.zarr`` and its ``overpass_times.json``.

    Parameters
    ----------
    site : str
    out_root : path
        A scratch cubes root (never ``cubes/``).
    cube : path, optional
        The legacy 10 m netCDF (default the site's v13 10 m cube).
    labels : path, optional
        The production ``overpass_times.json`` (default the site's).
    year : int
    n_dates : int
    size : (rows, cols)
    origin : (r0, c0), optional
        The window's upper-left pixel (default :func:`best_window` on the
        production intertidal mask, else the frame centre).
    mask : path, optional
        The intertidal mask GeoTIFF (default ``products_<site>/intertidal_mask.tif``).
    force : bool
        Replace an existing extract.

    Returns
    -------
    dict
        The extract description (also written as ``<year>.extract.json``).
    """
    import xarray as xr

    from experiments import download_cube_zarr as dcz

    t0 = time.time()
    out_root = Path(out_root)
    if out_root.resolve() == zc.cubes_root(None).resolve():
        raise ExtractError("the extract is written under a scratch root, never cubes/")
    out = zc.store_path(site, year, out_root)
    if out.exists() and not force:
        raise ExtractError(f"{out} exists (force=True replaces it)")
    cube = cube or config.site_config(site, 10).cube
    lab = _labels(site, labels)
    ds = xr.open_dataset(cube, mask_and_scale=False, decode_times=True)
    try:
        td = zc._time_dim(ds)
        xs, ys = np.asarray(ds["x"].values, float), np.asarray(ds["y"].values, float)
        H, W = len(ys), len(xs)
        h, w = int(size[0]), int(size[1])
        if origin is None:
            m = _mask_on_grid(mask or os.path.join(os.path.dirname(config.site_config(
                site, 10).extract or ""), "intertidal_mask.tif"), xs, ys)
            if m is not None and m.any():
                origin, how = best_window(m, (h, w)), "most intertidal pixels of the mask"
            else:
                origin, how = ((H - h) // 2, (W - w) // 2), "frame centre"
        else:
            how = "given"
        r0, c0 = (int(v) for v in origin)
        r1, c1 = r0 + h, c0 + w
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
            raise ExtractError(f"window rows {r0}:{r1}, cols {c0}:{c1} outside {H} x {W}")
        alld = pd.DatetimeIndex(ds[td].values)
        idx_y = np.flatnonzero((alld.year == int(year))
                               & np.isin(alld.strftime("%Y-%m-%d"), list(lab)))
        if not len(idx_y):
            raise ExtractError(f"{cube}: no {year} date with a catalogue time")
        step = max(1, len(idx_y) // int(n_dates))
        idx = idx_y[::step][:int(n_dates)]
        dates = [str(d)[:10] for d in alld[idx].strftime("%Y-%m-%d")]
        log(f"[extract] {site} {year}: rows {r0}:{r1}, cols {c0}:{c1} ({how}); "
            f"{len(dates)} of {len(idx_y)} dates (every {step})")
        real = {}
        for b in REAL_BANDS:
            if str(ds[b].dtype) != "int16" or zc.fill_of(ds[b]) != FILL:
                raise ExtractError(f"{cube}: {b} is not int16 with fill {FILL}")
            real[b] = np.asarray(ds[b].isel({td: idx, "y": slice(r0, r1),
                                              "x": slice(c0, c1)}).values).astype(np.int16)
        wkt = None
        for v in ("crs", "spatial_ref"):
            if v in ds.variables:
                wkt = ds[v].attrs.get("crs_wkt") or ds[v].attrs.get("spatial_ref")
                if wkt:
                    break
    finally:
        ds.close()
    sx, sy = xs[c0:c1], ys[r0:r1]
    grid = inputs.Grid(sx, sy, wkt)
    sample = sorted(set(np.linspace(0, len(dates) - 1, min(8, len(dates))).round().astype(int)))
    rep = inputs.measure_lattice20([("SCL", real["SCL"][k]) for k in sample],
                                   inputs.expected_lattice20(grid))
    lattice = tuple(rep["phase"]) if rep.get("phase") else (inputs.expected_lattice20(grid)
                                                            or (0, 0))
    syn = synthetic_bands(real["B03"], real["B08"], real["SCL"], lattice)
    data = dict(real, **syn)
    t = pd.DatetimeIndex(pd.to_datetime(dates))
    dsx = xr.Dataset({b: (("t", "y", "x"), data[b], {"long_name": b,
                                                     "synthetic": int(b in SYNTHETIC_BANDS)})
                      for b in zc.BANDS},
                     coords={"t": t, "y": sy, "x": sx})
    dsx["crs"] = xr.DataArray(np.int32(0), attrs={
        "crs_wkt": wkt, "spatial_ref": wkt,
        "GeoTransform": f"{sx[0] - 5.0} 10 0 {sy[0] + 5.0} 0 -10"})
    desc = {"format": EXTRACT_FORMAT, "site": site, "year": int(year),
            "source_cube": str(cube).replace("\\", "/"), "window": [r0, r1, c0, c1],
            "window_rule": how, "legacy_shape": [H, W], "dates_rule":
            f"every {step}-th {year} date with a catalogue time, first {n_dates}",
            "n_dates": len(dates), "lattice20": list(lattice), "lattice_report": rep,
            "real_bands": list(REAL_BANDS), "synthetic_bands": list(SYNTHETIC_BANDS),
            "synthetic_rules": SYNTHETIC_RULES, "purpose": PURPOSE}
    attrs = {"site": site, "extract": json.dumps(desc, sort_keys=True),
             "real_bands": list(REAL_BANDS), "synthetic_bands": list(SYNTHETIC_BANDS),
             "synthetic_rules": json.dumps(SYNTHETIC_RULES, sort_keys=True), "purpose": PURPOSE,
             "temporal_extent": json.dumps([dates[0], (t[-1] + pd.Timedelta(days=1))
                                            .strftime("%Y-%m-%d")])}
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    dcz.write_store(dsx, {b: FILL for b in zc.BANDS}, out, attrs, log=lambda m: None,
                    verify=True)
    op = zc.overpass_path(site, out_root)
    old = {}
    if op.is_file():
        with open(op, encoding="utf-8") as f:
            old = json.load(f)
    old.update({d: lab[d] for d in dates})
    tmp = op.with_name(op.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(sorted(old.items())), f, indent=1)
        f.write("\n")
    os.replace(tmp, op)
    # the real bands read back through the harness reader, bit for bit
    try:
        with inputs.ZarrSource(site, root=str(out_root), instants="catalogue",
                               lattice=lattice) as src:
            if src.dates != dates:
                raise ExtractError(f"{out}: the reader sees {src.T} dates, {len(dates)} written")
            back = src.read_block(np.arange(src.T), list(REAL_BANDS))
        for b in ("B03", "B08"):
            if not np.array_equal(back[b], real[b]):
                raise ExtractError(f"{out}: {b} does not read back as the legacy values")
        if not np.array_equal(back["SCL"], obs.scl_u8(real["SCL"], FILL)):
            raise ExtractError(f"{out}: SCL does not read back as the legacy values")
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    desc.update(store=str(out).replace("\\", "/"), overpass=str(op).replace("\\", "/"),
                dates=dates, readback="B03, B08 and SCL equal the legacy values bit for bit",
                seconds=round(time.time() - t0, 1),
                bytes=int(sum(f.stat().st_size for f in out.rglob("*") if f.is_file())))
    jp = zc.sidecar_path(site, year, "extract.json", out_root)
    with open(jp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(desc, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    log(f"[extract] {site} {year}: {out} ({desc['bytes'] / 1e6:.1f} MB, lattice {lattice}, "
        f"{desc['seconds']} s); real {list(REAL_BANDS)}, synthetic {list(SYNTHETIC_BANDS)}")
    return desc


def extract_info(site, year, root):
    """The ``extract`` attribute of a store (dict) or None for a real store."""
    a = zc.store_attrs(zc.store_path(site, year, root))
    e = a.get("extract")
    if isinstance(e, str):
        try:
            return json.loads(e)
        except ValueError:
            return None
    return e if isinstance(e, dict) else None
