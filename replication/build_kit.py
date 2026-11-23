# -*- coding: utf-8 -*-
"""Build the offline replication kit of the Villaviciosa product.

The kit lets anyone rerun steps 6-11 of MAREA (the tide at the mouth, the
sections, the delay fit, the per-pixel delay, the inversion and the
hypsometry) without the 3.09 GB cube and without the 2.3 GB EOT20 model,
and get arrays identical to products_marea/marea.npz (Route A of
guide/REPRODUCE.md, ``scripts/reproduce_villaviciosa.py --from-kit``).

It writes, into ``--out`` (default replication/):

villaviciosa_record_2023-2025.npz
    The record of the product: the 97 record rows of the extraction
    (``marea.extract`` over the authors' cube), Y (97, 32120) float32 NDWI
    and C (97, 32120) bool clear looks, with keep, dates, shape, sea, inter,
    bbox, crs_wkt and the source cube's sha256.
villaviciosa_eot20_levels.npz
    EOT20 at the registry centroid of Villaviciosa, at the 97 record
    instants read every whole-minute shift from -45 to 180 min.
villaviciosa_eot20_10min_2023-2025.npz
    EOT20 at the same point every 10 min over 2023-2025, starting 180 min
    early so a delay of up to 180 min can be read (emersion time).
MANIFEST.json
    The sha256, size and arrays of each file, the inputs and how they were
    made.

Run it where the inputs are (from the repository root):

    python -X utf8 replication/build_kit.py --extract PATH/marea_extract.npz \
        --tide-dir PATH/tide_models [--cube PATH/ndwi_cube_villaviciosa_grande_10y.nc]

``--extract`` is the 142.5 MB extraction the product was made from (all
1379 scenes; not public). ``--cube`` is optional: it gives the CRS text
exactly as the cube stores it and, with ``--hash-cube``, re-checks the
cube's sha256 (3.09 GB read). The record rows are streamed, so memory stays
small.
"""
import argparse
import datetime as _dt
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import marea_tools as mt  # noqa: E402

PROD = ROOT / "products_marea"
SHIFTS = np.arange(-45, 181, dtype=np.int16)        # whole minutes, -45..180
SERIES_START = pd.Timestamp("2023-01-01")            # emersion period, start
SERIES_END = pd.Timestamp("2026-01-01")              # end exclusive
SERIES_MARGIN_MIN = 180                              # the largest delay the bank holds
SERIES_STEP_MIN = 10
#: shifts whose levels are recomputed one call each and compared bit for bit
#: with the single batched call
VERIFY_SHIFTS = (-45, 0, 6, 12, 16, 90, 180)

RECORD = "villaviciosa_record_2023-2025.npz"
LEVELS = "villaviciosa_eot20_levels.npz"
SERIES = "villaviciosa_eot20_10min_2023-2025.npz"


def npz_rows(path, key, rows):
    """Rows ``rows`` of a C-ordered 2-D array inside an npz, streamed.

    ``path`` an npz (compressed or not); ``key`` the array's name; ``rows``
    DISTINCT row indices, in any order. The array is read row by row up to
    the largest index, so memory holds only the output. Returns
    ``(len(rows), n_columns)`` in the order of ``rows``.
    """
    from numpy.lib import format as npf

    rows = np.asarray(rows)
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as f:
        version = npf.read_magic(f)
        shape, fortran, dtype = (npf.read_array_header_1_0(f) if version == (1, 0)
                                 else npf.read_array_header_2_0(f))
        if fortran or len(shape) != 2:
            raise ValueError(f"{key}: expected a C-ordered 2-D array")
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        out = np.empty((len(rows), shape[1]), dtype)
        want = {int(r): i for i, r in enumerate(rows)}
        for r in range(shape[0]):
            buf = f.read(row_bytes)
            if r in want:
                out[want[r]] = np.frombuffer(buf, dtype)
            if r >= rows.max():
                break
    return out


def npz_small(path, key):
    """One (small) array of an npz, read whole (``path``, array name ``key``)."""
    with np.load(path, allow_pickle=False) as d:
        return d[key]


def describe(path):
    """{key: [dtype, shape]} of the npz at ``path`` (the MANIFEST's ``arrays``)."""
    with np.load(path, allow_pickle=False) as d:
        return {k: [str(d[k].dtype), list(d[k].shape)] for k in d.files}


def main(argv=None):
    """Build the three kit files and MANIFEST.json (see the module docstring); returns 0.

    The checks below are ``assert`` statements: they ARE the kit's checks
    (replication/README.md: "the script checks, and stops if one fails"),
    so never run this script with ``python -O``, which removes asserts.
    """
    # the --help text is the first line of the module docstring
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extract", required=True,
                    help="products_marea/marea_extract.npz of the authors (all 1379 scenes)")
    ap.add_argument("--tide-dir", required=True, help="folder holding EOT20/ocean_tides/")
    ap.add_argument("--cube", help="ndwi_cube_villaviciosa_grande_10y.nc (optional: CRS text)")
    ap.add_argument("--hash-cube", action="store_true",
                    help="also re-check the cube's sha256 against result.json (reads 3.09 GB)")
    ap.add_argument("--out", default=str(ROOT / "replication"))
    a = ap.parse_args(argv)
    t_start = time.time()
    started = _dt.datetime.now(_dt.timezone.utc)
    out = mt.guard_out_dir(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import pyintertidal as pit
    from pyintertidal.boundary import PyTMDBoundary

    res = json.loads((PROD / "result.json").read_text(encoding="utf-8"))
    times = mt.load_overpass(PROD / "overpass_times.json")
    with np.load(PROD / "marea.npz") as prod:
        keep_p, shape_p = prod["keep"], prod["shape"]
    rec = np.array(res["record_dates"])
    print(f"product: {len(rec)} record dates, {res['n_px']:,} intertidal px, "
          f"frame {tuple(int(v) for v in shape_p)}")

    # ── the record rows of the extraction ────────────────────────────────
    # (the asserts in this function are the kit's checks: do not run with -O)
    dates = npz_small(a.extract, "dates").astype(str)
    keep = npz_small(a.extract, "keep")
    shape = npz_small(a.extract, "shape")
    assert np.array_equal(keep, keep_p), "the extract's pixels are not the product's"
    assert np.array_equal(shape, shape_p), "the extract's frame is not the product's"
    rows = np.searchsorted(dates, rec)
    assert np.array_equal(dates[rows], rec), "a record date is not in the extract"
    t0 = time.time()
    Y = npz_rows(a.extract, "Y", rows)
    C = npz_rows(a.extract, "C", rows)
    print(f"extract: {len(dates)} scenes; record rows {len(rows)} read "
          f"[{time.time() - t0:.0f} s]; Y {Y.dtype}{Y.shape}, C {C.dtype}{C.shape}")

    aoi = pit.sites.get("villaviciosa")
    lat, lon = (float(v) for v in aoi.centroid)
    bb = aoi.bbox                     # the box the production call passed with the record
    if a.cube:
        import pyproj
        import xarray as xr
        with xr.open_dataset(a.cube) as ds:
            wkt = ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref")
        crs_wkt = str(pyproj.CRS.from_wkt(wkt).to_wkt())
        crs_source = "the cube's crs variable, as marea.extract reads it"
    else:
        import pyproj
        crs_wkt = str(pyproj.CRS.from_epsg(32630).to_wkt())
        crs_source = "EPSG:32630 from pyproj (no cube given)"
    cube_sha = res["inputs_sha"]["cube"]
    cube_check = None
    if a.cube and a.hash_cube:
        t0 = time.time()
        got = mt.sha256(a.cube)
        cube_check = got == cube_sha
        print(f"cube sha256 {got[:12]}... {'matches' if cube_check else 'DIFFERS FROM'} "
              f"result.json [{time.time() - t0:.0f} s]")
        assert cube_check, "the cube is not the product's"

    rec_path = out / RECORD
    mt.write_npz(rec_path, {
        "Y": Y, "C": C, "keep": keep, "dates": rec.astype("<U10"),
        "shape": shape.astype(np.int64),
        "sea": npz_small(a.extract, "sea"), "inter": npz_small(a.extract, "inter"),
        "bbox": np.array([bb["west"], bb["south"], bb["east"], bb["north"]], np.float64),
        "crs_wkt": np.array(crs_wkt), "source_cube_sha256": np.array(cube_sha)})
    del Y, C
    print(f"-> {mt.rel(rec_path)} ({rec_path.stat().st_size / 1e6:.1f} MB)")

    # ── EOT20 at the record instants, every whole-minute shift ───────────
    bnd = PyTMDBoundary("EOT20", lat, lon, directory=str(a.tide_dir))
    t_rec = pd.DatetimeIndex(pd.to_datetime([times[d] for d in rec])).tz_localize(None)
    inst = pd.DatetimeIndex(np.concatenate(
        [(t_rec - pd.Timedelta(minutes=float(s))).values for s in SHIFTS]))
    assert len(np.unique(inst.values)) == len(inst), "two shifted instants coincide"
    t0 = time.time()
    flat = np.asarray(bnd.levels(inst), float)        # returned sorted by time
    lev = np.empty(len(inst))
    lev[np.argsort(inst.values, kind="stable")] = flat
    lev = lev.reshape(len(SHIFTS), len(t_rec))
    print(f"EOT20 at {len(t_rec)} instants x {len(SHIFTS)} shifts: one call "
          f"[{time.time() - t0:.0f} s]")
    verified = {}
    for s in VERIFY_SHIFTS:
        one = np.asarray(bnd.levels(t_rec - pd.Timedelta(minutes=float(s))), float)
        verified[str(s)] = bool(np.array_equal(one, lev[list(SHIFTS).index(s)]))
    print(f"per-shift calls identical to the batched call: {verified}")
    assert all(verified.values()), "the batched levels differ from per-shift calls"
    h0 = lev[list(SHIFTS).index(0)]
    assert [float(h0.min()), float(h0.max())] == res["rango_marea_m"], \
        "the zero-delay range is not the product's rango_marea_m"
    lev_path = out / LEVELS
    mt.write_npz(lev_path, {
        "times_ns": t_rec.asi8.astype(np.int64),
        "times_utc": np.array([str(t) for t in t_rec], "<U19"),
        "dates": rec.astype("<U10"), "shifts_min": SHIFTS,
        "levels_m": lev, "model": np.array("EOT20"),
        "lat": np.float64(lat), "lon": np.float64(lon)})
    tb = mt.TableBoundary(lev_path)
    assert np.array_equal(tb.levels(t_rec - pd.Timedelta(minutes=12)), lev[list(SHIFTS).index(12)])
    print(f"-> {mt.rel(lev_path)} ({lev_path.stat().st_size / 1e6:.2f} MB)")

    # ── EOT20 every 10 min over 2023-2025, with the delay margin ─────────
    grid = pd.date_range(SERIES_START - pd.Timedelta(minutes=SERIES_MARGIN_MIN),
                         SERIES_END, freq=f"{SERIES_STEP_MIN}min", inclusive="left")
    t0 = time.time()
    ser = np.asarray(bnd.levels(grid), float)
    print(f"EOT20 every {SERIES_STEP_MIN} min, {grid[0]} .. {grid[-1]}: "
          f"{len(grid):,} values [{time.time() - t0:.0f} s]")
    ser_path = out / SERIES
    mt.write_npz(ser_path, {
        "times_ns": grid.asi8.astype(np.int64), "levels_m": ser.astype(np.float32),
        "step_min": np.int64(SERIES_STEP_MIN), "margin_min": np.int64(SERIES_MARGIN_MIN),
        "model": np.array("EOT20"), "lat": np.float64(lat), "lon": np.float64(lon)})
    print(f"-> {mt.rel(ser_path)} ({ser_path.stat().st_size / 1e6:.2f} MB)")

    # ── manifest ─────────────────────────────────────────────────────────
    from importlib import metadata
    eot = Path(a.tide_dir) / "EOT20" / "ocean_tides"
    files = {
        RECORD: "the 97 record rows of the extraction of the authors' cube "
                "(marea.extract), with keep, dates, shape, sea, inter, bbox (the "
                "registry polygon's box, which the production call passed), "
                "crs_wkt and the source cube's sha256",
        LEVELS: f"EOT20 at the registry centroid ({lat:.6f}, {lon:.6f}) at the 97 "
                "record instants minus every whole-minute shift -45..180 "
                "(levels_m[i, j] = level at times[j] - shifts_min[i])",
        SERIES: f"EOT20 at the same point every {SERIES_STEP_MIN} min from "
                f"{grid[0]} to {grid[-1]} UTC (float32); the first "
                f"{SERIES_MARGIN_MIN} min are the margin for delays up to "
                f"{SERIES_MARGIN_MIN} min",
    }
    manifest = {
        "kit": "MAREA replication kit, Ria de Villaviciosa, record 2023-2025",
        "reproduces": "products_marea/ (steps 6-11) with "
                      "scripts/reproduce_villaviciosa.py --from-kit replication",
        "made_by": "python -X utf8 replication/build_kit.py --extract <marea_extract.npz> "
                   "--tide-dir <tide_models>" + (" --cube <cube>" if a.cube else "")
                   + (" --hash-cube" if a.hash_cube else ""),
        "git_commit": mt._git("rev-parse", "HEAD") or None,
        "made_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ("numpy", "pandas", "pyTMD", "eo-tides")},
        "conventions": "times UTC (datatake start); levels in m on the EOT20 scale "
                       "(about mean sea level); shifts in min, the level read that many "
                       "minutes earlier",
        "files": {name: {"sha256": mt.sha256(out / name), "bytes": (out / name).stat().st_size,
                         "contents": desc, "arrays": describe(out / name)}
                  for name, desc in files.items()},
        "inputs": {
            "extraction": {"file": "products_marea/marea_extract.npz (not public)",
                           "sha256": mt.sha256(a.extract),
                           "bytes": Path(a.extract).stat().st_size,
                           "scenes": int(len(dates))},
            "source_cube": {"file": "ndwi_cube_villaviciosa_grande_10y.nc (not public)",
                            "sha256": cube_sha, "sha256_rechecked": cube_check,
                            "crs_wkt_from": crs_source},
            "product": {f: mt.sha256(PROD / f) for f in
                        ("result.json", "marea.npz", "overpass_times.json")},
            "tide_model": {"model": "EOT20", "point_lat_lon": [lat, lon],
                           "point": "pyintertidal.sites.get('villaviciosa').centroid",
                           "files": sorted(p.name for p in eot.glob("*.nc"))},
        },
        "checks": {
            "extract keep and shape equal products_marea/marea.npz": True,
            "record dates equal products_marea/result.json record_dates": True,
            "zero-shift range equals result.json rango_marea_m": True,
            "per-shift EOT20 calls identical to the batched call": verified,
        },
        "licences": "Contains modified Copernicus Sentinel data (2023-2025). EOT20: "
                    "Hart-Davis et al. (2021), SEANOE, doi:10.17882/79489, CC BY 4.0. "
                    "Kit: CC BY 4.0 (LICENSE-docs.md).",
        "doi": "to be added on release",
    }
    mt.write_json(out / "MANIFEST.json", manifest)
    print(f"-> {mt.rel(out / 'MANIFEST.json')}")
    for name, f in manifest["files"].items():
        print(f"   {name}  {f['bytes']:>10,} B  sha256 {f['sha256']}")
    print(f"done in {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
