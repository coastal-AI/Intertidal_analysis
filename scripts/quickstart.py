# -*- coding: utf-8 -*-
"""MAREA in one call: from a Sentinel-2 cube and a tide series to the maps.

:func:`map_estuary` runs steps 2 to 11 of ``guide/METHOD.md`` (section 10)
in the order of the table below. Step 4 comes first because step 2 reads
the scenes' dates and the frame's box from its output. It uses the settings
of the Villaviciosa product except the quality rule, which is on (3, the
library default), and calls existing functions of ``pyintertidal`` and
``scripts/marea_tools.py`` only. It adds no algorithm and changes no default
of the library:

====  ==============================================  =================================
step  what                                            function
====  ==============================================  =================================
4     the looks and the intertidal mask               ``marea.extract``
2     the acquisition time of every scene             the ``overpass_times`` argument, else
                                                      ``overpass.get_overpass_times``
3     the scenes clear over the transition zone       ``marea_tools.screen``
5     the record: the latest 3-year epoch             ``marea_tools.build_record``
6     the tide at the mouth                           the ``tide`` argument
7-10  mouth, sections, delays, elevation and width,   ``marea.reconstruct``
      and the hypsometry of step 11
11    emersion time and the GeoTIFFs                  ``marea_tools.emersion_time``,
                                                      ``marea_tools.export_geotiffs``
====  ==============================================  =================================

Step 1, the download, is ``pyintertidal.SentinelCube(...).ensure(...)``
(notebook 03, and the short version of ``guide/NEW_ESTUARY.md``).

Example: the Marismas de Santoña of notebook 03, from the repository root,
with the EOT20 files in ``tide_models/``::

    import sys
    sys.path.insert(0, "scripts")
    from quickstart import map_estuary, read_tide_csv
    from pyintertidal.boundary import GaugeBoundary, PyTMDBoundary

    # the tide at the mouth: EOT20 at (43.415 N, 3.48 W), the centroid of the
    # box of notebook 03. That point lies inland: it is a placeholder only.
    # Replace it with a point in open water off the mouth, checked on a map.
    tide = PyTMDBoundary("EOT20", 43.415, -3.48, directory="tide_models")
    # ... or a gauge series of your own (UTC instants, metres):
    # tide = GaugeBoundary(times_utc, levels_m, name="harbour gauge")
    # ... or a CSV file of your own (the format of --tide-csv, below):
    # tide = read_tide_csv("cubes/santona_tide.csv")
    paths = map_estuary("cubes/santona_2023-2025.nc", tide, "runs/santona",
                        mouth_side="north")

or, from a shell, with the same placeholder point (replace it as above)::

    python -X utf8 scripts/quickstart.py cubes/santona_2023-2025.nc runs/santona \\
        --lat 43.415 --lon -3.48 --mouth-side north

Offline: your own tide series and saved acquisition times
---------------------------------------------------------
With ``--tide-csv`` and ``--overpass-times`` the command needs neither a tide
model nor the network::

    python -X utf8 scripts/quickstart.py cubes/santona_2023-2025.nc runs/santona \\
        --tide-csv cubes/santona_tide.csv \\
        --overpass-times cubes/santona_overpass_times.json --mouth-side north

``--tide-csv`` takes the tide at the mouth from a CSV file instead of a tide
model, read by :func:`read_tide_csv`: a header row and the columns ``time``
and ``level_m``::

    time,level_m
    2022-12-31T21:00:00Z,0.5217
    2022-12-31T21:10:00Z,0.5876

``time`` is ISO 8601: an instant with an offset (``+01:00``) or ``Z`` is
converted to UTC, an instant without one is taken as UTC, and an instant
may not appear twice. ``level_m`` is the water level in metres, and the
heights come out on its zero. The samples can be regular or not; between two
of them the level is interpolated linearly, across a gap of any length too:
a straight line over a gap of a few hours can be far from the tide, so fill
the gaps first (from a tide model, for example), or use ``GaugeBoundary`` in
Python, which reads an instant more than an hour from a sample as missing.

An instant outside the series is an error, so the series must reach every
scene of the cube and, for the emersion time, every 10 minutes of the
record's calendar years (the latest 3 of the cube), from 3 hours before
their first 1 January (MAREA reads the tide up to 180 minutes early) to
23:50 on 31 December of the last. For a cube of 2023-2025 that is from
2022-12-31 21:00 to 2025-12-31 23:50 UTC. The command checks this before it
starts (:func:`tide_span_needed`). A cube that ends during a year makes that
whole calendar year part of the record, so with a series of your own,
download whole calendar years.

``--overpass-times`` reads the acquisition times from a JSON file in the
format of ``products_marea/overpass_times.json``,
``{"YYYY-MM-DD": "YYYY-MM-DD HH:MM:SS"}`` in UTC without an offset (an
offset is dropped, not converted), with ``marea_tools.load_overpass``; a
scene of the cube without an instant there leaves the record. Without it
they come from the Sentinel-2 STAC catalogue (``overpass.get_overpass_times``:
metadata only, no credits, but it needs the network). Nothing in this
edition writes that file; to save it once, while online, with the query
that :func:`map_estuary` makes::

    import json, sys
    sys.path.insert(0, "scripts")
    import marea_tools as mt
    from pyintertidal import overpass

    info = mt.cube_info("cubes/santona_2023-2025.nc")      # the cube's dates and box
    times = overpass.get_overpass_times(info["bbox"], (info["dates"][0], info["dates"][-1]))
    with open("cubes/santona_overpass_times.json", "w", encoding="utf-8") as f:
        json.dump({d: str(t) for d, t in times.items()}, f)  # "YYYY-MM-DD HH:MM:SS", UTC

Heights come out in metres in the datum of the tide at the mouth (EOT20:
about mean sea level); delays in minutes; times are UTC.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import marea_tools as mt                     # puts the repository root on sys.path
from pyintertidal import marea, overpass


def map_estuary(cube_path, tide, out_dir, overpass_times=None, site=None,
                mouth_side="any", epoch_years=3, min_scenes=40, n_boot=30,
                emersion_step_min=10):
    """Elevation, width, delay and emersion-time maps of one estuary.

    Parameters
    ----------
    cube_path : str or Path
        netCDF cube with B03, B08 and SCL on a 10 m UTM grid, e.g. from
        ``pyintertidal.SentinelCube(...).ensure(...)``. The frame must reach
        the open sea: the mouth is the permanent water touching its edge.
    tide : BoundaryProvider
        The water level at the mouth: anything whose ``levels(times)``
        returns metres for naive-UTC instants, e.g.
        ``pyintertidal.boundary.PyTMDBoundary`` (a tide model),
        ``GaugeBoundary`` (your gauge series), ``make_boundary``, or
        :func:`read_tide_csv` (a CSV file of your own). It must give a level
        at every usable scene of the cube, from 180 minutes before to 45
        minutes after each scene of the record (the delays searched), and,
        for the emersion time, every 10 minutes of the record's years from
        up to 180 minutes before 1 January (the delay of the inland pixels;
        :func:`tide_span_needed` gives the span for a cube):
        ``marea_tools.TableBoundary`` (the replication kit's table of record
        instants only) is not enough. Outside its record a ``GaugeBoundary``
        reads NaN (the scene leaves the record, the instant leaves the
        emersion time), while the ``marea_tools.TideSeries`` that
        :func:`read_tide_csv` returns raises ``ValueError``.
    out_dir : str or Path
        Output folder, created if needed (never ``products_marea/`` or
        ``products_villaviciosa/``).
    overpass_times : dict, optional
        ``{'YYYY-MM-DD': UTC timestamp}`` of the scenes, e.g.
        ``marea_tools.load_overpass(path)`` of a saved file. None queries the
        Sentinel-2 STAC catalogue over the cube's dates (metadata only: it
        needs the network and costs no openEO credits).
    site : str, optional
        Short name used in the file names (default: the cube's file name).
    mouth_side : str
        Which frame edge(s) hold the sea: ``"any"``, ``"north"``, ``"south"``,
        ``"east"``, ``"west"``, or several joined by ``"+"``.
    epoch_years : int
        The record is the screened scenes of the latest ``epoch_years`` years.
    min_scenes : int
        The run is refused with fewer scenes in the record (40, as for the
        Villaviciosa product; ``marea.reconstruct`` alone defaults to 80).
    n_boot : int
        Scene resamples for the 95 % interval of each section's delay (0
        skips the interval and is several times faster).
    emersion_step_min : int
        Step (min) of the tide grid on which emersion time is computed.

    Returns
    -------
    list of Path
        The GeoTIFFs ``marea_<site>_<epoch>_<layer>.tif`` for the layers
        ``z``, ``sigma``, ``tau_px``, ``s_km``, ``band`` and ``emersion``, on
        the cube's grid. ``out_dir`` also holds ``marea.npz``,
        ``result.json`` and ``figure.png`` from ``marea.reconstruct``.

    Raises
    ------
    RuntimeError
        When ``marea.reconstruct`` refuses the run (too few intertidal
        pixels or scenes: the reason is in ``result.json``).
    """
    cube_path = Path(cube_path)
    out_dir = mt.guard_out_dir(out_dir)              # never into the published products
    site = site or cube_path.stem

    # (4) the looks and the intertidal mask, in one streaming pass over the cube
    ex = marea.extract(str(cube_path))
    # (2) the acquisition time of every scene, UTC
    if overpass_times is None:
        overpass_times = overpass.get_overpass_times(
            ex["bbox"], (ex["dates"][0], ex["dates"][-1]), verbose=False)
    times = {str(d): pd.Timestamp(t) for d, t in overpass_times.items()}
    # (3) the scenes with at most 10 % cloud over the transition zone
    usable, _, _ = mt.screen(str(cube_path))
    # (5) the record: usable scenes with a time and a tide level, in the latest epoch
    ex_rec, _ = mt.build_record(ex, usable, times, tide, epoch_years=epoch_years)
    # (7)-(10) the mouth, the distance along the water, the 3-km sections, one
    # delay per section, the delay of every pixel, its elevation and width,
    # and the hypsometry of (11); the tide of (6) is the ``tide`` argument
    with warnings.catch_warnings():
        # s_km is NaN where water cannot reach; tide_estimators.make_bands_fixed
        # casts it to int before masking those pixels to band -1, which numpy
        # reports as an invalid cast. Harmless; silenced for this call only,
        # with the filter of reproduce_villaviciosa.py.
        warnings.filterwarnings("ignore", message="invalid value encountered in cast",
                                category=RuntimeWarning)
        result = marea.reconstruct(str(cube_path), str(out_dir), name=site,
                                   boundary=tide, extraction=ex_rec,
                                   overpass_times=times, mouth_side=mouth_side,
                                   min_scenes=min_scenes, n_boot=n_boot)
    if result is None:
        raise RuntimeError(f"reconstruct refused the run: see {out_dir / 'result.json'}")
    # (11) emersion time over the record's years, then the GeoTIFFs
    epoch = ex_rec["record_epoch"]                   # e.g. '2023-2025'
    first, last = (int(y) for y in epoch.split("-"))
    with np.load(out_dir / "marea.npz") as d:
        z, tau_px = d["z"], d["tau_px_min"]
    emersion = mt.emersion_time(z, tau_px, tide, f"{first}-01-01", f"{last + 1}-01-01",
                                step_min=emersion_step_min)   # share of time out of the water
    return mt.export_geotiffs(out_dir, str(cube_path), out_dir, site, epoch,
                              layers=("z", "sigma", "tau_px_min", "s_km", "band",
                                      "emersion"),
                              extra={"emersion": emersion})


def read_tide_csv(path):
    """The tide at the mouth from a CSV file of your own: what ``--tide-csv`` reads.

    Parameters
    ----------
    path : str or Path
        A CSV file with a header row and the columns ``time`` and
        ``level_m`` (any other column is ignored), e.g.::

            time,level_m
            2022-12-31T21:00:00Z,0.5217
            2022-12-31T22:10:00+01:00,0.5876

        ``time``: ISO 8601 instants, in any order and at any spacing, each
        one once. An instant with an offset (``+01:00``) or ``Z`` is
        converted to UTC; an instant without one is taken as UTC.
        ``level_m``: the water level at the mouth (m); the elevations come
        out on its zero.

    Returns
    -------
    marea_tools.TideSeries
        Named after the file. Its ``levels(times)`` interpolates linearly
        between the samples, across a gap too (fill long gaps first), and
        raises ``ValueError`` for an instant outside the series; ``span``
        gives its first and last instants (UTC).

    Raises
    ------
    ValueError
        Naming the problem: the file cannot be read as CSV text, a column is
        missing, a time is not an ISO 8601 instant, an instant appears twice
        (a record kept in local time without offsets repeats an hour when
        the clocks go back), a level is not a finite number, or there are
        fewer than two rows.
    """
    path = Path(path)
    try:
        # every cell as text, so that a bad value is reported, not guessed
        df = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: cannot be read as a CSV text file ({exc})") from None
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in ("time", "level_m") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: no column {' or '.join(repr(c) for c in missing)}; the "
                         f"header row must name the columns 'time' and 'level_m' (it names "
                         f"{', '.join(repr(c) for c in df.columns)})")
    if len(df) < 2:
        raise ValueError(f"{path}: a tide series needs at least two rows")
    # one row at a time, with the standard library: pandas.to_datetime(utc=True)
    # can read an instant without an offset in the offset of the rows before it,
    # and pandas.to_numeric can change the last digit of a level
    times, levels, row_of = [], [], {}
    for row, (text, value) in enumerate(zip(df["time"], df["level_m"])):
        where = f"{path}, row {row + 1} after the header"
        try:
            t = datetime.fromisoformat(str(text).strip().upper())   # ISO 8601 ('t', 'z' too)
        except ValueError:
            raise ValueError(f"{where}: time {text!r} is not an ISO 8601 instant") from None
        try:
            v = float(value)                                   # exactly the number written
        except ValueError:
            v = float("nan")
        if not np.isfinite(v):                                 # empty, not a number, or infinite
            raise ValueError(f"{where}: level_m {value!r} is not a number of metres")
        # an offset or Z is converted to UTC; an instant without one is taken as UTC
        t = t if t.tzinfo is None else t.astimezone(timezone.utc).replace(tzinfo=None)
        if t in row_of:                                        # two levels for one instant
            raise ValueError(f"{where}: time {text!r} is {t} UTC, the instant of row "
                             f"{row_of[t] + 1} again; a tide series needs each instant once "
                             f"(a record kept in local time repeats an hour when the clocks "
                             f"go back)")
        row_of[t] = row
        times.append(t)
        levels.append(v)
    return mt.TideSeries(times, levels, name=path.name)


def tide_span_needed(dates, epoch_years=3, emersion_step_min=10):
    """The first and last instant a tide series must reach for :func:`map_estuary`.

    What ``--tide-csv`` checks before the run, so that a short series stops
    it at once instead of after the fit.

    Parameters
    ----------
    dates : sequence of str
        The cube's scene dates, ``'YYYY-MM-DD'`` (``marea_tools.cube_info``).
    epoch_years, emersion_step_min : int
        As in :func:`map_estuary` (its defaults, which the command line uses).

    Returns
    -------
    (first, last, years)
        Naive UTC ``pandas.Timestamp``: the earlier of 00:00 on the cube's
        first date (step 5 reads the tide at every scene of the cube) and
        180 minutes before 1 January of the record's first year (the
        emersion time of step 11 reads the tide up to the largest delay
        early); and the last step of the emersion grid, 23:50 on
        31 December of the cube's last year. ``years`` is the record's
        calendar years, e.g. ``'2023-2025'``: the latest ``epoch_years`` of
        the cube, as ``marea_tools.build_record`` anchors them.
    """
    dates = sorted(str(d) for d in dates)
    year_first, year_last = int(dates[0][:4]), int(dates[-1][:4])
    record_first = max(year_last - epoch_years + 1, year_first)
    early = pd.Timedelta(minutes=marea.TAU_BOUNDS[1])           # the largest delay, 180 min
    first = min(pd.Timestamp(dates[0]), pd.Timestamp(f"{record_first}-01-01") - early)
    last = pd.Timestamp(f"{year_last + 1}-01-01") - pd.Timedelta(minutes=emersion_step_min)
    return first, last, f"{record_first}-{year_last}"


def main(argv=None):
    """The command line (``--help``; the formats are in the module docstring).

    ``argv`` is the list of arguments (default: those of the command line).
    Prints the paths of the GeoTIFFs of :func:`map_estuary`. Options that do
    not go together, a ``--tide-csv`` or ``--overpass-times`` file that is
    missing or cannot be read, or a ``--tide-csv`` series that does not reach
    the span the cube needs (:func:`tide_span_needed`), stop it before any
    computation with a usage message and exit code 2.
    """
    import argparse

    from pyintertidal.boundary import PyTMDBoundary

    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="With --tide-csv and --overpass-times the run needs neither a tide model nor "
               "the network. Run it from the repository root. The file formats, and how to "
               "save the acquisition times once: the docstring of scripts/quickstart.py.")
    ap.add_argument("cube", help="netCDF cube with B03, B08 and SCL at 10 m")
    ap.add_argument("out_dir", help="output folder (e.g. runs/<site>)")
    ap.add_argument("--mouth-side", default="any",
                    help="frame edge(s) holding the sea: any, north, south, east, west, "
                         "or several joined by '+'")
    ap.add_argument("--n-boot", type=int, default=30,
                    help="scene resamples for the delay intervals (0: none, faster)")
    tg = ap.add_argument_group(
        "the tide at the mouth",
        "a pyTMD tide model at a point (--lat, --lon; --model, --tide-dir), or a series of "
        "your own (--tide-csv)")
    tg.add_argument("--lat", type=float,
                    help="tide point at the mouth, degrees north (needed without --tide-csv; "
                         "not used with it)")
    tg.add_argument("--lon", type=float,
                    help="tide point at the mouth, degrees east, west is negative (needed "
                         "without --tide-csv; not used with it)")
    tg.add_argument("--model", help="pyTMD tide model (default EOT20)")
    tg.add_argument("--tide-dir", help="folder holding the model's files (default tide_models)")
    tg.add_argument("--tide-csv", metavar="PATH",
                    help="the tide at the mouth from a CSV file instead of a model: a header "
                         "row and the columns time (ISO 8601; an offset or Z is converted to "
                         "UTC, an instant without one is taken as UTC; each instant once) and "
                         "level_m (metres), regular or not, read by linear interpolation, "
                         "across gaps too (fill long gaps first). It must reach every scene "
                         "of the cube and the record's calendar years (the latest 3), from "
                         "3 hours before their first 1 January to 23:50 on 31 December of "
                         "the last; this is checked before the run. Not with --model or "
                         "--tide-dir")
    og = ap.add_argument_group("the acquisition time of every scene")
    og.add_argument("--overpass-times", metavar="PATH",
                    help="a JSON file in the format of products_marea/overpass_times.json "
                         "({\"YYYY-MM-DD\": \"YYYY-MM-DD HH:MM:SS\"}, UTC without an offset: "
                         "an offset is dropped, not converted). Without it the times are "
                         "read from the Sentinel-2 STAC catalogue, which needs the network")
    a = ap.parse_args(argv)

    # (6) the tide at the mouth: a tide model at a point, or a series of your own
    if a.tide_csv is None:
        missing = [flag for flag, value in (("--lat", a.lat), ("--lon", a.lon)) if value is None]
        if missing:
            ap.error(f"the following arguments are required: {', '.join(missing)} "
                     f"(or give --tide-csv)")
        tide = PyTMDBoundary("EOT20" if a.model is None else a.model, a.lat, a.lon,
                             directory="tide_models" if a.tide_dir is None else a.tide_dir)
    else:
        model_flags = [flag for flag, value in (("--model", a.model), ("--tide-dir", a.tide_dir))
                       if value is not None]
        if model_flags:
            ap.error(f"argument --tide-csv: not allowed with {' or '.join(model_flags)} "
                     f"(the CSV replaces the tide model)")
        if not Path(a.tide_csv).is_file():
            ap.error(f"argument --tide-csv: no file {a.tide_csv}")
        try:
            tide = read_tide_csv(a.tide_csv)
        except ValueError as exc:
            ap.error(f"argument --tide-csv: {exc}")
        first, last = tide.span
        print(f"tide at the mouth: {a.tide_csv}, {first} .. {last} UTC", flush=True)
        if a.lat is not None or a.lon is not None:
            print("--lat and --lon are not used with --tide-csv", flush=True)
        # the span the run will read, checked now: a series that falls short
        # would otherwise stop the run only at the emersion time, after the fit
        if not Path(a.cube).is_file():
            ap.error(f"argument cube: no file {a.cube}")
        need_first, need_last, years = tide_span_needed(mt.cube_info(a.cube)["dates"])
        if first > need_first or last < need_last:
            ap.error(f"argument --tide-csv: the series runs from {first} to {last} UTC, but "
                     f"this cube needs it from {need_first} to {need_last}: the tide at every "
                     f"scene of the cube and, for the emersion time, every 10 minutes of the "
                     f"record's calendar years, {years}, read up to 180 minutes early. Extend "
                     f"the series; the emersion time covers whole calendar years, so a cube "
                     f"that ends during a year needs the series to 31 December of that year")

    # (2) the acquisition times: a saved file, else the STAC catalogue (in map_estuary)
    times = None
    if a.overpass_times is not None:
        if not Path(a.overpass_times).is_file():
            ap.error(f"argument --overpass-times: no file {a.overpass_times}")
        try:
            times = mt.load_overpass(a.overpass_times)
        except (ValueError, TypeError, AttributeError) as exc:     # not JSON, or not instants
            ap.error(f"argument --overpass-times: {a.overpass_times} is not a JSON object "
                     f"{{\"YYYY-MM-DD\": \"YYYY-MM-DD HH:MM:SS\"}} ({exc})")
        if not times:
            ap.error(f"argument --overpass-times: {a.overpass_times} holds no instants")
        print(f"acquisition times: {a.overpass_times}, {len(times):,} dates from "
              f"{min(times)} to {max(times)}", flush=True)

    for path in map_estuary(a.cube, tide, a.out_dir, overpass_times=times,
                            mouth_side=a.mouth_side, n_boot=a.n_boot):
        print(path)


if __name__ == "__main__":
    main()
