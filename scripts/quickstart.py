# -*- coding: utf-8 -*-
"""MAREA in one call: from a Sentinel-2 cube and a tide record to the maps.

:func:`map_estuary` runs the steps of ``guide/METHOD.md`` (section 10) in
their order, with the settings of the Villaviciosa product, by calling
existing functions of ``pyintertidal`` and ``scripts/marea_tools.py`` only.
It adds no algorithm and changes no default of the library:

====  ==============================================  =================================
step  what                                            function
====  ==============================================  =================================
4     the looks and the intertidal mask               ``marea.extract``
2     the acquisition time of every scene             ``overpass.get_overpass_times``
3     the scenes clear over the transition zone       ``marea_tools.screen``
5     the record: the latest 3-year epoch             ``marea_tools.build_record``
6-10  mouth, sections, delays, elevation and width    ``marea.reconstruct``
11    emersion time and the GeoTIFFs                  ``marea_tools.emersion_time``,
                                                      ``marea_tools.export_geotiffs``
====  ==============================================  =================================

Step 1, the download, is ``pyintertidal.SentinelCube(...).ensure(...)``
(notebook 03).

Example: the Marismas de Santoña of notebook 03, from the repository root,
with the EOT20 files in ``tide_models/``::

    import sys
    sys.path.insert(0, "scripts")
    from quickstart import map_estuary
    from pyintertidal.boundary import PyTMDBoundary

    # the tide at the mouth: EOT20 at the centroid of the box of notebook 03
    tide = PyTMDBoundary("EOT20", 43.415, -3.48, directory="tide_models")
    # ... or a record of your own (UTC instants, metres):
    # tide = GaugeBoundary(times_utc, levels_m, name="harbour gauge")
    paths = map_estuary("cubes/santona_2023-2025.nc", tide, "runs/santona",
                        mouth_side="north")

or, from a shell::

    python -X utf8 scripts/quickstart.py cubes/santona_2023-2025.nc runs/santona \\
        --lat 43.415 --lon -3.48 --mouth-side north

Heights come out in metres in the datum of the tide at the mouth (EOT20:
about mean sea level); delays in minutes; times are UTC.
"""
from __future__ import annotations

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
        ``GaugeBoundary`` (your record) or ``make_boundary``. It must cover
        the whole record and, for the emersion time, every 10 minutes of its
        years: ``marea_tools.TableBoundary`` (the replication kit's table of
        record instants only) is not enough.
    out_dir : str or Path
        Output folder, created if needed (never ``products_marea/`` or
        ``products_villaviciosa/``).
    overpass_times : dict, optional
        ``{'YYYY-MM-DD': UTC timestamp}`` of the scenes. None queries the
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
    # (6)-(10) the mouth, the distance along the water, the 3-km sections, one
    # delay per section, the delay of every pixel, its elevation and width
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


if __name__ == "__main__":
    import argparse

    from pyintertidal.boundary import PyTMDBoundary

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cube", help="netCDF cube with B03, B08 and SCL at 10 m")
    ap.add_argument("out_dir", help="output folder (e.g. runs/<site>)")
    ap.add_argument("--lat", type=float, required=True,
                    help="tide point at the mouth, degrees north")
    ap.add_argument("--lon", type=float, required=True,
                    help="tide point at the mouth, degrees east (west is negative)")
    ap.add_argument("--model", default="EOT20", help="pyTMD tide model (default EOT20)")
    ap.add_argument("--tide-dir", default="tide_models",
                    help="folder holding the model's files (default tide_models)")
    ap.add_argument("--mouth-side", default="any",
                    help="frame edge(s) holding the sea: any, north, south, east, west, "
                         "or several joined by '+'")
    ap.add_argument("--n-boot", type=int, default=30,
                    help="scene resamples for the delay intervals (0: none, faster)")
    a = ap.parse_args()
    boundary = PyTMDBoundary(a.model, a.lat, a.lon, directory=a.tide_dir)
    for path in map_estuary(a.cube, boundary, a.out_dir, mouth_side=a.mouth_side,
                            n_boot=a.n_boot):
        print(path)
