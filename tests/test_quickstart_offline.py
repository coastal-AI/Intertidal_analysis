# -*- coding: utf-8 -*-
"""The command line of ``scripts/quickstart.py`` runs offline, with ``--tide-csv`` and ``--overpass-times``.

Two parts:

* fast units: :func:`quickstart.read_tide_csv`, the reader of ``--tide-csv`` (an offset is converted to UTC, an
  instant without one is taken as UTC, a missing column, a bad value or a repeated instant is named),
  :func:`quickstart.tide_span_needed`, and the argument checks of :func:`quickstart.main`, which stop before any
  computation (small files written into a temporary folder; the check of the series' span reads the dates of the
  notebook sample, and is skipped without it);
* an end-to-end run of the command line in its own process, as a user would run it, on the notebook sample
  ``data/villaviciosa_sample.nc`` (skipped without it; marked ``slow``, about 30 s): the tide at the mouth from a CSV
  written here, the acquisition times from ``products_marea/overpass_times.json``, no bootstrap (``--n-boot 0``),
  into a temporary folder. No tide model and no network are used. The sample's block does not reach the open sea
  (``guide/NEW_ESTUARY.md``, sections 2 and 5), so its maps mean little: the run checks the chain and its counts,
  which are those of the same call made in Python with ``map_estuary`` (92 scenes, 13,294 pixels, 10,491 with an
  elevation).

The CSV holds the sample's own EOT20 series (``series_time``, ``tide_series``: every 10 minutes from 2023-01-01 00:00
UTC) preceded by the 3 hours before it, taken from the replication kit's series, which is the same series with those
3 hours more. They are needed: the series is read as a ``marea_tools.TideSeries``, which refuses an instant outside
its span, the emersion time reads the tide of 1 January 2023 up to the largest pixel delay earlier (15 min here), and
the command line refuses, before the run, a series that does not reach 3 hours before 1 January.
"""
import json  # result.json
import os  # paths
import subprocess  # the command line runs in its own process
import sys  # the same Python as the tests
from pathlib import Path  # the printed paths

import numpy as np  # arrays
import pytest  # the test runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
SAMPLE = os.path.join(ROOT, "data", "villaviciosa_sample.nc")  # the notebook sample: 270 x 270 px, 97 scenes
KIT_SERIES = os.path.join(ROOT, "replication", "villaviciosa_eot20_10min_2023-2025.npz")  # from 21:00 on 2022-12-31
LAYERS = ("z", "sigma", "tau_px", "s_km", "band", "emersion")  # the six GeoTIFFs quickstart writes

sys.path.insert(0, os.path.join(ROOT, "scripts"))  # scripts/quickstart.py and scripts/marea_tools.py
import quickstart  # noqa: E402  (importing it puts the repository root on sys.path)


def write_csv(path, text):
    """Write ``text`` to ``path`` (UTF-8, LF) and return the path."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def test_offsets_are_converted_and_naive_instants_are_utc(tmp_path):
    """An instant with +01:00 or Z is converted to UTC, one without is UTC; levels are interpolated in between."""
    csv = write_csv(tmp_path / "tide.csv", "time,level_m\n"
                                           "2024-01-01T13:00:00+01:00,12.0\n"   # 12:00 UTC
                                           "2024-01-01T14:00:00,14.0\n"         # no offset: 14:00 UTC, not in +01:00
                                           "2024-01-01T13:00:00Z,13.0\n")       # 13:00 UTC (any order is fine)
    tide = quickstart.read_tide_csv(csv)
    first, last = tide.span  # naive UTC
    assert (str(first), str(last)) == ("2024-01-01 12:00:00", "2024-01-01 14:00:00")  # converted, not dropped
    assert tide.levels(["2024-01-01T12:00:00", "2024-01-01T13:30:00", "2024-01-01T14:00:00"]).tolist() == \
        [12.0, 13.5, 14.0]  # the level of 12:00 UTC is the one written at 13:00+01:00
    assert tide.name == "tide.csv"  # named after the file, as in its error messages


def test_an_instant_outside_the_series_is_an_error(tmp_path):
    """The series returned raises ValueError outside its span (as marea_tools.TideSeries does)."""
    csv = write_csv(tmp_path / "tide.csv", "time,level_m\n2024-01-01T12:00:00Z,1.0\n2024-01-01T13:00:00Z,2.0\n")
    tide = quickstart.read_tide_csv(csv)
    with pytest.raises(ValueError, match="covers"):
        tide.levels(["2024-01-01T11:59:00"])  # a minute before the first sample


def test_a_missing_column_is_named(tmp_path):
    """A file without the columns time and level_m gives a ValueError that names them and what the file has."""
    csv = write_csv(tmp_path / "gauge.csv", "date,level\n2024-01-01T12:00:00Z,1.0\n2024-01-01T13:00:00Z,2.0\n")
    with pytest.raises(ValueError) as err:
        quickstart.read_tide_csv(csv)
    msg = str(err.value)
    assert "'time'" in msg and "'level_m'" in msg  # what is needed
    assert "'date'" in msg and "'level'" in msg  # what the file holds


@pytest.mark.parametrize("row, what", [("12/01/2024 13:00,2.0", "time"),  # not ISO 8601
                                       ("2024-01-01T13:00:00Z,", "level_m"),  # an empty level
                                       ("2024-01-01T13:00:00Z,1.2 m", "level_m")])  # not a number
def test_a_bad_value_is_named_with_its_row(tmp_path, row, what):
    """A time that is not ISO 8601, or a level that is not a number, is named with its row."""
    csv = write_csv(tmp_path / "tide.csv", f"time,level_m\n2024-01-01T12:00:00Z,1.0\n{row}\n")
    with pytest.raises(ValueError, match=f"row 2 after the header: {what}"):
        quickstart.read_tide_csv(csv)


def test_a_repeated_instant_is_refused(tmp_path):
    """The same instant twice, here once with an offset and once in UTC, is named with both rows."""
    csv = write_csv(tmp_path / "tide.csv", "time,level_m\n"
                                           "2024-01-01T12:00:00Z,1.0\n"
                                           "2024-01-01T12:10:00Z,1.1\n"
                                           "2024-01-01T13:00:00+01:00,0.9\n")   # 12:00 UTC again
    with pytest.raises(ValueError, match="row 3 after the header: .* the instant of row 1 again"):
        quickstart.read_tide_csv(csv)


def test_the_span_needed_follows_the_record_years():
    """The tide must reach every scene and the record's calendar years, from 180 min before their first 1 January."""
    first, last, years = quickstart.tide_span_needed(["2023-01-06", "2024-06-01", "2025-12-28"])  # a 3-year cube
    assert (str(first), str(last), years) == ("2022-12-31 21:00:00", "2025-12-31 23:50:00", "2023-2025")
    first, last, years = quickstart.tide_span_needed(["2016-01-03", "2020-06-01", "2025-12-28"])  # 10 years
    assert (str(first), str(last), years) == ("2016-01-03 00:00:00", "2025-12-31 23:50:00", "2023-2025")
    first, last, years = quickstart.tide_span_needed(["2023-01-06", "2025-12-28", "2026-01-03"])  # into 2026
    assert (str(first), str(last), years) == ("2023-01-06 00:00:00", "2026-12-31 23:50:00", "2024-2026")


def run_main(argv, capsys):
    """Call quickstart.main and return (exit code, stderr); it must stop with a usage error."""
    with pytest.raises(SystemExit) as stop:
        quickstart.main(argv)
    return stop.value.code, capsys.readouterr().err


def test_the_tide_point_is_needed_without_a_csv(capsys):
    """Without --tide-csv, --lat and --lon are required, as before the option existed."""
    code, err = run_main(["cube.nc", "out"], capsys)
    assert code == 2 and "--lat, --lon" in err


def test_a_model_option_is_refused_with_a_csv(tmp_path, capsys):
    """--model and --tide-dir choose a tide model: they cannot be combined with --tide-csv."""
    csv = write_csv(tmp_path / "tide.csv", "time,level_m\n2024-01-01T12:00:00Z,1.0\n2024-01-01T13:00:00Z,2.0\n")
    for extra in (["--model", "EOT20"], ["--tide-dir", "tide_models"]):
        code, err = run_main(["cube.nc", "out", "--tide-csv", str(csv)] + extra, capsys)
        assert code == 2 and "--tide-csv: not allowed with " + extra[0] in err


def test_an_unreadable_csv_is_a_usage_error(tmp_path, capsys):
    """A CSV without the needed columns stops the command line with its message, not a traceback."""
    csv = write_csv(tmp_path / "gauge.csv", "date,level\n2024-01-01T12:00:00Z,1.0\n2024-01-01T13:00:00Z,2.0\n")
    code, err = run_main(["cube.nc", "out", "--tide-csv", str(csv)], capsys)
    assert code == 2 and "no column 'time' or 'level_m'" in err


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="data/villaviciosa_sample.nc is not present")
@pytest.mark.parametrize("start, end", [("2023-01-01T00:00:00Z", "2025-12-31T23:50:00Z"),   # starts at midnight
                                        ("2022-12-31T21:00:00Z", "2025-12-29T00:00:00Z")])  # ends before 31 December
def test_a_series_too_short_for_the_cube_stops_before_the_run(tmp_path, capsys, start, end):
    """A series that misses the 3 hours before 1 January, or the end of 31 December, is refused before any fit."""
    csv = write_csv(tmp_path / "tide.csv", f"time,level_m\n{start},0.0\n{end},0.0\n")  # only its span matters here
    code, err = run_main([SAMPLE, str(tmp_path / "run"), "--tide-csv", str(csv)], capsys)
    assert code == 2 and "needs it from 2022-12-31 21:00:00 to 2025-12-31 23:50:00" in err  # the span of 2023-2025
    assert not (tmp_path / "run").exists()  # nothing was computed or written


def test_lat_and_lon_with_a_csv_are_reported_as_unused(tmp_path, capsys):
    """--lat and --lon are not needed with --tide-csv: given anyway, the command line says it does not use them."""
    csv = write_csv(tmp_path / "tide.csv", "time,level_m\n2024-01-01T12:00:00Z,1.0\n2024-01-01T13:00:00Z,2.0\n")
    with pytest.raises(SystemExit):  # it stops at the missing cube, after the note
        quickstart.main([str(tmp_path / "no_cube.nc"), "out", "--tide-csv", str(csv), "--lat", "43.5", "--lon", "-5.4"])
    out = capsys.readouterr()
    assert "--lat and --lon are not used with --tide-csv" in out.out and "no file" in out.err


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="data/villaviciosa_sample.nc is not present")
def test_the_command_line_maps_the_sample_offline(tmp_path):
    """--tide-csv and --overpass-times: exit 0, the six GeoTIFFs, estado ok and the counts of the Python call."""
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(SAMPLE) as s:
        t_sample = s["series_time"].values  # every 10 min, 2023-01-01 00:00 .. 2025-12-31 23:50 (UTC)
        h_sample = s["tide_series"].values  # EOT20 at the sample's boundary point (m)
    with np.load(KIT_SERIES) as k:
        t_kit = k["times_ns"].astype("datetime64[ns]")  # the same series from 2022-12-31 21:00
        h_kit = k["levels_m"]
    before = t_kit < t_sample[0]  # the kit's 18 samples (3 h) before the sample's series
    assert np.array_equal(t_kit[~before], t_sample) and np.array_equal(h_kit[~before], h_sample)  # the same series
    times = pd.DatetimeIndex(np.concatenate([t_kit[before], t_sample])).strftime("%Y-%m-%dT%H:%M:%SZ")
    levels = np.concatenate([h_kit[before], h_sample]).astype(float).tolist()  # repr() below keeps every bit
    csv = tmp_path / "villaviciosa_tide.csv"
    with open(csv, "w", encoding="utf-8", newline="\n") as f:
        f.write("time,level_m\n")
        f.writelines(f"{t},{v!r}\n" for t, v in zip(times, levels))

    out = tmp_path / "run"  # never inside the repository
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    cmd = [sys.executable, "-X", "utf8", os.path.join("scripts", "quickstart.py"),
           os.path.join("data", "villaviciosa_sample.nc"), str(out),
           "--tide-csv", str(csv), "--overpass-times", os.path.join("products_marea", "overpass_times.json"),
           "--n-boot", "0"]
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=900)
    tail = (run.stdout + "\n" + run.stderr)[-4000:]  # what to show if it fails
    assert run.returncode == 0, tail

    tifs = [out / f"marea_villaviciosa_sample_2023-2025_{layer}.tif" for layer in LAYERS]
    assert all(p.is_file() for p in tifs), tail  # the six GeoTIFFs
    printed = [Path(line.strip()) for line in run.stdout.splitlines() if line.strip().endswith(".tif")]
    assert [p.resolve() for p in printed] == [p.resolve() for p in tifs], tail  # their paths: the output
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert res["estado"] == "ok"  # status
    assert res["overpass_source"] == "given"  # the acquisition times came from the file: no catalogue query
    assert (res["n_escenas"], res["n_px"], res["n_px_cota"]) == (92, 13294, 10491)  # scenes, pixels, elevations
