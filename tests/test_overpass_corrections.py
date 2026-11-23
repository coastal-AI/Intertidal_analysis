# -*- coding: utf-8 -*-
"""The overpass times: the known errors of the downloader's files are put right when they are read (configs/
overpass_corrections.json), a correction refuses a file that no longer holds the wrong instant, and the catalogue search
ignores a product whose tile lies in a far UTM zone (the one that gave Urdaibai's 2019-02-16 a time of 22:26 UTC).

These tests read pyintertidal.overpass.load_overpass, which applies the corrections; scripts/marea_tools.load_overpass
reads the same files without them. No network: the catalogue search itself is not called."""
import json                                                            # the files
import os                                                              # paths
import sys                                                             # the repository on the path

import pandas as pd                                                    # instants
import pytest                                                          # the test runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the repository's root
sys.path.insert(0, ROOT)                                               # import its package (also done by conftest.py)

from pyintertidal import overpass as ov  # noqa: E402  the module under test


def write(path, obj):
    """Write ``obj`` as a small JSON file (a stand-in overpass-time file)."""
    with open(path, "w", encoding="utf-8") as f:                       # a small JSON file
        json.dump(obj, f)


def test_the_corrections_file_holds_urdaibai_and_its_provenance():
    table = json.load(open(ov.CORRECTIONS, encoding="utf-8"))          # the project's corrections
    fix = table["products_urdaibai_10m/overpass_times.json"]["2019-02-16"]   # the one known error
    assert fix["was"] == "2019-02-16T22:26:01" and fix["time"] == "2019-02-16T11:11:29"
    assert "T30TWP" in fix["source"] and "T01MAV" in fix["source"]     # where both instants come from


def test_load_overpass_corrects_only_the_listed_file(tmp_path):
    site = tmp_path / "products_urdaibai_10m"                          # a file with the same name as the real one
    site.mkdir()
    write(site / "overpass_times.json", {"2019-02-11": "2019-02-11T11:06:31", "2019-02-16": "2019-02-16T22:26:01"})
    times, changed = ov.load_overpass(str(site / "overpass_times.json"))   # read with the corrections
    assert times["2019-02-16"] == pd.Timestamp("2019-02-16T11:11:29")  # put right
    assert times["2019-02-11"] == pd.Timestamp("2019-02-11T11:06:31")  # the rest untouched
    assert changed == {"2019-02-16": (pd.Timestamp("2019-02-16T22:26:01"), pd.Timestamp("2019-02-16T11:11:29"))}
    assert json.load(open(site / "overpass_times.json"))["2019-02-16"] == "2019-02-16T22:26:01"   # the file is not rewritten
    other = tmp_path / "products_foz_10m"                              # a file with no correction listed
    other.mkdir()
    write(other / "overpass_times.json", {"2019-02-16": "2019-02-16T22:26:01"})
    times, changed = ov.load_overpass(str(other / "overpass_times.json"))
    assert changed == {} and times["2019-02-16"] == pd.Timestamp("2019-02-16T22:26:01")   # nothing applied


def test_a_correction_refuses_a_file_that_changed(tmp_path):
    site = tmp_path / "products_urdaibai_10m"
    site.mkdir()
    write(site / "overpass_times.json", {"2019-02-16": "2019-02-16T10:00:00"})   # not the wrong instant listed
    with pytest.raises(ValueError, match="expects"):
        ov.load_overpass(str(site / "overpass_times.json"))


def test_far_tiles_are_recognised():
    assert ov.utm_zone(-2.7) == 30 and ov.utm_zone(-7.1) == 29 and ov.utm_zone(179.9) == 60   # the zones of a longitude
    assert ov._tile_zone("S2A_MSIL2A_20190216T222601_N0500_R072_T01MAV_20221128T090754") == 1   # the wrong product
    assert ov._tile_zone("S2B_MSIL2A_20190216T111129_N0500_R137_T30TWP_20221129T174552") == 30  # the right one
    assert ov._tile_zone("no tile here") is None
