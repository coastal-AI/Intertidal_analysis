# -*- coding: utf-8 -*-
"""Band subsets of the Zarr downloader (``--bands``, e.g. B03 B08 SCL), no network.

The subset is opt-in: these tests check that it reaches the process graph,
the job title and the store; that its stores never go into the nine-band
root (the root guard); that a year the reference netCDF covers lands bitwise
and passes the LIGHT-PATH GATE against the same year's nine-band store (and
that a store differing from it fails the gate); that a year without a
nine-band store of its own waits until a store of its bands passed that
gate; that a year without reference takes the site's known offset only when
the median composites of a registered NINE-BAND store (read from --root)
confirm it on a delivery of its array shape, and that its SCL is checked at
the 20 m phases; the dry run's input megapixels and credit estimate,
from the measured cost per megapixel; the converter's 2023 extraction gate
before a purchase at its sites, a light-path gate failing on the dates
alone, the nine-band root guard, and --expect-account (a fake connection). The nine-band behaviour is
pinned by tests/test_download_cube_zarr.py and tests/test_zarr_angles.py,
which run unchanged. Frames of 64 x 64 px, deliveries of at most 256 x 256
px, at most 5 dates.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import download_cube_zarr as dcz
from experiments import zarr_cubes as zc
from tests.test_download_cube_zarr import (D21, D22, D23, D24, FRAME64, Site, World, _ref_site,
                                           fake_build, frame_nc, frame_values, make_request,
                                           normalise_and_write, padded_delivery, place, synth,
                                           write_raw_v2, write_server_zarr, zip_store)

LIGHT = ("B03", "B08", "SCL")
D20 = pd.date_range("2020-06-06", periods=5, freq="5D")
D19 = pd.date_range("2019-06-07", periods=4, freq="5D")
D18 = pd.date_range("2018-06-08", periods=4, freq="5D")

assert fake_build                               # the fixture, imported for these tests


@pytest.fixture(scope="module")
def wkt():
    from pyproj import CRS

    return CRS.from_epsg(32630).to_wkt("WKT1_GDAL")


def deliver_light(site, year, world, dates, offset, shape, tweak=None, **kw):
    """What a B03 B08 SCL job of ``year`` delivers (``padded_delivery`` of
    the three bands, the backend's Zarr layout), as ``site``'s next delivery
    of that year."""
    data, xf, yf = padded_delivery(world, dates, offset, shape, **kw)
    data = {b: data[b] for b in LIGHT}
    if tweak:
        tweak(data)
    src = site.tmp / "src_light" / str(year)
    raw = write_server_zarr(src / "out.zarr", dates, yf, xf, data, site.wkt, extent=FRAME64)
    site.deliveries[year] = zip_store(raw, src / "out.zarr")
    return data


def light_request(site, year, light=None, main_root="nine"):
    """The B03 B08 SCL request of ``year`` under ``light`` (default
    ``<tmp>/cubes_light``), its nine-band root ``site.root`` (read only)."""
    main_root = site.root if main_root == "nine" else main_root
    return dcz.year_requests(["villaviciosa"], [year], light or site.tmp / "cubes_light", ["main"],
                             bands=LIGHT, main_root=main_root)[0]


def run_light(site, req):
    msgs = []
    res = dcz.run_request(site.ctx, req, msgs.append)
    return res, "\n".join(msgs)


def fake_gate_store(root, tmp, wkt, site="escalda", year=2023, how=None):
    """A tiny complete B03 B08 SCL store of ``site`` under ``root``, accepted
    as ``how`` (default: by the light-path gate, as escalda 2023 would be)."""
    req = make_request(root, year=year, site=site)
    req.bands = LIGHT
    t, y, x, data = synth(year, n=2, H=8, W=8)
    raw = write_raw_v2(tmp / "src_gate" / site / "a.zarr", t, y, x, {b: data[b] for b in LIGHT},
                       wkt)
    place(req, [("a.zarr", raw, "x")])
    normalise_and_write(req)
    dcz.mark_complete(req.store, {"verification": json.dumps({"how": how or dcz.NINE_BAND_HOW})})
    return req.store


def _attrs(store):
    return zc.store_attrs(store)


def write_extraction_gate(folder, site, ok, accepted=False, screening=True):
    """A converter 2023 extraction gate report (light_to_extraction ``gate``)."""
    g = {"ok": ok, "strict_ok": ok, "dates": {"equal": True},
         "screening": {"equal_record_dates": screening},
         "extraction": {"keep_px_in_static_mask": 0}, "legacy_dn": {"differ_pct_of_valid_both": 0.74},
         "store": {"path": f"cubes/{site}/2023.zarr"}, "utc": "2026-10-26T12:00:00"}
    if accepted:
        g["accepted"] = {"by": "light_to_extraction gate --accept-gate-differences", "utc": g["utc"]}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{site}_gate_2023.json").write_text(json.dumps(g), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  the request: graph, title, paths; the nine bands untouched
# ─────────────────────────────────────────────────────────────────────────────

def test_the_subset_reaches_the_graph_and_the_title_and_nothing_else(tmp_path, monkeypatch, wkt):
    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", FRAME64, wkt))
    cubes, light = tmp_path / "cubes", tmp_path / "cubes_light"
    nine = dcz.year_requests(["villaviciosa"], [2023], cubes, ["main"])[0]
    sub = dcz.year_requests(["villaviciosa"], [2023], light, ["main"], bands=LIGHT,
                            main_root=cubes)[0]
    assert not nine.is_subset and nine.bands == dcz.BANDS and nine.main_root is None
    assert nine.tag == "villaviciosa 2023" and dcz._subset_write_kw(nine) == {}
    assert sub.is_subset and sub.bands == LIGHT and sub.root == light and sub.main_root == cubes
    assert sub.extent == nine.extent                          # the nine bands' frame
    assert sub.store == light / "villaviciosa" / "2023.zarr" and Path(sub.part).parent == \
        light / "villaviciosa"
    assert sub.tag == "villaviciosa 2023 B03+B08+SCL"
    assert dcz._subset_write_kw(sub) == {"radiometry_bands": ("B12", "B11", "B08")}
    g9, gs = (dcz.build_process(r).flat_graph() for r in (nine, sub))
    assert gs["loadcollection1"]["arguments"]["bands"] == list(LIGHT)
    want = copy.deepcopy(g9)
    want["loadcollection1"]["arguments"]["bands"] = list(LIGHT)
    assert gs == want                                         # nothing else differs
    h9, hs = dcz.graph_hash(g9), dcz.graph_hash(gs)
    assert h9 != hs
    assert dcz.job_title(nine, h9) == f"zarr_cube_villaviciosa_2023_{h9[:8]}"
    assert dcz.job_title(sub, hs) == f"zarr_cube_B03_B08_SCL_villaviciosa_2023_{hs[:8]}"
    # the angles of a mixed call are never a subset
    both = dcz.year_requests(["villaviciosa"], [2023], light, ["main", "angles"], bands=LIGHT,
                             main_root=cubes)
    assert both[0].is_subset and not both[1].is_subset and both[1].bands == zc.ANGLE_BANDS
    # the registration plan: a subset also reads the nine-band root and needs a confirmation
    p9 = dcz.registration_plan(nine, dcz.Context(), None)
    ps = dcz.registration_plan(sub, dcz.Context(), None)
    assert p9.sibling_roots == () and not p9.require_confirmed and p9.root == cubes
    assert ps.sibling_roots == (cubes,) and ps.require_confirmed and ps.root == light
    assert dcz.subset_resampling(LIGHT) == (
        "resample_spatial(resolution=10, method='near') on the whole cube, as the existing netCDF "
        "cubes: identity for the native 10 m bands (B03 B08); nearest neighbour 20 m -> 10 m for SCL")


def test_the_options(tmp_path):
    a = dcz.parse_args(["escalda", "--years", "2023"])
    assert a.bands is None and a.store_root is None and a.products == ["main", "angles"]
    a = dcz.parse_args(["escalda", "--years", "2022", "--bands", "scl, b08,B03"])
    assert a.bands == LIGHT and a.products == ["main"] and a.store_root is None
    a = dcz.parse_args(["ems", "--bands", "B03,B08,B12,SCL", "--products", "main",
                        "--store-root", "x", "--offset-px", "288,271"])
    assert a.bands == ("B03", "B08", "B12", "SCL") and a.offset_px == (288, 271)
    assert dcz.parse_args(["ems", "--bands", ",".join(dcz.BANDS)]).bands is None   # the nine
    for bad in (["ems", "--bands", "B03,B08"], ["ems", "--bands", "B03,B8,SCL"],
                ["ems", "--store-root", "x"], ["ems", "--bands", "B03,B08,SCL", "--test"],
                ["--test", "--odd-window", "--bands", "B03,B08,SCL"],
                ["ems", "--bands", "B03,B08,SCL", "--lonlat-extent"],
                ["ems", "--bands", "B03,B08,SCL", "--products", "angles"],
                ["ems", "--bands", "B03,B08,SCL", "--products", "main,angles"]):
        with pytest.raises(SystemExit):
            dcz.parse_args(bad)
    with pytest.raises(ValueError, match=r"\['B08'\] missing"):
        dcz.parse_bands("B03,SCL")
    a = dcz.parse_args(["ems", "--expect-account", "fer@example.org"])         # any run
    assert a.expect_account == "fer@example.org" and a.bands is None
    assert dcz.parse_args(["ems"]).expect_account is None
    for bad in (["ems", "--extraction-gate-dir", "x"], ["ems", "--without-extraction-gate"],
                ["ems", "--expect-account", "fer"]):
        with pytest.raises(SystemExit):
            dcz.parse_args(bad)


def test_the_store_root_guard(tmp_path, wkt):
    """A band subset never writes beside stores of other bands: not the
    nine-band root, not inside it or around it, not a root holding a store of
    another band set (refused before anything, also by ``run_request``)."""
    cubes, light = tmp_path / "cubes", tmp_path / "cubes_light"
    assert dcz.subset_root_problem(light, LIGHT, cubes) is None
    assert "is the root of the nine-band stores" in dcz.subset_root_problem(cubes, LIGHT, cubes)
    assert "lies inside the nine-band root" in dcz.subset_root_problem(cubes / "light", LIGHT,
                                                                       cubes)
    assert "holds the nine-band root" in dcz.subset_root_problem(tmp_path, LIGHT, cubes)
    nine = make_request(light, year=2023)                     # nine bands under the light root
    t, y, x, data = synth(2023, n=2, H=8, W=8)
    place(nine, [("a.zarr", write_raw_v2(tmp_path / "src" / "a.zarr", t, y, x, data, wkt), "x")])
    normalise_and_write(nine)
    why = dcz.subset_root_problem(light, LIGHT, cubes)
    assert "2023.zarr holds the bands ['B02', 'B03'" in why and "not ['B03', 'B08', 'SCL']" in why
    sub = dcz.Request(site="villaviciosa", year=2022, start="2022-01-01", end="2023-01-01",
                      bbox={}, root=light, bands=LIGHT, main_root=cubes)
    with pytest.raises(dcz.FormatError, match="holds the bands"):
        dcz.run_request(dcz.Context(), sub, lambda m: None)
    assert not Path(sub.part).exists()
    nine.store.rename(nine.store.with_name("2023.zarr.elsewhere"))
    assert dcz.subset_root_problem(light, LIGHT, cubes) is None


def test_years_without_a_nine_band_store_wait_for_the_light_path_gate(tmp_path, wkt):
    """The cheap verification year comes first: a subset year without a
    nine-band store of its own is refused before any job until a store of the
    same bands (any site) was accepted by the light-path gate; a year with
    its own nine-band store is a gate year and never waits;
    --without-light-gate overrides."""
    cubes, light = tmp_path / "cubes", tmp_path / "cubes_light"
    sub = dcz.Request(site="wadden", year=2022, start="2022-01-01", end="2023-01-01", bbox={},
                      root=light, bands=LIGHT, main_root=cubes)
    assert "has passed the light-path gate yet" in dcz.subset_gate_problem(sub, dcz.Context())
    with pytest.raises(dcz.LightGateMissing, match="nothing started"):
        dcz.run_request(dcz.Context(), sub, lambda m: None)
    assert not Path(sub.part).exists()
    ctx = dcz.Context(dcz.parse_args(["wadden", "--bands", "B03,B08,SCL",
                                      "--without-light-gate", "--without-extraction-gate"]))
    assert ctx.without_light_gate and dcz.subset_gate_problem(sub, ctx) is None
    assert dcz.light_wait_problem(sub, ctx) is None
    with pytest.raises(SystemExit):
        dcz.parse_args(["wadden", "--without-light-gate"])
    nine = make_request(cubes, year=2023, site="wadden")       # the year's own nine-band store
    t, y, x, data = synth(2023, n=2, H=8, W=8)
    place(nine, [("a.zarr", write_raw_v2(tmp_path / "src" / "a.zarr", t, y, x, data, wkt), "x")])
    normalise_and_write(nine)
    sub23 = dcz.Request(site="wadden", year=2023, start="2023-01-01", end="2024-01-01", bbox={},
                        root=light, bands=LIGHT, main_root=cubes)
    assert dcz.subset_gate_problem(sub23, dcz.Context()) is None
    fake_gate_store(light, tmp_path, wkt, site="ferrol", how="not verified")   # does not count
    assert dcz.light_gate_passed(light, LIGHT) == []
    gate = fake_gate_store(light, tmp_path, wkt)                # escalda 2023 passed the gate
    assert dcz.light_gate_passed(light, LIGHT) == [str(gate)]
    assert dcz.light_gate_passed(light, ("B03", "B08", "B12", "SCL")) == []
    # the Wadden is a converter site: its own 2023 extraction gate must have passed too
    gdir = tmp_path / "gates"
    ctx = dcz.Context(dcz.parse_args(["wadden", "--bands", "B03,B08,SCL", "--extraction-gate-dir",
                                      str(gdir)]))
    assert "no readable 2023 extraction gate of wadden" in dcz.subset_gate_problem(sub, ctx)
    with pytest.raises(dcz.ExtractionGateMissing, match="nothing started"):
        dcz.run_request(ctx, sub, lambda m: None)
    assert not Path(sub.part).exists()
    write_extraction_gate(gdir, "wadden", ok=True)
    assert dcz.subset_gate_problem(sub, ctx) is None
    sub_v = dcz.Request(site="villaviciosa", year=2022, start="2022-01-01", end="2023-01-01",
                        bbox={}, root=light, bands=LIGHT, main_root=cubes)
    assert dcz.subset_gate_problem(sub_v, dcz.Context()) is None        # not a converter site


# ─────────────────────────────────────────────────────────────────────────────
#  a year the reference covers: bitwise, and the light-path gate
# ─────────────────────────────────────────────────────────────────────────────

def test_a_light_year_with_a_reference_lands_bitwise_and_passes_the_gate(tmp_path, monkeypatch,
                                                                         wkt, fake_build):
    """2024 (the reference covers it): the nine-band store first, then the
    B03 B08 SCL job: registered against the netCDF at the site's offset,
    bitwise the frame, and identical to the nine-band store (the light-path
    gate: accepted as such). The nine-band store is only read. --verify-only
    with --bands decides the same. A delivery that differs from the nine-band
    store in one pixel-date fails the gate (said, recorded), whatever the rule
    against the reference then decides."""
    world = World(seed=21)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    req9, res9, _ = site.run(2024)
    assert res9["registration"] == "netcdf [72, 72]" and res9["accepted"] == "identical"
    before = (req9.store / ".zmetadata").read_bytes()
    deliver_light(site, 2024, world, D24, (72, 72), (192, 192))
    req = light_request(site, 2024)
    res, text = run_light(site, req)
    assert res["registration"] == "netcdf [72, 72]" and res["verdict"] == "identical", text
    assert res["accepted"] == dcz.NINE_BAND_HOW
    assert res["light_path_gate"] == "passed" and res["job_costs"] == 2.5
    job = site.conn.jobs[res["job"]]
    assert job["title"].startswith("zarr_cube_B03_B08_SCL_villaviciosa_2024_")
    led = json.loads((Path(req.part) / "job.json").read_text())
    assert led["jobs"][-1]["process_graph"]["loadcollection1"]["arguments"]["bands"] == list(LIGHT)
    ds = zc.open_store(req.store)
    assert zc.band_names(ds) == list(LIGHT)
    want = frame_values(world, D24)
    for b in LIGHT:
        assert ds[b].dtype == np.int16
        np.testing.assert_array_equal(ds[b].values, want[b], err_msg=b)
    ds.close()
    a = _attrs(req.store)
    assert a["complete"] == 1 and json.loads(a["verification"])["how"] == dcz.NINE_BAND_HOW
    assert json.loads(a["bands"]) == list(LIGHT) == json.loads(a["band_subset"])
    assert json.loads(a["band_subset_of"]) == list(dcz.BANDS)
    assert a["nine_band_root"] == str(site.root) and a["format"] == zc.FORMAT_ID
    assert a["title"].endswith("(bands B03 B08 SCL)") and "for SCL" in a["resampling"]
    assert json.loads(a["radiometry"])["band"] == "B08"
    rec = json.loads(a["registration"])
    assert rec["netcdf"]["scl_confirmed"] and rec["phase_20m_scl"]["verdict"] == "confirmed"
    assert rec["phase_20m_scl"]["shares"]["00"] == 1.0
    gate = json.loads(req.sidecar("vs_nine_band.json").read_text())
    assert gate["passed"] and gate["comparison"]["verdict"] == "identical"
    assert gate["checks"]["delivered_offset_px"] == {"subset": [72, 72], "nine_band": [72, 72],
                                                     "equal": True}
    assert gate["checks"]["delivered_shape_px"]["equal"]
    assert all(s["identical"] and s["pixels"] == 4 * 64 * 64
               for s in gate["comparison"]["per_band"].values())
    rep = json.loads(req.sidecar("verify.json").read_text())
    assert rep["nine_band_store"]["passed"] and rep["acceptance"]["how"] == dcz.NINE_BAND_HOW
    assert "light-path gate PASSED: 4 = 4 dates, B03 B08 SCL bitwise equal" in text
    assert (req9.store / ".zmetadata").read_bytes() == before          # only read
    cube = zc.open_cube("villaviciosa", "2024-01-01", "2025-01-01", root=req.root)
    assert zc.band_names(cube) == list(LIGHT)
    cube.close()
    # --verify-only --bands: the same decision, from the command line (no network)
    zc.write_json({str(d.date()): f"{d.date()}T11:00:00" for d in D24},
                  zc.overpass_path("villaviciosa", req.root))
    monkeypatch.chdir(tmp_path)
    assert dcz.main(["villaviciosa", "--years", "2024", "--bands", "B03,B08,SCL", "--root",
                     str(site.root), "--store-root", str(req.root), "--verify-only"]) == 0
    acc = json.loads(req.sidecar("verify.json").read_text())["acceptance"]
    assert acc["how"] == dcz.NINE_BAND_HOW and acc["by"] == "--verify-only"
    # one pixel-date of B08 off by 1: the gate fails; the rule against the reference decides

    def tweak(data):
        data["B08"][1, 72 + 10, 72 + 10] += 1

    deliver_light(site, 2024, world, D24, (72, 72), (192, 192), tweak=tweak)
    req2 = light_request(site, 2024, light=tmp_path / "cubes_light_2")
    site.ctx.keep_raw = False                 # a near-identical nine-band year would drop its raw
    res2, text2 = run_light(site, req2)
    assert Path(req2.part).is_dir() and "a band subset keeps its raw delivery" in text2
    # the job of the same title is reattached (found on the backend), never paid twice
    assert site.conn.created == ["j-1", "j-2"] and res2["job"] == "j-2"
    assert "reattaching to job j-2" in text2
    assert res2["accepted"] == "near-identical" and res2["light_path_gate"] == "FAILED"
    assert "LIGHT-PATH GATE FAILED: the two stores differ: dates 4 / 4" in text2
    assert "; B08 1 differing pixel-dates" in text2
    gate2 = json.loads(req2.sidecar("vs_nine_band.json").read_text())
    assert not gate2["passed"] and gate2["comparison"]["per_band"]["B08"]["differing"] == 1
    assert gate2["comparison"]["per_band"]["B08"]["dates_differing"] == {str(D24[1].date()): 1}
    assert "the LIGHT-PATH GATE FAILED" in json.loads(_attrs(req2.store)["verification"])["why"]


# ─────────────────────────────────────────────────────────────────────────────
#  years without reference: the known offset, confirmed or refused
# ─────────────────────────────────────────────────────────────────────────────

def test_a_light_year_without_reference_needs_the_nine_band_stores_to_confirm_it(
        tmp_path, monkeypatch, wkt, fake_build):
    """The reference covers 2024 only. With the site's known offset
    (--offset-px) a B03 B08 SCL year is registered when the median composites
    against the registered NINE-BAND 2024 store (under --root, only read)
    confirm it: bitwise, 'not verified', its SCL at phase (0, 0). Refused:
    another offset; a delivery of another array shape (unless
    --accept-unverified-offset, recorded); a cloudy year the composites
    cannot measure (a nine-band year would be taken with a warning); no
    registered store anywhere; an SCL one pixel off."""
    world = World(seed=22)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    req24, _, _ = site.run(2024)
    site.ctx.offset_px = (72, 72)
    deliver_light(site, 2023, world, D23, (72, 72), (192, 192))
    req = light_request(site, 2023)
    jobs = list(site.conn.created)
    with pytest.raises(dcz.LightGateMissing, match=r"has passed the light-path gate yet"):
        run_light(site, req)                             # the gate year comes first: no job
    assert site.conn.created == jobs and not Path(req.part).exists()
    gate = fake_gate_store(req.root, tmp_path, wkt)      # escalda 2023 passed it (another site)
    res, text = run_light(site, req)
    assert res["registration"] == "expert --offset-px [72, 72]" and res["accepted"] == "not verified"
    assert res["light_path_gate"] is None                    # no nine-band 2023 to compare
    a = _attrs(req.store)
    assert json.loads(a["light_gate"]) == {"passed_by": [str(gate)], "passed_on_common_dates": [],
                                           "own_nine_band_store": False, "without_light_gate": False,
                                           "extraction_gate": None, "without_extraction_gate": False}
    rec = json.loads(a["registration"])
    assert rec["checked"] == "confirmed by the median composites against the registered 2024 store"
    assert rec["reference"] == str(req24.store) and rec["sibling"]["outcome"] == "confirmed"
    assert rec["phase_20m_scl"]["verdict"] == "confirmed"
    assert "this band subset's 20 m layer (SCL) is checked at the phases" in a["delivered_notes"]
    assert "20 m layer is not confirmed" not in a["delivered_warnings"]
    ds = zc.open_store(req.store)
    want = frame_values(world, D23)
    for b in LIGHT:
        np.testing.assert_array_equal(ds[b].values, want[b], err_msg=b)
    ds.close()
    assert not req.sidecar("vs_nine_band.json").exists()            # no nine-band 2023
    assert "SCL 20 m phase confirmed" in text
    # another offset: the composites put the frame elsewhere
    deliver_light(site, 2022, world, D22, (70, 72), (192, 192))
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"--offset-px 72,72 contradicts the median composites against the "
                             r"registered 2024 store, which put the frame at \(70, 72\)"):
        run_light(site, light_request(site, 2022))
    # the right offset in arrays of another shape: confirmed, but not from the same layout
    deliver_light(site, 2021, world, D21, (72, 72), (256, 256))
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"not the shape that store was delivered in \(\[192, 192\]\)"):
        run_light(site, light_request(site, 2021))
    site.ctx.accept_unverified_offset = True
    jobs = list(site.conn.created)
    res21, _ = run_light(site, light_request(site, 2021))
    assert site.conn.created == jobs                    # resumed from the kept download, no job
    rec21 = json.loads(_attrs(light_request(site, 2021).store)["registration"])
    assert res21["registration"] == "expert --offset-px [72, 72]"
    assert rec21["checked"].endswith("delivered in another array shape: --accept-unverified-offset")
    site.ctx.accept_unverified_offset = False
    # a cloudy year: nothing measures, so a subset refuses what the nine bands would take
    cloudy = World(seed=22, cloudy=D19)
    deliver_light(site, 2019, cloudy, D19, (72, 72), (192, 192))
    with pytest.raises(dcz.RegistrationRefused, match=r"is not confirmed by the median composites "
                                                      r"against the registered 2024 store"):
        run_light(site, light_request(site, 2019))
    # no registered store under either root
    deliver_light(site, 2018, world, D18, (72, 72), (192, 192))
    with pytest.raises(dcz.RegistrationRefused, match=r"--offset-px 72,72 cannot be checked"):
        run_light(site, light_request(site, 2018, main_root=tmp_path / "no_cubes"))
    # the SCL one pixel off the 10 m bands: the composites confirm, the 20 m phase refuses
    deliver_light(site, 2020, world, D20, (72, 72), (192, 192), offset20=(72, 73),
                  bands20=("SCL",))
    with pytest.raises(dcz.RegistrationRefused, match=r"SCL 20 m phase CONTRADICTED"):
        run_light(site, light_request(site, 2020))
    for y in (2022, 2019, 2018, 2020):
        assert not light_request(site, y).store.exists()
    assert zc.list_years("villaviciosa", site.root) == [2024]           # the nine bands: read only


# ─────────────────────────────────────────────────────────────────────────────
#  the dry run: input megapixels and credits from the measured cost per megapixel
# ─────────────────────────────────────────────────────────────────────────────

def test_the_credit_estimate():
    """Against a nine-band job a subset costs its share of the input: 3 / 9
    (every band on the 10 m grid) to 2.25 / 5.25 (the 20 m bands at their
    native resolution); a job that recorded its input picks one; a
    calibration of the same bands gives one number."""
    px = 1536 * 2176
    nine = {"site": "escalda", "year": 2023, "credits": 900.0, "dates": 146,
            "delivered_shape": [1536, 2176], "bands": list(dcz.BANDS)}
    est = dcz.credit_estimate(LIGHT, px, 146, [nine])
    assert est["credits_grid"] == pytest.approx(300.0)
    assert est["credits_native"] == pytest.approx(900.0 * 2.25 / 5.25)
    assert (est["credits_lo"], est["credits_hi"]) == (est["credits_grid"], est["credits_native"])
    assert est["input_grid_mpx"] == pytest.approx(px * 146 * 3 / 1e6)
    assert est["input_native_mpx"] == pytest.approx(px * 146 * 2.25 / 1e6)
    assert est["rate_grid"] == pytest.approx(900.0 / (px * 146 * 9 / 1e6))
    assert est["best"] is None and est["measured_k"] is None
    native = dcz.credit_estimate(LIGHT, px, 146, [dict(nine, input_mpx=px * 146 * 5.25 / 1e6)])
    assert native["best"] == pytest.approx(est["credits_native"]) and native["measured_k"] == 5.25
    assert "native" in native["convention"]
    grid = dcz.credit_estimate(LIGHT, px, 146, [dict(nine, input_mpx=px * 146 * 9.2 / 1e6)])
    assert grid["best"] == pytest.approx(300.0)
    same = dcz.credit_estimate(LIGHT, px, 73, [dict(nine, credits=330.0, bands=list(LIGHT))])
    assert same["credits_lo"] == pytest.approx(165.0) == same["credits_hi"]
    assert dcz.credit_estimate(LIGHT, px, 73, [])["credits_lo"] is None
    assert dcz.band_weight(dcz.BANDS) == 9 and dcz.band_weight(dcz.BANDS, native=True) == 5.25
    assert dcz.band_weight(LIGHT, native=True) == 2.25
    assert dcz.subset_bytes_per_px_date(LIGHT) == pytest.approx(2.3)
    assert dcz._input_mpx({"input_pixel": {"value": 1234.5, "unit": "mega-pixel"}}) == 1234.5
    assert dcz._input_mpx({"input_pixel": {"value": 2e6, "unit": "pixel"}}) == 2.0
    assert dcz._input_mpx({"cpu": {"value": 1}}) is None


def test_the_cost_calibration_uses_the_measured_jobs(tmp_path, wkt):
    """The site's measured jobs (``<year>.job.json`` with the job's costs,
    beside a complete store): of the same bands under the store root first,
    else the nine bands under --root, else QUOTED_COSTS."""
    cubes, light = tmp_path / "cubes", tmp_path / "cubes_light"
    sub = dcz.Request(site="escalda", year=2022, start="2022-01-01", end="2023-01-01", bbox={},
                      root=light, bands=LIGHT, main_root=cubes)
    entries, kind = dcz.cost_calibration(sub, [light, cubes])
    assert kind.startswith("quoted") and [e["credits"] for e in entries] == [890.0]
    est = dcz.credit_estimate(LIGHT, 1536 * 2176, 145, entries)
    assert 290 < est["credits_lo"] < 300 and 375 < est["credits_hi"] < 385
    t, y, x, data = synth(2023, n=3, H=8, W=16)
    for root, bands, credits in ((cubes, dcz.BANDS, 890.0), (light, LIGHT, 300.0)):
        req = make_request(root, year=2023, site="escalda")
        place(req, [("a.zarr", write_raw_v2(tmp_path / "src" / str(credits) / "a.zarr", t, y, x,
                                            {b: data[b] for b in bands}, wkt), "x")])
        req.bands = tuple(bands)
        normalise_and_write(req)
        usage = {"input_pixel": {"value": 8 * 16 * 3 * 5.25 / 1e6, "unit": "mega-pixel"}}
        zc.write_json({"ledger": {}, "job": {"id": f"j-{credits:g}", "costs": credits,
                                             "usage": usage}},
                      zc.sidecar_path("escalda", 2023, "job.json", root))
        if root == cubes:
            got = dcz.measured_costs("escalda", cubes)
            assert got == [{"site": "escalda", "year": 2023, "credits": 890.0, "dates": 3,
                            "delivered_shape": [8, 16], "bands": list(dcz.BANDS),
                            "input_mpx": pytest.approx(8 * 16 * 3 * 5.25 / 1e6), "job": "j-890",
                            "source": str(zc.sidecar_path("escalda", 2023, "job.json", cubes))}]
            entries, kind = dcz.cost_calibration(sub, [light, cubes])
            assert kind == "measured on the site's jobs of other bands" and len(entries) == 1
            est = dcz.credit_estimate(LIGHT, 8 * 16, 3, entries)
            assert est["best"] == pytest.approx(890.0 * 2.25 / 5.25)       # the input says native
    entries, kind = dcz.cost_calibration(sub, [light, cubes])
    assert kind == "measured on the site's jobs of the same bands"
    assert [e["credits"] for e in entries] == [300.0]
    assert dcz.credit_estimate(LIGHT, 8 * 16, 6, entries)["credits_lo"] == pytest.approx(600.0)
    # the share measured at escalda (300 / 890 per pixel-date) is the best estimate elsewhere
    assert dcz.measured_share(LIGHT, light, cubes) == [("escalda", round(300.0 / 890.0, 4))]
    ems = dcz.Request(site="ems", year=2022, start="2022-01-01", end="2023-01-01", bbox={},
                      root=light, bands=LIGHT, main_root=cubes)
    entries, kind = dcz.cost_calibration(ems, [light, cubes])
    assert kind.startswith("quoted") and [e["credits"] for e in entries] == [3240.0]
    est = dcz.credit_estimate(LIGHT, 3840 * 3968, 145, entries, share=0.337)
    assert est["best"] == pytest.approx(3240.0 * 0.337) and est["share"] == 0.337
    assert est["credits_lo"] < est["best"] < est["credits_hi"]


def test_the_dry_run_of_a_subset(tmp_path, monkeypatch, capsys, wkt):
    import socket

    def no_net(*a, **k):
        raise AssertionError("network used in a dry run")

    monkeypatch.setattr(socket, "create_connection", no_net)
    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", FRAME64, wkt, n=2),
              reference_period=("2024-01-01", "2025-01-01"))
    monkeypatch.chdir(tmp_path)
    cubes, light = tmp_path / "cubes", tmp_path / "cubes_light"
    assert dcz.main(["villaviciosa", "--years", "2023-2024", "--bands", "b03,scl,b08", "--root",
                     str(cubes), "--store-root", str(light), "--offset-px", "72,72",
                     "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"band subset B03 B08 SCL: stores under {light}; the nine-band stores under {cubes}" \
        in out
    assert "== villaviciosa 2023 B03+B08+SCL: window [2023-01-01, 2024-01-01)" in out
    assert "zarr_cube_B03_B08_SCL_villaviciosa_2024_" in out
    graph = out.split("process graph:")[1].split("outputs:")[0]
    assert '"B03"' in graph and '"SCL"' in graph and '"B8A"' not in graph
    assert "registration: against the reference netCDF ref.nc, which covers the year" in out
    assert ("registration: no reference for [2023-01-01, 2024-01-01) (ref.nc covers "
            "[2024-01-01, 2025-01-01)): --offset-px 72,72") in out
    assert "NO registered store of villaviciosa with this extent and grid" in out
    assert (f"light-path gate: NOT passed yet by any store of B03 B08 SCL under {light}: this "
            f"year would be REFUSED before any job") in out
    assert "input: 64 x 64 px (the frame: no delivered shape recorded) x ~2 dates x 3 bands" in out
    assert "(estimate 2.30 B per pixel-date for B03 B08 SCL)" in out
    assert out.count("credits: ~") == 2 and "calibration no job of villaviciosa measured" in out
    assert "TOTAL credits of the band-subset stores to make: ~" in out and "for 2 site-year(s)" in out
    assert "a band subset KEEPS its raw download (zip + extracted" in out and "removed after it" not in out
    assert not light.exists() and not cubes.exists()
    # the guard, before anything: the store root may not be the nine-band root
    assert dcz.main(["villaviciosa", "--years", "2023", "--bands", "B03,B08,SCL", "--root",
                     str(cubes), "--store-root", str(cubes), "--dry-run"]) == 2
    assert "REFUSED, nothing started" in capsys.readouterr().out
    # the nine bands' dry run says nothing of credits
    assert dcz.main(["villaviciosa", "--years", "2024", "--products", "main", "--root",
                     str(cubes), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "credits" not in out and "band subset" not in out
    assert "(estimate 6.0 B per pixel-date)" in out


# ─────────────────────────────────────────────────────────────────────────────
#  2026-10-26: the extraction gate, a gate failing on the dates alone, the
#  nine-band root guard, --expect-account
# ─────────────────────────────────────────────────────────────────────────────

def test_the_extraction_gate_decides_the_purchase(tmp_path):
    """At the converter's sites a year without its own nine-band store waits
    for the site's 2023 extraction gate: passed, or failed on extraction
    values only AND accepted by the user there; never a screening failure;
    --without-extraction-gate overrides (recorded)."""
    gdir = tmp_path / "gates"
    ems = dcz.Request(site="ems", year=2022, start="2022-01-01", end="2023-01-01", bbox={},
                      root=tmp_path / "cubes_light", bands=LIGHT, main_root=tmp_path / "cubes")
    ctx = dcz.Context(dcz.parse_args(["ems", "--bands", "B03,B08,SCL", "--extraction-gate-dir",
                                      str(gdir)]))
    write_extraction_gate(gdir, "ems", ok=False)
    why = dcz.extraction_gate_problem(ems, ctx)
    assert "FAILED on extraction values only (B03 / B08 vs the legacy cube: 0.74 %" in why
    assert "gate ems --accept-gate-differences" in why
    write_extraction_gate(gdir, "ems", ok=False, accepted=True)
    assert dcz.extraction_gate_problem(ems, ctx) is None
    assert dcz.extraction_gate_status("ems", ctx)[1]["accepted_by"].endswith("--accept-gate-differences")
    write_extraction_gate(gdir, "ems", ok=False, accepted=True, screening=False)
    assert "no flag accepts it" in dcz.extraction_gate_problem(ems, ctx)
    ctx.without_extraction_gate = True
    assert dcz.extraction_gate_problem(ems, ctx) is None


def test_a_gate_failing_on_the_dates_alone_lifts_the_wait(tmp_path, wkt):
    """Every common date bitwise with the same layout, one date more in the
    nine bands: the gate FAILS (the store is decided by the rule) but the
    years without a nine-band store may be bought (recorded); a gate failing
    on the delivered shape does not lift it and says why."""
    cubes, light = tmp_path / "cubes", tmp_path / "cubes_light"
    t, y, x, data = synth(2023, n=3, H=8, W=8)
    lay = {"delivered_offset_px": json.dumps([5, 1]), "delivered_shape_px": json.dumps([8, 8])}
    nine = make_request(cubes, year=2023, site="escalda")
    place(nine, [("a.zarr", write_raw_v2(tmp_path / "s9" / "a.zarr", t, y, x, data, wkt), "x")])
    normalise_and_write(nine)
    dcz.mark_complete(nine.store, lay)
    sub = make_request(light, year=2023, site="escalda")
    sub.bands, sub.main_root = LIGHT, cubes
    place(sub, [("a.zarr", write_raw_v2(tmp_path / "sl" / "a.zarr", t[:2], y, x,
                                        {b: data[b][:2] for b in LIGHT}, wkt), "x")])
    normalise_and_write(sub)
    dcz.mark_complete(sub.store, lay)
    msgs = []
    gate = dcz.nine_band_gate(sub, log=msgs.append)
    assert not gate["passed"] and gate["passed_on_common_dates"] and gate["layout_differs"] == []
    assert "LIGHT-PATH GATE FAILED on the dates only" in "\n".join(msgs)
    assert dcz.decide(sub, {"nine_band_store": gate})[1] != dcz.NINE_BAND_HOW      # not accepted
    assert dcz.light_gate_passed(light, LIGHT) == []
    assert dcz.light_gate_common_dates(light, LIGHT) == [str(sub.store)]
    old = dcz.Request(site="villaviciosa", year=2021, start="2021-01-01", end="2022-01-01",
                      bbox={}, root=light, bands=LIGHT, main_root=cubes)
    assert dcz.subset_gate_problem(old, dcz.Context()) is None
    dcz.mark_complete(sub.store, {"delivered_shape_px": json.dumps([8, 9])})
    gate = dcz.nine_band_gate(sub, log=msgs.append)
    assert gate["layout_differs"] == ["delivered_shape_px"] and not gate["passed_on_common_dates"]
    assert "do NOT hold for deliveries of these bands" in msgs[-1]
    assert dcz.light_gate_common_dates(light, LIGHT) == []
    assert "FAILED on the layout" in dcz.subset_gate_problem(old, dcz.Context())
    # the nine bands never go into a root of band subsets (refused before anything)
    assert dcz.nine_band_root_problem(cubes) is None
    assert "is a band-subset store" in dcz.nine_band_root_problem(light)


def test_expect_account_stops_before_any_job(tmp_path, monkeypatch, capsys, wkt):
    """--expect-account: the account is described on the connection before
    anything uses it; another account (or none named) stops the run with no
    job; without the option it is never asked."""
    import pyintertidal as pit

    me = {"user_id": "u-1", "name": "Fer", "info": {"oidc_userinfo": {"email": "fer@example.org",
                                                                       "sub": "u-1"}}}

    class FakeConn:
        def __init__(self, desc):
            self.desc, self.asked = desc, 0

        def describe_account(self):
            self.asked += 1
            if isinstance(self.desc, Exception):
                raise self.desc
            return self.desc

    assert dcz.account_email(me) == "fer@example.org"
    assert dcz.account_email({"user_id": "x@y.org", "links": [{"href": "mailto:a@b"}]}) == "x@y.org"
    assert dcz.account_email({"user_id": "u-1", "name": "Fer"}) is None
    conn = FakeConn(me)
    ctx = dcz.Context(dcz.parse_args(["escalda", "--expect-account", "FER@example.org"]),
                      connect=lambda: conn)
    assert ctx.connection() is conn and ctx.account["email"] == "fer@example.org"
    assert ctx.connection() is conn and conn.asked == 1                 # once per run
    for desc, msg in ((me, "authenticated as fer@example.org, another account"),
                      ({"user_id": "u-2"}, "names no e-mail"),
                      (RuntimeError("401"), "could not be described")):
        bad = dcz.Context(dcz.parse_args(["escalda", "--expect-account", "jorge@example.org"]),
                          connect=lambda d=desc: FakeConn(d))
        with pytest.raises(dcz.AccountMismatch, match=msg):
            bad.connection()
        assert bad._conn is None                                          # never usable
    plain = FakeConn(me)
    assert dcz.Context(dcz.parse_args(["escalda"]), connect=lambda: plain).connection() is plain
    assert plain.asked == 0                                               # the default: not asked
    # main(): refused before any request is run (no job created or looked up)
    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", FRAME64, wkt))
    monkeypatch.setattr(pit.scenes, "connect", lambda interactive=False: FakeConn(me))
    monkeypatch.setattr(pit.net, "use_system_certificates", lambda: None)
    ran = []
    monkeypatch.setattr(dcz, "run_request",
                        lambda ctx, req, log: ran.append(req.tag) or {"status": "done"})
    argv = ["villaviciosa", "--years", "2023", "--products", "main", "--root",
            str(tmp_path / "cubes")]
    assert dcz.main(argv + ["--expect-account", "jorge@example.org"]) == 2 and ran == []
    out = capsys.readouterr().out
    assert "openEO account: fer@example.org (user_id u-1); expected jorge@example.org" in out
    assert ("REFUSED, nothing started (no job, no download): --expect-account jorge@example.org"
            in out)
    assert dcz.main(argv + ["--expect-account", "fer@example.org"]) == 0
    assert ran == ["villaviciosa 2023"]
    assert "openEO account: fer@example.org (--expect-account)" in capsys.readouterr().out
