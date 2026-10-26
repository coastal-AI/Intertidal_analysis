"""Tests of experiments.benchmark.sensing_time and fetch.sensing (the OPTIONAL sensing-time diagnostic, spec §1.3, §10.3; user decision 2026-10-14).

They pin: the vendored NREL SPA (byte-identical to pvlib 0.13.0, the NREL
reference case of Reda & Andreas 2004 to < 0.001 deg, and the split into
time-only and place-only parts equal to pvlib's numpy path); the inversion
(a round trip time -> angles -> time, synthetic tile-mean angles generated
from a known instant recovered to < 2 s, quantised angles and their
uncertainty, azimuth wrapping through north); the circular-orbit ground
track and the physical window (the reviewer's numbers of §1.3); the
along-track check (consistent tiles pass, one reference time per datatake
fails); the per-date criteria, the group SD rule, the fallbacks and the
verdicts; double overpasses (unresolved without the openEO angle bands,
resolved with them); the STAC client on a recorded CDSE response (paging,
cache, resume, retry, refusal of another box), with no network; and the
recorded Villaviciosa 2025-06 items end to end, read back by
``inputs.read_sensing_times``.

Run:  python -X utf8 -m pytest tests/test_benchmark_sensing_time.py -q
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import inputs  # noqa: E402
from experiments.benchmark import sensing_time as st  # noqa: E402
from experiments.benchmark import vendor  # noqa: E402
from experiments.benchmark.fetch import sensing as fs  # noqa: E402
from experiments.benchmark.vendor import spa  # noqa: E402

RECORDED = Path(ROOT) / "tests" / "data" / "stac_villaviciosa_2025-06.json"
#: Villaviciosa frame centroid (legacy 10 m grid)
SITE = (-5.40504, 43.51003)
#: tile extents (EPSG, proj:bbox) of the recorded items, and the tile south of 30TUP
T30TUP = (32630, [300000.0, 4790220.0, 409800.0, 4900020.0])
T30TUN = (32630, [300000.0, 4690220.0, 409800.0, 4800020.0])
T29TQJ = (32629, [699960.0, 4790220.0, 809760.0, 4900020.0])


def _t(s):
    return st.unix(s)


def _features():
    with open(RECORDED, encoding="utf-8") as f:
        return json.load(f)["features"]


# ─────────────────────────────────────────────────────────────────────────────
#  synthetic items
# ─────────────────────────────────────────────────────────────────────────────

def _footprint(epsg, pbb):
    tf = st._transformer(epsg, 4326)
    xs = [pbb[0], pbb[2], pbb[2], pbb[0], pbb[0]]
    ys = [pbb[3], pbb[3], pbb[1], pbb[1], pbb[3]]
    lon, lat = tf.transform(xs, ys)
    ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
    return {"type": "Polygon", "coordinates": [ring]}, [min(lon), min(lat), max(lon), max(lat)]


def make_item(t_true, tile_ext, start, tile="30TUP", rel_orbit=37, platform="sentinel-2a",
              datatake="DT1", dec=None, el_shift=0.0, centre_only=False):
    """A STAC item record whose mean sun angles are SPA's at ``t_true`` over
    the tile's angle grid (or at its centre)."""
    epsg, pbb = tile_ext
    if centre_only:
        lon, lat = st._transformer(epsg, 4326).transform((pbb[0] + pbb[2]) / 2, (pbb[1] + pbb[3]) / 2)
        lon, lat = np.array([lon]), np.array([lat])
    else:
        lon, lat = st.tile_grid_lonlat(epsg, pbb[0], pbb[3])
    z, a = st.sun_angles(np.array([t_true]), lon, lat)
    zm, am = st.mean_angles(z, a)
    el, az = 90.0 - float(zm[0]) + el_shift, float(am[0])
    if dec is not None:
        el, az = round(el, dec), round(az, dec)
    c = st._transformer(epsg, 4326).transform((pbb[0] + pbb[2]) / 2, (pbb[1] + pbb[3]) / 2)
    geom, bbox = _footprint(epsg, pbb)
    return {"id": f"S2X_MSIL2A_{tile}_{datatake}_{int(t_true)}", "tile": tile,
            "platform": platform, "relative_orbit": rel_orbit, "datatake_id": datatake,
            "datetime": st.iso_utc(start), "sun_azimuth": az, "sun_elevation": el,
            "view_azimuth": 100.0 if rel_orbit == 137 else 280.0, "view_incidence": 5.0,
            "proj_epsg": epsg, "proj_bbox": list(pbb), "tile_centre_lonlat": [c[0], c[1]],
            "angle_decimals": {"sun_azimuth": dec, "sun_elevation": dec},
            "geometry": geom, "bbox": bbox}


def pass_items(t_site, start, tiles, rel_orbit=37, platform="sentinel-2a", datatake="DT1",
               site=SITE, single_reference=False, **kw):
    """Items of one pass whose per-tile instants are consistent with the site
    instant ``t_site`` (or all equal to it: one reference time)."""
    tr = st.track_model(site[1])
    out = []
    for name, ext in tiles:
        epsg, pbb = ext
        c = st._transformer(epsg, 4326).transform((pbb[0] + pbb[2]) / 2, (pbb[1] + pbb[3]) / 2)
        s, _ = st.along_across_km(c[0], c[1], site[0], site[1], tr["normal_deg"])
        tk = t_site if single_reference else t_site + float(s) / tr["normal_speed_km_s"]
        out.append(make_item(tk, ext, start, tile=name, rel_orbit=rel_orbit, platform=platform,
                             datatake=datatake, **kw))
    return out


TILES2 = [("30TUP", T30TUP), ("29TQJ", T29TQJ)]


def synthetic_run(n=8, perturb=None, base_tod="11:29:30", start_lead_s=-480.0, jitter_s=1.0,
                  site=SITE, systematics=False):
    """``n`` dates of one group (R037, sentinel-2a, 2025), every 10 days."""
    items, truth = {}, {}
    rng = np.random.default_rng(20261014)
    for k in range(n):
        d = (pd.Timestamp("2025-03-02") + pd.Timedelta(days=10 * k)).strftime("%Y-%m-%d")
        t = _t(f"{d}T{base_tod}Z") + float(rng.uniform(-jitter_s, jitter_s))
        kw = (perturb or {}).get(d, {})
        t = t + kw.pop("shift_s", 0.0)
        start = t + kw.pop("start_lead_s", start_lead_s)
        items[d] = pass_items(t, start, TILES2, datatake=f"DT{k}", site=site, **kw)
        truth[d] = t
    doc = st.sensing_times(items, sorted(items), site, systematics=systematics)
    return doc, truth


# ─────────────────────────────────────────────────────────────────────────────
#  the vendored ephemeris
# ─────────────────────────────────────────────────────────────────────────────

def test_vendored_spa_is_pristine_and_licensed():
    p = Path(ROOT) / "experiments" / "benchmark" / "vendor" / "spa.py"
    sha = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert sha == vendor.SPA_SHA256 == st.spa_file_sha256()
    lic = (p.parent / "LICENSE-pvlib.txt").read_text(encoding="utf-8")
    assert "BSD 3-Clause" in lic and "Sandia National Laboratories" in lic
    assert not spa.USE_NUMBA


def test_spa_nrel_reference_case():
    """Reda & Andreas (2004), table A4.1: 2003-10-17 12:30:30 LST (UTC-7),
    39.742476 N, 105.1786 W, 1830.14 m, 820 hPa, 11 C, delta-T 67 s:
    topocentric zenith 50.11162 deg, azimuth 194.34024 deg, JD 2452930.312847."""
    t = pd.Timestamp("2003-10-17 12:30:30", tz="Etc/GMT+7").tz_convert("UTC").value / 1e9
    assert abs(spa.julian_day(t) - 2452930.312847) < 1e-6
    z, a = st.sun_angles(np.array([t]), -105.1786, 39.742476, elev=1830.14, refraction=True,
                         pressure=820, temp=11, delta_t=67)
    assert abs(z[0, 0] - 50.11162) < 1e-3 and abs(a[0, 0] - 194.34024) < 1e-3
    assert abs(z[0, 0] - 50.11162) < 1e-5 and abs(a[0, 0] - 194.34024) < 1e-5
    o = spa.solar_position_numpy(np.array([t]), 39.742476, -105.1786, 1830.14, 820, 11, 67,
                                 0.5667, 1)
    assert o[0][0] == z[0, 0] and o[4][0] == a[0, 0]


def test_split_spa_equals_pvlib_numpy_path():
    rng = np.random.default_rng(7)
    ts = 1.5e9 + rng.uniform(0, 3e8, 40)
    lon = rng.uniform(-180, 180, 40)
    lat = rng.uniform(-80, 80, 40)
    o = spa.solar_position_numpy(ts, lat, lon, 0.0, 1013.25, 12.0, 69.0, 0.5667, 1)
    for i in range(40):
        z, a = st.sun_angles(ts[i:i + 1], lon[i], lat[i], delta_t=69.0)
        assert abs(z[0, 0] - o[1][i]) < 1e-9          # theta0: no refraction
        assert abs(st._wrap180(a[0, 0] - o[4][i])) < 1e-9
    # a (T, P) evaluation equals the point by point one
    z, a = st.sun_angles(ts[:3], lon[:5], lat[:5], delta_t=69.0)
    for i in range(3):
        for j in range(5):
            z1, a1 = st.sun_angles(ts[i:i + 1], lon[j], lat[j], delta_t=69.0)
            assert z[i, j] == z1[0, 0] and a[i, j] == a1[0, 0]


# ─────────────────────────────────────────────────────────────────────────────
#  the inversion
# ─────────────────────────────────────────────────────────────────────────────

def test_round_trip_time_angles_time():
    rng = np.random.default_rng(11)
    for _ in range(6):
        t_true = _t("2023-01-01T00:00:00Z") + rng.uniform(0, 3 * 365 * 86400)
        lat = float(rng.uniform(35, 60))
        lon = float(rng.uniform(-10, 10))
        # move t_true to ~11 UTC of its day, when the sites are imaged
        t_true = t_true - t_true % 86400 + 11 * 3600 + rng.uniform(-900, 900)
        z, a = st.sun_angles(np.array([t_true]), lon, lat)
        inv = st.invert_time(a[0, 0], 90 - z[0, 0], [lon], [lat], t_true - 400)
        assert abs(inv["t"] - t_true) < 0.05
        assert inv["residual_deg"] < 1e-5 and not inv["edge"]


@pytest.mark.parametrize("tile_ext", [T30TUP, T29TQJ])
def test_synthetic_tile_angles_recover_the_instant(tile_ext):
    t_true = _t("2025-06-10T11:29:20.4Z")
    rec = make_item(t_true, tile_ext, t_true - 470)
    r = st.invert_item(rec)
    assert r["status"] == "ok"
    assert abs(r["t_tile"] - t_true) < 2.0
    assert abs(r["t_tile"] - t_true) < 0.05
    assert r["residual_deg"] < 1e-5 and r["sigma_s"] < 0.05
    # the centre model is biased by the mean-of-angles nonlinearity only (< 1 s);
    # the cell-centre registration shifts ~7 s (2.5 km east and south)
    assert abs(r["model_bias_s"]) < 1.0
    assert 3.0 < abs(r["registration_s"]) < 12.0
    # the centre model on centre angles is exact too
    rc = st.invert_item(make_item(t_true, tile_ext, t_true - 470, centre_only=True),
                        model="centre", systematics=False)
    assert abs(rc["t_tile"] - t_true) < 0.05


def test_quantised_angles_and_their_uncertainty():
    t_true = _t("2025-06-10T11:29:20Z")
    errs, sig = [], []
    for k, dec in enumerate((3, 2, 1)):
        rec = make_item(t_true + 3.3 * k, T30TUP, t_true - 470, dec=dec)
        r = st.invert_item(rec, systematics=False)
        errs.append(abs(r["t_tile"] - (t_true + 3.3 * k)))
        sig.append(r["sigma_quant_s"])
        # the error is within the bound of the quantisation (half a step each way)
        q = 10.0 ** -dec
        assert errs[-1] <= 1.5 * q / r["rate_deg_s"] + 0.05
    assert errs[0] < 2.0
    assert sig[0] < sig[1] < sig[2]
    assert 2.0 < sig[2] / sig[1] < 20.0


def test_azimuth_wrapping_through_north():
    # southern hemisphere near local noon: the sun crosses north (azimuth 0/360)
    lon, lat = 3.0, -45.0
    t_noon = None
    ts = _t("2025-06-21T11:00:00Z") + np.arange(0, 7200, 5.0)
    z, a = st.sun_angles(ts, lon, lat)
    d = st._wrap180(a[:, 0])
    k = int(np.argmin(np.abs(d)))
    t_noon = ts[k]
    for off in (-40.0, 0.0, 40.0):
        t_true = t_noon + off
        z, a = st.sun_angles(np.array([t_true]), lon, lat)
        az = a[0, 0]
        for delivered in (az, az - 360.0 if az > 180 else az + 360.0):
            inv = st.invert_time(delivered, 90 - z[0, 0], [lon], [lat], t_true - 600)
            assert abs(inv["t"] - t_true) < 0.05
    # a grid straddling north averages to the right mean
    zz = np.array([[30.0, 30.0]])
    aa = np.array([[359.0, 1.0]])
    assert abs(st._wrap180(st.mean_angles(zz, aa)[1][0])) < 1e-12
    assert st.separation_deg(30.0, 359.9, 30.0, 0.1) < 0.11


# ─────────────────────────────────────────────────────────────────────────────
#  geometry
# ─────────────────────────────────────────────────────────────────────────────

def test_ground_track_and_the_physical_window():
    g = st.ground_track(43.51)
    assert 6.6 < g["speed_km_s"] < 6.8
    assert 193.0 < g["heading_deg"] < 196.5          # south-south-west
    assert 190.0 < g["heading_inertial_deg"] < g["heading_deg"]
    tr = st.track_model(43.51)
    assert tr["normal_deg"] == g["heading_inertial_deg"]
    assert tr["normal_speed_km_s"] < g["speed_km_s"]
    assert st.track_model(43.51, "ground")["normal_deg"] == g["heading_deg"]
    with pytest.raises(ValueError):
        st.track_model(43.51, "yaw")
    # spec §1.3: the reviewer's circular-orbit windows
    for (lon, lat), (lo, hi) in {(-5.405, 43.51): ("11:17:30", "11:31:54"),
                                 (7.0, 53.35): ("10:40:30", "10:58:00")}.items():
        w = st.physical_window(lon, lat, margin_min=0)
        assert abs(w["lo_nominal_s"] - _t(f"2025-01-01T{lo}Z") % 86400) < 30
        assert abs(w["hi_nominal_s"] - _t(f"2025-01-01T{hi}Z") % 86400) < 30
        w2 = st.physical_window(lon, lat)
        assert w2["lo_s"] == w["lo_s"] - 120 and w2["hi_s"] == w["hi_s"] + 120
    # the local plane round trip
    e, n = st.enu_km(-4.8, 43.75, *SITE)
    lo2, la2 = st.lonlat_from_enu(e, n, *SITE)
    assert abs(lo2 + 4.8) < 1e-9 and abs(la2 - 43.75) < 1e-9


def test_along_track_consistency_and_single_reference_time():
    t_site = _t("2025-06-10T11:29:30Z")
    tiles = [("30TUP", T30TUP), ("30TUN", T30TUN), ("29TQJ", T29TQJ)]
    tr = st.track_model(SITE[1])
    win = st.physical_window(*SITE)
    good = pass_items(t_site, t_site - 480, tiles)
    e, res = st.site_date(good, SITE, tr, win, systematics=False)
    assert e["status"] == "ok" and e["along_track"] == "pass"
    assert abs(st.unix(e["instant"]) - t_site) < 0.5
    chk = st.along_track_check(res, tr["normal_speed_km_s"])
    assert chk["status"] == "pass" and len(chk["pairs"]) == 3
    assert all(abs(p["resid_s"]) < 0.2 for p in chk["pairs"])
    # 30TUN lies ~100 km further along track: ~15 s later than 30TUP
    tup = next(r for r in res if r["tile"] == "30TUP")
    tun = next(r for r in res if r["tile"] == "30TUN")
    assert 12.0 < tun["t_tile"] - tup["t_tile"] < 18.0
    # one reference time for every tile of the datatake: detected, and a failure
    bad = pass_items(t_site, t_site - 480, tiles, single_reference=True)
    e2, res2 = st.site_date(bad, SITE, tr, win, systematics=False)
    assert st.along_track_check(res2, tr["normal_speed_km_s"])["status"] == "single_reference_time"
    assert e2["status"] == "check_failed" and not e2["checks"]["along_track"]


# ─────────────────────────────────────────────────────────────────────────────
#  criteria, groups, fallbacks, verdicts
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_run_passes_and_recovers_the_instants():
    doc, truth = synthetic_run()
    assert doc["verdict"] == "pass", doc["verdict_detail"]
    for d, t in truth.items():
        e = doc["dates"][d]
        assert e["status"] == "ok"
        assert abs(st.unix(e["instant"]) - t) < 0.5
        assert 7.5 < e["minus_start_min"] < 8.5
    g = doc["groups"]["R037/sentinel-2a/2025"]
    assert g["ok"] and g["sd_s"] < 2.0 and g["n_passing"] == 8


def test_residual_failure_falls_back_to_the_group_median():
    d0 = "2025-03-22"
    doc, truth = synthetic_run(n=10, perturb={d0: {"el_shift": 0.3}})
    e = doc["dates"][d0]
    assert e["checks"]["residual"] is False
    assert e["status"] == "fallback_group_median" and "fallback_group_median" in e["flags"]
    g = doc["groups"]["R037/sentinel-2a/2025"]
    assert abs(st.unix(e["instant"]) % 86400 - g["median_tod_s"]) < 1e-3
    assert doc["verdict"] == "fail"                    # 9/10 < 95 %
    assert any("ok share" in r for r in doc["verdict_detail"]["reasons"])


def test_group_spread_fails_the_site():
    rng = np.random.default_rng(3)
    shifts = {(pd.Timestamp("2025-03-02") + pd.Timedelta(days=10 * k)).strftime("%Y-%m-%d"):
              {"shift_s": float(rng.choice([-45.0, 45.0]))} for k in range(8)}
    doc, _ = synthetic_run(perturb=shifts)
    g = doc["groups"]["R037/sentinel-2a/2025"]
    # every kept date lies within 0.5 min of the median, so the spread shows up as
    # outliers: the removed share fails the group
    assert g["ok"] is False and g["sd_all_s"] > 30 and g["removed_share"] > 0.05
    assert doc["verdict"] == "fail"
    assert any("group" in r for r in doc["verdict_detail"]["reasons"])
    # one date far from its group: an outlier, falls back, the group still passes
    doc2, _ = synthetic_run(n=10, perturb={"2025-04-01": {"shift_s": 90.0}})
    e = doc2["dates"]["2025-04-01"]
    assert "group_outlier" in e["flags"] and e["status"] == "fallback_group_median"
    assert doc2["groups"]["R037/sentinel-2a/2025"]["outliers"] == ["2025-04-01"]


def test_instant_before_the_datatake_start_fails():
    doc, _ = synthetic_run(n=4, perturb={"2025-03-12": {"start_lead_s": 120.0}})
    e = doc["dates"]["2025-03-12"]
    assert e["checks"]["after_start"] is False and e["status"] == "fallback_group_median"
    assert e["minus_start_min"] < -1.9


def test_outside_the_physical_window_fails_every_date():
    doc, _ = synthetic_run(n=4, base_tod="11:41:00")
    assert all(not e["checks"]["window"] for e in doc["dates"].values())
    assert all(e["status"] == "failed" and e["instant"] is None for e in doc["dates"].values())
    assert doc["verdict"] == "fail"


def test_dates_without_items_and_double_overpasses():
    d1, d2 = "2025-06-04", "2025-06-05"
    ta = _t(f"{d1}T11:29:30Z")
    tb = _t(f"{d1}T11:19:30Z")
    items = {d1: (pass_items(ta, ta - 480, TILES2, rel_orbit=37, datatake="A")
                  + pass_items(tb, tb - 780, TILES2, rel_orbit=137, platform="sentinel-2b",
                               datatake="B"))}
    doc = st.sensing_times(items, [d1, d2], SITE, systematics=False)
    e = doc["dates"][d1]
    assert e["status"] == "double_unresolved" and e["instant"] is None
    assert "double_overpass" in e["flags"] and len(e["passes"]) == 2
    assert doc["dates"][d2]["status"] == "no_item"
    assert doc["verdict_detail"]["counts"]["double_unresolved"] == 1
    assert doc["verdict_detail"]["dates_without_items"] == [d2]
    # the openEO angle bands (whole degrees) at the site identify pass B
    z, a = st.sun_angles(np.array([tb]), *SITE)
    ang = pd.DataFrame({"sunZenithAngles": [round(z[0, 0])], "sunAzimuthAngles": [round(a[0, 0])],
                        "viewZenithMean": [5.0], "viewAzimuthMean": [100.0]},
                       index=pd.DatetimeIndex([d1]))
    doc2 = st.sensing_times(items, [d1], SITE, angles=ang, systematics=False)
    e2 = doc2["dates"][d1]
    assert e2["status"] == "ok" and "double_resolved_by_angles" in e2["flags"]
    assert e2["relative_orbit"] == 137 and abs(st.unix(e2["instant"]) - tb) < 0.5
    assert abs(e2["openeo_angles_delta_deg"]["azimuth"]) <= 0.5
    # a value half way between the passes stays unresolved
    za, aa = st.sun_angles(np.array([ta]), *SITE)
    mid = pd.DataFrame({"sunZenithAngles": [(z[0, 0] + za[0, 0]) / 2],
                        "sunAzimuthAngles": [(a[0, 0] + aa[0, 0]) / 2]},
                       index=pd.DatetimeIndex([d1]))
    e3 = st.sensing_times(items, [d1], SITE, angles=mid, systematics=False)["dates"][d1]
    assert e3["status"] == "double_unresolved"
    # int32-min no-data in the bands is not a value
    nod = pd.DataFrame({"sunZenithAngles": [-2147483648.0], "sunAzimuthAngles": [-2147483648.0]},
                       index=pd.DatetimeIndex([d1]))
    e4 = st.sensing_times(items, [d1], SITE, angles=nod, systematics=False)["dates"][d1]
    assert e4["status"] == "double_unresolved"


# ─────────────────────────────────────────────────────────────────────────────
#  the STAC client (recorded response, no network)
# ─────────────────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise IOError(f"HTTP {self.status}")

    def json(self):
        return self.data


class FakeSession:
    """Serves the recorded features in two pages: a POST ``next`` link with
    ``merge`` (token), then the last page. ``fail_first`` raises on the
    first call; ``forever`` always returns a ``next`` link."""

    def __init__(self, features, fail_first=0, forever=False, get_next=False):
        self.features = features
        self.calls = []
        self.fail_first = fail_first
        self.forever = forever
        self.get_next = get_next

    def _page(self, token):
        k = 12
        if self.forever:
            return {"features": [], "links": [{"rel": "next", "method": "POST", "merge": True,
                                               "body": {"token": "again"}, "href": fs.STAC_SEARCH}]}
        if token is None:
            nxt = ({"rel": "next", "href": fs.STAC_SEARCH + "?token=p2"} if self.get_next else
                   {"rel": "next", "method": "POST", "merge": True, "body": {"token": "p2"},
                    "href": fs.STAC_SEARCH})
            return {"features": self.features[:k], "links": [{"rel": "self"}, nxt]}
        return {"features": self.features[k:], "links": [{"rel": "self"}]}

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, dict(json or {})))
        if self.fail_first:
            self.fail_first -= 1
            raise ConnectionError("recorded outage")
        return FakeResponse(self._page((json or {}).get("token")))

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None))
        return FakeResponse(self._page("p2" if "token=" in url else None))

    def close(self):
        pass


class NoNetwork:
    def post(self, *a, **k):
        raise AssertionError("the cache should have answered")

    get = post


def test_item_record_from_a_recorded_feature():
    f = _features()[0]
    r = fs.item_record(f)
    assert r["tile"] == "30TUP" and r["relative_orbit"] == 37
    assert r["datatake_id"] == "GS2A_20250614T112131_052115_N05.11"
    assert r["datetime"] == "2025-06-14T11:21:31.024Z"
    assert r["proj_epsg"] == 32630 and r["proj_bbox"] == T30TUP[1]
    assert r["tile_centre_xy"] == [354900.0, 4845120.0]
    lon, lat = r["tile_centre_lonlat"]
    assert abs(lon + 4.80209) < 1e-4 and abs(lat - 43.74492) < 1e-4
    assert r["angle_decimals"]["sun_azimuth"] >= 10
    assert fs.decimals(149.18) == 2 and fs.decimals(150.0) == 1 and fs.decimals(None) is None
    assert fs.decimals(1.5e-5) == 6
    assert fs.year_chunks("2024-06-01", "2026-01-01") == [("2024-06-01", "2025-01-01"),
                                                         ("2025-01-01", "2026-01-01")]
    b = fs.query_body([-5.5, 43.4, -5.3, 43.6], "2025-01-01", "2026-01-01")
    assert b["datetime"] == "2025-01-01T00:00:00Z/2025-12-31T23:59:59.999Z"


def test_fetch_pages_follows_next_links_and_refuses_an_endless_search():
    feats = _features()
    sess = FakeSession(feats)
    got, pages = fs.fetch_pages(sess, {"bbox": [0, 0, 1, 1]}, sleep=lambda s: None)
    assert pages == 2 and len(got) == len(feats)
    assert sess.calls[1][2]["token"] == "p2" and sess.calls[1][2]["bbox"] == [0, 0, 1, 1]
    sess = FakeSession(feats, get_next=True)
    got, pages = fs.fetch_pages(sess, {}, sleep=lambda s: None)
    assert pages == 2 and sess.calls[1][0] == "GET" and len(got) == len(feats)
    with pytest.raises(fs.FetchError, match="next"):
        fs.fetch_pages(FakeSession(feats, forever=True), {}, max_pages=3, sleep=lambda s: None)


def test_fetch_items_cache_resume_retry_and_box(tmp_path):
    feats = _features()
    path = tmp_path / "stac_items.json"
    box = [-5.46, 43.47, -5.35, 43.55]
    sleeps = []
    sess = FakeSession(feats, fail_first=1)
    doc = fs.fetch_items("villaviciosa", box, "2025-06-01", "2025-06-16", session=sess, path=path,
                         sleep=sleeps.append)
    assert len(doc["items"]) == 20 and fs.chunks_complete(doc, "2025-06-01", "2025-06-16")
    assert doc["chunks"]["2025-06-01/2025-06-16"]["pages"] == 2
    assert any(s >= fs.CHUNK_BACKOFF_S for s in sleeps)             # the outage was retried
    on_disk = fs.load_items(path)
    assert on_disk["items"].keys() == doc["items"].keys()
    # resumable: a complete chunk is never fetched again
    again = fs.fetch_items("villaviciosa", box, "2025-06-01", "2025-06-16", session=NoNetwork(),
                           path=path, sleep=lambda s: None)
    assert again["items"].keys() == doc["items"].keys()
    by = fs.items_by_date(again, "2025-06-01", "2025-06-16")
    assert sorted(by) == ["2025-06-01", "2025-06-02", "2025-06-04", "2025-06-07", "2025-06-09",
                          "2025-06-11", "2025-06-12", "2025-06-14"]
    assert len(by["2025-06-04"]) == 4                                # two passes, two tiles each
    with pytest.raises(fs.FetchError, match="box"):
        fs.fetch_items("villaviciosa", [-5.0, 43.0, -4.9, 43.1], "2025-06-01", "2025-06-16",
                       session=NoNetwork(), path=path)
    # an outage that does not end leaves the chunk incomplete and raises
    with pytest.raises(fs.FetchError, match="did not answer"):
        fs.fetch_items("villaviciosa", box, "2025-06-01", "2025-06-16",
                       session=FakeSession(feats, fail_first=5), path=tmp_path / "x.json",
                       retries=2, sleep=lambda s: None)
    assert fs.load_items(tmp_path / "x.json")["chunks"]["2025-06-01/2025-06-16"]["complete"] is False


# ─────────────────────────────────────────────────────────────────────────────
#  the recorded Villaviciosa query end to end
# ─────────────────────────────────────────────────────────────────────────────

def test_recorded_villaviciosa_june_2025_end_to_end(tmp_path):
    feats = _features()
    labels = {}
    for f in feats:
        d = f["properties"]["datetime"][:10]
        if f["properties"]["sat:relative_orbit"] == 37 or d not in labels:
            labels[d] = f["properties"]["datetime"][:19].replace("T", " ")
    lab = tmp_path / "overpass_times.json"
    lab.write_text(json.dumps(labels), encoding="utf-8")
    cubes = tmp_path / "cubes"
    (cubes / "villaviciosa").mkdir(parents=True)
    w, s = SITE[0] - 0.05, SITE[1] - 0.04
    doc = st.run_site("villaviciosa", "2025-06-01", "2025-06-16", root=cubes,
                      session=FakeSession(feats), labels=lab,
                      bbox=[w, s, 2 * SITE[0] - w, 2 * SITE[1] - s], use_angles=False,
                      folder=tmp_path / "out", copy=True)
    assert doc["verdict"] == "pass", doc["verdict_detail"]
    det = doc["verdict_detail"]
    assert det["counts"]["ok"] == 6 and det["counts"]["double_unresolved"] == 2
    assert det["along_track"]["testable"] == 10 and det["along_track"]["pass"] == 10
    for d, e in doc["dates"].items():
        if e["status"] != "ok":
            continue
        tday = st.unix(e["instant"]) % 86400
        if e["relative_orbit"] == 37:
            assert 7.9 < e["minus_start_min"] < 8.2
            assert abs(tday - st.unix("2025-01-01T11:29:25Z") % 86400) < 20
        else:
            assert 12.2 < e["minus_start_min"] < 13.0
            assert abs(tday - st.unix("2025-01-01T11:19:30Z") % 86400) < 20
        assert e["residual_max_deg"] < 0.002 and e["sigma_s"] < 1.0 and e["spread_s"] < 0.5
        assert e["correction_min"] == pytest.approx(e["minus_start_min"], abs=0.01)
    rep = json.loads((tmp_path / "out" / st.REPORT_NAME).read_text(encoding="utf-8"))
    assert rep["along_track_pairs"]["resid_s_max_abs"] < 0.5
    assert 6.0 < rep["along_track_pairs"]["implied_speed_km_s_median"] < 7.5
    # the contract with inputs.py, and the copy beside the stores
    for p in (tmp_path / "out" / st.SENSING_NAME, cubes / "villaviciosa" / st.SENSING_NAME):
        inst = inputs.read_sensing_times(p)
        inputs._check_instants(inst)
        assert pd.isna(inst["2025-06-04"]) and inst["2025-06-02"].hour == 11
    assert (tmp_path / "out" / st.DATES_CSV).is_file()
    assert (tmp_path / "out" / st.ITEMS_CSV).read_text(encoding="utf-8").count("\n") == 21
    # resumable: the second run reads the cache only
    doc2 = st.run_site("villaviciosa", "2025-06-01", "2025-06-16", root=cubes, fetch=False,
                       labels=lab, bbox=[w, s, 2 * SITE[0] - w, 2 * SITE[1] - s],
                       use_angles=False, systematics=False, folder=tmp_path / "out", copy=False)
    assert {d: e["instant"] for d, e in doc2["dates"].items()} == \
        {d: e["instant"] for d, e in doc["dates"].items()}


def test_diag_cli_is_optional_and_gate_is_only_an_alias(tmp_path, capsys):
    """User decision of 2026-10-14: 'diag time' is an optional diagnostic.
    It never copies its file beside the stores unless asked, its verdict
    sets no exit code, 'gate' only prints that it is the diagnostic, and a
    site that cannot run is reported (exit 1) instead of raising."""
    feats = _features()
    labels = {}
    for f in feats:
        d = f["properties"]["datetime"][:10]
        if f["properties"]["sat:relative_orbit"] == 37 or d not in labels:
            labels[d] = f["properties"]["datetime"][:19].replace("T", " ")
    lab = tmp_path / "overpass_times.json"
    lab.write_text(json.dumps(labels), encoding="utf-8")
    cubes = tmp_path / "cubes"
    (cubes / "villaviciosa").mkdir(parents=True)
    w, s = SITE[0] - 0.05, SITE[1] - 0.04
    box = [str(v) for v in (w, s, 2 * SITE[0] - w, 2 * SITE[1] - s)]
    st.run_site("villaviciosa", "2025-06-01", "2025-06-16", root=cubes,
                session=FakeSession(feats), labels=lab, bbox=[float(v) for v in box],
                use_angles=False, folder=tmp_path / "cli" / "villaviciosa")
    assert not (cubes / "villaviciosa" / st.SENSING_NAME).exists()      # no copy by default
    argv = ["--sites", "villaviciosa", "--start", "2025-06-01", "--end", "2025-06-16",
            "--cubes-root", str(cubes), "--out", str(tmp_path / "cli"), "--no-fetch",
            "--labels", str(lab), "--no-angles", "--bbox", *box]
    assert st.main(["diag", *argv]) == 0
    out = capsys.readouterr().out
    assert "diagnostic verdict pass" in out and "optional" in out
    assert st.main(["gate", *argv]) == 0
    assert st.GATE_ALIAS_NOTE in capsys.readouterr().out
    assert not (cubes / "villaviciosa" / st.SENSING_NAME).exists()
    argv[argv.index("--out") + 1] = str(tmp_path / "empty")             # no items cache
    assert st.main(["diag", *argv]) == 1
    assert "could not run" in capsys.readouterr().out
