# -*- coding: utf-8 -*-
"""Validation v2 — OPTIONAL DIAGNOSTIC: how far each scene's site sensing time is from its catalogue time (spec §1.3).

Status (user decision of 2026-10-14)
------------------------------------
This module is an OPTIONAL DIAGNOSTIC, not a gate. The scene instant of
every run (level sources LB/LT/LM/LG, observations, package, designs,
judges, truths) is the catalogue / product-name time, i.e. the datatake
start, exactly as in every production result, without warning or flag.
The user judged the solar-angle correction to be over-engineering for now:
the validation first answers whether the errors are large, and if they are,
whether they are noise and whether they come from the lag or from the
elevation estimation. This diagnostic is no factor, setting or row of any
design; no command requires it (CLI ``validation_v2 diag time``, with
``gate time`` kept only as an alias that says so), and its
``sensing_times.json`` is read only when a run is given ``--sensing-times
<file>`` explicitly. It is never copied beside the stores unless asked
(``--copy-to-cubes``).

Question
--------
At what UTC instant did Sentinel-2 image the site on each date of the
record? The product-name time (``pyintertidal.overpass``, the cubes'
``overpass_times.json``) is the DATATAKE START: minutes before the site was
imaged, by an amount that changes from datatake to datatake (spec §1.3).
The diagnostic measures that difference per site, relative orbit and
platform.

Source of the sun angles (user decision of 2026-10-13, replacing the openEO
angle bands of method A)
-------------------------------------------------------------------------
* The openEO angle bands (``cubes/<site>/<year>.angles.zarr``) are
  quantised to whole degrees, with int32-min as no-data: at ~0.3-0.6 deg of
  sun azimuth per minute near noon they cannot time a scene to better than
  several minutes. They are kept as a diagnostic only, and to tell which of
  two same-date passes a mosaic holds at the site (rule 5).
* The CDSE STAC items (:mod:`experiments.benchmark.fetch.sensing`): one per
  MGRS tile and datatake, with the tile's MEAN sun azimuth and elevation in
  float precision, the relative orbit, the datatake and the tile's exact
  extent. The items' ``datetime`` is the datatake start.

Pre-registered method (frozen in :data:`ORBIT`, :data:`GRID`,
:data:`SEARCH` and :data:`CRITERIA`)
----------------------------------------------------------------
1. **Tile instant.** For each item, the UTC instant t_k at which the NREL
   solar position algorithm (Reda & Andreas 2004; pvlib 0.13.0's
   implementation, vendored unchanged in ``experiments/benchmark/vendor``,
   BSD-3-Clause) reproduces the item's mean sun angles, modelled as the
   arithmetic mean over the tile's 23 x 23 angle grid (5 km nodes from the
   tile's upper-left corner in its UTM CRS; ``model="grid"``, the default)
   or at the tile centre (``model="centre"``). Search in
   [datatake start - 5 min, datatake start + 30 min]: a 10-s scan of the
   great-circle separation between the modelled and the delivered sun
   directions (so azimuth wrapping is implicit) on the tile centre, then
   the model's minimum refined on 1-s and 0.05-s grids (parabolic vertex of
   the squared separation). Topocentric angles WITHOUT refraction at
   sea level; delta-T from SPA's polynomial; UT1 = UTC (|UT1 - UTC| < 0.9 s,
   declared). A minimum on the edge of the window fails the item.
2. **Uncertainty per tile.** Random part sqrt(sigma_res^2 + sigma_q^2):
   sigma_res = the residual separation / the sun's angular rate (the part
   of the delivered direction no instant explains, taken as if it lay along
   the sun's path: conservative); sigma_q = the delivered decimals'
   quantisation (uniform, q / sqrt 12) projected on the path. Systematic
   parts, reported, not added: the grid registration (corner vs
   cell-centre nodes, 2.5 km), the mean-of-angles model (grid vs centre),
   and the tile reference time (the angles may be computed at one instant
   per tile, e.g. its first line: up to half a tile, ~8 s).
3. **Site instant.** t_site = t_k - s_k / v, where s_k is the coordinate
   (km, positive in the direction of motion) of tile k's centre relative to
   the site, in the site's local tangent plane, along the normal of the
   push-broom's iso-time lines, and v the speed of those lines along it.
   The descending track at the site's latitude comes from the circular
   reference orbit (:data:`ORBIT`, Earth rotation included: ground speed
   ~6.7 km/s, ground heading ~195 deg). :data:`ISO_TIME` "inertial": the
   detector line is fixed in a platform without yaw steering, so the
   iso-time lines are perpendicular to the INERTIAL heading (~192 deg) and
   sweep at the ground speed x cos(~3 deg); "ground" (perpendicular to the
   ground track) is the alternative. Chosen on geometric consistency only
   (the 10 tile pairs of the 2025-06 Villaviciosa query fit "inertial"
   better; no truth involved; the two differ by < 0.5 s at a site).
   Tiles of one pass whose footprint contains the site centroid are
   combined by inverse-variance weighting; their spread must stay within
   ``same_pass_tol_s``.
4. **Along-track check (built in).** For every pair of tiles of one
   datatake whose centres are at least ``along_min_sep_km`` apart along
   track, (t_i - t_j) - (s_i - s_j) / v must lie within max(3 sigma_ij,
   ``along_tol_s``). Tiles far apart that give the same instant mean the
   angles were computed at one reference time per datatake (the risk named
   in §1.3): status ``single_reference_time``, a failure.
5. **Same-date passes.** Items of a date are grouped into passes by
   (platform, relative orbit); split datatakes of one pass are combined.
   Two passes on one date (a double overpass; frequent from 2025 with three
   platforms) are distinct instants minutes apart. The pass the cube's
   mosaic holds at the site is chosen with the openEO angle bands at the
   site centroid when available: the pass whose predicted site sun
   direction is within ``double_match_max_deg`` of the bands' and better
   than the other by ``double_margin_deg``. Otherwise the date gets no
   instant (status ``double_unresolved``, :data:`DOUBLE_POLICY`).
6. **Criteria per date (§1.3).** Converged inside the window; the largest
   tile residual <= ``residual_max_deg``; the pass's tiles agree; the
   along-track check passes (or is untestable); the instant lies inside
   the physical window of the site (descending pass, LTDN 10:30 mean solar
   time, cross-track +-145 km, +- ``window_margin_min``); the instant is
   not earlier than the datatake start by more than 0.5 min.
7. **Groups (site, relative orbit, platform, year)** with at least
   ``group_min_dates`` passing dates: a date more than
   ``group_dev_max_min`` from the group median time of day fails
   (``group_outlier``); the group passes iff the SD of the remaining time
   of day is <= ``group_sd_max_min`` and the removed share is <= 5 %. A
   failing date takes its group's median time of day (status
   ``fallback_group_median``, flagged), or no instant (``failed``) if its
   group has too few passing dates.
8. **Site verdict (diagnostic).** ``pass`` iff >= 95 % of the dates with
   items (double overpasses left unresolved excluded, counted) pass
   without fallback, every group passes, and >= 90 % of the testable
   datatakes pass the along-track check; else ``fail``. The verdict
   describes the inversion's own consistency; it gates nothing.

Outputs (spec §13.1): ``results/validation_v2/sensing/<site>/`` holds
``stac_items.json`` (the fetch cache), ``sensing_times.json`` (format
:data:`SENSING_FORMAT`, read by ``inputs.read_sensing_times`` when its path
is given: ``{"dates": {date: {"instant": iso | null, ...}}, "verdict":
...}``), ``sensing_report.json`` and the tables ``sensing_times.csv`` (per
date) and ``sensing_items.csv`` (per STAC item). ``sensing_times.json`` is
copied to ``cubes/<site>/`` only with ``copy=True`` (``--copy-to-cubes``);
even there no run reads it unless given ``--sensing-times``.

The optional SGP4 cross-check of §1.3 (method B) is not implemented here
(the user made it optional; sgp4/skyfield are not installed).

Importing this module does nothing; only :func:`main` chdirs to the
repository root.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SENSING_FORMAT = "validation-v2-sensing-times/1"
REPORT_FORMAT = "validation-v2-sensing-report/1"
SENSING_NAME = "sensing_times.json"
REPORT_NAME = "sensing_report.json"
DATES_CSV = "sensing_times.csv"
ITEMS_CSV = "sensing_items.csv"
#: the union period of the stores (end exclusive)
PERIOD = ("2017-01-01", "2026-01-01")

#: the Sentinel-2 reference orbit (circular approximation) and WGS84
ORBIT = {
    "altitude_km": 786.0,          # mean altitude
    "inclination_deg": 98.62,
    "ltdn_h": 10.5,                # local mean solar time at the descending node
    "swath_half_km": 145.0,        # 290 km swath
    "mu_km3_s2": 398600.4418,
    "earth_a_km": 6378.137,
    "earth_f": 1.0 / 298.257223563,
    "omega_e_rad_s": 7.2921150e-5,
}
#: the granule angle grid (MTD_TL.xml: 23 x 23 nodes, 5 km, from the upper-left corner)
GRID = {"n": 23, "step_m": 5000.0, "registration_offset_m": 2500.0}
#: the inversion search
SEARCH = {"before_s": 300.0, "after_s": 1800.0, "scan_step_s": 10.0, "refine_step_s": 1.0,
          "polish_step_s": 0.05, "refine_half_s": 60.0, "sigma_floor_s": 0.1}
#: default angle model: "grid" (mean over the angle grid) or "centre"
MODEL = "grid"
#: iso-time lines of the push-broom: "inertial" (perpendicular to the inertial velocity: a
#: platform without yaw steering) or "ground" (perpendicular to the ground track)
ISO_TIME = "inertial"
#: the pre-registered criteria (spec §1.3, adapted to the STAC tile angles)
CRITERIA = {
    "residual_max_deg": 0.05,
    "window_margin_min": 2.0,
    "start_lead_min": -0.5,         # instant - datatake start >= -0.5 min
    "along_min_sep_km": 10.0,
    "along_tol_s": 2.0,
    "along_sigma_k": 3.0,
    "same_pass_tol_s": 5.0,
    "group_min_dates": 3,
    "group_dev_max_min": 0.5,
    "group_sd_max_min": 0.5,
    "group_removed_max_share": 0.05,
    "site_ok_share_min": 0.95,
    "along_pass_share_min": 0.90,
    # whole-degree bands: a sky error up to ~0.55 deg; two passes 10 min apart are ~2.3 deg
    "double_match_max_deg": 1.0,
    "double_margin_deg": 1.0,
}
DOUBLE_POLICY = ("a double overpass takes the pass the openEO angle bands identify at the site "
                 "centroid; unresolved -> no instant (the date leaves the record, declared)")
#: date statuses
STATUSES = ("ok", "fallback_group_median", "failed", "double_unresolved", "no_item")
#: the openEO angle bands
ANGLE_COLS = ("sunZenithAngles", "sunAzimuthAngles", "viewZenithMean", "viewAzimuthMean")


def log(msg):
    print(msg, flush=True)


class SensingError(RuntimeError):
    """The sensing-time diagnostic cannot run (inputs missing or inconsistent)."""


# ─────────────────────────────────────────────────────────────────────────────
#  time helpers
# ─────────────────────────────────────────────────────────────────────────────

def unix(value):
    """Seconds since 1970-01-01 UTC of an ISO string or timestamp (naive =
    UTC)."""
    import pandas as pd

    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.value / 1e9


def iso_utc(t):
    """``YYYY-MM-DDTHH:MM:SS.mmmZ`` of a unix time (milliseconds truncated)."""
    d = dt.datetime.fromtimestamp(float(t), tz=dt.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


def hms(sec_of_day):
    """``HH:MM:SS.s`` of seconds of day."""
    if sec_of_day is None or not np.isfinite(sec_of_day):
        return None
    s = float(sec_of_day) % 86400.0
    h, r = divmod(s, 3600.0)
    m, r = divmod(r, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{r:04.1f}"


def tod(t):
    """Seconds of day (UTC) of a unix time."""
    return float(t) % 86400.0


def _wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# ─────────────────────────────────────────────────────────────────────────────
#  the solar ephemeris (NREL SPA, vendored)
# ─────────────────────────────────────────────────────────────────────────────

def _spa():
    from experiments.benchmark.vendor import spa

    return spa


def spa_file_sha256():
    """sha256 of the vendored ``spa.py`` (CRLF normalised to LF)."""
    p = Path(__file__).resolve().parent / "vendor" / "spa.py"
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def delta_t_s(t_unix):
    """TT - UT1 (s) from SPA's polynomial at the mean instant's year and month."""
    d = dt.datetime.fromtimestamp(float(np.mean(t_unix)), tz=dt.timezone.utc)
    return float(_spa().calculate_deltat(d.year, d.month))


def _geocentric(t_unix, delta_t=None):
    """The time-only part of SPA: apparent sidereal time, geocentric right
    ascension and declination, Earth radius vector (arrays over t)."""
    spa = _spa()
    t = np.atleast_1d(np.asarray(t_unix, np.float64))
    if delta_t is None:
        delta_t = delta_t_s(t)
    jd = spa.julian_day(t)
    jde = spa.julian_ephemeris_day(jd, delta_t)
    jc = spa.julian_century(jd)
    jce = spa.julian_ephemeris_century(jde)
    jme = spa.julian_ephemeris_millennium(jce)
    R = spa.heliocentric_radius_vector(jme)
    L = spa.heliocentric_longitude(jme)
    B = spa.heliocentric_latitude(jme)
    theta = spa.geocentric_longitude(L)
    beta = spa.geocentric_latitude(B)
    x0 = spa.mean_elongation(jce)
    x1 = spa.mean_anomaly_sun(jce)
    x2 = spa.mean_anomaly_moon(jce)
    x3 = spa.moon_argument_latitude(jce)
    x4 = spa.moon_ascending_longitude(jce)
    nut = np.empty((2, t.size))
    spa.longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4, nut)
    eps0 = spa.mean_ecliptic_obliquity(jme)
    eps = spa.true_ecliptic_obliquity(eps0, nut[1])
    dtau = spa.aberration_correction(R)
    lamd = spa.apparent_sun_longitude(theta, nut[0], dtau)
    v0 = spa.mean_sidereal_time(jd, jc)
    v = spa.apparent_sidereal_time(v0, nut[0], eps)
    alpha = spa.geocentric_sun_right_ascension(lamd, eps, beta)
    delta = spa.geocentric_sun_declination(lamd, eps, beta)
    return {"v": v, "alpha": alpha, "delta": delta, "R": R}


def sun_angles(t_unix, lon, lat, elev=0.0, refraction=False, pressure=1013.25, temp=12.0,
               delta_t=None):
    """Topocentric sun zenith and azimuth (deg, azimuth clockwise from north).

    Parameters
    ----------
    t_unix : array (T,)
    lon, lat : array (P,)
        Degrees (WGS84).
    elev : float or array (P,)
        Metres.
    refraction : bool
        Add SPA's refraction (``pressure`` hPa, ``temp`` deg C); the
        Sentinel-2 granule angles are geometric, so the default is False.
    delta_t : float, optional
        TT - UT1 in s (default :func:`delta_t_s`).

    Returns
    -------
    zenith, azimuth : arrays (T, P)
        Identical to ``spa.solar_position_numpy`` point by point (SPA split
        into its time-only and place-only parts).
    """
    spa = _spa()
    g = _geocentric(t_unix, delta_t)
    v, alpha, delta, R = (g[k][:, None] for k in ("v", "alpha", "delta", "R"))
    lon = np.atleast_1d(np.asarray(lon, np.float64))[None, :]
    lat = np.atleast_1d(np.asarray(lat, np.float64))[None, :]
    elev = np.atleast_1d(np.asarray(elev, np.float64))[None, :]
    H = spa.local_hour_angle(v, lon, alpha)
    xi = spa.equatorial_horizontal_parallax(R)
    u = spa.uterm(lat)
    x = spa.xterm(u, lat, elev)
    y = spa.yterm(u, lat, elev)
    d_alpha = spa.parallax_sun_right_ascension(x, xi, H, delta)
    delta_p = spa.topocentric_sun_declination(delta, x, y, xi, d_alpha, H)
    H_p = spa.topocentric_local_hour_angle(H, d_alpha)
    e0 = spa.topocentric_elevation_angle_without_atmosphere(lat, delta_p, H_p)
    if refraction:
        e0 = spa.topocentric_elevation_angle(
            e0, spa.atmospheric_refraction_correction(pressure, temp, e0, 0.5667))
    zen = spa.topocentric_zenith_angle(e0)
    az = spa.topocentric_azimuth_angle(spa.topocentric_astronomers_azimuth(H_p, delta_p, lat))
    return zen, az


def mean_angles(zen, az):
    """Arithmetic means over the points (axis 1); the azimuth is averaged
    relative to the first point, so a grid straddling north averages right."""
    zen = np.asarray(zen, float)
    az = np.asarray(az, float)
    ref = az[:, :1]
    am = (ref[:, 0] + _wrap180(az - ref).mean(axis=1)) % 360.0
    return zen.mean(axis=1), am


def separation_deg(zen1, az1, zen2, az2):
    """Great-circle angle between two sky directions (haversine form)."""
    e1 = np.radians(90.0 - np.asarray(zen1, float))
    e2 = np.radians(90.0 - np.asarray(zen2, float))
    daz = np.radians(np.asarray(az1, float) - np.asarray(az2, float))
    h = np.sin((e1 - e2) / 2.0) ** 2 + np.cos(e1) * np.cos(e2) * np.sin(daz / 2.0) ** 2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


# ─────────────────────────────────────────────────────────────────────────────
#  geometry: tiles, the local plane, the ground track, the physical window
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=64)
def _transformer(epsg_from, epsg_to):
    from pyproj import Transformer

    return Transformer.from_crs(f"EPSG:{int(epsg_from)}", f"EPSG:{int(epsg_to)}", always_xy=True)


@functools.lru_cache(maxsize=256)
def tile_grid_lonlat(epsg, xmin, ymax, n=GRID["n"], step=GRID["step_m"], offset=0.0):
    """lon, lat (flat, n*n) of a tile's angle-grid nodes: ``xmin + offset +
    step j``, ``ymax - offset - step i`` in the tile's CRS (``offset`` 0 =
    nodes from the upper-left corner, the registration used; ``offset`` 2.5
    km = cell centres, the systematic alternative)."""
    k = np.arange(int(n), dtype=np.float64) * float(step)
    X, Y = np.meshgrid(float(xmin) + offset + k, float(ymax) - offset - k)
    lon, lat = _transformer(epsg, 4326).transform(X.ravel(), Y.ravel())
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    lon.flags.writeable = False
    lat.flags.writeable = False
    return lon, lat


def _radii(lat0):
    a = ORBIT["earth_a_km"]
    f = ORBIT["earth_f"]
    e2 = f * (2.0 - f)
    s = math.sin(math.radians(lat0))
    N = a / math.sqrt(1.0 - e2 * s * s)
    M = a * (1.0 - e2) / (1.0 - e2 * s * s) ** 1.5
    return N, M


def enu_km(lon, lat, lon0, lat0):
    """East, north (km) of points in the tangent plane at (lon0, lat0)
    (WGS84 radii of curvature at lat0; < 0.1 % error within 150 km)."""
    N, M = _radii(lat0)
    e = np.radians(_wrap180(np.asarray(lon, float) - lon0)) * N * math.cos(math.radians(lat0))
    n = np.radians(np.asarray(lat, float) - lat0) * M
    return e, n


def lonlat_from_enu(e, n, lon0, lat0):
    N, M = _radii(lat0)
    lon = lon0 + np.degrees(np.asarray(e, float) / (N * math.cos(math.radians(lat0))))
    lat = lat0 + np.degrees(np.asarray(n, float) / M)
    return lon, lat


def along_across_km(lon, lat, lon0, lat0, heading_deg):
    """Along-track (positive in the direction of motion) and across-track
    (positive to the right of motion) coordinates, km."""
    e, n = enu_km(lon, lat, lon0, lat0)
    h = math.radians(heading_deg)
    return e * math.sin(h) + n * math.cos(h), e * math.cos(h) - n * math.sin(h)


def ground_track(lat_deg, descending=True, orbit=ORBIT):
    """The sub-satellite track of the circular reference orbit at a latitude.

    Returns
    -------
    dict
        ``heading_deg`` (clockwise from north, of the motion relative to the
        rotating Earth), ``speed_km_s`` (of the sub-satellite point on the
        ellipsoid), ``tau_s`` (time from the point to the descending node,
        or from the ascending node to the point), ``dlon_deg`` (Earth-fixed
        longitude of the point minus that node's), ``u_deg`` (argument of
        latitude).
    """
    a_e, f = orbit["earth_a_km"], orbit["earth_f"]
    a = a_e + orbit["altitude_km"]
    inc = math.radians(orbit["inclination_deg"])
    n = math.sqrt(orbit["mu_km3_s2"] / a ** 3)
    w = orbit["omega_e_rad_s"]
    phi = math.radians(lat_deg)
    s = math.sin(phi) / math.sin(inc)
    if abs(s) > 1.0:
        raise SensingError(f"latitude {lat_deg} beyond the orbit's reach")
    u = math.pi - math.asin(s) if descending else math.asin(s)
    pos = np.array([math.cos(u), math.sin(u) * math.cos(inc), math.sin(u) * math.sin(inc)])
    dpos = np.array([-math.sin(u), math.cos(u) * math.cos(inc), math.cos(u) * math.sin(inc)])
    alpha = math.atan2(pos[1], pos[0])
    east = np.array([-math.sin(alpha), math.cos(alpha), 0.0])
    north = np.array([-math.sin(phi) * math.cos(alpha), -math.sin(phi) * math.sin(alpha),
                      math.cos(phi)])
    b = a_e * (1.0 - f)
    c, sn = math.cos(phi), math.sin(phi)
    r = math.sqrt(((a_e * a_e * c) ** 2 + (b * b * sn) ** 2) / ((a_e * c) ** 2 + (b * sn) ** 2))
    vEi = r * n * float(dpos @ east)
    vE = vEi - w * r * c
    vN = r * n * float(dpos @ north)
    if descending:
        # the node is at u = pi (inertial longitude pi); the point is passed tau earlier, while
        # the Earth had rotated w * tau less: Earth-fixed lon(point) - lon(node)
        tau = (math.pi - u) / n
        d_alpha = (alpha - math.pi + math.pi) % (2.0 * math.pi) - math.pi     # wrap(alpha - pi)
        dlon = math.degrees(d_alpha + w * tau)
    else:
        tau = u / n
        dlon = math.degrees(alpha) - math.degrees(w * tau)
    return {"heading_deg": math.degrees(math.atan2(vE, vN)) % 360.0,
            "speed_km_s": math.hypot(vE, vN), "tau_s": tau, "dlon_deg": dlon,
            "u_deg": math.degrees(u),
            "heading_inertial_deg": math.degrees(math.atan2(vEi, vN)) % 360.0}


def track_model(lat_deg, iso_time=None, orbit=ORBIT):
    """:func:`ground_track` plus the iso-time geometry used to move a tile
    instant to the site: ``normal_deg`` (the direction along which the
    imaging time grows) and ``normal_speed_km_s`` (the speed of the
    push-broom line along it). ``iso_time`` "inertial" (default
    :data:`ISO_TIME`): the line is fixed in a platform without yaw steering,
    perpendicular to the inertial velocity, and sweeps the ground at the
    ground-track velocity; "ground": perpendicular to the ground track."""
    iso_time = ISO_TIME if iso_time is None else iso_time
    g = ground_track(lat_deg, orbit=orbit)
    if iso_time == "ground":
        normal, vn = g["heading_deg"], g["speed_km_s"]
    elif iso_time == "inertial":
        normal = g["heading_inertial_deg"]
        vn = g["speed_km_s"] * math.cos(math.radians(g["heading_deg"] - normal))
    else:
        raise ValueError(f"iso_time must be 'inertial' or 'ground', not {iso_time!r}")
    g.update(iso_time=iso_time, normal_deg=normal, normal_speed_km_s=vn)
    return g


def physical_window(lon, lat, margin_min=None, orbit=ORBIT, n=61):
    """The UTC times of day (s) at which a descending Sentinel-2 pass can
    image the point: LTDN = ``orbit["ltdn_h"]`` mean solar time, the point
    anywhere in the swath (cross-track +- ``swath_half_km``), widened by
    ``margin_min`` (default ``CRITERIA["window_margin_min"]``).

    Returns
    -------
    dict
        ``lo_s``, ``hi_s`` (with the margin), ``lo_nominal_s``,
        ``hi_nominal_s``, ``nadir_s`` (the point on the track), ``margin_min``.
    """
    margin = CRITERIA["window_margin_min"] if margin_min is None else float(margin_min)
    h = math.radians(ground_track(lat, orbit=orbit)["heading_deg"])
    ts = []
    for x in np.linspace(-orbit["swath_half_km"], orbit["swath_half_km"], int(n)):
        e, nn = -x * math.cos(h), x * math.sin(h)
        lp, fp = lonlat_from_enu(e, nn, lon, lat)
        g = ground_track(float(fp), orbit=orbit)
        lon_node = float(lp) - g["dlon_deg"]
        utc_node = orbit["ltdn_h"] * 3600.0 - lon_node / 15.0 * 3600.0
        ts.append((utc_node - g["tau_s"]) % 86400.0)
    ts = np.array(ts)
    return {"lo_s": float(ts.min() - margin * 60), "hi_s": float(ts.max() + margin * 60),
            "lo_nominal_s": float(ts.min()), "hi_nominal_s": float(ts.max()),
            "nadir_s": float(ts[len(ts) // 2]), "margin_min": margin}


def covers(rec, lon, lat):
    """Whether the item's footprint contains the point (True without a
    footprint)."""
    g = rec.get("geometry")
    if not g:
        return True
    from shapely.geometry import Point, shape

    try:
        return bool(shape(g).covers(Point(float(lon), float(lat))))
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  the inversion
# ─────────────────────────────────────────────────────────────────────────────

def invert_time(sun_azimuth, sun_elevation, lon, lat, t0, before_s=SEARCH["before_s"],
                after_s=SEARCH["after_s"], scan_step_s=SEARCH["scan_step_s"],
                refine_step_s=SEARCH["refine_step_s"], polish_step_s=SEARCH["polish_step_s"],
                refraction=False, delta_t=None, t_guess=None,
                half_width_s=SEARCH["refine_half_s"]):
    """The instant whose mean sun direction over the points matches the
    delivered one (least squares on the sky, azimuth wrapping implicit).

    Parameters
    ----------
    sun_azimuth, sun_elevation : float
        Delivered mean angles (deg).
    lon, lat : arrays (P,)
        The points the delivered angles average (a tile's grid, or one point).
    t0 : float
        Unix time the window is anchored at (the datatake start).
    t_guess : float, optional
        Skip the scan and refine in ``t_guess +- half_width_s``.

    Notes
    -----
    The scan (``scan_step_s``) brackets the minimum; the squared separation
    is then evaluated on a ``refine_step_s`` grid over the bracket and on a
    ``polish_step_s`` grid around its best node, each in ONE vectorised SPA
    call, and the minimum is the vertex of the parabola through the best
    node and its neighbours (the squared separation of a point moving at a
    constant rate past a fixed one is quadratic in time).

    Returns
    -------
    dict
        ``t``, ``residual_deg``, ``dzen_deg``, ``daz_deg`` (model - delivered),
        ``rate_deg_s`` (the sun's angular rate on the sky), ``el_rate``,
        ``az_rate`` (deg/s), ``zen_model``, ``az_model``, ``edge`` (minimum
        on the window edge), ``nfev``.
    """
    lon = np.atleast_1d(np.asarray(lon, np.float64))
    lat = np.atleast_1d(np.asarray(lat, np.float64))
    zen_obs = 90.0 - float(sun_elevation)
    az_obs = float(sun_azimuth) % 360.0
    dT = delta_t_s(t0) if delta_t is None else float(delta_t)
    nfev = [0]

    def model(ts):
        z, a = sun_angles(ts, lon, lat, refraction=refraction, delta_t=dT)
        nfev[0] += len(np.atleast_1d(ts))
        return mean_angles(z, a)

    def sep(ts):
        zm, am = model(ts)
        return separation_deg(zm, am, zen_obs, az_obs)

    lo_w, hi_w = float(t0) - float(before_s), float(t0) + float(after_s)
    edge = False
    if t_guess is None:
        ts = np.arange(lo_w, hi_w + scan_step_s / 2.0, scan_step_s)
        s = sep(ts)
        k = int(np.argmin(s))
        edge = k == 0 or k == len(ts) - 1
        lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
    else:
        lo, hi = float(t_guess) - half_width_s, float(t_guess) + half_width_s
    t = _grid_vertex(sep, lo, hi, refine_step_s)
    t = _grid_vertex(sep, t - 10 * polish_step_s, t + 10 * polish_step_s, polish_step_s)
    zm, am = model(np.array([t - 1.0, t, t + 1.0]))
    el_rate = -(zm[2] - zm[0]) / 2.0
    az_rate = float(_wrap180(am[2] - am[0])) / 2.0
    cos_e = math.cos(math.radians(90.0 - zm[1]))
    rate = math.hypot(el_rate, az_rate * cos_e)
    if t_guess is None:
        edge = edge or t - lo_w < scan_step_s or hi_w - t < scan_step_s
    return {"t": t, "residual_deg": float(separation_deg(zm[1], am[1], zen_obs, az_obs)),
            "dzen_deg": float(zm[1] - zen_obs), "daz_deg": float(_wrap180(am[1] - az_obs)),
            "rate_deg_s": rate, "el_rate": el_rate, "az_rate": az_rate,
            "zen_model": float(zm[1]), "az_model": float(am[1]), "edge": bool(edge),
            "nfev": nfev[0], "delta_t_s": dT}


def _grid_vertex(sep, lo, hi, step):
    """The parabolic vertex of ``sep(t)**2`` around its best node on the
    grid ``lo, lo + step, ..., hi`` (offsets from ``lo`` keep the precision
    of the unix times)."""
    n = max(int(round((hi - lo) / step)), 2)
    off = np.arange(n + 1, dtype=np.float64) * step
    f = sep(lo + off) ** 2
    k = int(np.argmin(f))
    if 0 < k < n:
        den = f[k - 1] - 2.0 * f[k] + f[k + 1]
        if den > 0:
            return lo + off[k] + 0.5 * step * (f[k - 1] - f[k + 1]) / den
    return lo + off[k]


def time_sigma(inv, dec_az=None, dec_el=None):
    """``(sigma, sigma_res, sigma_q)`` in seconds (see the module docstring,
    rule 2). ``dec_*``: decimals the angles were delivered with (None = no
    quantisation term)."""
    w = max(inv["rate_deg_s"], 1e-9)
    s_res = inv["residual_deg"] / w
    q_el = 10.0 ** (-dec_el) if dec_el is not None else 0.0
    q_az = 10.0 ** (-dec_az) if dec_az is not None else 0.0
    cos_e = math.cos(math.radians(90.0 - inv["zen_model"]))
    var_q = (inv["el_rate"] ** 2 * q_el ** 2 + inv["az_rate"] ** 2 * cos_e ** 4 * q_az ** 2) / 12.0
    s_q = math.sqrt(var_q) / w ** 2
    return math.hypot(s_res, s_q), s_res, s_q


def item_points(rec, model=MODEL, offset=0.0):
    """``(lon, lat, centre_lonlat, centre_source)`` of an item for the
    angle model: the grid nodes (``grid``) or the tile centre (``centre``);
    without a tile extent, the footprint box centre (flagged)."""
    if rec.get("proj_epsg") and rec.get("proj_bbox"):
        xmin, _, _, ymax = rec["proj_bbox"]
        c = rec.get("tile_centre_lonlat")
        if c is None:
            x0, y0, x1, y1 = rec["proj_bbox"]
            c = list(_transformer(rec["proj_epsg"], 4326).transform((x0 + x1) / 2, (y0 + y1) / 2))
        if model == "grid":
            lon, lat = tile_grid_lonlat(int(rec["proj_epsg"]), float(xmin), float(ymax),
                                        offset=float(offset))
            return lon, lat, c, "proj_bbox"
        return np.array([c[0]]), np.array([c[1]]), c, "proj_bbox"
    bb = rec.get("bbox")
    if not bb:
        return None, None, None, None
    c = [(bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0]
    return np.array([c[0]]), np.array([c[1]]), c, "footprint_bbox"


def invert_item(rec, model=MODEL, refraction=False, systematics=True):
    """The tile instant of one STAC item record (rules 1-2).

    Returns
    -------
    dict
        The item's identity, ``t0`` (datatake start), ``t_tile``, ``sigma_s``,
        ``sigma_res_s``, ``sigma_quant_s``, ``residual_deg``, ``dzen_deg``,
        ``daz_deg``, ``rate_deg_s``, ``edge``, ``centre_lonlat``,
        ``centre_source``, and with ``systematics`` ``model_bias_s`` (grid -
        centre) and ``registration_s`` (cell-centre nodes - corner nodes);
        ``status`` "ok" or why not.
    """
    out = {k: rec.get(k) for k in ("id", "tile", "platform", "relative_orbit", "datatake_id",
                                   "datetime", "sun_azimuth", "sun_elevation", "view_azimuth",
                                   "view_incidence")}
    out.update(t0=None, t_tile=None, status="ok", model=model)
    if rec.get("datetime") is None:
        out["status"] = "no_datetime"
        return out
    out["t0"] = unix(rec["datetime"])
    if rec.get("sun_azimuth") is None or rec.get("sun_elevation") is None:
        out["status"] = "no_angles"
        return out
    lon, lat, c, src = item_points(rec, model)
    if lon is None:
        out["status"] = "no_extent"
        return out
    if src != "proj_bbox":
        model = "centre"
        out["model"] = "centre"
    dec = rec.get("angle_decimals") or {}
    az, el = rec["sun_azimuth"], rec["sun_elevation"]
    # the window scan runs on the tile centre (one point, cheap); the grid model is then
    # refined around it (the two differ by well under a second: model_bias_s)
    ic = invert_time(az, el, np.array([c[0]]), np.array([c[1]]), out["t0"], refraction=refraction)
    if model == "grid":
        inv = invert_time(az, el, lon, lat, out["t0"], refraction=refraction, t_guess=ic["t"],
                          half_width_s=10.0)
        inv["edge"] = ic["edge"]
    else:
        inv = ic
    sig, s_res, s_q = time_sigma(inv, dec.get("sun_azimuth"), dec.get("sun_elevation"))
    out.update(t_tile=inv["t"], sigma_s=sig, sigma_res_s=s_res, sigma_quant_s=s_q,
               residual_deg=inv["residual_deg"], dzen_deg=inv["dzen_deg"],
               daz_deg=inv["daz_deg"], rate_deg_s=inv["rate_deg_s"], edge=inv["edge"],
               centre_lonlat=[float(c[0]), float(c[1])], centre_source=src,
               model_bias_s=None, registration_s=None)
    if inv["edge"]:
        out["status"] = "edge"
    if model == "grid":
        out["model_bias_s"] = inv["t"] - ic["t"]
    if systematics and model == "grid":
        lo, la, _, _ = item_points(rec, "grid", offset=GRID["registration_offset_m"])
        ir = invert_time(az, el, lo, la, out["t0"], refraction=refraction, t_guess=inv["t"],
                         half_width_s=30.0)
        out["registration_s"] = ir["t"] - inv["t"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  per date: passes, the along-track check, the site instant
# ─────────────────────────────────────────────────────────────────────────────

def along_track_check(results, speed_km_s, crit=CRITERIA):
    """Rule 4 over the tiles of ONE datatake (results of :func:`invert_item`
    with ``along_km``).

    Returns
    -------
    dict
        ``status`` (untested | pass | fail | single_reference_time) and the
        ``pairs`` (tiles, ds_km, expected_s, observed_s, resid_s, tol_s, ok,
        single_reference, implied_speed_km_s).
    """
    rs = [r for r in results if r.get("t_tile") is not None and not r.get("edge")
          and r.get("along_km") is not None]
    pairs = []
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            a, b = rs[i], rs[j]
            ds = a["along_km"] - b["along_km"]
            if abs(ds) < crit["along_min_sep_km"]:
                continue
            exp = ds / speed_km_s
            obs = a["t_tile"] - b["t_tile"]
            sig = math.hypot(a["sigma_s"], b["sigma_s"])
            tol = max(crit["along_sigma_k"] * sig, crit["along_tol_s"])
            single = abs(obs) <= max(crit["along_sigma_k"] * sig, 1.0) and abs(exp) > tol
            pairs.append({"tiles": [a["tile"], b["tile"]], "ds_km": round(ds, 3),
                          "expected_s": round(exp, 3), "observed_s": round(obs, 3),
                          "resid_s": round(obs - exp, 3), "tol_s": round(tol, 3),
                          "ok": abs(obs - exp) <= tol, "single_reference": bool(single),
                          "implied_speed_km_s": (round(ds / obs, 3) if abs(obs) > 0.5 else None)})
    if not pairs:
        st = "untested"
    elif any(p["single_reference"] for p in pairs):
        st = "single_reference_time"
    elif all(p["ok"] for p in pairs):
        st = "pass"
    else:
        st = "fail"
    return {"status": st, "pairs": pairs}


def _pass_key(r):
    return (str(r.get("platform")), int(r["relative_orbit"]) if r.get("relative_orbit") is not None
            else -1)


def date_passes(results, site, track, crit=CRITERIA):
    """The passes of one date (rule 3, 5): per (platform, relative orbit),
    the site instant from the tiles covering the site, the along-track check
    per datatake, the datatake start."""
    passes = []
    for key in sorted({_pass_key(r) for r in results}):
        rs = [r for r in results if _pass_key(r) == key]
        good = [r for r in rs if r.get("t_site") is not None and not r.get("edge")]
        use = [r for r in good if r.get("covers_site")]
        flags = []
        if good and not use:
            use = good
            flags.append("site_outside_footprints")
        dts = {}
        for r in rs:
            dts.setdefault(str(r.get("datatake_id")), []).append(r)
        checks = {d: along_track_check(v, track["normal_speed_km_s"], crit)
                  for d, v in dts.items()}
        p = {"platform": key[0], "relative_orbit": key[1], "datatakes": sorted(dts),
             "tiles": [r["tile"] for r in use], "n_items": len(rs), "along": checks,
             "datatake_start": min(r["t0"] for r in rs if r.get("t0") is not None),
             "flags": flags, "t_site": None}
        if use:
            sig = np.array([max(r["sigma_s"], SEARCH["sigma_floor_s"]) for r in use])
            w = 1.0 / sig ** 2
            ts = np.array([r["t_site"] for r in use])
            t = float((w * ts).sum() / w.sum())
            spread = float(ts.max() - ts.min())
            p.update(t_site=t, spread_s=spread,
                     sigma_s=max(math.sqrt(1.0 / w.sum()), spread / 2.0),
                     residual_max_deg=max(r["residual_deg"] for r in use),
                     view_azimuth=use[0].get("view_azimuth"),
                     view_incidence=use[0].get("view_incidence"))
        p["edge"] = any(r.get("edge") for r in rs)
        passes.append(p)
    return passes


def _angles_value(row, col):
    if row is None:
        return None
    try:
        v = float(row[col])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(v) or v < -1e6 or v > 1e6:
        return None
    return v


def resolve_double(passes, site, angles_row, crit=CRITERIA):
    """Rule 5: the index of the pass the openEO angle bands identify at the
    site, or None; and the per-pass sky separations (deg)."""
    zen = _angles_value(angles_row, "sunZenithAngles")
    az = _angles_value(angles_row, "sunAzimuthAngles")
    cands = [i for i, p in enumerate(passes) if p.get("t_site") is not None]
    if zen is None or az is None or not cands:
        return None, {}
    seps = {}
    for i in cands:
        z, a = sun_angles(np.array([passes[i]["t_site"]]), site[0], site[1])
        seps[i] = float(separation_deg(z[0, 0], a[0, 0], zen, az))
    order = sorted(seps, key=seps.get)
    best = order[0]
    second = seps[order[1]] if len(order) > 1 else np.inf
    if seps[best] <= crit["double_match_max_deg"] and second - seps[best] >= crit["double_margin_deg"]:
        return best, seps
    return None, seps


def site_date(recs, site, track, window, crit=CRITERIA, label=None, angles_row=None,
              model=MODEL, refraction=False, systematics=True, results=None):
    """The instant of one date at the site (rules 1-6, before the groups).

    Parameters
    ----------
    recs : list of dict
        The date's STAC item records.
    site : (lon, lat)
    track : dict
        :func:`ground_track` at the site.
    window : dict
        :func:`physical_window` of the site.
    label : float, optional
        The cube's label (unix), reported.
    angles_row : mapping, optional
        The openEO angle bands at the site centroid on that date.
    results : list, optional
        Precomputed :func:`invert_item` results for ``recs``.

    Returns
    -------
    (entry, item_results)
    """
    lon0, lat0 = site
    res = results if results is not None else [invert_item(r, model, refraction, systematics)
                                               for r in recs]
    for r, rec in zip(res, recs):
        r["covers_site"] = covers(rec, lon0, lat0)
        if r.get("t_tile") is not None:
            s, x = along_across_km(r["centre_lonlat"][0], r["centre_lonlat"][1], lon0, lat0,
                                   track["normal_deg"])
            r["along_km"], r["across_km"] = float(s), float(x)
            r["t_site"] = r["t_tile"] - float(s) / track["normal_speed_km_s"]
        else:
            r["along_km"] = r["across_km"] = r["t_site"] = None
    e = {"instant": None, "status": "no_item", "label": iso_utc(label) if label else None,
         "flags": [], "checks": {}, "n_items": len(recs), "passes": []}
    if not recs:
        return e, res
    passes = date_passes(res, site, track, crit)
    e["passes"] = [_pass_summary(p) for p in passes]
    valid = [i for i, p in enumerate(passes) if p.get("t_site") is not None]
    if len(passes) > 1:
        e["flags"].append("double_overpass")
    if not valid:
        e["status"] = "check_failed"
        e["flags"].append("no_valid_tile")
        chosen = passes[0]
    elif len(passes) == 1:
        chosen = passes[valid[0]]
    else:
        k, seps = resolve_double(passes, site, angles_row, crit)
        e["double_separation_deg"] = {f"{passes[i]['platform']}/R{passes[i]['relative_orbit']:03d}":
                                      round(v, 3) for i, v in seps.items()}
        if k is None:
            e["status"] = "double_unresolved"
            e["platform"] = None
            e["relative_orbit"] = None
            return e, res
        chosen = passes[k]
        e["flags"].append("double_resolved_by_angles")
    e.update(platform=chosen["platform"], relative_orbit=chosen["relative_orbit"],
             datatake_id=chosen["datatakes"][0] if len(chosen["datatakes"]) == 1
             else chosen["datatakes"], tiles=chosen["tiles"],
             datatake_start=iso_utc(chosen["datatake_start"]))
    e["flags"] += chosen["flags"]
    if chosen.get("t_site") is None:
        return e, res
    t = chosen["t_site"]
    along = [c["status"] for c in chosen["along"].values()]
    tday = tod(t)
    checks = {
        "converged": not chosen["edge"],
        "residual": chosen["residual_max_deg"] <= crit["residual_max_deg"],
        "tiles_agree": chosen["spread_s"] <= crit["same_pass_tol_s"],
        "along_track": all(s in ("pass", "untested") for s in along),
        "window": window["lo_s"] <= tday <= window["hi_s"],
        "after_start": (t - chosen["datatake_start"]) / 60.0 >= crit["start_lead_min"],
    }
    z, a = sun_angles(np.array([t]), lon0, lat0)
    e.update(t=t, sigma_s=round(chosen["sigma_s"], 3), spread_s=round(chosen["spread_s"], 3),
             residual_max_deg=round(chosen["residual_max_deg"], 5),
             along_track=("single_reference_time" if "single_reference_time" in along else
                          "fail" if "fail" in along else "pass" if "pass" in along else
                          "untested"),
             sun_zenith=round(float(z[0, 0]), 5), sun_azimuth=round(float(a[0, 0]), 5),
             view_zenith=chosen.get("view_incidence"), view_azimuth=chosen.get("view_azimuth"),
             minus_start_min=round((t - chosen["datatake_start"]) / 60.0, 4),
             checks=checks)
    if label:
        e["correction_min"] = round((t - label) / 60.0, 4)
    if angles_row is not None:
        zs = _angles_value(angles_row, "sunZenithAngles")
        azs = _angles_value(angles_row, "sunAzimuthAngles")
        if zs is not None and azs is not None:
            e["openeo_angles_delta_deg"] = {"zenith": round(float(z[0, 0]) - zs, 3),
                                            "azimuth": round(float(_wrap180(a[0, 0] - azs)), 3)}
    e["status"] = "ok" if all(checks.values()) else "check_failed"
    e["instant"] = iso_utc(t) if e["status"] == "ok" else None
    return e, res


def _pass_summary(p):
    out = {k: p.get(k) for k in ("platform", "relative_orbit", "datatakes", "tiles", "n_items",
                                 "flags")}
    out["datatake_start"] = iso_utc(p["datatake_start"])
    out["instant"] = iso_utc(p["t_site"]) if p.get("t_site") is not None else None
    for k in ("spread_s", "sigma_s", "residual_max_deg"):
        if p.get(k) is not None:
            out[k] = round(float(p[k]), 5)
    out["along"] = {d: c["status"] for d, c in p["along"].items()}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  groups, fallbacks and the verdict
# ─────────────────────────────────────────────────────────────────────────────

def group_key(date, e):
    return f"R{int(e['relative_orbit']):03d}/{e['platform']}/{date[:4]}"


def apply_groups(dates, crit=CRITERIA):
    """Rule 7, in place on ``{date: entry}``; returns the group table."""
    groups = {}
    for d, e in dates.items():
        if e.get("t") is None or e.get("relative_orbit") is None:
            continue
        groups.setdefault(group_key(d, e), []).append(d)
    table = {}
    for g, ds in sorted(groups.items()):
        passing = [d for d in ds if dates[d]["status"] == "ok"]
        tods = np.array([tod(dates[d]["t"]) for d in passing])
        row = {"n_dates": len(ds), "n_passing": len(passing), "dates": ds}
        if len(passing) >= crit["group_min_dates"]:
            med = float(np.median(tods))
            dev = np.abs(tods - med)
            out = [d for d, v in zip(passing, dev) if v > crit["group_dev_max_min"] * 60]
            kept = np.array([t for d, t in zip(passing, tods) if d not in out])
            sd = float(np.std(kept, ddof=1)) if kept.size > 1 else 0.0
            removed = len(out) / max(len(passing), 1)
            row.update(median_tod_s=med, median_tod=hms(med), sd_s=round(sd, 3),
                       sd_all_s=round(float(np.std(tods, ddof=1)), 3) if tods.size > 1 else 0.0,
                       outliers=out, removed_share=round(removed, 4),
                       ok=(sd <= crit["group_sd_max_min"] * 60
                           and removed <= crit["group_removed_max_share"]))
            for d in out:
                dates[d]["status"] = "check_failed"
                dates[d]["flags"].append("group_outlier")
                dates[d]["instant"] = None
        else:
            row.update(median_tod_s=None, ok=None,
                       note=f"fewer than {crit['group_min_dates']} passing dates: untested")
        labels = [unix(dates[d]["label"]) for d in ds if dates[d].get("label")]
        if len(labels) > 1:
            row["label_sd_s"] = round(float(np.std([tod(x) for x in labels], ddof=1)), 3)
        starts = [unix(dates[d]["datatake_start"]) for d in ds if dates[d].get("datatake_start")]
        if len(starts) > 1:
            row["datatake_start_sd_s"] = round(float(np.std([tod(x) for x in starts], ddof=1)), 3)
        table[g] = row
    for d, e in dates.items():
        if e["status"] != "check_failed":
            continue
        g = table.get(group_key(d, e)) if e.get("relative_orbit") is not None else None
        if g and g.get("median_tod_s") is not None:
            day = unix(d + "T00:00:00Z")
            e["instant"] = iso_utc(day + g["median_tod_s"])
            e["status"] = "fallback_group_median"
            e["flags"].append("fallback_group_median")
        else:
            e["status"] = "failed"
            e["instant"] = None
    return table


def verdict(dates, groups, crit=CRITERIA):
    """Rule 8: ``(verdict, detail)``."""
    st = {s: 0 for s in STATUSES}
    for e in dates.values():
        st[e["status"]] = st.get(e["status"], 0) + 1
    with_items = sum(1 for e in dates.values() if e["status"] != "no_item")
    denom = with_items - st["double_unresolved"]
    ok_share = st["ok"] / denom if denom else 0.0
    along = {"testable": 0, "pass": 0, "fail": 0, "single_reference_time": 0}
    seen = set()
    for e in dates.values():
        for p in e.get("passes", []):
            for dtk, s in p.get("along", {}).items():
                if dtk in seen or s == "untested":
                    continue
                seen.add(dtk)
                along["testable"] += 1
                along[s if s in along else "fail"] += 1
    along["share"] = round(along["pass"] / along["testable"], 4) if along["testable"] else None
    bad_groups = [g for g, r in groups.items() if r.get("ok") is False]
    reasons, warnings = [], []
    if denom == 0:
        reasons.append("no date with STAC items")
    if ok_share < crit["site_ok_share_min"]:
        reasons.append(f"ok share {ok_share:.3f} < {crit['site_ok_share_min']}")
    if bad_groups:
        reasons.append(f"group SD or outliers fail: {bad_groups}")
    if along["testable"] and along["share"] < crit["along_pass_share_min"]:
        reasons.append(f"along-track check passes on {along['pass']}/{along['testable']} datatakes")
    if along["single_reference_time"]:
        reasons.append(f"{along['single_reference_time']} datatakes look like one reference "
                       "time per datatake")
    if not along["testable"]:
        warnings.append("along-track check untested (no datatake with two tiles >= "
                        f"{crit['along_min_sep_km']} km apart along track)")
    if st["double_unresolved"]:
        warnings.append(f"{st['double_unresolved']} double-overpass dates without an instant")
    untested = [g for g, r in groups.items() if r.get("ok") is None]
    if untested:
        warnings.append(f"groups with too few dates for the SD test: {untested}")
    detail = {"counts": st, "n_dates": len(dates), "n_with_items": with_items,
              "ok_share": round(ok_share, 4), "along_track": along,
              "groups_failing": bad_groups, "reasons": reasons, "warnings": warnings}
    return ("pass" if not reasons else "fail"), detail


# ─────────────────────────────────────────────────────────────────────────────
#  the site run
# ─────────────────────────────────────────────────────────────────────────────

def sensing_times(items, dates, site, labels=None, angles=None, model=MODEL, refraction=False,
                  systematics=True, crit=CRITERIA, meta=None, iso_time=None):
    """The sensing-time diagnostic for one site (pure: no I/O).

    Parameters
    ----------
    items : {date: [STAC item records]}
        ``fetch.sensing.items_by_date``.
    dates : list of str
        The cube's dates (each gets an entry; dates without items get
        ``no_item``). None = every date of ``items``.
    site : (lon, lat)
        The frame centroid.
    labels : {date: unix or ISO}, optional
        The cube's product-name labels (reported, and the correction).
    angles : pandas.DataFrame, optional
        ``zarr_cubes.angles_at`` at the site (index: dates), the diagnostic
        and the double-overpass resolver.
    meta : dict, optional
        Merged into the document (provenance).

    Returns
    -------
    dict
        The ``sensing_times.json`` document, plus ``"_items"`` (per-item
        results, for the tables; not written to the JSON).
    """
    lon0, lat0 = float(site[0]), float(site[1])
    track = track_model(lat0, iso_time)
    window = physical_window(lon0, lat0, crit["window_margin_min"])
    dates = sorted(items) if dates is None else sorted({str(d)[:10] for d in dates})
    labels = {str(k)[:10]: (unix(v) if not isinstance(v, (int, float)) else float(v))
              for k, v in (labels or {}).items() if v is not None}
    arows = {}
    if angles is not None and len(angles):
        for idx, row in angles.iterrows():
            arows[str(idx)[:10]] = row
    out, item_rows = {}, []
    for d in dates:
        recs = items.get(d, [])
        e, res = site_date(recs, (lon0, lat0), track, window, crit, label=labels.get(d),
                           angles_row=arows.get(d), model=model, refraction=refraction,
                           systematics=systematics)
        out[d] = e
        for r in res:
            item_rows.append(dict(r, date=d))
    groups = apply_groups(out, crit)
    v, detail = verdict(out, groups, crit)
    extra = sorted(set(items) - set(dates))
    detail["item_dates_not_in_cube"] = len(extra)
    detail["dates_without_items"] = [d for d, e in out.items() if e["status"] == "no_item"]
    detail["double_overpass_dates"] = [d for d, e in out.items()
                                       if "double_overpass" in e["flags"]]
    sysrows = [r for r in item_rows if r.get("registration_s") is not None]
    systematic = {
        "registration_s_median": (round(float(np.median([abs(r["registration_s"])
                                                         for r in sysrows])), 3)
                                  if sysrows else None),
        "model_bias_s_median": (round(float(np.median([abs(r["model_bias_s"])
                                                       for r in sysrows])), 3)
                                if sysrows else None),
        "tile_reference_time_s": round(0.5 * 109.8 / track["speed_km_s"], 2),
        "note": ("not added to sigma_s: the grid registration (corner vs cell-centre nodes), "
                 "the mean-of-angles model (grid vs centre) and the tile reference time (the "
                 "angles may refer to one instant per tile, up to half a tile from its centre)")}
    for e in out.values():
        e.pop("t", None)
    doc = {"format": SENSING_FORMAT, "site_lonlat": [lon0, lat0],
           "method": ("sun-angle inversion of the CDSE STAC per-tile mean sun angles with the "
                      "NREL SPA, moved to the site along track (user decision 2026-10-13)"),
           "model": model, "refraction": bool(refraction),
           "track": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in track.items()},
           "physical_window": {"lo": hms(window["lo_s"]), "hi": hms(window["hi_s"]),
                               "lo_nominal": hms(window["lo_nominal_s"]),
                               "hi_nominal": hms(window["hi_nominal_s"]),
                               "nadir": hms(window["nadir_s"]),
                               "margin_min": window["margin_min"]},
           "criteria": dict(crit), "orbit": dict(ORBIT), "grid": dict(GRID),
           "search": dict(SEARCH), "double_policy": DOUBLE_POLICY,
           "systematic": systematic,
           "spa": {"source": "experiments/benchmark/vendor/spa.py (pvlib 0.13.0, BSD-3-Clause)",
                   "sha256": spa_file_sha256()},
           "verdict": v, "verdict_detail": detail, "groups": groups, "dates": out}
    if meta:
        doc.update(meta)
    doc["_items"] = item_rows
    return doc


def report(doc):
    """The diagnostic report of one site: per relative orbit and per group, the
    correction (instant - label, instant - datatake start), the spread of
    labels and instants, the along-track check, residuals, uncertainties."""
    dates = doc["dates"]
    per_orbit = {}
    for d, e in dates.items():
        if e.get("relative_orbit") is None or not e.get("instant"):
            continue
        r = per_orbit.setdefault(f"R{int(e['relative_orbit']):03d}", {"dates": []})
        r["dates"].append(d)
    for ro, r in per_orbit.items():
        es = [dates[d] for d in r["dates"]]
        ok = [e for e in es if e["status"] == "ok"]
        inst = [tod(unix(e["instant"])) for e in ok]
        corr = [e["correction_min"] for e in ok if e.get("correction_min") is not None]
        lead = [e["minus_start_min"] for e in ok if e.get("minus_start_min") is not None]
        r.update(n=len(es), n_ok=len(ok),
                 instant_median=hms(float(np.median(inst))) if inst else None,
                 instant_sd_s=round(float(np.std(inst, ddof=1)), 3) if len(inst) > 1 else None,
                 correction_vs_label_median_min=(round(float(np.median(corr)), 3)
                                                 if corr else None),
                 correction_vs_label_range_min=([round(min(corr), 3), round(max(corr), 3)]
                                                if corr else None),
                 instant_minus_datatake_start_median_min=(round(float(np.median(lead)), 3)
                                                          if lead else None),
                 sigma_median_s=(round(float(np.median([e["sigma_s"] for e in ok])), 3)
                                 if ok else None),
                 residual_max_deg=(round(max(e["residual_max_deg"] for e in ok), 5)
                                   if ok else None))
        r.pop("dates")
    rows = doc.get("_items", [])
    good = [r for r in rows if r.get("t_tile") is not None]
    res = [r["residual_deg"] for r in good]
    return {"format": REPORT_FORMAT, "site": doc.get("site"), "period": doc.get("period"),
            "verdict": doc["verdict"], "verdict_detail": doc["verdict_detail"],
            "physical_window": doc["physical_window"], "track": doc["track"],
            "per_relative_orbit": per_orbit, "groups": doc["groups"],
            "items": {"n": len(rows), "n_inverted": len(good),
                      "residual_deg_median": round(float(np.median(res)), 5) if res else None,
                      "residual_deg_max": round(float(np.max(res)), 5) if res else None,
                      "sigma_s_median": (round(float(np.median([r["sigma_s"] for r in good])), 3)
                                         if good else None),
                      "statuses": _count(r.get("status") for r in rows)},
            "systematic": doc["systematic"], "along_track_pairs": _pairs(doc)}


def _count(it):
    out = {}
    for x in it:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def _pairs(doc):
    """Along-track pair statistics over the run (from the item rows)."""
    rows = doc.get("_items", [])
    by = {}
    for r in rows:
        if r.get("t_tile") is None or r.get("edge"):
            continue
        by.setdefault((r["date"], str(r.get("datatake_id"))), []).append(r)
    sp = doc["track"]["normal_speed_km_s"]
    res, imp, n = [], [], 0
    for rs in by.values():
        c = along_track_check(rs, sp)
        for p in c["pairs"]:
            n += 1
            res.append(p["resid_s"])
            if p["implied_speed_km_s"] is not None:
                imp.append(p["implied_speed_km_s"])
    return {"n_pairs": n,
            "resid_s_median_abs": round(float(np.median(np.abs(res))), 3) if res else None,
            "resid_s_max_abs": round(float(np.max(np.abs(res))), 3) if res else None,
            "implied_speed_km_s_median": round(float(np.median(imp)), 3) if imp else None,
            "model_speed_km_s": round(sp, 4)}


# ─────────────────────────────────────────────────────────────────────────────
#  inputs of the site run: geometry, dates, labels, angles
# ─────────────────────────────────────────────────────────────────────────────

def _grid_corners(grid):
    xs, ys = grid.xs, grid.ys
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 10.0
    dy = float(ys[0] - ys[1]) if len(ys) > 1 else 10.0
    x0, x1 = float(xs[0]) - dx / 2, float(xs[-1]) + dx / 2
    y0, y1 = float(ys[-1]) - dy / 2, float(ys[0]) + dy / 2
    from pyproj import CRS, Transformer

    tf = Transformer.from_crs(CRS.from_user_input(grid.crs_wkt), "EPSG:4326", always_xy=True)
    cx, cy = tf.transform([x0, x1, x1, x0], [y1, y1, y0, y0])
    lon, lat = tf.transform(float(np.mean(xs)), float(np.mean(ys)))
    return [float(lon), float(lat)], [[float(a), float(b)] for a, b in zip(cx, cy)]


def site_geometry(site, root=None, legacy=None, bbox=None):
    """The frame centroid and corners in lon/lat and the query box.

    Order: an explicit ``bbox``; the first Zarr store of the site; the
    legacy netCDF cube (``legacy`` or the downloader's reference); the
    downloader's site box.

    Returns
    -------
    dict
        ``centre`` [lon, lat], ``corners``, ``bbox`` [w, s, e, n], ``source``.
    """
    from experiments import zarr_cubes as zc

    if bbox is None:
        years = zc.list_years(site, root)
        grid, src = None, None
        if years:
            from experiments.benchmark import inputs

            ds = zc.open_store(zc.store_path(site, years[0], root), bands=["SCL"])
            try:
                grid, src = inputs.grid_of(ds), f"zarr {years[0]}"
            finally:
                ds.close()
        else:
            from experiments import download_cube_zarr as dcz

            p = Path(legacy) if legacy else ROOT / dcz.SITE_CONFIG[site]["reference"]
            p = p if p.is_absolute() else ROOT / p
            if p.is_file():
                import xarray as xr

                from experiments.benchmark import inputs

                with xr.open_dataset(p, mask_and_scale=False, decode_times=False) as ds:
                    grid, src = inputs.grid_of(ds), f"legacy {p.name}"
            else:
                bbox = dcz.site_bbox(site)
                src = "downloader site box"
        if grid is not None:
            centre, corners = _grid_corners(grid)
            lons = [c[0] for c in corners]
            lats = [c[1] for c in corners]
            return {"centre": centre, "corners": corners,
                    "bbox": [min(lons), min(lats), max(lons), max(lats)], "source": src}
    else:
        src = "explicit box"
    if isinstance(bbox, dict):
        bbox = [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
    w, s, e, n = (float(v) for v in bbox)
    return {"centre": [(w + e) / 2, (s + n) / 2],
            "corners": [[w, n], [e, n], [e, s], [w, s]], "bbox": [w, s, e, n], "source": src}


def site_inputs(site, root=None, start=PERIOD[0], end=PERIOD[1], labels=None):
    """The cube's dates and labels in ``[start, end)``.

    Dates: the Zarr stores' dates; else the stores' ``overpass_times.json``;
    else the pipeline's production labels (no store yet: declared). Labels:
    ``labels`` (a path) or the first of those files that exists.

    Returns
    -------
    (dates or None, {date: label}, dates_source, labels_source)
    """
    from experiments import zarr_cubes as zc

    dates, dsrc = None, None
    years = zc.list_years(site, root)
    if years:
        dates = []
        for y in years:
            ds = zc.open_store(zc.store_path(site, y, root), bands=["SCL"])
            try:
                dates += [str(d)[:10] for d in zc.dates(ds)]
            finally:
                ds.close()
        dsrc = "zarr stores"
    lab, lsrc = {}, None
    cand = [Path(labels)] if labels else []
    cand.append(zc.overpass_path(site, root))
    try:
        from experiments import download_cube_zarr as dcz

        po = dcz.SITE_CONFIG.get(site, {}).get("pipeline_overpass")
        if po:
            cand.append(ROOT / po)
    except Exception:
        pass
    for p in cand:
        p = p if p.is_absolute() else ROOT / p
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            lab = {str(k)[:10]: v for k, v in raw.items() if v}
            lsrc = str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
            break
    lab = {d: v for d, v in lab.items() if start <= d < end}
    if dates is None and lab:
        dates, dsrc = sorted(lab), f"labels ({lsrc}; no Zarr store)"
    if dates is not None:
        dates = sorted(d for d in set(dates) if start <= d < end)
    return dates, lab, dsrc, lsrc


def angles_table(site, centre, start, end, root=None):
    """``zarr_cubes.angles_at`` at the centroid, or None (no angles store)."""
    from experiments import zarr_cubes as zc

    if not zc.list_angle_years(site, root):
        return None
    try:
        df = zc.angles_at(site, centre[0], centre[1], start, end, root=root, missing="skip")
    except Exception as exc:
        log(f"[diag time] {site}: angle bands unreadable ({exc}); no diagnostic")
        return None
    df = df.copy()
    for c in df.columns:
        v = df[c].astype("float64")
        df[c] = v.where((v > -1e6) & (v < 1e6))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  outputs
# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return _clean(_jsonable(o))
    return o


def write_json(path, obj):
    """Atomic, LF, sorted keys, no NaN."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(_clean(obj), sort_keys=True, indent=1, default=_jsonable, allow_nan=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data + "\n")
    os.replace(tmp, path)


DATE_COLS = ("date", "status", "instant", "label", "datatake_start", "correction_min",
             "minus_start_min", "platform", "relative_orbit", "tiles", "sigma_s", "spread_s",
             "residual_max_deg", "along_track", "sun_zenith", "sun_azimuth", "view_zenith",
             "view_azimuth", "flags")
ITEM_COLS = ("date", "id", "tile", "platform", "relative_orbit", "datatake_id", "datetime",
             "status", "covers_site", "sun_azimuth", "sun_elevation", "t_tile_iso", "t_site_iso",
             "along_km", "across_km", "sigma_s", "sigma_res_s", "sigma_quant_s", "residual_deg",
             "dzen_deg", "daz_deg", "model_bias_s", "registration_s", "centre_source")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}" if abs(v) < 1e6 else f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return "" if v is None else str(v)


def write_tables(doc, folder):
    """``sensing_times.csv`` (per date) and ``sensing_items.csv`` (per item)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / DATES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(DATE_COLS)
        for d, e in doc["dates"].items():
            w.writerow([d] + [_fmt(e.get(c)) for c in DATE_COLS[1:]])
    with open(folder / ITEMS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(ITEM_COLS)
        for r in doc.get("_items", []):
            r = dict(r)
            r["t_tile_iso"] = iso_utc(r["t_tile"]) if r.get("t_tile") is not None else None
            r["t_site_iso"] = iso_utc(r["t_site"]) if r.get("t_site") is not None else None
            w.writerow([_fmt(r.get(c)) for c in ITEM_COLS])


def write_outputs(doc, folder, cubes_dir=None):
    """Write ``sensing_times.json``, the report and the tables into
    ``folder``; copy ``sensing_times.json`` into ``cubes_dir`` if that
    folder exists. Returns the written paths."""
    folder = Path(folder)
    body = {k: v for k, v in doc.items() if k != "_items"}
    p = folder / SENSING_NAME
    write_json(p, body)
    rp = folder / REPORT_NAME
    write_json(rp, report(doc))
    write_tables(doc, folder)
    out = {"sensing_times": p, "report": rp, "dates_csv": folder / DATES_CSV,
           "items_csv": folder / ITEMS_CSV}
    if cubes_dir is not None and Path(cubes_dir).is_dir():
        dst = Path(cubes_dir) / SENSING_NAME
        shutil.copyfile(p, dst)
        out["cubes_copy"] = dst
    return out


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_site(site, start=PERIOD[0], end=PERIOD[1], root=None, out_root=None, fetch=True,
             session=None, refresh=False, model=MODEL, labels=None, bbox=None, legacy=None,
             copy=False, use_angles=True, systematics=True, folder=None, iso_time=None):
    """Fetch (resumably) and run the sensing-time diagnostic for one site;
    write the outputs.

    Parameters
    ----------
    site : str
    start, end : str
        The period, end exclusive (default the union period).
    root : path, optional
        The cubes root (``zarr_cubes.cubes_root``).
    out_root : path, optional
        Default ``results/validation_v2/sensing``.
    fetch : bool
        Query the STAC (cached chunks are reused); False reads the cache.
    session : optional
        A requests-like session (tests).
    labels : path, optional
        A labels file (default: the stores' overpass file, else the
        pipeline's).
    bbox : (w, s, e, n), optional
        Override the frame (default :func:`site_geometry`).
    copy : bool
        Copy ``sensing_times.json`` to ``cubes/<site>/`` when that folder
        exists (default False: no run reads it unless given explicitly).
    folder : path, optional
        Output folder (default ``<out_root>/<site>``).

    Returns
    -------
    dict
        The document (with ``_items``) and ``paths``.
    """
    from experiments import zarr_cubes as zc
    from experiments.benchmark.fetch import sensing as fs

    t0 = time.time()
    geom = site_geometry(site, root, legacy=legacy, bbox=bbox)
    log(f"[diag time] {site}: centre {geom['centre'][0]:.5f}, {geom['centre'][1]:.5f} "
        f"({geom['source']}); query box {[round(v, 5) for v in geom['bbox']]}")
    folder = Path(folder) if folder else fs.sensing_dir(site, out_root)
    ipath = folder / fs.ITEMS_NAME
    if fetch:
        idoc = fs.fetch_items(site, geom["bbox"], start, end, session=session, refresh=refresh,
                              path=ipath)
    else:
        if not ipath.is_file():
            raise SensingError(f"{ipath} missing: run the fetch step first")
        idoc = fs.load_items(ipath)
    if not fs.chunks_complete(idoc, start, end):
        raise SensingError(f"{ipath}: the STAC chunks of {start}..{end} are not all complete")
    items = fs.items_by_date(idoc, start, end)
    dates, lab, dsrc, lsrc = site_inputs(site, root, start, end, labels)
    if dates is None:
        dates, dsrc = sorted(items), "STAC dates (no cube dates and no labels)"
    log(f"[diag time] {site}: {len(items)} STAC dates, {len(dates)} cube dates ({dsrc}); "
        f"labels: {lsrc}")
    angles = angles_table(site, geom["centre"], start, end, root) if use_angles else None
    meta = {"site": site, "period": [start, end], "geometry": geom, "dates_source": dsrc,
            "labels_source": lsrc, "angles_diagnostic": angles is not None,
            "items_cache": {"path": ipath.relative_to(ROOT).as_posix()
                            if str(ipath).startswith(str(ROOT)) else str(ipath),
                            "sha256": _sha_file(ipath), "n_items": len(idoc.get("items", {}))},
            "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    doc = sensing_times(items, dates, geom["centre"], labels=lab, angles=angles, model=model,
                        systematics=systematics, meta=meta, iso_time=iso_time)
    paths = write_outputs(doc, folder, zc.site_dir(site, root) if copy else None)
    doc["paths"] = {k: str(v) for k, v in paths.items()}
    det = doc["verdict_detail"]
    log(f"[diag time] {site}: verdict {doc['verdict'].upper()}; counts {det['counts']}; "
        f"along-track {det['along_track']}; {time.time() - t0:.1f} s")
    for r in det["reasons"]:
        log(f"[diag time] {site}:   reason: {r}")
    for w in det["warnings"]:
        log(f"[diag time] {site}:   warning: {w}")
    return doc


def print_table(doc, n=None):
    """Log one line per date: status, datatake start, site instant, the
    correction, sigma."""
    log(f"{'date':10} {'status':22} {'orbit':10} {'datatake start':>15} {'site instant':>15} "
        f"{'-start min':>10} {'sigma s':>8} {'resid deg':>9} along")
    for i, (d, e) in enumerate(doc["dates"].items()):
        if n is not None and i >= n:
            break
        ro = (f"{e.get('platform') or '-'}/R{int(e['relative_orbit']):03d}"
              if e.get("relative_orbit") is not None else "-")
        ds = (e.get("datatake_start") or "")[11:23]
        ins = (e.get("instant") or "")[11:23]
        log(f"{d:10} {e['status']:22} {ro:10} {ds:>15} {ins:>15} "
            f"{_fmt(e.get('minus_start_min')):>10} {_fmt(e.get('sigma_s')):>8} "
            f"{_fmt(e.get('residual_max_deg')):>9} {e.get('along_track') or '-'}")
        if "double_overpass" in e["flags"]:
            for p in e["passes"]:
                log(f"{'':10}   pass {p['platform']}/R{int(p['relative_orbit']):03d}: datatake "
                    f"start {p['datatake_start'][11:23]}, site instant "
                    f"{(p.get('instant') or '')[11:23]}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def cmd_fetch(args):
    from experiments.benchmark.fetch import sensing as fs

    for site in args.sites:
        geom = site_geometry(site, args.cubes_root, bbox=args.bbox)
        folder = Path(args.out) / site if args.out else fs.sensing_dir(site)
        fs.fetch_items(site, geom["bbox"], args.start, args.end, refresh=args.refresh,
                       path=folder / fs.ITEMS_NAME)
    return 0


#: what ``gate`` (the old name) prints before running the diagnostic
GATE_ALIAS_NOTE = ("'gate time' is now the OPTIONAL diagnostic 'diag time' (user decision "
                   "2026-10-14): it gates nothing; every run uses the catalogue times, and "
                   "its sensing_times.json is read only when given with --sensing-times")


def cmd_diag(args):
    """Run the diagnostic per site. Its verdict describes the inversion and
    gates nothing: the exit code is 0 once every site ran (1 only when a
    site could not be run)."""
    failed = []
    for site in args.sites:
        folder = Path(args.out) / site if args.out else None
        try:
            doc = run_site(site, args.start, args.end, root=args.cubes_root,
                           fetch=not args.no_fetch, refresh=args.refresh, model=args.model,
                           labels=args.labels, bbox=args.bbox, copy=bool(args.copy_to_cubes),
                           use_angles=not args.no_angles, folder=folder,
                           iso_time=args.iso_time)
        except Exception as e:                  # a diagnostic failure is reported, not raised
            log(f"[diag time] {site}: could not run ({type(e).__name__}: {e})")
            failed.append(site)
            continue
        print_table(doc, None if args.all_rows else 40)
        log(f"[diag time] {site}: diagnostic verdict {doc['verdict']} (optional; no run "
            "reads it unless given --sensing-times)")
    return 1 if failed else 0


def main(argv=None):
    """``python -X utf8 -m experiments.benchmark.sensing_time {fetch,diag,gate} ...``
    (``gate`` is the old name of ``diag``)."""
    from experiments import zarr_cubes as zc

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("fetch", "diag", "gate"),
                    help="fetch the STAC items; diag runs the optional diagnostic (gate: alias)")
    ap.add_argument("--sites", nargs="+", default=list(zc.SITES))
    ap.add_argument("--start", default=PERIOD[0])
    ap.add_argument("--end", default=PERIOD[1], help="end exclusive")
    ap.add_argument("--cubes-root", default=None)
    ap.add_argument("--out", default=None, help="output root (default results/validation_v2/sensing)")
    ap.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("W", "S", "E", "N"))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--model", choices=("grid", "centre"), default=MODEL)
    ap.add_argument("--iso-time", choices=("inertial", "ground"), default=ISO_TIME)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--copy-to-cubes", action="store_true",
                    help="also copy sensing_times.json beside the stores (no run reads it "
                         "unless given --sensing-times)")
    ap.add_argument("--no-angles", action="store_true")
    ap.add_argument("--all-rows", action="store_true")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "gate":
        log(f"[diag time] {GATE_ALIAS_NOTE}")
    return cmd_diag(args)


if __name__ == "__main__":
    sys.exit(main())
