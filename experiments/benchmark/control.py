# -*- coding: utf-8 -*-
"""Validation v2 — the positive control: a synthetic site 'pc' as Zarr stores (spec §10.2).

Question
--------
Does every lag method recover a planted lag profile when the truth is
known, in a world that is deliberately NOT MAREA's model (sub-pixel
wetness, linear spectral mixing, waves, clouds, haze, a continuous lag
profile instead of 3-km bands)? And how do the methods fail in worlds that
break their assumptions (hysteresis, missing-not-at-random masking,
amplification)?

The world (spec §10.2; plan INTERFACES §11)
-------------------------------------------
* Frame: H x W pixels of 10 m (default 60 x 1400: 600 m across, 14 km
  along), UTM 31N, the first pixel's edges on multiples of 20 m so the
  native 20 m lattice has phase (0, 0). The open sea is on the WEST edge
  (``mouth_side`` 'west'); the estuary runs east to land; dykes close the
  north and south edges.
* Planted DEM at 1 m sub-pixels (:func:`planted_dem`): a channel below the
  lowest tide, banks, flats on both sides whose low and high edges vary
  along the estuary (several slopes; a linear north profile and a convex
  south one), creeks cutting the flats, centimetre ripples, supratidal
  dykes and land. The elevation estimand is the 10 m mean (``z_true``).
* Boundary tide (:class:`PlantedTide`): M2, S2, N2, K2, K1, O1 and M4 with
  fixed amplitudes and phases (no nodal modulation), the h_B of the site.
* Lag: tau(s) = 120 min x min(s / 12 km, 1) ('profile') or 0 ('zero'), s
  the along-estuary distance from the sea edge; amplitude ratio 1 (gating).
  A sub-pixel is wet iff the local level A(s) h_B(t - tau(s)) plus a 2 cm
  wave term (a plane wave of 25 m wavelength, random direction and phase
  per scene) exceeds its elevation.
* Reflectance: linear mixing per pixel of library endmembers by their
  sub-pixel fractions: clear water with low suspended sediment, wet sand
  (exposed intertidal sediment) and dry sand (supratidal), plus Gaussian
  noise (2 % + 20 DN). The 10 m bands (B02, B03, B04, B08) are mixed at
  10 m; the 20 m bands (B8A, B11, B12) and the masks (SCL, CLD) at 20 m,
  then replicated 2 x 2 on the 10 m grid at phase (0, 0).
* Scenes: about 104 per year (one every 3.5 days) at 10:50 UTC +- 3 min,
  two alternating relative orbits; per scene a cloud class (clear 35 %,
  partly cloudy 40 %, mostly cloudy 15 %, overcast 10 %) with elliptical
  cloud blobs (SCL 8/9, CLD 55-100), thin cirrus (SCL 10), shadows offset
  from the clouds (SCL 3, x 0.35); haze on 2 % of the scenes (+400 DN in
  every band, undetected by SCL); no-data strips (fill, SCL 0) on 3 % of the
  scenes and isolated band fills; clear water SCL 6, sand SCL 5, 1 % SCL 7.
  No BOA offset (processing baseline 05.11).
* Misspecified worlds (descriptive, never gating): 'hysteresis' (ebb
  retention: on the falling limb a sub-pixel drains 20 min later),
  'mnar' (dark wet mud on the low flats, labelled SCL 2 or 3 at low tide)
  and 'amplification' (the amplitude ratio grows to 1.2 inland).

Stores and sidecars (the downloader's format, so the control exercises the
same reader): ``cubes/pc/<year>.zarr`` written by
``download_cube_zarr.write_store`` (imported, never edited), with
``<year>.verify.json`` (not compared: no legacy cube), ``<year>.format.txt``,
``<year>.job.json``, ``overpass_times.json`` and ``scene_meta.json`` (the
STAC-like sidecar with platforms, relative orbits and baselines). Beside
them: ``planted.npz`` (z_true, tau, amp (H, W) float32; ``z`` as truth.py's
planted_frame reads it) and ``world.json`` (every parameter, the dates,
instants, orbits, the store and planted hashes).

Seeds: FROZEN positive_control seed (20261013); the DEM, the schedule and
every scene's clouds, waves and noise depend only on the seed, the year and
the scene, so the scenarios and worlds share them (common random numbers).
"""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pandas as pd

from experiments.benchmark import config

SITE = config.CONTROL_SITE
WORLDS = ("gating", "hysteresis", "mnar", "amplification")
#: the world that gates the printing rule; the others are descriptive
GATING_WORLDS = ("gating",)
SCENARIOS = tuple(config.CONTROL_SCENARIOS)            # ("zero", "profile")
WORLD_FORMAT = "v2-control-world-1"
PLANTED_NAME = "planted.npz"
WORLD_NAME = "world.json"

_PC = config.FROZEN["positive_control"]
SEED = int(_PC["seed"])
FILL = -32768
PIXEL_M = 10.0
EPSG = 32631
#: the frame's north-west corner (multiples of 20 m: lattice phase (0, 0))
X_EDGE, Y_EDGE = 640_000.0, 5_900_000.0
DEFAULT_YEARS = (2024, 2025)
DEFAULT_FRAME = {"H": 60, "W": 1400, "scenes_per_year": 104, "sub_m": 1.0}

TIDE = {"ref_utc": "2000-01-01T12:00:00",
        "rule": "h_B(t) = sum A cos(speed (t - ref) - phase), hours, no nodal modulation",
        "constituents": [["M2", 28.9841042, 1.50, 112.0], ["S2", 30.0000000, 0.40, 168.0],
                         ["N2", 28.4397295, 0.28, 87.0], ["K2", 30.0821373, 0.11, 165.0],
                         ["K1", 15.0410686, 0.08, 40.0], ["O1", 13.9430356, 0.10, 245.0],
                         ["M4", 57.9682084, 0.08, 300.0]]}
LAG = {"max_min": 120.0, "ramp_km": 12.0,
       "profile": "tau(s) = 120 x min(s / 12 km, 1) min", "zero": "tau = 0"}
AMPLIFICATION = {"inland_ratio": 1.2, "ramp_km": 12.0,
                 "rule": "A(s) = 1 + 0.2 x min(s / 12 km, 1) in the 'amplification' world, "
                         "else 1"}
HYSTERESIS = {"ebb_delay_min": 20.0,
              "rule": "wet iff z < max(L(t), L(t - 20 min)) + wave: on the falling limb a "
                      "sub-pixel drains 20 min later"}
MNAR = {"mud_below_m": -0.6, "low_tide_quantile": 0.3, "p_scl2": 0.7,
        "rule": "flat sub-pixels below -0.6 m are dark wet mud when exposed; at scenes with "
                "h_B below its 30 % quantile their exposed 20 m pixels are labelled SCL 2 "
                "(p 0.7) or 3"}
WAVE = {"amp_m": 0.02, "wavelength_m": 25.0,
        "rule": "a plane wave 0.02 sin(k (x cos th + y sin th) + phi), th and phi uniform per "
                "scene"}
DEM = {"sea_m": -5.0, "land_m": 4.0, "channel_axis_m": [-4.5, -4.0], "channel_edge_rise_m": 0.5,
       "flat_knots": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
       "flat_low_m": [-1.9, -1.3, -1.7, -1.1], "flat_high_m": [1.5, 1.8, 1.0, 1.6],
       "profile_exponent": {"north": 1.0, "south": 1.6},
       "creek": {"spacing_m": 1200.0, "reach_u": 0.75, "half_width_m": [12.0, 4.0],
                 "depth_m": 0.7},
       "ripple": {"amp_m": 0.015, "lx_m": 37.0, "ly_m": 23.0},
       "widths": {"sea_share": 0.03, "land_share": 0.03, "dyke_share": 0.04,
                  "channel_share": 0.16, "bank_share": 0.07},
       "supratidal_m": 2.5}
ENDMEMBERS = {   # DN = reflectance x 10000 (illustrative library values, declared)
    "water": {"B02": 520, "B03": 480, "B04": 300, "B08": 150, "B8A": 120, "B11": 50, "B12": 30},
    "wet_sand": {"B02": 650, "B03": 800, "B04": 950, "B08": 1200, "B8A": 1250, "B11": 1050,
                 "B12": 750},
    "dry_sand": {"B02": 1400, "B03": 1750, "B04": 2100, "B08": 2600, "B8A": 2700, "B11": 3300,
                 "B12": 2900},
    "wet_mud": {"B02": 300, "B03": 350, "B04": 380, "B08": 450, "B8A": 460, "B11": 380,
                "B12": 260},
    "cloud": {"B02": 5000, "B03": 4800, "B04": 4700, "B08": 5200, "B8A": 5100, "B11": 3500,
              "B12": 2700},
}
NOISE = {"rel": 0.02, "dn": 20.0}
CLOUDS = {"classes": [["clear", 0.35, 0.0, 0.0], ["partly", 0.40, 0.05, 0.40],
                      ["mostly", 0.15, 0.50, 0.90], ["overcast", 0.10, 1.0, 1.0]],
          "blob_x_share": [0.02, 0.25], "blob_y_share": [0.15, 0.6],
          "cld": {"clear": [0, 6], "cloud": [55, 101], "cirrus": [20, 46], "shadow": [0, 9]},
          "scl9_min_cld": 80, "cirrus_share_partly": 0.3, "cirrus_dn": 600,
          "shadow_offset_px20": [4, 10], "shadow_factor": 0.35, "scl7_share": 0.01}
HAZE = {"share": float(_PC["haze"]["share"]), "dn": int(_PC["haze"]["dn"]), "cld_add": [5, 16]}
FILLS = {"nodata_strip_share": 0.03, "strip_cols_share": [0.05, 0.15],
         "band_fill_share": 0.05, "band_fill_px": 3}
OVERPASS = {"time_utc": "10:50", "jitter_min": 3.0, "orbits": [8, 51],
            "platforms": ["sentinel-2a", "sentinel-2b"], "baseline": "05.11", "tile": "31UFU"}
SPECTRAL10 = ("B02", "B03", "B04", "B08")
SPECTRAL20 = ("B8A", "B11", "B12")

CONFIG = {"site": SITE, "worlds": list(WORLDS), "gating_worlds": list(GATING_WORLDS),
          "scenarios": list(SCENARIOS), "seed": SEED, "pixel_m": PIXEL_M, "epsg": EPSG,
          "frame_edge": [X_EDGE, Y_EDGE], "default_years": list(DEFAULT_YEARS),
          "default_frame": dict(DEFAULT_FRAME), "tide": TIDE, "lag": LAG,
          "amplification": AMPLIFICATION, "hysteresis": HYSTERESIS, "mnar": MNAR,
          "wave": WAVE, "dem": DEM, "endmembers": ENDMEMBERS, "noise": NOISE,
          "clouds": CLOUDS, "haze": HAZE, "fills": FILLS, "overpass": OVERPASS,
          "lattice20": [0, 0], "offsets": "0 DN on every scene (no BOA offset)",
          "tolerances": config._plain(_PC["tolerances"]),
          "must_pass": config._plain(_PC["must_pass"])}


def log(msg):
    print(msg, flush=True)


def _check_world(world, scenario):
    if world not in WORLDS:
        raise ValueError(f"unknown control world {world!r}; expected one of {WORLDS}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown control scenario {scenario!r}; expected one of {SCENARIOS}")


# ─────────────────────────────────────────────────────────────────────────────
#  the boundary tide
# ─────────────────────────────────────────────────────────────────────────────

def tide_levels(ns, tide=None):
    """The analytic boundary tide at int64 nanoseconds (float64 metres)."""
    tide = TIDE if tide is None else tide
    ref = pd.Timestamp(tide["ref_utc"]).value
    hours = (np.asarray(ns, np.int64) - ref) / 3.6e12
    out = np.zeros(np.shape(hours), np.float64)
    for _, speed, amp, phase in tide["constituents"]:
        out += float(amp) * np.cos(np.radians(float(speed) * hours - float(phase)))
    return out


class PlantedTide:
    """The control's boundary provider: ``levels(times)`` answers in TIME
    ORDER (as ``PyTMDBoundary``), float64, deterministic."""

    def __init__(self, tide=None, name="planted tide"):
        self.tide = json.loads(json.dumps(TIDE if tide is None else tide))
        self.name = str(name)

    def levels(self, times):
        ns = np.sort(pd.DatetimeIndex(times).as_unit("ns").asi8)
        return tide_levels(ns, self.tide)


def planted_provider(world_doc):
    """The analytic boundary tide of a world document."""
    return PlantedTide(world_doc.get("tide", TIDE),
                       name=f"planted tide ({world_doc.get('world')}, "
                            f"{world_doc.get('scenario')})")


# ─────────────────────────────────────────────────────────────────────────────
#  lag and amplitude profiles
# ─────────────────────────────────────────────────────────────────────────────

def lag_profile(s_km, scenario):
    """tau(s) in minutes: 120 x min(s / 12 km, 1) ('profile') or 0 ('zero')."""
    s = np.clip(np.asarray(s_km, np.float64), 0.0, None)
    if scenario == "zero":
        return np.zeros_like(s)
    if scenario == "profile":
        return LAG["max_min"] * np.minimum(s / LAG["ramp_km"], 1.0)
    raise ValueError(f"unknown scenario {scenario!r}")


def amp_profile(s_km, world):
    """The amplitude ratio A(s): 1, or 1 -> 1.2 at 12 km in 'amplification'."""
    s = np.clip(np.asarray(s_km, np.float64), 0.0, None)
    if world == "amplification":
        return 1.0 + (AMPLIFICATION["inland_ratio"] - 1.0) * np.minimum(
            s / AMPLIFICATION["ramp_km"], 1.0)
    return np.ones_like(s)


# ─────────────────────────────────────────────────────────────────────────────
#  the planted DEM
# ─────────────────────────────────────────────────────────────────────────────

def _frame(frame):
    f = dict(DEFAULT_FRAME)
    f.update(frame or {})
    H, W = int(f["H"]), int(f["W"])
    sub = float(f["sub_m"])
    if H < 8 or W < 40 or H % 2 or W % 2:
        raise ValueError(f"the control frame must be even and at least 8 x 40 px, got {H} x {W}")
    k = PIXEL_M / sub
    if abs(k - round(k)) > 1e-9 or round(k) < 1:
        raise ValueError(f"sub_m must divide {PIXEL_M} m, got {sub}")
    return {"H": H, "W": W, "sub_m": sub, "factor": int(round(k)),
            "scenes_per_year": int(f["scenes_per_year"])}


def geometry_m(H, W):
    """The zone widths (m) of a frame of H x W pixels."""
    w = DEM["widths"]
    across, along = H * PIXEL_M, W * PIXEL_M
    half = across / 2.0
    out = {"across_m": across, "along_m": along, "half_m": half,
           "sea_m": max(40.0, w["sea_share"] * along),
           "land_m": max(40.0, w["land_share"] * along),
           "dyke_m": max(10.0, w["dyke_share"] * across),
           "ch_half_m": max(8.0, w["channel_share"] * half),
           "bank_m": max(4.0, w["bank_share"] * half)}
    out["flat_m"] = half - out["dyke_m"] - out["ch_half_m"] - out["bank_m"]
    out["estuary_m"] = along - out["sea_m"] - out["land_m"]
    if out["flat_m"] <= 0:
        raise ValueError("the frame is too narrow for flats")
    return out


def s_of_x(x_m, geom):
    """The along-estuary distance (km) from the sea edge of an x (m from the
    frame's west edge): 0 in the sea."""
    return np.clip((np.asarray(x_m, np.float64) - geom["sea_m"]) / 1000.0, 0.0, None)


def planted_dem(H, W, sub_m=1.0):
    """The planted DEM at sub-pixel resolution.

    Returns
    -------
    dict
        ``z_sub`` (H f, W f) float32 (f = 10 / sub_m), ``z_true`` (H, W)
        float32 (the 10 m mean), ``zone`` (H, W) int8 (0 sea, 1 channel,
        2 bank, 3 flat, 4 creek, 5 dyke/land; the majority of the
        sub-pixels), ``x_m`` (W f,), ``y_m`` (H f,), ``geom``.
    """
    f = int(round(PIXEL_M / float(sub_m)))
    g = geometry_m(H, W)
    x = (np.arange(W * f) + 0.5) * float(sub_m)
    y = (np.arange(H * f) + 0.5) * float(sub_m)
    X = x[None, :]
    Y = y[:, None]
    dist = np.abs(Y - g["half_m"])                          # from the channel axis
    xi = np.clip((X - g["sea_m"]) / g["estuary_m"], 0.0, 1.0)  # along the estuary, 0..1
    zl = np.interp(xi, DEM["flat_knots"], DEM["flat_low_m"])
    zh = np.interp(xi, DEM["flat_knots"], DEM["flat_high_m"])
    p = np.where(Y < g["half_m"], DEM["profile_exponent"]["north"],
                 DEM["profile_exponent"]["south"])
    u = (dist - g["ch_half_m"] - g["bank_m"]) / g["flat_m"]
    uc = np.clip(u, 0.0, 1.0)
    z = zl + (zh - zl) * uc ** p
    zone = np.full(z.shape, 3, np.int8)
    # creeks across the flats (alternate sides)
    cr = DEM["creek"]
    k, xc = 1, g["sea_m"] + cr["spacing_m"]
    while xc < g["along_m"] - g["land_m"] - 300.0:
        north = bool(k % 2)
        side = (Y < g["half_m"]) if north else (Y >= g["half_m"])
        hw = cr["half_width_m"][0] + (cr["half_width_m"][1] - cr["half_width_m"][0]) * \
            np.clip(uc / cr["reach_u"], 0.0, 1.0)
        dx = np.abs(X - xc)
        m = side & (u >= 0.0) & (u <= cr["reach_u"]) & (dx < hw)
        depth = cr["depth_m"] * (1.0 - (dx / hw) ** 2) * (1.0 - 0.5 * uc / cr["reach_u"])
        z = np.where(m, z - depth, z)
        zone = np.where(m, np.int8(4), zone)
        k += 1
        xc += cr["spacing_m"]
    rp = DEM["ripple"]
    z = z + rp["amp_m"] * np.sin(2 * np.pi * X / rp["lx_m"]) * np.sin(2 * np.pi * Y / rp["ly_m"])
    # channel and banks
    axis = DEM["channel_axis_m"][0] + (DEM["channel_axis_m"][1] - DEM["channel_axis_m"][0]) * xi
    z_ch = axis + DEM["channel_edge_rise_m"] * (dist / g["ch_half_m"]) ** 2
    ch = dist <= g["ch_half_m"]
    z = np.where(ch, z_ch, z)
    zone = np.where(ch, np.int8(1), zone)
    bank = (dist > g["ch_half_m"]) & (dist < g["ch_half_m"] + g["bank_m"])
    w_b = (dist - g["ch_half_m"]) / g["bank_m"]
    z_edge = axis + DEM["channel_edge_rise_m"]
    z = np.where(bank, z_edge + (zl - z_edge) * w_b, z)
    zone = np.where(bank, np.int8(2), zone)
    # dykes, sea and land
    dyke = (Y < g["dyke_m"]) | (Y > g["across_m"] - g["dyke_m"])
    sea = X < g["sea_m"]
    land = X > g["along_m"] - g["land_m"]
    z = np.where(dyke | land, DEM["land_m"], z)
    zone = np.where(dyke | land, np.int8(5), zone)
    z = np.where(sea, DEM["sea_m"], z)
    zone = np.where(sea, np.int8(0), zone)
    z = np.broadcast_to(z, (H * f, W * f)).astype(np.float32)
    zone = np.broadcast_to(zone, (H * f, W * f))
    z_true = z.reshape(H, f, W, f).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    counts = np.stack([(zone == c).reshape(H, f, W, f).sum(axis=(1, 3)) for c in range(6)])
    zone_px = counts.argmax(axis=0).astype(np.int8)
    return {"z_sub": np.ascontiguousarray(z), "z_true": z_true, "zone": zone_px, "x_m": x,
            "y_m": y, "geom": g, "factor": f}


# ─────────────────────────────────────────────────────────────────────────────
#  the schedule
# ─────────────────────────────────────────────────────────────────────────────

def scene_schedule(years, scenes_per_year, seed=SEED):
    """Dates, instants, orbits, platforms and the per-scene draws (cloud
    class and cover, haze, no-data strips, band fills).

    Every year has ``scenes_per_year`` dates, one every 365 / n days from
    January 2, at 10:50 UTC + U(-3, 3) min (whole seconds); the relative
    orbit and the platform alternate. Haze falls on round(2 % x n) scenes of
    the whole schedule (exactly), no-data strips on round(3 % x n).

    Returns
    -------
    pandas.DataFrame
        One row per scene: date, instant, year, k, orbit, platform,
        cloud_class, cloud_cover, haze, nodata_strip, band_fill.
    """
    if not 1 <= int(scenes_per_year) <= 365:
        raise ValueError(f"scenes_per_year must be 1..365 (one mosaic per date), got "
                         f"{scenes_per_year}")
    rows = []
    hh, mm = (int(v) for v in OVERPASS["time_utc"].split(":"))
    for y in years:
        n = int(scenes_per_year)
        rng = np.random.default_rng([seed, 1, int(y)])
        day = np.unique(np.floor(np.arange(n) * 365.0 / n + 1.0).astype(int))
        jit = rng.uniform(-OVERPASS["jitter_min"], OVERPASS["jitter_min"], len(day))
        for k, (d, j) in enumerate(zip(day, jit)):
            date = pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(days=int(d))
            if date.year != int(y):
                continue
            inst = date + pd.Timedelta(hours=hh, minutes=mm) + pd.Timedelta(
                seconds=int(round(j * 60.0)))
            rows.append({"date": str(date.date()), "instant": inst, "year": int(y), "k": k,
                         "orbit": int(OVERPASS["orbits"][k % 2]),
                         "platform": OVERPASS["platforms"][(k // 2) % 2]})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng([seed, 2])
    names = [c[0] for c in CLOUDS["classes"]]
    probs = np.array([c[1] for c in CLOUDS["classes"]], float)
    cls = rng.choice(len(names), size=len(df), p=probs / probs.sum())
    lo = np.array([CLOUDS["classes"][c][2] for c in cls])
    hi = np.array([CLOUDS["classes"][c][3] for c in cls])
    df["cloud_class"] = [names[c] for c in cls]
    df["cloud_cover"] = lo + (hi - lo) * rng.random(len(df))
    n_all = len(df)
    for col, share in (("haze", HAZE["share"]), ("nodata_strip", FILLS["nodata_strip_share"]),
                       ("band_fill", FILLS["band_fill_share"])):
        pick = np.zeros(n_all, bool)
        n_pick = int(round(share * n_all))
        if n_pick:
            pick[rng.choice(n_all, n_pick, replace=False)] = True
        df[col] = pick
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  one scene
# ─────────────────────────────────────────────────────────────────────────────

def _wet_sub(z_sub, level, x_m, y_m, rng):
    """Sub-pixel wetness: z < level(x) + the 2 cm plane wave."""
    A = float(WAVE["amp_m"])
    d = z_sub - level[None, :].astype(np.float32)
    wet = d < -A
    amb = np.abs(d) <= A
    th, ph = rng.uniform(0.0, 2 * np.pi, 2)
    if amb.any():
        ii, jj = np.nonzero(amb)
        kk = 2 * np.pi / float(WAVE["wavelength_m"])
        w = A * np.sin(kk * (x_m[jj] * np.cos(th) + y_m[ii] * np.sin(th)) + ph)
        wet[ii, jj] = d[ii, jj] < w
    return wet


def _block_share(mask, H, W, f):
    return mask.reshape(H, f, W, f).sum(axis=(1, 3), dtype=np.int64) / float(f * f)


def _mix(frac, bands, rng, shape):
    """Linear mixing of the endmembers by their fractions, plus noise."""
    out = {}
    for b in bands:
        r = np.zeros(shape, np.float64)
        for em, fr in frac.items():
            r += fr * float(ENDMEMBERS[em][b])
        r = r * (1.0 + rng.normal(0.0, NOISE["rel"], shape)) + rng.normal(0.0, NOISE["dn"], shape)
        out[b] = r
    return out


def _ellipses(rng, Hb, Wb, cover, max_blobs=60):
    """A cloud mask of about ``cover`` (share of the 20 m frame)."""
    m = np.zeros((Hb, Wb), bool)
    if cover <= 0:
        return m
    if cover >= 1.0:
        m[:] = True
        return m
    rr, cc = np.mgrid[0:Hb, 0:Wb]
    for _ in range(max_blobs):
        if m.mean() >= cover:
            break
        ax = rng.uniform(*CLOUDS["blob_x_share"]) * Wb
        ay = rng.uniform(*CLOUDS["blob_y_share"]) * Hb
        cy, cx = rng.uniform(0, Hb), rng.uniform(0, Wb)
        m |= ((cc - cx) / max(ax, 1.0)) ** 2 + ((rr - cy) / max(ay, 1.0)) ** 2 <= 1.0
    return m


def world_dem(H, W, sub_m=1.0):
    """:func:`planted_dem` plus what the scenes need: the supratidal
    sub-pixels (dry sand) and their 10 m / 20 m shares, and the low-flat mud
    sub-pixels of the 'mnar' world."""
    dem = planted_dem(H, W, sub_m)
    f = dem["factor"]
    supra_sub = dem["z_sub"] >= DEM["supratidal_m"]
    dem["supra_sub"] = supra_sub
    dem["supra10"] = _block_share(supra_sub, H, W, f)
    dem["supra20"] = _block_share(supra_sub, H // 2, W // 2, 2 * f)
    zone_sub = np.repeat(np.repeat(dem["zone"], f, axis=0), f, axis=1)
    dem["mud_sub"] = (dem["z_sub"] < MNAR["mud_below_m"]) & np.isin(zone_sub, (3, 4))
    return dem


def _scene(dem, row, world, scenario, tide, low_tide, seed):
    """The raw int16 bands (H, W) of one scene (the world's physics)."""
    from experiments.benchmark import obs

    z_sub, f = dem["z_sub"], dem["factor"]
    H, W = dem["z_true"].shape
    Hb, Wb = H // 2, W // 2
    rng = np.random.default_rng([seed, 3, int(row["year"]), int(row["k"])])
    rngw = np.random.default_rng([seed, 4, int(row["year"]), int(row["k"]),
                                  WORLDS.index(world)])
    s = s_of_x(dem["x_m"], dem["geom"])
    tau = lag_profile(s, scenario)
    amp = amp_profile(s, world)
    t_ns = pd.Timestamp(row["instant"]).value
    level = amp * tide_levels(t_ns - np.round(tau * 6e10).astype(np.int64), tide)
    if world == "hysteresis":
        prev = amp * tide_levels(t_ns - np.round((tau + HYSTERESIS["ebb_delay_min"]) * 6e10)
                                 .astype(np.int64), tide)
        level = np.maximum(level, prev)
    wet = _wet_sub(z_sub, level, dem["x_m"], dem["y_m"], rng)
    supra = dem["supra_sub"]
    fr10 = {"water": _block_share(wet, H, W, f), "dry_sand": dem["supra10"]}
    fr20 = {"water": _block_share(wet, Hb, Wb, 2 * f), "dry_sand": dem["supra20"]}
    if world == "mnar":
        mud = ~wet & dem["mud_sub"]
        fr10["wet_mud"] = _block_share(mud, H, W, f)
        fr20["wet_mud"] = _block_share(mud, Hb, Wb, 2 * f)
    for fr in (fr10, fr20):
        fr["wet_sand"] = np.clip(1.0 - sum(v for k, v in fr.items()), 0.0, 1.0)
    del wet, supra
    b10 = _mix(fr10, SPECTRAL10, rng, (H, W))
    b20 = _mix(fr20, SPECTRAL20, rng, (Hb, Wb))
    # surface classes (SCL) and cloud probability (CLD) at 20 m
    scl = np.where(fr20["water"] >= 0.5, 6, 5).astype(np.int16)
    scl = np.where(rng.random((Hb, Wb)) < CLOUDS["scl7_share"], 7, scl).astype(np.int16)
    cld = rng.integers(*CLOUDS["cld"]["clear"], (Hb, Wb)).astype(np.int16)
    if world == "mnar" and low_tide:
        mud20 = fr20.get("wet_mud", np.zeros((Hb, Wb))) >= 0.5
        lab = np.where(rngw.random((Hb, Wb)) < MNAR["p_scl2"], 2, 3)
        scl = np.where(mud20, lab, scl).astype(np.int16)
    cloud = _ellipses(rng, Hb, Wb, float(row["cloud_cover"]))
    cirrus = np.zeros((Hb, Wb), bool)
    if row["cloud_class"] == "partly" and rng.random() < CLOUDS["cirrus_share_partly"]:
        cirrus = _ellipses(rng, Hb, Wb, 0.1) & ~cloud
    dy, dx = CLOUDS["shadow_offset_px20"]
    shadow = np.zeros((Hb, Wb), bool)
    shadow[dy:, dx:] = cloud[:Hb - dy, :Wb - dx] if (dy < Hb and dx < Wb) else False
    shadow &= ~cloud & ~cirrus
    cprob = rng.integers(*CLOUDS["cld"]["cloud"], (Hb, Wb))
    scl = np.where(cloud, np.where(cprob >= CLOUDS["scl9_min_cld"], 9, 8), scl).astype(np.int16)
    cld = np.where(cloud, cprob, cld).astype(np.int16)
    scl = np.where(cirrus, 10, scl).astype(np.int16)
    cld = np.where(cirrus, rng.integers(*CLOUDS["cld"]["cirrus"], (Hb, Wb)), cld).astype(np.int16)
    scl = np.where(shadow, 3, scl).astype(np.int16)
    cld = np.where(shadow, rng.integers(*CLOUDS["cld"]["shadow"], (Hb, Wb)), cld).astype(np.int16)
    phase = (0, 0)
    cloud10 = obs.expand20(cloud, phase, (H, W))
    cirrus10 = obs.expand20(cirrus, phase, (H, W))
    shadow10 = obs.expand20(shadow, phase, (H, W))
    out = {}
    for b in SPECTRAL10:
        v = b10[b]
        v = np.where(cloud10, float(ENDMEMBERS["cloud"][b]) * (1 + rng.normal(0, 0.05, (H, W))), v)
        v = np.where(cirrus10, v + CLOUDS["cirrus_dn"], v)
        v = np.where(shadow10, v * CLOUDS["shadow_factor"], v)
        out[b] = v
    for b in SPECTRAL20:
        v = b20[b]
        v = np.where(cloud, float(ENDMEMBERS["cloud"][b]) * (1 + rng.normal(0, 0.05, (Hb, Wb))), v)
        v = np.where(cirrus, v + CLOUDS["cirrus_dn"], v)
        v = np.where(shadow, v * CLOUDS["shadow_factor"], v)
        out[b] = obs.expand20(v, phase, (H, W))
    if bool(row["haze"]):
        for b in out:
            out[b] = out[b] + HAZE["dn"]
        cld = np.clip(cld + rng.integers(*HAZE["cld_add"], (Hb, Wb)), 0, 100).astype(np.int16)
    raw = {b: np.clip(np.round(v), 1, 15000).astype(np.int16) for b, v in out.items()}
    raw["SCL"] = obs.expand20(scl, phase, (H, W)).astype(np.int16)
    raw["CLD"] = obs.expand20(cld, phase, (H, W)).astype(np.int16)
    if bool(row["nodata_strip"]):
        share = rng.uniform(*FILLS["strip_cols_share"])
        c0 = int(W - 2 * max(1, round(share * W / 2.0)))     # even: the 20 m lattice
        for b in raw:
            raw[b][:, c0:] = FILL
    if bool(row["band_fill"]):
        for b in ("B03", "B08"):
            r_ = rng.integers(0, H, FILLS["band_fill_px"])
            c_ = rng.integers(0, W, FILLS["band_fill_px"])
            raw[b][r_, c_] = FILL
    return raw


# ─────────────────────────────────────────────────────────────────────────────
#  writing a world
# ─────────────────────────────────────────────────────────────────────────────

def _wkt():
    from pyproj import CRS
    return CRS.from_epsg(EPSG).to_wkt("WKT1_GDAL")


def _coords(H, W):
    return (X_EDGE + PIXEL_M / 2 + PIXEL_M * np.arange(W),
            Y_EDGE - PIXEL_M / 2 - PIXEL_M * np.arange(H))


def _write_json(path, obj):
    from experiments.benchmark import product
    product.write_json(path, obj)


def _scene_meta(df, frame):
    """The STAC-like scene sidecar (``inputs.scene_meta_from_items``)."""
    from pyproj import Transformer

    from experiments.benchmark import inputs

    xs, ys = _coords(frame["H"], frame["W"])
    tr = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(float(xs[0]), float(ys[-1]))
    lon1, lat1 = tr.transform(float(xs[-1]), float(ys[0]))
    feats = []
    for r in df.itertuples():
        stamp = pd.Timestamp(r.instant)
        plat = str(r.platform)
        pid = f"S2{plat[-1].upper()}_MSIL2A_{stamp:%Y%m%dT%H%M%S}_N0511_R{int(r.orbit):03d}_" \
              f"T{OVERPASS['tile']}_{stamp:%Y%m%dT%H%M%S}"
        feats.append({"id": pid, "bbox": [lon0, lat0, lon1, lat1], "properties": {
            "datetime": stamp.strftime("%Y-%m-%dT%H:%M:%S.000000Z"), "platform": plat,
            "processing:version": OVERPASS["baseline"], "sat:relative_orbit": int(r.orbit),
            "sat:absolute_orbit": int(30000 + 143 * int(r.k) + int(r.orbit)),
            "sat:orbit_state": "descending", "eo:cloud_cover": round(100 * float(r.cloud_cover), 2),
            "grid:code": f"MGRS-{OVERPASS['tile']}", "product:timeliness_category": "synthetic"}})
    return inputs.scene_meta_from_items(feats, site=SITE,
                                        query={"synthetic": "experiments.benchmark.control"})


def write_world(out_root, world="gating", scenario="profile", years=DEFAULT_YEARS, frame=None,
                seed=None):
    """Write one control world: the yearly stores of site 'pc' with their
    sidecars, ``planted.npz`` and ``world.json`` under ``out_root``.

    Parameters
    ----------
    out_root : str
        The world's folder (the stores go to ``<out_root>/cubes/pc/``).
    world : str
        One of :data:`WORLDS` ('gating' decides the printing rule).
    scenario : str
        'profile' (0 -> 120 min at 12 km) or 'zero'.
    years : sequence of int
        Default (2024, 2025): window W2 spans more than 365 days (HD26).
    frame : dict, optional
        ``H``, ``W`` (10 m pixels, even), ``scenes_per_year``, ``sub_m``
        (default 60 x 1400, 104, 1 m).
    seed : int, optional
        Default FROZEN positive_control seed.

    Returns
    -------
    dict
        ``{"cubes_root", "planted_path", "world_doc"}``.
    """
    import xarray as xr

    from experiments import download_cube_zarr as dcz
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs, product

    _check_world(world, scenario)
    seed = SEED if seed is None else int(seed)
    fr = _frame(frame)
    H, W = fr["H"], fr["W"]
    years = tuple(int(y) for y in years)
    out_root = str(out_root).replace("\\", "/")
    cubes_root = f"{out_root}/cubes"
    os.makedirs(cubes_root, exist_ok=True)
    dem = world_dem(H, W, fr["sub_m"])
    df = scene_schedule(years, fr["scenes_per_year"], seed)
    tide = TIDE
    h_inst = tide_levels(pd.DatetimeIndex(df["instant"]).as_unit("ns").asi8, tide)
    low_cut = float(np.quantile(h_inst, MNAR["low_tide_quantile"])) if len(h_inst) else 0.0
    df["h_B"] = h_inst
    df["low_tide"] = h_inst < low_cut
    xs, ys = _coords(H, W)
    wkt = _wkt()
    site_dir = zc.site_dir(SITE, cubes_root)
    os.makedirs(site_dir, exist_ok=True)
    stores = {}
    for y in years:
        dy = df[df["year"] == y]
        if not len(dy):
            raise ValueError(f"no scene in {y}")
        data = {b: np.empty((len(dy), H, W), np.int16) for b in zc.BANDS}
        for i, row in enumerate(dy.to_dict("records")):
            sc = _scene(dem, row, world, scenario, tide, bool(row["low_tide"]), seed)
            for b in zc.BANDS:
                data[b][i] = sc[b]
        # an UNNAMED index: xarray writes a coordinate built from an index named
        # otherwise ('date') with the wrong time values
        t = pd.DatetimeIndex(np.asarray(dy["date"], dtype="datetime64[ns]"))
        ds = xr.Dataset({b: (("t", "y", "x"), data[b], {"long_name": b}) for b in zc.BANDS},
                        coords={"t": t, "y": ys, "x": xs})
        ds["crs"] = xr.DataArray(np.int32(0), attrs={
            "crs_wkt": wkt, "spatial_ref": wkt,
            "GeoTransform": f"{X_EDGE} {PIXEL_M:g} 0 {Y_EDGE} 0 -{PIXEL_M:g}"})
        attrs = {"site": SITE, "synthetic": 1, "world": world, "scenario": scenario,
                 "job_id": f"synthetic-{SITE}-{world}-{scenario}-{y}",
                 "collection": "synthetic positive control (experiments.benchmark.control)",
                 "grid_epsg": EPSG, "bands": json.dumps(list(zc.BANDS)),
                 "input_baselines": json.dumps({d: [OVERPASS["baseline"]] for d in dy["date"]})}
        out = zc.store_path(SITE, y, cubes_root)
        dcz.write_store(ds, {b: FILL for b in zc.BANDS}, out, attrs, log=lambda m: None,
                        verify=True)
        del ds, data
        _write_json(str(zc.sidecar_path(SITE, y, inputs.VERIFY_NAME, cubes_root)),
                    {"verdict": "not compared", "site": SITE, "year": int(y), "synthetic": True,
                     "reason": "synthetic positive-control store: there is no legacy netCDF "
                               "cube to compare with"})
        with open(zc.sidecar_path(SITE, y, "format.txt", cubes_root), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write("\n".join([
                f"synthetic positive-control store {SITE} {y} ({world}, {scenario})",
                "written by experiments.benchmark.control.write_world with "
                "download_cube_zarr.write_store",
                f"bands {', '.join(zc.BANDS)}: int16, _FillValue {FILL}, no scale or offset",
                "20 m bands (B8A, B11, B12, SCL, CLD) generated at 20 m and replicated 2 x 2 "
                "at lattice phase (0, 0)",
                f"processing baseline {OVERPASS['baseline']}: no BOA offset in the values",
                f"{len(dy)} dates, {H} x {W} px of {PIXEL_M:g} m, EPSG:{EPSG}", ""]))
        _write_json(str(zc.sidecar_path(SITE, y, "job.json", cubes_root)),
                    {"job_id": attrs["job_id"], "synthetic": True, "site": SITE, "year": int(y),
                     "world": world, "scenario": scenario, "seed": seed,
                     "generator": "experiments.benchmark.control.write_world"})
        stores[str(y)] = {"path": str(out).replace("\\", "/"),
                          "zmetadata_sha256": product.sha256_file(str(out / ".zmetadata"))}
        log(f"[control] {world}/{scenario}: wrote {out} ({len(dy)} scenes)")
    inst_map = {r["date"]: pd.Timestamp(r["instant"]).isoformat() for r in df.to_dict("records")}
    _write_json(str(zc.overpass_path(SITE, cubes_root)), inst_map)
    inputs.write_scene_meta(inputs.scene_meta_path(SITE, cubes_root), _scene_meta(df, fr))
    s_px = s_of_x((np.arange(W) + 0.5) * PIXEL_M, dem["geom"])
    tau_px = np.broadcast_to(lag_profile(s_px, scenario)[None, :], (H, W)).astype(np.float32)
    amp_px = np.broadcast_to(amp_profile(s_px, world)[None, :], (H, W)).astype(np.float32)
    planted_path = f"{out_root}/{PLANTED_NAME}"
    product.save_npz(planted_path, {
        "z_true": dem["z_true"], "z": dem["z_true"], "tau": np.ascontiguousarray(tau_px),
        "amp": np.ascontiguousarray(amp_px), "zone": dem["zone"],
        "s_true_km": np.broadcast_to(s_px[None, :], (H, W)).astype(np.float32)})
    doc = {"format": WORLD_FORMAT, "site": SITE, "world": world, "scenario": scenario,
           "gating": world in GATING_WORLDS, "years": list(years), "seed": seed,
           "frame": {"H": H, "W": W, "pixel_m": PIXEL_M, "sub_m": fr["sub_m"],
                     "scenes_per_year": fr["scenes_per_year"], "x_edge": X_EDGE,
                     "y_edge": Y_EDGE, "epsg": EPSG, "lattice20": [0, 0],
                     "geometry_m": dem["geom"]},
           "tide": tide, "config": CONFIG, "config_sha256": config.sha256_json(CONFIG),
           "code_sha256": config.sha256_source(os.path.abspath(__file__)),
           "dates": list(df["date"]), "instants": inst_map,
           "orbits": {r["date"]: int(r["orbit"]) for r in df.to_dict("records")},
           "platforms": {r["date"]: str(r["platform"]) for r in df.to_dict("records")},
           "offsets": {d: 0 for d in df["date"]},
           "scenes": {r["date"]: {"cloud_class": r["cloud_class"],
                                  "cloud_cover": round(float(r["cloud_cover"]), 4),
                                  "haze": bool(r["haze"]), "nodata_strip": bool(r["nodata_strip"]),
                                  "band_fill": bool(r["band_fill"]),
                                  "low_tide": bool(r["low_tide"])}
                      for r in df.to_dict("records")},
           "haze_share": float(df["haze"].mean()) if len(df) else 0.0,
           "low_tide_cut_m": low_cut,
           "paths": {"out_root": out_root, "cubes_root": cubes_root, "planted": planted_path,
                     "world": f"{out_root}/{WORLD_NAME}"},
           "stores": stores, "planted_sha256": product.sha256_file(planted_path),
           "tolerances": CONFIG["tolerances"], "must_pass": CONFIG["must_pass"],
           "expected_offsets": "the cards' control_offset_min (control.expected_offset)",
           "created_utc": product.utc_stamp()}
    _write_json(doc["paths"]["world"], doc)
    return {"cubes_root": cubes_root, "planted_path": planted_path, "world_doc": doc}


def load_world(out_root):
    """The ``world.json`` of a written world."""
    from experiments.benchmark import product

    doc = product.read_json(f"{str(out_root).rstrip('/')}/{WORLD_NAME}")
    if doc.get("format") != WORLD_FORMAT:
        raise ValueError(f"{out_root}/{WORLD_NAME}: not a {WORLD_FORMAT} document")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
#  the package of a world
# ─────────────────────────────────────────────────────────────────────────────

def control_site_config(out_root, world_doc):
    """The ``config.SiteConfig`` of site 'pc' (built here; config.py's
    site_config serves the v13 20 m control only)."""
    cubes_root = world_doc["paths"]["cubes_root"]
    return config.SiteConfig(
        site=SITE, res_m=int(PIXEL_M), pixel_m=PIXEL_M, cube=None, extract=None,
        overpass=f"{cubes_root}/{SITE}/overpass_times.json",
        boundary=types.MappingProxyType({"kind": "planted", "name": planted_provider(
            world_doc).name, "world": world_doc["world"], "scenario": world_doc["scenario"]}),
        mouth_side="west", elevation_truth=("planted",), judges=(),
        published=types.MappingProxyType({}), transition_result=None,
        root=str(out_root).replace("\\", "/"), tag="planted", scenario=world_doc["scenario"],
        extra=types.MappingProxyType({"planted": world_doc["paths"]["planted"],
                                      "world": world_doc["paths"]["world"]}))


def control_site2(out_root, world_doc):
    """The ``package.V2Site`` of a world: site 'pc', mouth 'west', no judges,
    the world's instants, offsets {date: 0}, lattice (0, 0), its years and
    the P0 observation key only. Build it with ``package.prepare_v2(site2,
    provider=planted_provider(world_doc))``."""
    from experiments.benchmark import package, settings

    return package.V2Site(site=SITE, cfg=control_site_config(out_root, world_doc),
                          root=str(out_root).replace("\\", "/"),
                          cubes_root=world_doc["paths"]["cubes_root"],
                          instants=dict(world_doc["instants"]),
                          offsets={d: 0 for d in world_doc["dates"]}, lattice=(0, 0),
                          years=tuple(int(y) for y in world_doc["years"]),
                          keys=(settings.P0.id,))


def open_package(site2, world_doc, dom_version=None):
    """``package.V2Package`` of a built control package with the planted
    tide as its live provider (``V2Package.provider`` cannot build a
    'planted' boundary itself)."""
    from experiments.benchmark import package

    pkg = package.V2Package(site2, dom_version)
    pkg._provider = planted_provider(world_doc)
    return pkg


def planted_band_lags(planted_path, domain):
    """The planted lag per band of a domain: the mean of the planted tau
    over the domain's keep pixels of each band (``band_of >= 0``).

    Returns
    -------
    dict
        ``{"bands": [int], "tau_mean_min": [float], "n_px": [int]}``.
    """
    with np.load(planted_path, allow_pickle=False) as z:
        tau = np.asarray(z["tau"], np.float64).ravel()
    keep = np.asarray(domain.keep, np.int64)
    band = np.asarray(domain.band_of, np.int64)
    if band.shape != keep.shape:
        raise ValueError("domain.band_of must align with domain.keep")
    tk = tau[keep]
    bands = sorted(int(b) for b in np.unique(band) if b >= 0)
    return {"bands": bands,
            "tau_mean_min": [float(np.mean(tk[band == b])) for b in bands],
            "n_px": [int(np.sum(band == b)) for b in bands]}


def expected_offset(product_id):
    """The expected offset (min) of a lag product in the gating world: its
    card's ``control_offset_min`` (``lags.card``), 0 when the card gives
    none. An unknown product raises KeyError."""
    from experiments.benchmark import lags

    card = lags.card(product_id)
    v = card.get("control_offset_min") if isinstance(card, dict) else None
    return 0.0 if v is None else float(v)


__all__ = ["WORLDS", "SCENARIOS", "write_world", "planted_provider", "control_site2",
           "planted_band_lags", "expected_offset", "load_world", "open_package",
           "scene_schedule", "planted_dem", "PlantedTide"]
