"""
sites.py — A registry of study areas
=====================================

Named areas of interest, so notebooks say ``sites.get("villaviciosa")``
instead of carrying a wall of coordinates. Coordinates written once and
reused are coordinates that stay consistent between the study, the
validation and the field campaign.

The registry ships the sites of this project (Cantabrian and Galician
estuaries) and takes your own with :func:`register`. Nothing here is
special-cased in the library: a registered site is just an
:class:`~pyintertidal.aoi.AOI`.
"""

from __future__ import annotations

from .aoi import AOI

#: Built-in study areas: ``key -> (name, polygon vertices, description)``.
#: Polygons are deliberately generous — they include the surrounding land and
#: open water, which the pipeline classifies away, so no intertidal fringe is
#: clipped off by an over-tight boundary.
SITES = {
    "vadehavet": {
        "name": "Vadehavet (Danish Wadden)",
        "description": "Danish Wadden Sea — our SECOND validation site, and a "
                       "HARDER one than the Tagus rather than merely a "
                       "different one: the modelled tidal range here is "
                       "1.90 m against the Tagus's 3.45 m, so there is little "
                       "over half the vertical signal to fit a sigmoid to, "
                       "while the surveyed flat spans 2.19 m — wider than the "
                       "tide itself, so its upper part cannot be reached by "
                       "any image. A method that holds up here is not winning "
                       "on an easy site. Reference: EMODnet "
                       "IB_Danske_Vadden_2020_64 "
                       "at 16.5 m, 292 km² of flat measured, every cell "
                       "between +0.34 and +2.53 m above datum and 99 % "
                       "non-integer — a survey OF the flats, not a boat "
                       "survey that stops at the low-water line. This "
                       "polygon is the 10 km cell holding 26.4 km² of it.",
        "polygon": [(8.4157, 55.2493), (8.5735, 55.2493),
                    (8.5735, 55.3391), (8.4157, 55.3391)],
        "tide_model": "EOT20",
    },
    "tejo": {
        "name": "Tejo (Tagus)",
        "description": "Tagus estuary (Portugal) — our VALIDATION site. Not a "
                       "study area in itself: it is the only Iberian estuary "
                       "we found with a float-precision survey of the flats "
                       "themselves (EMODnet IB_Tagus_2020_64, 22.6 m, 65.7 "
                       "km² measured between +0.5 and +3.4 m above LAT). "
                       "Atlantic mesotidal, so the regime is comparable to "
                       "the Cantabrian sites the method is applied to. This "
                       "polygon is the 10 km cell holding the most surveyed "
                       "flat: 23.1 km², thirteen times Villaviciosa's mapped "
                       "intertidal.",
        "polygon": [(-9.0347, 38.7808), (-8.9195, 38.7808),
                    (-8.9195, 38.8707), (-9.0347, 38.8707)],
        "tide_model": "EOT20",
    },
    "santander": {
        "name": "Bahía de Santander",
        "description": "Bahía de Santander (Cantabria) — a large sheltered bay "
                       "with extensive sand and mud flats, a marsh system at "
                       "its head, and a working port. The mixture of natural "
                       "flats and hard infrastructure in one scene makes it a "
                       "good place to compare water detectors against each "
                       "other: the failure modes of SCL and of an index "
                       "diverge exactly where the substrate is unusual.",
        "polygon": [(-3.880, 43.360), (-3.680, 43.360),
                    (-3.680, 43.480), (-3.880, 43.480)],
        "tide_model": "EOT20",
    },
    "villaviciosa": {
        "name": "Villaviciosa",
        "description": "Ría de Villaviciosa (Asturias) — mesotidal estuary, "
                       "sandy flats and marsh; our reference site, validated "
                       "by an RTK GNSS field survey (361 fixed solutions at "
                       "1.3 cm, at the lowest spring tide of the month). The "
                       "IGN LiDAR that covers it is NOT usable on the flat: "
                       "across the intertidal mask it takes seven distinct "
                       "int16 values and 83 % of it sits at exactly +2.00 m, "
                       "the water surface at flight time. That is why the "
                       "field campaign exists.",
        "polygon": [(-5.455, 43.478), (-5.425, 43.470), (-5.385, 43.482),
                    (-5.375, 43.520), (-5.385, 43.545), (-5.425, 43.548),
                    (-5.455, 43.520)],
        "tide_model": "EOT20",
    },
    "santona": {
        "name": "Santoña",
        "description": "Marismas de Santoña (Cantabria) — the largest marsh "
                       "system on the north coast; wide flats, strong tidal "
                       "range, a Ramsar site.",
        "polygon": [(-3.50, 43.40), (-3.42, 43.40), (-3.42, 43.46),
                    (-3.50, 43.46)],
        "tide_model": "EOT20",
    },
    "foz": {
        "name": "Foz",
        "description": "Ría de Foz (Galicia) — small, sheltered estuary; the "
                       "original pilot site of the project.",
        "polygon": [(-7.265, 43.545), (-7.235, 43.545), (-7.235, 43.585),
                    (-7.265, 43.585)],
        "tide_model": "EOT20",
    },
    "arousa": {
        "name": "Arousa",
        "description": "Ría de Arousa (Rías Baixas) — the widest Galician "
                       "ría: open, Bijagós-like geometry with large flats "
                       "at Carril-Rianxo and Cambados.",
        "polygon": [(-9.06, 42.44), (-8.64, 42.44), (-8.64, 42.76),
                    (-9.06, 42.76)],
        "tide_model": "EOT20",
    },
    "urdaibai": {
        "name": "Urdaibai",
        "description": "Urdaibai / Mundaka (Basque Country) — UNESCO "
                       "biosphere reserve with extensive tidal flats.",
        "polygon": [(-2.72, 43.35), (-2.66, 43.35), (-2.66, 43.42),
                    (-2.72, 43.42)],
        "tide_model": "EOT20",
    },
    "escalda": {
        "name": "Escalda (Westerschelde)",
        "description": "Westerschelde (Netherlands), Vlissingen to Terneuzen "
                       "-- the project's gauge-instrumented site. Three IOC "
                       "tide gauges sit inside the box: Vlissingen (vlis) and "
                       "Breskens (brsk) at the mouth, Terneuzen (trnz) 15 km "
                       "up the axis. That is what a tide-adjustment method "
                       "needs to be judged on: the boundary comes from the "
                       "mouth, the interior clock is measured from imagery, "
                       "and the inner gauge, never used in the fit, says how "
                       "close the estimate got. Cube at 20 m, 2023-2025.",
        "polygon": [(3.55, 51.33), (3.85, 51.33), (3.85, 51.46),
                    (3.55, 51.46)],
        "tide_model": "EOT20",
    },
    "ems": {
        "name": "Ems-Dollard",
        "description": "Ems estuary and the Dollard (NL/DE) -- Wadden Sea "
                       "flats with two IOC gauges on the axis: Borkum (bork) "
                       "at the mouth, Delfzijl (delf) 28 km up the estuary. "
                       "Gauge-judged site for the tide-adjustment comparison; "
                       "cube at 20 m, 2023-2025.",
        "polygon": [(6.80, 53.22), (7.30, 53.22), (7.30, 53.50),
                    (6.80, 53.50)],
        "tide_model": "EOT20",
    },
    "wadden": {
        "name": "Wadden (Vlie-Harlingen)",
        "description": "Dutch Wadden Sea behind the Vlie inlet -- the flats "
                       "between Terschelling Noordzee (ters, sea side) and "
                       "Harlingen (harl, 28 km in, behind the flats). "
                       "Gauge-judged site; cube at 20 m, 2023-2025.",
        "polygon": [(5.15, 53.15), (5.55, 53.15), (5.55, 53.45),
                    (5.15, 53.45)],
        "tide_model": "EOT20",
    },
    "ferrol": {
        "name": "Ferrol",
        "description": "Ria de Ferrol (Galicia) -- a deep Cantabrian-type "
                       "ria with small inner flats and the only Spanish IOC "
                       "gauge pair on one axis: Ferrol1 (fer1, outer) and "
                       "Ferrol2 (fer2, inner, 6.4 km). Gauge-judged site; "
                       "cube at 10 m, 2023-2025.",
        "polygon": [(-8.36, 43.44), (-8.15, 43.44), (-8.15, 43.52),
                    (-8.36, 43.52)],
        "tide_model": "EOT20",
    },
}


def get(key, name=None):
    """Return a registered site as an :class:`~pyintertidal.aoi.AOI`.

    >>> aoi = sites.get("villaviciosa")
    """
    key = str(key).lower().replace(" ", "").replace("ñ", "n")
    if key not in SITES:
        raise KeyError(f"unknown site {key!r}; available: "
                       f"{', '.join(sorted(SITES))}")
    entry = SITES[key]
    return AOI.from_polygon(entry["polygon"], name=name or entry["name"])


def tide_model(key, default="EOT20"):
    """The tide model a site declares, or ``default``.

    Sites carry this so a study area that needs a particular model — a site
    where the default's grid has no ocean cell nearby, say — says so once here
    instead of in every notebook that uses it. Check the distance with
    :func:`pyintertidal.tidecheck.nearest_ocean_km` before overriding.
    """
    key = str(key).lower().replace(" ", "").replace("ñ", "n")
    if key not in SITES:
        raise KeyError(f"unknown site {key!r}; available: "
                       f"{', '.join(sorted(SITES))}")
    return SITES[key].get("tide_model") or default


def register(key, name, polygon, description="", tide_model="EOT20"):
    """Add a site to the registry for this session.

    Persist it by adding an entry to :data:`SITES` in this file — that is how
    a study area becomes reusable across notebooks and by other people.
    """
    SITES[str(key).lower()] = {"name": name, "polygon": list(polygon),
                               "description": description,
                               "tide_model": tide_model}
    return get(key)


def describe(key=None):
    """Print the registry (or one site) with descriptions and areas."""
    keys = [key] if key else sorted(SITES)
    for k in keys:
        entry = SITES[str(k).lower()]
        aoi = get(k)
        print(f"{entry['name']} ({k}) — {aoi.area_km2:.1f} km²")
        print(f"  {entry['description']}")
    return keys


def list_sites():
    """Registered site keys."""
    return sorted(SITES)
