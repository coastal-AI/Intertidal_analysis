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
    "villaviciosa": {
        "name": "Villaviciosa",
        "description": "Ría de Villaviciosa (Asturias) — mesotidal estuary, "
                       "sandy flats and marsh; our reference site, with an "
                       "IGN LiDAR survey and a field transect at Misiego.",
        "polygon": [(-5.455, 43.478), (-5.425, 43.470), (-5.385, 43.482),
                    (-5.375, 43.520), (-5.385, 43.545), (-5.425, 43.548),
                    (-5.455, 43.520)],
        "tide_model": "GOT4.10",
    },
    "santona": {
        "name": "Santoña",
        "description": "Marismas de Santoña (Cantabria) — the largest marsh "
                       "system on the north coast; wide flats, strong tidal "
                       "range, a Ramsar site.",
        "polygon": [(-3.50, 43.40), (-3.42, 43.40), (-3.42, 43.46),
                    (-3.50, 43.46)],
        "tide_model": "GOT4.10",
    },
    "foz": {
        "name": "Foz",
        "description": "Ría de Foz (Galicia) — small, sheltered estuary; the "
                       "original pilot site of the project.",
        "polygon": [(-7.265, 43.545), (-7.235, 43.545), (-7.235, 43.585),
                    (-7.265, 43.585)],
        "tide_model": "GOT4.10",
    },
    "urdaibai": {
        "name": "Urdaibai",
        "description": "Urdaibai / Mundaka (Basque Country) — UNESCO "
                       "biosphere reserve with extensive tidal flats.",
        "polygon": [(-2.72, 43.35), (-2.66, 43.35), (-2.66, 43.42),
                    (-2.72, 43.42)],
        "tide_model": "GOT4.10",
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


def register(key, name, polygon, description="", tide_model="GOT4.10"):
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
