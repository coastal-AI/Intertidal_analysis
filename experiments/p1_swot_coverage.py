"""FASE P: does SWOT fly over Villaviciosa, and how often?

SWOT (NASA/CNES) measures water surface height directly. If its swaths cross
the ria, its Level-2 river/lake products would give MEASURED water levels to
contrast the operator against — an external truth with no tide gauge.

Requires the user's Earthdata credentials in ~/.netrc (never in the repo).
Searches the two relevant HR collections over the ria's bbox and reports
granule counts and dates. This is a COVERAGE census, not a download: what
exists, how often, over what period. Downloading and reading actual heights
is the next step, only worthwhile if coverage is non-trivial.

Run:  python -m experiments.p1_swot_coverage
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join("results", "p1_swot")
# bbox of the ria (the confirmed one; the enlarged download AOI does not
# change this)
BBOX = (-5.47, 43.46, -5.33, 43.56)      # W, S, E, N
COLLECTIONS = [
    "SWOT_L2_HR_Raster_2.0",             # 100/250 m height grid
    "SWOT_L2_HR_PIXC_2.0",               # point cloud (pixel cloud)
]


def main():
    from pyintertidal.net import use_system_certificates
    use_system_certificates()
    import earthaccess

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise SystemExit("Earthdata: login via ~/.netrc failed — check "
                         "username/password")
    report = {"bbox": BBOX, "colecciones": {}}
    for short_name in COLLECTIONS:
        try:
            grans = earthaccess.search_data(
                short_name=short_name, bounding_box=BBOX,
                temporal=("2023-01-01", "2026-08-18"))
        except Exception as e:
            report["colecciones"][short_name] = {"error": str(e)}
            continue
        dates = sorted({str(g["umm"]["TemporalExtent"]["RangeDateTime"]
                            ["BeginningDateTime"])[:10] for g in grans})
        report["colecciones"][short_name] = {
            "n_granulos": len(grans),
            "primera_fecha": dates[0] if dates else None,
            "ultima_fecha": dates[-1] if dates else None,
            "n_dias_distintos": len(dates),
            "fechas_muestra": dates[:10],
        }
        print(f"{short_name}: {len(grans)} granules, "
              f"{len(dates)} distinct days "
              f"({dates[0] if dates else '-'}..{dates[-1] if dates else '-'})",
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
