"""Find an estuary with two tide gauges: one at the mouth, one inside.

The transfer operator can only be believed if something independent confirms
it, and Villaviciosa has no gauge at all. But the test does not need a field
campaign: an estuary with a gauge near its mouth and another upstream gives
the operator its own examination. Feed it the outer record, ask for the inner
one, compare against a series it never saw.

The IOC Sea Level Monitoring facility publishes many stations openly, with no
registration. This lists what exists around Iberia and works out which pairs
sit in the same estuary and how far apart along the water they are.

What makes a good pair, in order of importance:

* both inside the SAME tidal system, one near the open sea and one well
  upstream — otherwise there is no propagation to measure;
* far enough apart that the estuary has done something measurable. At a
  typical 1 %/km amplification, 10 km changes M2 by a tenth, which a gauge
  resolves easily and a 2 km separation would not;
* overlapping records, at a sampling interval short enough to resolve the
  semidiurnal band. Anything hourly or better will do.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
BASE = "http://www.ioc-sealevelmonitoring.org/service.php"
# generous box: Iberia plus the French Basque coast
BOX = dict(lon=(-10.5, 3.5), lat=(35.5, 44.5))


def fetch(query, **kw):
    url = BASE + "?query=" + query
    for k, v in kw.items():
        url += f"&{k}={v}"
    req = urllib.request.Request(url, headers={"User-Agent": "pyintertidal"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def haversine(a, b):
    R = 6371.0
    p1, p2 = np.radians(a[0]), np.radians(b[0])
    dp = p2 - p1
    dl = np.radians(b[1] - a[1])
    h = (np.sin(dp / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(h))


def main():
    cache = os.path.join(SC, "ioc_stations.json")
    if os.path.exists(cache):
        stations = json.load(open(cache))
        print("lista de estaciones leida de cache")
    else:
        stations = fetch("stationlist", showall="all")
        json.dump(stations, open(cache, "w"))
    print(f"{len(stations)} estaciones en el servicio IOC\n")

    local = []
    for s in stations:
        try:
            lat, lon = float(s.get("lat")), float(s.get("lon"))
        except (TypeError, ValueError):
            continue
        if not (BOX["lat"][0] <= lat <= BOX["lat"][1]
                and BOX["lon"][0] <= lon <= BOX["lon"][1]):
            continue
        local.append({"code": s.get("Code") or s.get("code"),
                      "name": (s.get("Location") or s.get("location")
                               or "").strip(),
                      "country": s.get("Country") or s.get("country"),
                      "lat": lat, "lon": lon})
    # one entry per code (the service lists a row per sensor)
    seen, uniq = set(), []
    for s in local:
        if s["code"] and s["code"] not in seen:
            seen.add(s["code"])
            uniq.append(s)
    print(f"{len(uniq)} estaciones distintas en Iberia\n")
    print(f"{'codigo':10s} {'nombre':30s} {'pais':6s} {'lat':>8s} {'lon':>9s}")
    print("-" * 68)
    for s in sorted(uniq, key=lambda x: (x["country"] or "", x["name"])):
        print(f"{s['code']:10s} {s['name'][:30]:30s} "
              f"{str(s['country'])[:6]:6s} {s['lat']:8.3f} {s['lon']:9.3f}")

    # ── candidate pairs: close in space, so plausibly the same system ────
    print(f"\n\nPAREJAS A MENOS DE 60 km (candidatas a boca + interior)")
    print(f"{'estacion A':22s} {'estacion B':22s} {'km':>6s}")
    print("-" * 54)
    pairs = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            d = haversine((a["lat"], a["lon"]), (b["lat"], b["lon"]))
            if 3.0 < d < 60.0:
                pairs.append((d, a, b))
    for d, a, b in sorted(pairs)[:40]:
        print(f"{a['name'][:22]:22s} {b['name'][:22]:22s} {d:6.1f}")
    print(f"\n{len(pairs)} parejas en ese rango")
    print("\nLa distancia en linea recta NO es la distancia por el canal: hay")
    print("que mirar cuales estan de verdad en el mismo estuario, y medir la")
    print("separacion siguiendo el agua.")

    json.dump(uniq, open(os.path.join(SC, "ioc_iberia.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
