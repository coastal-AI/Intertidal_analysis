"""Find gauge pairs worldwide, to learn from what length an estuary matters.

Two measurements exist so far: 85 km up the Guadalquivir the estuarine
transfer cuts the error by 76 %, and 6 km into the ria de Ferrol it buys
8.8 mm. Two points do not make a curve, and the question they leave open is
the one that decides whether anyone needs this correction: at what separation
does an estuary start doing something worth modelling?

Answering it needs pairs of gauges spanning the range in between, and there is
no reason to restrict the search to Spain. The IOC facility publishes stations
worldwide.

The design avoids having to decide in advance which pairs are estuarine. Every
pair within a plausible distance gets measured, and the two populations
separate themselves: a pair on open coast has nothing between it, so it must
return a gain of one and a lag near zero, and those pairs are the control. A
pair up an estuary returns amplification and a real lag. Nobody has to
classify anything by hand, and the control is not a separate experiment but
the same one.

This step only DISCOVERS pairs — no series are downloaded. At roughly 136 000
records per station per three months, deciding what to fetch before fetching
it is worth a few minutes.
"""
import json
import os
import sys
import urllib.request
from collections import defaultdict

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
BASE = "http://www.ioc-sealevelmonitoring.org/service.php"
MIN_KM, MAX_KM = 2.0, 130.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def main():
    cache = os.path.join(SC, "ioc_stations.json")
    if os.path.exists(cache):
        stations = json.load(open(cache))
    else:
        url = BASE + "?query=stationlist&showall=all"
        req = urllib.request.Request(url,
                                     headers={"User-Agent": "pyintertidal"})
        with urllib.request.urlopen(req, timeout=120) as r:
            stations = json.loads(r.read().decode("utf-8", "replace"))
        json.dump(stations, open(cache, "w"))
    print(f"{len(stations)} filas en el servicio IOC")

    seen, st = set(), []
    for s in stations:
        code = s.get("Code") or s.get("code")
        try:
            lat, lon = float(s.get("lat")), float(s.get("lon"))
        except (TypeError, ValueError):
            continue
        if not code or code in seen:
            continue
        seen.add(code)
        st.append({"code": code, "lat": lat, "lon": lon,
                   "name": (s.get("Location") or s.get("location")
                            or "").strip(),
                   "country": s.get("Country") or s.get("country") or ""})
    print(f"{len(st)} estaciones distintas en el mundo\n")

    lat = np.array([x["lat"] for x in st])
    lon = np.array([x["lon"] for x in st])
    pairs = []
    for i in range(len(st)):
        d = haversine(lat[i], lon[i], lat[i + 1:], lon[i + 1:])
        for off, dist in enumerate(d):
            if MIN_KM < dist < MAX_KM:
                pairs.append((float(dist), st[i], st[i + 1 + off]))
    print(f"{len(pairs)} parejas entre {MIN_KM:.0f} y {MAX_KM:.0f} km\n")

    # bucket by distance so the final selection spans the range rather than
    # piling up wherever gauges happen to be dense
    edges = [2, 5, 10, 20, 35, 60, 90, 130]
    buckets = defaultdict(list)
    for d, a, b in pairs:
        for lo, hi in zip(edges, edges[1:]):
            if lo <= d < hi:
                buckets[(lo, hi)].append((d, a, b))
                break

    print(f"{'tramo (km)':14s} {'parejas':>8s}   ejemplos")
    print("-" * 78)
    for lo, hi in zip(edges, edges[1:]):
        b = sorted(buckets[(lo, hi)], key=lambda x: x[0])
        ex = "; ".join(f"{x[1]['name'][:14]}-{x[2]['name'][:14]}"
                       for x in b[:2])
        print(f"{f'{lo}-{hi}':14s} {len(b):8d}   {ex[:52]}")

    # a shortlist: countries with dense networks and famous estuaries first
    PREF = ("NLD", "BEL", "DEU", "GBR", "FRA", "PRT", "ESP", "USA", "CAN",
            "AUS", "NZL", "IRL", "ARG", "BRA")
    print(f"\n\nCANDIDATAS por tramo, priorizando redes densas")
    print(f"{'km':>6s}  {'A':22s} {'B':22s} {'pais':5s}")
    print("-" * 62)
    short = []
    for lo, hi in zip(edges, edges[1:]):
        b = sorted(buckets[(lo, hi)], key=lambda x: (
            x[1]["country"] not in PREF, x[0]))
        for d, a, c in b[:4]:
            print(f"{d:6.1f}  {a['name'][:22]:22s} {c['name'][:22]:22s} "
                  f"{str(a['country'])[:5]:5s}")
            short.append({"km": d, "a": a["code"], "b": c["code"],
                          "name_a": a["name"], "name_b": c["name"],
                          "country": a["country"]})
    json.dump(short, open(os.path.join(SC, "pares_mundo.json"), "w"), indent=1)
    print(f"\n{len(short)} candidatas guardadas en pares_mundo.json")
    print("\nNinguna esta clasificada como estuario o costa abierta: eso lo")
    print("dira la propia medida, y las de costa abierta son el control.")


if __name__ == "__main__":
    main()
