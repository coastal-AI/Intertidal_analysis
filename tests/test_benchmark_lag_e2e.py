# -*- coding: utf-8 -*-
"""Validation v2, Stage L end to end (integration; green once packages A-D
have landed): the small v2 package of ``tests/benchmark_zarr_site.py`` with
one synthetic in-estuary judge (a gauge reading the boundary tide 30 min
late) and its per-window lag truth, the lag plan of the site at W3 plus a
W5 row (its years are not in the package), the queue with the REAL lag
adapters, and ``scoring.score_lag(pre_lock=True)``.

Asserts: every planned W3 unit leaves a product (or a ledger status), one
lag_points row per judge x product x truth at W3 (status rows included),
the not_applicable plan rows read 'not applicable' / 'NOT AVAILABLE', the
W5 row reads 'NOT AVAILABLE: store years ...', and every file carries the
PRE-LOCK banner.

Run:  python -X utf8 -m pytest -q tests/test_benchmark_lag_e2e.py
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark import config, designs, judges, package, settings, truth
from experiments.benchmark import instance as I
from experiments.benchmark import queue_v2 as Q

SITE = "escalda"
P0 = settings.P0.id
LAG_MIN = 30
CONS = ("m2", "s2", "o1", "k1", "n2", "m4")
#: the adapter modules the end-to-end run needs (skip while one is not built)
NEEDED = ("experiments.benchmark.methods.elev_marea_pair", "experiments.benchmark.lags.aggregate",
          "experiments.benchmark.lags.gm20", "experiments.benchmark.lags.hd26",
          "experiments.benchmark.lags.ppl", "experiments.benchmark.lags.bt26",
          "experiments.benchmark.lags.models", "experiments.benchmark.lags.regional",
          "experiments.benchmark.lags.physics", "experiments.benchmark.lags.references",
          "experiments.benchmark.lags.marea_family", "experiments.benchmark.scoring")


def _require_packages():
    import importlib
    for m in NEEDED:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            if e.name == m:
                pytest.skip(f"{m} is not built yet (packages A-D)")
            raise


def _read_scores(path):
    """A score CSV; its leading '# PRE-LOCK ...' banner lines are skipped."""
    with open(path, encoding="utf-8") as f:
        n = 0
        for line in f:
            if not line.startswith("#"):
                break
            n += 1
    return pd.read_csv(path, skiprows=n)


def _gauge_csv(path, provider):
    t = pd.date_range("2023-01-01 00:00", "2025-12-31 23:50", freq="10min").as_unit("ns")
    v = 0.25 + provider.value((t - pd.Timedelta(minutes=LAG_MIN)).asi8)
    v = v + np.random.default_rng(3).normal(0.0, 0.005, len(t))
    stamps = np.char.replace(np.datetime_as_string(t.values, unit="s"), "T", " ")
    pd.DataFrame({"time_utc": stamps, "level_m": v}).to_csv(path, index=False,
                                                              lineterminator="\n")


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    _require_packages()
    from tests import benchmark_zarr_site as zs
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("v2_e2e")
    s = zs.make_site(base, mp)
    inp, site2, prov = s.inp, s.site2, s.provider
    # the judge: an RWS-like 10-min gauge (a documented convention), 30 min late
    st = json.load(open(inp["stations"], encoding="utf-8"))
    st[0].update(provider="RWS", sampling_min=10)
    json.dump(st, open(inp["stations"], "w", encoding="utf-8"))
    _gauge_csv(os.path.join(inp["root"], "st_flat.csv"), prov)
    mp.setitem(config.JUDGES, SITE, ("st_flat",))
    mp.setattr(config, "STATIONS_JSON", inp["stations"])
    mp.setattr(config, "ESTUARY_TRUTH", inp["root"])
    package.prepare_v2(site2, steps=("series",), provider=prov)
    s.pkg.close()
    pkg = package.V2Package(site2)
    pkg._provider = prov
    table = judges.build(pkg, path=f"{site2.truth_dir}/judges.json")
    truth.build_lag_truth_v2(pkg, windows=("W3",), table=table, constituents=CONS)
    # the plan: the lag design of the site at W3, plus a W5 row (years missing)
    rows = [r for r in designs.plan_rows("lag", sites=(SITE,)) if r["window"] == "W3"]
    rows.append(Q.make_row("lag", SITE, "W5", P0, "L17"))
    os.makedirs(os.path.join(site2.root, "plans"), exist_ok=True)
    with open(os.path.join(site2.root, "plans", "lag.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(designs.plan_csv(rows))
    opts = Q.QueueOptions(root=site2.root, plans=("lag",), site2_for=lambda x: site2,
                          provider_for=lambda x: prov, ram_total_gb=1000.0, threads=1)
    counts = Q.run(opts)
    series = {"st_flat": judges.read_series(table[0])}
    out = __import__("experiments.benchmark.scoring", fromlist=["x"]).score_lag(
        site2.root, sites=[SITE], windows=["W3", "W5"], pre_lock=True, B=20,
        packages={SITE: pkg}, series_by_code=series)
    yield {"root": site2.root, "rows": rows, "counts": counts, "out": out, "pkg": pkg,
           "table": table}
    pkg.close()
    mp.undo()


def test_every_planned_w3_unit_left_a_product_or_a_status(e2e):
    root = e2e["root"]
    for r in e2e["rows"]:
        n = I.norm_row(r)
        if n["status"] != "planned":
            continue
        unit = Q.unit_of(n)
        runs = [Q.read_run(Q.product_dir(root, unit, p)) for p in n["products"]]
        led = Q.read_ledger(root, n["unit_id"])
        if n["window"] == "W5":
            assert led["status"] == "unavailable" and "2021" in led["reason"]
            continue
        assert all(x is not None for x in runs) or led is not None, n["method"]
    assert e2e["counts"].get("ran", 0) >= 10


def test_one_point_row_per_judge_and_product(e2e):
    pts = _read_scores(e2e["out"]["points"])
    w3 = pts[(pts["window"] == "W3") & (pts["judge"] == "st_flat")]
    want = set()
    for r in e2e["rows"]:
        n = I.norm_row(r)
        if n["window"] == "W3":            # the lag products (E0a/E0b are scored in V3)
            want |= {p for p in n["products"] if p not in Q.ELEV_PRODUCTS}
    got = set(w3["product"].str.split("@").str[0])
    assert want <= got, sorted(want - got)
    for p in want:
        sub = w3[(w3["product"] == p) & (w3["truth"] == "a")]
        assert len(sub) == 1, (p, len(sub))
    na = w3[w3["product"] == "L16"]
    assert not na.empty and str(na["status"].iloc[0]).lower().startswith(("not applicable",
                                                                          "not available"))


def test_window_without_its_years_reads_not_available(e2e):
    pts = _read_scores(e2e["out"]["points"])
    w5 = pts[pts["window"] == "W5"]
    assert not w5.empty and w5["status"].astype(str).str.contains("NOT AVAILABLE").all()


def test_pre_lock_banner_and_the_table(e2e):
    md = open(e2e["out"]["table_md"], encoding="utf-8").read()
    assert "PRE-LOCK" in md and "st_flat" in md
    for k in ("points", "summary"):
        assert open(e2e["out"][k], encoding="utf-8").readline().startswith("# PRE-LOCK")
    l01 = _read_scores(e2e["out"]["points"])
    row = l01[(l01["product"] == "L17") & (l01["truth"] == "a") & (l01["window"] == "W3")]
    assert len(row) == 1
    tau_a = float(row["tau_true"].iloc[0])
    assert abs(tau_a - LAG_MIN) <= 2
