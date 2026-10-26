"""Tests of experiments.benchmark.designs, the design tables and the plan (spec §5, §14).

They pin: every design builds with valid setting ids, windows, methods and
statuses; the effective-key dedupe (a detector-blind method collapses G1'
to 12 effective cells, §2.9); explicit not_applicable rows (LG at
Villaviciosa, §4.4); P0 is applicable to every Tier-1 method (§4.4 rule 3);
the OFAT and G1' cell counts, the matched-N draws (3, frame methods 1), the
K1' grid and its batch kinds; the cut list (each step's predicate, the
never-cut rows) and the ten-day projection, which applies the cuts in
order; the CSV writer (deterministic bytes, the index sha); the observation
keys the package counts. User decision of 2026-10-14: the catalogue times
are every unit's instants (no 'label instants' row, no G-time cut item or
stage); the rows of the error attribution exist (no delay and MAREA from
one E0 unit, no delay under the measured gauge level LG, seeds, halves,
calendar years); the CLI's optional 'diag time' ('gate time' an alias),
its resolutions (--res 20: legacy cubes, production boundary) and the
explicit --sensing-times. Reviews of 2026-10-14: LM and L16 rows are
planned only where a regional archive covers the whole window (NOT
AVAILABLE rows elsewhere; the Dutch LM rows at W1), the E0 'nodelay' unit
writes E0a only, the never-cut items are checked on the kept units, the
v2 block-size lattice is frozen, and the CLI guards (the sensing file's
site, even --block-rows, SIGTERM as Ctrl-C, the output root's .gitignore).
2026-10-15/16: check-env's inputs section (a missing, empty or cut-short
input of a Campaign 1 step is a FAIL with the step and what to copy, a
present one OK with its size or file count, a later milestone's a WARN; an
entry that cannot be checked does not stop the section), the judge CSVs and
Vaklodingen sheets named from this machine's stations.json and catalogue,
the boundary gauges against the union IOC cache, and the INPUTS table, which
names the code's own paths.

Run:  python -X utf8 -m pytest tests/test_benchmark_designs.py -q
"""
import csv
import hashlib
import io
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, designs as D, settings  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return {d: D.build(d) for d in D.DESIGNS}


def test_every_design_builds_valid_units(built):
    assert set(D.DESIGNS) == set(D.QUEUE_1A) | set(D.QUEUE_1B)
    for name, units in built.items():
        assert units, name
        for u in units:
            assert u.design == name
            settings.Setting.parse(u.setting)
            settings.window(u.window)
            assert u.method in D.METHODS and u.site in config.SITES
            assert u.status in D.STATUSES
            assert u.tier in ("1A'", "1B")
            row = u.row(1)
            assert tuple(row) == D.CSV_COLUMNS
            assert u.est_hours >= 0 and u.est_ram_gb > 0
            if u.status != "planned":
                assert u.est_hours == 0
    assert all(u.tier == "1A'" for d in D.QUEUE_1A for u in built[d])
    assert all(u.tier == "1B" for d in D.QUEUE_1B for u in built[d])


def test_effective_key_dedupe_in_g1(built):
    g1 = built["g1"]
    cells = {(u.setting, u.window) for u in g1}
    assert len(cells) == 24
    for site in ("escalda", "ferrol"):
        e8 = [u for u in g1 if u.method == "E8" and u.site == site]
        if site == "ferrol":
            assert not e8                       # elevation units: not at Ferrol
            continue
        assert len(e8) == 24
        assert sum(u.status == "planned" for u in e8) == 12      # D is not read
        assert sum(u.status == "shared" for u in e8) == 12
        e0 = [u for u in g1 if u.method == "E0" and u.site == site]
        assert all(u.status == "planned" for u in e0)             # reads the domain
    assert not any(u.method == "E3" for u in g1)                  # Granadeiro not in G1'
    # every shared row names a planned unit with its effective key
    ids = {u.unit_id: u for u in g1 if u.status == "planned"}
    for u in g1:
        if u.status == "shared":
            ref = ids[u.note.split()[-1]]
            assert ref.dedupe_key == u.dedupe_key


def test_not_applicable_rows_are_explicit(built):
    lv = built["levels"]
    lg_v = [u for u in lv if u.site == "villaviciosa" and u.setting.endswith("@LG")]
    assert lg_v and all(u.status == "not_applicable" for u in lg_v)
    assert all("not available at villaviciosa" in u.note for u in lg_v)
    # no delay runs under LT/LM/LG as E0+nodelay, MAREA never does; the only
    # other E0 rows are the gauge-lag attribution rows at P0 under LB (2026-10-15)
    e0 = [u for u in lv if u.method == "E0" and u.status == "planned"]
    assert e0 and all(u.variant == "nodelay" for u in e0 if u.variant != "gaugelag")
    assert all(u.setting == settings.P0.id and u.products == ("E0g",)
               for u in e0 if u.variant == "gaugelag")
    assert not any(u.method == "E3" for u in lv)
    ok, why = D.applicable("L02", "C1.floor0.ndwi.D1@LT", "escalda")
    assert not ok and "level" in why


def test_p0_is_applicable_to_every_tier1_method():
    for m in D.METHODS:
        for site in D.ELEV_SITES:
            assert D.applicable(m, settings.P0.id, site)[0], (m, site)


def test_ofat_cells_and_matched_draws(built):
    of = built["ofat"]
    cells = {u.setting for u in of if u.family in ("rand", "rule") and not u.variant}
    assert len(cells - {settings.P0.id}) == 16
    m = [u for u in of if u.family.startswith("matched-")]
    assert {u.family for u in m} == {"matched-C2", "matched-C6", "matched-C8"}
    for u in m:
        draws = {x.rep for x in m if (x.site, x.method, x.family) == (u.site, u.method, u.family)}
        assert draws == ({0} if D.METHODS[u.method].frame else {0, 1, 2})
        assert u.setting == settings.P0.id and u.N == f"rule:{u.family[8:]}"
    scene = [u for u in of if settings.Setting.parse(u.setting).scene_rule]
    assert scene and all(u.family == "rule" for u in scene)
    c0 = [u for u in of if u.setting.startswith("C0.") and u.status == "planned"]
    assert c0 and all("S0 only" in u.note for u in c0)
    vb = [u for u in of if u.variant in ("bnd-gijon", "bnd-cantabrian")]
    assert vb and {u.site for u in vb} == {"villaviciosa"}


def test_k1_grid_and_kinds(built):
    k1 = built["curves-k1"]
    T = D.window_T("W3")
    reps = {}
    for u in k1:
        reps.setdefault(u.N, set()).add(u.rep)
        assert u.kind == settings.subset_kind(T, u.N, len(reps[u.N]) and
                                              dict(((20, 5), (45, 5), (90, 3), (180, 2),
                                                    (330, 2)))[u.N])
    assert {n: len(r) for n, r in reps.items()} == {20: 5, 45: 5, 90: 3, 180: 2, 330: 2}
    assert {u.kind for u in k1 if u.N == 330} == {"nested"}
    assert {u.kind for u in k1 if u.N == 20} == {"disjoint"}
    assert not any(u.method == "E3" for u in k1)


def test_cut_steps_and_never_cut(built):
    def steps(name):
        return {u.cut_step for u in built[name]}

    assert {u.cut_step for u in built["curves-k1"] if u.N == 20 and u.rep >= 3} == {1}
    assert {u.cut_step for u in built["curves-k1"] if u.N == 330 and u.rep == 0} == {0}
    assert {u.cut_step for u in built["ofat"] if u.family.startswith("matched")
            and u.rep >= 1} == {2}
    assert {u.cut_step for u in built["g1"]
            if settings.Setting.parse(u.setting).detect == "D3"} == {3}
    assert {(u.site, u.cut_step) for u in built["granadeiro-nir"]} == {
        ("villaviciosa", 4), ("escalda", 0)}
    assert {u.cut_step for u in built["ofat"]
            if settings.Setting.parse(u.setting).index in ("awei_sh", "ndwi20")} == {5}
    assert {u.cut_step for u in built["levels"] if u.site == "villaviciosa"
            and settings.Setting.parse(u.setting).level in ("LT", "LM")} == {6}
    assert {u.site for u in built["tuning"] if u.cut_step == 7} == {"villaviciosa"}
    assert {u.cut_step for u in built["ofat"] if u.variant == "bnd-gijon"} == {8}
    # never cut: P0 base, the as-published rows, the native cells, G-base
    for name in ("base", "published", "native", "granadeiro-base", "lag", "years"):
        assert steps(name) == {0}, name


def test_projection_applies_the_cuts_in_order():
    p = D.projection()
    assert p["fits"] and p["applied"] == 0
    s0 = p["steps"][0]
    assert 7.0 < s0["days"] < 11.0            # spec §14.2: about 9.5 days
    assert s0["days_with_contingency"] > s0["days"]
    tight = D.projection(cap_days=s0["days"] - 0.3)
    days = [s["days"] for s in tight["steps"]]
    assert len(days) >= 2 and all(a > b for a, b in zip(days, days[1:]))
    assert tight["applied"] == len(days) - 1 and tight["fits"]
    assert [s["step"] for s in tight["steps"]] == list(range(len(days)))
    none = D.projection(cap_days=1.0)
    assert not none["fits"] and none["applied"] is None and len(none["steps"]) == 9
    faster = D.projection(wh_per_day=150.0)
    assert faster["steps"][0]["days"] < s0["days"]


def test_campaign_counts_each_effective_key_once():
    units = D.campaign_units(("ofat", "g1"))
    keys = [u.dedupe_key for u in units]
    assert len(keys) == len(set(keys))
    p0_e0 = [u for u in D.campaign_units(("lag", "base")) if u.method == "E0"
             and u.setting == settings.P0.id and u.window == "W3" and u.variant == "boot30"
             and u.family == "rand"]
    assert len(p0_e0) == 5 and {u.design for u in p0_e0} == {"lag"}


def test_plan_csv_is_deterministic_and_indexed(tmp_path):
    path, summ = D.write_plan("native", root=str(tmp_path), sites=("escalda", "ferrol"))
    b1 = open(path, "rb").read()
    path2, summ2 = D.write_plan("native", root=str(tmp_path), sites=("escalda", "ferrol"))
    assert open(path2, "rb").read() == b1 and b"\r\n" not in b1
    assert summ["sha256"] == hashlib.sha256(b1).hexdigest() == summ2["sha256"]
    rows = list(csv.DictReader(io.StringIO(b1.decode("utf-8"))))
    assert tuple(rows[0]) == D.CSV_COLUMNS
    assert {r["site"] for r in rows} == {"escalda", "ferrol"}
    idx = json.load(open(os.path.join(str(tmp_path), "plans", "index.json"), encoding="utf-8"))
    assert idx["plans"]["native"]["sha256"] == summ["sha256"]
    assert idx["plans"]["native"]["rows"] == len(rows) == summ["n_rows"]
    with pytest.raises(KeyError):
        D.build("nope")
    with pytest.raises(ValueError):
        D.build("native", sites=("atlantis",))


def test_observation_keys_map_scene_rules_to_their_pixel_rule():
    keys = D.observation_keys()
    assert settings.P0.id in keys
    assert not any(settings.Setting.parse(k).scene_rule for k in keys)
    assert not any("@" in k for k in keys)
    assert "C11.floor0.ndwi.D1" in keys and "C3.floor0.ndwi.D1" in keys
    assert "C1.floor0.mndwi.D2" in keys and "C1.floor0.nir.D1" in keys
    sids = D.planned_settings()
    assert "C2.floor0.ndwi.D1" in sids and "C1.floor0.ndwi.D1@LM" in sids


def test_catalogue_instants_are_no_factor_nor_row(built):
    """User decision of 2026-10-14: the catalogue times are every unit's
    instants; the sensing-time diagnostic is no row, no cut item, no fixed
    stage of the projection."""
    for name, units in built.items():
        for u in units:
            assert u.variant != "labels", (name, u)
            assert "G-time" not in u.note and "label instants" not in u.note, (name, u)
    assert not any("G-time" in x for x in D.NEVER_CUT)
    assert not any("G-time" in k for k in D.FIXED_DAYS)
    assert "time" not in settings.FACTORS and "instants" not in settings.FACTORS


def test_attribution_rows_exist_on_the_same_scenes_pixels_and_settings(built):
    """No lag (E0a), MAREA's band lags (E0b) and the measured gauge level
    (no delay under LG) at P0, W3, N = all, with the noise rows; LG only
    where the judges exist (not_applicable at Villaviciosa)."""
    att = D.attribution_rows()
    assert set(att) == set(D.ELEV_SITES)
    for site, r in att.items():
        assert r["present"]["E0a_E0b_LB"] and len(r["rows"]["E0a_E0b_LB"]) == 1
        assert len(r["rows"]["seeds"]) == 4 and len(r["rows"]["halves"]) == 2
        assert len(r["rows"]["years"]) == 3
        assert r["present"]["E0a_LG"] == (site in settings.LEVEL["LG"].sites)
    assert any("not available at villaviciosa" in m for m in att["villaviciosa"]["missing"])
    # E0a and E0b come from ONE unit (same scenes, pixels and setting)
    assert D.METHODS["E0"].products[:2] == ("E0a", "E0b")
    lg = [u for u in built["levels"] if u.method == "E0" and u.setting.endswith("@LG")
          and u.status == "planned"]
    assert {u.site for u in lg} == set(settings.LEVEL["LG"].sites)
    assert all(u.variant == "nodelay" and u.window == "W3" and u.N == "all" for u in lg)
    # the level factor does not enter the domain: the LG and LB rows share it
    base = settings.effective_key(settings.P0.id, "W3", D.METHODS["E0"].signature)
    for u in lg:
        assert u.effective_key.split("@")[0] == base.split("@")[0]
    assert [c[0] for c in D.ATTRIBUTION_CONTRASTS] == ["lag effect", "lag headroom",
                                                       "level error", "ignoring the lag",
                                                       "lag estimation", "band model"]


def test_costs_scale_with_window_frame_and_n():
    u = D.Unit(design="x", site="ems", window="W3", setting=settings.P0.id, method="E2")
    v = D.Unit(design="x", site="villaviciosa", window="W3", setting=settings.P0.id,
               method="E2")
    w = D.Unit(design="x", site="ems", window="W9", setting=settings.P0.id, method="E2")
    c = D.Unit(design="x", site="ems", window="W3", setting=settings.P0.id, method="E2", N=90)
    assert v.est_hours < u.est_hours < w.est_hours
    assert c.est_hours == pytest.approx(u.est_hours * 90 / D.window_T("W3"), rel=1e-3)
    assert D.Unit(design="x", site="ems", window="W3", setting=settings.P0.id,
                  method="L17").est_read_gb == 0.0
    g = D.Unit(design="x", site="ems", window="W3", setting=settings.P0.id, method="E3")
    assert g.est_hours > 5                      # Granadeiro: hours of its own pane


# ─────────────────────────────────────────────────────────────────────────────
#  the CLI: plan, and the later milestones' commands
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_plan_writes_the_plans_under_the_root(tmp_path, monkeypatch, capsys):
    from experiments import validation_v2 as cli
    monkeypatch.setenv("VALIDATION_V2_ROOT", str(tmp_path))
    rc = cli.main(["plan", "--design", "native,granadeiro-base", "--sites", "escalda"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Tier 1A' projection" in out and "never cut" in out
    for d in ("native", "granadeiro-base"):
        assert os.path.exists(os.path.join(str(tmp_path), "plans", f"{d}.csv"))
    log = os.path.join(str(tmp_path), "logs", "commands.jsonl")
    rec = json.loads(open(log, encoding="utf-8").read().splitlines()[-1])
    assert rec["command"] == "plan" and rec["exit"] == 0
    assert cli.main(["plan", "--design", "nope"]) == 2


#: milestone V2 replaced the V2 stubs (run, score --part lag, lock, the V2
#: gates, control --kind lag, timing, pilot attain): they now run their
#: command (tests/test_benchmark_cli_lag.py); the later milestones still exit 2
@pytest.mark.parametrize("argv,ms", [
    (["report"], "V6"), (["pilot", "units"], "V3"),
    (["score", "--part", "elev"], "V3"), (["score", "--part", "lag", "--blocks"], "V3"),
    (["gate", "nidem"], "V3"), (["control", "--kind", "elev"], "V3")])
def test_cli_later_commands_exit_with_their_milestone(argv, ms, capsys, tmp_path, monkeypatch):
    from experiments import validation_v2 as cli
    monkeypatch.setenv("VALIDATION_V2_ROOT", str(tmp_path))
    assert cli.main(argv) == 2
    out = capsys.readouterr().out
    assert f"available from milestone {ms}" in out
    assert not os.path.exists(os.path.join(str(tmp_path), "logs", "commands.jsonl"))


def test_cli_gate_time_is_only_an_alias_of_the_optional_diag(monkeypatch, capsys, tmp_path):
    """User decision of 2026-10-14: 'diag time' is optional and gates nothing;
    'gate time' only says so and runs it."""
    from experiments import validation_v2 as cli
    from experiments.benchmark import sensing_time as st
    monkeypatch.setenv("VALIDATION_V2_ROOT", str(tmp_path))
    seen = []
    monkeypatch.setattr(st, "main", lambda argv=None: seen.append(list(argv)) or 0)
    assert cli.main(["gate", "time", "--sites", "ems", "--no-fetch"]) == 0
    out = capsys.readouterr().out
    assert cli.GATE_TIME_NOTE in out and "OPTIONAL" in out
    assert seen[-1][:3] == ["diag", "--sites", "ems"] and "--no-fetch" in seen[-1]
    assert cli.main(["diag", "time", "--sites", "ems"]) == 0
    assert "--copy-to-cubes" not in seen[-1] and cli.DIAG_NOTE in capsys.readouterr().out
    assert "diag" in cli.build_parser().format_help()


def test_cli_resolutions_and_explicit_sensing_times(monkeypatch, tmp_path):
    """--res 20: the legacy netCDF cubes with the production boundary under
    results/validation_v2_20m; the catalogue times unless --sensing-times
    (a file, or the diagnostic's folder of <site>/sensing_times.json)."""
    from experiments import validation_v2 as cli
    monkeypatch.delenv("VALIDATION_V2_ROOT", raising=False)
    assert cli._root(10) == "results/validation_v2"
    assert cli._root(20) == "results/validation_v2_20m"
    monkeypatch.setenv("VALIDATION_V2_ROOT", str(tmp_path))
    assert cli._root(20) == str(tmp_path)
    assert cli._sites("all", 20) == ["escalda", "wadden", "ems"]
    with pytest.raises(ValueError, match="no 20 m input"):
        cli._sites("ferrol", 20)
    p = cli.build_parser()
    s2 = cli._site2("escalda", p.parse_args(["prepare", "escalda", "--res", "20"]))
    assert (s2.source, s2.boundary, s2.pixel_m, s2.instants) == ("netcdf", "production", 20.0,
                                                                 "catalogue")
    assert s2.bands == ("B03", "B08", "SCL") and s2.root == str(tmp_path)
    s2 = cli._site2("escalda", p.parse_args(["prepare", "escalda"]))
    assert (s2.source, s2.boundary, s2.pixel_m, s2.instants) == ("zarr", "union", 10.0,
                                                                 "catalogue")
    d = tmp_path / "sensing" / "escalda"
    d.mkdir(parents=True)
    (d / "sensing_times.json").write_text("{}", encoding="utf-8")
    a = p.parse_args(["prepare", "escalda", "--sensing-times", str(tmp_path / "sensing")])
    assert cli._site2("escalda", a).instants == os.path.join(str(tmp_path / "sensing"),
                                                             "escalda", "sensing_times.json")
    a = p.parse_args(["prepare", "ems", "--sensing-times", str(tmp_path / "sensing")])
    with pytest.raises(ValueError, match="no ems/sensing_times.json"):
        cli._site2("ems", a)


# ─────────────────────────────────────────────────────────────────────────────
#  the reviews of 2026-10-14: regional archives, never-cut check, CLI guards
# ─────────────────────────────────────────────────────────────────────────────

def test_lm_and_l16_are_planned_only_where_the_regional_archive_covers(built):
    """AMM15 starts 2024-08-04: at the Dutch sites LM and L16 run at W1 (=
    Y2025) and are NOT AVAILABLE rows elsewhere; the rías keep W3 (the IBI
    reanalysis, hourly means: the stamp convention must be measured)."""
    lm = [u for u in built["levels"] if settings.Setting.parse(u.setting).level == "LM"]
    for site in D.DUTCH_SITES:
        w3 = [u for u in lm if u.site == site and u.window == "W3"]
        w1 = [u for u in lm if u.site == site and u.window == "W1"]
        assert w3 and all(u.status == "not_applicable" and u.note.startswith("NOT AVAILABLE")
                          for u in w3)
        assert w1 and all(u.status == "planned" and "AMM15" in u.note for u in w1)
        assert {u.method for u in w1} == {u.method for u in w3}
    vv = [u for u in lm if u.site == "villaviciosa"]
    assert vv and all(u.window == "W3" and u.status == "planned" for u in vv)
    assert all("hourly MEANS" in u.note and "degenerate" in u.note for u in vv)
    l16 = {(u.site, u.window): u for u in built["lag"] if u.method == "L16"}
    for site in D.DUTCH_SITES:
        assert l16[(site, "W1")].status == "planned"
        for wid in ("W3", "W2", "Y2023", "Y2024"):
            assert l16[(site, wid)].status == "not_applicable"
            assert "NOT AVAILABLE" in l16[(site, wid)].note
    assert all(l16[(s, w)].status == "planned" for s in ("ferrol", "villaviciosa")
               for w in D.LAG_WINDOWS)
    ok, why = D.applicable("L16", D.P0, "ems", "W3")
    assert not ok and why.startswith("NOT AVAILABLE")
    assert D.applicable("L16", D.P0, "ems", "W1") == (True, "")
    assert D.applicable("L16", D.P0, "ems") == (True, "")        # no window: no archive check


def test_nodelay_units_write_no_delay_only():
    u = D.Unit(design="levels", site="escalda", window="W3", setting=D._sid(level="LG"),
               method="E0", variant="nodelay")
    assert u.products == ("E0a",) and u.row(1)["products"] == "E0a"
    b = D.Unit(design="base", site="escalda", window="W3", setting=D.P0, method="E0")
    assert b.products == tuple(D.METHODS["E0"].products) and "E0b" in b.products


def test_never_cut_items_are_checked_on_the_kept_units(built):
    units = D.campaign_units(D.QUEUE_1A)
    nc = D.never_cut_check(units)
    assert nc["LB-vs-LM at three or more sites"]["ok"]
    assert nc["LB-vs-LT at three or more sites"]["ok"]
    assert nc["the MAREA, NIDEM, DEA and Granadeiro gates"]["ok"] is None
    # without the W1 rows of the Dutch sites only Villaviciosa compares LB with LM
    no_w1 = [u for u in units if not (u.design == "levels" and u.window == "W1")]
    bad = D.never_cut_check(no_w1)["LB-vs-LM at three or more sites"]
    assert not bad["ok"] and "1 site(s)" in bad["detail"]
    p = D.projection()
    assert p["fits"] and p["never_cut_ok"] and p["within_cap"]


def test_frozen_block_size_rule_is_the_v2_lattice():
    v = config.FROZEN["inference"]["variogram"]
    assert list(v["lattice_m"]) == [250, 500, 1000, 2000, 4000, 8000, 16000]
    assert v["range_max_m"] == 16000.0 and "largest" in v["pooling"]
    assert "shared" in v["weights"]
    assert config.FROZEN["inference"]["boot"]["B_primary"] == 20000


def test_cli_guards_sensing_file_site_even_rows_and_signals(tmp_path, monkeypatch):
    import argparse
    import signal

    from experiments import validation_v2 as cli
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"site": "ems", "dates": {}}), encoding="utf-8")
    a = argparse.Namespace(sensing_times=str(f))
    assert cli._sensing_file("ems", a) == str(f)
    with pytest.raises(ValueError, match="file of 'ems', not 'escalda'"):
        cli._sensing_file("escalda", a)
    g = tmp_path / "plain.json"
    g.write_text(json.dumps({"2024-01-02": "2024-01-02T10:57:00"}), encoding="utf-8")
    b = argparse.Namespace(sensing_times=str(g))
    assert cli._sensing_file("escalda", b) == str(g)
    with pytest.raises(ValueError, match="folder form"):
        cli._sensing_file("escalda", b, n_sites=2)
    p = cli.build_parser()
    assert p.parse_args(["prepare", "escalda", "--block-rows", "128"]).block_rows == 128
    with pytest.raises(SystemExit):
        p.parse_args(["prepare", "escalda", "--block-rows", "127"])
    assert p.parse_args(["truth", "escalda", "--sensing-times", str(f)]).sensing_times == str(f)
    # SIGTERM (kill, tmux kill-server) raises KeyboardInterrupt like Ctrl-C
    old = cli._interrupt_on_signals()
    try:
        assert signal.SIGTERM in old
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
            for _ in range(1000):                  # the handler runs between bytecodes
                pass
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def test_cli_output_root_gitignore_keeps_the_heavy_outputs_out(tmp_path):
    from experiments import validation_v2 as cli
    root = tmp_path / "v2"
    cli._ignore_output_root(str(root))
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "common/*/" in text and "*.npy" in text and "*.tmp-*" in text
    (root / ".gitignore").write_text("mine\n", encoding="utf-8")
    cli._ignore_output_root(str(root))                       # never overwritten
    assert (root / ".gitignore").read_text(encoding="utf-8") == "mine\n"


def test_gauge_lag_rows_split_the_level_error():
    """User decision of 2026-10-15: one E0 'gaugelag' unit (MAREA with the
    judges' measured band lags, E0g) per elevation site with judges, at P0,
    W3, on the base E0 unit's scenes, pixels and setting; not_applicable at
    Villaviciosa; never counted as MAREA's own LB row; never cut."""
    units = [u for u in D.build("levels") if u.variant == "gaugelag"]
    by_site = {u.site: u for u in units}
    assert set(by_site) == set(D.ELEV_SITES)
    for site in D.DUTCH_SITES:
        u = by_site[site]
        assert u.status == "planned" and u.products == ("E0g",)
        assert u.setting == D.P0 and u.window == "W3" and u.N == "all" and u.cut_step == 0
        base = D.Unit(design="base", site=site, window="W3", setting=D.P0, method="E0",
                      variant="boot30")
        assert u.effective_key == base.effective_key and u.unit_id != base.unit_id
    assert by_site["villaviciosa"].status == "not_applicable"
    assert "no in-estuary judges" in by_site["villaviciosa"].note
    names = {c[0]: (c[1], c[2]) for c in D.ATTRIBUTION_CONTRASTS}
    assert names["ignoring the lag"] == ("E0g@LB", "E0a@LB")
    assert names["lag estimation"] == ("E0b@LB", "E0g@LB")
    assert names["band model"] == ("E0a@LG", "E0g@LB")
    att = D.attribution_rows()
    for site in D.DUTCH_SITES:
        assert att[site]["present"]["E0g_LB"]
        assert not any("E0g" in m for m in att[site]["missing"])
    assert any("E0g" in m and "no in-estuary judges" in m
               for m in att["villaviciosa"]["missing"])
    # the gauge-lag unit alone never stands in for MAREA's LB row (Stage L at W3)
    only_g = [u for u in units if u.status == "planned"]
    assert not D.never_cut_check(only_g, sites=D.DUTCH_SITES)["Stage L at W3"]["ok"]


# ─────────────────────────────────────────────────────────────────────────────
#  check-env's inputs section (2026-10-15: the server lacked estuary_truth/)
# ─────────────────────────────────────────────────────────────────────────────

_EST = "data_v4/gauges/estuary_truth/"
_COPY_EST = "copy from the laptop: data_v4/gauges/estuary_truth/ (291 MB)"


def test_cli_check_env_inputs_fail_ok_warn(tmp_path, monkeypatch):
    """A missing, empty or cut-short input that a Campaign 1 step needs is a
    FAIL that names the step and what to copy from the laptop (only the bad
    files while their folder is here, the folder when it is not); a present
    one is OK with its size or file count; a later milestone's is a WARN,
    checked at 10 m only; an input of a site that is not checked is skipped;
    an entry that cannot be checked is reported and the section goes on."""
    import shutil

    from experiments import v13_benchmark as v13
    from experiments import validation_v2 as cli

    est = tmp_path / "data_v4" / "gauges" / "estuary_truth"
    est.mkdir(parents=True)
    (est / "escalda__a.csv").write_bytes(b"x" * 2000)
    (est / "escalda__c.csv").write_bytes(b"x" * 999)              # an interrupted copy
    (tmp_path / "data_v4" / "gauges" / "ioc_stations.json").write_bytes(b"[]" * 1500)
    ocean = tmp_path / "tide_models" / "M" / "ocean_tides"
    ocean.mkdir(parents=True)
    for k in range(3):
        (ocean / f"c{k}_ocean_m.nc").write_bytes(b"y" * 10)
    (tmp_path / "products_e").mkdir()
    (tmp_path / "products_e" / "marea_extract.npz").write_bytes(b"")
    sizes = {_EST + "escalda__a.csv": 2000, _EST + "escalda__c.csv": 1000}
    table = (
        cli.Need("IOC station list", ("data_v4/gauges/ioc_stations.json",), ("escalda", "wadden"),
                 "prepare (h_B)", "before prepare", "copy from the laptop: data_v4/gauges/"
                 "ioc_stations.json", copy="data_v4/gauges/ioc_stations.json"),
        cli.Need("tide model", ("tide_models/M/ocean_tides/*_ocean_m.nc",), (), "prepare (h_B)",
                 "before prepare", "copy from the laptop: tide_models/M/", copy="tide_models/M/",
                 count=3),
        cli.Need("judge stations", (_EST + "stations.json",), ("escalda", "ferrol"),
                 "fetch judges", "before fetch judges", _COPY_EST, copy=_EST),
        cli.Need("judge series", (_EST + "escalda__a.csv", _EST + "escalda__b.csv",
                                  _EST + "escalda__c.csv"), ("escalda",),
                 "truth escalda (lag truths)", "before truth", _COPY_EST, copy=_EST),
        cli.Need("production extraction", ("products_e/marea_extract.npz",), ("ems",),
                 "the gate G-keep", "V2", "made on the server", note="identity not checked"),
        cli.Need("RTK grid", ("products_v/grid.tif",), ("villaviciosa",), "truth villaviciosa",
                 "before truth", "tracked in git: git pull"),
    )
    ck = v13.Checks()
    out = cli.check_inputs(ck, ["escalda", "ems"], root=str(tmp_path), table=table, sizes=sizes,
                           boundary=False)
    got = {i["what"]: (i["level"], i["detail"]) for i in ck.items}
    assert got["input IOC station list"] == ("OK", "data_v4/gauges/ioc_stations.json, 3 KB "
                                                   "(escalda)")
    assert got["input tide model"] == ("OK", "3 files, 30 B in tide_models/M/ocean_tides/ "
                                             "(every site)")
    assert got["input judge stations"] == (
        "FAIL", "data_v4/gauges/estuary_truth/stations.json is missing (escalda); needed now "
                "(fetch judges is the first V1 step); used by fetch judges; get it: " + _COPY_EST)
    level, detail = got["input escalda judge series"]
    assert level == "FAIL" and detail == (
        "2 of 3 files missing or incomplete in data_v4/gauges/estuary_truth/ (escalda__b.csv, "
        "escalda__c.csv (has 999 B, not the laptop's 1,000 B (an interrupted copy?))); needed "
        "before truth; used by truth escalda (lag truths); get it: " + _COPY_EST)
    assert got["input ems production extraction"] == (
        "WARN", "products_e/marea_extract.npz is empty (0 B); needed at milestone V2; used by "
                "the gate G-keep; get it: made on the server")
    assert "input villaviciosa RTK grid" not in got           # villaviciosa is not checked
    # the folder is here: only the bad files are to be copied
    assert out == {"OK": 2, "FAIL": 2, "WARN": 1,
                   "copy": [_EST + "stations.json", _EST + "escalda__b.csv",
                            _EST + "escalda__c.csv"]}
    assert got["inputs summary"][1].endswith(
        "(WARN); copy from the laptop to the same repository-relative paths with scp -p (it "
        "keeps the file times; a folder goes with scp -p -r into its parent folder): "
        + ", ".join(out["copy"]))
    (tmp_path / "products_e" / "marea_extract.npz").write_bytes(b"z")
    ck = v13.Checks()
    cli.check_inputs(ck, ["ems"], root=str(tmp_path), table=table, boundary=False)
    assert {i["what"]: i["detail"] for i in ck.items}["input ems production extraction"] == (
        "products_e/marea_extract.npz, 1 B; identity not checked")
    # a glob short of its count is a FAIL; at 20 m the later milestones are not checked;
    # without its folder here, the folder is to be copied (and none of its files)
    (ocean / "c2_ocean_m.nc").unlink()
    shutil.rmtree(est)
    ck = v13.Checks()
    out = cli.check_inputs(ck, ["escalda", "ems"], res=20, root=str(tmp_path), table=table,
                           sizes=sizes, boundary=False)
    got = {i["what"]: (i["level"], i["detail"]) for i in ck.items}
    assert got["input tide model"] == ("FAIL", (
        "tide_models/M/ocean_tides/*_ocean_m.nc matches 2 of 3 files (every site); needed "
        "before prepare; used by prepare (h_B); get it: copy from the laptop: tide_models/M/"))
    assert "input ems production extraction" not in got
    assert out == {"OK": 1, "FAIL": 3, "WARN": 0, "copy": ["tide_models/M/", _EST]}
    # an entry that cannot be checked (a stale mount, say) is reported at its level
    def stale(*a, **k):
        raise OSError("stale file handle")

    monkeypatch.setattr(cli, "_scan_input", stale)
    ck = v13.Checks()
    out = cli.check_inputs(ck, ["ems"], root=str(tmp_path), table=table, boundary=False)
    got = {i["what"]: (i["level"], i["detail"]) for i in ck.items}
    assert got["input tide model"][0] == "FAIL" and got["input tide model"][1].startswith(
        "cannot be checked (OSError: stale file handle) (every site); needed before prepare")
    assert got["input ems production extraction"][0] == "WARN"
    assert out["FAIL"] == 1 and out["WARN"] == 1 and "inputs summary" in got


def test_cli_check_env_inputs_names_from_this_machines_files(tmp_path, monkeypatch):
    """The judges' CSVs are named from this machine's stations.json (as
    judges.stations reads it) and the sheets from sheets_for on this
    machine's catalogue; while those files are absent, the laptop's names.
    A stations.json without a judge of config.JUDGES, or a catalogue that
    sheets_for cannot read, is a FAIL."""
    from experiments import v13_benchmark as v13
    from experiments import validation_v2 as cli

    monkeypatch.chdir(ROOT)
    from experiments import v0_vaklodingen_fetch as v0

    table = (cli.Need("judge series", (_EST + "laptop.csv",), ("wadden",), "truth wadden",
                      "before fetch judges", _COPY_EST, copy=_EST, derive="judges"),
             cli.Need("Vaklodingen sheets", ("data_v4/truth/vaklodingen/vaklodingenKB1.nc",),
                      ("ems",), "truth ems --part elev", "before truth",
                      "copy from the laptop: data_v4/truth/vaklodingen/",
                      copy="data_v4/truth/vaklodingen/", derive="vak"))

    def run():
        ck = v13.Checks()
        cli.check_inputs(ck, ["wadden", "ems"], root=str(tmp_path), table=table, sizes={},
                         boundary=False)
        return {i["what"]: (i["level"], i["detail"]) for i in ck.items}

    got = run()                                                  # neither file here
    assert got["input wadden judge series"][1].startswith(
        "data_v4/gauges/estuary_truth/laptop.csv is missing (names from the laptop's list: "
        "stations.json is absent here); needed now")
    assert got["input ems Vaklodingen sheets"][1].startswith(
        "data_v4/truth/vaklodingen/vaklodingenKB1.nc is missing (names from the laptop's list: "
        "the catalogue is absent here); needed before truth")
    # stations.json here: the 'file' of each judge
    est = tmp_path / "data_v4" / "gauges" / "estuary_truth"
    est.mkdir(parents=True)
    rows = [{"code": c, "site": "wadden", "file": f"wadden__{c}.csv", "in_boundary": False}
            for c in config.JUDGES["wadden"]]
    (est / "stations.json").write_text(json.dumps(rows), encoding="utf-8")
    for r in rows:
        (est / r["file"]).write_bytes(b"z" * 10)
    assert run()["input wadden judge series"] == (
        "OK", "2 files, 20 B in data_v4/gauges/estuary_truth/ (names from stations.json)")
    (est / rows[1]["file"]).unlink()
    assert run()["input wadden judge series"][1].startswith(
        f"1 of 2 files missing in data_v4/gauges/estuary_truth/ ({rows[1]['file']}) (names "
        "from stations.json); needed now")
    (est / "stations.json").write_text(json.dumps(rows[:1]), encoding="utf-8")
    level, detail = run()["input wadden judge series"]
    assert level == "FAIL" and detail.startswith(
        "data_v4/gauges/estuary_truth/stations.json cannot list the wadden judges (KeyError: "
        "data_v4/gauges/estuary_truth/stations.json lacks the judges ['harlingen.waddenzee'] "
        "of wadden); needed now")
    # the catalogue here: sheets_for's names, but only for the repository whose
    # catalogue sheets_for reads (v0.ROOT)
    (tmp_path / "data_v4" / "truth").mkdir(parents=True)
    (tmp_path / "data_v4" / "truth" / "vaklodingen_catalog.nc").write_bytes(b"c")

    def broken(site):
        raise OSError("not a netCDF file")

    monkeypatch.setattr(v0, "sheets_for", broken)
    assert run()["input ems Vaklodingen sheets"][1].startswith(
        "data_v4/truth/vaklodingen/vaklodingenKB1.nc is missing (names from the laptop's list); "
        "needed before truth")
    monkeypatch.setattr(v0, "ROOT", str(tmp_path))
    monkeypatch.setattr(v0, "sheets_for", lambda site: [("vaklodingenKB9.nc", None, None)])
    got = run()["input ems Vaklodingen sheets"]
    assert got[0] == "FAIL" and got[1].startswith(
        "data_v4/truth/vaklodingen/vaklodingenKB9.nc is missing (names from the catalogue); "
        "needed before truth")
    (tmp_path / "data_v4" / "truth" / "vaklodingen").mkdir()
    (tmp_path / "data_v4" / "truth" / "vaklodingen" / "vaklodingenKB9.nc").write_bytes(b"s")
    assert run()["input ems Vaklodingen sheets"] == (
        "OK", "data_v4/truth/vaklodingen/vaklodingenKB9.nc, 1 B (names from the catalogue)")
    monkeypatch.setattr(v0, "sheets_for", broken)
    assert run()["input ems Vaklodingen sheets"] == (
        "FAIL", "data_v4/truth/vaklodingen_catalog.nc cannot list the ems sheets (OSError: not "
                "a netCDF file); needed before truth; used by truth ems --part elev; get it: "
                "copy from the laptop: data_v4/truth/vaklodingen/")


def test_cli_check_env_boundary_gauges_against_the_union_cache(tmp_path):
    """At 10 m, the boundary codes of this machine's ioc_stations.json
    (boundary_codes) must be among fetch ioc's UNION_CODES and ready in the
    union cache (union_ready); a pytmd site has no line, nor has any site
    while the list is absent (its own line FAILs)."""
    from experiments import v13_benchmark as v13
    from experiments import validation_v2 as cli
    from experiments.benchmark.fetch import ioc as fioc

    gauges = tmp_path / "data_v4" / "gauges"
    cache = gauges / "ioc_2017-2025"
    cache.mkdir(parents=True)
    st = [{"Code": "vlis", "Lat": 51.4429, "Lon": 3.5970},
          {"Code": "brsk", "Lat": 51.4014, "Lon": 3.5477},
          {"Code": "trnz", "Lat": 51.3362, "Lon": 3.8195}]          # held out at escalda
    (gauges / "ioc_stations.json").write_text(json.dumps(st), encoding="utf-8")
    man = {"codes": {}}
    for code in ("vlis", "brsk"):
        for y in fioc.UNION_YEARS:
            (cache / f"ioc_{code}_{y}-01-01_{y}-12-31.json").write_text("[1]")
        (cache / fioc.marker_name(code)).write_text("[]")
        man["codes"][code] = {"years": {str(y): {"ok": True, "kind": "downloaded"}
                                        for y in fioc.UNION_YEARS}}
    (cache / fioc.MANIFEST_NAME).write_text(json.dumps(man), encoding="utf-8")

    def run(sites=("escalda", "villaviciosa")):
        ck = v13.Checks()
        out = cli.check_boundary_cache(ck, list(sites), root=str(tmp_path))
        return out, {i["what"]: (i["level"], i["detail"]) for i in ck.items}

    out, got = run()
    assert out == {"OK": 1, "FAIL": 0} and list(got) == ["input escalda boundary gauges"]
    assert got["input escalda boundary gauges"] == (
        "OK", "['vlis', 'brsk'] from data_v4/gauges/ioc_stations.json, ready in "
              "data_v4/gauges/ioc_2017-2025/ (every year 2017-2025, markers, assessed by the "
              "last fetch ioc)")
    # inside the inputs section at 10 m only, counted in its summary
    ck = v13.Checks()
    res = cli.check_inputs(ck, ["escalda"], root=str(tmp_path), table=())
    assert res["OK"] == 1 and [i["what"] for i in ck.items] == [
        "inputs", "input escalda boundary gauges", "inputs summary"]
    ck = v13.Checks()
    cli.check_inputs(ck, ["escalda"], res=20, root=str(tmp_path), table=())
    assert "input escalda boundary gauges" not in [i["what"] for i in ck.items]
    # a missing marker: not ready
    (cache / fioc.marker_name("brsk")).unlink()
    out, got = run()
    level, detail = got["input escalda boundary gauges"]
    assert out["FAIL"] == 1 and level == "FAIL" and detail.startswith(
        "['vlis', 'brsk'] from data_v4/gauges/ioc_stations.json: the union IOC cache is not "
        "ready (brsk: no union marker data_v4/gauges/ioc_2017-2025/ioc_brsk_2017-01-01_"
        "2025-12-31.json); needed before prepare; used by prepare --step record")
    assert detail.endswith("get it: run 'fetch ioc' (exit 0 only when the cache is ready)")
    # a list whose nearest gauge fetch ioc never fetches
    (gauges / "ioc_stations.json").write_text(
        json.dumps(st + [{"Code": "zzzz", "Lat": 51.395, "Lon": 3.70}]), encoding="utf-8")
    level, detail = run()[1]["input escalda boundary gauges"]
    assert level == "FAIL" and detail.startswith(
        "data_v4/gauges/ioc_stations.json gives ['zzzz', 'vlis']: ['zzzz'] not among fetch "
        "ioc's UNION_CODES")
    # no union cache at all
    (gauges / "ioc_stations.json").write_text(json.dumps(st), encoding="utf-8")
    import shutil
    shutil.rmtree(cache)
    assert "(data_v4/gauges/ioc_2017-2025/ does not exist)" in \
        run()[1]["input escalda boundary gauges"][1]
    # no list: no line (the 'IOC station list' entry FAILs)
    (gauges / "ioc_stations.json").unlink()
    assert run() == ({"OK": 0, "FAIL": 0}, {})


def test_cli_check_env_inputs_table_names_the_code_paths(monkeypatch):
    """The INPUTS table holds the code's own paths: the judges' stations, the
    IOC list, the PNOA DTM and the RTK files; one judge-series entry per site
    with judges, one CSV per listed judge (the stations.json 'file' where the
    file is here), LG named only where LG runs; the sheets of every
    Vaklodingen site (sheets_for where the catalogue is here); the 10 m
    production files of config.site_config; the laptop sizes of the
    laptop-only copies. No Zarr store and no CLI output is listed; a copy
    hint names its path."""
    from experiments import v1_vaklodingen_validate as v1
    from experiments import v10_validation_report as v10
    from experiments import validation_v2 as cli
    from pyintertidal import rtk

    monkeypatch.chdir(ROOT)
    assert set(cli.WHEN_PHRASE) == set(cli.WHEN) and set(cli.CAMPAIGN1_WHEN) < set(cli.WHEN)
    assert (cli._STATIONS, cli._ESTUARY, cli._IOC_STATIONS, cli._VAK_DIR, cli._MDT02) == (
        config.STATIONS_JSON, config.ESTUARY_TRUTH + "/", config.IOC_STATIONS,
        v1.TRUTH_DIR + "/", v10.MDT02)
    assert cli._LG_SITES == tuple(settings.LEVEL["LG"].sites)
    for n in cli.INPUTS:
        assert n.when in cli.WHEN
        assert n.what and n.needed_by and n.how and set(n.sites) <= set(config.SITES)
        assert n.paths and sum("*" in p for p in n.paths) <= 1
        for p in n.paths:
            assert "\\" not in p and not os.path.isabs(p) and ".." not in p.split("/")
            assert not p.endswith("/")
            assert not p.startswith(("cubes/", "results/", "data_v4/gauges/ioc_2017-2025"))
        if n.copy:
            assert n.how.startswith(f"copy from the laptop: {n.copy}")
        if n.derive:
            assert n.derive in ("judges", "vak") and len(n.sites) == 1
    by = {(n.what, n.sites): n for n in cli.INPUTS}
    paths = {p for n in cli.INPUTS for p in n.paths}
    assert {config.STATIONS_JSON, config.IOC_STATIONS, v10.MDT02, rtk.DEFAULT_CSV,
            rtk.DEFAULT_ALICANTE_CSV, rtk.DEFAULT_GRID} <= paths
    gauge_sites = tuple(s for s in config.SITES if config.BOUNDARY[s]["kind"] == "make_boundary")
    judge_sites = tuple(s for s in config.SITES if config.JUDGES[s])
    assert by[("IOC station list", gauge_sites)].when == "before prepare"
    assert by[("judge stations", judge_sites)].when == "before fetch judges"
    series = {n.sites[0]: [os.path.basename(p) for p in n.paths]
              for n in cli.INPUTS if n.what == "judge series"}
    assert {s: len(f) for s, f in series.items()} == {s: len(config.JUDGES[s])
                                                      for s in judge_sites}
    for n in cli.INPUTS:
        if n.what == "judge series":
            assert n.derive == "judges" and n.when == "before fetch judges"
            assert ("(LG)" in n.needed_by) == (n.sites[0] in settings.LEVEL["LG"].sites)
    if os.path.exists(config.STATIONS_JSON):
        with open(config.STATIONS_JSON, encoding="utf-8") as f:
            st = {r["code"]: r["file"] for r in json.load(f)}
        assert series == {s: [st[c] for c in config.JUDGES[s]] for s in judge_sites}
    # the laptop sizes: every judge CSV, stations.json and the PNOA DTM, equal where here
    assert set(cli.LAPTOP_BYTES) == {config.STATIONS_JSON, v10.MDT02} | {
        cli._ESTUARY + f for files in series.values() for f in files}
    for p, nbytes in cli.LAPTOP_BYTES.items():
        if os.path.exists(p):
            assert os.path.getsize(p) == nbytes, p
    vak = {n.sites[0]: [os.path.basename(p) for p in n.paths]
           for n in cli.INPUTS if n.what == "Vaklodingen sheets"}
    assert set(vak) == {s for s in config.SITES if "vaklodingen" in config.ELEVATION_TRUTH[s]}
    if os.path.exists("data_v4/truth/vaklodingen_catalog.nc"):
        from experiments import v0_vaklodingen_fetch as v0
        assert vak == {s: [name for name, _, _ in v0.sheets_for(s)] for s in vak}
    for site in config.SITES:
        cfg = config.site_config(site, 10)
        assert by[("production extraction", (site,))].paths == (cfg.extract,)
        assert by[("production MAREA result", (site,))].paths == (cfg.transition_result,)


def test_l09_is_not_applicable_where_the_window_cannot_hold_its_span():
    """HD26-H needs >= 365 days between its first and last scene: a common
    calendar year or W1 holds at most 364, so the lag design plans L09 there
    as not applicable and its unit writes L10 only (structural, from the
    window bounds; the rule is not relaxed)."""
    assert D.max_span_days("Y2023") == 364 and D.max_span_days("Y2024") == 365
    assert D.max_span_days("W1") == 364 and D.max_span_days("W3") > 365
    assert D.l09_span_rule("Y2024")[0] and not D.l09_span_rule("Y2023")[0]
    rows = D.plan_rows("lag", ("ems",))
    l09 = [r for r in rows if r["method"] == "L09"]
    by = {}
    for r in l09:
        by.setdefault(r["window"], []).append((r["status"], r["products"], r["variant"]))
    for w in ("Y2023", "W1"):
        assert sorted(by[w]) == [("not_applicable", "L09", D.L09_SPAN_VARIANT),
                                 ("planned", "L10", "")], w
    for w in ("W3", "W2", "Y2024"):
        assert by[w] == [("planned", "L09 L10", "")], w
    na = next(r for r in l09 if r["window"] == "Y2023" and r["status"] == "not_applicable")
    assert "365" in na["note"] and "spec §4.2" in na["note"]


def test_write_plan_keeps_the_other_sites_rows(tmp_path):
    root = str(tmp_path)
    D.write_plan("lag", root=root, sites=("escalda",))
    D.write_plan("lag", root=root, sites=("ems",))
    import csv
    with open(os.path.join(root, "plans", "lag.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["site"] for r in rows} == {"escalda", "ems"}
    full = D.plan_rows("lag", ("escalda", "ems"))
    assert [r["unit_id"] for r in rows] == [r["unit_id"] for r in full]
    idx = json.load(open(os.path.join(root, "plans", "index.json"), encoding="utf-8"))
    assert idx["plans"]["lag"]["sites"] == ["escalda", "ems"]
    D.write_plan("lag", root=root, sites=("ems",), replace=True)
    with open(os.path.join(root, "plans", "lag.csv"), encoding="utf-8", newline="") as f:
        assert {r["site"] for r in csv.DictReader(f)} == {"ems"}
