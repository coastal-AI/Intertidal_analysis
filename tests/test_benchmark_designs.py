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


@pytest.mark.parametrize("argv,ms", [
    (["run", "--plans", "lag", "--worker-id", "1"], "V2"),
    (["score", "--part", "lag"], "V2"), (["lock", "--phase", "A"], "V2"),
    (["gate", "zarr"], "V2"), (["control"], "V2"), (["timing"], "V2"),
    (["report"], "V6"), (["pilot", "units"], "V3"), (["pilot", "attain"], "V2")])
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
