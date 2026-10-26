"""Tests of experiments.benchmark.product: units, products, atomic IO, keys.

Pins the pre-registered rules of the module docstring (v13 spec §1.8,
§2.11, §7.4):

* ``.npz`` files are byte-stable (fixed entry times) and their content hash
  ignores the zip container; object arrays are refused;
* JSON and text are written atomically, NaN round-trips;
* ``subset_key`` is the sha256 of the sorted int64 positions; a Unit
  validates its fields and refuses a mismatched subset key; equality is by
  description and subset;
* a Product casts its per-pixel arrays to the frozen dtypes and refuses
  ragged, 2-D or out-of-range arrays and unknown statuses;
* ``save`` publishes atomically, keeps a previous attempt as ``.old-``,
  round-trips every array (also those nested in method_bands, diag,
  domain, provenance); ``is_done`` needs a final status AND the same
  inputs_key; ``iter_runs`` skips temporary and superseded attempts;
* the package key hashes exactly files (minus manifest.json and
  judges.json), inputs, obs_sha and the package FROZEN sections; config,
  code and inputs keys change with each of their parts and only with them.

Run:  python -X utf8 -m pytest tests/test_benchmark_product.py -q
"""
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, product  # noqa: E402

SITE, RES = "escalda", 20


def _unit(sel=(0, 3, 7), **kw):
    d = dict(site=SITE, res_m=RES, group="marea_pair", variant="", family="rand",
             mode="A", N=len(sel), rep=1, sens={}, sel=np.asarray(sel))
    d.update(kw)
    return product.Unit(**d)


def _product(unit=None, n=6, status="ok", **kw):
    rng = np.random.default_rng(3)
    d = dict(method="marea", variant="", unit=unit or _unit(), status=status,
             z=rng.normal(size=n), z_raw=rng.normal(size=n), conf=rng.random(n),
             clear_obs=np.arange(n), lag=np.full(n, 30.0))
    d.update(kw)
    return product.Product(**d)


# ─────────────────────────────────────────────────────────────────────────────
#  atomic IO and hashing
# ─────────────────────────────────────────────────────────────────────────────

def test_save_npz_is_byte_stable_and_loadable(tmp_path):
    arrays = {"a": np.arange(5, dtype=np.int16), "b": np.linspace(0, 1, 7),
              "s": np.array("text"), "m": np.ones((2, 3), bool)}
    p1, p2 = tmp_path / "one.npz", tmp_path / "two.npz"
    product.save_npz(p1, arrays)
    time.sleep(1.1)                  # zip entry times have a 2-s resolution
    product.save_npz(p2, arrays)
    assert p1.read_bytes() == p2.read_bytes()
    back = product.load_npz(p1)
    assert list(back) == list(arrays)
    for k, v in arrays.items():
        assert back[k].dtype == np.asarray(v).dtype and np.array_equal(back[k], v)
    with np.load(p1, allow_pickle=False) as z:          # an ordinary .npz
        assert np.array_equal(z["a"], arrays["a"])
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]


def test_save_npz_refuses_object_arrays_and_leaves_nothing(tmp_path):
    p = tmp_path / "bad.npz"
    with pytest.raises(TypeError):
        product.save_npz(p, {"ok": np.arange(3), "obj": np.array([{"a": 1}], dtype=object)})
    assert os.listdir(tmp_path) == []


def test_npz_hash_is_the_content_not_the_container(tmp_path):
    arrays = {"x": np.arange(10.0), "y": np.array([1, 2, 3], np.int32)}
    a, b, c = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"
    product.save_npz(a, arrays, compress=True)
    np.savez(b, **arrays)                          # another container, other times
    product.save_npz(c, {"x": np.arange(10.0), "y": np.array([1, 2, 4], np.int32)})
    assert a.read_bytes() != b.read_bytes()
    assert product.sha256_file(a) == product.sha256_file(b)
    assert product.sha256_file(a) != product.sha256_file(c)
    # any other file: the raw bytes
    t = tmp_path / "t.txt"
    t.write_bytes(b"abc\n")
    assert product.sha256_file(t) == hashlib.sha256(b"abc\n").hexdigest()
    e = product.file_entry(t)
    assert e["size"] == 4 and e["sha256"] == hashlib.sha256(b"abc\n").hexdigest()
    assert e["mtime_ns"] == os.stat(t).st_mtime_ns
    assert "sha256" not in product.file_entry(t, sha=False)


def test_write_json_atomic_nan_and_numpy(tmp_path):
    p = tmp_path / "sub" / "doc.json"
    obj = {"nan": float("nan"), "i": np.int64(3), "f": np.float32(0.5),
           "a": np.arange(3), "s": {2, 1}, "t": (1, 2)}
    product.write_json(p, obj)
    back = product.read_json(p)
    assert np.isnan(back["nan"]) and back["i"] == 3 and back["f"] == 0.5
    assert back["a"] == [0, 1, 2] and back["s"] == [1, 2] and back["t"] == [1, 2]
    with pytest.raises(TypeError):
        product.write_json(tmp_path / "bad.json", {"x": object()})
    assert not (tmp_path / "bad.json").exists()
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]
    product.write_text(tmp_path / "log.txt", "line\n")
    assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "line\n"


def test_utc_stamp_format():
    s = product.utc_stamp()
    assert len(s) == 22 and s[8] == "T" and s.endswith("Z") and s[:8].isdigit()
    assert s[9:21].isdigit()                      # HHMMSS and microseconds


# ─────────────────────────────────────────────────────────────────────────────
#  units
# ─────────────────────────────────────────────────────────────────────────────

def test_subset_key_is_the_sha_of_the_int64_positions():
    sel = np.array([0, 4, 9], np.int32)
    want = hashlib.sha256(sel.astype("<i8").tobytes()).hexdigest()
    assert product.subset_key(sel) == want
    assert product.subset_key([0.0, 4.0, 9.0]) == want
    assert product.subset_key([]) == hashlib.sha256(b"").hexdigest()
    for bad, exc in (([3, 1], ValueError), ([1, 1], ValueError), ([-1, 2], ValueError),
                     ([0.5, 2.0], TypeError), (np.array([True, False]), TypeError),
                     (np.zeros((2, 2), int), ValueError)):
        with pytest.raises(exc):
            product.subset_key(bad)


def test_unit_validation_and_equality():
    u = _unit()
    assert u.subset_key == product.subset_key([0, 3, 7]) and u.n == 3
    assert u.sel.dtype == np.int64
    assert u == _unit() and u != _unit(rep=2) and u != _unit(sel=(0, 3, 8))
    assert u.__hash__ is None
    assert _unit(N="all").N == "all"
    assert u.label == "escalda 20m marea_pair rand-A N=3 r1"
    assert _unit(variant="tau120", sens={"mask": "blocklist"}).label.endswith(
        "[tau120] rand-A N=3 r1 mask-blocklist")
    for kw in (dict(family="other"), dict(mode="B"), dict(N=0), dict(N=2.5), dict(N=True),
               dict(rep=-1), dict(rep=1.5), dict(subset_key="0" * 64),
               dict(site="nowhere"), dict(res_m=30), dict(site="ferrol")):
        with pytest.raises(ValueError):
            _unit(**kw)
    back = product.Unit.from_json(json.loads(json.dumps(u.to_json())), u.sel)
    assert back == u


def test_senstag_and_unit_dir():
    assert product.senstag({}) == ""
    assert product.senstag({"mask": "blocklist", "floor": 80}) == "floor-80+mask-blocklist"
    assert product.senstag({"a b": "x/y"}) == "a-b-x-y"
    u = _unit(rep=3)
    assert product.unit_dir("R\\", u, "marea", "") == "R/runs/escalda/marea/rand-A/N003_r03"
    v = _unit(N="all", rep=0, mode="F", family="win", sens={"index": "raw"})
    assert product.unit_dir("R", v, "nidem", "v2") == \
        "R/runs/escalda/nidem__v2/win-F/Nall_r00__index-raw"


def test_not_computable_carries_stage_reason_and_diag():
    e = product.NotComputable("no_intertidal", "1,000 keep px", {"n_keep": 1000})
    assert (e.stage, e.reason, e.diag) == ("no_intertidal", "1,000 keep px", {"n_keep": 1000})
    assert str(e) == "no_intertidal: 1,000 keep px"
    e = product.NotComputable("record_floor", {"T": 3})
    assert e.reason is None and e.diag == {"T": 3} and str(e) == "record_floor"


# ─────────────────────────────────────────────────────────────────────────────
#  products
# ─────────────────────────────────────────────────────────────────────────────

def test_product_casts_and_validates():
    p = _product()
    for name, dt in product.ARRAY_DTYPES.items():
        assert getattr(p, name).dtype == np.dtype(dt)
    assert p.n_support == 6
    assert _product(z=None, z_raw=None, conf=None, clear_obs=None, lag=None).n_support == 0
    with pytest.raises(ValueError):
        _product(status="done")
    with pytest.raises(ValueError):
        _product(lag_kind="angular")
    with pytest.raises(ValueError):
        _product(conf=np.ones(5))                     # ragged
    with pytest.raises(ValueError):
        _product(z=np.ones((2, 3)))                   # not 1-D
    with pytest.raises(ValueError):
        _product(clear_obs=np.array([0, 1, 2, 3, 4, 40000]))      # beyond int16
    with pytest.raises(ValueError):
        _product(clear_obs=np.array([0, 1, 2, 3, 4, np.nan]))     # not finite


def test_save_and_load_round_trip(tmp_path):
    p = _product(method_bands={"tau": np.array([0.0, 15.0]), "k": [1, 2]},
                 diag={"curve": np.arange(4.0), "n": 3, "nested": {"m": np.eye(2)}},
                 domain={"status": "ok", "keep": np.array([5, 9])},
                 provenance={"git": {"head": None}}, fallback="zero",
                 keys={"inputs_key": "k1"}, timings={"total_s": 1.5}, peak_rss_mb=12.5,
                 stage=None, reason=None)
    out = str(tmp_path / "unit")
    assert product.save(p, out, log_text="hello\n") == os.path.normpath(out)
    q = product.load(out)
    for name in product.ARRAY_DTYPES:
        a, b = getattr(p, name), getattr(q, name)
        assert a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))
    assert q.unit == p.unit and q.status == "ok" and q.fallback == "zero"
    assert np.array_equal(q.method_bands["tau"], [0.0, 15.0]) and q.method_bands["k"] == [1, 2]
    assert np.array_equal(q.diag["curve"], np.arange(4.0)) and q.diag["n"] == 3
    assert np.array_equal(q.diag["nested"]["m"], np.eye(2))
    assert np.array_equal(q.domain["keep"], [5, 9]) and q.domain["status"] == "ok"
    assert q.keys == {"inputs_key": "k1"} and q.peak_rss_mb == 12.5
    assert (tmp_path / "unit" / "log.txt").read_text(encoding="utf-8") == "hello\n"
    run = product.read_run(out)
    assert run["format"] == product.PRODUCT_FORMAT and run["n_support"] == 6
    assert run["arrays"] == sorted(list(product.ARRAY_DTYPES) + ["unit_sel"])


def test_save_keeps_the_previous_attempt_and_moves_files(tmp_path):
    out = str(tmp_path / "unit")
    product.save(_product(status="error", reason="boom"), out)
    extra = tmp_path / "frame.tif"
    extra.write_bytes(b"raster")
    product.save(_product(), out, files={"rasters/frame.tif": str(extra)})
    olds = [d for d in os.listdir(tmp_path) if d.startswith("unit.old-")]
    assert len(olds) == 1
    assert product.read_run(str(tmp_path / olds[0]))["status"] == "error"
    assert product.read_run(out)["status"] == "ok"
    assert (tmp_path / "unit" / "rasters" / "frame.tif").read_bytes() == b"raster"
    assert not extra.exists()
    assert not [d for d in os.listdir(tmp_path) if ".tmp-" in d]
    with pytest.raises(ValueError):
        product.save(_product(), str(tmp_path / "u2"), files={"../x": str(out)})


def test_is_done_needs_a_final_status_and_the_same_inputs_key(tmp_path):
    for status in product.STATUSES:
        out = str(tmp_path / status)
        product.save(_product(status=status, keys={"inputs_key": "K"}), out)
        assert product.is_done(out, "K") == (status in product.DONE_STATUSES)
        assert not product.is_done(out, "other")
    assert not product.is_done(str(tmp_path / "absent"), "K")
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "run.json").write_text("{not json", encoding="utf-8")
    assert product.read_run(str(bad)) is None and not product.is_done(str(bad), "K")


def test_iter_runs_skips_temporary_and_superseded_attempts(tmp_path):
    root = str(tmp_path)
    u = _unit()
    d = product.unit_dir(root, u, "marea", "")
    product.save(_product(unit=u), d)
    product.save(_product(unit=u, status="budget"), d)       # leaves an .old- copy
    tmp = d + ".tmp-host-1"
    os.makedirs(tmp)
    with open(os.path.join(tmp, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "ok"}, f)
    got = list(product.iter_runs(root))
    assert [g[0] for g in got] == [d] and got[0][1]["status"] == "budget"
    assert [g[0] for g in product.iter_runs(root, site=SITE)] == [d]
    assert list(product.iter_runs(root, site="wadden")) == []


def test_load_refuses_an_unknown_format(tmp_path):
    out = str(tmp_path / "unit")
    product.save(_product(), out)
    run = product.read_run(out)
    run["format"] = "v12-product"
    product.write_json(os.path.join(out, "run.json"), run)
    with pytest.raises(ValueError):
        product.load(out)


# ─────────────────────────────────────────────────────────────────────────────
#  keys
# ─────────────────────────────────────────────────────────────────────────────

def _manifest():
    return {"files": {"b03.npy": {"sha256": "1", "size": 10},
                      "manifest.json": {"sha256": "m"}, "judges.json": {"sha256": "j"}},
            "inputs": {"cube": {"sha256": "c", "mtime_ns": 5}},
            "obs_sha": "o",
            "frozen": {s: f"sha-{s}" for s in product.PACKAGE_KEY_SECTIONS} | {"scoring": "x"},
            "git": {"head": "abc"}}


def test_package_key_hashes_exactly_its_payload():
    m = _manifest()
    k = product.package_key(m)
    assert k == config.sha256_json(product.manifest_payload(m))
    assert set(product.manifest_payload(m)["files"]) == {"b03.npy"}
    same = json.loads(json.dumps(m))
    same["git"] = {"head": "def"}
    same["files"]["judges.json"]["sha256"] = "j2"
    same["files"]["manifest.json"]["sha256"] = "m2"
    same["files"]["b03.npy"]["size"] = 11
    same["inputs"]["cube"]["mtime_ns"] = 6
    same["frozen"]["scoring"] = "y"
    assert product.package_key(same) == k
    for path in (("files", "b03.npy"), ("inputs", "cube")):
        other = json.loads(json.dumps(m))
        other[path[0]][path[1]]["sha256"] = "changed"
        assert product.package_key(other) != k
    other = json.loads(json.dumps(m))
    other["obs_sha"] = "o2"
    assert product.package_key(other) != k
    for s in product.PACKAGE_KEY_SECTIONS:
        other = json.loads(json.dumps(m))
        other["frozen"][s] = "changed"
        assert product.package_key(other) != k, s
    other = json.loads(json.dumps(m))
    del other["obs_sha"]
    with pytest.raises(KeyError):
        product.package_key(other)


def test_config_code_and_inputs_keys():
    sec = "marea"
    base = product.config_key({"a": 1}, sec, "A", {}, "")
    assert base == product.config_key({"a": 1}, sec, "A", None, None)
    for args in (({"a": 2}, sec, "A", {}, ""), ({"a": 1}, sec, "F", {}, ""),
                 ({"a": 1}, sec, "A", {"mask": "blocklist"}, ""),
                 ({"a": 1}, sec, "A", {}, "tau120"), ({"a": 1}, None, "A", {}, "")):
        assert product.config_key(*args) != base, args
    assert product.config_key({"a": 1}, sec, "A", {}, "", group="marea_pair") != base
    with pytest.raises(KeyError):
        product.config_key({}, "no_such_section", "A", {}, "")
    with pytest.raises(ValueError):
        product.config_key({}, sec, "X", {}, "")
    ck = product.code_key("experiments/benchmark/product.py")
    assert ck == product.code_key(os.path.join(ROOT, "experiments", "benchmark", "product.py"))
    assert product.code_key("experiments/benchmark/product.py",
                            deps=product.pyintertidal_files("marea")) != ck
    assert product.pyintertidal_files("marea", "geometry") == (
        "pyintertidal/marea.py", "pyintertidal/geometry.py")
    with pytest.raises(FileNotFoundError):
        product.code_key("experiments/benchmark/no_such_module.py")
    u = _unit()
    ik = product.inputs_key("P", u, "marea", "", "C", ck)
    assert ik == product.inputs_key({"package_key": "P"}, u.subset_key, "marea", None, "C", ck)
    for args in (("P2", u, "marea", "", "C", ck), ("P", _unit(sel=(1, 2)), "marea", "", "C", ck),
                 ("P", u, "nidem", "", "C", ck), ("P", u, "marea", "v", "C", ck),
                 ("P", u, "marea", "", "C2", ck), ("P", u, "marea", "", "C", "other")):
        assert product.inputs_key(*args) != ik, args
    with pytest.raises(TypeError):
        product.inputs_key(object(), u, "marea", "", "C", ck)


def test_code_key_ignores_line_endings(tmp_path, monkeypatch):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    # the same path with either content: the key depends on the content only
    a = product.code_key(str(lf))
    os.replace(crlf, lf)
    assert product.code_key(str(lf)) == a


def test_npz_written_by_save_is_a_zip_with_fixed_times(tmp_path):
    p = tmp_path / "x.npz"
    product.save_npz(p, {"a": np.arange(3)})
    with zipfile.ZipFile(p) as zf:
        infos = zf.infolist()
    assert [i.filename for i in infos] == ["a.npy"]
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in infos)


def test_git_state_keeps_head_when_the_status_is_slow(monkeypatch):
    import subprocess
    import types as _types
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1] == "rev-parse":
            return _types.SimpleNamespace(stdout="abc123\n", returncode=0)
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert product.git_state() == {"head": "abc123", "dirty": None}
    assert calls[1][:4] == ["git", "status", "--porcelain", "--"]
    assert set(calls[1][4:]) == set(product.CODE_PATHS)
