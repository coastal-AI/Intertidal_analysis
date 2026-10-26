# -*- coding: utf-8 -*-
"""experiments/fixedcfg/dea_intertidal.py + dea_worker.py (DEA Intertidal 2.5.3).

Fast tests (no DEA): a missing side interpreter gives a clear error; the
worker is launched with ``-P``, outside the repository, without PYTHONPATH,
and its temp dir is removed; the streamed input npz round-trips; obs
validation; times; the registry entry and card.

DEA tests (skipped without the side interpreter, or with
MAREA_SKIP_DEA_TESTS=1): ONE worker run of a few hundred KB in a module
fixture. It takes 1.5-4 min on the laptop because
Windows spawns DEA's pool processes and each re-imports DEA (three pools:
two chunks and the gate); on the server (fork) it takes seconds. On a tiny
synthetic site (181 px x 100 scenes) with a planted ramp, a step NDWI,
clouds carrying a garbage NDWI, 3 % flipped observations and two holes in
keep, the adapter:

* recovers the ramp within DEA's expected precision. That precision is set
  by DEA's rolling windows (spacing 1 % and radius 15 % of the tide range R)
  and by the trailing mean over 20 of 200 interpolated intervals (about
  10 % of R). The dry and wet NDWI levels straddle the 0.1 threshold
  symmetrically, so the smoothing adds no bias. What remains is the tide
  sampling of each window and the flipped observations, a fraction of the
  smoothing length: |median error| <= 0.02 R, RMSE <= 0.03 R, max <= 0.08 R.
  The 2026-10-20 run gave 0.002 R, 0.018 R and 0.060 R. A wrong pixel map
  or misaligned tides would give errors of the order of the ramp (0.6 R);
* is bitwise equal to DEA's own unchunked ``elevation()`` (pixel_tides
  replaced by the same h_B broadcast) in z and sigma, across two chunks;
* applies clean_edge_pixels at keep's holes, removing z > 0 neighbours only;
* ran in the side venv, outside the repository, with DEA 2.5.3 byte-exact.

Run:  python -m pytest -q tests/test_fixedcfg_dea.py
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.fixedcfg import dea_intertidal as dea  # noqa: E402
from experiments.fixedcfg import dea_worker as worker  # noqa: E402

RANGE = 4.0                                 # tide range of the synthetic site (m)
HOLE_HIGH = (slice(4, 6), slice(11, 13))    # no-data hole where the bed is above 0
HOLE_LOW = (slice(7, 9), slice(2, 4))       # no-data hole where the bed is below 0


def _side_python():
    try:
        return dea.find_side_python()
    except dea.DEAEnvironmentError:
        return None


SIDE = None if os.environ.get("MAREA_SKIP_DEA_TESTS") == "1" else _side_python()
needs_side = pytest.mark.skipif(SIDE is None, reason="no DEA side interpreter (venv-dea or MAREA_DEA_PYTHON) "
                                                     "or MAREA_SKIP_DEA_TESTS=1")


def synthetic_site(T=100, H=12, W=16, seed=0):
    """A planted ramp z = -1.2 .. +1.2 m across columns under tides uniform in
    [-2, 2] m; NDWI -0.25 (dry) / +0.45 (wet) + N(0, 0.05); 15 % cloudy with
    a wet-looking garbage NDWI (must be masked); 2 % no data; 3 % of the
    observations flipped wet/dry (misclassified); two 2 x 2 holes and a
    corner outside keep."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-RANGE / 2, RANGE / 2, T)
    zt = np.tile(-1.2 + 2.4 * np.arange(W) / (W - 1), (H, 1))
    wet = h[:, None, None] > zt[None]
    flip = rng.random((T, H, W)) < 0.03
    nd = np.where(wet ^ flip, 0.45, -0.25) + rng.normal(0, 0.05, (T, H, W))
    clear = rng.random((T, H, W)) >= 0.15
    nd = np.where(clear, nd, 0.9)                       # clouds: garbage that looks wet
    nd = np.where(rng.random((T, H, W)) < 0.02, np.nan, nd)
    km = np.ones((H, W), bool)
    km[HOLE_HIGH] = False
    km[HOLE_LOW] = False
    km[0, :3] = False
    keep = np.flatnonzero(km.ravel())
    Y = nd.reshape(T, -1)[:, keep].astype(np.float32)
    C = clear.reshape(T, -1)[:, keep]
    t = pd.date_range("2023-01-01 11:00", periods=T, freq="5D", tz="UTC")
    obs = dict(Y=Y, C=C, W=C & (Y > 0), h=h, t=t, keep=keep, shape=(H, W), pixel_m=10, site="synthetic")
    return obs, zt.reshape(-1)[keep], km


def tiny_obs(T=6, P=5, shape=(2, 3)):
    rng = np.random.default_rng(1)
    Y = rng.normal(0, 0.3, (T, P)).astype(np.float32)
    C = rng.random((T, P)) > 0.3
    return dict(Y=Y, C=C, W=C & (Y > 0), h=rng.normal(0, 1, T), t=pd.date_range("2024-01-01", periods=T),
                keep=np.arange(P, dtype=np.int64), shape=shape, pixel_m=10, site="tiny")


# ─────────────────────────────────────────────────────────────────────────────
#  fast tests (no DEA)
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_side_interpreter_clear_error(monkeypatch, tmp_path):
    bad = tmp_path / "no_venv" / "bin" / "python"
    monkeypatch.setenv(dea.ENV_PYTHON, str(bad))
    with pytest.raises(dea.DEAEnvironmentError, match=dea.ENV_PYTHON) as e:
        dea.run(tiny_obs())
    assert str(bad) in str(e.value) and "dea-intertidal" in str(e.value)
    # no variable and no default location: the message lists what was tried
    monkeypatch.delenv(dea.ENV_PYTHON)
    tried = [tmp_path / "a" / "venv-dea" / "bin" / "python", tmp_path / "b" / "python.exe"]
    monkeypatch.setattr(dea, "side_python_candidates", lambda: tried)
    with pytest.raises(dea.DEAEnvironmentError, match="not found") as e:
        dea.find_side_python()
    assert all(str(p) in str(e.value) for p in tried) and dea.ENV_PYTHON in str(e.value)


def test_launch_with_P_outside_repo_without_pythonpath(monkeypatch):
    monkeypatch.setenv(dea.ENV_PYTHON, sys.executable)
    monkeypatch.setenv("PYTHONPATH", ROOT)
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=list(cmd), **kw)
        seen["input_present"] = Path(cmd[cmd.index("--input") + 1]).is_file()
        kw["stdout"].write(b"fake worker: boom\n")
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr(dea.subprocess, "run", fake_run)
    with pytest.raises(dea.DEAWorkerError, match="exit code 7") as e:
        dea.run(tiny_obs(), {"dea_max_workers": 1})
    assert "fake worker: boom" in str(e.value)
    cmd, cwd = seen["cmd"], seen["cwd"]
    assert cmd[0] == sys.executable and cmd[1] == "-P" and cmd[2:4] == ["-X", "utf8"]
    assert Path(cmd[4]) == dea.WORKER_PATH and Path(cmd[4]).is_absolute()
    assert not worker._inside(cwd, ROOT), f"worker cwd {cwd} is inside the repository"
    assert not worker._inside(cmd[cmd.index("--input") + 1], ROOT)
    assert "PYTHONPATH" not in seen["env"] and seen["env"]["TQDM_DISABLE"] == "1"
    assert seen["input_present"]
    assert not os.path.exists(cwd), "the temp dir must be removed after a failure"
    # a temp root inside the repository is refused before anything runs
    seen.clear()
    with pytest.raises(dea.DEAEnvironmentError, match="inside the repository"):
        dea.run(tiny_obs(), {"dea_tmp": os.path.join(ROOT, "results")})
    assert not seen


def test_estimator_seconds_are_the_workers_pixel_steps(monkeypatch, tmp_path):
    """run() reports DEA's own estimator seconds (the worker's per-pixel steps
    and edge cleaning) apart from the side interpreter's start-up, imports
    and the npz I/O (the runner's timing reads meta['seconds_estimator'])."""
    monkeypatch.setenv(dea.ENV_PYTHON, sys.executable)
    obs = tiny_obs()
    P = len(obs["keep"])

    def fake_run(cmd, **kw):
        out = cmd[cmd.index("--output") + 1]
        meta = {"n_candidates": 3, "n_no_obs": 0, "n_edge_removed": 0, "n_sigma_zero": 0, "n_nonfinite_h": 0,
                "n_chunks": 1, "gate": None, "max_workers": 2,
                "seconds": {"pixel_steps": 1.25, "clean_edge_pixels": 0.25, "import": 7.0, "total": 9.0},
                **{k: None for k in ("dea_version", "elevation_py", "elevation_py_sha256", "python", "executable",
                                     "cwd", "sys_path", "versions", "start_method",
                                     "xarray_use_bottleneck_default", "parameters", "used", "tide_dtype")}}
        np.savez(out, z=np.zeros(P, np.float32), sigma=np.zeros(P, np.float32), meta=np.array(json.dumps(meta)))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dea.subprocess, "run", fake_run)
    res = dea.run(obs, {"dea_max_workers": 2, "dea_tmp": str(tmp_path)})
    m = res["meta"]
    assert m["seconds_estimator"] == 1.5 and m["seconds"]["worker_detail"]["import"] == 7.0
    assert m["seconds"]["overhead"] >= 0 and "pixel_steps" in m["seconds"]["estimator_rule"]
    assert m["max_workers"] == 2


def test_input_npz_roundtrip(tmp_path):
    obs = tiny_obs(T=9, P=23, shape=(4, 6))
    Y, C, h, keep, shape, t = dea.obs_arrays(obs)
    t_ns, src = dea.time_ns(t, len(h))
    p = tmp_path / "in.npz"
    dea.write_input(p, Y, C, h, t_ns, keep, shape, {"chunk_px": 7, "x": [1, 2]}, block=5)
    want = np.where(C, Y, np.nan).astype(np.float32).T
    got = np.concatenate([b for _, b in worker.iter_pixel_blocks(p, "ndwi_pt", 7)])
    assert got.dtype == np.float32 and np.array_equal(got, want, equal_nan=True)
    assert [p0 for p0, _ in worker.iter_pixel_blocks(p, "ndwi_pt", 7)] == [0, 7, 14, 21]
    with np.load(p, allow_pickle=False) as f:
        assert np.array_equal(f["h"], h) and np.array_equal(f["keep"], keep)
        assert tuple(f["shape"]) == shape and np.array_equal(f["time_ns"], t_ns)
        assert json.loads(f["params"].item()) == {"chunk_px": 7, "x": [1, 2]}
        assert np.array_equal(f["ndwi_pt"], want, equal_nan=True)
    assert src == "obs t"


def test_obs_validation_and_times():
    obs = tiny_obs()
    bad = dict(obs, h=obs["h"][:-1])
    with pytest.raises(ValueError, match="one h_B per scene"):
        dea.obs_arrays(bad)
    with pytest.raises(ValueError, match="frame"):
        dea.obs_arrays(dict(obs, keep=np.arange(5) + 3))
    with pytest.raises(ValueError, match="repeated"):
        dea.obs_arrays(dict(obs, keep=np.array([0, 1, 1, 2, 3])))
    t = pd.date_range("2024-03-01 11:00", periods=3, freq="D", tz="Europe/Madrid")
    ns, src = dea.time_ns(t, 3)
    assert src == "obs t" and ns[0] == pd.Timestamp("2024-03-01 10:00").value
    ns, src = dea.time_ns(None, 3)
    assert src.startswith("placeholder") and np.all(np.diff(ns) > 0)
    assert dea.default_chunk_px(125) == 50_000 and dea.default_chunk_px(40) == 100_000
    assert dea.default_chunk_px(5000) == 5_000


def test_register_and_card():
    METHODS = {"other": ("x", None, {})}
    assert dea.register(METHODS) is METHODS
    label, fn, card = METHODS["dea"]
    assert fn is dea.run and label == dea.LABEL and card is dea.CARD and "other" in METHODS
    assert card["version"] == "2.5.3" and card["fidelity"] == "F1" and card["needs_full_frame"] is False
    assert card["doi"] == "10.26186/149403" and card["tuning"] == []
    assert any("Bishop-Taylor" in r["cite"] and "doi" in r["doi_or_url"] for r in card["references"])
    items = [d["item"] for d in card["deviations"]]
    assert items[:2] == ["imagery, masks, index", "tide"] and all(d["why"] for d in card["deviations"])
    assert "W" in card["inputs"] and "not used" in card["inputs"]["W"]
    p = card["parameters"]
    assert p["elevation"]["ndwi_thresh"] == 0.1 and p["ds_to_flat"]["ndwi_thresh"] == 0.0
    assert p["pixel_rolling_median"]["window_offset"] == 5 and p["pixel_dem"]["smooth_radius"] == 20
    assert p["pixel_uncertainty"]["min_misclassified"] == 3 and p["elevation"]["min_correlation"] == 0.15


def test_methods_registers_dea_lazily_as_E2(monkeypatch, tmp_path):
    """experiments/fixedcfg/methods.py registers DEA through dea_intertidal.register
    under the catalogue id E2 when the side interpreter exists, and skips it
    cleanly (nothing registered, REGISTRY_SKIPPED says why) when it does not."""
    from experiments.fixedcfg import methods as M

    monkeypatch.setenv(dea.ENV_PYTHON, str(tmp_path / "no_venv" / "bin" / "python"))
    reg, skipped = {}, {}
    assert M.register_dea(reg, skipped) is None and reg == {}
    assert "no side interpreter" in skipped["E2"] and dea.ENV_PYTHON in skipped["E2"]
    monkeypatch.setenv(dea.ENV_PYTHON, sys.executable)
    reg, skipped = {}, {"E2": "stale"}
    assert M.register_dea(reg, skipped) == Path(sys.executable) and skipped == {}
    label, fn, card = reg["E2"]
    assert fn is dea.run and label == dea.LABEL and card["fidelity"] == "F1" and card["id"] == dea.METHOD_ID
    assert "Bishop-Taylor" in card["reference"] and card["references"] == dea.CARD["references"]
    assert all(k in card for k in M.base.CARD_KEYS) and "reference" not in dea.CARD     # the adapter is untouched
    with pytest.raises(ValueError):
        M.register_dea(reg, {})                                               # never twice
    # the module's own registry: E2 registered or skipped, never under 'dea', never an error
    assert ("E2" in M.METHODS) != ("E2" in M.REGISTRY_SKIPPED)
    assert "dea" not in M.METHODS and "dea_intertidal" not in M.REGISTRY_ERRORS
    assert list(M.METHODS)[:6] == ["E4", "E5a", "E5b", "E5c", "E7", "E12"]            # built-ins stay first


# ─────────────────────────────────────────────────────────────────────────────
#  DEA (one worker run in the side interpreter)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dea_run():
    obs, zt, km = synthetic_site()
    P = len(obs["keep"])
    cfg = {"dea_chunk_px": P // 2 + 1, "dea_max_workers": 1, "dea_gate": True}
    res = dea.run(obs, cfg)
    return obs, zt, km, res


def _interior(obs, km):
    """Keep pixels whose four neighbours are all in keep (or off the frame)."""
    H, W = obs["shape"]
    pad = np.pad(km, 1, constant_values=True)
    ok = km & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return ok.reshape(-1)[obs["keep"]]


@needs_side
def test_dea_recovers_planted_ramp(dea_run):
    obs, zt, km, res = dea_run
    z, s, meta = res["z"], res["sigma"], res["meta"]
    assert z.dtype == np.float32 and s.dtype == np.float32 and z.shape == zt.shape == s.shape
    inner = _interior(obs, km)
    fin = np.isfinite(z)
    assert fin[inner].mean() >= 0.95, f"coverage of the interior {fin[inner].mean():.3f}"
    err = (z - zt)[fin]
    med, rmse, mx = float(np.median(err)), float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err)))
    assert abs(med) <= 0.02 * RANGE, med
    assert rmse <= 0.03 * RANGE, rmse
    assert mx <= 0.08 * RANGE, mx
    # DEA's uncertainty: finite exactly where z is, >= 0, and > 0 where flips misclassify >= 3 scenes
    assert np.array_equal(np.isfinite(s), fin) and np.all(s[fin] >= 0)
    assert meta["sigma_nan_where_z_nan"] and np.mean(s[fin] > 0) > 0.25   # 0.55 on 2026-10-20
    assert meta["n_z"] == int(fin.sum()) and meta["coverage"] == meta["n_z"] / len(z)
    assert meta["n_chunks"] == 2 and meta["n_nonfinite_h"] == 0 and meta["time_source"] == "obs t"


@needs_side
def test_dea_identity_gate_bitwise(dea_run):
    _, _, _, res = dea_run
    g = res["meta"]["gate"]
    assert g is not None and g["passed"], g
    assert g["z"]["bitwise"] and g["sigma"]["bitwise"] and g["z"]["n_differ"] == 0


@needs_side
def test_dea_clean_edge_pixels_at_keep_holes(dea_run):
    obs, zt, km, res = dea_run
    H, W = obs["shape"]
    z, keep = res["z"], obs["keep"]
    frame = np.full(H * W, np.nan)
    frame[keep] = z
    frame = frame.reshape(H, W)

    def ring(sl):
        m = np.zeros((H, W), bool)
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        m[r0 - 1, c0:c1] = m[r1, c0:c1] = True
        m[r0:r1, c0 - 1] = m[r0:r1, c1] = True
        return m & km

    high, low = ring(HOLE_HIGH), ring(HOLE_LOW)
    assert np.all(np.isnan(frame[high])), "z > 0 pixels next to no data must be cleaned"
    assert np.isfinite(frame[low]).mean() >= 0.75, "z <= 0 pixels next to no data are kept"
    assert res["meta"]["n_edge_removed"] >= high.sum()


@needs_side
def test_dea_ran_in_side_venv_outside_repo(dea_run):
    _, _, _, res = dea_run
    d = res["meta"]["dea"]
    assert d["dea_version"] == "2.5.3" and d["elevation_py_sha256"] == worker.ELEVATION_PY_SHA256
    assert not worker._inside(d["cwd"], ROOT), d["cwd"]
    assert all(os.path.normcase(os.path.realpath(p)) != os.path.normcase(os.path.realpath(ROOT))
               for p in d["sys_path"] if p)
    venv = Path(res["meta"]["interpreter"]).resolve().parents[1]
    assert worker._inside(d["elevation_py"], venv), (d["elevation_py"], venv)
    assert not worker._inside(d["elevation_py"], os.path.join(ROOT, "intertidal"))
    assert d["used"]["ndwi_thresh"] == 0.1 and d["used"]["windows_n"] == 100
    assert d["tide_dtype"].startswith("float32")


def test_worker_imports_light():
    """The worker's top level (also run by DEA's spawned pool processes and by
    these tests in the main venv) must not import DEA or xarray."""
    code = ("import sys; sys.path.insert(0, %r); import experiments.fixedcfg.dea_worker as w; "
            "print(int('intertidal' in sys.modules), int('xarray' in sys.modules))" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "0"]
