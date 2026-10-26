# -*- coding: utf-8 -*-
"""experiments/v18_compute_cost.py on tiny synthetic data: the stage
stopwatch and its memory probe, the gates before timing, the equal-input
helpers (record levels, NIDEM composites at keep, the observation rule, the
pre-flight), the MAREA and NIDEM stage groups (reproduced bitwise against the
direct calls), the bootstrap resample against bootstrap_lags, the optional
cube stages on a tiny netCDF cube, the log parsers, the report, the output
guard and the RTK guard.

Run:  python -m pytest -q tests/test_compute_cost.py
"""
import ast
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402
from experiments import v18_compute_cost as v18  # noqa: E402
from pyintertidal import marea  # noqa: E402
from pyintertidal import tide_estimators as te  # noqa: E402
from pyintertidal import waterline_contour as wc  # noqa: E402

M2_H = 12.4206012


class FakeBoundary:
    """A boundary like PyTMDBoundary: levels returned sorted by time, every
    call counted."""

    def __init__(self, amp=1.4):
        self.amp, self.calls = amp, 0

    def levels(self, times):
        t = pd.DatetimeIndex(times)
        self.calls += 1
        hrs = t.sort_values().values.astype("datetime64[s]").astype(np.int64) / 3600.0
        return self.amp * np.sin(2 * np.pi * hrs / M2_H) + 0.2 * np.sin(2 * np.pi * hrs / 12.0)


def _times(T, seed):
    rng = np.random.default_rng(seed)
    return pd.DatetimeIndex(sorted(pd.Timestamp("2024-01-01 11:00")
                                   + pd.to_timedelta(rng.uniform(0, 900, T), unit="D")))


# ── measurement ─────────────────────────────────────────────────────────────

def test_stage_warms_up_once_times_and_sees_memory():
    calls = []

    def fn():
        calls.append("run")
        a = np.ones(12_000_000)                       # 92 MiB, held for 80 ms
        time.sleep(0.08)
        return float(a[0])

    for mode, expect in (("small", ["warm_s", "run"]), ("full", ["warm_f", "run"]), ("none", ["run"])):
        calls.clear()
        rec = v18.Recorder(None, "synthetic", warmup=mode)
        v, row = rec.stage("screened", "marea", "inversion", fn, lambda: calls.append("warm_s"),
                           lambda: calls.append("warm_f"), n_scenes=5, n_px=7)
        assert v == 1.0 and calls == expect
        assert row["wall_s"] >= 0.08 and row["cpu_s"] >= 0 and row["status"] == "ok"
        assert row["warmup"] == (mode if mode != "none" else "none")
        assert (row["warmup_wall_s"] is None) == (mode == "none")
        assert row["work_mb"] > 40, row                # the 92 MiB allocation is seen
        assert row["peak_rss_mb"] >= row["rss_start_mb"]


def test_memprobe_sampling_fallback_sees_a_peak():
    p = v18.MemProbe(interval_s=0.002, force_sampling=True).start()
    a = np.ones(8_000_000)                            # 61 MiB
    time.sleep(0.05)
    del a
    m = p.stop()
    assert "sampled" in m["mem_method"] and m["work_mb"] > 30


def test_threads_gate():
    one = {"env": {}, "pools": [{"api": "blas", "num_threads": 1}, {"api": "openmp", "num_threads": 1}],
           "max_threads": 1}
    two = {"env": {}, "pools": [{"api": "blas", "num_threads": 2}], "max_threads": 2}
    none = {"env": {"OMP_NUM_THREADS": "1"}, "pools": None, "max_threads": None}
    assert v18.threads_gate(one)["ok"] and not v18.threads_gate(two)["ok"]
    g = v18.threads_gate(none)
    assert g["ok"] and g["verified"] is False


def test_time_refuses_a_busy_machine_before_any_data(tmp_path, monkeypatch):
    monkeypatch.setattr(v18, "load1", lambda: 50.0)
    out = tmp_path / "busy"
    with pytest.raises(SystemExit):
        v18.main(["time", "villaviciosa", "--out", str(out), "--max-load", "6"])
    j = json.load(open(out / "timing_villaviciosa.json", encoding="utf-8"))
    assert j["status"] == "refused" and j["gates"]["machine_quiet"]["ok"] is False
    assert not (out / "timing_villaviciosa.csv").exists()     # nothing was timed


# ── equal-input helpers ─────────────────────────────────────────────────────

def test_record_levels_on_record_instants_in_row_space():
    times = _times(30, 1)
    rows = np.array([2, 5, 9, 17, 29])
    B = FakeBoundary()
    table = v14.LevelTable(B, times[rows])
    lev = v18.RecordLevels(table, rows, len(times))
    lev.ensure([0, 15, 40])
    assert B.calls == 1                                   # one call, the record instants only
    for tau in (0, 15, 40):
        full = lev[tau]
        assert full.shape == (30,) and np.isnan(np.delete(full, rows)).all()
        direct = FakeBoundary().levels(times[rows] - pd.Timedelta(minutes=tau))
        assert np.array_equal(full[rows], direct)
    with pytest.raises(ValueError):
        v18.RecordLevels(table, rows[:3], 30)


def _frame(T=48, H=9, W=11, seed=4, nan_obs=6):
    """A sloping flat seen T times at random levels: raw NDWI with a few
    undefined values under a clear flag."""
    rng = np.random.default_rng(seed)
    h = rng.permutation(np.linspace(-1.2, 1.2, T))
    yy, xx = np.mgrid[0:H, 0:W]
    zs = -1.0 + 2.0 * xx / (W - 1.0) + 0.1 * np.sin(yy / 2.0)
    ndwi = (np.where(h[:, None, None] > zs[None], 0.45, -0.3) + rng.normal(0, 0.03, (T, H, W))).astype(np.float32)
    clear = rng.random((T, H, W)) < 0.85
    t_, r_, c_ = np.nonzero(clear)
    pick = rng.choice(len(t_), nan_obs, replace=False)
    ndwi[t_[pick], r_[pick], c_[pick]] = np.nan          # clear but undefined NDWI
    ndwi[~clear & (rng.random((T, H, W)) < 0.2)] = np.nan
    return h, ndwi, clear


def test_composites_keep_equal_composites_from_arrays_bitwise():
    h, ndwi, clear = _frame()
    T, H, W = ndwi.shape
    j = wc.interval_assignment(h)[0]
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(H * W, 70, replace=False))
    Y = ndwi.reshape(T, -1)[:, keep].copy()
    C = clear.reshape(T, -1)[:, keep].copy()
    bad = v18.clear_nan_exceptions(Y, C)
    assert sum(len(v) for v in bad.values()) == int((C & np.isnan(Y)).sum()) > 0
    Y0 = np.nan_to_num(Y, nan=0.0)
    ck = v18.composites_keep(Y0, C, bad, np.arange(T), j, block=16)       # several blocks
    fr = v18.to_frame(ck, keep, (H, W))
    # the reference: ITEM's composites of the frame with every pixel outside keep unobserved
    cl = clear.reshape(T, -1).copy()
    outside = np.setdiff1d(np.arange(H * W), keep)
    cl[:, outside] = False
    ref = wc.composites_from_arrays(ndwi, cl.reshape(T, H, W), j)
    for k in ("median", "count", "sd_mean", "sd_mean_bounded", "n_ndwi_above_bound", "n_ndwi_below_minus_bound"):
        assert np.array_equal(fr[k], ref[k], equal_nan=True), k
    assert fr["max_abs_ndwi"] == ref["max_abs_ndwi"] and fr["shape"] == (H, W)
    # a column slice is the same columns of the whole
    part = v18.composites_keep(Y0, C, bad, np.arange(T), j, 10, 40, block=7)
    assert np.array_equal(part["median"], ck["median"][:, 10:40], equal_nan=True)
    assert np.array_equal(part["count"], ck["count"][:, 10:40])
    # the pre-flight counts (NIDEM's observation rule) equal the composites' count
    counts, _ = wc.clear_counts_by_interval(v18.ObsRows(C, bad, np.arange(T)), None, j)
    assert np.array_equal(counts, ck["count"].astype(np.int32))


def _nidem_ctx(h, ndwi, clear, record="all"):
    T, H, W = ndwi.shape
    keep = np.arange(H * W)
    Y = ndwi.reshape(T, -1).copy()
    C = clear.reshape(T, -1).copy()
    bad = v18.clear_nan_exceptions(Y, C)
    return types.SimpleNamespace(
        site="synthetic", rows={record: np.arange(T)}, P=H * W, h_space=np.asarray(h, float),
        Y0=np.nan_to_num(Y, nan=0.0), C0=C, bad=bad, keep=keep, shape=(H, W), frame_px=H * W,
        stored={record: {"npz": None, "diag": None}}, gates={}, outputs={},
        recorder=v18.Recorder(None, "synthetic"), warmup="small")


def test_nidem_group_reproduces_nidem_from_arrays():
    h, ndwi, clear = _frame(T=60, H=24, W=28, seed=6)
    ctx = _nidem_ctx(h, ndwi, clear)
    v18.nidem_group(ctx, "all")
    ref, (j, _, Z, U, sizes) = wc.nidem_from_arrays(ndwi, clear, h)
    o = ctx.outputs["all"]["nidem"]
    assert o["computable"] and o["sizes"] == sizes.tolist()
    assert o["n_vertices"] == ref["n_vertices"] > 0
    assert o["filtered_keep_px"] == int(np.isfinite(ref["filtered"]).sum()) > 0
    assert ctx.gates["all"]["nidem_preflight_counts_vs_composites"]["ok"]
    assert ctx.gates["all"]["nidem_intervals_vs_stored_run"]["checked"] is False
    st = [(r["stage"], r["status"]) for r in ctx.recorder.rows]
    assert st == [("preflight", "ok"), ("composites", "ok"), ("contours_tin", "ok")]


def test_nidem_group_not_computable_when_an_interval_is_unobserved(tmp_path):
    h, ndwi, clear = _frame(T=60, H=12, W=14, seed=7)
    j = wc.interval_assignment(h)[0]
    clear[j == 1] = False                                  # every scene of interval 1 overcast
    ctx = _nidem_ctx(h, ndwi, clear)
    # a stored run that was NOT COMPUTABLE too: the status gate agrees
    diag = tmp_path / "nidem_diagnostics.json"
    diag.write_text(json.dumps({"status": "NOT COMPUTABLE", "reason": "pre-flight"}), encoding="utf-8")
    ctx.stored["all"] = {"npz": str(tmp_path / "missing.npz"), "diag": str(diag)}
    v18.nidem_group(ctx, "all")
    o = ctx.outputs["all"]["nidem"]
    assert o["computable"] is False and "expected valid share" in o["reason"]
    st = [(r["stage"], r["status"]) for r in ctx.recorder.rows]
    assert st == [("preflight", "ok"), ("composites", "not_computable"), ("contours_tin", "not_computable")]
    g = ctx.gates["all"]["nidem_intervals_vs_stored_run"]
    assert g["checked"] and g["ok"] and g["stored"] == "NOT COMPUTABLE"


# ── MAREA ───────────────────────────────────────────────────────────────────

def _marea_ctx(T=90, per=120, lag=40, seed=8):
    """A two-band site whose inner band floods ``lag`` minutes late, its
    'production' product made by the direct calls."""
    rng = np.random.default_rng(seed)
    times = _times(T, seed)
    lev = v14.LevelTable(FakeBoundary(amp=1.6), times)
    lev.ensure([0, lag])
    band = np.repeat([0, 1], per)
    P = 2 * per
    z = rng.uniform(-1.2, 1.2, P)
    Hl = np.where(band[None, :] == 0, lev[0][:, None], lev[lag][:, None])
    Y = (np.where(Hl > z[None, :], 0.4, -0.3) + rng.normal(0, 0.04, (T, P))).astype(np.float32)
    C = rng.random((T, P)) < 0.9
    s_km = np.where(band == 0, rng.uniform(0.2, 2.8, P), rng.uniform(3.2, 5.8, P))
    fit_kw = dict(sigma0=0.2, sg_grid=(0.06, 0.1, 0.15, 0.22), tau_bounds=marea.TAU_BOUNDS, z_points=40,
                  max_px_band=1200, monotone=True)
    m2a = dict(Y=Y, C=C, band=band, centers=np.array([1.5, 4.5]), nb=2, band_km=3.0, fit_kw=fit_kw,
               seed=20260819)
    rows = np.arange(T)
    lev.ensure(marea.BANK_TAUS)
    tau, _, tp = v14.refit_lags(types.SimpleNamespace(m2a=m2a, levels=lev, s_km=s_km), rows)
    lev.ensure(np.unique(tp))
    h0 = lev[0]
    z_ref = np.full(P, np.nan)
    for tv in np.unique(tp):
        cols = np.flatnonzero(tp == tv)
        z_ref[cols], _ = marea.invert_series(Y[:, cols], C[:, cols], lev[tv], float(h0.min()), float(h0.max()))
    ctx = types.SimpleNamespace(
        site="synthetic", rows={"screened": rows}, n_space=T, P=P, t_space=times, h_space=np.array(h0, float),
        boundary=FakeBoundary(amp=1.6), boundary_name="fake", m2a=m2a, s_km=s_km,
        res={"tau_min": np.asarray(tau).tolist(), "n_boot": 30}, tau_px_all=np.asarray(tp, float),
        z_stored=z_ref, Y0=Y, C0=C, bad={}, keep=np.arange(P), shape=(12, 20), frame_px=240, gates={},
        outputs={}, recorder=v18.Recorder(None, "synthetic"), warmup="small")
    return ctx, tau, tp, z_ref


@pytest.mark.parametrize("warmup", ["small", "full", "none"])
def test_marea_group_reproduces_the_direct_product_and_passes_its_gates(warmup):
    ctx, tau, tp, z_ref = _marea_ctx()
    ctx.recorder = v18.Recorder(None, "synthetic", warmup=warmup)
    ctx.warmup = warmup
    v18.scene_levels_group(ctx, "screened")
    v18.marea_group(ctx, "screened", boot=1, boot_here=True)
    g = ctx.gates["screened"]
    assert g["scene_levels_vs_build_record"]["ok"]
    assert g["lag_fit_vs_result_json_tau_min"]["ok"] and g["lag_fit_vs_result_json_tau_min"]["max_abs_diff_min"] == 0
    assert g["pixel_lags_vs_production_tau_px_min"]["differing"] == 0
    assert g["quality_rule_post_filter_vs_invert_series"]["ok"]
    gi = g["inversion_vs_stored_product"]
    assert gi["ok"] and gi["validity_flips"] == 0 and gi["n_abs_dz_gt_tol"] == 0   # bitwise in practice
    assert v18.failed_gates(ctx.gates) == []
    o = ctx.outputs["screened"]["marea"]
    assert abs(o["tau_bands_min"][1] - 40.0) < 8.0 and o["tau_bands_min"][0] == 0.0
    assert o["estimates_mb0"] == int(np.isfinite(z_ref).sum()) and o["bootstrap_resamples"] == 1
    st = [r["stage"] for r in ctx.recorder.rows]
    assert st == ["scene_levels", "lagged_levels", "lag_fit", "bootstrap_resample", "inversion"]
    assert all(r["status"] == "ok" and r["wall_s"] >= 0 for r in ctx.recorder.rows)
    # two boundary calls in all: lag 0, then the bank and every whole-minute lag
    assert len(ctx.tables["screened"][0].calls) == 2
    expect = {"small": "setup's boundary calls", "full": "full", "none": "none"}[warmup]
    assert ctx.recorder.rows[1]["warmup"] == expect
    assert (ctx.recorder.rows[2]["warmup_wall_s"] is None) == (warmup == "none")


def test_bootstrap_resample_equals_bootstrap_lags():
    ctx, *_ = _marea_ctx(T=60, per=80, seed=9)
    lev = v18.RecordLevels(v14.LevelTable(FakeBoundary(amp=1.6), ctx.t_space), np.arange(60), 60)
    lev.ensure(marea.BANK_TAUS)
    m = ctx.m2a
    bank = te.ShiftBank(marea.BANK_TAUS, [lev[t] for t in marea.BANK_TAUS])
    wet = m["C"] & (m["Y"] > 0)
    ref = te.bootstrap_lags(wet, m["C"], bank, m["band"], m["centers"], m["nb"], n_boot=3, seed=m["seed"],
                            **{**m["fit_kw"], "max_px_band": v18.BOOT_PX_BAND})
    draws = v18.bootstrap_draws(60, 3, m["seed"])
    for b in range(2):
        mine = v18.bootstrap_resample(m, lev, np.arange(60), draws[b], b, m["seed"])
        assert np.allclose(mine, ref[b], rtol=0, atol=1e-9)


def test_compare_z_tolerates_a_rare_flip_only():
    z = np.linspace(-1, 1, 200_000)
    assert v18.compare_z(z, z)["ok"]
    z2 = z.copy()
    z2[:2] = np.nan                                       # 2 flips in 200,000 = 1e-5
    g = v18.compare_z(z2, z)
    assert g["validity_flips"] == 2 and g["ok"]
    z2[:10] += 0.05
    assert not v18.compare_z(z2, z)["ok"]


# ── the cube stages on a tiny netCDF cube ──────────────────────────────────

def _tiny_cube(path, T=40, H=16, W=18, seed=5):
    import pyproj
    import xarray as xr

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-03", periods=T, freq="5D")
    h = rng.permutation(np.linspace(-1.2, 1.2, T))
    yy, xx = np.mgrid[0:H, 0:W]
    zs = -1.0 + 2.0 * xx / (W - 1.0) + 0.1 * np.sin(yy / 2.0)
    nd = np.where(h[:, None, None] > zs[None], 0.45, -0.3) + rng.normal(0, 0.03, (T, H, W))
    b08 = np.full((T, H, W), 1000.0) + rng.normal(0, 5, (T, H, W))
    b03 = b08 * (1 + nd) / (1 - nd)
    cloudy = rng.random((T, H, W)) < 0.12
    scl = np.where(cloudy, 9.0, np.where(nd > 0, 6.0, 5.0))
    xs = 300000.0 + 10.0 * np.arange(W)
    ys = 4800000.0 - 10.0 * np.arange(H)
    ds = xr.Dataset({"B03": (("t", "y", "x"), b03.astype(np.float32)),
                     "B08": (("t", "y", "x"), b08.astype(np.float32)),
                     "SCL": (("t", "y", "x"), scl.astype(np.float32))},
                    coords={"t": dates, "y": ys, "x": xs})
    ds["crs"] = xr.DataArray(0, attrs={"crs_wkt": pyproj.CRS.from_epsg(32630).to_wkt()})
    ds.to_netcdf(path)
    return h


def test_cube_stages_reproduce_the_published_path(tmp_path):
    cube = str(tmp_path / "cube.nc")
    h = _tiny_cube(cube)
    ex = marea.extract(cube)
    keep = ex["keep"]
    assert len(keep) > 50
    T = len(h)
    j, _, Z, U, _ = wc.interval_assignment(h)
    comp = wc.stream_composites(cube, np.arange(T), j, rule="item", verbose=False)
    stored = wc.nidem(comp, Z, U)
    npz = str(tmp_path / "nidem.npz")
    np.savez_compressed(npz, filtered=stored["filtered"])
    ctx = types.SimpleNamespace(
        site="synthetic", rows={"all": np.arange(T)}, P=len(keep), h_space=h, ti_space=np.arange(T),
        keep=keep, cube=cube, shape=ex["shape"], frame_px=int(np.prod(ex["shape"])),
        stored={"all": {"npz": npz, "diag": None}}, gates={}, outputs={}, recorder=v18.Recorder(None, "synthetic"),
        warmup="small", Y0=np.nan_to_num(ex["Y"], nan=0.0), C0=ex["C"].astype(bool))
    v18.nidem_frame_group(ctx, "all")
    assert ctx.gates["all"]["nidem_frame_filtered_vs_stored_bitwise"]["ok"]
    assert [r["stage"] for r in ctx.recorder.rows] == ["prepass", "preflight", "composites", "contours_tin"]
    v18.marea_extract_group(ctx)
    assert ctx.gates["marea_extract_vs_stored_extraction"]["ok"]


# ── harvest ─────────────────────────────────────────────────────────────────

V7_LOG = """[v7] escalda (SENSITIVITY (NIDEM on MAREA's 68-scene record), ITEM pixel-quality mask) -> out
  grid 1467 x 2101, 464 cube scenes, keep 291,799 px; pixel mask 'item'
  pre-pass (207 s): observations on keep 18,168,804
  record: 68 scenes (2023-01-18 .. 2025-12-30), boundary consensus
  [composites] rows 0-256 of 1467 (4 s)
  [composites] rows 1280-1467 of 1467 (31 s)
  NIDEM: 251,217 contour vertices, empty levels [], 280,504 filtered px (8 s)
  wrote out/nidem_diagnostics.json; total 368 s, peak 867 MB
"""
V7_NC = """[v7] ems (primary: ITEM all-observations record, ITEM pixel-quality mask) -> out
  grid 3206 x 3413, 462 cube scenes, keep 1,886,417 px; pixel mask 'item'
  pre-pass (590 s): observations on keep 250,190,341
  record: 462 scenes (2023-01-02 .. 2025-12-30)
[v7] NOT COMPUTABLE: ems: pre-flight stop
"""
STATUS = """2026-10-19T20:52:50Z START nidem ems
2026-10-19T20:52:50Z START nidem escalda
2026-10-19T20:59:02Z OK nidem escalda
2026-10-19T21:16:05Z OK nidem ems
"""


def test_harvest_nidem_logs(tmp_path):
    (tmp_path / "nidem_escalda.log").write_text(V7_LOG, encoding="utf-8")
    (tmp_path / "nidem_ems.log").write_text(V7_NC, encoding="utf-8")
    (tmp_path / "STATUS").write_text(STATUS, encoding="utf-8")
    rows, files = v18.harvest_nidem_logs(str(tmp_path))
    d = pd.DataFrame(rows)
    e = d[d["site"] == "escalda"].set_index(["stage", "quantity"])["value"]
    assert e[("prepass", "wall_s")] == 207 and e[("composites", "wall_s")] == 31
    assert e[("contours_tin", "wall_s")] == 8 and e[("total", "wall_s")] == 368
    assert e[("total", "rss_at_end_mb")] == 867
    assert set(d.loc[d["site"] == "escalda", "record"]) == {"screened"}
    assert set(d["concurrent_runs"]) == {2}
    m = d[d["site"] == "ems"]
    assert set(m["record"]) == {"all"} and (m["quantity"] == "not_computable").any()
    assert int(m["n_px"].iloc[0]) == 1886417


def _cell(src, start, end, text=""):
    return {"cell_type": "code", "source": src,
            "metadata": {"execution": {"iopub.execute_input": start, "shell.execute_reply": end}},
            "outputs": [{"output_type": "stream", "name": "stdout", "text": text}] if text else []}


def test_harvest_notebook_cells(tmp_path):
    nb = {"cells": [
        _cell('SITE = "ems"\n', "2026-10-02T09:33:42.0Z", "2026-10-02T09:33:42.5Z"),
        _cell("reference_map, cloud_pct = pit.reference_and_clouds(cube)", "2026-10-02T09:35:17Z",
              "2026-10-02T10:16:15Z"),
        _cell("water_freq = pit.water_frequency(cube, dates=usable)", "2026-10-02T10:16:18Z", "2026-10-02T10:31:07Z"),
        _cell("viz.plot_water_frequency(water_freq, transform)", "2026-10-02T10:31:07Z", "2026-10-02T10:31:08Z"),
        _cell("marea_res = marea.reconstruct(cube.cache_path)", "2026-10-02T10:36:23Z", "2026-10-02T12:15:56Z",
              "extracted and saved x.npz\n1,886,417 intertidal px in the record\nMAREA record: 54 of 462 cube "
              "scenes\n[Ems] bootstrap of 30 scene resamples [611 s]\n[Ems] listo en 1522 s -> out\n"),
        _cell("z_plain, sg = marea.invert_series(Y, C, h0, lo, hi)", "2026-10-02T12:15:56Z", "2026-10-02T12:28:29Z",
              "plain inversion: 1,716,119 px  [69 s]\nMAREA inversion: 1,770,621 px, 108 distinct lags  [727 s]\n")],
        "metadata": {}}
    (tmp_path / "tide_boundary_comparison_ems.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    rows, _ = v18.harvest_notebooks(str(tmp_path))
    r = pd.DataFrame(rows)
    assert (r["stage"] == "water_frequency").sum() == 1     # the plotting cell is not the computation
    d = r.set_index("stage")["value"]
    cell = (pd.Timestamp("2026-10-02T12:15:56") - pd.Timestamp("2026-10-02T10:36:23")).total_seconds()
    assert d["water_frequency"] == 889
    assert d["screening"] == 2458 and d["reconstruct"] == 1522 and d["bootstrap"] == 611
    assert d["extraction"] == cell - 1522
    assert d["plain_inversion"] == 69 and d["marea_inversion_with_per_lag_calls"] == 727 - 69
    assert set(r["site"]) == {"ems"} and set(r["n_scenes"].dropna()) == {54}


def test_harvest_v14_settings_and_job_scaling(tmp_path):
    st = {"site": "escalda", "lags": "fixed", "blas": {"OMP_NUM_THREADS": "2"},
          "runs": [{"started": "2026-10-20T07:30:54", "finished": "2026-10-20T08:00:52"}],
          "timing": {"load_seconds": 78.9, "gate_inversion_seconds_mb0": 8.744, "gate_N": 68, "inverted_px": 200000,
                     "level_calls": [{"n_lags": 9, "instants": 4176, "seconds": 5.6}]}}
    (tmp_path / "escalda_fixed_settings.json").write_text(json.dumps(st), encoding="utf-8")
    N = np.array([10, 20, 40, 80, 160, 464] * 2)
    pd.DataFrame({"site": "escalda", "lags": "fixed", "pool": "all", "pool_size": 464, "N": N, "mb": 0,
                  "n_set": 200000, "seconds": 2.0 + 0.1 * N}).to_csv(tmp_path / "escalda_fixed.csv", index=False)
    rows, files = v18.harvest_v14(str(tmp_path))
    d = pd.DataFrame(rows).set_index(["stage", "quantity"])["value"]
    assert d[("inversion", "us_per_px_scene")] == pytest.approx(1e6 * 8.744 / (68 * 200000))
    assert d[("job_vs_N", "job_s_per_scene")] == pytest.approx(0.1)
    assert d[("job_vs_N", "job_s_intercept")] == pytest.approx(2.0)
    assert d[("job_vs_N", "job_s_per_scene_per_Mpx")] == pytest.approx(0.5)


def test_harvest_granadeiro_and_overlaps(tmp_path):
    (tmp_path / "granadeiro_escalda.log").write_text(
        "[v8] Granadeiro et al. 2021 (faithful) at escalda -> out; 16 R workers\n"
        "  scene rule: 33 of 464 2023-2025 scenes\n"
        "  wrote x.json; total 4621 s; python peak 754 MB, python + R peak 2956 MB\n", encoding="utf-8")
    rows, _ = v18.harvest_granadeiro([str(tmp_path / "granadeiro_*.log")])
    d = pd.DataFrame(rows).set_index("quantity")["value"]
    assert d["wall_s"] == 4621 and d["python_plus_r_peak_mb"] == 2956
    t = pd.Timestamp
    c = v18.overlap_counts({"a": (t("2026-01-01 10:00"), t("2026-01-01 11:00")),
                            "b": (t("2026-01-01 10:30"), t("2026-01-01 12:00")),
                            "c": (t("2026-01-01 11:30"), t("2026-01-01 13:00")), "d": (None, None)})
    assert c == {"a": 2, "b": 3, "c": 2, "d": None}


# ── report ──────────────────────────────────────────────────────────────────

def _timing_rows():
    base = dict(site="villaviciosa", tag="", n_px=32120, frame_px=837225, warmup="small", threads="openblas:1",
                cpu_util=1.0, mem_method="psutil", inputs_mb=1.0, rss_start_mb=300.0)
    rows = []
    for record, n in (("screened", 97), ("all", 465)):
        for method, stage, w, mem in (("shared", "scene_levels", 1.0, 290.0), ("marea", "lagged_levels", 1.5, 300.0),
                                      ("marea", "lag_fit", 4.0, 40.0), ("marea", "inversion", 3.0, 25.0),
                                      ("nidem", "preflight", 0.2, 5.0), ("nidem", "composites", 2.0, 80.0),
                                      ("nidem", "contours_tin", 5.0, 120.0)):
            st = "ok"
            if record == "all" and stage in ("composites", "contours_tin"):
                st, w, mem = "not_computable", float("nan"), float("nan")
            rows.append(dict(base, record=record, n_scenes=n, method=method, stage=stage, status=st, wall_s=w,
                             cpu_s=w, work_mb=mem, peak_rss_mb=300 + (mem if mem == mem else 0), n_out=1,
                             note=("NOT COMPUTABLE: test" if st != "ok" else "")))
    rows.append(dict(base, record="screened", n_scenes=97, method="marea", stage="bootstrap_resample", status="ok",
                     wall_s=6.0, cpu_s=6.0, work_mb=10.0, peak_rss_mb=310.0, n_out=2, note=""))
    return rows


def test_report_end_to_end(tmp_path):
    out = tmp_path / "rep"
    v18.guard_out(str(out))
    pd.DataFrame(_timing_rows(), columns=v18.TIMING_COLUMNS).to_csv(out / "timing_villaviciosa.csv", index=False)
    (out / "timing_villaviciosa.json").write_text(json.dumps({
        "site": "villaviciosa", "status": "complete", "machine": {"node": "test", "cpu_count": 4},
        "gates": {"threads_pinned_to_1": {"ok": True}, "screened": {"inversion_vs_stored_product": {"ok": True}}},
        "outputs": {"screened": {"marea": {"bootstrap_n_production": 30}}},
        "inputs": {"pools": {"screened": {"size": 97}, "all": {"size": 465}},
                   "inputs": {"extract": {"grid": [915, 915], "n_keep": 32120}}}}), encoding="utf-8")
    empty = tmp_path / "v14"
    empty.mkdir()
    summary, asrun, facts = v18.cmd_report(types.SimpleNamespace(out=str(out), harvest=None, timing=None,
                                                                 v14_dir=str(empty)))
    by = {(r["record"], r["method"]): r for r in summary}
    m, n = by[("screened", "marea")], by[("screened", "nidem")]
    assert m["compute_s"] == pytest.approx(4.0 + 3.0) and m["wall_s"] == pytest.approx(1.0 + 1.5 + 4.0 + 3.0)
    assert m["n_tide_calls"] == 2 and m["tide_s"] == pytest.approx(2.5)
    assert n["compute_s"] == pytest.approx(0.2 + 2.0 + 5.0) and n["wall_s"] == pytest.approx(1.0 + 7.2)
    assert n["n_tide_calls"] == 1
    # working memory of the computing stages; the tide calls apart
    assert n["work_mb"] == 120.0 and m["work_mb"] == 40.0 and m["tide_work_mb"] == 300.0
    assert by[("all", "nidem")]["status"] == "not_computable"
    assert by[("all", "nidem")]["detect_s"] == pytest.approx(1.0 + 0.2)
    assert by[("screened", "marea_bootstrap")]["wall_s"] == pytest.approx(3.0)
    for f in ("compute_cost.csv", "inputs_table.csv", "site_facts.csv", "compute_cost.md", "compute_cost.es.md",
              "compute_cost_en.png", "compute_cost_es.png", "compute_cost_en.pdf", "compute_cost_es.pdf"):
        assert (out / f).exists(), f
    en = (out / "compute_cost.md").read_text(encoding="utf-8")
    es = (out / "compute_cost.es.md").read_text(encoding="utf-8")
    assert "NOT COMPUTABLE" in en and "NO CALCULABLE" in es and "What each method needs" in en
    assert "Qué necesita cada método" in es and "compute_cost_es.png" in es
    it = pd.read_csv(out / "inputs_table.csv")
    assert list(it.columns) == v18.T["en"]["cols_inputs"] and len(it) == len(v18.INPUT_ROWS)


# ── guards ──────────────────────────────────────────────────────────────────

def test_output_guard(tmp_path):
    for bad in ("products_marea/x", "results/scene_count_curves/v18", "results/server_10m/v18",
                "results/method_comparison_same_scenes", "experiments/out", "."):
        with pytest.raises(SystemExit):
            v18.guard_out(bad)
    ok = tmp_path / "new"
    assert v18.guard_out(str(ok)) == str(ok) and (ok / v18.MARKER).exists()
    v18.guard_out(str(ok))                                # its own folder again: fine
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.txt").write_text("someone else's", encoding="utf-8")
    with pytest.raises(SystemExit):
        v18.guard_out(str(other))
    logs_only = tmp_path / "run"                          # the server creates logs/ first
    (logs_only / "logs").mkdir(parents=True)
    (logs_only / "logs" / "escalda.log").write_text("started", encoding="utf-8")
    assert v18.guard_out(str(logs_only)) == str(logs_only)


def test_rtk_guard_passes():
    from tests import test_r1_guard
    static = [n for n in dir(test_r1_guard) if n.startswith("test_static")]
    assert static
    for n in static:
        getattr(test_r1_guard, n)()                       # scans experiments/, this script included
    tree = ast.parse(open(v18.__file__, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "attr", None) == "load_rtk" or getattr(n.func, "id", None) == "load_rtk")]
    assert all(not c.args and not c.keywords for c in calls)
