"""truth on a site whose stores cover fewer years than the default (2026-10-20).

``truth`` has no ``--years``: its site carries the default years, and at a
site whose record covers one year (ferrol 2024, escalda and ems 2023, the
CDSE credits ran out) the series step got the default years and refused them
("years [2024, 2025] are not in the record"). The truth command now passes the
record's own years.
"""
import dataclasses
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import validation_v2 as v2  # noqa: E402
from experiments.benchmark import package  # noqa: E402
from tests import benchmark_zarr_site as zs  # noqa: E402


@pytest.fixture
def one_year(tmp_path):
    """A synthetic site whose record covers 2024 only (default years 2023-2025)."""
    mp = pytest.MonkeyPatch()
    zs.small(mp)
    inp = zs.make_inputs(str(tmp_path / "inputs"))
    base = zs.site2_for(inp, tmp_path / "out")
    provider = zs.FakeTide(nan_at=[inp["nan_instant"]])
    package.prepare_v2(dataclasses.replace(base, years=(2024,)), steps=("record",),
                       provider=provider)
    yield base, provider
    mp.undo()


def test_series_on_a_one_year_record_needs_the_record_years(one_year):
    base, provider = one_year
    assert tuple(base.years) == (2023, 2024, 2025)
    assert package._read_record(base)["years"] == ["2024"]
    with pytest.raises(package.PackageError, match="not in the record"):
        package.prepare_v2(base, steps=["series"], provider=provider)
    ran = package.prepare_v2(base, steps=["series"], years=[2024], provider=provider)
    assert list(ran) == ["series"]


class _Stop(Exception):
    pass


def test_truth_passes_the_record_years_to_the_series_step(monkeypatch, tmp_path):
    seen = {}
    s2 = types.SimpleNamespace(site="ems", boundary="union", years=(2023, 2024, 2025),
                               truth_dir=str(tmp_path / "truth"))

    class OneYearPackage:
        def __init__(self, site2):
            self.record = {"boundary": {"kind": "union"}}
            self.T = 145
            self.years = [2023]

        def provider(self):
            return "the provider"

    def prepare_v2(site2, steps=None, years=None, provider=None, **kw):
        seen.update(steps=list(steps), years=years, provider=provider)
        raise _Stop("stop after the series call")

    monkeypatch.setattr(v2, "_sites", lambda site, res: ["ems"])
    monkeypatch.setattr(v2, "_site2", lambda site, args, n_sites=1: s2)
    monkeypatch.setattr(v2, "_instants_line", lambda rec: "catalogue")
    monkeypatch.setattr(package, "V2Package", OneYearPackage)
    monkeypatch.setattr(package, "check_record_instants", lambda site2, rec: None)
    monkeypatch.setattr(package, "prepare_v2", prepare_v2)
    args = types.SimpleNamespace(site="ems", part=None, windows="Y2023", res=10)
    assert v2.cmd_truth(args) == 1          # the stand-in stops the run after the call
    assert seen == {"steps": ["series"], "years": [2023], "provider": "the provider"}
