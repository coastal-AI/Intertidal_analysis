# data/: the RTK-GNSS survey of the Ría de Villaviciosa

The authors' field survey of the intertidal flats of the Ría de Villaviciosa
(Asturias, Spain): 361 points measured on 14 August 2026 with an RTK-GNSS
receiver, every one with a fixed (FIX) RTK solution. With the PNOA airborne
LiDAR, it is one of the two independent checks of elevation at
Villaviciosa; it is never used to make the maps.

| file | contents |
|---|---|
| `villaviciosa_rtk_gnss.csv` | the survey as exported by the receiver: one row per point, with longitude, latitude and ellipsoidal height, the antenna height, the solution status and correction type, the precision of each coordinate (RMS), the averaging time and the satellites used |
| `villaviciosa_rtk_gnss_alicante.csv` | the same points with one more column, `MSL height (Alicante REDNAP)`: the height above the Spanish national vertical datum (Alicante) |

## How the code reads it

Read the survey through `pyintertidal.rtk.load_rtk()`. It places the points
on the canonical 10 m grid of Villaviciosa
([`../products_villaviciosa/`](../products_villaviciosa/README.md)) and
returns the **development points** only. The survey is split into
development points and a reserved part; the split is computed in
`pyintertidal/rtk.py`, not stored in these files, so the files hold every
point.

`tests/test_r1_guard.py` checks that no file of `pyintertidal/` or
`experiments/` asks for the reserved part, except
`experiments/v3_open_reserved.py`, which logs every opening to
`results/v3_aperturas.jsonl`. That log holds one opening, on 2026-08-21
([`../experiments/README.md`](../experiments/README.md), section 3), so the
reserved points are not an untouched blind test. Validation v2 scores the
development points only. The guard binds this repository's code, not the
files: anyone can read the CSVs.

## Heights and comparisons

MAREA's elevations are on the vertical scale of the tide series they were
inverted on (about mean sea level), not on a national datum. Every
comparison with the survey therefore removes the median offset first and
reports it separately as the bias
([`../docs/products.md`](../docs/products.md#conventions)).
