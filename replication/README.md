# replication/: the offline replication kit of the Villaviciosa product

* **What this folder is for:** rerunning steps 6-11 of the Villaviciosa product (the tide at the mouth, the sections, the delays, every pixel's elevation and the hypsometry) with no account, no download and no tide model.
* **Who it is for:** reviewers.
* **Time and needs:** under 2 minutes; the pinned environment of [`../requirements.txt`](../requirements.txt) (set-up in [`../guide/REPRODUCE.md`](../guide/REPRODUCE.md)); 12.5 MB of disk for the kit, already in the repository.

The kit holds what steps 1-5 produced from the authors' 10-year cube, which
is not yet public: the record of the 97 screened scenes of 2023-2025 at the
32,120 intertidal pixels, and the EOT20 tide at the mouth at the instants
the method reads. Use it with Route A of
[`../guide/REPRODUCE.md`](../guide/REPRODUCE.md); the step numbers and the
conventions for times and heights are those of
[`../scripts/marea_tools.py`](../scripts/marea_tools.py) (times UTC, heights in metres on the scale
of the tide at the mouth).

## Files

| file | contents | dtype/shape | sha256 | made by |
|---|---|---|---|---|
| `villaviciosa_record_2023-2025.npz` (10,997,346 bytes) | the record (steps 1-5): `Y`, the NDWI of every look; `C`, whether the look is clear; `keep`, the flat indices of the intertidal pixels in the frame; `dates` of the 97 scenes; the frame `shape`; the masks `sea` and `inter` (from all 1379 scenes); `bbox`, the box of the registered Villaviciosa polygon (west, south, east, north, degrees), which the production call passed with the record; `crs_wkt`, the cube's CRS (EPSG:32630); and `source_cube_sha256` | `Y` (97, 32120) float32; `C` (97, 32120) bool; `keep` (32120,) int64; `dates` (97,) str; `shape` (2,) int64 = (915, 915); `sea`, `inter` (915, 915) bool; `bbox` (4,) float64 | `21e222759d6ea5f3e2cf955bdced9932a70b7c21f4401016d9f4b8f876e0fdde` | `marea.extract` on the authors' cube, rows cut to the 97 record dates of `products_marea/result.json` |
| `villaviciosa_eot20_levels.npz` (97,379 bytes) | EOT20 at the centroid of the registered Villaviciosa polygon (43.509024 N, 5.416135 W), at the 97 record instants, for every whole-minute delay from -45 to 180 min: `levels_m[i, j]` is the level at `times_ns[j]` minus `shifts_min[i]` minutes; also `times_utc`, `dates`, `model`, `lat`, `lon` | `times_ns` (97,) int64, ns since 1970 UTC; `shifts_min` (226,) int16; `levels_m` (226, 97) float64, m | `6d6f675c35ae1b411fc1ae34372d7b9c03da4890ace3447465be0bc80a6360ef` | `PyTMDBoundary("EOT20", *pyintertidal.sites.get("villaviciosa").centroid)`, one call for all 21,922 instants |
| `villaviciosa_eot20_10min_2023-2025.npz` (1,379,419 bytes) | the continuous EOT20 series at the same point every 10 min, from 2022-12-31 21:00 to 2025-12-31 23:50 UTC: the first 180 min are the margin for delays up to 180 min; for the emersion time; also `step_min` (10), `margin_min` (180), `model`, `lat`, `lon` | `times_ns` (157842,) int64, ns since 1970 UTC; `levels_m` (157842,) float32, m | `d80ad559a9b373f5bd1a5e3da07a1bf7bc6a4339fc72a316f544d4198c19f488` | `PyTMDBoundary` at the same point |
| `MANIFEST.json` | the sha256, size and arrays of each file above; the inputs (the sha256 of the extraction, of the source cube and of the three product files, the EOT20 point and files); the checks made while building; the licences | JSON | – | `build_kit.py` |
| `build_kit.py` | the script that built the kit | Python | – | – |

The source cube is `ndwi_cube_villaviciosa_grande_10y.nc` (sha256
`5f3ccb2bf9162037798111b86cc295e77fe39159480e665a4911a18c53aaa337`,
3,092,569,908 bytes; not public yet).

## How it was made

On 2026-10-27, on a Windows 10 laptop, at a development commit (`67f80cf`, not published), from
the repository root:

```bash
python -X utf8 replication/build_kit.py --extract <marea_extract.npz> --tide-dir <tide_models> --cube <cube> --hash-cube
```

`--extract` is the extraction the product was made from (`marea.extract`
over all 1379 scenes of the authors' cube, 142.5 MB, not public); `--cube`
gives the CRS text exactly as the cube stores it, and `--hash-cube`
re-checks the cube's sha256 against `products_marea/result.json`. The script
reads only the 97 record rows of the extraction, so its memory stays small.
`MANIFEST.json` records the commit, the time (`made_utc`), Python 3.13.3 and
the versions of numpy, pandas, pyTMD and eo-tides.

Before it writes `MANIFEST.json`, the script checks, and stops if one
fails:

* the extraction's pixels (`keep`) and frame (`shape`) are those of
  `products_marea/marea.npz`;
* every record date of `products_marea/result.json` is in the extraction;
* the levels at zero delay span exactly `rango_marea_m` of
  `products_marea/result.json` (-2.172 to 1.146 m);
* the batched EOT20 call gives the same levels, bit for bit, as one call per
  delay, for the delays -45, 0, 6, 12, 16, 90 and 180 min.

Route A reads the table through `scripts/marea_tools.TableBoundary`, which
returns the stored level at an exact instant and raises a `KeyError` for any
other instant: it never interpolates, so a run that strays from the record
fails loudly instead of reading a wrong tide. The 10-min series is read by
`marea_tools.TideSeries`, which interpolates linearly between samples and
raises a `ValueError` outside its period.

## Use

```bash
python -X utf8 scripts/reproduce_villaviciosa.py --from-kit replication --out runs/villaviciosa_repro
```

The script first checks every kit file against its sha256 in
`MANIFEST.json` and stops if one differs. You can check them yourself with
`certutil -hashfile <file> SHA256` (Windows) or `sha256sum <file>` (Linux).
The expected output and the check table are in Route A of
[`../guide/REPRODUCE.md`](../guide/REPRODUCE.md); on 2026-10-27 all 41 checks
passed in 57 s (20 s with `--fast`).

## Licences

* The record contains modified Copernicus Sentinel data: the looks of
  2023-2025, and masks computed from the scenes of 2016-2025, processed with
  openEO on the Copernicus Data Space Ecosystem.
* The tide files hold values of the EOT20 model (Hart-Davis et al. 2021,
  [DOI 10.17882/79489](https://doi.org/10.17882/79489)), CC BY 4.0.
* The kit's data files are released under CC BY 4.0
  ([`../LICENSE-docs.md`](../LICENSE-docs.md)); `build_kit.py` under MIT
  ([`../LICENSE`](../LICENSE)).

## Next steps

* Run Route A: [`../guide/REPRODUCE.md`](../guide/REPRODUCE.md).
* What each array of the product means: [`../guide/DATA.md`](../guide/DATA.md).
* The method, step by step: [`../guide/METHOD.md`](../guide/METHOD.md).
* The same steps on an estuary of your own: [`../guide/NEW_ESTUARY.md`](../guide/NEW_ESTUARY.md).
