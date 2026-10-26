# experiments/benchmark/: the validation v2 harness

The code behind `python -X utf8 -m experiments.validation_v2 <command>`. It
was written for the controlled benchmark v13 ([`../../docs/v13_benchmark_spec.md`](../../docs/v13_benchmark_spec.md))
and extended for validation v2 ([`../../docs/validation_v2_spec.md`](../../docs/validation_v2_spec.md),
which supersedes v13 where the two differ). Every module cites the section of
the specification it implements. Validation v2 is in progress: the modules
below are those committed through milestone V2 (the lag benchmark); the
specification (§12) lists the ones still to come (`reader.py`, `blocks.py`,
`report.py` and most elevation adapters in `methods/`).

## How the pieces connect

```
fetch/*            external inputs: boundary gauges (IOC), in-estuary gauges, tide models,
                   EMODnet, CMEMS, scene metadata, LiDAR and Vaklodingen provenance
   |
inputs.py          the scenes every method reads, from the Zarr stores (experiments/zarr_cubes.py)
obs.py             THE observation: cloud masks, band preprocessing, water indices, wet/dry detection
domain.py          the pixel domain and the along-estuary geometry (3-km sections)
package.py         the common input package of a site, built once and shared by every method
levels.py          the water level each method receives (LB boundary, LT per-pixel model,
                   LM regional model, LG gauge network)
   |
truth.py, judges.py   the truths: in-estuary gauge lags; RTK, LiDAR and Vaklodingen elevations
settings.py, designs.py   the factors (cloud, index, detection, preprocessing, window, number of
                          images) and the design tables that say which units to run
   |
runner.py, queue_v2.py, instance.py   the run queue: claims, resume, logs, memory and threads
lags/*, methods/*                     the method adapters; each writes one product (product.py)
   |
readout.py         one read-out rule for every method's lag
scoring.py, inference.py, lagtable.py   scores, bootstrap tests and tables
gates.py, control.py   reproduction gates and a synthetic positive control, run before scoring
lock.py            the pre-registration lock (LOCK.json) written before results are seen
timing.py          the stage stopwatch and the timing runs
```

## Modules

| file | what it does (spec section) |
|---|---|
| `config.py` | sites, constants, seeds and the sectioned FROZEN plan; paths of every output |
| `inputs.py` | the scene inputs from the Zarr stores, the per-scene BOA offset, the inputs report (§1.2, §1.10, §1.11) |
| `obs.py` | the observation every method receives (cloud masks C0-C8 and C11, the ones implemented so far; indices; detectors D1-D5) |
| `domain.py` | the pixel domain and the geometry along the estuary (§1.5, §2.6, §3.1) |
| `package.py` | the common input package of a site (§1.4, §2.2, §2.5-§2.11) |
| `levels.py` | the boundary tide and the other water-level sources (§1.3, §2.7) |
| `truth.py` | the truths every method is scored against (§1.7, §5.1, §5.5) |
| `judges.py` | the in-estuary gauges that judge every lag, and the rules that admit them (§1.6, §2.8) |
| `settings.py` | the settings factors as registries, with a hash per entry (§2) |
| `designs.py` | the design tables, the plan CSV and the ten-day cap (§5, §13.3, §14) |
| `instance.py` | what every lag method of one cell receives |
| `runner.py` | the unit queue: plan, claims, resume, logs, resources |
| `queue_v2.py` | the instance-major run queue of the lag benchmark (§5.1, §13.2-§13.5) |
| `product.py` | units, products, statuses, atomic writes and code keys |
| `readout.py` | lag read-outs: one rule for every method's output (§6.3) |
| `scoring.py` | lag scoring against the in-estuary truths |
| `inference.py` | bootstrap weights, sign-flip tests and families of the lag inference (§6.4, §6.5, §9) |
| `lagtable.py` | the lag tables (fixed-column CSVs) and their markdown report |
| `gates.py` | reproduction and identity gates (§10.1) |
| `control.py` | the positive control: a synthetic site in the Zarr format with a planted delay (§10.2) |
| `pilots.py` | the truth-free computability of the curve points |
| `lock.py` | the pre-registration lock `LOCK.json` (§9.8, §9.9) |
| `timing.py` | stage stopwatch and dedicated timing runs (§8, §13.5) |
| `sensing_time.py` | an optional diagnostic: each scene's sensing time from its sun angles (not a gate, not a setting) |
| `extract.py` | a small Zarr extract of a legacy netCDF cube, for local checks only |
| `fetch/` | `ioc` (boundary gauges), `judges` (in-estuary gauge series), `tides` (per-pixel global tide model), `cmems` (regional models), `emodnet` (EMODnet Bathymetry DTM), `stac_meta`, `sensing`, `pnoa_meta`, `vak_meta` |
| `lags/` | the lag methods: MAREA's family (`marea_family`: L01, L02, L03, L05), band aggregates (`aggregate`: L04, L07, L10), `bt26` (L12), `gm20` (L08, Geyman and Maloof 2020), `hd26` (L09, L10), `ppl` (L11), global tide models at the gauges (`models`: L14, L15), regional models (`regional`: L16), zero lag and shallow-water celerity (`physics`: L17, L18), the gauge-network reference (`references`: L20); shared helpers in `common` |
| `methods/` | elevation adapters; so far `elev_marea_pair` (E0a, the inversion with no delay; E0b, MAREA; E0g, MAREA's inversion with the band delays measured by the in-estuary gauges) |
| `vendor/` | third-party code vendored verbatim: the NREL Solar Position Algorithm from pvlib 0.13.0 (BSD-3-Clause, `LICENSE-pvlib.txt`) |

## Outputs

`results/validation_v2/` (10 m, the reported numbers) holds `common/<site>/`
(input packages), `truth/<site>/`, `runs/<site>/`, `gates/<site>/`, `logs/`,
`LOCK.json` and `attain.json`; `results/validation_v2_20m/` holds local
checks on the 20 m legacy cubes only, never reported. The heavy files are
never committed (`results/validation_v2/.gitignore`). See
[`../../docs/products.md`](../../docs/products.md).

## Editing these files

Products, gates, truths and the lock record the sha256 of the source files
they depend on (`product.code_key`, `gates.GATE_CODE`, `truth._code_shas`,
the adapters' `CODE_DEPS`, `settings.entry_sha`). Changing one of those files,
even a comment, marks the products and gate verdicts that rest on it as
stale. Change them only between runs, and expect the affected units to rerun.

## Tests

`tests/test_benchmark_*.py`, on the small synthetic sites of
`tests/benchmark_site.py` and `tests/benchmark_zarr_site.py`
([`../../tests/README.md`](../../tests/README.md)).
