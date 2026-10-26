# notebooks/

Two Jupyter notebooks. Run them with the repository root as the working
directory: they read their inputs by paths relative to the root.

| notebook | what it shows | what it needs |
|---|---|---|
| [`pixel_walkthrough_villaviciosa.ipynb`](pixel_walkthrough_villaviciosa.ipynb) | one real pixel of the Ría de Villaviciosa through MAREA: its looks, the candidate delays, the likelihood step by step, the decision and the inversion. It is the source of [`../docs/one_pixel_marea_example.md`](../docs/one_pixel_marea_example.md) ([Spanish](../docs/one_pixel_marea_example.es.md)) and keeps its outputs, so it can be read on GitHub without running it | the 10-year Villaviciosa netCDF cube and a MAREA product made from it, neither in the repository; development state of September 2026 (it still shows the 10-minute detection threshold, later removed) |
| [`validation/00_explore_zarr_cubes.ipynb`](validation/00_explore_zarr_cubes.ipynb) | a look inside the Zarr validation cubes: dates and bands, an RGB view, the water indices of one date, the SCL and CLD cloud masks, the time series of one pixel and a grid of quick looks; it reads one date, one pixel or a reduced view at a time | the stores `cubes/<site>/<year>.zarr`, downloaded with `experiments/download_cube_zarr.py` (see the [repository README](../README.md#minimal-examples), example 3); stored without outputs |

The notebooks of the validation results (the lag benchmark, the
scene-count curves of MAREA and NIDEM, the fixed-configuration tables) are
being written and will be added with the results, once the validation is
complete. [`../docs/README.md`](../docs/README.md) lists them.
