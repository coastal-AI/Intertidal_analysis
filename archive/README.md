# archive/: earlier material, kept for the record

**Not maintained.** These files show how the project reached the current
method; they are not needed to use MAREA, and many of them no longer run as
they are. Paths inside them refer to where they lived before they were moved
here, some text is in Spanish, and the numbers they show are development
numbers under earlier versions of the method, not validation results. For
the current method, start at the [repository README](../README.md).

| folder | what it is |
|---|---|
| `intertidal/` | the first Python package of the project (docstrings in Spanish), replaced by [`pyintertidal/`](../pyintertidal/README.md) |
| `notebooks/` | the founding and per-site study notebooks, with their outputs: a benchmark of the old and new processing pipelines, a comparison of water-detection methods, tide-model tests at the Ebro and at Foz, the intertidal topography of Villaviciosa and Santander, the north-coast campaign, and the four per-site comparison notebooks (`tide_boundary_comparison_<site>.ipynb`) behind [`../docs/tide_adjustment_comparison.md`](../docs/tide_adjustment_comparison.md) |
| [`research/`](research/README.md) | the research notebooks of the development phase, one line of reasoning each, built from the stored results of the experiments (`results/`, and `runs/`, which is not public); they change directory to a path on the authors' machine |
| [`modelo_generativo/`](modelo_generativo/README.md) | the description, in Spanish, of a dataset of Sentinel-2 images of the Ría de Foz paired with tide heights, made for a generative model; the generator notebook is not published |
| `docs/` | planning notes of August 2026: `INVENTORY.md` (the project's assets against the phases of plan v4) and `STORY.md` (the argument of the method, drafted for the paper, the poster and the repository) |
| `poster_marea_2026-08/` | the earlier MAREA poster draft (August 2026): LaTeX sources, figures, TikZ diagrams and build scripts. It does not build from here: its preamble stayed in `poster/preamble/`, now used by the CARTOGRASS poster |

The notebooks that import `intertidal` were written for a package at the
repository root; it now lives in `archive/intertidal/`. The others import
`pyintertidal` and `experiments` as they were in September 2026, and read
data cubes and products that are not in the repository.
