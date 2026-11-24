# MAREA map viewer

An interactive map of the MAREA results at fourteen estuaries of northern Spain (four drawn inside masks cleaned by
hand, ten preliminary ones inside masks made by a rule and without an accuracy assessment): elevation, uncertainty,
emersion time, delay and tide lines over an orthophoto or a grey map, with a pixel readout and, on wide screens, a 3D
view. It is the whole website: one full-screen page, a React + Tailwind CSS application around MapLibre GL JS, built
with Vite into plain static files that GitHub Pages serves from the `docs/` folder of this branch.

## Run it

Node 22 or newer (built with Node 24 and npm 11).

```bash
cd site
npm ci              # once: installs the exact versions of package-lock.json
npm run dev         # development server: http://localhost:5173/Intertidal_analysis/
npm run build       # static site into ../docs (vite build, then scripts/postbuild.mjs)
npm run preview     # serves the build: http://localhost:4173/Intertidal_analysis/
npm run typecheck   # TypeScript, no output
npm test            # unit tests (Vitest)
```

Every dependency is pinned to an exact version (`.npmrc`: `save-exact=true`); the versions were resolved with
`npm install --before=2026-11-09`, so none was younger than two weeks when the site was built.

## Build variables

The base path is **one** variable. Every asset and data file is resolved from it, so the same code serves any host.

| Variable | Default | Meaning |
|---|---|---|
| `SITE_BASE` | `/Intertidal_analysis/` | Path the site is served under: `/Intertidal_analysis/` for the GitHub Pages site of this repository (`https://<owner>.github.io/Intertidal_analysis/`), `/<repository>/` for another repository, `/` for a custom domain |
| `SITE_OUT_DIR` | `../docs` | Output folder (emptied first; it may not be the site folder or a folder above it) |
| `SITE_PUBLIC_DIR` | `public` | Folder copied unchanged into the build; it replaces `public/` as a whole |
| `SITE_CACHE_DIR` | `node_modules/.vite` | Vite's cache |
| `VITE_REPO_URL` | `https://github.com/coastal-AI/Intertidal_analysis/tree/CARTOGRASS26` | Repository page the info panel links to; its file links and the citation follow it |
| `VITE_ENABLE_3D` | `desktop` | Where the viewer may show its 2D/3D toggle: `on`, `desktop` (wide screens only) or `off` |

Examples:

```bash
npm run build                                                     # GitHub Pages of this repository
SITE_BASE=/marea/ VITE_REPO_URL=https://github.com/<owner>/marea npm run build   # a separate repository
SITE_BASE=/ npm run build                                         # a custom domain
```

On Windows PowerShell, set a variable first: `$env:SITE_BASE = '/marea/'; npm run build`. Git Bash rewrites a
value that starts with `/` into a Windows folder before npm sees it; keep it as typed with
`MSYS2_ENV_CONV_EXCL=SITE_BASE SITE_BASE=/marea/ npm run build`. The build refuses a base path that is a folder.

## What the build writes

`vite build` writes the page (`index.html`) and its assets into `SITE_OUT_DIR`; `scripts/postbuild.mjs` then adds
`.nojekyll`, `licenses/` (the licence texts of the bundled packages, one file each and `THIRD-PARTY-NOTICES.txt`,
linked from the info panel), and runs a scan of the text files for local folders, file URLs and IP addresses and a
size report: the build fails on a hit, above 60 MB in all or above 5 MB for one file. The viewer keeps its state in
the URL hash (`#site=<id>&layer=<id>&base=photo|map&3d=0|1`), so it needs no other page and no 404 fallback.

## Code map

```
site/
  index.html             the page; an inline script applies the theme before the first paint
  vite.config.js         Vite, React, Tailwind, the "@" alias for src/, Vitest
  scripts/
    site-env.mjs         the build variables (shared by vite.config.js and postbuild.mjs)
    postbuild.mjs        .nojekyll, licences, scan, sizes
    data_layers.py       writes public/data/ from the map layers, drawing the overlays from the values (NumPy, Pillow)
  public/                copied as is: favicon.svg, .nojekyll, data/
  src/
    main.tsx, App.tsx    entry; the full-screen frame that lazy-loads the viewer
    config.ts            build-time configuration (base path, repository link, 3D setting)
    content/site.ts      texts of the info panel: title, authors, data sources, citation
    styles/index.css     design tokens (light on :root, dark on .dark) and base styles
    styles/fonts.css     Inter, self-hosted (latin and greek subsets)
    lib/                 assets.ts (URLs under the base path, cached JSON), format.ts (numbers, units),
                         sites.ts (the site list contract), theme.ts, useResource.ts
    components/ui/       the UI kit
    components/layout/   ErrorBoundary
    features/contracts.ts   the props App.tsx passes to the viewer
    features/explorer/   the viewer (its own chunk, with MapLibre GL JS): map, panel, legend, readout, info panel;
                         ramps.json, the colour scales of the maps
```

## Data files

All under `public/data/` (served at `<base>data/`), written by `scripts/data_layers.py`:

```bash
python scripts/data_layers.py --layers <folder of the layer files>    # deterministic: a second run writes the same bytes
```

The layer folder (value images on each site's grid, `heights.json` with each site's NMMA constant C, the tide levels
and lines) was made with the MAREA library outside this branch; only its result, `public/data/`, is published here.
How to read the value images, and turn them into GeoTIFFs:
[`../guide/DATA.md`](../guide/DATA.md#6-map-viewer-files).

Every layer of a site is drawn on one domain, named in the "domain" block of the layer folder's manifest: the pixels
with an elevation inside an intertidal mask cleaned by hand ("hand-cleaned"), or inside a mask made by a rule
("automatic"). The script refuses a layer folder without that block. A site with an automatic mask is listed with
`"preliminary": true`: the viewer lists it in the "Preliminary · automatic mask" group of the site switcher, writes
"preliminary" on its chip on phones, and gives one line on what that means under its name.

Code reads them with `dataUrl()` and `loadJSON()` from `src/lib/assets.ts`, never with a hand-written path.

- `data/sites/index.json`: the site list (schema `cg26-sites/1`, checked by `useSitesIndex()` in `src/lib/sites.ts`):
  for each site, west to east, its names, images and period, centre and bounds, heights (`{datum: "NMMA", C, sigma,
  source}` or `{datum: "model", C: null, note}`), thumbnail, manifest, layer ids and `preliminary` (true or false).
  Paths are relative to `data/`.
- `data/sites/<id>/manifest.json`: the site's grid, layers, legends and tide levels (schema `cg26-explorer-site/2`),
  with its overlay images, value images (code = 256 R + G, value = offset + scale × code, code 0 = no data), tide
  lines and, at Mundaka, lines of equal delay. The viewer draws whatever layers a manifest lists.
- Overlay images: drawn by `data_layers.py` from the value images, not copied: at twice the grid's resolution (2.5 m
  on the ground), each pixel interpolated bilinearly from the grid pixels around it that have a value, the edge of the
  data faded the same way, and coloured with `src/features/explorer/ramps.json`. The viewer draws them with linear
  resampling; the readout reads the value images. An overlay over 4096 px a side comes in tiles, each with its own
  corners, cut where the overlay hardly changes, so the seams do not show.

## Design system

- **Colours**: the poster palette as tokens in `src/styles/index.css`, light on `:root` and dark on `.dark`, mapped
  to Tailwind colours (`canvas`, `surface`, `raised`, `sunken`, `ink`, `muted`, `line`, `control`, `primary`, …).
  `src/styles/contrast.test.ts` checks every text pair at 4.5:1 or more and every control pair at 3:1 or more, in both
  themes.
- **Map colours**: one scale per layer, defined only in `src/features/explorer/ramps.json`, which colours the overlay
  images and the legends alike, with the colours of the poster's maps: elevation matplotlib's viridis (low ground
  dark purple, high ground yellow), uncertainty ColorBrewer Purples (one colour per class), emersion time Greens,
  delay Oranges without its lightest 15 %; all sequential, light to dark (dark to light for the elevation). This
  product includes color specifications and designs developed by Cynthia Brewer (http://colorbrewer.org/).
- **Theme**: the system preference, unless the visitor picks one with the toggle of the control panel (stored in
  `localStorage` when the browser allows it).
- **Type**: Inter (variable); tabular figures for values.
- **UI kit** (`src/components/ui`): `Button`, `ButtonAnchor`, `IconButton`, `Segmented` (native radio group), `Chip`,
  `Slider`, `ColorBar` (legend ramp, OKLab where supported), `Notice`, `ExternalLink`, `Sheet` (the control panel:
  bottom sheet on phones, floating card on wide screens, with the readout in a footer that stays in view),
  `Skeleton`. Components take layout classes through `className`; size and colour come from props.
- **Accessibility**: labelled native inputs, a visible focus ring, `aria-live` for the readout, keyboard panning with
  a crosshair readout, a modal info panel (`<dialog>`), colour never the only cue, `lang="en"`.

## Licences

The site's code is MIT and its data CC BY 4.0, as the rest of the repository. The bundled third-party code keeps its
own licences, copied into `docs/licenses/` by the build. Basemaps (IGN) and terrain tiles are loaded from their
services at run time and are not copied into the repository.
