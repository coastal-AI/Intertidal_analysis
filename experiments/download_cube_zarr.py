# -*- coding: utf-8 -*-
"""Download the validation cubes again as Zarr (nine bands, and their sun/view angles), per year.

What
----
For the validation v2 (indices NIR/NDWI/MNDWI/AWEI/WRI, SCL and CLD cloud
filters, water detection) the five validation sites are fetched again from
the Copernicus Data Space openEO backend with the bands
``B02 B03 B04 B08 B8A B11 B12 SCL CLD`` of SENTINEL2_L2A, on the SAME 10 m
grids as the existing netCDF cubes, and kept as ONE ZARR STORE PER SITE AND
YEAR, ``cubes/<site>/<year>.zarr`` (format: ``experiments/zarr_cubes.py``;
read any window with ``zarr_cubes.open_cube(site, start, end)``).

The process is the one of the existing cubes (``SentinelCube.ensure`` and,
for Villaviciosa, the notebook download of ``ndwi_cube_villaviciosa_grande_
10y.nc``): ``load_collection(SENTINEL2_L2A, extent, [start, end), bands,
max_cloud_cover=100)`` -> ``resample_spatial(resolution=10, method="near")``
-> ``save_result``, with THREE INTENDED DIFFERENCES and no other (a test
compares the two graphs): more bands, ``format="Zarr"``, and the SPATIAL
EXTENT, which is no longer the site's lon/lat box but the site's reference
frame in its UTM CRS, ``{"west", "south", "east", "north", "crs": <EPSG
code>}`` (the bounding-box form of ``load_collection``'s ``spatial_extent``
in the openEO process specification: ``crs`` an integer EPSG code, default
4326; the client passes the dict through unchanged).

Why the UTM extent. The netCDF path of the backend snapped a lon/lat box to
the native 10 m lattice of the Sentinel-2 tiles (the existing cubes: pixel
edges on multiples of 10 m in UTM). Its Zarr path does not (format test of
2026-10-13): the grid is anchored on the upper-left corner of the box
reprojected to UTM (305408.66 / 4822375.74 for the 2 km test box), so pixel
centres sat 0.134 and 0.450 px off the lattice of the reference cube, which
with nearest-neighbour resampling mislabels native pixels by up to half a
pixel; the arrays were also larger than asked (256 x 256 px for a ~206 x
207 px box: whole chunks of 128 px) and x/y came as float32 (the graph's
``resample_spatial`` carries the client's default ``align="upper-left"``,
unchanged from the netCDF cubes' graph). So the request
is an extent whose edges ARE lattice lines: the reference frame of the site
(the existing 10 m netCDF cube's pixel centres +- 5 m, exactly, e.g.
Villaviciosa 301020..310170 x 4815690..4824840 in EPSG:32630; a site
without that cube on the machine: a complete store of the site, else the
lon/lat box projected to UTM and snapped OUTWARD to multiples of 60 m, a
multiple of the 10 and 20 m pixels: ``FRAME_SNAP_M``). ``--lonlat-extent``
sends the old lon/lat box instead, to reproduce the old request for
diagnosis (its offset grid is refused unless ``--allow-grid-offset``). The
stores record the extent sent (``spatial_extent``), where it came from
(``spatial_extent_source``) and the old lon/lat box
(``spatial_extent_lonlat``). A changed extent changes the process graph and
so the job title: an old download or job is never reused for it.

Resampling: B02/B03/B04/B08 are native
10 m (the call is the identity for them); B8A/B11/B12, SCL and CLD are 20 m
and reach 10 m by NEAREST NEIGHBOUR, the pipeline's method (its only 20 m
band so far was SCL): no new interpolation, and the 20 m bands can be
aggregated back exactly. The FORMAT REPORT measures it on the delivered
data (share of pixel pairs inside one 20 m cell holding the same value:
100 % = nearest from the native grid, measured on the dates with most valid
pixels). A store whose grid is in another CRS or off the existing cube's
10 m lattice is refused; a store on that lattice with another extent is
cropped / padded (with the fill) to the reference grid, so every year of a
site has the same grid. The reference grid is the site's 10 m netCDF cube,
else a complete store of the site, else the bbox snapped outward to 60 m
(the log says which). A PRODUCTION run refuses, before any job is made,
a request whose frame is that snapped box (``--allow-snapped-frame``
overrides): such a store would sit on the native lattice but not on the
netCDF frame, and would then be the reference of the site's other years; the
dry run marks those rows. Delivered x/y in float32 are on the reference
lattice when every value is the float32 rounding of a lattice value (pixel
centres in whole metres below 2^24 m are exact in float32, so a delivery on
the lattice deviates by 0; anything beyond half a float32 step, 0.016-0.03 m
in x and 0.25 m in y, is not rounding) and then take the lattice's exact
float64 values; otherwise the lattice check decides on the regular axis
through their median, and an offset is refused. Arrays larger than the
request (the Zarr writer pads to whole chunks) are cropped onto the
reference frame AT THE REGISTERED OFFSET (below: the delivered coordinates
do not place the pixels), and the FORMAT REPORT says what the cropped-away
margin held (a note: the fill, or data outside the frame, on a sample of
dates). A delivery that does not cover the frame
(it would be padded with the fill, which reads as no-data inside the frame)
stops a production year (``--allow-padding`` overrides); --test only warns.
A time axis in ``milliseconds since 01/01/1970 - 00:00:00`` (and
other ``<unit> since <date>`` forms, any case, day-first slashes included)
is decoded explicitly; float times narrower than float64 that cannot resolve
a second at their values are refused. Values are kept as delivered (int16, fill -32768);
nothing is rescaled. The no-data value is the one the delivered data
DECLARE (``_FillValue``/``nodata`` attribute) or a Zarr ``fill_value`` of
-32768; zarr's default ``fill_value`` 0 is never taken as no-data on its
own (0 is clear sky in CLD): if -32768 occurs in the data it is the no-data,
otherwise the run stops (AMBIGUOUS NO-DATA) unless ``--nodata`` is given.

Registration (2026-10-19)
-------------------------
Every main store written from the backend's Zarr until 2026-10-19 was
displaced by an integer number of pixels against the legacy netCDF cube of
its site (same grid, same dates): the delivered arrays are padded (e.g.
1152 x 1152 px for Villaviciosa's 915 x 915 px frame) and their x/y LABEL
them as if the data started at the requested corner, but the frame's data
start at row dy, column dx of them (Villaviciosa (72, 72), Ferrol (4, 2),
Escalda (5, 1), Wadden (0, 5), Ems (288, 271); constant across the years of
a site; the backend's rule is unknown). The coordinates therefore never
place the pixels of a padded delivery; ``register_delivery`` establishes the
offset (dy, dx) of the frame in the delivered (padded) arrays, and every
band (the 10 m ones, the nearest-resampled 20 m ones, SCL, CLD) is cropped at
``raw[dy:dy + H, dx:dx + W]`` with the frame's coordinates:

* against the site's reference netCDF when it covers the year: the common
  dates whose delivered SCL is clear (classes 4/5/6 on >=
  REGISTRATION_MIN_CLEAR of its valid pixels, counted one date at a time in
  the part of the arrays inside the frame at every offset) are tried
  clearest first, at most REGISTRATION_MAX_TRIED, until REGISTRATION_DATES
  count: per date, the masked FFT cross-correlation of the reference's B08
  frame against the FULL delivered array (every offset 0 <= dy <= rows - H,
  0 <= dx <= cols - W; no cyclic shift; B03 must confirm) must peak with
  Pearson r >= REGISTRATION_MIN_R_ANY on the frame, REGISTRATION_MIN_MARGIN
  above the best offset outside the peak (REGISTRATION_PEAK_PX) and a
  CENTRED peak (REGISTRATION_MAX_SUBPIXEL); a date that does not (its
  reference frame without data, a mosaic that differs) is set aside and
  listed, a date whose peak passes ELSEWHERE refuses the year; at least
  REGISTRATION_MIN_DATES dates must count, all at one offset, at least
  REGISTRATION_MIN_DATES of them with B08 r >= REGISTRATION_MIN_R; the
  delivered SCL must not match the reference's better within
  REGISTRATION_SCL_SEARCH_PX of that offset. ``--offset-px ROWS,COLS`` must
  agree with it (when the registration cannot measure without contradicting
  it, e.g. too few clear dates, the expert offset stands unconfirmed, with a
  warning, and the verification after writing decides);
* a year the reference does not cover (campaign 2): ``--offset-px`` is
  CHECKED against a store of the site registered against its reference (same
  extent and grid): the per-pixel MEDIAN COMPOSITES of the
  REGISTRATION_SIBLING_DATES clearest dates of this delivery (full arrays)
  and of that store (its frame) must not put the frame elsewhere (refused),
  and confirm it when they can (recorded); when nothing can check it, a
  delivery of another array shape than the site's stores' is refused unless
  ``--accept-unverified-offset``. Without ``--offset-px``, the offset of a
  registered store of the SAME request (same extent, grid and delivered
  shape) is taken only when those composites put the frame there for B08
  and B03 (peak rules; r >= REGISTRATION_SIBLING_MIN_R: two years'
  composites differ). Single dates of two years cannot do this (haze the SCL
  calls clear, product geolocation 1 px apart): composites can;
* arrays of the frame's shape whose x/y agree have no other offset (in a
  year without reference checked within REGISTRATION_FRAME_SIZED_PX of (0, 0)
  against a registered store's composites); arrays smaller than the frame
  are placed by their coordinates (padding rules);
* the 20 m layer (B8A B11 B12 SCL CLD) must not contradict the offset: the
  delivery's own B8A against its B08 frame (REGISTRATION_20M_*), the SCL
  against the reference's (above), and the 20 m cells of the placed frame
  at phase (0, 0) of the 20 m lattice (REGISTRATION_PHASE_*).

Anything else refuses the year before a store is written (--test warns and
keeps the coordinates). The store records ``delivered_offset_px``,
``delivered_shape_px``, ``registration_method``, ``registration_reference``,
``registration_dates``, ``registration_r`` and the whole ``registration``
record. After writing, the main store is verified against the reference
(``verify_and_complete``): 'identical', or NEAR-IDENTICAL (on the identical
grid, B03 and B08 equal on >= 99.9 % and SCL on >= 99 % of the pixel-dates
valid in either store: VERIFY_MIN_EQUAL; no date of the reference missing;
no date displaced) marks it complete; anything else leaves it NOT complete
and stops the year with a message pointing at ``<year>.verify.json`` (the
raw download is kept), which says per date whether the store is in place
(``geometry_check``) and where the values differ (per 100-pixel block), so
``--accept-differences`` (recorded) is decided on evidence: it accepts
differing values, never a displaced store; a store written from the same
delivery and refused is then verified again, not rewritten.
``--renormalise`` rebuilds the stores of the sites / years given from their
kept raw delivery (``<year>.part``, kept by ``--keep-raw``) without any
network call or job: the raw normalised again with the registration and
written, the store there moved aside to ``<year>.zarr.superseded-<UTC>``
(never deleted) only once the new one is whole (its sidecars copied with the
same suffix first), the new one verified, and ``<year>.verify.json``,
``<year>.format.txt``, ``<year>.job.json`` and the ledger's notes updated;
refused when the raw is missing or the disk cannot hold a second copy.
``--renormalise --dry-run`` prints the measured offsets, the 20 m checks and
what would be moved, and does nothing. The angles stores are placed by their
x/y (no band to register them); a padded angles delivery is recorded and
flagged (``angles_placement``, ``verify_angles``).

One openEO batch job per year (windows [YYYY-01-01, YYYY+1-01-01), end-
exclusive like ``experiments/download_cube.py``), with the same
``JOB_OPTIONS``. Each year: job -> download into ``<year>.part/`` ->
normalise into ``<year>.zarr`` (registered; every block read back and
compared with what the backend delivered) -> compare with the existing
netCDF cube of the site where it covers the year (``<year>.verify.json``;
the store is marked complete only when the comparison accepts it) ->
overpass times of the
new dates into ``cubes/<site>/overpass_times.json`` (a NEW file; the
pipeline's files are only read). Resumable: a finished year is skipped; a
year that failed is the only one repeated; a job left running by a killed
session is reattached (job id in ``<year>.part/job.json``, or found by its
title on the backend) instead of paid for twice; a job that failed leaves its
backend error log in ``<year>.part/`` and the next job is made with twice the
executor memory (capped at 8G); a ``<year>.part/`` made for another request
(other ``--format-options``, ``--backend-format``, bands or spatial extent:
the job title carries the graph hash) is moved aside to
``<year>.part.superseded-<time>``
and a new job is made, so an old download is never normalised for a new
request, and files of two jobs never mix. Every store records the share of
low B12 digital numbers per date and the processing baselines of its input
products (``radiometry`` / ``input_baselines``), so ``open_cube`` can warn
when a window joins years with and without the +1000 L2A offset.

The Zarr output of the backend is flagged experimental and nobody here has
seen it, so the normaliser accepts a directory store, a zipped store, a
store delivered file by file (one asset per file), several assets (split by
band or by date), Zarr v2 or v3 (sharded, big-endian, without dimension
names or group metadata), per-band variables or a ``bands`` dimension, and
netCDF as a fallback, and one store per date without a time dimension (the
date from the STAC item / asset metadata or the file name); it stops with a
clear message otherwise, and also when the delivered grid is in another CRS
or off the 10 m lattice of the existing cube (``--allow-grid-offset``
overrides). Whatever goes wrong while reading the delivered files, the
FORMAT REPORT gathered so far (with the raw Zarr metadata when zarr cannot
open a store, and the traceback) is written to ``<year>.format.txt`` and
printed. ``--test`` runs ONE tiny job per product first (a window of 2,040 x
2,040 m = 204 x 204 px inside the Villaviciosa frame, corners on multiples
of 60 m, near the old 2 km test box: ``TEST_SIZE_M``; 2025-06-01..15: all
nine bands, then the angles on that window snapped outward to 300 m), prints
what the backend says about its Zarr format and a FORMAT REPORT of what came
back for each, crops the delivery onto the window, writes
``<root>/_test/villaviciosa/2025.zarr`` and ``2025.angles.zarr`` and opens
them through ``zarr_cubes.open_cube`` / ``open_angles`` / ``angles_at``. Run
it again to re-normalise the same downloads (no new job); add ``--new-job``
(once) to pay for new ones: deleting ``<root>/_test`` is not enough, the
finished jobs would be found again on the backend by their titles. A
``<year>.part`` made for another request (another extent, too: the job
title carries the hash of the process graph) is moved aside to
``<year>.part.superseded-<time>``, never deleted by --test, and the job of
the new request is made. ``--test --lonlat-extent`` repeats the old
request under ``<root>/_test_lonlat`` (its finished job is found again by
its unchanged title, and its download is copied from the folder the --test
of the UTM extent set aside, so no new job and no new download).
``--test --odd-window`` (main product only, one more small job, under
``<root>/_test_odd``) requests a 2,050 x 2,050 m window (205 x 205 px, an
odd size) whose west and south edges are lines 10 m past a multiple of 20 m
(north and east on multiples of 20 m): the production frames have such
edges (Villaviciosa east/south, Ferrol east/south), the --test window has
none, so this shows whether the backend snaps or rounds such an extent.

The ANGLES product
------------------
The scene times of the pipeline (``overpass_times.json``) are datatake START
times, not the time the satellite passed over a site (Ems and Wadden share
them on all 462 dates; they spread 7-8 min within one relative orbit). The
validation recomputes the sensing time per site and date by inverting the
SUN ANGLES at the site with a solar ephemeris (validation code, not here).
For that every site and year also gets ``cubes/<site>/<year>.angles.zarr``:
the bands ``sunZenithAngles sunAzimuthAngles viewZenithMean
viewAzimuthMean`` of the same collection (float32 degrees, which the backend
derives from the 5 km angle grids of the tile metadata; the view angles are
for diagnostics), from ONE SEPARATE small job per site and year:
``load_collection(SENTINEL2_L2A, 300 m frame, same window, the four angle
bands, max_cloud_cover=100)`` -> ``resample_spatial(resolution=300,
method="bilinear")`` -> ``save_result(format="Zarr")``, where the 300 m
frame is the main frame snapped OUTWARD to multiples of 300 m in the same
UTM CRS (the existing rule of the angles reference grid; a complete angles
store of the site, when there is one, gives its own frame), sent as a UTM
extent like the main one. The property filter
is the main job's (it keeps every scene), so both jobs select the same
products; there is no cloud filtering. Never in the main job: float bands in
the main cube could change the dtype the backend delivers for all of it.

Why 300 m. The angle fields are smooth (the sun zenith changes by at most
~0.01 deg per km) and sampled every 5 km: 300 m samples the backend's
interpolation ~17 times per 5 km cell, so the bilinear interpolation of the
store at any point (``zarr_cubes.angles_at``) reproduces it far below the
precision of the angle grids, while the ~2 km --test window still gets 8 x 7
cells (enough to measure a spread inside a frame; 500 m would leave 4 x 4)
and the largest site 115 x 94 cells (0.17 MB a date as float32, ~30 MB a
year raw, a few MB on disk; ~1/2000 of the values of the main job; 100 m
would only add cells). Bilinear, because nearest would turn the 5 km grid
into 5 km steps.
Alignment: 300 m is a multiple of 10, 20 and 60 m, so the 300 m lattice of
the site's UTM CRS (cell edges at multiples of 300 m, shifted by the 10 m
frame's own offset from multiples of 10 m when it has one) is a sub-lattice
of the 10 m lattice of the main grid: every cell edge is a line of that
lattice, every cell is exactly 30 x 30 pixels of the main grid, and the
angle frame is the main frame snapped outward to 300 m (it contains it;
Ems: 108 x 114 cells around 3206 x 3413 px), which the angles job requests
as its UTM extent (the backend's Zarr path anchors its grid on the extent's
upper-left corner, so the extent's edges must be those lines). The guard
checks the delivered grid against that frame (another CRS or cell edges
off the 10 m lattice stop; another extent, e.g. the padding to whole
chunks, is cropped / padded with NaN); when the backend's 300 m cells sit
at another phase of the 10 m lattice than predicted, the first store of the
site is kept as delivered and becomes the reference of its other years.

Store: float32 degrees, NaN = no data (a delivered no-data value, declared
or the Zarr fill_value, becomes NaN; CF scale_factor/add_offset would be
decoded; nothing else is changed), dims (t, y, x) with the main store's ``t``
labels, chunks {t: 64, y: 256, x: 256}, consolidated, the provenance
attributes of the main stores, ``format = pyintertidal-zarr-angles/1`` and
``complete = 1`` written last. Every piece of machinery is the main
product's: job ledger and reattachment by title (``zarr_angles_<site>_
<year>_<hash>``), ``<year>.angles.part/``, the defensive normalisation of
whatever layout comes back and its FORMAT REPORT
(``<year>.angles.format.txt``), the site lock, resumability, logging.
Checks (``<year>.angles.verify.json`` and the log): the dates against those
of the main store of the year (differences listed: e.g. a date at a swath
edge with angles but no pixels); the physical ranges (zenith 0-90,
azimuth 0-360; sun angles outside them stop a production year before
anything is written: not degrees; sun angles exactly 0, impossible here,
are flagged as a no-data nobody declared); the per-date spread inside the
frame against a geometric bound (the sun zeniths of two points differ by at
most their angular distance; the azimuths by about that over sin(zenith)
plus the convergence of the meridians: a date above it mixes datatakes or
carries a bad grid); and the sun zenith / azimuth at the frame centre for a
few dates. ``--products main,angles`` (default both) selects the products;
a complete store is skipped, so re-running a year whose main store is
complete fetches only its angles.

Band subsets (``--bands``, 2026-10-26)
--------------------------------------
Opt-in and off by default: without ``--bands`` nothing above changes
(process, job title, stores, attributes, reports, dry run). ``--bands
B03,B08,SCL`` (any subset of the nine that holds B03, B08 and SCL:
SUBSET_REQUIRED; the registration measures with B08 / B03 and chooses its
dates by SCL, the verification compares B03 B08 SCL) makes the MAIN product
with those bands only, for about a third of the credits (the method
comparison needs NDWI = (B03 - B08) / (B03 + B08) and SCL only). Same
collection, extent, window, resampling, property filter and job options:
only the band list of ``load_collection`` differs, so the graph hash and the
job title (``zarr_cube_B03_B08_SCL_<site>_<year>_<hash>``) differ, and a
nine-band job or download is never reattached or reused for a subset, nor
the reverse. Its stores go to ``--store-root`` (default ``SUBSET_ROOT``,
``cubes_light``): ``<store-root>/<site>/<year>.zarr`` in the same format
(``FORMAT_ID``), with sidecars, ``download.log``, ``overpass_times.json`` and
a site lock of their own. ONE BAND SET PER ROOT (``open_cube`` refuses a
window whose years differ in bands): a store root that is the nine-band
``--root``, lies inside it, holds it, or holds a store of another band set
is refused before anything is made (``subset_root_problem``). The stores
under ``--root`` are only READ:

* the frame and the reference grid are the nine bands' (the site's netCDF
  cube, else a complete store under ``--root``); where the reference covers
  the year the registration is the nine bands'. In a year it does not cover,
  ``--offset-px`` (the site's known offset, one site per command) or the
  offset of a registered store of the same request is checked against the
  median composites of the netCDF-registered stores of the site under the
  store root AND under ``--root`` (the nine-band 2023-2025 years), and,
  stricter than for the nine bands, taken only when the composites CONFIRM
  it on a store whose delivery had this delivery's array shape
  (``--accept-unverified-offset`` takes it otherwise, recorded);
* without B8A / B11 / B12 the 20 m checks of the nine bands cannot run; the
  subset's SCL (its only 20 m band) is checked at the four phases of the
  20 m lattice instead (``scl_phase_check``: nearest neighbour from the
  native grid makes phase (0, 0) the best; another phase beating it refuses
  the year);
* a year that also has a complete nine-band store under ``--root`` is
  compared with it after writing (``nine_band_gate``, ``<year>.vs_nine_band.
  json``): the subset's bands BITWISE on every pixel and date (the fill
  included), the same dates and grid, the same registration offset,
  delivered shape and input baselines. Passing, the store is accepted as
  'identical to the nine-band store' (the LIGHT-PATH GATE: one cheap year
  that has a nine-band store proves the subset's layout and values before
  years without reference are bought); failing, the usual rule against the
  reference decides and the log says the gate FAILED. A year WITHOUT a
  nine-band store of its own is refused before any job until a store of the
  same bands (any site) has passed the gate (``subset_gate_problem``;
  ``--without-light-gate`` overrides, recorded in ``light_gate``);
* ``radiometry`` is recorded on B12, else B11, else B08 (a weaker indicator
  over turbid water); the store records ``band_subset`` and
  ``nine_band_root``; the dry run's disk estimate is per band;
* ``--dry-run`` prints per site-year the registration it will meet, the
  expected input megapixels and the credits estimated from the cost per
  megapixel MEASURED on the site's jobs (``<year>.job.json`` under the store
  root, else under ``--root``; else QUOTED_COSTS): a range, since against
  nine-band jobs B03 B08 SCL cost 1/3 (bands counted on the 10 m output
  grid) to 0.43 (the 20 m bands at their native resolution) of the nine; a subset
  job of the site settles it (same bands: one number), and the share of the
  nine bands' cost MEASURED at a site with both (``measured_share``: the
  Westerschelde 2023 job of the gate) gives the other sites' best estimate;
  the raw deliveries a subset keeps are counted in the disk total.

Gates added 2026-10-26 (band subsets only):

* a light-path gate that fails on the DATES alone (every common date bitwise,
  the same offset, delivered shape and input baselines; e.g. the CDSE
  catalogue changed since the nine-band job) still lifts the wait of the
  years without a nine-band store (``light_gate_common_dates``, recorded in
  their ``light_gate``); the store itself is accepted as NINE_BAND_HOW only
  when the gate passes. A gate that fails on the offset or the delivered
  shape says that the nine-band offsets do not hold for the subset;
* at EXTRACTION_GATE_SITES (Westerschelde, Wadden, Ems) a year without a
  nine-band store of its own is bought only once the converter's FREE 2023
  extraction gate of the site (``experiments/light_to_extraction.py gate``,
  ``results/fixedcfg/light/<site>_gate_2023.json``) passed, or failed on
  extraction values only and the user accepted it there (``gate
  --accept-gate-differences``); ``extraction_gate_problem``,
  ``--without-extraction-gate`` (recorded).

``--l2a-nodata-equal`` (band subsets only, off by default; 2026-10-26): the
light-path gate may pass UP TO THE NO-DATA ENCODING. Found at escalda 2023
(B03 B08 SCL against the nine bands, the same 294 input products, offset and
delivered shape): every differing pixel-date was no data in both stores,
written once as the Sentinel-2 L2A no-data (DN 0, SCL 0 = NO_DATA, in a
chunk the backend wrote) and once as the Zarr fill (-32768: a 128 x 128 chunk
the backend did not write; 110 all-zero chunks of the nine-band delivery
were absent from the subset's). With the flag, a gate whose bitwise
comparison differs runs ``nodata_encoding_check``: a differing pixel-date
counts as encoding only when, in BOTH stores, every compared band (SCL among
them) is 0 or the fill; any other difference (a value, a real 0 of a dark
pixel with a valid SCL) still fails. Passing that way (the same dates, grid,
offset, delivered shape and input baselines too), the store is accepted as
NINE_BAND_NODATA_HOW, recorded with the counts in ``<year>.vs_nine_band.json``
(``nodata_encoding``), and lifts the wait of the years without a nine-band
store as NINE_BAND_HOW does (``light_gate_passed``). Without the flag
nothing changes. Nothing the n-tables read changes either: the extraction
reads 0 / 0 as NaN NDWI, SCL 0 as not clear, and the screening sets the fill
to SCL 0.

``--expect-account <e-mail>`` (any run, off by default): before any job is
submitted or looked up, ``connection.describe_account()`` is called, the
account's e-mail printed, and the run stops when it is another account (the
refresh token in use is chosen by ``OPENEO_CONFIG_HOME``).

Order of the pre-2023 purchases (2026-10-26): the free steps first
(``light_to_extraction reference`` and ``gate`` of each site, the dry runs),
then the one cheap light-path year (escalda 2023), then ONE site-year per
command, newest first, the cheapest and best-surveyed site first: escalda
2022, 2021, 2020, then wadden, then ems; after each year the converter's
``build`` prints the screened scenes still missing for n = 125 (``n_target``)
and an older year is bought only while some are missing. A loop over years
stops on an INCOMPLETE store, not on the exit code (1 also means overpass
times missing).

Resuming, the ledger, the reattachment by title, ``--renormalise``,
``--verify-only`` and ``--accept-differences`` work as above (give the same
``--bands`` and ``--store-root``); a subset's raw delivery (``<year>.part``)
is always kept, as with ``--keep-raw``. Not with --test, --odd-window,
--lonlat-extent or the angles (main product only).

Usage (from the repository root; on the server after ``source env.sh``)::

    $PY -X utf8 -m experiments.download_cube_zarr --login          # once, interactive
    $PY -X utf8 -m experiments.download_cube_zarr --test --dry-run # the UTM extents, no network
    $PY -X utf8 -m experiments.download_cube_zarr --test           # the format test, both products
    $PY -X utf8 -m experiments.download_cube_zarr --test --lonlat-extent   # the old request (diagnosis)
    $PY -X utf8 -m experiments.download_cube_zarr --test --odd-window      # odd edges, main only
    $PY -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023-2025 --dry-run
    $PY -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023-2025
    $PY -X utf8 -m experiments.download_cube_zarr ems --years 2023-2025 --products angles
    $PY -X utf8 -m experiments.download_cube_zarr ems --years 2017-2022
    $PY -X utf8 -m experiments.download_cube_zarr ems --years 2023 --verify-only
    $PY -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023 --renormalise --dry-run
    $PY -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023 --renormalise
    $PY -X utf8 -m experiments.download_cube_zarr ems --years 2022 --offset-px 288,271
    $PY -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023 --accept-differences
    $PY -X utf8 -m experiments.download_cube_zarr escalda --years 2023 --bands B03,B08,SCL --dry-run
    $PY -X utf8 -m experiments.download_cube_zarr escalda --years 2023 --bands B03,B08,SCL --keep-raw
    $PY -X utf8 -m experiments.download_cube_zarr ems --years 2022 --bands B03,B08,SCL --offset-px 288,271 --keep-raw
    $PY -X utf8 -m experiments.download_cube_zarr escalda --years 2022 --bands B03,B08,SCL --offset-px 5,1 --keep-raw --expect-account <e-mail>

To force a fresh job for a year, pass ``--new-job`` ONCE: the year's
``<year>.part`` (and ``<year>.angles.part``) is moved aside and no job is
reattached (deleting the folder alone is not enough: a finished job of the
same title would be found again on the backend). Do not keep ``--new-job``
when resuming that run. A store that is complete is skipped even with
``--new-job`` (nothing is paid for twice); move ``<year>.zarr`` or
``<year>.angles.zarr`` aside first to make it again (or ``--renormalise``
a main store whose raw delivery was kept). A store found where a new one is
written is moved aside to ``<year>.zarr.superseded-<UTC>``, never deleted.

Importing this module does nothing (no chdir, no network); ``main`` works
from the repository root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import socket
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import zarr_cubes as zc  # noqa: E402
from experiments.validation_grid import (DUTCH_PERIOD, DUTCH_SITES, dutch_cube,  # noqa: E402
                                         dutch_products)

# ─────────────────────────────────────────────────────────────────────────────
#  what is downloaded
# ─────────────────────────────────────────────────────────────────────────────

BACKEND = "openeo.dataspace.copernicus.eu"
COLLECTION = "SENTINEL2_L2A"
BANDS = zc.BANDS
RESOLUTION_M = 10
#: resample_spatial method of the existing cubes (cube.py, notebook_compat.py)
RESAMPLE_METHOD = "near"
#: as the existing cubes: a property filter that keeps every scene
MAX_CLOUD_COVER = 100
#: the same as experiments.download_cube.JOB_OPTIONS (a test checks it)
JOB_OPTIONS = {"executor-memory": "4G", "executor-memoryOverhead": "4G"}
BACKEND_FORMAT = "Zarr"
DEFAULT_YEARS = (2023, 2024, 2025)

RESAMPLING = (f"resample_spatial(resolution={RESOLUTION_M}, method='{RESAMPLE_METHOD}') on the "
              "whole cube, as the existing netCDF cubes: identity for the native 10 m bands "
              "(B02 B03 B04 B08); nearest neighbour 20 m -> 10 m for B8A B11 B12 SCL CLD")
BAND_NOTES = {
    "SCL": "Sen2Cor scene classification (class codes), 20 m, nearest neighbour to 10 m",
    "CLD": "Sen2Cor cloud probability (%), 20 m, nearest neighbour to 10 m",
}
SPECTRAL_NOTE = ("L2A surface reflectance x 10000 (digital number as delivered by the backend; "
                 "not rescaled)")

#: the sites, the box each existing cube was requested with, that cube, the
#: period it was REQUESTED for (end-exclusive) and the pipeline's overpass
#: file (read only, to keep the times the products used). bbox None = the
#: registry polygon's box, as experiments.download_cube and the notebooks.
#: Villaviciosa's cube came from a wider box than the registry polygon
#: (915 x 915 px; experiments/prototypes/dl_villaviciosa_10y.py): every
#: Villaviciosa product sits on that grid, so that box is used here.
SITE_CONFIG = {
    "villaviciosa": {"bbox": {"west": -5.46, "south": 43.47, "east": -5.35, "north": 43.55},
                     "reference": "ndwi_cube_villaviciosa_grande_10y.nc",
                     "reference_period": ("2016-01-01", "2025-12-31"),
                     "pipeline_overpass": "products_marea/overpass_times.json"},
    "ferrol": {"bbox": None, "reference": "ndwi_cube_ferrol_2023-2025_10m.nc",
               "reference_period": ("2023-01-01", "2025-12-31"),
               "pipeline_overpass": "products_ferrol/overpass_times.json"},
}
SITE_CONFIG.update({s: {"bbox": None, "reference": dutch_cube(s), "reference_period": DUTCH_PERIOD,
                        "pipeline_overpass": f"{dutch_products(s)}/overpass_times.json"}
                    for s in DUTCH_SITES})
SITES = zc.SITES

#: --test: a TEST_SIZE_M x TEST_SIZE_M m window (204 x 204 px, a multiple of
#: 60 m) on the Villaviciosa lattice around the outer ria (channel, flats
#: and shore), corners on multiples of FRAME_SNAP_M m (``test_window``),
#: 2025-06-01..2025-06-15, every band. TEST_HALF_M: the old lon/lat test box
#: (``test_bbox``: provenance, and the request of --lonlat-extent)
TEST_SITE = "villaviciosa"
TEST_CENTRE = (-5.395, 43.52)          # lon, lat
TEST_HALF_M = 1000.0
TEST_SIZE_M = 2040
TEST_WINDOW = ("2025-06-01", "2025-06-16")
TEST_ROOT = "_test"
#: --test --lonlat-extent (the old request, for diagnosis) works under
#: <root>/_test_lonlat, so it never sets aside or overwrites what the --test
#: of the UTM extent made (and its finished job of 2026-10-13 is found again
#: by its title: same graph, same title, no new credits unless --new-job)
TEST_ROOT_LONLAT = "_test_lonlat"
#: --test --odd-window: a window of this size (205 x 205 px, odd) whose
#: west edge is TEST_ODD_SHIFT_M east of the --test window's (a line 10 m
#: past a multiple of 20 m) and whose north edge is the --test window's: west
#: and south edges on odd 10 m lines, as the east / south edges of the
#: production frames; under <root>/_test_odd, main product only
TEST_ODD_SIZE_M = 2050
TEST_ODD_SHIFT_M = 10
TEST_ROOT_ODD = "_test_odd"

#: the frame of a site without its reference netCDF cube on the machine (and
#: no complete store): the lon/lat box projected to UTM and snapped OUTWARD
#: to multiples of this (m). The Sentinel-2 tile corners sit on multiples of
#: 60 m, so such edges are lines of the native 10, 20 and 60 m lattices.
FRAME_SNAP_M = 60
#: x/y delivered as float32 (the backend's Zarr writer) carry up to half a
#: float32 step of rounding (0.25 m at northings below 8.4e6 m, where the
#: step is 0.5 m; 0.016-0.03 m at eastings). ``interpret`` accepts a float32
#: axis as regular within this many 10 m pixels of a regular axis (an axis
#: anchored anywhere, e.g. the lon/lat box of 2026-10-13, rounds by up to half
#: a step); ON THE LATTICE is stricter (``snap_to_lattice``): every value must
#: be the float32 rounding of a lattice value, and an offset of the regular
#: axis is refused by the lattice check (1e-3 px) unless --allow-grid-offset.
F32_TOL_PX = 0.05
F32_TOL_M = F32_TOL_PX * RESOLUTION_M
#: what ``spatial_extent_source`` says of a --lonlat-extent request
LONLAT_SOURCE = ("--lonlat-extent: the lon/lat box (EPSG:4326), the request before 2026-10-13, "
                 "for diagnosis (its grid is anchored on the reprojected box, off the lattice)")
#: the short origin of a --lonlat-extent request (the dry run prints it)
LONLAT_ORIGIN = "--lonlat-extent: the old lon/lat box, NOT on the lattice (diagnosis only)"

#: REGISTRATION (2026-10-19; module docstring, "Registration"). The bands
#: whose masked cross-correlation places the frame in a padded delivery: the
#: first measures, the second must confirm (same offset, same rules)
REGISTRATION_BANDS = ("B08", "B03")
#: SCL classes that make a pixel clear for choosing the dates (vegetation,
#: not vegetated, water), counted on the DELIVERED SCL in the ALWAYS-INSIDE
#: core of the arrays (``core_window``: rows Hr - H .. H, columns Wr - W ..
#: W, inside the frame at every candidate offset; the whole array when that
#: core is empty), one date at a time (``clear_counts``)
REGISTRATION_CLEAR_SCL = (4, 5, 6)
#: a date is usable when at least this share of its valid delivered SCL
#: pixels is clear ...
REGISTRATION_MIN_CLEAR = 0.5
#: ... and its valid SCL pixels number at least this share of the counted
#: window (the reference frame of a date must also hold data on this share)
REGISTRATION_MIN_COVER = 0.5
#: the dates counted (at most), and the fewest accepted
REGISTRATION_DATES = 5
REGISTRATION_MIN_DATES = 3
#: against the reference netCDF the usable dates are tried clearest first,
#: at most this many: a date that does not measure (its reference frame
#: without data, r below REGISTRATION_MIN_R_ANY, a peak not unique or not
#: centred) is SET ASIDE (listed) and the next clearest one taken; a date
#: whose peak passes at ANOTHER offset refuses the year (disagreement)
REGISTRATION_MAX_TRIED = 10
#: Pearson r on the frame at the registered offset: every counted date and
#: band reaches REGISTRATION_MIN_R_ANY, and at least REGISTRATION_MIN_DATES
#: counted dates reach REGISTRATION_MIN_R in B08. The same products give r =
#: 1.0000 at the true offset (Ferrol, Escalda, Wadden, Ems); two downloads of
#: Villaviciosa differ in a mosaic strip of the westernmost ~90 columns (the
#: other tile), which costs r: B08 0.9947 and B03 0.9836 on 2025-10-02 with
#: every argmax at the true offset (margins 0.10-0.20), so requiring 0.995 of
#: every date and band tested the products' identity, not the geometry
REGISTRATION_MIN_R = 0.995
REGISTRATION_MIN_R_ANY = 0.98
#: offsets within this many pixels (Chebyshev) of the best belong to its
#: peak; the best must beat every offset outside it by REGISTRATION_MIN_MARGIN
#: in r. Measured on clear dates of the legacy cubes, at 2 px or more from
#: the true offset: identical data (one date) B08 r <= 0.94 at Ferrol (10 m),
#: <= 0.976 at Wadden (a 20 m pixel = 2 px of 10 m), B03 <= 0.87 / 0.91, so
#: margins >= 0.024; composites of two years (the sibling check) B08 / B03
#: margins 0.050-0.133 at Ferrol, 0.024-0.061 at Wadden
REGISTRATION_PEAK_PX = 1
REGISTRATION_MIN_MARGIN = 0.005
#: the peak must also be CENTRED on the offset found: the parabola through
#: the best r and its two neighbours, on each axis, puts the maximum within
#: this many pixels of it. The margin rule cannot see a flat-topped peak whose
#: argmax is the wrong pixel (Ferrol, 2023-10-06 vs 2024-10-10: argmax 39
#: with r 0.657, the true 40 at 0.652, sub-pixel maximum 39.38); identical
#: data give a symmetric peak (~0)
REGISTRATION_MAX_SUBPIXEL = 0.25
#: an offset counts only when the frame's valid pixels and the delivery's
#: overlap there on at least this share of the frame's valid pixels
REGISTRATION_MIN_OVERLAP = 0.5
#: a year without a reference netCDF takes the offset of a registered store
#: of the site (the sibling) only when the per-pixel MEDIAN COMPOSITES of the
#: REGISTRATION_SIBLING_DATES clearest usable dates of each (this delivery's
#: full arrays; the sibling's frame) put the correlation peak there, for
#: every band, with the peak rules above and r >= REGISTRATION_SIBLING_MIN_R.
#: Single dates of two years cannot do it: on the legacy cubes, with the
#: true offset known, the pairs of clear dates refused all 6 year pairs at
#: Ferrol and at Wadden (B03 r 0.06-0.40 on hazy dates the SCL calls clear,
#: argmax 1 px off on 7 of 60 pairs, product geolocation); the composites
#: found the exact offset on 12 of 12 (Ferrol r 0.93-0.98, margin
#: 0.050-0.133; Wadden r 0.90-0.98, margin 0.024-0.061)
REGISTRATION_SIBLING_DATES = 12
REGISTRATION_SIBLING_MIN_R = 0.85
#: the composites are computed in blocks of this many rows (memory: rows x
#: columns x dates float32, ~100 MB at Ems)
REGISTRATION_COMPOSITE_ROWS = 512
#: a delivery of the frame's shape in a year without reference is checked
#: against a registered store within this many pixels of (0, 0) (the
#: composite padded with an invalid margin)
REGISTRATION_FRAME_SIZED_PX = 8
#: the 20 m layer (B8A B11 B12 SCL CLD, nearest-resampled) takes the offset
#: measured on the 10 m bands: on the registration dates the delivery's own
#: B8A (full arrays) is correlated with its own B08 frame at that offset and
#: must peak there (r >= REGISTRATION_20M_MIN_R where it measures; a peak
#: that passes at another offset refuses the year). B8A, not B11: B11
#: against B08 gives r 0.46-0.72 and once an argmax 1 px off (Villaviciosa)
REGISTRATION_20M_BAND = "B8A"
REGISTRATION_20M_MIN_R = 0.9
#: the 20 m PHASE check of the placed frame: for these 20 m spectral bands,
#: the share of valid pixel pairs inside one 20 m cell of the UTM lattice
#: holding equal values (``block_equal_counts``) at the four phases of that
#: lattice (x and / or y shifted by 10 m), on a few dates: nearest
#: neighbour from the native grid gives ~100 % at phase (0, 0) (Villaviciosa
#: B11 0.987 vs 0.50 / 0.50 / 0.02); a 20 m layer displaced by an odd number
#: of pixels makes ANOTHER phase the ~100 % one, which refuses the year when
#: it beats phase (0, 0) by REGISTRATION_PHASE_MIN_GAIN (on at least
#: REGISTRATION_PHASE_MIN_PAIRS pairs). SCL / CLD are left out: classes and
#: cloud probabilities are equal over large areas at any phase
REGISTRATION_PHASE_BANDS = ("B8A", "B11", "B12")
REGISTRATION_PHASE_MIN_GAIN = 0.2
REGISTRATION_PHASE_MIN_PAIRS = 1000
#: against the reference netCDF the delivered SCL (20 m) is compared with the
#: reference's on every counted date at the registered offset and at every
#: offset within REGISTRATION_SCL_SEARCH_PX of it: another offset matching
#: better by more than REGISTRATION_SCL_TOL (equal share of the pixels valid
#: in both) refuses the year (a displaced 20 m layer); the registered one
#: matching best by more than that confirms the 20 m layer
REGISTRATION_SCL_SEARCH_PX = 2
REGISTRATION_SCL_TOL = 0.001
#: the verification after writing (``compare_with_netcdf``) accepts a store
#: that is 'identical' to the reference, or NEAR-IDENTICAL: on the identical
#: grid, per band, at least this share of the compared pixel-dates valid in
#: either store equal (same value, or no data in both); anything else is
#: refused (the store is not marked complete) unless --accept-differences
VERIFY_MIN_EQUAL = {"B03": 0.999, "B08": 0.999, "SCL": 0.99}
#: a store that DIFFERS from the reference gets a per-date GEOMETRY verdict
#: (``geometry_check``): on every common date with at least
#: VERIFY_GEOMETRY_MIN_VALID valid B08 pixels in both, the reference's
#: central crop (at most VERIFY_GEOMETRY_CROP_PX square) is correlated with
#: the store's within VERIFY_GEOMETRY_SEARCH_PX: 'same' when the peak is at
#: (0, 0) with r >= VERIFY_GEOMETRY_MIN_R, 'displaced' when it passes
#: elsewhere, 'undetermined' otherwise. It tells differing VALUES (another
#: mosaic, a reprocessed product) from a displacement: --accept-differences
#: accepts the former, never a store displaced on a date or the same on
#: fewer than half of the dates checked
VERIFY_GEOMETRY_CROP_PX = 1024
VERIFY_GEOMETRY_SEARCH_PX = 3
VERIFY_GEOMETRY_MIN_R = 0.98
VERIFY_GEOMETRY_MIN_VALID = 0.5

#: dates per year in the existing cubes (Villaviciosa 2016-2025; the other
#: sites have the same within a few), for size estimates of years no cube has
TYPICAL_DATES = {2016: 63, 2017: 126, 2018: 145, 2019: 145, 2020: 144, 2021: 146, 2022: 145,
                 2023: 145, 2024: 147, 2025: 173}
#: estimated bytes on disk per pixel and date for the nine bands, from Blosc/
#: Zstd-5 ratios measured on the Ferrol 10 m cube: 10 m bands ~1.75x, 20 m
#: bands upsampled by nearest ~4.5x, SCL ~100x, CLD ~30x (assumed)
BYTES_PER_PIXEL_DATE = 6.0
#: a failed job is retried with twice the executor memory, up to this (GB)
MAX_RETRY_MEMORY_GB = 8
#: radiometric convention check: band, and the DN below which a valid value
#: means "no +1000 offset" (DN < 500 is reflectance < 0.05 without the
#: offset, which water has in B12; with the offset it would be < -0.05)
RADIOMETRY_BANDS = ("B12", "B11")
LOW_DN = 500
#: a delivered chunk larger than this (bytes) is reported as a memory risk
BIG_CHUNK_BYTES = 2e9
#: dates sampled from a year for the no-data, value and resampling reports
REPORT_CANDIDATES = 12
REPORT_DATES = 4

#: the ANGLES product (module docstring): the sun and view angle bands of
#: SENTINEL2_L2A (float32 degrees, from the 5 km tile metadata grids)
ANGLE_BANDS = zc.ANGLE_BANDS
#: their grid (m): see "Why 300 m" in the module docstring
ANGLES_RES_M = zc.ANGLES_RES_M
#: smooth 5 km fields: bilinear (nearest would make 5 km steps)
ANGLES_METHOD = "bilinear"
#: the native spacing of the angle grids of the Sentinel-2 tile metadata
ANGLES_NATIVE_M = 5000
ANGLES_RESAMPLING = (f"resample_spatial(resolution={ANGLES_RES_M}, method='{ANGLES_METHOD}') of "
                     f"the backend's angle bands (Sentinel-2 tile metadata grids, "
                     f"{ANGLES_NATIVE_M} m): {ANGLES_RES_M} m cells whose edges are lines of the "
                     f"{RESOLUTION_M} m lattice of the main grid")
#: (long name, CF standard name, description) of each angle band
ANGLE_INFO = {
    "sunZenithAngles": ("sun zenith angle", "solar_zenith_angle",
                        "sun zenith angle at the cell (degrees from the vertical)"),
    "sunAzimuthAngles": ("sun azimuth angle", "solar_azimuth_angle",
                         "sun azimuth angle at the cell (degrees clockwise from north)"),
    "viewZenithMean": ("viewing zenith angle (mean)", "sensor_zenith_angle",
                       "viewing zenith angle at the cell, mean of the detectors and bands"),
    "viewAzimuthMean": ("viewing azimuth angle (mean)", "sensor_azimuth_angle",
                        "viewing azimuth angle at the cell, mean of the detectors and bands"),
}
#: CF packing attributes decoded for the angles (never for the main bands)
CF_PACKING = ("scale_factor", "add_offset")
#: bytes on disk per cell and date for the four float32 angle bands (16 B
#: raw): Blosc/Zstd-5 with byte shuffle gave 4.3 B on bilinear interpolations
#: of a 5 km grid with irregular low-order bits (Ems frame, 173 dates); 6 B
#: leaves a margin for the view angles' detector steps
ANGLES_BYTES_PER_CELL_DATE = 6.0
#: the spread bound of the sun angles inside a frame (module docstring) adds
#: these tolerances (degrees): seconds between the tiles of one datatake
#: and the angle grids' own interpolation
SPREAD_TOL_ZENITH = 0.1
SPREAD_TOL_AZIMUTH = 0.25
#: a value this far outside its physical range (degrees) is out of range
RANGE_EPS = 1e-3
#: dates whose angles at the frame centre the reports print
CENTRE_DATES = 5

#: BAND SUBSETS (--bands; module docstring, "Band subsets"). Off by default:
#: nothing below is used by a request of the nine bands. The default root of
#: the subsets' stores (--store-root), relative to the repository root
SUBSET_ROOT = "cubes_light"
#: the bands every subset holds: the registration measures with B08 / B03 and
#: chooses its dates by SCL; the verification compares B03 B08 SCL
SUBSET_REQUIRED = REGISTRATION_BANDS + ("SCL",)
#: the radiometry indicator of a subset: B12, else B11 (as the nine bands),
#: else B08 (NIR: water below DN 500 without the +1000 offset, as B12, but
#: turbid water reaches it less often)
SUBSET_RADIOMETRY_BANDS = RADIOMETRY_BANDS + ("B08",)
#: bytes on disk per pixel and date of each band (int16, Blosc/Zstd-5 ratios
#: of BYTES_PER_PIXEL_DATE: 10 m bands ~1.75x, 20 m bands nearest-resampled
#: ~4.5x, SCL ~100x, CLD ~30x; the nine sum to ~6): a subset's disk estimate
BYTES_PER_BAND_PIXEL_DATE = {"B02": 1.14, "B03": 1.14, "B04": 1.14, "B08": 1.14, "B8A": 0.44,
                             "B11": 0.44, "B12": 0.44, "SCL": 0.02, "CLD": 0.07}
#: the SCL of a subset without B8A / B11 / B12 at the four phases of the 20 m
#: lattice (``scl_phase_check``): nearest neighbour from the native 20 m grid
#: makes every 20 m cell of the placed frame uniform, so phase (0, 0) holds the
#: highest share of equal pixel pairs. Measured on the Ferrol 10 m cube
#: (2023-2025, 25 dates): phase (0, 0) 1.0 on every date, the other phases
#: 0.894-0.9986 (the structure of the scene); the same SCL one pixel off gives
#: phase (0, 0) 0.948-0.9986 and 1.0 at another phase. Refused when another
#: phase beats (0, 0) by more than SUBSET_SCL_PHASE_TOL (on at least
#: REGISTRATION_PHASE_MIN_PAIRS pairs); confirmed when (0, 0) beats every other
#: phase by at least SUBSET_SCL_PHASE_MIN_GAP; else not confirmed (a warning)
SUBSET_SCL_PHASE_TOL = 0.001
SUBSET_SCL_PHASE_MIN_GAP = 0.002
#: the credits of NINE-BAND 10 m site-year jobs on the CDSE account (approximate,
#: as measured in October 2026), with the dates and the delivered (padded)
#: arrays of those years (results/server_logs_2026-10-19/zarr_verification): the
#: dry run's calibration of the cost per megapixel of a site when no job
#: sidecar of it (``<year>.job.json`` with the job's ``costs``) is on this
#: machine. Not the same at every site: ~0.20 credits per band-megapixel-date
#: of the delivered arrays at Westerschelde, ~0.16 at Wadden and Ems
QUOTED_COSTS = (
    {"site": "escalda", "year": 2023, "credits": 890.0, "dates": 146,
     "delivered_shape": (1536, 2176)},
    {"site": "wadden", "year": None, "credits": 2000.0, "dates": 145,
     "delivered_shape": (3456, 2816)},
    {"site": "wadden", "year": 2025, "credits": 2400.0, "dates": 172,
     "delivered_shape": (3456, 2816)},
    {"site": "ems", "year": 2023, "credits": 3240.0, "dates": 145,
     "delivered_shape": (3840, 3968)},
)
#: a calibration job's measured input (``usage.input_pixel``) picks the band
#: convention of the estimate when it is within this share of one of them
COST_CONVENTION_TOL = 0.1
#: the sites whose band-subset years without a nine-band store of their own
#: feed the method comparison through experiments/light_to_extraction.py: such
#: a year is bought only once that converter's free 2023 EXTRACTION gate of the
#: site (``<EXTRACTION_GATE_DIR>/<site>_gate_2023.json``) passed, or failed on
#: extraction values only and the user accepted it there (``gate
#: --accept-gate-differences``, recorded in the report);
#: ``extraction_gate_problem``, --without-extraction-gate
EXTRACTION_GATE_SITES = ("escalda", "wadden", "ems")
EXTRACTION_GATE_DIR = "results/fixedcfg/light"


class FormatError(RuntimeError):
    """The backend delivered something the normaliser does not understand."""


class AmbiguousNoData(FormatError):
    """No declared no-data value, and the data cannot tell (``--nodata``)."""


class JobFailed(RuntimeError):
    """The openEO job ended in error (logs saved in the part folder)."""


class LockedError(RuntimeError):
    """Another download of the same site is running."""


class FrameRefused(RuntimeError):
    """A production request whose frame is the snapped lon/lat box, not a
    reference grid's (``--allow-snapped-frame``)."""


class RegistrationRefused(FormatError):
    """The frame's place in a padded delivery could not be established by
    the rules (``register_delivery``): nothing is written."""


class VerificationRefused(RuntimeError):
    """A written store differs from the reference beyond the declared rule
    (``verification_decision``): it is not marked complete
    (``--accept-differences``)."""


class RawMissing(RuntimeError):
    """--renormalise: the store's raw delivery (``<year>.part``) is gone."""


class LightGateMissing(RuntimeError):
    """A band-subset year without a nine-band store of its own, while no store
    of those bands has passed the light-path gate yet (``subset_gate_problem``;
    --without-light-gate): no job is made."""


class ExtractionGateMissing(LightGateMissing):
    """A band-subset year of an EXTRACTION_GATE_SITES site without a nine-band
    store of its own, while the converter's 2023 extraction gate of the site
    has not passed nor been accepted by the user (``extraction_gate_problem``;
    --without-extraction-gate): no job is made."""


class AccountMismatch(RuntimeError):
    """--expect-account: the openEO connection is authenticated as another
    account (or names none): nothing is started."""


# ─────────────────────────────────────────────────────────────────────────────
#  requests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Product:
    """One kind of store made per site and year (``--products``)."""
    name: str
    bands: tuple
    resolution: int              # m: resample_spatial resolution
    method: str                  # resample_spatial method
    format_id: str               # the store's ``format`` attribute
    chunks: dict                 # the store's chunks
    suffix: str                  # after the year in the store, part and sidecar names
    title_prefix: str            # openEO job titles
    raw_itemsize: int            # bytes per delivered value (dry-run estimate)
    disk_per_px_date: float      # bytes on disk per pixel and date (dry-run estimate)
    store_dtype: str             # what the dry run says the values are
    description: str


#: the nine-band 10 m cube (the values as delivered: int16, fill -32768)
MAIN = Product("main", BANDS, RESOLUTION_M, RESAMPLE_METHOD, zc.FORMAT_ID, zc.CHUNKS, "",
               "zarr_cube", 2, BYTES_PER_PIXEL_DATE, "int16",
               f"the nine bands on the {RESOLUTION_M} m grid, int16 as delivered")
#: the sun and view angles of the same scenes (float32 degrees, NaN)
ANGLES = Product("angles", ANGLE_BANDS, ANGLES_RES_M, ANGLES_METHOD, zc.ANGLES_FORMAT_ID,
                 zc.ANGLES_CHUNKS, ".angles", "zarr_angles", 4, ANGLES_BYTES_PER_CELL_DATE,
                 "float32", f"sun and view angles on {ANGLES_RES_M} m cells, float32 degrees")
PRODUCTS = {p.name: p for p in (MAIN, ANGLES)}


def parse_products(text):
    """``"main,angles"`` -> the product names, in the order of PRODUCTS."""
    names = {p.strip().lower() for p in str(text or "").replace(";", ",").split(",") if p.strip()}
    unknown = sorted(names - set(PRODUCTS))
    if unknown:
        raise ValueError(f"unknown product(s) {unknown}; choose from {list(PRODUCTS)}")
    if not names:
        raise ValueError(f"no product given; choose from {list(PRODUCTS)}")
    return [p for p in PRODUCTS if p in names]


def parse_bands(text):
    """``"B03,B08,SCL"`` (any case, any order) -> the bands of a subset of the
    main product in store order (BANDS). ValueError for an unknown band, an
    empty list, or one without SUBSET_REQUIRED (B03, B08, SCL)."""
    names = [b.strip().upper() for b in str(text or "").replace(";", ",").split(",") if b.strip()]
    unknown = sorted(set(names) - set(BANDS))
    if unknown:
        raise ValueError(f"unknown band(s) {unknown}; choose from {list(BANDS)}")
    if not names:
        raise ValueError(f"no band given; choose from {list(BANDS)}")
    need = [b for b in SUBSET_REQUIRED if b not in names]
    if need:
        raise ValueError(f"a band subset holds {list(SUBSET_REQUIRED)} (the registration measures "
                         f"with B08 / B03 and chooses its dates by SCL; the verification compares "
                         f"B03 B08 SCL): {need} missing")
    return tuple(b for b in BANDS if b in names)


@dataclass
class Request:
    """One store to produce: a site, a year (the store name), a window and
    the product (``main``: the nine-band cube; ``angles``: its angles).

    ``extent``: the ``spatial_extent`` the job requests, ``{"west", "south",
    "east", "north", "crs": EPSG code}`` in the site's UTM CRS, edges on the
    reference lattice (``year_requests`` / ``test_request`` fill it in);
    None requests the lon/lat ``bbox`` instead (the request before
    2026-10-13: ``--lonlat-extent``). ``bbox`` is always the lon/lat box (the
    stores keep it in their provenance); ``extent_source`` says where the
    extent comes from, ``extent_origin`` the same in a few words (reference
    netCDF, complete store, or SNAPPING of the lon/lat box).

    ``bands`` other than the product's: a BAND SUBSET of the main product
    (``is_subset``; --bands, module docstring): ``root`` is then the subset's
    store root and ``main_root`` the root of the nine-band stores, only read
    (the reference grid, the registered stores that check an offset, the
    same year's nine-band store of the light-path gate, the cost calibration)."""
    site: str
    year: int
    start: str
    end: str                     # exclusive
    bbox: dict
    root: Path
    bands: tuple = None          # None: the product's bands
    test: bool = False
    product: str = MAIN.name
    extent: dict = None          # UTM extent requested; None: the lon/lat bbox
    extent_source: str = ""
    extent_origin: str = ""
    main_root: Path = None       # a band subset: the nine-band root (read only)

    def __post_init__(self):
        if self.product not in PRODUCTS:
            raise ValueError(f"unknown product {self.product!r}; choose from {list(PRODUCTS)}")
        if self.bands is None:
            self.bands = PRODUCTS[self.product].bands

    @property
    def spatial_extent(self):
        """What ``load_collection`` gets: the UTM extent, else the lon/lat box."""
        return dict(self.extent) if self.extent is not None else dict(self.bbox)

    @property
    def spec(self):
        return PRODUCTS[self.product]

    @property
    def is_angles(self):
        return self.product == ANGLES.name

    @property
    def is_subset(self):
        """A main request of fewer bands than the nine (--bands)."""
        return self.product == MAIN.name and tuple(self.bands) != tuple(MAIN.bands)

    @property
    def tag(self):
        what = f" {self.product}" if self.product != MAIN.name else ""
        sub = f" {'+'.join(self.bands)}" if self.is_subset else ""
        return f"{self.site} {self.year}{what}{sub}{' TEST' if self.test else ''}"

    @property
    def store(self):
        if self.is_angles:
            return zc.angles_path(self.site, self.year, self.root)
        return zc.store_path(self.site, self.year, self.root)

    @property
    def part(self):
        return zc.site_dir(self.site, self.root) / f"{self.year}{self.spec.suffix}.part"

    def sidecar(self, kind):
        sfx = self.spec.suffix.lstrip(".")
        return zc.sidecar_path(self.site, self.year, f"{sfx}.{kind}" if sfx else kind, self.root)

    def complete(self):
        """True when this request's store is finished (``zarr_cubes.is_complete``)."""
        return zc.is_complete(self.store, self.spec.format_id)


def site_bbox(site):
    cfg = SITE_CONFIG[site]
    if cfg["bbox"] is not None:
        return dict(cfg["bbox"])
    import pyintertidal as pit          # the registry; heavy import, only when needed

    return dict(pit.sites.get(site).bbox)


def test_bbox():
    """The --test box: TEST_HALF_M around TEST_CENTRE, in degrees."""
    lon, lat = TEST_CENTRE
    dlat = TEST_HALF_M / 111_320.0
    dlon = TEST_HALF_M / (111_320.0 * math.cos(math.radians(lat)))
    return {"west": round(lon - dlon, 6), "south": round(lat - dlat, 6),
            "east": round(lon + dlon, 6), "north": round(lat + dlat, 6)}


def parse_years(tokens):
    """``["2017-2022", "2025"]`` or ``["2023,2025"]`` -> sorted unique years."""
    years = set()
    for tok in tokens or []:
        for part in str(tok).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"(\d{4})\s*[-:]\s*(\d{4})", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if b < a:
                    raise ValueError(f"empty year range {part!r}")
                years.update(range(a, b + 1))
            elif re.fullmatch(r"\d{4}", part):
                years.add(int(part))
            else:
                raise ValueError(f"not a year or a range: {part!r}")
    bad = [y for y in years if not 2015 <= y <= dt.date.today().year]
    if bad:
        raise ValueError(f"years outside Sentinel-2 L2A: {sorted(bad)}")
    return sorted(years)


def year_requests(sites, years, root, products=(MAIN.name,), lonlat=False, extents=True,
                  bands=None, main_root=None):
    """The requests of a run: per site, per year, the products in order
    (the angles of a year right after its main store, so their dates can be
    compared at once). Each carries its UTM extent: the site's 10 m frame
    (``site_frame``) for the main store, its 300 m frame (``angles_frame``)
    for the angles; ``lonlat``: the lon/lat box instead (--lonlat-extent);
    ``extents=False``: none (--verify-only needs none). ``bands`` (a subset
    of the nine, --bands): the main requests are a BAND SUBSET whose stores go
    under ``root`` (the store root) while ``main_root`` (the nine-band root,
    only read) gives their frame: the nine bands' (``Request.main_root``)."""
    frames, out = {}, []
    sub = bands is not None and tuple(bands) != tuple(MAIN.bands)
    for s in sites:
        for y in years:
            for p in products:
                subset = sub and p == MAIN.name
                fr_root = Path(main_root) if subset and main_root is not None else root
                ext, src, org = None, "", ""
                if lonlat:
                    src, org = LONLAT_SOURCE, LONLAT_ORIGIN
                elif extents:
                    if (s, p) not in frames:
                        fr = angles_frame(s, root) if p == ANGLES.name else site_frame(s, fr_root)
                        frames[(s, p)] = (extent_of(fr), fr["source"], fr["origin"])
                    ext, src, org = frames[(s, p)]
                kw = (dict(bands=tuple(bands), main_root=None if main_root is None
                           else Path(main_root)) if subset else {})
                out.append(Request(site=s, year=y, start=zc.year_window(y)[0],
                                   end=zc.year_window(y)[1], bbox=site_bbox(s), root=root,
                                   product=p, extent=ext, extent_source=src, extent_origin=org,
                                   **kw))
    return out


def test_request(root, product=MAIN.name, lonlat=False, odd=False):
    """The --test request of ``product`` (``root``: the cubes root; the
    test writes under ``<root>/_test``): the TEST_SIZE_M window on the
    Villaviciosa lattice (``test_window``), for the angles that window
    snapped outward to the site's 300 m cells; ``lonlat``: the old 2 km
    lon/lat box (``test_bbox``), under ``<root>/_test_lonlat``; ``odd``
    (main only): the window with odd edges (``test_odd_window``), under
    ``<root>/_test_odd``."""
    ext, src, org, sub = None, LONLAT_SOURCE, LONLAT_ORIGIN, TEST_ROOT_LONLAT
    if odd and (lonlat or product != MAIN.name):
        raise ValueError("the odd --test window is for the main product of the UTM extent only")
    if not lonlat:
        fr = (test_angles_window(root) if product == ANGLES.name else
              test_odd_window(site_frame(TEST_SITE, root)) if odd else
              test_window(site_frame(TEST_SITE, root)))
        ext, src, org = extent_of(fr), fr["source"], fr["origin"]
        sub = TEST_ROOT_ODD if odd else TEST_ROOT
    return Request(site=TEST_SITE, year=int(TEST_WINDOW[0][:4]), start=TEST_WINDOW[0],
                   end=TEST_WINDOW[1], bbox=test_bbox(), root=Path(root) / sub, test=True,
                   product=product, extent=ext, extent_source=src, extent_origin=org)


def test_requests(root, products=tuple(PRODUCTS), lonlat=False, odd=False):
    """The --test requests, one per product (``odd``: the main product only)."""
    if odd:
        return [test_request(root, MAIN.name, odd=True)]
    return [test_request(root, p, lonlat=lonlat) for p in products]


def snapped(req):
    """Is the request's extent the lon/lat box snapped to 60 m (no reference
    grid on this machine), rather than a reference grid's frame?"""
    return req.extent is not None and str(req.extent_origin).startswith("SNAPPING")


# ─────────────────────────────────────────────────────────────────────────────
#  logging and locking
# ─────────────────────────────────────────────────────────────────────────────

def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def make_log(req):
    """``log(msg)``: a stamped line on stdout (flushed) and in the site's log."""
    path = zc.site_dir(req.site, req.root) / "download.log"

    def log(msg):
        line = f"{utcnow():%Y-%m-%d %H:%M:%S}Z [{req.tag}] {msg}"
        print(line, flush=True)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    return log


def _pid_alive(pid):
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    if os.name == "nt":                 # os.kill(pid, 0) TERMINATES a process on Windows
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SiteLock:
    """One download per site at a time: ``<site>/.download.lock`` (O_EXCL).

    A lock left by a process that is gone (same host) is taken over; a live
    one stops this run before any job is created, so two sessions never pay
    for the same year twice.
    """

    def __init__(self, directory):
        self.path = Path(directory) / ".download.lock"
        self.mine = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        info = {"host": socket.gethostname(), "pid": os.getpid(),
                "started_utc": utcnow().isoformat(timespec="seconds")}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    other = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    other = {}
                stale = (not other or (other.get("host") == info["host"]
                                       and not _pid_alive(other.get("pid", -1))))
                if stale:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise LockedError(
                    f"{self.path} is held by pid {other.get('pid')} on {other.get('host')} since "
                    f"{other.get('started_utc')}: another download of this site is running "
                    f"(delete the file if it is not)")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(info, f)
            self.mine = True
            return self
        raise LockedError(f"could not take {self.path}")

    def __exit__(self, *exc):
        if self.mine:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        return False


def ensure_gitignore(root):
    """``cubes/.gitignore`` with ``*``: the stores never show up in git."""
    root = Path(root)
    try:
        root.resolve().relative_to(ROOT)
    except ValueError:
        return
    root.mkdir(parents=True, exist_ok=True)
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("# Zarr cubes written by experiments/download_cube_zarr.py: data, not code\n*\n",
                      encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  the openEO process
# ─────────────────────────────────────────────────────────────────────────────

def build_process(req, connection=None, backend_format=BACKEND_FORMAT, format_options=None):
    """``load_collection -> resample_spatial -> save_result``: (10, near)
    for the main product, (300, bilinear) and the angle bands for the angles.
    The spatial extent is the request's UTM extent (``Request.extent``,
    with its EPSG code as ``crs``), or the lon/lat box for --lonlat-extent.

    Without a connection the graph is built offline (dry run, job title);
    with one, the client also checks the bands and the output format
    against the backend.
    """
    from openeo.rest.datacube import DataCube

    spec = req.spec
    kw = dict(spatial_extent=req.spatial_extent, temporal_extent=[req.start, req.end],
              bands=list(req.bands), max_cloud_cover=MAX_CLOUD_COVER)
    if connection is not None:
        cube = connection.load_collection(COLLECTION, **kw)
    else:
        cube = DataCube.load_collection(COLLECTION, connection=None, **kw)
    cube = cube.resample_spatial(resolution=spec.resolution, method=spec.method)
    return cube.save_result(format=backend_format, options=format_options or None)


def graph_hash(flat_graph):
    blob = json.dumps(flat_graph, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def job_title(req, ghash):
    """``zarr_cube_<site>_<year>_<hash>`` (``zarr_angles_...`` for the
    angles; ``..._test_<site>_<start>_<end>_...`` for --test): a job left
    on the backend is found again by it, never for another product. A band
    subset (--bands) carries its bands, ``zarr_cube_B03_B08_SCL_<site>_
    <year>_<hash>`` (its graph, hence its hash, differs from the nine's)."""
    prefix = req.spec.title_prefix
    if req.test:
        return f"{prefix}_test_{req.site}_{req.start}_{req.end}_{ghash[:8]}"
    if req.is_subset:
        return f"{prefix}_{'_'.join(req.bands)}_{req.site}_{req.year}_{ghash[:8]}"
    return f"{prefix}_{req.site}_{req.year}_{ghash[:8]}"


def predict_grid(bbox, res=RESOLUTION_M):
    """The grid openEO makes of a lon/lat box: its UTM envelope (zone of the
    box centre) snapped outward to the ``res`` lattice. Matches the existing
    Villaviciosa and Ferrol 10 m cubes and the Dutch 20 m cubes exactly."""
    from pyproj import Transformer

    lon_c = (bbox["west"] + bbox["east"]) / 2.0
    lat_c = (bbox["south"] + bbox["north"]) / 2.0
    zone = int((lon_c + 180.0) // 6.0) + 1
    epsg = (32600 if lat_c >= 0 else 32700) + zone
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    n = 100
    lon = np.r_[np.linspace(bbox["west"], bbox["east"], n), np.full(n, bbox["east"]),
                np.linspace(bbox["east"], bbox["west"], n), np.full(n, bbox["west"])]
    lat = np.r_[np.full(n, bbox["south"]), np.linspace(bbox["south"], bbox["north"], n),
                np.full(n, bbox["north"]), np.linspace(bbox["north"], bbox["south"], n)]
    X, Y = tf.transform(lon, lat)
    x0 = math.floor(X.min() / res) * res
    x1 = math.ceil(X.max() / res) * res
    y0 = math.floor(Y.min() / res) * res
    y1 = math.ceil(Y.max() / res) * res
    return {"epsg": epsg, "shape": [int(round((y1 - y0) / res)), int(round((x1 - x0) / res))],
            "x_edges": [x0, x1], "y_edges": [y0, y1]}


def lattice_phase(edge, step=RESOLUTION_M):
    """Offset (m, in [0, step)) of a frame edge from the multiples of
    ``step``: 0 for the sites' frames (edges on multiples of 10 m)."""
    p = float(edge) % float(step)
    return 0.0 if min(p, float(step) - p) < 1e-6 else round(p, 6)


def coarse_axes(x, y, res=ANGLES_RES_M):
    """Cell centres (x ascending, y descending) of the ``res`` m grid whose
    cells cover a finer grid of pixel centres ``x``, ``y``: the finer
    frame's edges snapped outward to the lines ``phase + k * res``, where
    ``phase`` is the finer frame's offset from multiples of RESOLUTION_M
    (``lattice_phase``; 0 for the sites' frames, so multiples of ``res``, as
    the backend snaps a bbox: ``predict_grid``). With ``res`` a multiple of
    the finer spacing, every coarse cell edge is then a line of the finer
    lattice, whatever its phase. Applied to the 10 m grid of a site (edges
    on multiples of 10 m) it is ``predict_grid(bbox, res)``."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    dx = abs(float(x[1] - x[0])) if len(x) > 1 else float(RESOLUTION_M)
    dy = abs(float(y[1] - y[0])) if len(y) > 1 else float(RESOLUTION_M)
    px, py = lattice_phase(x[0] - dx / 2.0), lattice_phase(y[0] + dy / 2.0)
    x0 = px + math.floor((x[0] - dx / 2.0 - px) / res + 1e-9) * res
    x1 = px + math.ceil((x[-1] + dx / 2.0 - px) / res - 1e-9) * res
    y0 = py + math.floor((y[-1] - dy / 2.0 - py) / res + 1e-9) * res
    y1 = py + math.ceil((y[0] + dy / 2.0 - py) / res - 1e-9) * res
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    return (x0 + res / 2.0 + res * np.arange(nx, dtype="float64"),
            y1 - res / 2.0 - res * np.arange(ny, dtype="float64"))


# ─────────────────────────────────────────────────────────────────────────────
#  the request extent: the site's frame in its UTM CRS, edges on the lattice
# ─────────────────────────────────────────────────────────────────────────────

def _num(v):
    """An edge for the process graph: an int when whole (301020, not 301020.0)."""
    v = float(v)
    return int(v) if v.is_integer() else v


def _wkt_of_epsg(epsg):
    from pyproj import CRS

    return CRS.from_epsg(int(epsg)).to_wkt("WKT1_GDAL")


def frame_from_axes(x, y, crs_wkt, res, source="", kind="", origin=""):
    """The frame of pixel-centre axes ``x`` (ascending) and ``y``
    (descending), ``res`` m apart: ``{"west", "south", "east", "north",
    "res", "epsg", "crs_wkt", "source", "kind", "origin"}`` (edges = outer
    centres +- res / 2; ``source``: where the frame comes from, ``origin``:
    the same in a few words). Irregular axes raise ValueError."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if not len(x) or not len(y):
        raise ValueError("an empty grid has no frame")
    for name, v, sign in (("x", x, 1.0), ("y", y, -1.0)):
        if len(v) > 1 and not np.allclose(np.diff(v), sign * res, rtol=0, atol=1e-6 * res):
            raise ValueError(f"the {name} axis is not a regular {res:g} m axis "
                             f"({'ascending' if sign > 0 else 'descending'})")
    h = res / 2.0
    epsg = zc._epsg_from_wkt(crs_wkt)
    return {"west": float(x[0] - h), "south": float(y[-1] - h), "east": float(x[-1] + h),
            "north": float(y[0] + h), "res": float(res), "epsg": int(epsg) if epsg else None,
            "crs_wkt": crs_wkt, "source": source, "kind": kind, "origin": origin}


def frame_shape(frame, res=None):
    """``[rows, cols]`` of a frame (or of an extent dict) at ``res`` m."""
    res = float(res or frame["res"])
    return [int(round((float(frame["north"]) - float(frame["south"])) / res)),
            int(round((float(frame["east"]) - float(frame["west"])) / res))]


def frame_axes(frame, res=None):
    """Pixel-centre axes of a frame (or of an extent dict): x ascending, y
    descending, float64, ``res`` m apart from the edges (exact for edges and
    spacings in whole metres)."""
    res = float(res or frame["res"])
    ny, nx = frame_shape(frame, res)
    return (float(frame["west"]) + res / 2.0 + res * np.arange(nx, dtype="float64"),
            float(frame["north"]) - res / 2.0 - res * np.arange(ny, dtype="float64"))


def extent_of(frame):
    """The ``spatial_extent`` of ``load_collection`` for a frame: its edges
    and ``crs`` = its EPSG code as an integer (the bounding-box form of the
    openEO process specification: an EPSG code or a WKT2 string, default
    4326)."""
    if not frame.get("epsg"):
        raise ValueError(f"the frame ({frame.get('source')}) has no EPSG code to request it with")
    return {"west": _num(frame["west"]), "south": _num(frame["south"]),
            "east": _num(frame["east"]), "north": _num(frame["north"]), "crs": int(frame["epsg"])}


def describe_extent(ext, res):
    """``EPSG:32630 west 305400 south 4820340 east 307440 north 4822380
    (204 x 204 px of 10 m)``."""
    ny, nx = frame_shape(ext, res)
    unit = "px" if res <= RESOLUTION_M else "cells"
    return (f"EPSG:{ext.get('crs', ext.get('epsg'))} west {_num(ext['west'])} south "
            f"{_num(ext['south'])} east {_num(ext['east'])} north {_num(ext['north'])} "
            f"({ny} x {nx} {unit} of {res:g} m)")


def snap_frame(frame, step, origin_x=0.0, origin_y=0.0):
    """``frame`` with its edges snapped OUTWARD to the lines ``origin + k *
    step`` (multiples of ``step`` by default): it contains ``frame``."""
    w = origin_x + math.floor((frame["west"] - origin_x) / step + 1e-9) * step
    e = origin_x + math.ceil((frame["east"] - origin_x) / step - 1e-9) * step
    s = origin_y + math.floor((frame["south"] - origin_y) / step + 1e-9) * step
    n = origin_y + math.ceil((frame["north"] - origin_y) / step - 1e-9) * step
    return dict(frame, west=float(w), east=float(e), south=float(s), north=float(n))


def snapped_bbox_frame(bbox, step=FRAME_SNAP_M, res=RESOLUTION_M):
    """The frame of a site without a reference grid: the lon/lat box's UTM
    envelope (zone of the box centre, ``predict_grid``) snapped OUTWARD to
    multiples of ``step`` m, on the ``res`` m lattice."""
    g = predict_grid(bbox, res=step)
    (x0, x1), (y0, y1) = g["x_edges"], g["y_edges"]
    return {"west": float(x0), "south": float(y0), "east": float(x1), "north": float(y1),
            "res": float(res), "epsg": int(g["epsg"]), "crs_wkt": _wkt_of_epsg(g["epsg"]),
            "source": (f"the lon/lat box projected to EPSG:{g['epsg']} and snapped outward to "
                       f"multiples of {step} m"), "kind": "predicted",
            "origin": f"SNAPPING: the lon/lat box outward to multiples of {step} m"}


def site_frame(site, root=None, log=None):
    """The 10 m frame of ``site``, which its main jobs request: the frame of
    its reference grid (``reference_grid``: the site's 10 m netCDF cube, else
    a complete store under ``root``, else the lon/lat box snapped outward to
    multiples of FRAME_SNAP_M m), ``source`` / ``origin`` saying which."""
    ref = reference_grid(site, root=root, log=log)
    if ref is None:
        raise ValueError(f"{site}: no reference grid (not even one predicted from the bbox), so "
                         f"no UTM extent to request (--lonlat-extent requests the lon/lat box)")
    fr = frame_from_axes(ref["x"], ref["y"], ref["crs_wkt"], RESOLUTION_M, kind=ref["kind"])
    shape = f"{len(ref['y'])} x {len(ref['x'])} px"
    why = ref.get("why", f"no {RESOLUTION_M} m reference netCDF")
    if ref["kind"] == "netcdf":
        fr["source"] = f"the frame of the reference netCDF {Path(ref['path']).name} ({shape})"
        fr["origin"] = f"reference netCDF {Path(ref['path']).name}"
    elif ref["kind"] == "store":
        fr["source"] = f"the frame of the complete store {ref['path']} ({shape}; {why})"
        fr["origin"] = f"complete store {Path(ref['path']).name} ({why})"
    else:
        fr["source"] = (f"the lon/lat box of {site} projected to EPSG:{fr['epsg']} and snapped "
                        f"outward to multiples of {FRAME_SNAP_M} m ({shape}; {why}, no complete "
                        f"store)")
        fr["origin"] = (f"SNAPPING: the lon/lat box outward to multiples of {FRAME_SNAP_M} m "
                        f"({why}, no complete store)")
    off = [k for k in ("west", "south", "east", "north")
           if abs(fr[k] / RESOLUTION_M - round(fr[k] / RESOLUTION_M)) > 1e-6]
    if off:                          # requested as it is: the reference grid is the target
        fr["source"] += (f" - NOTE: its {'/'.join(off)} edge(s) are not on multiples of "
                         f"{RESOLUTION_M} m, i.e. not the native Sentinel-2 lattice")
    return fr


def angles_frame(site, root=None, log=None, res=ANGLES_RES_M):
    """The 300 m frame of ``site``, which its angles jobs request: the frame
    of ``angles_reference_grid`` (a complete angles store of the site, else
    the 10 m frame snapped outward to multiples of ``res`` m, which contains
    it: the rule of ``coarse_axes``; a 10 m frame whose edges are not on
    multiples of 10 m keeps its offset, so the cell edges stay lines of its
    lattice), ``source`` / ``origin`` saying which."""
    ref = angles_reference_grid(site, root=root, log=log, res=res)
    if ref is None:
        raise ValueError(f"{site}: no {res} m reference grid, so no UTM extent for the angles")
    fr = frame_from_axes(ref["x"], ref["y"], ref["crs_wkt"], res, kind=ref["kind"])
    shape = f"{len(ref['y'])} x {len(ref['x'])} cells of {res} m"
    if ref["kind"] == "store":
        fr["source"] = f"the frame of the complete angles store {ref['path']} ({shape})"
        fr["origin"] = f"complete angles store {Path(ref['path']).name}"
    else:
        fine = site_frame(site, root)
        ph = (lattice_phase(fine["west"]), lattice_phase(fine["north"]))
        lines = (f"multiples of {res} m" if ph == (0.0, 0.0) else
                 f"multiples of {res} m plus ({ph[0]:g}, {ph[1]:g}) m, the {RESOLUTION_M} m "
                 f"frame's own offset")
        fr["source"] = (f"the {RESOLUTION_M} m frame snapped outward to {lines} "
                        f"({shape}; the {RESOLUTION_M} m frame: {fine['source']})")
        fr["origin"] = f"{fine['origin']}, snapped outward to {lines}"
    return fr


def test_window(frame, centre=None, size=None, align=FRAME_SNAP_M):
    """The --test window inside ``frame`` (a site's 10 m frame): ``size`` x
    ``size`` m (TEST_SIZE_M) around ``centre`` (lon, lat; TEST_CENTRE), its
    edges lines of the frame's lattice: on multiples of ``align`` m when the
    lattice has such lines (the sites' 10 m lattices: edges on multiples of
    10 m), else on the lattice; moved inside the frame when it would stick
    out. A frame too small for it raises ValueError."""
    from pyproj import CRS, Transformer

    centre = tuple(TEST_CENTRE if centre is None else centre)
    size = float(TEST_SIZE_M if size is None else size)
    res = float(frame["res"])
    if abs(size / res - round(size / res)) > 1e-9:
        raise ValueError(f"a {size:g} m window is not a whole number of {res:g} m pixels")
    crs = CRS.from_user_input(frame.get("crs_wkt") or f"EPSG:{frame['epsg']}")
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*centre)

    def snap(v, step, phase, how):
        q = (v - phase) / step
        k = (round(q) if how == "round" else math.floor(q + 1e-9) if how == "floor"
             else math.ceil(q - 1e-9))
        return phase + k * step

    def place(lo, hi, want_lo, step, phase, down):
        """The window's lower edge along one axis (``down``: y, the edge
        given is the upper one), inside [lo, hi], on the lines phase + k step."""
        if down:                                   # north edge, the window below it
            e = snap(want_lo, step, phase, "round")
            if e > hi + 1e-6:
                e = snap(hi, step, phase, "floor")
            if e - size < lo - 1e-6:
                e = snap(lo + size, step, phase, "ceil")
            return e if (e <= hi + 1e-6 and e - size >= lo - 1e-6) else None
        e = snap(want_lo, step, phase, "round")    # west edge, the window east of it
        if e < lo - 1e-6:
            e = snap(lo, step, phase, "ceil")
        if e + size > hi + 1e-6:
            e = snap(hi - size, step, phase, "floor")
        return e if (e >= lo - 1e-6 and e + size <= hi + 1e-6) else None

    def options(edge):          # the lines the window's edges may sit on, preferred first
        phase = float(np.mod(edge, res))
        lat = [(res, phase)]
        if min(phase, res - phase) < 1e-6 and align % res == 0:
            return [(float(align), 0.0)] + [(res, 0.0)]
        return lat

    west = north = None
    on = []
    for step, phase in options(frame["west"]):
        west = place(frame["west"], frame["east"], cx - size / 2.0, step, phase, False)
        if west is not None:
            on.append(step)
            break
    for step, phase in options(frame["north"]):
        north = place(frame["south"], frame["north"], cy + size / 2.0, step, phase, True)
        if north is not None:
            on.append(step)
            break
    if west is None or north is None:
        raise ValueError(f"a {size:g} x {size:g} m test window does not fit in the frame "
                         f"{frame_shape(frame)} px ({frame.get('source')})")
    n = int(round(size / res))
    where = (f"corners on multiples of {align} m"
             if all(s == align for s in on) and abs(size % align) < 1e-9 else
             f"corners on the {res:g} m lattice of the frame")
    return dict(frame, west=float(west), east=float(west + size), north=float(north),
                south=float(north - size), kind="test",
                source=(f"the --test window, {size:g} x {size:g} m ({n} x {n} px) around lon/lat "
                        f"{centre[0]:g}, {centre[1]:g}, {where}, inside {frame.get('source')}"),
                origin=f"--test window ({where}) inside the frame of: {frame.get('origin')}")


def test_odd_window(frame, shift=TEST_ODD_SHIFT_M, size=TEST_ODD_SIZE_M):
    """The --test --odd-window window: the --test window's west edge moved
    ``shift`` m east (a line 10 m past a multiple of 20 m), its north edge
    kept, ``size`` m (205 px) on each side: the west and south edges are odd
    10 m lines, the north and east edges multiples of 20 m (the production
    frames have odd east / south edges; the --test window has none)."""
    res = float(frame["res"])
    win = test_window(frame)
    w, n = float(win["west"]) + shift, float(win["north"])
    e, so = w + size, n - size
    if (abs(w / res - round(w / res)) > 1e-9 or abs(size / res - round(size / res)) > 1e-9
            or e > float(frame["east"]) + 1e-6 or so < float(frame["south"]) - 1e-6):
        raise ValueError(f"the odd --test window ({w:g}..{e:g} x {so:g}..{n:g}) is not on the "
                         f"{res:g} m lattice or not inside the frame ({frame.get('source')})")
    n_px = int(round(size / res))
    odd = [k for k, v in (("west", w), ("south", so), ("east", e), ("north", n))
           if abs((v / 20.0) - round(v / 20.0)) > 1e-9]
    return dict(frame, west=w, east=e, north=n, south=so, kind="test",
                source=(f"the --test --odd-window window, {size:g} x {size:g} m ({n_px} x {n_px} px, "
                        f"odd), its {'/'.join(odd) or 'no'} edge(s) 10 m past a multiple of 20 m, "
                        f"inside {frame.get('source')}"),
                origin=(f"--test --odd-window ({'/'.join(odd) or 'no'} edge(s) on odd 10 m lines) "
                        f"inside the frame of: {frame.get('origin')}"))


def test_angles_window(root=None, res=ANGLES_RES_M):
    """The --test frame of the angles job: the --test window snapped outward
    to the site's ``res`` m cells (the lattice of ``angles_frame``: multiples
    of 300 m, unless an angles store of the site fixes another phase)."""
    cells = angles_frame(TEST_SITE, root, res=res)
    win = test_window(site_frame(TEST_SITE, root))
    fr = snap_frame(win, res, origin_x=cells["west"], origin_y=cells["north"])
    ny, nx = frame_shape(fr, res)
    fr.update(res=float(res), kind="test",
              source=(f"the --test window ({_num(win['west'])}..{_num(win['east'])} x "
                      f"{_num(win['south'])}..{_num(win['north'])}) snapped outward to the {res} m "
                      f"cells of the site ({ny} x {nx} cells; {cells['source']})"),
              origin=f"--test window snapped outward to the {res} m cells of: {cells['origin']}")
    return fr


def request_grid(req):
    """The grid of a request's UTM extent at its product's resolution, as
    a reference-grid dict (``kind`` "request"); None for a lon/lat request."""
    if req.extent is None:
        return None
    res = req.spec.resolution
    x, y = frame_axes(req.extent, res)
    return {"path": f"the requested extent ({describe_extent(req.extent, res)})", "kind": "request",
            "x": x, "y": y, "crs_wkt": _wkt_of_epsg(req.extent["crs"]), "res": res}


def reference_summary(site):
    """Grid and dates per year of the site's existing netCDF cube (metadata
    only), or None when it is not on this machine."""
    path = ROOT / SITE_CONFIG[site]["reference"]
    if not path.is_file():
        return None
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(path, mask_and_scale=False) as ds:
        tdim = "t" if "t" in ds.dims else "time"
        t = pd.DatetimeIndex(ds[tdim].values)
        g = zc.grid_info(ds)
    return {"path": str(path), "shape": g["shape"], "epsg": g["epsg"],
            "per_year": {int(k): int(v) for k, v in pd.Series(t.year).value_counts().items()}}


# ─────────────────────────────────────────────────────────────────────────────
#  jobs: ledger, orphans, start, wait, error logs
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE = ("created", "queued", "running")
LEDGER = "job.json"


class Ledger:
    """``<year>.part/job.json``: every job created or adopted for this store.

    The id is written BEFORE the job is started, so a killed session leaves
    a job that the next run finds and reattaches to.
    """

    def __init__(self, part, title=None):
        self.path = Path(part) / LEDGER
        self.data = {"title": title, "jobs": []}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        if title:
            self.data["title"] = title

    def save(self):
        zc.write_json(self.data, self.path)

    def current(self):
        """The last job not known to be useless (None when there is none)."""
        for j in reversed(self.data["jobs"]):
            if j.get("status") not in ("error", "canceled", "gone", "expired"):
                return j
        return None

    def known(self, job_id):
        return any(j["job_id"] == job_id for j in self.data["jobs"])

    def failed_created(self):
        """Jobs made by this downloader for this store that ended in error."""
        return sum(1 for j in self.data["jobs"]
                   if j.get("status") == "error" and j.get("source") == "created")

    def add(self, job_id, status, source, **extra):
        """A new entry; ``extra``: the submitted process graph, job options."""
        self.data["jobs"].append(dict({"job_id": job_id, "status": status, "source": source,
                                       "title": self.data.get("title"),
                                       "added_utc": utcnow().isoformat(timespec="seconds")},
                                      **extra))
        self.save()
        return self.data["jobs"][-1]

    def update(self, entry, **kw):
        entry.update(kw)
        entry["updated_utc"] = utcnow().isoformat(timespec="seconds")
        self.save()


def job_summary(info):
    """The parts of a job description worth keeping (no process graph)."""
    keep = ("id", "title", "status", "created", "updated", "progress", "costs", "usage",
            "plan", "budget")
    return {k: info.get(k) for k in keep if k in info}


def find_orphan(conn, title, ledger, log):
    """A job of the same title still useful on the backend (newest first).
    Looks through the last 1000 jobs (the client's default is 100)."""
    try:
        jobs = list(conn.list_jobs(limit=1000))
    except Exception as exc:              # listing is a convenience, never fatal
        log(f"could not list the backend's jobs ({exc}); no orphan search")
        return None
    cands = [j for j in jobs if j.get("title") == title
             and j.get("status") in ACTIVE + ("finished",) and not ledger.known(j.get("id"))]
    cands.sort(key=lambda j: str(j.get("created", "")), reverse=True)
    return cands[0] if cands else None


def _api_status(exc):
    return getattr(exc, "http_status_code", None)


def start_job(job, ctx, log):
    """Start, retrying while the backend refuses for capacity reasons."""
    from openeo.rest import OpenEoApiPlainError

    t0 = time.time()
    while True:
        try:
            job.start()
            return
        except OpenEoApiPlainError as exc:
            msg = f"{getattr(exc, 'code', '')} {exc}"
            status = _api_status(exc)
            if status == 402 or re.search(r"credit|payment|insufficient funds", msg, re.I):
                raise JobFailed(f"the backend refused to start job {job.job_id}: {msg} "
                                f"(credits exhausted?)") from exc
            transient = status in (429, 502, 503, 504) or re.search(
                r"concurren|too many|rate.?limit|capacity|try again", msg, re.I)
            if not transient or time.time() - t0 > ctx.max_wait_s:
                raise
            log(f"backend refused to start job {job.job_id} ({msg.strip()[:200]}); retrying in "
                f"{ctx.capacity_wait_s // 60} min")
            ctx.sleep(ctx.capacity_wait_s)


def wait_job(job, ctx, log):
    """Poll until the job leaves created/queued/running; return its info."""
    import requests
    from openeo.rest import OpenEoApiPlainError

    t0 = time.time()
    interval = ctx.poll_s
    last, last_print, soft = None, 0.0, 0
    while True:
        try:
            info = job.describe()
            soft = 0
        except (requests.ConnectionError, requests.Timeout) as exc:
            soft += 1
            if soft > 30:
                raise
            log(f"connection problem while polling job {job.job_id} ({exc}); retrying")
            ctx.sleep(30)
            continue
        except OpenEoApiPlainError as exc:
            if _api_status(exc) in (502, 503, 504) and soft < 30:
                soft += 1
                log(f"backend unavailable while polling job {job.job_id} ({exc}); retrying")
                ctx.sleep(30)
                continue
            raise
        status = info.get("status", "N/A")
        now = time.time()
        if status != last or now - last_print > 600:
            prog = info.get("progress")
            log(f"job {job.job_id}: {status}"
                + (f" ({prog:.0f} %)" if isinstance(prog, (int, float)) else "")
                + f" after {dt.timedelta(seconds=int(now - t0))}")
            last, last_print = status, now
        if status not in ACTIVE + ("submitted",):
            return info
        if now - t0 > ctx.max_wait_s:
            raise TimeoutError(
                f"job {job.job_id} still {status} after {ctx.max_wait_s / 3600:.0f} h; it keeps "
                f"running on the backend: run the same command again to reattach")
        ctx.sleep(interval)
        interval = min(ctx.poll_max_s, interval * 1.25)


def save_error_logs(job, part, log):
    """Write the backend's error log of a failed job and show its start."""
    try:
        entries = [dict(e) for e in job.logs(level="error")]
    except Exception as exc:
        log(f"could not fetch the logs of job {job.job_id}: {exc}")
        return None
    path = Path(part) / f"job_{job.job_id}.errors.json"
    zc.write_json(entries, path)
    log(f"job {job.job_id}: {len(entries)} error log entries -> {path}")
    for e in entries[:15]:
        log(f"  backend error: {str(e.get('message', e))[:800]}")
    return path


def _memory_gb(value):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([gGmM])[bB]?\s*", str(value))
    if not m:
        return None
    v = float(m.group(1))
    return v if m.group(2).lower() == "g" else v / 1024.0


def escalate_job_options(options, times=1, cap_gb=MAX_RETRY_MEMORY_GB):
    """Job options with executor memory and overhead doubled ``times`` times
    (capped at ``cap_gb``): a year that lost its Spark executors is not
    resubmitted with the memory it failed with."""
    out = dict(options)
    for _ in range(int(times)):
        for key in ("executor-memory", "executor-memoryOverhead"):
            gb = _memory_gb(out.get(key)) if key in out else None
            if gb is not None and gb < cap_gb:
                out[key] = f"{int(min(cap_gb, math.ceil(gb * 2)))}G"
    return out


def submitted_graph(info):
    """The process graph a job description carries (None when absent)."""
    proc = (info or {}).get("process") or {}
    return proc.get("process_graph") if isinstance(proc, dict) else None


def ensure_finished_job(ctx, req, title, log):
    """A FINISHED job for ``req``: the ledger's, an orphan of the same title,
    or a new one. Returns ``(job, info)``; raises JobFailed after
    ``ctx.retries`` new jobs failed in this run. A new job after failed ones
    gets more executor memory (``escalate_job_options``); its submitted
    process graph and options are kept in the ledger. ``ctx.new_job`` skips
    the orphan search (a paid fresh job)."""
    conn = ctx.connection()
    led = Ledger(req.part, title)
    created_here = 0
    while True:
        cur = led.current()
        if cur is None and not ctx.new_job:
            orphan = find_orphan(conn, title, led, log)
            if orphan is not None:
                log(f"reattaching to job {orphan['id']} ({orphan.get('status')}, created "
                    f"{orphan.get('created')}) found on the backend with this title")
                cur = led.add(orphan["id"], orphan.get("status"), "orphan")
        if cur is None:
            if created_here > ctx.retries:
                raise JobFailed(f"{created_here} job(s) for {req.tag} failed in this run; see the "
                                f"error logs in {req.part}")
            sr = build_process(req, connection=conn, backend_format=ctx.backend_format,
                               format_options=ctx.format_options)
            flat = sr.flat_graph()
            zc.write_json(flat, Path(req.part) / "process_graph.json")
            nfail = led.failed_created()
            opts = escalate_job_options(ctx.job_options, nfail)
            job = sr.create_job(title=title, job_options=opts)
            created_here += 1
            cur = led.add(job.job_id, "created", "created", job_options=opts, process_graph=flat)
            log(f"created job {job.job_id} '{title}' (job options {opts}"
                + (f", memory raised after {nfail} failed job(s)" if opts != ctx.job_options else "")
                + ")")
        job = conn.job(cur["job_id"])
        try:
            info = job.describe()
        except Exception as exc:
            if _api_status(exc) in (404, 410):
                log(f"job {cur['job_id']} is gone from the backend; a new one will be made")
                led.update(cur, status="gone")
                continue
            raise
        if not cur.get("process_graph") and submitted_graph(info):
            led.update(cur, process_graph=submitted_graph(info))
        status = info.get("status")
        if status == "created":
            log(f"starting job {job.job_id}")
            start_job(job, ctx, log)
            status = "queued"
        if status in ACTIVE + ("submitted",):
            info = wait_job(job, ctx, log)
            status = info.get("status")
        led.update(cur, status=status, info=job_summary(info))
        if status == "finished":
            costs, usage = info.get("costs"), info.get("usage")
            log(f"job {job.job_id} finished" + (f"; costs {costs}" if costs is not None else "")
                + (f"; usage {json.dumps(usage)[:400]}" if usage else ""))
            return job, info
        log(f"job {job.job_id} ended '{status}'")
        save_error_logs(job, req.part, log)


# ─────────────────────────────────────────────────────────────────────────────
#  results: assets and their download
# ─────────────────────────────────────────────────────────────────────────────

ASSETS_DONE = "assets.json"


def collect_assets(job, meta):
    """``[(name, href, metadata)]`` of a job's results: the collection's
    ``assets`` or, when the backend lists results as STAC items, theirs."""
    assets = dict(meta.get("assets") or {})
    if not assets:
        for link in meta.get("links") or []:
            if link.get("rel") != "item" or not link.get("href"):
                continue
            item = job.connection.get(link["href"], expected_status=200).json()
            props = item.get("properties") or {}
            when = props.get("datetime") or props.get("start_datetime")
            for k, v in (item.get("assets") or {}).items():
                name = k if k not in assets else f"{item.get('id', 'item')}/{k}"
                if isinstance(v, dict) and when and not asset_datetime(v):
                    v = dict(v, item_datetime=when)
                assets[name] = v
    return [(k, v["href"], v) for k, v in assets.items() if isinstance(v, dict) and v.get("href")]


def asset_datetime(md):
    """The acquisition date/time an asset (or its STAC item) declares."""
    for k in ("datetime", "start_datetime", "item_datetime"):
        if md.get(k):
            return str(md[k])
    return None


def results_gone(job):
    """True when the backend no longer has the job's results (404/410)."""
    try:
        job.get_results().get_metadata(force=True)
        return False
    except Exception as exc:
        return _api_status(exc) in (404, 410)


def safe_relpath(name):
    parts = [p for p in re.split(r"[\\/]+", str(name)) if p not in ("", ".", "..")]
    return Path(*parts) if parts else Path("asset")


def asset_relpath(name, href):
    """Where an asset lands under ``<part>/assets``.

    A store delivered file by file (one asset per Zarr file) may name its
    assets by the bare file name ('0.0.0', '.zarray'), which would collide:
    the path inside the store is then taken from the href, from the
    component ending in '.zarr' on. Otherwise the asset name is used.
    """
    from urllib.parse import unquote, urlparse

    comps = [c for c in unquote(urlparse(str(href or "")).path).split("/") if c]
    for i, c in enumerate(comps[:-1]):
        if c.lower().endswith(".zarr"):
            return safe_relpath("/".join(comps[i:]))
    return safe_relpath(name)


DOWNLOAD_MARKER = "download_job.json"


def clear_download(part, log, why):
    """Remove the files of a previous download from ``part``."""
    part = Path(part)
    gone = [n for n in ("assets", "extracted") if (part / n).exists()]
    for n in gone:
        shutil.rmtree(part / n)
    for n in (ASSETS_DONE, DOWNLOAD_MARKER):
        if (part / n).exists():
            (part / n).unlink()
            gone.append(n)
    if gone:
        log(f"{why}: removed {gone} from {part}")


def download_results(job, req, log, workers=4):
    """Download every asset into ``<part>/assets`` (resumable per file) and
    write ``assets.json`` when all are complete.

    Files are only reused from a download of the SAME job
    (``download_job.json``): when the job changes (results expired, a new
    job after a failure) whatever an older job left is removed first, so a
    store is never made of files from two jobs."""
    from concurrent.futures import ThreadPoolExecutor
    from openeo.rest.job import ResultAsset

    part = Path(req.part)
    part.mkdir(parents=True, exist_ok=True)
    marker = part / DOWNLOAD_MARKER
    prev = None
    if marker.is_file():
        try:
            prev = json.loads(marker.read_text(encoding="utf-8")).get("job_id")
        except ValueError:
            prev = None
    if prev != job.job_id and ((part / "assets").exists() or (part / ASSETS_DONE).exists()):
        clear_download(part, log, f"files of {'job ' + str(prev) if prev else 'an unknown job'} "
                                  f"in the part folder, now downloading job {job.job_id}")
    zc.write_json({"job_id": job.job_id, "started_utc": utcnow().isoformat(timespec="seconds")},
                  marker)
    results = job.get_results()
    meta = results.get_metadata()
    zc.write_json(meta, part / "job-results.json")
    items = collect_assets(job, meta)
    if not items:
        raise FormatError(f"job {job.job_id} finished with no result assets "
                          f"(results metadata saved in {part / 'job-results.json'})")
    raw = part / "assets"
    raw.mkdir(parents=True, exist_ok=True)
    total = sum(int(m.get("file:size") or 0) for _, _, m in items)
    log(f"{len(items)} asset(s) to download" + (f", {total / 1e9:.2f} GB" if total else ""))

    rels = {}
    for name, href, _ in items:
        rels.setdefault(asset_relpath(name, href), []).append(name)
    clash = {str(k): v for k, v in rels.items() if len(v) > 1}
    if clash:
        raise FormatError(f"job {job.job_id}: several assets would land on the same file "
                          f"({dict(list(clash.items())[:5])}); results metadata in "
                          f"{part / 'job-results.json'}")

    def fetch(item):
        name, href, md = item
        rel = asset_relpath(name, href)
        target = raw / rel
        size = md.get("file:size")
        if target.is_file() and (size is None or target.stat().st_size == int(size)):
            return str(rel), target.stat().st_size, md, "kept"   # same job (see the marker)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".download")
        err = None
        for attempt in range(1, 5):
            try:
                ResultAsset(job=job, name=name, href=href, metadata=md).download(tmp)
                got = tmp.stat().st_size
                if size is not None and got != int(size):
                    raise IOError(f"{name}: {got} bytes downloaded, {size} announced")
                os.replace(tmp, target)
                return str(rel), got, md, "downloaded"
            except Exception as exc:          # signed URLs expire: refresh them
                err = exc
                log(f"download of {name} failed ({exc}); attempt {attempt}/4")
                time.sleep(min(120, 15 * attempt))
                try:
                    fresh = {k: (h, m) for k, h, m in collect_assets(
                        job, results.get_metadata(force=True))}
                    href, md = fresh.get(name, (href, md))
                except Exception:
                    pass
        raise IOError(f"could not download {name}: {err}")

    t0 = time.time()
    done = []
    if len(items) > 8:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(fetch, items), 1):
                done.append(r)
                if k % 200 == 0:
                    log(f"  {k}/{len(items)} assets")
    else:
        done = [fetch(it) for it in items]
    got = sum(d[1] for d in done)
    log(f"downloaded {len(done)} asset(s), {got / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")
    files = {}
    for rel, size_, md, _ in done:
        files[rel] = {"size": size_, "type": md.get("type"), "job_id": job.job_id}
        if asset_datetime(md):
            files[rel]["datetime"] = asset_datetime(md)
    zc.write_json({"job_id": job.job_id, "complete": True, "files": files}, part / ASSETS_DONE)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
#  what came back: zips, stores, files
# ─────────────────────────────────────────────────────────────────────────────

ZARR_MARKERS = {".zgroup", ".zarray", ".zattrs", ".zmetadata", "zarr.json"}


def extract_zips(part, log=print, dry_run=False):
    """Unzip every zip asset into ``<part>/extracted/<name without .zip>``
    (members checked to stay inside it). Returns the extraction folders.
    ``dry_run``: nothing is (re)extracted or removed; a zip not extracted
    yet stops with a FormatError."""
    part = Path(part)
    done = json.loads((part / ASSETS_DONE).read_text(encoding="utf-8"))
    out = []
    for rel, md in sorted(done["files"].items()):
        src = part / "assets" / rel
        is_zip = rel.lower().endswith(".zip") or "zip" in str(md.get("type") or "").lower()
        if not is_zip:
            continue
        if not zipfile.is_zipfile(src):
            raise FormatError(f"{src} is announced as a zip but is not one")
        dest = part / "extracted" / (rel[:-4] if rel.lower().endswith(".zip") else rel + ".d")
        flag = dest.with_name(dest.name + ".extracted")
        out.append(dest)
        if flag.is_file() and dest.is_dir():
            continue
        if dry_run:
            raise FormatError(f"dry run: {rel} is not extracted in {dest} (no {flag.name} flag); "
                              f"nothing is extracted or removed by a dry run: run without --dry-run")
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        base = dest.resolve()
        n = 0
        with zipfile.ZipFile(src) as zf:
            for m in zf.infolist():
                rel_m = safe_relpath(m.filename)
                target = (dest / rel_m).resolve()
                if base != target and base not in target.parents:
                    raise FormatError(f"{src}: member {m.filename!r} would land outside {dest}")
                if m.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as fi, open(target, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
                n += 1
        flag.write_text(str(n), encoding="utf-8")
        log(f"extracted {n} files from {rel} -> {dest}")
    return out


def _is_array_dir(p):
    p = Path(p)
    if (p / ".zarray").is_file():
        return True
    zj = p / "zarr.json"
    if zj.is_file():
        try:
            return json.loads(zj.read_text(encoding="utf-8")).get("node_type") == "array"
        except ValueError:
            return False
    return False


def discover(dirs):
    """Zarr stores, netCDF and GeoTIFF files under ``dirs``.

    A Zarr root is the top-most folder holding Zarr metadata. Arrays whose
    folder has no group metadata above them (a writer that skips the root
    ``.zgroup``) are gathered into an implicit group of their parent.
    """
    roots, ncs, tifs, other = [], [], [], []
    for base in dirs:
        base = Path(base)
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            names = set(filenames)
            if names & ZARR_MARKERS:
                roots.append(Path(dirpath))
                dirnames[:] = []
                continue
            dirnames.sort()
            for fn in sorted(filenames):
                low = fn.lower()
                if low.endswith((".nc", ".nc4", ".netcdf")):
                    ncs.append(Path(dirpath) / fn)
                elif low.endswith((".tif", ".tiff")):
                    tifs.append(Path(dirpath) / fn)
                elif not low.endswith((".zip", ".extracted", ".json", ".download")):
                    other.append(Path(dirpath) / fn)
    stores, implicit = [], {}
    for r in roots:
        if _is_array_dir(r):
            implicit.setdefault(r.parent, []).append(r)
        else:
            stores.append({"kind": "zarr", "path": r, "arrays": None})
    for parent, arrs in sorted(implicit.items()):
        stores.append({"kind": "zarr-implicit", "path": parent, "arrays": sorted(arrs)})
    return {"stores": stores, "netcdf": ncs, "geotiff": tifs, "other": other}


# ─────────────────────────────────────────────────────────────────────────────
#  reading what came back (zarr-python, not xarray's zarr backend: xarray
#  2025.1 cannot open several Zarr v3 / string layouts written by others)
# ─────────────────────────────────────────────────────────────────────────────

T_NAMES = {"t", "time", "times", "date", "dates", "datetime"}
Y_NAMES = {"y", "lat", "latitude", "northing", "row", "rows"}
X_NAMES = {"x", "lon", "long", "longitude", "easting", "col", "cols", "column", "columns"}
B_NAMES = {"bands", "band", "band_name", "band_names", "spectral", "variable", "variables",
           "channel", "channels", "wavelength"}
CRS_NAMES = {"crs", "spatial_ref", "grid_mapping", "transverse_mercator", "projection"}


def axis_of(name):
    n = str(name).split("/")[-1].lower()
    for axis, names in (("t", T_NAMES), ("y", Y_NAMES), ("x", X_NAMES), ("bands", B_NAMES)):
        if n in names:
            return axis
    return None


def _py(v):
    """numpy scalar -> python (for reports and attrs)."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    try:
        v = np.asarray(v).item()
    except (ValueError, TypeError):
        return str(v)
    if isinstance(v, float) and np.isnan(v):
        return float("nan")
    return v


def _attrs_clean(attrs):
    out = {}
    for k, v in dict(attrs).items():
        if k == "_ARRAY_DIMENSIONS":
            continue
        if isinstance(v, (np.generic, np.ndarray)):
            v = np.asarray(v).tolist()
        out[str(k)] = v
    return out


def _codecs(arr):
    bits = []
    for name in ("filters", "serializer", "compressors"):
        v = getattr(arr, name, None)
        if v:
            bits.append(f"{name}={v}")
    return " ".join(bits) or "?"


@dataclass
class RawData:
    """What one delivered store/file holds, read lazily, values untouched."""
    source: str
    kind: str                    # zarr | zarr-implicit | netcdf
    ds: object                   # xarray.Dataset (dask-backed)
    info: dict                   # variable -> metadata as delivered
    attrs: dict                  # root / global attributes
    zarr_format: object = None
    notes: list = field(default_factory=list)


def _array_dims(arr):
    md = arr.metadata
    names = getattr(md, "dimension_names", None)
    if names and all(n is not None for n in names):
        return tuple(str(n) for n in names), "dimension_names (v3)"
    d = arr.attrs.get("_ARRAY_DIMENSIONS")
    if d:
        return tuple(str(n) for n in d), "_ARRAY_DIMENSIONS"
    return None, None


CONVENTIONAL = {2: ("y", "x"), 3: ("t", "y", "x"), 4: ("t", "bands", "y", "x")}


def _infer_dims(name, shape, axis_len, notes):
    """Dims of an array without dimension names, from the coordinate
    lengths. When several coordinates share a length (a square tile, as many
    dates as rows) the conventional order (t, [bands], y, x) of image
    writers decides, and the notes say so: the netCDF comparison catches a
    transposition at once."""
    conv = CONVENTIONAL.get(len(shape))
    dims, used, assumed = [], set(), False
    for k, n in enumerate(shape):
        cands = {ax: cn for ax, (cn, ln) in axis_len.items() if ln == n and ax not in used}
        if len(cands) == 1:
            ax = next(iter(cands))
        elif len(cands) > 1 and conv and conv[k] in cands:
            ax, assumed = conv[k], True
        else:
            raise FormatError(
                f"array {name!r} {tuple(shape)} has no dimension names and its axis {k} (length "
                f"{n}) matches {len(cands) or 'no'} coordinate(s) {list(cands.values())}; "
                f"coordinates seen: {axis_len}")
        dims.append(cands[ax])
        used.add(ax)
    if assumed:
        notes.append(f"{name}: coordinates share a length; read as the conventional order "
                     f"{tuple(dims)}")
    return tuple(dims)


def read_zarr(store):
    """Open one discovered Zarr store with zarr-python into a RawData."""
    import dask.array as da
    import xarray as xr
    import zarr

    path = Path(store["path"])
    notes = []
    if store["arrays"] is None:
        try:
            node = zarr.open(store=str(path), mode="r")
        except Exception as exc:
            arrays = [p for p in sorted(path.iterdir()) if p.is_dir() and _is_array_dir(p)]
            if not arrays:
                raise FormatError(f"{path}: zarr cannot open it ({type(exc).__name__}: {exc}) "
                                  f"and no array folder sits in it") from exc
            notes.append(f"zarr could not open the root ({type(exc).__name__}: {exc})")
            store = {"kind": "zarr-implicit", "path": path, "arrays": arrays}
    if store["arrays"] is None:
        if isinstance(node, zarr.Array):
            members = {path.name: node}
            gattrs = {}
        else:
            members = {n: m for n, m in node.members(max_depth=None) if isinstance(m, zarr.Array)}
            gattrs = _attrs_clean(node.attrs)
        zfmt = node.metadata.zarr_format
    else:
        members = {p.name: zarr.open_array(store=str(p), mode="r") for p in store["arrays"]}
        gattrs = {}
        zfmt = sorted({m.metadata.zarr_format for m in members.values()})
        notes.append("no group metadata at the store root: arrays read one by one")
    if not members:
        raise FormatError(f"{path}: a Zarr store with no arrays")
    # short names (last path component) unless they collide
    short = {}
    for n in members:
        s = n.split("/")[-1]
        short[n] = s if sum(1 for m in members if m.split("/")[-1] == s) == 1 else n.replace("/", "_")
    info, dims = {}, {}
    for n, arr in members.items():
        d, src = _array_dims(arr)
        if d is None and arr.ndim == 1 and axis_of(short[n]):
            d, src = (short[n],), "own name (1-D)"
        dims[n] = d
        info[short[n]] = {"path": n, "shape": list(arr.shape), "dtype": str(arr.dtype),
                          "chunks": list(arr.chunks),
                          "shards": list(arr.shards) if getattr(arr, "shards", None) else None,
                          "fill_value": _py(arr.fill_value), "dims": list(d) if d else None,
                          "dims_source": src, "codecs": _codecs(arr),
                          "zarr_format": arr.metadata.zarr_format,
                          "attrs": _attrs_clean(arr.attrs)}
    axis_len = {}
    for n, arr in members.items():
        if arr.ndim == 1 and dims[n]:
            ax = axis_of(dims[n][0]) or axis_of(short[n])
            if ax and ax not in axis_len:
                axis_len[ax] = (dims[n][0], arr.shape[0])
    for n, arr in members.items():
        if dims[n] is None and arr.ndim >= 2:
            dims[n] = _infer_dims(short[n], arr.shape, axis_len, notes)
            info[short[n]]["dims"] = list(dims[n])
            info[short[n]]["dims_source"] = "inferred from coordinate lengths"
        elif dims[n] is None and arr.ndim == 0:
            dims[n] = ()
        elif dims[n] is None:
            dims[n] = (short[n],)
    variables = {}
    for n, arr in members.items():
        attrs = _attrs_clean(arr.attrs)
        if arr.ndim <= 1:
            data = np.asarray(arr[...])
            if data.dtype.kind in ("T", "O"):          # numpy StringDType / objects
                data = np.array([str(v) for v in np.ravel(data).tolist()],
                                dtype=str).reshape(data.shape)
        else:
            data = da.from_zarr(arr)
        variables[short[n]] = xr.Variable(dims[n], data, attrs)
    coords = {k: v for k, v in variables.items() if v.ndim == 1 and v.dims == (k,)}
    data_vars = {k: v for k, v in variables.items() if k not in coords}
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs)
    return RawData(source=str(path), kind=store["kind"], ds=ds, info=info, attrs=gattrs,
                   zarr_format=zfmt, notes=notes)


def read_netcdf(path):
    """A delivered netCDF file (``--backend-format netCDF``, or a backend
    that ignored the Zarr request) as RawData, values untouched."""
    import xarray as xr

    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=True, chunks={})
    info = {}
    for n, v in ds.variables.items():
        info[n] = {"shape": list(v.shape), "dtype": str(v.dtype), "dims": list(v.dims),
                   "dims_source": "netCDF", "chunks": v.encoding.get("chunksizes"),
                   "fill_value": _py(v.attrs.get("_FillValue")),
                   "codecs": f"zlib={v.encoding.get('zlib')} complevel={v.encoding.get('complevel')}",
                   "attrs": _attrs_clean(v.attrs)}
    return RawData(source=str(path), kind="netcdf", ds=ds, info=info, attrs=dict(ds.attrs))


#: calendars whose dates are the plain proleptic Gregorian ones (decoded by
#: ``decode_time_values``; any other goes to xarray's CF decoder)
STANDARD_CALENDARS = ("", "standard", "gregorian", "proleptic_gregorian")
#: nanoseconds per unit of a ``<unit> since <date>`` time axis
TIME_UNIT_NS = {}
for _names, _ns in ((("nanoseconds", "nanosecond", "nsec", "nsecs", "ns"), 1),
                    (("microseconds", "microsecond", "usec", "usecs", "us"), 10 ** 3),
                    (("milliseconds", "millisecond", "msec", "msecs", "ms"), 10 ** 6),
                    (("seconds", "second", "secs", "sec", "s"), 10 ** 9),
                    (("minutes", "minute", "mins", "min"), 60 * 10 ** 9),
                    (("hours", "hour", "hrs", "hr", "h"), 3600 * 10 ** 9),
                    (("days", "day", "d"), 86400 * 10 ** 9)):
    TIME_UNIT_NS.update({_n: _ns for _n in _names})
del _names, _ns


def parse_time_units(units):
    """``"<unit> since <reference>"`` -> ``(nanoseconds per unit, reference
    as datetime64[ns], UTC)``. The reference is ISO (``1970-01-01``,
    ``1970-01-01 00:00:00``, ``1970-01-01T00:00:00Z``) or slashed as the
    openEO Zarr writer has it, ``01/01/1970 - 00:00:00`` (time optional):
    day first, month first only where the day cannot be a month; a slashed
    date whose day and month cannot be told apart raises ValueError
    ("ambiguous"), as does any other form."""
    import pandas as pd

    m = re.fullmatch(r"\s*([A-Za-z]+)\s+since\s+(.+?)\s*", str(units), re.I)
    if not m:
        raise ValueError(f"time units {units!r} are not '<unit> since <date>'")
    unit, ref = m.group(1).lower(), m.group(2).strip()
    if unit not in TIME_UNIT_NS:
        raise ValueError(f"time unit {m.group(1)!r} of {units!r} not understood")
    s = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})(?:\s*(?:-|T|\s)\s*(\d{1,2}):(\d{2})"
                     r"(?::(\d{2})(?:\.(\d{1,9}))?)?)?\s*(?:Z|UTC|GMT|[+-]00:?00)?", ref, re.I)
    if s:
        a, b, year = int(s.group(1)), int(s.group(2)), int(s.group(3))
        if a == b or (a > 12 >= b):
            day, month = a, b
        elif b > 12 >= a:
            day, month = b, a
        else:
            raise ValueError(f"reference date {ref!r} of {units!r} is ambiguous: day and month "
                             f"cannot be told apart")
        iso = (f"{year:04d}-{month:02d}-{day:02d}T{int(s.group(4) or 0):02d}:"
               f"{int(s.group(5) or 0):02d}:{int(s.group(6) or 0):02d}"
               + (f".{s.group(7)}" if s.group(7) else ""))
        return TIME_UNIT_NS[unit], np.datetime64(iso, "ns")
    if not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:[T\s]+\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
                        r"\s*(?:Z|UTC|GMT|[+-]\d{2}:?\d{2})?", ref, re.I):
        raise ValueError(f"reference date {ref!r} of {units!r} not understood")
    ts = pd.Timestamp(re.sub(r"\s*(UTC|GMT)$", "", ref, flags=re.I))
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return TIME_UNIT_NS[unit], np.datetime64(ts.to_datetime64(), "ns")


def decode_time_values(values, units):
    """Numeric time values (any byte order) in ``units`` (``parse_time_units``)
    -> datetime64[ns], exactly for integers; floats are split into their
    whole units (exact) and the fraction (rounded to the ns), so whole
    milliseconds as float64 stay exact too."""
    ns, t0 = parse_time_units(units)
    v = np.asarray(values)
    if v.dtype.kind not in "iuf":
        raise ValueError(f"time values of dtype {v.dtype} are not numbers")
    v = v.astype(v.dtype.newbyteorder("="))
    if v.dtype.kind in "iu":
        v = v.astype("int64")
        if v.size and int(np.abs(v).max()) > np.iinfo("int64").max // ns:
            raise ValueError(f"time values up to {int(np.abs(v).max())} {units!r} overflow")
        off = v * ns
    else:
        if not np.all(np.isfinite(v)):
            raise ValueError("non-finite time values")
        v = v.astype("float64")
        whole = np.floor(v)
        if v.size and float(np.abs(whole).max()) > np.iinfo("int64").max // ns:
            raise ValueError(f"time values up to {float(np.abs(v).max())} {units!r} overflow")
        off = whole.astype("int64") * ns + np.round((v - whole) * ns).astype("int64")
    return t0 + off.astype("timedelta64[ns]")


def _decode_time(var, notes=None):
    import pandas as pd

    v = np.asarray(var.values)
    if np.issubdtype(v.dtype, np.datetime64):
        return v.astype("datetime64[ns]")
    if v.dtype.kind in "iuf":
        units = var.attrs.get("units")
        if not units or not re.search(r"\bsince\b", str(units), re.I):
            raise FormatError(f"the time coordinate is numeric without CF units "
                              f"(attrs {var.attrs}); cannot tell what {v[:3]} mean")
        if v.dtype.kind == "f" and v.dtype.itemsize < 8 and v.size:
            try:
                unit_ns = parse_time_units(units)[0]
            except ValueError:
                unit_ns = None
            vn = v.astype(v.dtype.newbyteorder("="))
            if unit_ns and np.all(np.isfinite(vn)):
                step_s = float(np.spacing(np.abs(vn).max())) * unit_ns / 1e9
                if step_s > 1.0:
                    raise FormatError(f"the time coordinate is {v.dtype} in {str(units)!r}: at "
                                      f"these values one {vn.dtype.name} step is {step_s:g} s, so "
                                      f"the times are rounded by up to {step_s / 2:g} s; refused "
                                      f"(times must be int64 or float64)")
        cal = var.attrs.get("calendar")
        if cal is None or str(cal).lower() in STANDARD_CALENDARS:
            try:
                out = decode_time_values(v, units)
            except ValueError as exc:
                if "ambiguous" in str(exc):
                    raise FormatError(f"the time coordinate: {exc}") from exc
                out = None                      # another form: xarray's CF decoder below
            if out is not None:
                if notes is not None and "/" in str(units):
                    t0 = parse_time_units(units)[1]
                    notes.append(f"time units {str(units)!r} decoded explicitly (reference date "
                                 f"read as {str(t0)[:19]}, {str(var.dtype)} values)")
                return out
        from xarray.coding.times import decode_cf_datetime

        out = decode_cf_datetime(v, str(units), var.attrs.get("calendar", "standard"))
        return pd.to_datetime([str(x) for x in np.ravel(out)]).values.astype("datetime64[ns]") \
            if np.asarray(out).dtype == object else np.asarray(out).astype("datetime64[ns]")
    if v.dtype.kind in "USO":
        t = pd.to_datetime(np.asarray(v).astype(str), utc=True)
        return t.tz_convert(None).values.astype("datetime64[ns]")
    raise FormatError(f"time coordinate of dtype {v.dtype} not understood")


# ─────────────────────────────────────────────────────────────────────────────
#  interpretation: delivered layout -> canonical (t, y, x) bands
# ─────────────────────────────────────────────────────────────────────────────

#: non-spectral SENTINEL2_L2A layers recognised after a prefix ('S2_SCL')
OTHER_LAYERS = ("SCL", "CLD", "SNW", "AOT", "WVP", "CLP", "CLM")
#: the angle bands, and their aliases in the CDSE collection metadata
ANGLE_KEYS = {b.upper(): b.upper() for b in ANGLE_BANDS}
ANGLE_KEYS.update({"SZA": "SUNZENITHANGLES", "SAA": "SUNAZIMUTHANGLES", "VZA": "VIEWZENITHMEAN",
                   "VAA": "VIEWAZIMUTHMEAN", "APPROXIMATEVIEWZENITH": "VIEWZENITHMEAN",
                   "APPROXIMATEVIEWAZIMUTH": "VIEWAZIMUTHMEAN"})


def band_key(name):
    """Comparable band name: 'b2', 'B02', 'SENTINEL2_L2A:B02',
    'SENTINEL2_L2A_B02' -> 'B02'; 'b8a' -> 'B8A'; 'scl', 'S2_SCL' -> 'SCL';
    'sunZenithAngles', 'S2_sunZenithAngles', 'SZA' -> 'SUNZENITHANGLES'."""
    s = re.split(r"[:/]", str(name).strip())[-1].upper()
    tail = re.split(r"[_.\-\s]+", s)[-1]
    for cand in (s, tail):
        m = re.fullmatch(r"B0?(\d{1,2})(A?)", cand)
        if m:
            n = int(m.group(1))
            return f"B{n}A" if m.group(2) else f"B{n:02d}"
    for cand in (s, tail):
        if cand in ANGLE_KEYS:
            return ANGLE_KEYS[cand]
    return tail if tail in OTHER_LAYERS else s


FILL_ATTRS = ("_FillValue", "missing_value", "nodata", "no_data", "nodata_value", "NoData",
              "no_data_value")
SCALE_ATTRS = ("scale_factor", "add_offset", "scale", "offset")


def _wkt_from(attrs):
    """A CRS from attributes: WKT text (``_CRS``: the openEO Zarr writer's),
    or an EPSG code made into WKT."""
    for key in ("crs_wkt", "spatial_ref", "proj:wkt2", "wkt", "crs", "_CRS"):
        v = attrs.get(key)
        if isinstance(v, str) and ("PROJCS" in v or "PROJCRS" in v or "GEOGCS" in v
                                   or "GEOGCRS" in v):
            return v, key
    for key in ("epsg", "epsg_code", "proj:epsg", "proj:code", "crs", "EPSG"):
        v = attrs.get(key)
        if v is None:
            continue
        m = re.fullmatch(r"(?:EPSG:)?(\d{4,5})", str(v).strip(), re.I)
        if m:
            code = int(m.group(1))
            try:
                from pyproj import CRS

                return CRS.from_epsg(code).to_wkt("WKT1_GDAL"), key
            except Exception:
                return f"EPSG:{code}", key
    return None, None


def _axes_from_transform(raw, ds, crs_hints=()):
    """Pixel-centre x/y from georeferencing attributes when the store has no
    coordinate arrays: GDAL ``GeoTransform`` (x0 dx 0 y0 0 dy) or an affine
    ``transform`` / ``proj:transform`` (dx 0 x0 0 dy y0) on a grid-mapping
    variable, the root, a band, or the job's STAC metadata."""
    if "y" not in ds.dims or "x" not in ds.dims:
        return None
    H, W = ds.sizes["y"], ds.sizes["x"]
    sources = [(f"variable {v!r}", ds[v].attrs) for v in ds.variables
               if str(v).lower() in CRS_NAMES or ds[v].attrs.get("GeoTransform")]
    sources += [("root attributes", raw.attrs)]
    sources += [(f"attributes of {v!r}", ds[v].attrs) for v in ds.data_vars]
    sources += list(crs_hints)
    for label, attrs in sources:
        gt = attrs.get("GeoTransform")
        if gt is not None:
            vals = [float(v) for v in (gt.split() if isinstance(gt, str) else gt)]
            if len(vals) == 6 and vals[2] == 0 and vals[4] == 0:
                x0, dx, _, y0, _, dy = vals
                return (x0 + dx * (np.arange(W) + 0.5), y0 + dy * (np.arange(H) + 0.5),
                        f"GeoTransform of {label}")
        for key in ("proj:transform", "spatial:transform", "transform"):
            tr = attrs.get(key)
            if isinstance(tr, (list, tuple)) and len(tr) in (6, 9):
                a, b, c, d, e, f = (float(v) for v in tr[:6])
                if b == 0 and d == 0:
                    shape = attrs.get("proj:shape") or attrs.get("spatial:shape")
                    if shape and list(shape)[-2:] != [H, W]:
                        continue
                    return (c + a * (np.arange(W) + 0.5), f + e * (np.arange(H) + 0.5),
                            f"{key} of {label}")
    return None


@dataclass
class Interpretation:
    bands: dict                  # band -> DataArray (t, y, x), delivered values
    t: object
    y: object
    x: object
    crs_wkt: str
    crs_source: str
    fills: dict                  # band -> fill (None: none)
    fill_sources: dict
    delivered: dict              # band -> how it was found
    notes: list
    warnings: list
    source: str
    attrs: dict = field(default_factory=dict)    # root attributes as delivered
    zarr_fills: dict = field(default_factory=dict)  # band -> Zarr fill_value as delivered
    nodata_lines: list = field(default_factory=list)  # the no-data decision, for the report
    groups: list = field(default_factory=list)   # bands sharing delivered chunks
    coord_dtypes: dict = field(default_factory=dict)  # x / y dtype as delivered ('>f4', ...)
    raw_axes: dict = field(default_factory=dict)  # float32 x / y as delivered (in float64)
    registration: dict = None    # where the frame sits in the delivered arrays (register_delivery)
    placement: dict = None       # the angles: how they were placed (angles_placement)


def _low_precision(dtype):
    """A float dtype narrower than float64 (the backend's Zarr x/y: '>f4')."""
    try:
        dt = np.dtype(dtype)
    except TypeError:
        return False
    return dt.kind == "f" and dt.itemsize < 8


def date_from_path(path):
    """A YYYY-MM-DD / YYYYMMDD date in the last components of a path."""
    import pandas as pd

    for comp in reversed(Path(str(path)).parts[-3:]):
        for m in re.finditer(r"(?<!\d)(20\d{2})-?(0[1-9]|1[0-2])-?(0[1-9]|[12]\d|3[01])(?!\d)", comp):
            try:
                return str(pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}").date())
            except ValueError:
                continue
    return None


def _as_naive_ns(value):
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64(), "ns")


def _counts(a, values, max_dates=REPORT_CANDIDATES):
    """``({value: count}, dates read)`` over up to ``max_dates`` dates of a
    (t, y, x) band spread over the axis (all dates of a short window)."""
    T = a.sizes["t"]
    idx = np.unique(np.linspace(0, T - 1, min(T, max_dates)).round().astype(int)) if T else []
    arr = np.asarray(a.isel(t=list(idx)).values)
    out = {}
    for v in values:
        if v is None:
            continue
        hit = np.isnan(arr) if isinstance(v, float) and np.isnan(v) else arr == v
        out[v] = int(hit.sum())
    return out, len(idx), int(arr.size)


def _same_value(a, b):
    if a is None or b is None:
        return a is b
    fa, fb = isinstance(a, float) and np.isnan(a), isinstance(b, float) and np.isnan(b)
    return (fa and fb) or (not fa and not fb and a == b)


def interpret(raw, bands, resolution=RESOLUTION_M, crs_hints=(), ref_grid=None,
              require_all=True, date_hint=None, nodata=None, strict_nodata=True,
              count_nodata=False, product=None):
    """Find the requested bands, the axes, the grid and the CRS in one
    RawData, with the values untouched. Raises FormatError, with what was
    seen, whenever the layout leaves a doubt. ``require_all=False`` (one of
    several assets) accepts a subset of the bands, but not none of them.

    ``date_hint``: the date of a store without a time dimension (one store
    per date), from the job's STAC metadata; else a date in its path.
    ``nodata``: the no-data value of bands that declare none (``--nodata``).
    ``strict_nodata``: an undecidable no-data value raises (production);
    False (``--test``) reports it and goes on with -32768.
    ``count_nodata``: count the candidate no-data values of every band for
    the report (always done for an ambiguous band).
    ``product``: ``ANGLES`` for the angle bands, whose no-data is decided
    otherwise (a Zarr fill_value, even zarr's default 0, IS their no-data:
    no angle of these sites is exactly that value, and the store maps it to
    NaN with NaN itself) and whose dtype is expected float32; None or
    ``MAIN``: the main bands, as always."""
    angles = product is not None and product.name == ANGLES.name
    ds = raw.ds
    notes, warns = list(raw.notes), []
    # axes
    rename, axes = {}, {}
    for d in ds.dims:
        ax = axis_of(d)
        if ax and ax not in axes:
            axes[ax] = d
    if "t" not in axes and "y" in axes and "x" in axes:
        when, wsrc = (date_hint, "the job's STAC metadata") if date_hint else \
            (date_from_path(raw.source), "the path")
        if when is None:
            raise FormatError(f"{raw.source}: no 't' dimension among {list(ds.dims)} and no date in "
                              f"the job's metadata or in the path")
        yx = {axes["y"], axes["x"]}
        ds = ds.assign({v: ds[v].expand_dims("t") for v in ds.data_vars if yx <= set(ds[v].dims)})
        ds = ds.assign_coords(t=("t", np.array([_as_naive_ns(when)])))
        axes["t"] = "t"
        notes.append(f"no time dimension: one date {when} taken from {wsrc}")
    for ax in ("t", "y", "x"):
        if ax not in axes:
            raise FormatError(f"{raw.source}: no {ax!r} dimension among {list(ds.dims)}")
    for ax, d in axes.items():
        if d != ax:
            rename[d] = ax
    if rename:
        ds = ds.rename(rename)
        notes.append(f"dimensions renamed {rename}")
    # a coordinate stored under another name than its dimension ('time' on 't')
    for ax in ("t", "y", "x", "bands"):
        if ax in ds.dims and ax not in ds.coords:
            alt = [v for v in ds.variables if v != ax and ds[v].dims == (ax,)
                   and axis_of(v) == ax]
            if len(alt) == 1:
                ds = ds.assign_coords({ax: (ax, np.asarray(ds[alt[0]].values), ds[alt[0]].attrs)})
                ds = ds.drop_vars(alt[0])
                notes.append(f"coordinate {alt[0]!r} used for dimension {ax!r}")
    if ("x" not in ds.coords or "y" not in ds.coords):
        got = _axes_from_transform(raw, ds, crs_hints)
        if got is not None:
            xs, ys, src = got
            ds = ds.assign_coords(x=("x", xs), y=("y", ys))
            notes.append(f"x/y computed from {src} (pixel centres)")
    if "t" not in ds.coords:
        raise FormatError(f"{raw.source}: the time dimension has no coordinate values")
    tvals = _decode_time(ds["t"], notes)
    found, delivered, src_attrs = {}, {}, {}
    groups = None
    wanted = {band_key(b): b for b in bands}
    banddim = axes.get("bands")
    if banddim is not None:
        holders = [v for v in ds.data_vars if "bands" in ds[v].dims]
        if len(holders) != 1:
            raise FormatError(f"{raw.source}: {len(holders)} variables carry the bands dimension "
                              f"({holders}); expected one")
        hv = holders[0]
        if "bands" in ds.coords:
            labels = [str(_py(b)) for b in np.asarray(ds["bands"].values)]
        elif ds.sizes["bands"] == len(bands):
            labels = list(bands)
            warns.append(f"variable {hv!r} has a bands dimension without labels; bands taken in "
                         f"the requested order {list(bands)}")
        else:
            raise FormatError(f"{raw.source}: {ds.sizes['bands']} unlabelled bands for "
                              f"{len(bands)} requested")
        for i, lab in enumerate(labels):
            k = band_key(lab)
            if k in wanted:
                found[wanted[k]] = ds[hv].isel(bands=i, drop=True)
                delivered[wanted[k]] = f"{hv}[bands={lab}]"
                src_attrs[wanted[k]] = (hv, ds[hv].attrs)
        notes.append(f"layout: one variable {hv!r} with a bands dimension, labels {labels}")
        data = ds[hv].data
        cb = (int(data.chunksize[ds[hv].dims.index("bands")])
              if getattr(data, "chunksize", None) else ds.sizes["bands"])
        if cb > 1 and len(found) > 1:
            groups = [[b for b in bands if b in found]]
            notes.append(f"{hv!r} holds {cb} bands per chunk: the bands are written together, so "
                         f"each delivered chunk is decoded once per block")
    else:
        cands = [v for v in ds.data_vars if {"y", "x"} <= set(ds[v].dims) and "t" in ds[v].dims]
        keys = {}
        for v in cands:
            keys.setdefault(band_key(v), []).append(v)
        for k, b in wanted.items():
            hits = keys.get(k, [])
            if len(hits) > 1:
                raise FormatError(f"{raw.source}: several variables look like band {b}: {hits}")
            if hits:
                found[b] = ds[hits[0]]
                delivered[b] = hits[0]
                src_attrs[b] = (hits[0], ds[hits[0]].attrs)
        if not found and len(cands) == len(bands) and all(
                re.fullmatch(r"(band|b|layer|data)?_?\d+", str(c), re.I) for c in cands):
            for c, b in zip(cands, bands):
                found[b], delivered[b], src_attrs[b] = ds[c], c, (c, ds[c].attrs)
            warns.append(f"generic variable names {cands} mapped to the requested order")
        notes.append(f"layout: one variable per band ({len(cands)} band-like variables: {cands})")
    missing = [b for b in bands if b not in found]
    if missing and (require_all or not found):
        raise FormatError(f"{raw.source}: requested bands {missing} not found; data variables "
                          f"{ {v: list(ds[v].dims) for v in ds.data_vars} }")
    others = [v for v in ds.data_vars if v not in {d.split('[')[0] for d in delivered.values()}]
    if others:
        notes.append(f"other variables (not bands, kept out of the store): {others}")
    # dims per band
    for b in list(found):
        a = found[b]
        extra = [d for d in a.dims if d not in ("t", "y", "x")]
        for d in extra:
            if a.sizes[d] == 1:
                a = a.isel({d: 0}, drop=True)
            else:
                raise FormatError(f"band {b} has an extra dimension {d!r} of length {a.sizes[d]}")
        a = a.transpose("t", "y", "x")
        # positions only from here on: the axes travel in the Interpretation
        found[b] = a.drop_vars(list(a.coords))
    # grid
    if "x" not in ds.coords or "y" not in ds.coords:
        raise FormatError(f"{raw.source}: no x/y coordinate values (coords {list(ds.coords)}); "
                          f"refusing to place the pixels on a grid by guesswork")
    cdtypes = {"x": str(ds["x"].dtype), "y": str(ds["y"].dtype)}      # as delivered
    x = np.asarray(ds["x"].values, dtype="float64")
    y = np.asarray(ds["y"].values, dtype="float64")
    order_t = np.argsort(tvals, kind="stable")
    if len(tvals) != len(np.unique(tvals)):
        import pandas as pd

        dup = pd.Index(tvals).duplicated(keep=False)
        raise FormatError(f"{raw.source}: duplicated time labels {sorted(set(map(str, tvals[dup])))[:10]}")
    flips = {}
    if len(y) > 1 and y[1] > y[0]:
        flips["y"] = slice(None, None, -1)
        y = y[::-1]
        notes.append("y was ascending: flipped to descending (north up), values untouched")
    if len(x) > 1 and x[1] < x[0]:
        flips["x"] = slice(None, None, -1)
        x = x[::-1]
        notes.append("x was descending: flipped to ascending, values untouched")
    if not np.all(order_t == np.arange(len(tvals))):
        notes.append("time labels were not sorted: sorted")
    for b in found:
        a = found[b]
        if flips:
            a = a.isel(flips)
        if not np.all(order_t == np.arange(len(tvals))):
            a = a.isel(t=order_t)
        found[b] = a
    tvals = tvals[order_t]
    regular, raw_axes = {}, {}
    for name, v, sign in (("x", x, 1.0), ("y", y, -1.0)):
        if len(v) > 1 and _low_precision(cdtypes[name]):
            # float32 (the backend's Zarr writer): up to half a float32 step
            # of rounding per value; regular within F32_TOL_M, then exact
            k = np.arange(len(v), dtype="float64")
            reg = float(np.median(v - sign * resolution * k)) + sign * resolution * k
            dev = float(np.max(np.abs(v - reg)))
            if dev > F32_TOL_M + 1e-9:
                raise FormatError(f"{raw.source}: {name} (delivered as {cdtypes[name]}) is not the "
                                  f"{resolution} m grid: up to {dev:.3f} m from a regular axis (float32 "
                                  f"rounding allows {F32_TOL_M:g} m); spacing "
                                  f"{np.unique(np.round(np.diff(v), 6))[:5]}")
            notes.append(f"{name} delivered as {cdtypes[name]} (float32): "
                         + (f"within {dev:.3f} m of a regular {resolution:g} m axis; regularised "
                            f"in float64" if dev > 0 else f"a regular {resolution:g} m axis"))
            raw_axes[name] = v.copy()       # snap_to_lattice measures the rounding on these
            v = reg
        elif len(v) > 1:
            d = np.diff(v)
            if not np.allclose(d, sign * resolution, rtol=0, atol=1e-6 * resolution):
                raise FormatError(f"{raw.source}: {name} spacing {np.unique(np.round(d, 6))[:5]} "
                                  f"is not the {resolution} m grid")
        regular[name] = v
    x, y = regular["x"], regular["y"]
    half = resolution / 2.0

    def off_half(v0, name):
        tol = (F32_TOL_M / resolution) if _low_precision(cdtypes[name]) else 1e-6
        q = (v0 - half) / resolution
        return abs(q - round(q)) > tol

    if len(x) and (off_half(x[0], "x") or off_half(y[0], "y")):
        warns.append(f"pixel centres x0={x[0]} y0={y[0]} are not at half-pixel positions of the "
                     f"{resolution} m lattice (the netCDF cubes' are): corners instead of centres?")
    # CRS
    wkt, csrc = None, None
    gm_names = {str(src_attrs[b][1].get("grid_mapping")) for b in found
                if src_attrs[b][1].get("grid_mapping")}
    for gm in sorted(gm_names) + [v for v in ds.variables if str(v).lower() in CRS_NAMES]:
        if gm in ds.variables:
            w, key = _wkt_from(ds[gm].attrs)
            if w:
                wkt, csrc = w, f"variable {gm!r} attribute {key!r}"
                break
    if wkt is None:
        w, key = _wkt_from(raw.attrs)
        if w:
            wkt, csrc = w, f"global attribute {key!r}"
    if wkt is None:
        for b in found:
            w, key = _wkt_from(src_attrs[b][1])
            if w:
                wkt, csrc = w, f"attribute {key!r} of {src_attrs[b][0]!r}"
                break
    if wkt is None:
        for label, hint in crs_hints:
            w, key = _wkt_from(hint)
            if w:
                wkt, csrc = w, f"{label} ({key})"
                break
    if wkt is None and ref_grid is not None:
        def overlap(v, r, name):            # float32 axes: within their rounding of the lattice
            r = np.asarray(r, dtype="float64")
            if _low_precision(cdtypes[name]) and len(r) > 1 and len(v):
                step = r[1] - r[0]
                on = r[0] + np.round((v - r[0]) / step) * step
                if np.max(np.abs(v - on)) <= F32_TOL_M + 1e-9:
                    v = on
            return zc._overlap(v, r)

        ox = overlap(x, ref_grid["x"], "x")
        oy = overlap(y, ref_grid["y"], "y")
        if ox is not None and oy is not None and ref_grid.get("crs_wkt"):
            wkt = ref_grid["crs_wkt"]
            label = {"store": "reference store", "predicted": "reference grid",
                     "request": "requested extent"}.get(ref_grid.get("kind"),
                                                        "reference netCDF cube")
            csrc = (f"{label} {ref_grid['path']} (the delivered coordinates lie on its grid; the "
                    f"delivered data carry no CRS)")
            warns.append(f"CRS not in the delivered data: taken from the {label}")
    if wkt is None:
        raise FormatError(f"{raw.source}: no CRS found (no grid-mapping variable, no crs/EPSG "
                          f"attribute, nothing in the job metadata)")
    # fill and scale, per band. A declared attribute decides; else a Zarr
    # fill_value equal to the standard no-data (-32768, NaN for floats);
    # zarr's DEFAULT fill_value (0) is not a declaration: 0 is a valid value
    # (clear sky in CLD), so the data decide, and an undecidable case stops.
    fills, fsrc, zfills, nlines, ambiguous = {}, {}, {}, [], []
    for b in found:
        vname, attrs = src_attrs[b]
        dtype = found[b].dtype
        std = zc.FILL if dtype.kind in "iu" else float("nan")
        declared, key = None, None
        for k in FILL_ATTRS:
            if attrs.get(k) is not None:
                declared, key = _py(attrs[k]), k
                break
        zfill = raw.info.get(vname, {}).get("fill_value")
        if raw.kind == "netcdf":
            zfill = None                  # netCDF's fill is the declared _FillValue
        if zfill is not None and isinstance(zfill, float) and np.isnan(zfill) and dtype.kind != "f":
            zfill = None
        zfills[b] = zfill
        if declared is not None:
            fill, src = declared, f"attribute {key!r}"
            if zfill is not None and not _same_value(zfill, declared):
                notes.append(f"band {b}: Zarr fill_value {zfill!r} differs from the declared "
                             f"{key} {declared!r}; the declared one is the no-data")
        elif nodata is not None:
            fill, src = nodata, "--nodata"
        elif zfill is None:
            fill, src = std, ("none declared: -32768 assumed" if dtype.kind in "iu"
                              else "none declared: NaN")
        elif _same_value(zfill, std):
            fill, src = zfill, "Zarr fill_value"
        elif angles:
            fill, src = zfill, (f"Zarr fill_value {zfill!r} (angles: no angle of these sites is "
                                f"exactly {zfill!r}; it becomes NaN in the store, as NaN does)")
        else:
            cnt, nd, npx = _counts(found[b], [std, zfill])
            line = (f"  {b}: Zarr fill_value {zfill!r} with no no-data attribute; in {nd} date(s) "
                    f"({npx} px): {std!r} x {cnt.get(std, 0)}, {zfill!r} x {cnt.get(zfill, 0)}")
            if cnt.get(std, 0) > 0:
                fill = std
                src = (f"{std!r} found in the data; the Zarr fill_value {zfill!r} is not declared "
                       f"as no-data (kept as delivered_zarr_fill_value)")
                warns.append(f"band {b}: {src}")
                nlines.append(line + f" -> no-data {std!r}")
            else:
                fill, src = std, (f"AMBIGUOUS: Zarr fill_value {zfill!r} undeclared and {std!r} "
                                  f"absent; {std!r} used (pass --nodata to decide)")
                ambiguous.append(b)
                nlines.append(line + " -> AMBIGUOUS")
        if count_nodata and not any(ln.startswith(f"  {b}:") for ln in nlines):
            cands = ([std] + ([zfill] if zfill is not None else [])
                     + ([0] if dtype.kind in "iu" else []))
            if angles:              # NaN once; exact zeros too (an undeclared no-data?)
                cands = [c for i, c in enumerate(cands + [0.0])
                         if not any(_same_value(c, d) for d in (cands + [0.0])[:i])]
            cnt, nd, npx = _counts(found[b], cands)
            nlines.append(f"  {b}: no-data {fill!r} ({src}); in {nd} date(s) ({npx} px): "
                          + ", ".join(f"{k!r} x {v}" for k, v in cnt.items()))
        if dtype.kind in "iu" and fill is not None and not isinstance(fill, (int, np.integer)):
            if isinstance(fill, float) and float(fill).is_integer():
                fill = int(fill)
            else:
                raise FormatError(f"band {b}: integer data with a non-integer fill {fill!r}")
        fills[b], fsrc[b] = fill, src
        if angles:
            if np.dtype(dtype).newbyteorder("=") != np.dtype("float32"):
                warns.append(f"band {b} delivered as {dtype}, not float32: converted to float32 "
                             f"(no-data -> NaN)")
            for k in SCALE_ATTRS:
                if k in attrs:
                    warns.append(f"band {b}: delivered attribute {k}={attrs[k]!r} "
                                 + ("DECODED (CF packing; recorded as applied_" + k + ")"
                                    if k in CF_PACKING else f"NOT applied (kept as delivered_{k})"))
            continue
        if np.dtype(dtype).newbyteorder("=") != np.dtype("int16"):
            warns.append(f"band {b} delivered as {dtype}, not int16: kept as delivered")
        elif fill != zc.FILL:
            warns.append(f"band {b}: delivered fill {fill} ({src}), not -32768: kept as delivered")
        for k in SCALE_ATTRS:
            if k in attrs:
                warns.append(f"band {b}: delivered attribute {k}={attrs[k]!r} NOT applied "
                             f"(kept as delivered_{k})")
    if ambiguous:
        block = (["AMBIGUOUS NO-DATA: these bands declare no no-data value, their Zarr fill_value "
                  "is zarr's default and -32768 does not occur in the sampled dates, so whether "
                  "the fill_value marks no data cannot be told from the data:"]
                 + [ln for ln in nlines if ln.split(":")[0].strip() in ambiguous]
                 + ["  decide with --nodata VALUE (e.g. --nodata 0 if the fill_value is the "
                    "backend's no-data), after looking at the bands' values above"])
        if strict_nodata:
            raise AmbiguousNoData("\n".join(block))
        warns.append(f"AMBIGUOUS NO-DATA for {ambiguous}: -32768 used (see the no-data block)")
        nlines = block + [ln for ln in nlines if ln.split(":")[0].strip() not in ambiguous]
    if groups is None:
        groups = [[b] for b in bands if b in found]
    return Interpretation(bands=found, t=tvals, y=y, x=x, crs_wkt=wkt, crs_source=csrc,
                          fills=fills, fill_sources=fsrc, delivered=delivered, notes=notes,
                          warnings=warns, source=raw.source, attrs=dict(raw.attrs),
                          zarr_fills=zfills, nodata_lines=nlines, groups=groups,
                          coord_dtypes=cdtypes, raw_axes=raw_axes)


def combine(parts):
    """One Interpretation from several assets: different bands on one grid
    and dates (merge), or the same bands on one grid with disjoint dates
    (concatenate in time). Anything else stops with a FormatError."""
    if len(parts) == 1:
        return parts[0]
    import xarray as xr

    p0 = parts[0]
    for p in parts[1:]:
        if not (np.array_equal(p.x, p0.x) and np.array_equal(p.y, p0.y)):
            raise FormatError(f"assets on different grids ({p0.source} vs {p.source}); the "
                              f"normaliser does not mosaic tiles")
        if not zc.crs_equal(p.crs_wkt, p0.crs_wkt):
            raise FormatError(f"assets in different CRS ({p0.source} vs {p.source})")
    same_t = all(np.array_equal(p.t, p0.t) for p in parts)
    band_sets = [set(p.bands) for p in parts]
    notes = [f"{len(parts)} assets combined"] + [n for p in parts for n in p.notes]
    warns = [w for p in parts for w in p.warnings]
    nlines = [ln for p in parts for ln in p.nodata_lines]
    if same_t and all(not (a & b) for i, a in enumerate(band_sets) for b in band_sets[i + 1:]):
        merged = {}
        for p in parts:
            merged.update(p.bands)
        return Interpretation(
            bands=merged, t=p0.t, y=p0.y, x=p0.x, crs_wkt=p0.crs_wkt, crs_source=p0.crs_source,
            fills={b: f for p in parts for b, f in p.fills.items()},
            fill_sources={b: f for p in parts for b, f in p.fill_sources.items()},
            delivered={b: f"{Path(p.source).name}:{d}" for p in parts for b, d in p.delivered.items()},
            notes=notes + ["assets hold different bands: merged"], warnings=warns,
            source="; ".join(p.source for p in parts), attrs=p0.attrs,
            zarr_fills={b: f for p in parts for b, f in p.zarr_fills.items()},
            nodata_lines=nlines, groups=[g for p in parts for g in p.groups],
            coord_dtypes=p0.coord_dtypes, raw_axes=p0.raw_axes)
    if all(s == band_sets[0] for s in band_sets):
        allt = np.concatenate([p.t for p in parts])
        if len(np.unique(allt)) != len(allt):
            raise FormatError("assets share bands and overlap in time: cannot combine")
        order = np.argsort(allt, kind="stable")
        bands = {}
        for b in parts[0].bands:
            cat = xr.concat([p.bands[b] for p in parts], dim="t")
            bands[b] = cat.isel(t=order)
        fills = parts[0].fills
        for p in parts[1:]:
            for b in fills:
                if not _same_value(fills[b], p.fills[b]):
                    raise FormatError(f"band {b}: different fills across assets "
                                      f"({fills[b]!r} in {p0.source}, {p.fills[b]!r} in {p.source})")
        return Interpretation(
            bands=bands, t=allt[order], y=p0.y, x=p0.x, crs_wkt=p0.crs_wkt,
            crs_source=p0.crs_source, fills=fills, fill_sources=p0.fill_sources,
            delivered=p0.delivered, notes=notes + ["assets hold different dates: concatenated"],
            warnings=warns, source="; ".join(p.source for p in parts), attrs=p0.attrs,
            zarr_fills=p0.zarr_fills, nodata_lines=nlines, groups=p0.groups,
            coord_dtypes=p0.coord_dtypes, raw_axes=p0.raw_axes)
    raise FormatError(f"{len(parts)} assets that neither split the bands nor the dates: "
                      f"{[sorted(s) for s in band_sets]}")


def _pad_tyx(arr, top, bottom, left, right, fill):
    """A (t, y, x) array padded with ``fill`` on y and x.

    dask's own ``pad`` cuts the margin into blocks as large as the edge
    chunk (a small window put on a large grid becomes ~10^4-10^5 tasks).
    Here each margin is one block across the padded axis, chunked like the
    data along the other one, so nothing is rechunked and the delivered
    chunks are read as they are."""
    if not hasattr(arr, "dask"):
        return np.pad(np.asarray(arr), ((0, 0), (top, bottom), (left, right)),
                      constant_values=fill)
    import dask.array as da

    tch, (T, H, W) = arr.chunks[0], arr.shape

    def full(ychunks, xchunks):
        return da.full((T, sum(ychunks), sum(xchunks)), fill, dtype=arr.dtype,
                       chunks=(tch, ychunks, xchunks))

    row = (([full(arr.chunks[1], (left,))] if left else []) + [arr]
           + ([full(arr.chunks[1], (right,))] if right else []))
    mid = da.concatenate(row, axis=2) if len(row) > 1 else arr
    col = (([full((top,), mid.chunks[2])] if top else []) + [mid]
           + ([full((bottom,), mid.chunks[2])] if bottom else []))
    return da.concatenate(col, axis=1) if len(col) > 1 else mid


def margin_summary(interp, r0, r1, c0, c1, max_dates=REPORT_CANDIDATES):
    """What the delivered rows / columns outside rows ``r0:r1`` and columns
    ``c0:c1`` hold (the part an alignment crops away: the Zarr writer pads
    its arrays to whole chunks), per band, over up to ``max_dates`` dates
    spread over the axis, counted chunk by chunk (dask): ``{"shape",
    "rows", "cols", "dates", "of", "bands": {band: {"cells", "fill", "nan",
    "zero", "other", "min", "max"}}, "other"}``; ``other`` = values that are
    neither the band's fill, NaN nor 0. None when nothing is cropped."""
    import dask
    import dask.array as da

    if not interp.bands:
        return None
    T, H, W = next(iter(interp.bands.values())).shape
    regions = [(slice(0, r0), slice(0, W)), (slice(r1, H), slice(0, W)),
               (slice(r0, r1), slice(0, c0)), (slice(r0, r1), slice(c1, W))]
    regions = [(ys, xs) for ys, xs in regions if ys.stop > ys.start and xs.stop > xs.start]
    if not regions or not T:
        return None
    idx = [int(i) for i in np.unique(np.linspace(0, T - 1, min(T, max_dates)).round())]
    lazy, cells = {}, {}
    for b, a in interp.bands.items():
        d = a.data if hasattr(a.data, "dask") else da.from_array(np.asarray(a.data), chunks=-1)
        d = d[idx]
        fill = interp.fills.get(b)
        real_fill = fill is not None and not (isinstance(fill, float) and np.isnan(fill))
        per = []
        cells[b] = 0
        for ys, xs in regions:
            r = d[:, ys, xs]
            cells[b] += int(np.prod(r.shape))
            # dtype by keyword: dask 2026 takes a positional one as float64
            none = da.zeros(r.shape, dtype=bool, chunks=r.chunks)
            nan = da.isnan(r) if r.dtype.kind == "f" else none
            fil = ((r == fill) & ~nan) if real_fill else none
            zero = (r == 0) & ~fil & ~nan
            other = ~(fil | nan | zero)
            v = r.astype("float64")
            per.append([fil.sum(), nan.sum(), zero.sum(), other.sum(),
                        da.where(other, v, np.inf).min(), da.where(other, v, -np.inf).max()])
        lazy[b] = per
    (got,) = dask.compute(lazy)
    out = {"shape": [int(H), int(W)], "rows": [int(r0), int(r1)], "cols": [int(c0), int(c1)],
           "dates": len(idx), "of": int(T), "bands": {}, "other": 0, "zero": 0}
    for b, per in got.items():
        s = {"cells": cells[b], "fill": int(sum(p[0] for p in per)),
             "nan": int(sum(p[1] for p in per)), "zero": int(sum(p[2] for p in per)),
             "other": int(sum(p[3] for p in per))}
        lo, hi = min(float(p[4]) for p in per), max(float(p[5]) for p in per)
        s["min"], s["max"] = (lo, hi) if s["other"] else (None, None)
        out["bands"][b] = s
        out["other"] += s["other"]
        out["zero"] += s["zero"]
    return out


def format_margin(m, fills=None, how="coordinates"):
    """``(line, clean)`` of a ``margin_summary``: ``clean`` when the margin
    held nothing but the fill / NaN. Either way the line is a NOTE: the
    margin lies outside the frame (cropped by coordinates on the checked
    lattice, or at the registered offset: ``how``), so data there are pixels
    the request did not ask for, not pixels lost from the store. On a sample
    of the dates, the line says so."""
    nb = next(iter(m["bands"].values()))["cells"] // max(1, m["dates"])
    sampled = m["dates"] < m["of"]
    head = (f"cropped-away margin: the delivered rows / columns outside rows {m['rows'][0]}:"
            f"{m['rows'][1]}, columns {m['cols'][0]}:{m['cols'][1]} of {m['shape'][0]} x "
            f"{m['shape'][1]} ({nb} values per band and date; {m['dates']} of {m['of']} dates"
            + (" sampled" if sampled else "") + ")")
    on = " on the sampled dates" if sampled else ""
    if not m["other"] and not m["zero"]:
        what = sorted({repr(f) for f in (fills or {}).values()}) if fills else []
        return (head + ": only the fill" + (f" ({', '.join(what)})" if what else "") + " / NaN"
                + on), True
    parts = []
    for b, s in m["bands"].items():
        n = max(1, s["cells"])
        bits = [f"{k} {s[k] / n * 100:.1f} %" for k in ("fill", "nan", "zero", "other") if s[k]]
        if s["other"]:
            bits.append(f"(values {s['min']:g}..{s['max']:g})")
        parts.append(f"{b} " + " ".join(bits))
    return (head + f" holds data{on}, not only the fill: " + "; ".join(parts)
            + f" - pixels OUTSIDE the requested frame, cut off by {how} (not in the store; "
            "nothing inside the frame is lost)"), False


def align_to_grid(interp, ref_x, ref_y, res=RESOLUTION_M):
    """Crop / pad the bands of ``interp`` (same 10 m lattice, another
    extent) onto the reference axes, padding with each band's fill: an exact
    index shift, values untouched. Returns ``(note, padded, margin)``: what
    was done, whether anything was padded, and what the cropped-away rows /
    columns held (``margin_summary``; None when nothing was cut)."""
    import xarray as xr

    rx = np.asarray(ref_x, dtype="float64")
    ry = np.asarray(ref_y, dtype="float64")
    x, y = np.asarray(interp.x), np.asarray(interp.y)
    ox = int(round((x[0] - rx[0]) / res))          # reference column of delivered column 0
    oy = int(round((ry[0] - y[0]) / res))          # reference row of delivered row 0 (y down)
    c0, c1 = max(0, -ox), min(len(x), len(rx) - ox)
    r0, r1 = max(0, -oy), min(len(y), len(ry) - oy)
    if c1 <= c0 or r1 <= r0:
        raise FormatError(f"the delivered grid ({len(y)} x {len(x)} px from ({x[0]}, {y[0]})) does "
                          f"not overlap the reference grid ({len(ry)} x {len(rx)} px from "
                          f"({rx[0]}, {ry[0]}))")
    left, top = ox + c0, oy + r0
    right, bottom = len(rx) - left - (c1 - c0), len(ry) - top - (r1 - r0)
    margin = margin_summary(interp, r0, r1, c0, c1)
    for b, a in list(interp.bands.items()):
        a = a.isel(y=slice(r0, r1), x=slice(c0, c1))
        if top or bottom or left or right:
            fill = interp.fills.get(b)
            if fill is None:
                raise FormatError(f"band {b} has no fill value to pad it onto the reference grid")
            if a.dims != ("t", "y", "x"):
                raise FormatError(f"band {b}: dims {a.dims}, expected ('t', 'y', 'x')")
            a = xr.DataArray(_pad_tyx(a.data, top, bottom, left, right, fill), dims=a.dims,
                             attrs=a.attrs, name=a.name)
        interp.bands[b] = a
    interp.x, interp.y = rx.copy(), ry.copy()
    note = (f"delivered grid {len(y)} x {len(x)} px from ({x[0]}, {y[0]}) put on the reference grid "
            f"{len(ry)} x {len(rx)} px from ({rx[0]}, {ry[0]}): cropped rows {r0}:{r1} cols "
            f"{c0}:{c1}, padded with the fill top {top} bottom {bottom} left {left} right {right}")
    return note, bool(top or bottom or left or right), margin


def canonical_dataset(interp, bands, product=None):
    """The store's dataset (lazy): bands, coordinates, ``crs``. The main
    bands keep their delivered values and dtype; ``product`` ANGLES:
    float32 degrees with NaN (:func:`canonical_angles`)."""
    if product is not None and product.name == ANGLES.name:
        return canonical_angles(interp, bands)
    data_vars = {}
    for b in bands:
        a = interp.bands[b]
        attrs = {"long_name": b, "units": "", "grid_mapping": "crs",
                 "native_resolution_m": zc.NATIVE_RES_M.get(b, RESOLUTION_M),
                 "resampling": ("none (native 10 m grid)" if zc.NATIVE_RES_M.get(b) == 10 else
                                "nearest neighbour 20 m -> 10 m (resample_spatial method 'near')"),
                 "description": BAND_NOTES.get(b, SPECTRAL_NOTE),
                 "delivered_as": interp.delivered.get(b, ""),
                 "delivered_dtype": str(a.dtype)}
        for k in SCALE_ATTRS:
            if k in a.attrs:
                attrs[f"delivered_{k}"] = _py(a.attrs[k])
        zf = interp.zarr_fills.get(b)
        if zf is not None and not _same_value(zf, interp.fills.get(b)):
            attrs["delivered_zarr_fill_value"] = zf
        data = a.data
        native = np.dtype(a.dtype).newbyteorder("=")
        if np.dtype(a.dtype) != native:
            data = data.astype(native)
        data_vars[b] = (("t", "y", "x"), data, attrs)
    return _on_grid(data_vars, interp, RESOLUTION_M)


def _on_grid(data_vars, interp, res):
    """``data_vars`` as a dataset on the interpretation's axes, with the
    ``crs`` grid-mapping variable (``res``: the spacing of a one-cell axis)."""
    import xarray as xr

    ds = xr.Dataset(data_vars=data_vars,
                    coords={"t": ("t", np.asarray(interp.t, dtype="datetime64[ns]")),
                            "y": ("y", np.asarray(interp.y, dtype="float64")),
                            "x": ("x", np.asarray(interp.x, dtype="float64"))})
    ds["t"].attrs = {"standard_name": "time", "long_name": "acquisition date as labelled by openEO",
                     "axis": "T"}
    ds["y"].attrs = {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y",
                     "long_name": "y coordinate of projection (pixel centre)"}
    ds["x"].attrs = {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X",
                     "long_name": "x coordinate of projection (pixel centre)"}
    x, y = interp.x, interp.y
    dx = float(x[1] - x[0]) if len(x) > 1 else float(res)
    dy = float(y[1] - y[0]) if len(y) > 1 else -float(res)
    gt = f"{x[0] - dx / 2:.6f} {dx:.6f} 0 {y[0] - dy / 2:.6f} 0 {dy:.6f}"
    epsg = zc._epsg_from_wkt(interp.crs_wkt)
    ds["crs"] = xr.DataArray(np.int32(0), attrs={
        "crs_wkt": interp.crs_wkt, "spatial_ref": interp.crs_wkt, "GeoTransform": gt,
        "epsg_code": f"EPSG:{epsg}" if epsg else "", "long_name": "coordinate reference system"})
    return ds


def angles_float32(data, fill, attrs):
    """Delivered angle values -> float32 degrees with NaN, lazily.

    ``fill`` (the delivered no-data; NaN or None: nothing to map) becomes
    NaN, as does -32768 in integer data; CF ``scale_factor`` / ``add_offset``
    are decoded when declared. Returns ``(data, attributes to record)``."""
    native = np.dtype(data.dtype).newbyteorder("=")
    if np.dtype(data.dtype) != native:
        data = data.astype(native)
    lazy = hasattr(data, "dask")
    if lazy:
        import dask.array as xp
    else:
        xp = np
        data = np.asarray(data)
    bad = None
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        bad = data == fill
    if data.dtype.kind == "i" and np.iinfo(data.dtype).min <= zc.FILL:
        bad = (data == zc.FILL) if bad is None else (bad | (data == zc.FILL))
    notes = {}
    scale, offset = attrs.get("scale_factor"), attrs.get("add_offset")
    out = data.astype("float64") if (scale is not None or offset is not None) else data
    if scale is not None:
        out = out * float(_py(scale))
        notes["applied_scale_factor"] = float(_py(scale))
    if offset is not None:
        out = out + float(_py(offset))
        notes["applied_add_offset"] = float(_py(offset))
    out = out.astype("float32")
    if bad is not None:
        out = xp.where(bad, np.float32(np.nan), out)
    for k in SCALE_ATTRS:
        if k in attrs and k not in CF_PACKING:
            notes[f"delivered_{k}"] = _py(attrs[k])
    return out, notes


def store_fills(interp, product=None):
    """The no-data of the bands AS STORED: the delivered fills for the main
    product (values kept as delivered), NaN for the angles."""
    if product is not None and product.name == ANGLES.name:
        return {b: float("nan") for b in interp.bands}
    return interp.fills


def canonical_angles(interp, bands):
    """The angles store's dataset (lazy): float32 degrees, NaN = no data,
    CF names and units, what was delivered recorded per band."""
    data_vars = {}
    for b in bands:
        a = interp.bands[b]
        fill = interp.fills.get(b)
        long_name, std_name, desc = ANGLE_INFO.get(b, (b, "", b))
        attrs = {"long_name": long_name, "units": "degree", "grid_mapping": "crs",
                 "native_resolution_m": ANGLES_NATIVE_M, "resampling": ANGLES_RESAMPLING,
                 "description": desc, "physical_range": list(zc.ANGLE_RANGES.get(b, ())),
                 "delivered_as": interp.delivered.get(b, ""), "delivered_dtype": str(a.dtype),
                 "delivered_nodata": "none" if fill is None else repr(fill)}
        if std_name:
            attrs["standard_name"] = std_name
        zf = interp.zarr_fills.get(b)
        if zf is not None and not _same_value(zf, fill):
            attrs["delivered_zarr_fill_value"] = repr(zf)
        data, notes = angles_float32(a.data, fill, a.attrs)
        attrs.update(notes)
        data_vars[b] = (("t", "y", "x"), data, attrs)
    return _on_grid(data_vars, interp, ANGLES_RES_M)


# ─────────────────────────────────────────────────────────────────────────────
#  writing the store (block by block, every block read back)
# ─────────────────────────────────────────────────────────────────────────────

def _available_memory():
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def radiometry_counts(block, fill, low=LOW_DN):
    """Per date of a (t, y, x) spectral block: (valid pixels, valid pixels
    with 1 <= DN < ``low``). 0 is left out (the L2A products' own no-data)."""
    valid = block != 0
    if block.dtype.kind == "f":
        valid &= ~np.isnan(block)
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        valid &= block != fill
    lowv = valid & (block >= 1) & (block < low)
    return valid.sum(axis=(1, 2)), lowv.sum(axis=(1, 2))


def radiometry_summary(per_date, n_pixels, band, low=LOW_DN):
    """The offset state of a year from per-date ``[valid, low]`` counts.

    With the +1000 BOA offset in the values a DN below ``low`` would be a
    reflectance below -0.05, which does not happen: the share is ~0 on
    every date. Without it, water (B12 reflectance ~0.01) gives DN below
    ``low``: on clear dates the share is about the water fraction. The 90th
    percentile over dates with data (>= 1 % of the grid) decides:
    ``"+1000 in the values"`` below 0.2 %, ``"no offset"`` above 1 %,
    ``"unclear"`` between (e.g. a short cloudy window)."""
    shares = [lo / v for v, lo in per_date.values() if v >= max(1, 0.01 * n_pixels)]
    p90 = float(np.percentile(shares, 90)) if shares else None
    state = ("unknown (no date with data)" if p90 is None else
             "+1000 in the values" if p90 < 0.002 else "no offset" if p90 > 0.01 else "unclear")
    return {"band": band, "low_dn": low, "dates_used": len(shares),
            "p90_share_below_low": None if p90 is None else round(p90, 5),
            "offset_state": state,
            "per_date_valid_low": {d: [int(v), int(lo)] for d, (v, lo) in per_date.items()}}


def write_store(ds, fills, out, attrs, log=print, chunks=None, verify=True, groups=None,
                format_id=None, complete=True, supersede_to=None,
                radiometry_bands=RADIOMETRY_BANDS):
    """Write the canonical dataset to ``out`` (``<out>.tmp`` then rename).
    ``format_id``: the store's ``format`` attribute (the cubes'
    ``zarr_cubes.FORMAT_ID`` unless given; the angles pass theirs).
    ``complete=False``: the store is written whole but with ``complete = 0``
    and ``verification = "pending"``: the main product is marked complete
    (``mark_complete``) only once its verification against the reference
    accepts it. A store already at ``out`` is moved aside to
    ``<out>.superseded-<UTC>`` (or ``supersede_to``, a path chosen in
    advance), never deleted, and only once the new one is written whole and
    read back: until then the old store stays where it was.

    Template first (metadata and coordinates), then one region write per
    band and time block of ``CHUNKS['t']`` dates (or of the delivered time
    chunk rounded up to a multiple of it, so a delivered chunk is decoded
    once), each read back and compared bit for bit with the delivered
    block. ``groups``: lists of bands that share delivered chunks (a
    ``bands`` dimension chunked across bands), computed together so each
    delivered chunk is decoded once per block. A block that would not fit
    in the available memory stops the write before anything is read. The
    per-date share of low B12 (else B11) digital numbers is computed on the
    way and stored as the ``radiometry`` attribute (see
    ``radiometry_summary``; ``radiometry_bands``: the bands tried, in order;
    a band subset passes SUBSET_RADIOMETRY_BANDS). ``complete = 1`` and the
    consolidated metadata are written last, then the folder is renamed: a
    store under its final name is always whole.
    """
    import dask.array as da
    import xarray as xr
    import zarr

    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    bands = zc.band_names(ds)
    groups = [[b for b in g if b in bands] for g in (groups or [[b] for b in bands])]
    groups = [g for g in groups if g]
    groups += [[b] for b in bands if not any(b in g for g in groups)]
    T, H, W = ds.sizes["t"], ds.sizes["y"], ds.sizes["x"]
    ct, cy, cx = zc.chunk_shape((T, H, W), chunks)
    tmpl = xr.Dataset(
        {b: (("t", "y", "x"), da.empty((T, H, W), dtype=ds[b].dtype, chunks=(ct, cy, cx)),
             ds[b].attrs) for b in bands},
        coords={k: ds[k] for k in ("t", "y", "x")})
    tmpl["crs"] = ds["crs"]
    tmpl.attrs = dict(attrs, complete=0, format=format_id or zc.FORMAT_ID)
    enc = zc.encoding(tmpl, bands, fills, chunks)
    dates = zc.dates(ds)
    src_ct = max([int(ds[b].data.chunksize[0]) for b in bands
                  if getattr(ds[b].data, "chunksize", None)] or [1])
    bt = min(T, ct * max(1, -(-src_ct // ct)))
    item = max(np.dtype(ds[b].dtype).itemsize for b in bands)
    widest = max(len(g) for g in groups)

    def need(nt):                 # a block of the widest group, plus the read-back of one band
        return nt * H * W * item * (widest + 1)

    avail = _available_memory()
    if bt > ct and avail is not None and need(bt) > 0.5 * avail:
        log(f"the delivered data come in chunks of {src_ct} dates, but blocks of {bt} dates would "
            f"need ~{need(bt) / 1e9:.1f} GB of the {avail / 1e9:.1f} GB available: blocks of {ct} "
            f"dates instead (each delivered chunk is decoded up to {-(-bt // ct)} times: slower, "
            f"same values)")
        bt = ct
    elif bt > ct:
        log(f"the delivered data come in chunks of {src_ct} dates: blocks of {bt} dates "
            f"(~{bt * H * W * item / 1e9:.1f} GB per band)")
    if avail is not None and need(bt) > 0.8 * avail:
        raise MemoryError(f"writing {out.name} needs ~{need(bt) / 1e9:.1f} GB per block ({bt} dates "
                          f"x {H} x {W} px x {widest} band(s) decoded together, plus the read-back) "
                          f"and {avail / 1e9:.1f} GB are available: free memory first")
    tmpl.to_zarr(tmp, mode="w", compute=False, encoding=enc, zarr_format=2, consolidated=False)
    rad_band = next((b for b in radiometry_bands if b in bands), None)
    rad = {}
    nblk = (T + bt - 1) // bt
    t0 = time.time()
    for k, a in enumerate(range(0, T, bt)):
        b_ = min(T, a + bt)
        nbytes = 0
        for group in groups:
            blk = ds[group].isel(t=slice(a, b_)).compute()      # shared chunks decoded once
            for b in group:
                block = np.asarray(blk[b].values)
                native = block.dtype.newbyteorder("=")
                if block.dtype != native:
                    block = block.astype(native)
                nbytes += block.nbytes
                part = xr.Dataset({b: (("t", "y", "x"), da.from_array(block, chunks=(ct, cy, cx)))})
                part.to_zarr(tmp, region={"t": slice(a, b_), "y": slice(0, H), "x": slice(0, W)},
                             mode="r+", consolidated=False)
                if verify:
                    back = zarr.open_array(store=str(tmp / b), mode="r")[a:b_]
                    same = (np.array_equal(back, block, equal_nan=True) if block.dtype.kind == "f"
                            else np.array_equal(back, block))
                    del back
                    if not same:
                        raise RuntimeError(f"{b}, dates {dates[a]}..{dates[b_ - 1]}: the block read "
                                           f"back from {tmp} differs from the delivered one")
                if b == rad_band:
                    nv, nl = radiometry_counts(block, fills.get(b))
                    for j in range(len(nv)):
                        rad[dates[a + j]] = (int(nv[j]), int(nl[j]))
                del block
            del blk
        log(f"block {k + 1}/{nblk}: {dates[a]}..{dates[b_ - 1]} ({nbytes / 1e6:.0f} MB) "
            f"written{' and read back identical' if verify else ''} "
            f"[{(time.time() - t0) / 60:.1f} min]")
    g = zarr.open_group(str(tmp), mode="r+")
    if rad_band is not None:
        summ = radiometry_summary(rad, H * W, rad_band)
        g.attrs["radiometry"] = json.dumps(summ)
        log(f"radiometry: {rad_band} share of valid DN in [1, {LOW_DN}) per date, p90 "
            f"{summ['p90_share_below_low']} over {summ['dates_used']} dates -> "
            f"{summ['offset_state']}")
    if complete:
        g.attrs["complete"] = 1
    else:
        g.attrs["verification"] = PENDING
    zarr.consolidate_metadata(str(tmp))
    chk = zc.open_store(tmp)
    if chk.sizes["t"] != T or zc.dates(chk) != dates or sorted(zc.band_names(chk)) != sorted(bands):
        raise RuntimeError(f"{tmp}: the written store does not reopen as written")
    chk.close()
    if out.exists():
        move_aside(out, "a store was already there (written again)", log, dest=supersede_to)
    os.replace(tmp, out)
    return out


#: the ``verification`` attribute of a main store written but not yet accepted
PENDING = ("pending: written whole, marked complete once its verification against the "
           "reference accepts it (verification_decision)")


def superseded_name(path):
    """A free ``<path>.superseded-<UTC>`` name beside ``path``."""
    path = Path(path)
    stamp = f"{path.name}.superseded-{utcnow():%Y%m%dT%H%M%S}"
    dest, k = path.with_name(stamp), 1
    while dest.exists():
        dest, k = path.with_name(f"{stamp}-{k}"), k + 1
    return dest


def move_aside(path, why, log=print, dest=None):
    """Move ``path`` (a store) to ``dest`` (a free path chosen in advance) or
    ``<path>.superseded-<UTC>``: never deleted. Returns the new path."""
    path = Path(path)
    dest = Path(dest) if dest is not None and not Path(dest).exists() else superseded_name(path)
    os.replace(path, dest)
    log(f"{why}: {path} moved aside to {dest} (kept, never deleted)")
    return dest


def mark_complete(store, attrs=None):
    """Set ``complete = 1`` (and ``attrs``) on a store written with
    ``complete=False``, and consolidate its metadata again."""
    import zarr

    g = zarr.open_group(str(store), mode="r+")
    for k, v in (attrs or {}).items():
        g.attrs[k] = v
    g.attrs["complete"] = 1
    zarr.consolidate_metadata(str(store))
    return Path(store)


# ─────────────────────────────────────────────────────────────────────────────
#  the format report
# ─────────────────────────────────────────────────────────────────────────────

def _short(v, n=300):
    s = json.dumps(v, default=str, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"


OUTPUT_FORMAT = "output_format.json"


def describe_output_format(conn, fmt, part, log):
    """What the backend says about the output format (GET /file_formats):
    title, flags, parameters; saved as ``<part>/output_format.json`` for the
    FORMAT REPORT. Never fatal."""
    try:
        out = (conn.list_file_formats() or {}).get("output", {})
    except Exception as exc:
        log(f"could not list the backend's file formats ({type(exc).__name__}: {exc})")
        return None
    key = next((k for k in out if k.lower() == str(fmt).lower()), None)
    if key is None:
        log(f"the backend does not list {fmt!r} among its output formats {sorted(out)}")
        return None
    desc = dict(out[key], name=key)
    zc.write_json(desc, Path(part) / OUTPUT_FORMAT)
    for ln in report_output_format(desc):
        log(ln)
    return desc


def report_output_format(desc):
    if not desc:
        return []
    flags = {k: desc[k] for k in ("experimental", "deprecated") if k in desc}
    L = [f"backend output format {desc.get('name')!r}: title {desc.get('title')!r}, gis_data_types "
         f"{desc.get('gis_data_types')}" + (f", {flags}" if flags else "")]
    if desc.get("description"):
        L.append(f"  description: {_short(desc['description'], 400)}")
    params = desc.get("parameters") or {}
    L.append(f"  save_result options: {len(params)}" + (" (pass with --format-options JSON)"
                                                        if params else ""))
    for name, spec in params.items():
        spec = spec if isinstance(spec, dict) else {"description": spec}
        L.append(f"    {name}: type {spec.get('type')}, default {spec.get('default')!r}; "
                 f"{_short(str(spec.get('description', '')), 240)}")
    return L


def report_job(info, meta, files):
    L = []
    if info:
        L.append(f"job: {info.get('id')} status {info.get('status')} created {info.get('created')} "
                 f"updated {info.get('updated')}")
        if info.get("costs") is not None or info.get("usage"):
            L.append(f"  costs {info.get('costs')} usage {_short(info.get('usage'), 600)}")
    if meta:
        L.append(f"results metadata: type {meta.get('type')} stac_version "
                 f"{meta.get('stac_version')} keys {sorted(meta)}")
        links = meta.get("links") or []
        rels = {}
        for lk in links:
            rels[lk.get("rel")] = rels.get(lk.get("rel"), 0) + 1
        L.append(f"  links by rel: {rels}")
        assets = meta.get("assets") or {}
        L.append(f"  assets listed: {len(assets)}")
        for k, (name, a) in enumerate(sorted(assets.items())):
            if k >= 40:
                L.append(f"  ... {len(assets) - 40} more")
                break
            extra = {kk: a[kk] for kk in a if kk.startswith(("proj:", "file:", "eo:", "raster:"))
                     or kk in ("bands", "roles", "type", "title")}
            host = re.sub(r"^(https?://[^/]+).*$", r"\1", str(a.get("href", "")))
            L.append(f"  asset {name!r}: {_short(extra, 500)} href {host}/...")
    if files:
        tot = sum(v.get("size", 0) for v in files.values())
        L.append(f"downloaded: {len(files)} file(s), {tot / 1e6:.1f} MB")
        for k, (rel, v) in enumerate(sorted(files.items())):
            if k >= 20:
                L.append(f"  ... {len(files) - 20} more")
                break
            L.append(f"  {rel}  {v.get('size', 0) / 1e6:.2f} MB  type {v.get('type')}")
    return L


def report_tree(path, max_lines=30):
    """Directory layout: folders with their file counts and sizes."""
    L, rows = [], []
    path = Path(path)
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        size = sum((Path(dirpath) / f).stat().st_size for f in filenames)
        rel = Path(dirpath).relative_to(path)
        sample = sorted(filenames)[:6]
        rows.append(f"  {str(rel) if str(rel) != '.' else '.'}/  {len(filenames)} files "
                    f"{size / 1e6:.2f} MB  e.g. {sample}")
    L.extend(rows[:max_lines])
    if len(rows) > max_lines:
        L.append(f"  ... {len(rows) - max_lines} more folders")
    return L


def report_raw(raw):
    L = [f"store {raw.source}: kind {raw.kind}, Zarr format {raw.zarr_format}"]
    if raw.attrs:
        L.append(f"  root attributes: {_short(raw.attrs, 800)}")
    for n, i in sorted(raw.info.items()):
        L.append(f"  {n:<14} shape {i['shape']} {i['dtype']} chunks {i.get('chunks')}"
                 + (f" shards {i['shards']}" if i.get("shards") else "")
                 + f" fill {i.get('fill_value')!r} dims {i.get('dims')} ({i.get('dims_source')})")
        L.append(f"  {'':<14} codecs {i.get('codecs')}; attrs {_short(i.get('attrs'), 400)}")
    for n in raw.notes:
        L.append(f"  note: {n}")
    return L


def report_raw_metadata(path, max_arrays=40):
    """The Zarr metadata files of a store, read as JSON (no zarr-python):
    what can still be said about a store zarr cannot open (an unknown
    codec, a data type it does not support)."""
    path = Path(path)
    L = [f"  raw Zarr metadata under {path}:"]
    n = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        rel = Path(dirpath).relative_to(path).as_posix()
        for fn in sorted(set(filenames) & {".zgroup", ".zarray", ".zattrs", "zarr.json"}):
            try:
                md = json.loads((Path(dirpath) / fn).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                L.append(f"    {rel}/{fn}: unreadable ({type(exc).__name__}: {exc})")
                continue
            if fn == ".zarray" or md.get("node_type") == "array":
                n += 1
                if n > max_arrays:
                    continue
                keep = {k: md.get(k) for k in ("shape", "chunks", "dtype", "data_type", "fill_value",
                                               "compressor", "filters", "codecs", "order",
                                               "dimension_names", "chunk_grid", "attributes")
                        if k in md}
                L.append(f"    array {rel}: {_short(keep, 700)}")
            elif fn == ".zattrs" or md.get("node_type") == "group":
                L.append(f"    {rel}/{fn}: {_short(md, 500)}")
    if n > max_arrays:
        L.append(f"    ... {n - max_arrays} more arrays")
    return L


def report_dates(ds, fills, band=None, candidates=REPORT_CANDIDATES, n=REPORT_DATES):
    """The dates the value and resampling reports use: among up to
    ``candidates`` dates spread over the axis, the ``n`` with most valid
    pixels of ``band`` (B03 by default). Returns ``(indices, line)``."""
    T = ds.sizes["t"]
    if not T:
        return [], "no dates"
    names = zc.band_names(ds)
    band = band or ("B03" if "B03" in names else names[0])
    cand = np.unique(np.linspace(0, T - 1, min(T, candidates)).round().astype(int))
    a = np.asarray(ds[band].isel(t=list(cand)).values)
    valid = _valid(a, fills.get(band))
    counts = valid.reshape(len(cand), -1).sum(axis=1)
    order = np.argsort(-counts, kind="stable")[:n]
    chosen = sorted(int(cand[i]) for i in order if counts[i] > 0) or [int(cand[order[0]])]
    dts = zc.dates(ds)
    npx = a[0].size
    line = (f"dates used: {[dts[i] for i in chosen]} (the {len(chosen)} with most valid {band} "
            f"pixels among {len(cand)} of {T}; valid share "
            + ", ".join(f"{dts[int(c)]} {k / npx * 100:.0f} %" for c, k in zip(cand, counts))[:900]
            + ")")
    return chosen, line


def _valid(a, fill):
    valid = ~np.isnan(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        valid &= a != fill
    return valid


def input_products(meta):
    """``{date: [processing baselines]}`` of the Sentinel-2 L2A products a
    job read, from any product name in its results metadata (CDSE lists
    them as ``derived_from`` links)."""
    blob = json.dumps(meta or {}, default=str)
    pat = r"S2[A-D]_MSIL2A_(\d{8})T\d{6}_N(\d{2})(\d{2})_R\d{3}_T\w{5}_\d{8}T\d{6}"
    out = {}
    for m in re.finditer(pat, blob):
        d = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
        out.setdefault(d, set()).add(f"{m.group(2)}.{m.group(3)}")
    return {d: sorted(v) for d, v in sorted(out.items())}


def report_xarray_direct(path):
    import warnings
    import xarray as xr

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_zarr(str(path), consolidated=None, mask_and_scale=False)
        msg = f"xarray.open_zarr opens the delivered store directly: dims {dict(ds.sizes)}, " \
              f"variables {list(ds.data_vars)}"
        ds.close()
        return [msg]
    except Exception as exc:
        return [f"xarray.open_zarr cannot open the delivered store as is "
                f"({type(exc).__name__}: {str(exc)[:200]}); read with zarr-python instead"]


def report_interp(interp, ds):
    import pandas as pd

    t = pd.DatetimeIndex(interp.t)
    L = ["interpretation:"]
    L.append(f"  time: {len(t)} dates {t.min().date() if len(t) else ''} .. "
             f"{t.max().date() if len(t) else ''}; times of day {sorted(set(t.strftime('%H:%M:%S')))}")
    if len(t) <= 40:
        L.append(f"  dates: {[str(d.date()) for d in t]}")
    x, y = interp.x, interp.y
    L.append(f"  grid: {len(y)} rows x {len(x)} cols; x {x[0]:.1f}..{x[-1]:.1f}, y {y[0]:.1f}.."
             f"{y[-1]:.1f} (pixel centres), spacing {x[1] - x[0] if len(x) > 1 else '?'} / "
             f"{y[1] - y[0] if len(y) > 1 else '?'} m")
    L.append(f"  CRS: EPSG {zc._epsg_from_wkt(interp.crs_wkt)} from {interp.crs_source}")
    reg = interp.registration
    if reg:
        o = reg.get("offset")
        L.append(f"  registration: {reg.get('method')}: "
                 + (f"frame {reg['frame_shape'][0]} x {reg['frame_shape'][1]} px at row {o[0]}, col "
                    f"{o[1]} of the delivered {reg['delivered_shape'][0]} x "
                    f"{reg['delivered_shape'][1]} px (their x/y: row {reg['coordinate_offset'][0]}, "
                    f"col {reg['coordinate_offset'][1]}); " if o is not None else "")
                 + str(reg.get("why") or ""))
    for b in ds.data_vars:
        if b == "crs":
            continue
        L.append(f"  {b:<4} <- {interp.delivered.get(b)}: dtype {ds[b].attrs.get('delivered_dtype')}, "
                 f"fill {interp.fills.get(b)!r} ({interp.fill_sources.get(b)})")
    if interp.nodata_lines:
        L.append("  no-data:")
        L += ["  " + ln for ln in interp.nodata_lines]
    for n in interp.notes:
        if n:
            L.append(f"  note: {n}")
    for w in interp.warnings:
        L.append(f"  WARNING: {w}")
    return L


def report_summary(files, found, raws, interp, product=None):
    """The answers in a few lines: assets and layout, variables and dims,
    dtypes, fills, scale attributes, chunks, CRS, time, band naming."""
    angles = product is not None and product.name == ANGLES.name
    import pandas as pd

    types = sorted({str(v.get("type")) for v in files.values()})
    zips = [k for k, v in files.items() if k.lower().endswith(".zip") or "zip" in str(v.get("type"))]
    L = ["SUMMARY",
         f"  assets: {len(files)} file(s), types {types}; zipped: {'yes (' + ', '.join(zips[:3]) + ')' if zips else 'no'}",
         f"  layout: {len(found['stores'])} Zarr store(s) "
         f"{sorted({str(r.zarr_format) for r in raws if r.kind != 'netcdf'})} "
         f"[{', '.join(sorted({s['kind'] for s in found['stores']}))}], {len(found['netcdf'])} netCDF; "
         + next((n for n in interp.notes if n.startswith("layout:")), "layout: ?")]
    by_var = {}
    for r in raws:
        for name, i in r.info.items():
            by_var.setdefault(name, i)
    held = []
    for b, d in interp.delivered.items():
        info = by_var.get(d.split("[")[0]) or by_var.get(d.split(":")[-1].split("[")[0], {})
        held.append((b, d, info))
    dims = sorted({tuple(i.get("dims") or ()) for _, _, i in held})
    dtypes = sorted({i.get("dtype") for _, _, i in held if i.get("dtype")})
    chunks = sorted({str(i.get("chunks")) for _, _, i in held})
    codecs = sorted({str(i.get("codecs")) for _, _, i in held})
    L.append(f"  band naming: {', '.join(f'{b}<-{d}' for b, d, _ in held)}")
    L.append(f"  dims as delivered: {dims}; dtypes {dtypes}; chunks {chunks}")
    L.append(f"  codecs: {codecs[:2]}{' ...' if len(codecs) > 2 else ''}")
    fills = sorted({f"{interp.fills[b]!r} ({interp.fill_sources[b]})" for b in interp.fills})
    L.append(f"  no-data: {fills}")
    scale = {b: {k: i.get('attrs', {}).get(k) for k in SCALE_ATTRS if k in i.get('attrs', {})}
             for b, _, i in held}
    scale = {b: v for b, v in scale.items() if v}
    L.append(f"  scale/offset attributes: {scale if scale else 'none'} "
             + ("(angles: CF scale_factor/add_offset decoded, others never applied; stored as "
                "float32 degrees, no-data -> NaN)" if angles else "(never applied)"))
    x, y = interp.x, interp.y
    L.append(f"  grid: EPSG {zc._epsg_from_wkt(interp.crs_wkt)} from {interp.crs_source}; "
             f"{len(y)} x {len(x)} {'cells' if angles else 'px'}, spacing "
             f"{x[1] - x[0] if len(x) > 1 else '?'} / "
             f"{y[1] - y[0] if len(y) > 1 else '?'} m, first centre ({x[0]}, {y[0]})")
    tinfo = next((r.info.get(n) for r in raws for n in ("t", "time", "times", "date", "dates")
                  if n in r.info), {}) or {}
    t = pd.DatetimeIndex(interp.t)
    L.append(f"  time: {len(t)} labels, stored as {tinfo.get('dtype')} "
             f"{tinfo.get('attrs', {}).get('units', '')!r}; times of day "
             f"{sorted(set(t.strftime('%H:%M:%S')))[:5]}")
    if interp.warnings:
        L.append(f"  warnings: {len(interp.warnings)} (above)")
    return L


def block_equal_counts(a, x, y, valid, cell=20.0):
    """``(equal, pairs)``: adjacent pixel pairs inside one ``cell`` m cell of
    the UTM lattice, both valid, and how many of them hold the same value
    (all of them when a 20 m band reached 10 m by nearest neighbour from its
    native grid; well below when interpolated). ``a``: (y, x) or (t, y, x)."""
    a, valid = np.asarray(a), np.asarray(valid)
    if a.ndim == 2:
        a, valid = a[None], valid[None]
    ix = np.floor(np.asarray(x) / cell)
    iy = np.floor(np.asarray(y) / cell)
    sc, sr = ix[1:] == ix[:-1], iy[1:] == iy[:-1]
    eq_c = (a[:, :, 1:] == a[:, :, :-1])[:, :, sc]
    ok_c = (valid[:, :, 1:] & valid[:, :, :-1])[:, :, sc]
    eq_r = (a[:, 1:, :] == a[:, :-1, :])[:, sr, :]
    ok_r = (valid[:, 1:, :] & valid[:, :-1, :])[:, sr, :]
    return int(eq_c[ok_c].sum() + eq_r[ok_r].sum()), int(ok_c.sum() + ok_r.sum())


def block_equal_share(a, x, y, valid, cell=20.0):
    """Share of :func:`block_equal_counts` (NaN without valid pairs)."""
    eq, n = block_equal_counts(a, x, y, valid, cell)
    return eq / n if n else float("nan")


def report_resampling(ds, fills, date_indices=(0,), dates_line=None):
    """How the 20 m bands reached the 10 m grid, pooled over the dates with
    most valid pixels (``report_dates``), with the number of pairs."""
    x, y = ds["x"].values, ds["y"].values
    idx = list(date_indices)
    res = {}
    for b in zc.band_names(ds):
        a = np.asarray(ds[b].isel(t=idx).values)
        res[b] = block_equal_counts(a, x, y, _valid(a, fills.get(b)))
    fmt = ", ".join(f"{b} {(eq / n * 100 if n else float('nan')):.1f} % (n={n})"
                    for b, (eq, n) in res.items())
    dts = zc.dates(ds)
    return [f"resampling check on {[dts[i] for i in idx]}: share of valid pixel pairs inside one "
            f"20 m cell of the UTM lattice with equal values: {fmt}",
            "  (100 % for B8A B11 B12 SCL CLD = nearest neighbour from the native 20 m grid; the "
            "10 m bands give the share expected by chance; n = valid pairs, n=0 = no data)"]


def report_values(ds, fills, date_indices=(0,), dates_line=None):
    """min / max / percentiles / no-data / zeros of every band, pooled over
    the dates with most valid pixels, and the low-DN share of B12 / B11."""
    idx = list(date_indices)
    dts = zc.dates(ds)
    L = [f"values on {[dts[i] for i in idx]}:"]
    if dates_line:
        L.append(f"  {dates_line}")
    for b in zc.band_names(ds):
        a = np.asarray(ds[b].isel(t=idx).values)
        nod = ~_valid(a, fills.get(b))
        v = a[~nod]
        pct = (np.percentile(v, [1, 50, 99]).round(1).tolist() if v.size else "-")
        low = (f"; DN in [1, {LOW_DN}): {((v >= 1) & (v < LOW_DN)).mean() * 100:.1f} % of valid"
               if b in RADIOMETRY_BANDS and v.size else "")
        L.append(f"  {b:<4} min {v.min() if v.size else '-'} max {v.max() if v.size else '-'} "
                 f"p1/p50/p99 {pct} no-data {nod.mean() * 100:.1f} % zeros "
                 f"{(a == 0).mean() * 100:.1f} % (-32768: {(a == -32768).sum()} px){low}")
    spectral = [b for b in zc.band_names(ds) if b not in ("SCL", "CLD")]
    if spectral:
        L.append("  (spectral bands: L2A digital numbers as delivered. Since processing baseline "
                 "04.00 (2022-01-25) the L2A products carry BOA_ADD_OFFSET = -1000: if p1 of "
                 f"B11/B12 over water sits near 1000 (and no valid DN is below {LOW_DN}) the "
                 "offset is still in the values; near 0, the backend removed it or the products "
                 "predate 04.00. Every store records this per date: attribute 'radiometry')")
    return L


# ─────────────────────────────────────────────────────────────────────────────
#  the angles: values, frame, physical ranges, spread, frame centre
# ─────────────────────────────────────────────────────────────────────────────

def report_angle_values(ds, date_indices=(0,), dates_line=None):
    """min / max / percentiles / no-data of every angle (as stored: float32
    degrees, NaN), pooled over the dates with most valid cells."""
    idx = list(date_indices)
    dts = zc.dates(ds)
    L = [f"values on {[dts[i] for i in idx]} (float32 degrees as stored, NaN = no data):"]
    if dates_line:
        L.append(f"  {dates_line}")
    for b in zc.band_names(ds):
        a = np.asarray(ds[b].isel(t=idx).values, dtype="float64")
        v = a[np.isfinite(a)]
        pct = np.percentile(v, [1, 50, 99]).round(4).tolist() if v.size else "-"
        L.append(f"  {b:<16} min {f'{v.min():.4f}' if v.size else '-'} max "
                 f"{f'{v.max():.4f}' if v.size else '-'} p1/p50/p99 {pct} no-data "
                 f"{(~np.isfinite(a)).mean() * 100 if a.size else 0:.1f} %")
    return L


def _great_circle_deg(lon1, lat1, lon2, lat2):
    """Angular distance (degrees) between two points on the sphere."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2.0) ** 2)
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(h))))


def frame_geometry(ds):
    """The frame of a (t, y, x) dataset: cell-edge extent, centre (x/y and
    lon/lat), corners in lon/lat, the diagonal as an arc (degrees), the
    longitude span and the largest |latitude|. Without a readable CRS the
    arc comes from the projected diagonal and the lon/lat parts are None."""
    x = np.asarray(ds["x"].values, dtype="float64")
    y = np.asarray(ds["y"].values, dtype="float64")
    cx = cy = None
    gt = ds["crs"].attrs.get("GeoTransform") if "crs" in ds.variables else None
    if gt:
        try:
            vals = [float(v) for v in str(gt).split()]
            cx, cy = abs(vals[1]), abs(vals[5])
        except (ValueError, IndexError):
            pass
    dx = abs(float(x[1] - x[0])) if len(x) > 1 else float(cx or 0.0)
    dy = abs(float(y[1] - y[0])) if len(y) > 1 else float(cy or 0.0)
    xe = (float(x[0]) - dx / 2.0, float(x[-1]) + dx / 2.0)
    ye = (float(y[-1]) - dy / 2.0, float(y[0]) + dy / 2.0)
    xc, yc = (xe[0] + xe[1]) / 2.0, (ye[0] + ye[1]) / 2.0
    geo = {"cells": [int(len(y)), int(len(x))], "cell_m": [dx, dy], "x_edges": list(xe),
           "y_edges": list(ye), "centre_xy": [xc, yc],
           "diagonal_m": float(math.hypot(xe[1] - xe[0], ye[1] - ye[0])),
           "centre_lonlat": None, "corners_lonlat": None, "lon_span_deg": None,
           "lat_max_abs": None}
    geo["diagonal_deg"] = geo["diagonal_m"] / 111195.0
    wkt = zc.grid_info(ds)["crs_wkt"]
    try:
        from pyproj import CRS, Transformer

        tf = Transformer.from_crs(CRS.from_user_input(wkt), "EPSG:4326", always_xy=True)
        corners = [tf.transform(px, py) for px, py in ((xe[0], ye[1]), (xe[1], ye[1]),
                                                       (xe[1], ye[0]), (xe[0], ye[0]))]
        lonc, latc = tf.transform(xc, yc)
        lons, lats = [c[0] for c in corners], [c[1] for c in corners]
        geo.update(centre_lonlat=[round(lonc, 6), round(latc, 6)],
                   corners_lonlat=[[round(lo, 6), round(la, 6)] for lo, la in corners],
                   lon_span_deg=max(lons) - min(lons), lat_max_abs=max(abs(v) for v in lats),
                   diagonal_deg=max(_great_circle_deg(*corners[0], *corners[2]),
                                    _great_circle_deg(*corners[1], *corners[3])))
    except Exception as exc:                  # informative: the bound falls back
        geo["crs_note"] = f"frame not in lon/lat ({type(exc).__name__}: {exc})"
    return geo


def _per_date(fn, a):
    """``fn`` (nanmin, nanmax, nanmedian) over each date of (t, y, x), NaN
    for a date without data, without the all-NaN warnings."""
    import warnings

    if not a.shape[0]:
        return np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a.reshape(a.shape[0], -1), axis=1)


def angle_checks(ds, n_centre=CENTRE_DATES):
    """The checks of an angles dataset (the normalised delivery, or a store).

    * ranges: per angle min / max, finite share and the values outside the
      physical range (zenith 0-90, azimuth 0-360 degrees);
    * spread: per date, max - min over the frame (azimuths around their
      median, so 359/1 is a spread of 2), summarised over the dates; the sun
      angles are held against a geometric bound: the sun zeniths of two
      points differ by at most their angular distance (triangle inequality
      on the sphere), so zenith spread <= diagonal + ``SPREAD_TOL_ZENITH``;
      the azimuths by about distance x cot(zenith) plus the convergence of
      the meridians, bounded by diagonal / sin(min zenith) + longitude span
      x sin(max |latitude|) + ``SPREAD_TOL_AZIMUTH``. A date above the
      bound mixes datatakes or carries a bad angle grid;
    * the dates without sun angles, and those with no-data cells in the frame;
    * the four angles at the frame centre on every date (bilinear,
      ``zarr_cubes.sample_point``).
    """
    names = [b for b in zc.ANGLE_BANDS if b in ds.data_vars]
    dts = zc.dates(ds)
    T = len(dts)
    geo = frame_geometry(ds)
    out = {"bands": names, "n_dates": T, "frame": geo, "ranges": {}, "spread": {}}
    stats = {}
    for b in names:
        a = np.array(ds[b].values, dtype="float64")         # a site-year: a few MB per angle
        f = zc.fill_of(ds[b])
        if f is not None:
            a[a == f] = np.nan
        fin = np.isfinite(a)
        lo, hi = zc.ANGLE_RANGES.get(b, (-np.inf, np.inf))
        with np.errstate(invalid="ignore"):
            oor = fin & ((a < lo - RANGE_EPS) | (a > hi + RANGE_EPS))
        v = a[fin]
        bad_dates = np.flatnonzero(oor.reshape(T, -1).any(axis=1)) if T else []
        out["ranges"][b] = {
            "range": [lo, hi], "min": float(v.min()) if v.size else None,
            "max": float(v.max()) if v.size else None,
            "finite_share": float(fin.mean()) if a.size else 0.0, "zeros": int((v == 0.0).sum()),
            "out_of_range": int(oor.sum()), "out_of_range_dates": [dts[i] for i in bad_dates][:20]}
        anyf = fin.reshape(T, -1).any(axis=1) if T else np.zeros(0, bool)
        allf = fin.reshape(T, -1).all(axis=1) if T else np.zeros(0, bool)
        if b in zc.AZIMUTH_BANDS:
            med = _per_date(np.nanmedian, a)
            d = np.mod(a - med[:, None, None] + 180.0, 360.0) - 180.0
            spread = _per_date(np.nanmax, d) - _per_date(np.nanmin, d)
        else:
            spread = _per_date(np.nanmax, a) - _per_date(np.nanmin, a)
        spread = np.where(anyf, spread, np.nan)
        stats[b] = {"spread": spread, "min": _per_date(np.nanmin, a), "any": anyf, "all": allf}
        s = spread[np.isfinite(spread)]
        out["spread"][b] = {"max": float(s.max()) if s.size else None,
                            "median": float(np.median(s)) if s.size else None,
                            "p95": float(np.percentile(s, 95)) if s.size else None}
        del a, fin, oor
    diag = float(geo["diagonal_deg"])
    if geo.get("lon_span_deg") is not None and geo.get("lat_max_abs") is not None:
        lon_term = geo["lon_span_deg"] * math.sin(math.radians(geo["lat_max_abs"]))
    else:
        lon_term = 2.0 * diag
    zen, azi = stats.get("sunZenithAngles"), stats.get("sunAzimuthAngles")
    over, without, nan_cells = {}, [], []
    for i, d in enumerate(dts):
        row = {}
        bz = diag + SPREAD_TOL_ZENITH
        if zen is not None and np.isfinite(zen["spread"][i]) and zen["spread"][i] > bz:
            row["sunZenithAngles"] = [round(float(zen["spread"][i]), 4), round(bz, 4)]
        zmin = float(zen["min"][i]) if zen is not None and np.isfinite(zen["min"][i]) else 0.0
        ba = diag / math.sin(math.radians(max(zmin, 5.0))) + lon_term + SPREAD_TOL_AZIMUTH
        if azi is not None and np.isfinite(azi["spread"][i]) and azi["spread"][i] > ba:
            row["sunAzimuthAngles"] = [round(float(azi["spread"][i]), 4), round(ba, 4)]
        if row:
            over[d] = row
        if any(s is not None and not s["any"][i] for s in (zen, azi)):
            without.append(d)
        elif any(s is not None and not s["all"][i] for s in (zen, azi)):
            nan_cells.append(d)
    out["spread_bound"] = {
        "diagonal_deg": round(diag, 5), "lon_term_deg": round(lon_term, 5),
        "zenith": f"spread <= {diag:.4f} (frame diagonal) + {SPREAD_TOL_ZENITH} deg",
        "azimuth": (f"spread <= {diag:.4f} / sin(min sun zenith of the date) + {lon_term:.4f} "
                    f"(meridian convergence) + {SPREAD_TOL_AZIMUTH} deg")}
    out["dates_over_bound"] = over
    out["dates_without_sun"] = without
    out["dates_with_nan_cells"] = nan_cells
    xc, yc = geo["centre_xy"]
    centre = {"x": xc, "y": yc, "lon": (geo["centre_lonlat"] or [None, None])[0],
              "lat": (geo["centre_lonlat"] or [None, None])[1], "per_date": {}}
    if names and T:
        df = zc.sample_point(ds, xc, yc, bands=names)
        for d, row in zip(dts, df.itertuples(index=False)):
            centre["per_date"][d] = {b: (None if not np.isfinite(v) else round(float(v), 5))
                                     for b, v in zip(names, row)}
    out["centre"] = centre
    return out


def sun_out_of_range(chk):
    """The sun angles of an ``angle_checks`` result outside their physical
    range: ``["sunZenithAngles: 3 values ..."]`` (empty when none)."""
    bad = []
    for b in ("sunZenithAngles", "sunAzimuthAngles"):
        r = chk["ranges"].get(b)
        if r and r["out_of_range"]:
            bad.append(f"{b}: {r['out_of_range']} values outside {r['range']} (min {r['min']}, max "
                       f"{r['max']}; dates {r['out_of_range_dates'][:5]})")
    return bad


def sun_zeros(chk):
    """Sun angles exactly 0 in an ``angle_checks`` result: impossible at
    these sites at the overpass (zenith > 15 deg, azimuth 100-260 deg), so a
    no-data written as 0 without being declared. Kept as delivered and
    reported: ``["sunZenithAngles: 12 cells exactly 0"]``."""
    return [f"{b}: {chk['ranges'][b]['zeros']} cells exactly 0"
            for b in ("sunZenithAngles", "sunAzimuthAngles")
            if chk["ranges"].get(b, {}).get("zeros")]


def _fmt_deg(v, width=9):
    return f"{v:{width}.4f}" if v is not None else f"{'-':>{width}}"


def report_angle_checks(chk, n=CENTRE_DATES):
    """Log lines of an ``angle_checks`` result: ranges, frame, spread and its
    bound, the dates over it, and the angles at the frame centre for ``n``
    dates spread over the window."""
    L = ["angles check:"]
    for b, r in chk["ranges"].items():
        L.append(f"  {b:<16} {_fmt_deg(r['min'], 0).strip()} .. {_fmt_deg(r['max'], 0).strip()} "
                 f"deg, finite {r['finite_share'] * 100:.1f} %, outside [{r['range'][0]:g}, "
                 f"{r['range'][1]:g}]: {r['out_of_range']}"
                 + (f" (dates {r['out_of_range_dates'][:5]})" if r["out_of_range"] else "")
                 + (f"; exactly 0: {r['zeros']}" if r.get("zeros") else ""))
    g = chk["frame"]
    L.append(f"  frame: {g['cells'][0]} x {g['cells'][1]} cells of {g['cell_m'][0]:g} m, diagonal "
             f"{g['diagonal_m'] / 1000:.2f} km = {g['diagonal_deg']:.4f} deg; centre lon/lat "
             f"{g['centre_lonlat']}" + (f" ({g['crs_note']})" if g.get("crs_note") else ""))
    sp = chk["spread"]
    L.append("  spread inside the frame per date, max / median over the dates: "
             + ", ".join(f"{b} {_fmt_deg(s['max'], 0).strip()} / {_fmt_deg(s['median'], 0).strip()}"
                         for b, s in sp.items()) + " deg")
    bnd = chk["spread_bound"]
    L.append(f"  bound of the sun angles: zenith {bnd['zenith']}; azimuth {bnd['azimuth']}")
    over = chk["dates_over_bound"]
    L.append(f"  dates over the bound: {len(over)}"
             + (f" (spread, bound) {dict(list(over.items())[:6])}" if over else ""))
    if chk["dates_without_sun"]:
        L.append(f"  dates WITHOUT sun angles in the frame: {len(chk['dates_without_sun'])} "
                 f"{chk['dates_without_sun'][:10]}")
    if chk["dates_with_nan_cells"]:
        L.append(f"  dates with no-data cells in the frame: {len(chk['dates_with_nan_cells'])} "
                 f"{chk['dates_with_nan_cells'][:10]}")
    c = chk["centre"]
    per = c["per_date"]
    good = [d for d, v in per.items() if v.get("sunZenithAngles") is not None]
    at = np.unique(np.linspace(0, len(good) - 1, min(n, len(good))).round()) if good else []
    pick = [good[int(i)] for i in at]
    where = (f"lon {c['lon']:.5f}, lat {c['lat']:.5f}; " if c.get("lon") is not None else "")
    L.append(f"  at the frame centre ({where}x {c['x']:.1f}, y {c['y']:.1f}), {len(pick)} of "
             f"{len(per)} dates (all in the verification JSON):")
    L.append(f"    {'date':<10}  {'sunZenith':>9}  {'sunAzimuth':>10}  {'viewZenith':>10}  "
             f"{'viewAzimuth':>11}")
    for d in pick:
        v = per[d]
        L.append(f"    {d}  {_fmt_deg(v.get('sunZenithAngles'))}  "
                 f"{_fmt_deg(v.get('sunAzimuthAngles'), 10)}  "
                 f"{_fmt_deg(v.get('viewZenithMean'), 10)}  "
                 f"{_fmt_deg(v.get('viewAzimuthMean'), 11)}")
    return L


# ─────────────────────────────────────────────────────────────────────────────
#  verification and overpass times
# ─────────────────────────────────────────────────────────────────────────────

def reference_grid(site, root=None, log=None):
    """The 10 m grid a store of ``site`` must be on, with where it came from
    (``kind``): the site's netCDF cube when it is on this machine and on a
    10 m grid; else a complete store of the site under ``root`` (a year
    already downloaded); else the site's lon/lat box projected to UTM and
    snapped OUTWARD to multiples of FRAME_SNAP_M m (``snapped_bbox_frame``:
    what is requested then; the netCDF cubes are its 10 m-snapped
    ``predict_grid``, with the same lattice). Its frame is also the
    extent the main jobs request (``site_frame``). ``log`` gets a line
    saying which, louder when it is not the netCDF cube."""
    say = log or (lambda m: None)
    import xarray as xr

    path = ROOT / SITE_CONFIG[site]["reference"]
    why = f"{path.name} is not on this machine"
    if path.is_file():
        with xr.open_dataset(path, mask_and_scale=False) as r:
            g = zc.grid_info(r)
            if np.isclose(abs(g["dx"]), RESOLUTION_M) and np.isclose(abs(g["dy"]), RESOLUTION_M):
                say(f"grid guard: reference grid {path.name} ({g['shape'][0]} x {g['shape'][1]} px, "
                    f"EPSG {g['epsg']})")
                return {"path": str(path), "kind": "netcdf", "x": np.asarray(r["x"].values),
                        "y": np.asarray(r["y"].values), "crs_wkt": g["crs_wkt"]}
            why = f"{path.name} is on a {abs(g['dx']):g} m grid"
    for year in zc.list_years(site, root):
        sp = zc.store_path(site, year, root)
        ds = zc.open_store(sp)
        try:
            g = zc.grid_info(ds)
            say(f"grid guard: {why}; reference grid = the complete store {sp} ({g['shape'][0]} x "
                f"{g['shape'][1]} px, EPSG {g['epsg']}) - weaker than the netCDF cube")
            return {"path": str(sp), "kind": "store", "x": np.asarray(ds["x"].values),
                    "y": np.asarray(ds["y"].values), "crs_wkt": g["crs_wkt"], "why": why}
        finally:
            ds.close()
    try:
        fr = snapped_bbox_frame(site_bbox(site))
    except Exception as exc:
        say(f"GRID GUARD OFF: {why}, no complete store of {site} and no predicted grid "
            f"({type(exc).__name__}: {exc}): the CRS and lattice of the delivered grid are NOT "
            f"checked")
        return None
    x, y = frame_axes(fr)
    say(f"GRID GUARD WEAKER: {why} and no complete store of {site}: reference grid = the bbox "
        f"projected and snapped outward to multiples of {FRAME_SNAP_M} m ({len(y)} x {len(x)} px, "
        f"EPSG {fr['epsg']})")
    return {"path": f"grid predicted from the {site} bbox", "kind": "predicted", "x": x, "y": y,
            "crs_wkt": fr["crs_wkt"], "why": why}


def angles_reference_grid(site, root=None, log=None, res=ANGLES_RES_M):
    """The 300 m grid an angles store of ``site`` must be on (``kind``):
    a complete angles store of the site under ``root`` ("store": every year
    of a site on one grid), else the site's 10 m reference grid
    (``reference_grid``: the netCDF cube, a main store or the bbox) with
    its frame snapped outward to the ``res`` lattice ("predicted"). The
    10 m frame is kept (``fine_edges``) to check that the angles cover it."""
    say = log or (lambda m: None)
    for year in zc.list_angle_years(site, root):
        sp = zc.angles_path(site, year, root)
        ds = zc.open_store(sp)
        try:
            g = zc.grid_info(ds)
            if np.isclose(abs(g["dx"]), res) and np.isclose(abs(g["dy"]), res):
                say(f"angles grid guard: reference grid = the complete angles store {sp} "
                    f"({g['shape'][0]} x {g['shape'][1]} cells of {res} m, EPSG {g['epsg']})")
                return {"path": str(sp), "kind": "store", "x": np.asarray(ds["x"].values),
                        "y": np.asarray(ds["y"].values), "crs_wkt": g["crs_wkt"], "res": res}
        finally:
            ds.close()
    fine = reference_grid(site, root=root, log=None)
    if fine is None:
        say(f"ANGLES GRID GUARD OFF: no {RESOLUTION_M} m reference grid for {site}: the CRS and "
            f"the lattice of the delivered angles are NOT checked")
        return None
    fx, fy = np.asarray(fine["x"], dtype="float64"), np.asarray(fine["y"], dtype="float64")
    cx, cy = coarse_axes(fx, fy, res)
    h = RESOLUTION_M / 2.0
    if fine.get("kind") == "predicted":
        name = f"the {RESOLUTION_M} m grid predicted from the {site} bbox"
    else:
        name = f"{Path(str(fine['path'])).name}"
    say(f"angles grid guard: reference grid = "
        + (name if fine.get("kind") == "predicted" else
           f"the {RESOLUTION_M} m grid of {name} ({fine['kind']})")
        + f" snapped outward to the {res} m lattice ({len(cy)} x {len(cx)} cells, EPSG "
        f"{zc._epsg_from_wkt(fine['crs_wkt'])})")
    return {"path": f"{res} m cells around {name}", "kind": "predicted", "x": cx, "y": cy,
            "crs_wkt": fine["crs_wkt"], "res": res,
            "fine_edges": [float(fx[0] - h), float(fx[-1] + h), float(fy[-1] - h),
                           float(fy[0] + h)]}


def verify_request(req, log, ctx=None):
    """Compare the store with the site's netCDF cube; JSON beside the store.
    ``ctx``: only its --l2a-nodata-equal is read (a band subset's light-path
    gate, ``nine_band_gate``)."""
    cfg = SITE_CONFIG[req.site]
    ref = ROOT / cfg["reference"]
    base = {"site": req.site, "year": req.year, "window": [req.start, req.end],
            "store": str(req.store), "reference": str(ref),
            "checked_utc": utcnow().isoformat(timespec="seconds")}
    p0, p1 = cfg["reference_period"]
    if not ref.is_file():
        rep = dict(base, verdict="no reference", reason=f"{ref} is not on this machine")
    elif not (req.start < p1 and req.end > p0):
        rep = dict(base, verdict="no reference",
                   reason=f"the reference cube covers [{p0}, {p1}), not this window")
    else:
        rep = zc.compare_with_netcdf(req.store, ref, bands=("B03", "B08", "SCL"),
                                     start=req.start, end=req.end, ref_period=(p0, p1),
                                     log=lambda m: log(f"  {m}"))
        rep = dict(base, **rep)
        if rep.get("verdict") == "differs":       # values, or the geometry? (geometry_check)
            rep["geometry"] = geometry_check(req.store, ref, rep)
    written = (Path(req.store) / ".zmetadata").is_file()       # complete, or pending its verdict
    others = [y for y in zc.list_years(req.site, req.root) if y != req.year]
    if others and written:
        a = zc.open_store(req.store)
        b = zc.open_store(zc.store_path(req.site, others[0], req.root))
        rep["same_grid_as_year"] = {str(others[0]): bool(zc._same_grid(a, b))}
        a.close()
        b.close()
    if written:
        attrs = zc.store_attrs(req.store)
        if attrs.get("registration"):
            rep["registration"] = {k: _json_attr(attrs.get(k)) if k != "registration_method"
                                   else attrs.get(k)
                                   for k in ("delivered_offset_px", "delivered_shape_px",
                                             "registration_method", "registration_dates",
                                             "registration_r")}
        rad = zc.store_radiometry(attrs)
        if rad:
            rep["radiometry"] = rad
        bl = zc.store_baselines(attrs)
        if bl:
            rep["input_baselines"] = bl
    zc.write_json(rep, req.sidecar("verify.json"))
    for line in zc.report_lines(rep) if rep.get("dates") else [
            f"verification: {rep['verdict']} ({rep.get('reason')})"]:
        log(line)
    for b, s in (rep.get("bands") or {}).items() if rep.get("verdict") == "differs" else ():
        blk = s.get("differing_by_block") or {}
        if s.get("differing_pixels") and blk:
            log(f"  {b} differing share per {blk['block_px']}-pixel block: columns "
                f"{blk['columns']}; rows {blk['rows']}")
    if rep.get("geometry"):
        log(f"  geometry: {geometry_summary(rep['geometry'])}")
    if rep.get("radiometry"):
        r = rep["radiometry"]
        log(f"radiometry: {r.get('band')} p90 share of valid DN in [1, {r.get('low_dn')}) over "
            f"{r.get('dates_used')} dates = {r.get('p90_share_below_low')} -> {r.get('offset_state')}"
            + (f"; input baselines {rep['input_baselines']['counts']}"
               if rep.get("input_baselines") else ""))
    if req.is_subset and written:   # the light-path gate: the same year's nine-band store
        gate = nine_band_gate(req, log, l2a_nodata=bool(getattr(ctx, "l2a_nodata_equal", False)))
        if gate is not None:
            rep["nine_band_store"] = gate
            zc.write_json(rep, req.sidecar("verify.json"))
    return rep


def geometry_check(store, nc_path, rep, band="B08"):
    """The per-date GEOMETRY of a store that differs from its reference
    (VERIFY_GEOMETRY_*): on every common date of ``rep`` (the report of
    ``compare_with_netcdf``: its window, the reference period and the common
    grid window) whose central crop (at most VERIFY_GEOMETRY_CROP_PX square)
    holds at least VERIFY_GEOMETRY_MIN_VALID valid ``band`` pixels in both,
    the reference's crop less a margin of VERIFY_GEOMETRY_SEARCH_PX is
    registered in the store's crop (``offset_of``, r >=
    VERIFY_GEOMETRY_MIN_R): 'same' (the best offset is (0, 0) with that r and
    a centred peak; a smooth, cloudy date need not be unique within the few
    pixels searched: two Villaviciosa downloads gave 120 such dates of 465,
    every one at (0, 0)), 'displaced' (the peak passes every rule elsewhere:
    ``{date: [dy, dx, r]}``) or 'undetermined' (``{date: reason}``). The
    store is read in blocks of dates."""
    import xarray as xr

    s = VERIFY_GEOMETRY_SEARCH_PX
    cw = (rep.get("grid") or {}).get("common_window")
    out = {"checked": False, "band": band, "search_px": s, "min_r": VERIFY_GEOMETRY_MIN_R,
           "dates_checked": 0, "same": 0, "displaced": {}, "undetermined": {}, "skipped": 0}
    if not cw:
        return dict(out, why="no common grid window")
    zy0, zy1 = cw["store_rows"]
    zx0, zx1 = cw["store_cols"]
    ry0, rx0 = cw["reference_rows"][0], cw["reference_cols"][0]
    Hc, Wc = zy1 - zy0, zx1 - zx0
    h, w = min(Hc, VERIFY_GEOMETRY_CROP_PX), min(Wc, VERIFY_GEOMETRY_CROP_PX)
    if h <= 2 * s + 2 or w <= 2 * s + 2:
        return dict(out, why=f"the common window ({Hc} x {Wc} px) is too small")
    oy, ox = (Hc - h) // 2, (Wc - w) // 2
    out.update(crop=[h, w], crop_rows=[zy0 + oy, zy0 + oy + h], crop_cols=[zx0 + ox, zx0 + ox + w])
    z = zc.open_store(store, bands=[band])
    r = xr.open_dataset(nc_path, mask_and_scale=False)
    try:
        rt = zc._time_dim(r)
        (w0, w1), (p0, p1) = rep["window"], rep["reference_period"]
        zd = {d: i for i, d in enumerate(zc._dates_of(z["t"].values)) if w0 <= d < w1}
        rd = {}
        for j, d in enumerate(zc._dates_of(r[rt].values)):
            if max(w0, p0) <= d < min(w1, p1):
                rd.setdefault(d, j)
        common = sorted(set(zd) & set(rd))
        zf, rf = zc.fill_of(z[band]), zc.fill_of(r[band])
        tb = int(zc.CHUNKS["t"])
        for k in range(0, len(common), tb):
            block = common[k:k + tb]
            za = np.asarray(z[band].isel(t=[zd[d] for d in block], y=slice(zy0 + oy, zy0 + oy + h),
                                         x=slice(zx0 + ox, zx0 + ox + w)).values)
            for m, d in enumerate(block):
                ca = np.asarray(r[band].isel({rt: rd[d], "y": slice(ry0 + oy, ry0 + oy + h),
                                              "x": slice(rx0 + ox, rx0 + ox + w)}).values)
                a = za[m]
                oka, okc = _valid_px(a, zf), _valid_px(ca, rf)
                if oka.mean() < VERIFY_GEOMETRY_MIN_VALID or okc.mean() < VERIFY_GEOMETRY_MIN_VALID:
                    out["skipped"] += 1
                    continue
                out["dates_checked"] += 1
                v = offset_of(a, oka, ca[s:h - s, s:w - s], okc[s:h - s, s:w - s],
                              VERIFY_GEOMETRY_MIN_R)
                o = v.get("offset")
                rel = None if o is None else [int(o[0]) - s, int(o[1]) - s]
                if rel == [0, 0] and v.get("r_ok") and v.get("centred"):
                    out["same"] += 1                # a smooth (cloudy) date: flat, but AT (0, 0)
                elif v.get("pass"):
                    out["displaced"][d] = rel + [v.get("r")]
                else:
                    out["undetermined"][d] = (str(v.get("why"))
                                              + (f" (peak at {tuple(rel)}, r {v.get('r')})"
                                                 if rel is not None else ""))
            del za
    finally:
        z.close()
        r.close()
    return dict(out, checked=True)


def geometry_summary(geo):
    """One line on ``geometry_check``'s record."""
    if not geo or not geo.get("checked"):
        return f"not checked ({(geo or {}).get('why')})"
    n, same = geo["dates_checked"], geo["same"]
    disp, und = geo["displaced"], geo["undetermined"]
    head = (f"the store's {geo['band']} is in place (peak at (0, 0) within {geo['search_px']} px, r "
            f">= {geo['min_r']}) on {same} of the {n} date(s) checked")
    if disp:
        head = (f"DISPLACED on {len(disp)} date(s) ("
                + "; ".join(f"{d} by {tuple(v[:2])} r {v[2]}" for d, v in list(disp.items())[:5])
                + f"); {head}")
    if und:
        head += (f"; undetermined on {len(und)} ("
                 + "; ".join(f"{d}: {w}" for d, w in list(und.items())[:3])[:400] + ")")
    if geo.get("skipped"):
        head += f"; {geo['skipped']} date(s) with too few valid pixels not checked"
    return head


def geometry_blocks(geo):
    """True when ``geometry_check`` says the store is NOT the reference's
    geometry: a displaced date, or in place on fewer than half the dates
    checked. --accept-differences never accepts such a store."""
    if not geo or not geo.get("checked"):
        return False
    return bool(geo["displaced"]) or geo["same"] < 0.5 * geo["dates_checked"]


def _equal_share(s):
    """``(share, n)`` of a band's comparison: the pixel-dates equal (same
    value, or no data in both) over those valid in either store (no data in
    both left out of the count)."""
    n = int(s.get("pixels") or 0) - int(s.get("nodata_both") or 0)
    if n <= 0:
        return None, 0
    return 1.0 - float(s.get("differing_pixels") or 0) / float(n), n


def verification_decision(rep):
    """``(accepted, how, why)`` of a ``verify_request`` report, by the
    declared rule: 'identical' (or 'identical on the common window', the
    --test window) is accepted; 'no reference' is accepted as NOT VERIFIED
    (nothing to compare with: the registration's record says how the frame
    was placed); 'differs' is accepted as NEAR-IDENTICAL when the grid is
    the reference's, no date is displaced (``geometry_check``), no date of
    the reference is missing from the store, and for every band of
    VERIFY_MIN_EQUAL at least that share of the pixel-dates valid in either
    store is equal (same value, or no data in both; dates only in the store
    are listed in the report, not counted); anything else ('not comparable',
    a band missing, a share below its threshold) is refused. The ``why``
    carries the geometry verdict, so --accept-differences is decided on it."""
    v = rep.get("verdict")
    if v in ("identical", "identical on the common window"):
        return True, "identical", v
    if v == "no reference":
        return True, "not verified", f"no reference ({rep.get('reason')})"
    if v != "differs":
        return False, "refused", f"verdict {v!r} ({rep.get('reason')})"
    grid = rep.get("grid") or {}
    if not grid.get("identical"):
        return False, "refused", "the store's grid is not the reference's"
    geo = rep.get("geometry") or {}
    d = rep.get("dates") or {}
    shares, low = {}, []
    if geo.get("displaced"):
        low.append(f"GEOMETRY: {geometry_summary(geo)}")
    for b, need in VERIFY_MIN_EQUAL.items():
        s = (rep.get("bands") or {}).get(b)
        eq, n = _equal_share(s) if s else (None, 0)
        if eq is None:
            low.append(f"{b} not compared")
            continue
        shares[b] = round(eq, 6)
        if eq < need:
            low.append(f"{b} equal on {eq:.4%} of {n} pixel-dates valid in either store "
                       f"(< {need:.1%})")
    if d.get("only_reference"):
        low.append(f"{len(d['only_reference'])} date(s) of the reference missing from the store "
                   f"({d['only_reference'][:6]}{' ...' if len(d['only_reference']) > 6 else ''})")
    tail = (f"; dates only in the store {len(d.get('only_store') or [])}, only in the reference "
            f"{len(d.get('only_reference') or [])}"
            + (f"; geometry: {geometry_summary(geo)}" if geo.get("checked") else ""))
    if low:
        return False, "refused", "; ".join(low) + tail
    return True, "near-identical", ("equal shares "
                                    + ", ".join(f"{b} {v:.4%}" for b, v in shares.items())
                                    + f" >= {VERIFY_MIN_EQUAL}" + tail)


def verify_and_complete(ctx, req, log):
    """Verify a main store written with ``complete=False`` against the
    reference (``verify_request``) and decide (``verification_decision``):
    accepted -> ``complete = 1`` and the decision in the ``verification``
    attribute; refused -> the store stays NOT complete and
    VerificationRefused says where verify.json is, unless
    ``ctx.accept_differences`` (--accept-differences: accepted, recorded;
    never a store whose geometry is not the reference's, ``geometry_blocks``)
    or --test (reported, never refused). The decision is also written into
    ``<year>.verify.json`` (``acceptance``). A band subset is decided by
    ``decide`` (the light-path gate first)."""
    rep = verify_request(req, log, ctx)
    ok, how, why = decide(req, rep)
    geo = rep.get("geometry") or {}
    acc = {"accepted": ok, "how": how, "why": why, "rule": dict(VERIFY_MIN_EQUAL),
           "verdict": rep.get("verdict"), "checked_utc": rep.get("checked_utc")}
    flag = bool(getattr(ctx, "accept_differences", False))
    blocked = not ok and flag and geometry_blocks(geo)
    if not ok and flag and not blocked:
        acc.update(accepted=True, how="accepted with --accept-differences",
                   why=f"refused by the rule ({why}), accepted with --accept-differences")
    elif blocked:
        acc["accept_differences"] = ("--accept-differences given, NOT applied: the geometry is not "
                                     "the reference's (" + geometry_summary(geo) + ")")
    elif not ok and req.test:
        acc.update(accepted=True, how="--test: reported, not refused", why=why)
    rep["acceptance"] = acc
    zc.write_json(rep, req.sidecar("verify.json"))
    if acc["accepted"]:
        mark_complete(req.store, {"verification": json.dumps(acc)})
        log(f"verification decision: {acc['how']} ({acc['why']}); {req.store.name} marked complete")
        return rep
    raise VerificationRefused(
        f"{req.tag}: the written store differs from the reference beyond the rule "
        f"{VERIFY_MIN_EQUAL}: {why}. It is NOT marked complete ({req.store} stays with complete = "
        f"0); see {req.sidecar('verify.json')} (per band, the differing share per "
        f"{zc.DIFF_BLOCK_PX}-pixel block of columns / rows; 'geometry': each date in place or not); "
        f"the raw download stays in {req.part}. "
        + ("--accept-differences was NOT applied: it accepts differing values, never a store whose "
           "geometry is not the reference's" if blocked else
           "Once the differences are explained, run again with --accept-differences to accept it "
           "(recorded in the store; the written store is verified again, not rewritten)"))


def update_overpass(req, dates, log, network=True):
    """Overpass times of the store's dates into ``<site>/overpass_times.json``.

    Seeded from the pipeline's own file of the site (read only), so a date
    keeps the time the existing products used; the rest asked to the STAC
    catalogue, like ``experiments.download_cube.check_overpass``. Returns
    the dates still without a time.
    """
    import pandas as pd

    path = zc.overpass_path(req.site, req.root)
    have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    missing = [d for d in dates if d not in have]
    seeded = 0
    pipe = ROOT / SITE_CONFIG[req.site]["pipeline_overpass"]
    if missing and pipe.is_file():
        old = json.loads(pipe.read_text(encoding="utf-8"))
        for d in missing:
            if d in old:
                have[d] = pd.Timestamp(old[d]).to_pydatetime().isoformat()
                seeded += 1
    missing = [d for d in dates if d not in have]
    asked = 0
    if missing and network:
        import pyintertidal as pit

        pit.net.use_system_certificates()
        fresh = pit.overpass.get_overpass_times(req.bbox, (min(missing), max(missing)),
                                                verbose=False)
        for d in missing:
            if d in fresh:
                have[d] = pd.Timestamp(fresh[d]).to_pydatetime().isoformat()
                asked += 1
    zc.write_json(dict(sorted(have.items())), path)
    still = [d for d in dates if d not in have]
    log(f"overpass times: {len(dates) - len(still)} of {len(dates)} dates in {path} ({seeded} from "
        f"{pipe.name if pipe.is_file() else 'no pipeline file'}, {asked} from the catalogue)"
        + (f"; STILL MISSING {still}" if still else ""))
    return still


def compare_dates(angle_dates, req):
    """The dates of an angles store against those of the main store of the
    same year (same root): ``{"compared", "angles", "main", "common",
    "only_angles", "only_main", "main_store"}``; not compared while the main
    store is not complete."""
    main = zc.store_path(req.site, req.year, req.root)
    out = {"main_store": str(main), "angles": len(angle_dates)}
    if not zc.is_complete(main):
        return dict(out, compared=False, main=None,
                    reason=f"{main.name} is not complete (not downloaded yet?): compare later with "
                           f"--verify-only")
    ds = zc.open_store(main)
    try:
        mdates = zc.dates(ds)
    finally:
        ds.close()
    sa, sm = set(angle_dates), set(mdates)
    return dict(out, compared=True, main=len(mdates), common=len(sa & sm),
                only_angles=sorted(sa - sm), only_main=sorted(sm - sa))


def report_angle_dates(dates):
    if not dates.get("compared"):
        return [f"dates: {dates['angles']} in the angles store; NOT compared with the main store "
                f"({dates.get('reason')})"]
    same = not dates["only_angles"] and not dates["only_main"]
    return [f"dates: angles {dates['angles']}, main store {dates['main']}, common {dates['common']}"
            + ("; IDENTICAL" if same else
               f"; only in the angles {dates['only_angles'][:12]}"
               f"{' ...' if len(dates['only_angles']) > 12 else ''}; only in the main store "
               f"{dates['only_main'][:12]}{' ...' if len(dates['only_main']) > 12 else ''}")]


def unregistered_angles(attrs):
    """The problem of an angles store placed by its x/y from a PADDED
    delivery (not registered: ``angles_placement``), or None. A store written
    before 2026-10-19 has no ``placement`` attribute: its ``delivered_notes``
    tell (the coordinate crop of a larger delivered grid)."""
    rec = _json_attr(attrs.get("placement"))
    if isinstance(rec, dict):
        if not rec.get("padded"):
            return None
        return (f"padded delivery ({rec.get('delivered_shape_cells')} cells for the "
                f"{rec.get('frame_shape_cells')} cell frame) placed by its x/y, NOT registered: not "
                f"for the sensing-time inversion until registered")
    notes = " ".join(str(n) for n in (_json_attr(attrs.get("delivered_notes")) or []))
    m = re.search(r"delivered grid (\d+) x (\d+) px from .*? put on the reference grid (\d+) x (\d+) "
                  r"px", notes)
    if m and (int(m.group(1)) > int(m.group(3)) or int(m.group(2)) > int(m.group(4))):
        return (f"written before 2026-10-19 from a padded delivery ({m.group(1)} x {m.group(2)} "
                f"cells for the {m.group(3)} x {m.group(4)} cell frame) placed by its x/y, NOT "
                f"registered: not for the sensing-time inversion until registered")
    return None


def verify_angles(req, log):
    """Check a written angles store; JSON beside it (``<year>.angles.
    verify.json``) and the log. The verdict lists the problems: dates that
    differ from the main store of the year, values outside the physical
    ranges, sun angles exactly 0 (an undeclared no-data), dates whose
    sun-angle spread exceeds the geometric bound, dates without sun angles
    (``angle_checks``); "ok" otherwise."""
    ds = zc.open_store(req.store)
    try:
        dts = zc.dates(ds)
        chk = angle_checks(ds)
    finally:
        ds.close()
    dates = compare_dates(dts, req)
    problems = []
    if dates["compared"] and (dates["only_angles"] or dates["only_main"]):
        problems.append(f"dates differ from the main store ({len(dates['only_angles'])} only in "
                        f"the angles, {len(dates['only_main'])} only in the main store)")
    oor = [b for b, r in chk["ranges"].items() if r["out_of_range"]]
    if oor:
        problems.append(f"values outside the physical range in {oor}")
    zeros = sun_zeros(chk)
    if zeros:
        problems.append("sun angles exactly 0 (an undeclared no-data?): " + "; ".join(zeros))
    if chk["dates_over_bound"]:
        problems.append(f"{len(chk['dates_over_bound'])} date(s) with a sun-angle spread above the "
                        f"bound")
    if chk["dates_without_sun"]:
        problems.append(f"{len(chk['dates_without_sun'])} date(s) without sun angles in the frame")
    unreg = unregistered_angles(zc.store_attrs(req.store))
    if unreg:
        problems.append(unreg)
    verdict = "; ".join(problems) if problems else (
        "ok" if dates["compared"] else "ok (dates not compared: no complete main store)")
    rep = {"site": req.site, "year": req.year, "window": [req.start, req.end],
           "store": str(req.store), "checked_utc": utcnow().isoformat(timespec="seconds"),
           "verdict": verdict, "problems": problems, "dates": dates}
    rep.update(chk)
    zc.write_json(rep, req.sidecar("verify.json"))
    log("angles verification: " + ("PROBLEMS: " + verdict if problems else
                                   "OK" if verdict == "ok" else "OK, " + verdict[4:-1]))
    for line in report_angle_dates(dates) + report_angle_checks(chk):
        log(f"  {line}")
    return rep


# ─────────────────────────────────────────────────────────────────────────────
#  one store, start to end
# ─────────────────────────────────────────────────────────────────────────────

class Context:
    """Run-wide settings and the cached openEO connection."""

    def __init__(self, args=None, connect=None, sleep=time.sleep):
        a = args or argparse.Namespace()
        self.job_options = (json.loads(a.job_options) if getattr(a, "job_options", None)
                            else dict(JOB_OPTIONS))
        self.format_options = (json.loads(a.format_options) if getattr(a, "format_options", None)
                               else None)
        self.backend_format = getattr(a, "backend_format", None) or BACKEND_FORMAT
        self.retries = int(getattr(a, "retries", 1))
        self.max_wait_s = float(getattr(a, "max_wait_hours", 24.0)) * 3600.0
        self.keep_raw = bool(getattr(a, "keep_raw", False))
        self.allow_offset = bool(getattr(a, "allow_grid_offset", False))
        self.allow_padding = bool(getattr(a, "allow_padding", False))
        self.allow_snapped = bool(getattr(a, "allow_snapped_frame", False))
        self.nodata = getattr(a, "nodata", None)
        self.new_job = bool(getattr(a, "new_job", False))
        #: --offset-px (rows, cols) of the frame in the delivered arrays, or None
        self.offset_px = getattr(a, "offset_px", None)
        #: --accept-differences: a store refused by the verification rule is accepted (recorded)
        self.accept_differences = bool(getattr(a, "accept_differences", False))
        #: --accept-unverified-offset: an --offset-px nothing can check is taken although the
        #: delivered arrays are not the shape the site's stores were delivered in (recorded)
        self.accept_unverified_offset = bool(getattr(a, "accept_unverified_offset", False))
        #: --without-light-gate (band subsets): buy a year without a nine-band store of its own
        #: before any store of its bands passed the light-path gate (recorded)
        self.without_light_gate = bool(getattr(a, "without_light_gate", False))
        #: --l2a-nodata-equal (band subsets): the light-path gate may pass up to the no-data
        #: encoding (L2A 0 / fill; nodata_encoding_check), recorded as NINE_BAND_NODATA_HOW
        self.l2a_nodata_equal = bool(getattr(a, "l2a_nodata_equal", False))
        #: band subsets of EXTRACTION_GATE_SITES: where the converter's 2023 extraction gates are
        #: (--extraction-gate-dir) and --without-extraction-gate (recorded)
        self.extraction_gate_dir = getattr(a, "extraction_gate_dir", None) or EXTRACTION_GATE_DIR
        self.without_extraction_gate = bool(getattr(a, "without_extraction_gate", False))
        #: --expect-account: the e-mail the connection must be authenticated as (None: unchecked)
        self.expect_account = getattr(a, "expect_account", None)
        self.account = None
        self.network = True
        self.poll_s, self.poll_max_s, self.capacity_wait_s = 10.0, 60.0, 300
        self.sleep = sleep
        self._connect = connect
        self._conn = None

    def connection(self):
        if self._conn is None:
            if self._connect is not None:
                conn = self._connect()
            else:
                import pyintertidal as pit

                pit.net.use_system_certificates()
                conn = pit.scenes.connect(interactive=False)
            if self.expect_account:         # before anything is submitted on it (AccountMismatch)
                self.account = check_account(conn, self.expect_account)
            self._conn = conn
        return self._conn


def account_email(desc):
    """The e-mail of an openEO ``describe_account()`` (``GET /me``): the first
    ``email`` value found (CDSE: ``info.oidc_userinfo.email``), else a user id,
    name or preferred user name holding an '@'; None when there is none."""
    found = []

    def walk(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, str(k).lower())
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, key)
        elif isinstance(o, str) and "@" in o:
            found.append((0 if key == "email" else 1 if key in ("preferred_username", "user_id", "name",
                                                                 "username") else 2, o.strip()))

    walk(desc)
    found = [f for f in found if f[0] < 2]
    return min(found, key=lambda f: f[0])[1] if found else None


def check_account(conn, expected, log=print):
    """--expect-account: ``conn.describe_account()``, the account's e-mail
    printed; AccountMismatch when it is not ``expected`` (case-insensitive) or
    names no e-mail. Returns ``{"email", "user_id", "name"}``."""
    try:
        desc = conn.describe_account()
    except Exception as exc:  # noqa: BLE001 - any failure: the account is unknown, nothing starts
        raise AccountMismatch(f"--expect-account {expected}: the account could not be described "
                              f"({type(exc).__name__}: {exc}); nothing started") from None
    email = account_email(desc)
    who = {"email": email, "user_id": (desc or {}).get("user_id") if isinstance(desc, dict) else None,
           "name": (desc or {}).get("name") if isinstance(desc, dict) else None}
    log(f"openEO account: {email or '(no e-mail in the account description)'} (user_id "
        f"{who['user_id']}); expected {expected}")
    if not email:
        raise AccountMismatch(f"--expect-account {expected}: the account description names no e-mail "
                              f"(keys {sorted(desc) if isinstance(desc, dict) else type(desc).__name__}); "
                              f"nothing started")
    if email.strip().lower() != str(expected).strip().lower():
        raise AccountMismatch(f"--expect-account {expected}: the connection is authenticated as {email}, "
                              f"another account (OPENEO_CONFIG_HOME selects the refresh token); nothing "
                              f"started")
    return who


def snap_to_lattice(interp, ref, step=RESOLUTION_M):
    """Undo the float32 rounding of delivered x/y: an axis delivered as
    float32 whose values are ALL the float32 rounding of lines ``ref[axis][0]
    + k * step`` (the reference's 10 m lattice; the angles' cell centres lie
    on lines of it too) takes those lines' exact float64 values. Pixel
    centres in whole metres are exact in float32, so a delivery on the
    lattice deviates by 0 here; a value farther than half a float32 step
    from the lattice (0.016-0.03 m at eastings, 0.25 m at northings of
    4-8e6 m) is not rounding: the axis is left as ``interpret`` regularised
    it (through the median of the delivered values) for the lattice check,
    which refuses any offset of it (1e-3 px). float64 axes are never
    touched. Returns the notes."""
    if ref is None:
        return []
    name = Path(str(ref["path"])).name if ref.get("kind") in ("netcdf", "store") else ref["path"]
    notes = []
    for axis in ("x", "y"):
        dt = interp.coord_dtypes.get(axis)
        if not dt or not _low_precision(dt):
            continue
        native = np.dtype(dt).newbyteorder("=")
        # the values as delivered (interpret regularised the axis around their median)
        v = np.asarray(interp.raw_axes.get(axis, getattr(interp, axis)), dtype="float64")
        r = np.asarray(ref[axis], dtype="float64")
        if not len(v) or not len(r):
            continue
        exact = r[0] + np.round((v - r[0]) / step) * step
        dev = float(np.max(np.abs(v - exact)))
        image = exact.astype(native).astype("float64")
        off = np.flatnonzero(image != v.astype(native).astype("float64"))
        half = 0.5 * float(np.spacing(np.abs(v).max().astype(native)))
        if not len(off):
            setattr(interp, axis, exact)
            notes.append(f"{axis} (delivered as {dt}) {dev:.3f} m ({dev / RESOLUTION_M:.3f} px) from "
                         f"the {step:g} m lattice of {name}: "
                         + ("exactly its float32 values" if dev == 0 else
                            f"float32 rounding (at most half a float32 step, {half:.3f} m here)")
                         + "; the lattice's exact float64 values are used")
        else:
            notes.append(f"{axis} (delivered as {dt}): {len(off)} of {len(v)} values are not the "
                         f"float32 rounding of the {step:g} m lattice of {name} (up to {dev:.3f} m "
                         f"= {dev / RESOLUTION_M:.3f} px off; half a float32 step here is "
                         f"{half:.3f} m): not rounding, so the lattice check decides on the regular "
                         f"axis through their median")
    return notes


def lattice_check(interp, ref_grid, whole=True):
    """``(error, extent)`` of the delivered grid against the reference grid.

    error: another CRS, or pixel centres off the reference's 10 m lattice
    (half a pixel would be corners delivered as centres): such a store would
    not be on the grid of the existing cubes. extent: same lattice, another
    extent (a message; the caller crops / pads onto the reference grid;
    ignored for --test, ``whole=False``)."""
    if ref_grid is None:
        return None, None
    if ref_grid.get("crs_wkt") and not zc.crs_equal(interp.crs_wkt, ref_grid["crs_wkt"]):
        return (f"the delivered CRS (EPSG {zc._epsg_from_wkt(interp.crs_wkt)}) is not the CRS of "
                f"{ref_grid['path']} (EPSG {zc._epsg_from_wkt(ref_grid['crs_wkt'])})"), None
    rx = np.asarray(ref_grid["x"], dtype="float64")
    ry = np.asarray(ref_grid["y"], dtype="float64")
    if len(rx) < 2 or len(ry) < 2 or not len(interp.x) or not len(interp.y):
        return None, None
    dxr, dyr = float(rx[1] - rx[0]), float(ry[1] - ry[0])
    if not (np.isclose(abs(dxr), RESOLUTION_M) and np.isclose(abs(dyr), RESOLUTION_M)):
        return (f"the reference grid {ref_grid['path']} is not a {RESOLUTION_M} m grid "
                f"({dxr:g} / {dyr:g} m)"), None
    fx = ((float(interp.x[0]) - rx[0]) / dxr) % 1.0
    fy = ((float(interp.y[0]) - ry[0]) / dyr) % 1.0
    fx, fy = min(fx, 1.0 - fx), min(fy, 1.0 - fy)
    if fx > 1e-3 or fy > 1e-3:
        f32 = [a for a in ("x", "y") if _low_precision(interp.coord_dtypes.get(a))]
        return (f"the delivered pixel centres are offset by ({fx:.3f}, {fy:.3f}) px from the grid "
                f"of {ref_grid['path']} (0.5 = pixel corners delivered as centres?)"
                + (f"; {'/'.join(f32)} delivered as float32, whose rounding (half a float32 "
                   f"step) is absorbed: this is a real offset" if f32 else "")), None
    if whole and (len(interp.x) != len(rx) or len(interp.y) != len(ry)
                  or not np.isclose(interp.x[0], rx[0], atol=1e-3)
                  or not np.isclose(interp.y[0], ry[0], atol=1e-3)):
        return None, (f"same 10 m lattice as {Path(ref_grid['path']).name} but another extent: "
                      f"{len(interp.y)} x {len(interp.x)} px from ({interp.x[0]}, {interp.y[0]}) "
                      f"against {len(ry)} x {len(rx)} px from ({rx[0]}, {ry[0]})")
    return None, None


def _phase(a, b, step):
    """Distance of ``a - b`` from the nearest multiple of ``step``, in steps."""
    f = ((float(a) - float(b)) / step) % 1.0
    return min(f, 1.0 - f)


def angles_grid_check(interp, ref, whole=True, res=ANGLES_RES_M):
    """``(error, extent, warning)`` of a delivered angles grid against the
    angles reference grid (``angles_reference_grid``); the spacing (``res``)
    was checked by ``interpret``.

    error: another CRS; cell edges that are not lines of the 10 m lattice of
    the main grid; or, against an angles store of the site already there,
    another phase of the 300 m lattice (its years would not share a grid).
    extent: the reference's 300 m lattice with another extent (the caller
    crops / pads onto the reference; ignored for --test, ``whole=False``).
    warning: the cells sit at another phase of the 10 m lattice than
    predicted (the first store of a site: kept as delivered, and it becomes
    the reference of the site's other years), or the angle frame does not
    cover the 10 m frame."""
    if ref is None:
        return None, None, None
    if ref.get("crs_wkt") and not zc.crs_equal(interp.crs_wkt, ref["crs_wkt"]):
        return (f"the delivered CRS (EPSG {zc._epsg_from_wkt(interp.crs_wkt)}) is not the CRS of "
                f"{ref['path']} (EPSG {zc._epsg_from_wkt(ref['crs_wkt'])})"), None, None
    x, y = np.asarray(interp.x, dtype="float64"), np.asarray(interp.y, dtype="float64")
    rx, ry = np.asarray(ref["x"], dtype="float64"), np.asarray(ref["y"], dtype="float64")
    if not (len(x) and len(y) and len(rx) and len(ry)):
        return None, None, None
    name = Path(str(ref["path"])).name if ref.get("kind") == "store" else ref["path"]
    warn, extent = None, None
    if _phase(x[0], rx[0], res) > 1e-6 or _phase(y[0], ry[0], res) > 1e-6:
        ex, ey = _phase(x[0], rx[0], RESOLUTION_M), _phase(y[0], ry[0], RESOLUTION_M)
        off = (f"({(x[0] - rx[0]) % res:g}, {(y[0] - ry[0]) % res:g}) m from the cell centres of "
               f"{name}")
        if ex > 1e-3 or ey > 1e-3:
            return (f"the delivered {res} m cells are {off}: their edges are not lines of the "
                    f"{RESOLUTION_M} m lattice of the main grid"), None, None
        if ref.get("kind") == "store":
            return (f"the delivered {res} m cells are {off}, another phase of the {RESOLUTION_M} m "
                    f"lattice than the site's angles store: the years would not share a "
                    f"grid"), None, None
        warn = (f"the delivered {res} m cells are {off} (not at multiples of {res} m, as "
                f"predicted), but their edges are lines of the {RESOLUTION_M} m lattice of the "
                f"main grid: kept as delivered; this store becomes the reference grid of the "
                f"site's other years")
    elif whole and (len(x) != len(rx) or len(y) != len(ry) or not np.isclose(x[0], rx[0], atol=1e-3)
                    or not np.isclose(y[0], ry[0], atol=1e-3)):
        extent = (f"same {res} m lattice as {name} but another extent: {len(y)} x {len(x)} cells "
                  f"from ({x[0]}, {y[0]}) against {len(ry)} x {len(rx)} cells from ({rx[0]}, "
                  f"{ry[0]})")
    fe = ref.get("fine_edges")
    if whole and fe is not None and extent is None:
        h = res / 2.0
        xe0, xe1, ye0, ye1 = x[0] - h, x[-1] + h, y[-1] - h, y[0] + h
        if xe0 > fe[0] + 1e-6 or xe1 < fe[1] - 1e-6 or ye0 > fe[2] + 1e-6 or ye1 < fe[3] - 1e-6:
            cover = (f"the angle frame (x {xe0:.0f}..{xe1:.0f}, y {ye0:.0f}..{ye1:.0f}) does not "
                     f"cover the {RESOLUTION_M} m frame (x {fe[0]:.0f}..{fe[1]:.0f}, y "
                     f"{fe[2]:.0f}..{fe[3]:.0f}): angles_at refuses points outside it")
            warn = cover if warn is None else f"{warn}; {cover}"
    return None, extent, warn


def _same_axes(x, y, grid):
    """Axes ``x``, ``y`` with the extent of ``grid`` (lengths, first centres to 1 mm)?"""
    gx, gy = np.asarray(grid["x"], dtype="float64"), np.asarray(grid["y"], dtype="float64")
    return bool(len(x) == len(gx) and len(y) == len(gy) and len(gx) and len(gy)
                and np.isclose(x[0], gx[0], rtol=0, atol=1e-3)
                and np.isclose(y[0], gy[0], rtol=0, atol=1e-3))


def _same_extent(interp, grid):
    return _same_axes(interp.x, interp.y, grid)


def _on_lattice_of(xs, ys, grid, step):
    """Do axes starting at ``xs``, ``ys`` lie on the ``step`` lattice of ``grid``?"""
    gx, gy = np.asarray(grid["x"], dtype="float64"), np.asarray(grid["y"], dtype="float64")
    return _phase(xs, gx[0], step) <= 1e-6 and _phase(ys, gy[0], step) <= 1e-6


def delivered_extent_note(attrs, req):
    """The delivered ``extent`` attribute (the openEO Zarr writer's [west,
    south, east, north]) against the UTM extent requested, or None."""
    v = (attrs or {}).get("extent")
    if req.extent is None or not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        got = [float(a) for a in v]
    except (TypeError, ValueError):
        return None
    want = [float(req.extent[k]) for k in ("west", "south", "east", "north")]
    d = max(abs(a - b) for a, b in zip(got, want))
    if d <= 0.01:
        return f"the delivered 'extent' attribute {list(v)} is the requested extent"
    return (f"the delivered 'extent' attribute {list(v)} is NOT the requested extent {want} (up to "
            f"{d:.2f} m off); the coordinates place the pixels, not this attribute")


def placement_target(interp, req, guard, rgrid, extent):
    """``(target, why)``: the grid the store must have when the delivery's
    extent is not it, else ``(None, None)`` (the choice of
    ``put_on_target``). Production: the reference grid (``extent``, the
    lattice check's message, says the delivery has another extent). --test
    with a UTM extent: the requested window (``rgrid``), which must lie on
    the reference lattice (FormatError otherwise)."""
    res = req.spec.resolution
    target, why = None, None
    if req.test and rgrid is not None:
        if guard is not rgrid:
            if guard.get("crs_wkt") and not zc.crs_equal(rgrid["crs_wkt"], guard["crs_wkt"]):
                raise FormatError(f"the requested window is in EPSG:{req.extent['crs']}, the reference "
                                  f"grid {guard['path']} in EPSG {zc._epsg_from_wkt(guard['crs_wkt'])}")
            gx = np.asarray(guard["x"], dtype="float64")
            step = abs(float(gx[1] - gx[0])) if len(gx) > 1 else float(res)
            if not _on_lattice_of(rgrid["x"][0], rgrid["y"][0], guard, step):
                raise FormatError(f"the requested window ({rgrid['path']}) is not on the lattice of "
                                  f"{guard['path']}")
        if not _same_extent(interp, rgrid):
            target, why = rgrid, (f"the delivered grid ({len(interp.y)} x {len(interp.x)} from "
                                  f"({interp.x[0]}, {interp.y[0]})) is not the --test window, "
                                  f"{rgrid['path']}")
    elif extent:
        target, why = guard, extent
    return target, why


def put_on_target(interp, req, guard, rgrid, extent, ref_grid, allow_padding=False,
                  registered=None):
    """After the lattice check passed: crop / pad the delivery onto the grid
    the store must have (``placement_target``). ``registered``: the record of
    ``register_delivery``; when it holds an offset, every band is cropped AT
    THAT OFFSET onto the target (``place_at_offset``: the delivered x/y are
    not used to place the pixels). Otherwise by coordinates
    (``align_to_grid``): an alignment that only crops is a note, and so is
    what the cropped-away margin held (``format_margin``: the Zarr writer's
    padding to whole chunks, or pixels outside the frame). PADDING (the
    delivery does not cover the target: rows / columns inside the frame
    would be the fill, which reads as no-data) stops a production year
    (FormatError, nothing written) unless ``allow_padding``
    (--allow-padding), and is a warning otherwise. A production extent that
    is not the reference frame is a warning too."""
    res = req.spec.resolution
    target, why = placement_target(interp, req, guard, rgrid, extent)
    if (not req.test and rgrid is not None and ref_grid is not None
            and not _same_axes(rgrid["x"], rgrid["y"], ref_grid)):
        interp.warnings.append(f"{rgrid['path']} is not the frame of {ref_grid['path']}: the store "
                               f"follows the reference grid")
    if (registered or {}).get("offset") is not None:
        place_at_offset(interp, target, registered, why)
        return
    if target is None:
        return
    if not _on_lattice_of(interp.x[0], interp.y[0], target, float(res)):
        interp.warnings.append(f"{why}; but the delivered grid is not on its lattice: kept as "
                               f"delivered")
        return
    note, padded, margin = align_to_grid(interp, target["x"], target["y"], res=res)
    if padded and not req.test and not allow_padding:
        raise FormatError(f"{why}; {note}: the delivery does not cover the whole reference grid, "
                          f"so rows / columns INSIDE the frame would be the fill, which reads as "
                          f"no-data (a request or backend problem, not a scene without data); "
                          f"nothing written (run again with --allow-padding to write it padded)")
    (interp.warnings if padded else interp.notes).append(f"{why}; {note}")
    if margin:
        interp.notes.append(format_margin(margin, interp.fills)[0])


# ─────────────────────────────────────────────────────────────────────────────
#  registration: where the frame's pixels sit in a padded delivery
#  (module docstring, "Registration")
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegistrationPlan:
    """Where the place of the frame in a delivery may come from
    (``registration_plan``): the reference netCDF covering the window, an
    expert offset (--offset-px), registered stores of the same request under
    ``root``. ``strict``: production refuses what the rules do not
    establish; --test warns and keeps the delivered coordinates."""
    reference: str = None             # the site's reference netCDF, when it covers the window
    reference_period: tuple = None    # the period it was requested for (end-exclusive)
    no_reference: str = ""            # why there is none
    expert: tuple = None              # --offset-px (rows, cols)
    strict: bool = True
    root: object = None               # registered sibling stores are looked for here (None: nowhere)
    accept_unverified: bool = False   # --accept-unverified-offset (an expert offset nothing can check)
    #: a band subset (--bands): more roots whose registered stores check an offset (the
    #: nine-band root, read only), and an offset taken only when the composites CONFIRM it on
    #: a store delivered in this delivery's array shape (unless accept_unverified)
    sibling_roots: tuple = ()
    require_confirmed: bool = False


def registration_plan(req, ctx=None, ref_grid=None):
    """The ``RegistrationPlan`` of a main request: the site's reference
    netCDF when it is on this machine, is the 10 m reference grid
    (``ref_grid``, from ``reference_grid``) and covers the request's window;
    ``ctx.offset_px`` (--offset-px; production only) and
    ``ctx.accept_unverified_offset``; registered stores of the site under the
    request's root (production only). A band subset (--bands): also those
    under its ``main_root`` (read only), and ``require_confirmed``."""
    cfg = SITE_CONFIG[req.site]
    path = ROOT / cfg["reference"]
    p0, p1 = cfg["reference_period"]
    expert = None if req.test else getattr(ctx, "offset_px", None)
    common = dict(expert=tuple(int(v) for v in expert) if expert else None, strict=not req.test,
                  root=None if req.test else Path(req.root),
                  accept_unverified=bool(getattr(ctx, "accept_unverified_offset", False)))
    if req.is_subset and not req.test:
        common.update(require_confirmed=True, sibling_roots=tuple(
            Path(r) for r in (req.main_root,) if r is not None
            and Path(r).resolve() != Path(req.root).resolve()))
    if not path.is_file():
        why = f"{path.name} is not on this machine"
    elif not (req.start < p1 and req.end > p0):
        why = f"{path.name} covers [{p0}, {p1}), not [{req.start}, {req.end})"
    elif (ref_grid is None or ref_grid.get("kind") != "netcdf"
          or Path(str(ref_grid.get("path"))).resolve() != path.resolve()):
        why = f"{path.name} is not the {RESOLUTION_M} m reference grid of {req.site}"
    else:
        return RegistrationPlan(reference=str(path), reference_period=(p0, p1), **common)
    return RegistrationPlan(no_reference=why, **common)


def registration_rule():
    """The constants of the registration, as recorded with every store."""
    return {"bands": list(REGISTRATION_BANDS), "clear_scl": list(REGISTRATION_CLEAR_SCL),
            "min_clear": REGISTRATION_MIN_CLEAR, "min_cover": REGISTRATION_MIN_COVER,
            "dates": REGISTRATION_DATES, "min_dates": REGISTRATION_MIN_DATES,
            "max_tried": REGISTRATION_MAX_TRIED, "min_r": REGISTRATION_MIN_R,
            "min_r_any": REGISTRATION_MIN_R_ANY, "peak_px": REGISTRATION_PEAK_PX,
            "min_margin": REGISTRATION_MIN_MARGIN, "max_subpixel": REGISTRATION_MAX_SUBPIXEL,
            "min_overlap": REGISTRATION_MIN_OVERLAP,
            "sibling_dates": REGISTRATION_SIBLING_DATES,
            "sibling_min_r": REGISTRATION_SIBLING_MIN_R,
            "frame_sized_px": REGISTRATION_FRAME_SIZED_PX,
            "band_20m": REGISTRATION_20M_BAND, "min_r_20m": REGISTRATION_20M_MIN_R,
            "phase_bands": list(REGISTRATION_PHASE_BANDS),
            "phase_min_gain": REGISTRATION_PHASE_MIN_GAIN,
            "scl_search_px": REGISTRATION_SCL_SEARCH_PX, "scl_tol": REGISTRATION_SCL_TOL}


def _valid_px(a, fill):
    """Valid pixels of a band array: not the fill, not 0 (the L2A no-data), finite."""
    a = np.asarray(a)
    ok = a != 0
    if a.dtype.kind == "f":
        ok &= np.isfinite(a)
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        ok &= a != fill
    return ok


def core_window(shape, frame_shape):
    """``(rows, cols)`` slices of delivered arrays of ``shape`` (rows, cols)
    that lie inside a frame of ``frame_shape`` at EVERY offset the frame can
    take in them (rows Hr - H .. H, columns Wr - W .. W): where the clear
    pixels of a date are counted before the offset is known, so the margin
    (data outside the frame, or the fill) does not choose the dates. The
    whole axis when that core is empty (arrays more than twice the frame)."""
    (Hr, Wr), (H, W) = (int(v) for v in shape), (int(v) for v in frame_shape)
    rows = slice(Hr - H, H) if 0 <= Hr - H < H <= Hr else slice(0, Hr)
    cols = slice(Wr - W, W) if 0 <= Wr - W < W <= Wr else slice(0, Wr)
    return rows, cols


def _window_px(rows, cols, frame_shape=None):
    """Pixels of the window rows x cols, at most the frame's (the
    whole-array fallback of core_window holds more than one frame)."""
    n = int((rows.stop - rows.start) * (cols.stop - cols.start))
    return n if frame_shape is None else min(n, int(frame_shape[0]) * int(frame_shape[1]))


def clear_counts(band, idx, fill, rows=None, cols=None):
    """``(valid, clear)`` pixel counts of a (t, y, x) SCL band on the dates
    ``idx`` (valid: neither the fill nor 0; clear: the REGISTRATION_CLEAR_SCL
    classes), in the window ``rows`` x ``cols`` (default: the whole array),
    one date at a time, or the dates of one time chunk together when the
    array is chunked over several dates (a store: each chunk decoded once):
    each read touches its own chunks and nothing else (one dask graph over
    every date of a padded delivery, ~460 dates x 930 chunks at Ems, took
    minutes and gigabytes before computing anything)."""
    idx = [int(i) for i in idx]
    nv, nc = np.zeros(len(idx), "int64"), np.zeros(len(idx), "int64")
    real = fill is not None and not (isinstance(fill, float) and np.isnan(fill))
    data = band.data if hasattr(band, "dims") else band
    rows = rows if rows is not None else slice(None)
    cols = cols if cols is not None else slice(None)
    tch = getattr(data, "chunks", None)
    bounds = np.cumsum((0,) + tuple(tch[0])) if tch else None
    groups = {}
    for k, i in enumerate(idx):           # dates of one time chunk are read together
        g = int(np.searchsorted(bounds, i, side="right")) if bounds is not None else k
        groups.setdefault(g, []).append((k, i))
    for members in groups.values():
        block = np.asarray(data[[i for _, i in members], rows, cols])
        for m, (k, _) in enumerate(members):
            a = block[m]
            ok = a != 0
            if real:
                ok &= a != fill
            nv[k] = int(ok.sum())
            nc[k] = int(np.isin(a, REGISTRATION_CLEAR_SCL).sum())
        del block
    return nv, nc


def usable_dates(dates, nv, nc, frame_px, n=REGISTRATION_DATES):
    """The dates of ``clear_counts`` the registration may use: clear on at
    least REGISTRATION_MIN_CLEAR of their valid SCL pixels, valid on at least
    REGISTRATION_MIN_COVER of ``frame_px`` (the pixel count of the window
    counted); the ``n`` clearest (most clear pixels; None: all) first.
    Returns ``(chosen [(date, position)], number usable)``."""
    ok = []
    for k, (d, v, c) in enumerate(zip(dates, nv, nc)):
        share = float(c) / float(v) if v else 0.0
        if v >= REGISTRATION_MIN_COVER * frame_px and share >= REGISTRATION_MIN_CLEAR:
            ok.append((-int(c), str(d), k))
    ok.sort()
    return [(d, k) for _, d, k in ok[:n]], len(ok)


def _fft_pair():
    """``(rfft2, irfft2)`` at a given size: scipy's (all cores) when present."""
    try:
        from scipy import fft as sfft

        return (lambda a, s: sfft.rfft2(a, s=s, workers=-1),
                lambda A, s: sfft.irfft2(A, s=s, workers=-1))
    except ImportError:                     # pragma: no cover - scipy is installed here
        return (lambda a, s: np.fft.rfft2(a, s=s), lambda A, s: np.fft.irfft2(A, s=s))


def _standardised(a, ok):
    """``a`` as float64, centred and scaled on its valid pixels, 0 elsewhere
    (r does not change; the sums of the correlation stay small)."""
    v = np.where(ok, np.asarray(a, dtype="float64"), 0.0)
    if ok.any():
        m = float(v[ok].mean())
        s = float(v[ok].std()) or 1.0
        v = np.where(ok, (v - m) / s, 0.0)
    return v


def masked_correlation(image, image_ok, template, template_ok):
    """Pearson r of ``template`` (h x w) against the window ``image[dy:dy +
    h, dx:dx + w]`` at EVERY offset 0 <= dy <= H - h, 0 <= dx <= W - w, over
    the pixels valid in both (masked normalised cross-correlation by FFT,
    Padfield 2012): ``(r, n)``, arrays of shape (H - h + 1, W - w + 1), n the
    pixels compared, r NaN where fewer than 2 or a side is constant. The
    transforms are taken at the image's size with the template at the
    origin: a window inside the image never wraps around, so these are the
    linear correlations of the full delivered array, not a cyclic shift."""
    H, W = image.shape
    h, w = template.shape
    if h > H or w > W:
        raise ValueError(f"a {h} x {w} frame cannot lie inside a {H} x {W} array")
    fw, bw = _fft_pair()
    s, oh, ow = (H, W), H - h + 1, W - w + 1

    def corr(A, B):
        return np.array(bw(A * B, s)[:oh, :ow])          # a copy: the full array is freed

    mt = template_ok.astype("float64")
    t = _standardised(template, template_ok)
    Cm, Ct, Ct2 = np.conj(fw(mt, s)), np.conj(fw(t, s)), np.conj(fw(t * t, s))
    del mt, t
    F = fw(image_ok.astype("float64"), s)
    n = np.rint(corr(F, Cm))
    st, st2 = corr(F, Ct), corr(F, Ct2)
    a = _standardised(image, image_ok)
    F = fw(a, s)
    sa, sta = corr(F, Cm), corr(F, Ct)
    F = fw(a * a, s)
    del a
    sa2 = corr(F, Cm)
    del F, Cm, Ct, Ct2
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sta - st * sa / n
        vt = st2 - st * st / n
        va = sa2 - sa * sa / n
        r = cov / np.sqrt(vt * va)
    r[(n < 2) | ~(vt > 1e-9 * n) | ~(va > 1e-9 * n)] = np.nan
    return r, n


def window_r(image, image_ok, template, template_ok, dy, dx):
    """``(r, n)``: Pearson r of the frame against the delivered window at
    offset (dy, dx), computed directly over the pixels valid in both."""
    h, w = template.shape
    ok = template_ok & image_ok[dy:dy + h, dx:dx + w]
    n = int(ok.sum())
    if n < 2:
        return None, n
    a = image[dy:dy + h, dx:dx + w][ok].astype("float64")
    b = template[ok].astype("float64")
    a -= a.mean()
    b -= b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return (float((a * b).sum()) / den if den > 0 else None), n


def _subpixel(rr, dy, dx):
    """``[delta_y, delta_x]``: where the parabola through ``rr[dy, dx]`` and
    its two neighbours on each axis has its maximum, in pixels from (dy, dx)
    (None on an axis without two finite neighbours; 0.5 for a flat top)."""
    out = []
    for m, p in (((dy - 1, dx), (dy + 1, dx)), ((dy, dx - 1), (dy, dx + 1))):
        inside = all(0 <= a < s for q in (m, p) for a, s in zip(q, rr.shape))
        vm, vp = (float(rr[m]), float(rr[p])) if inside else (np.nan, np.nan)
        c = float(rr[dy, dx])
        if not (np.isfinite(vm) and np.isfinite(vp)):
            out.append(None)
            continue
        den = vm - 2.0 * c + vp
        out.append(0.5 if den >= 0 else round(float(0.5 * (vm - vp) / den), 4))
    return out


def offset_of(image, image_ok, template, template_ok, min_r):
    """Register the frame ``template`` in the delivered ``image``: the best
    offset of ``masked_correlation`` among those whose overlap holds
    REGISTRATION_MIN_OVERLAP of the frame's valid pixels, its r computed
    directly on the frame, the best r outside its peak (REGISTRATION_PEAK_PX)
    and whether the rules pass: r >= ``min_r`` (``r_ok``), a margin of at
    least REGISTRATION_MIN_MARGIN over that second best (``unique``; also
    when every offset lies in the peak) and a CENTRED peak (``centred``: the
    parabolic sub-pixel maximum within REGISTRATION_MAX_SUBPIXEL of the
    offset on each axis that has two neighbours)."""
    nt = int(template_ok.sum())
    if nt < 2:
        return {"offset": None, "pass": False, "r_ok": False, "unique": False, "centred": False,
                "why": "no valid pixel in the frame"}
    r, n = masked_correlation(image, image_ok, template, template_ok)
    rr = np.where(n >= REGISTRATION_MIN_OVERLAP * nt, r, np.nan)
    if not np.isfinite(rr).any():
        return {"offset": None, "pass": False, "r_ok": False, "unique": False, "centred": False,
                "why": (f"no offset where the delivery's valid pixels cover "
                        f"{REGISTRATION_MIN_OVERLAP:.0%} of the frame's")}
    dy, dx = (int(v) for v in np.unravel_index(int(np.nanargmax(rr)), rr.shape))
    best = float(rr[dy, dx])
    p = REGISTRATION_PEAK_PX
    rest = rr.copy()
    rest[max(0, dy - p):dy + p + 1, max(0, dx - p):dx + p + 1] = np.nan
    second = at = None
    if np.isfinite(rest).any():
        sy, sx = (int(v) for v in np.unravel_index(int(np.nanargmax(rest)), rest.shape))
        second, at = float(rest[sy, sx]), [sy, sx]
    sub = _subpixel(rr, dy, dx)
    rx, nx = window_r(image, image_ok, template, template_ok, dy, dx)
    margin = None if second is None else best - second
    r_ok = rx is not None and rx >= min_r
    unique = margin is None or margin >= REGISTRATION_MIN_MARGIN
    centred = all(v is None or abs(v) <= REGISTRATION_MAX_SUBPIXEL for v in sub)
    why = []
    if not r_ok:
        why.append(f"r {'-' if rx is None else f'{rx:.5f}'} < {min_r}")
    if not unique:
        why.append(f"not unique: r {second:.5f} at offset {tuple(at)} is within "
                   f"{REGISTRATION_MIN_MARGIN} of the best {best:.5f}")
    if not centred:
        why.append(f"not centred: the peak's parabolic maximum lies ({sub[0]}, {sub[1]}) px from "
                   f"{(dy, dx)} (> {REGISTRATION_MAX_SUBPIXEL})")
    return {"offset": [dy, dx], "r": None if rx is None else round(rx, 6),
            "r_fft": round(best, 6), "second": None if second is None else round(second, 6),
            "second_at": at, "margin": None if margin is None else round(margin, 6),
            "subpixel": sub, "overlap": nx, "frame_valid": nt, "r_ok": r_ok, "unique": unique,
            "centred": centred, "pass": not why, "why": "; ".join(why)}


def _r_table(per):
    """``{band: {date: r}}`` of per-date, per-band ``offset_of`` results."""
    out = {}
    for d, bands in per.items():
        for b, v in bands.items():
            out.setdefault(b, {})[d] = v.get("r")
    return out


def scl_offset_check(scl, scl_fill, ref, ref_fill, off, frame_shape,
                     px=REGISTRATION_SCL_SEARCH_PX):
    """The delivered SCL of one date (the FULL arrays) against the
    reference's frame on the same date: the share of equal classes over the
    pixels valid in both, at the registered offset ``off`` and at every
    offset within ``px`` pixels of it that keeps the frame inside the arrays.
    Returns ``{"at_offset", "best_other", "best_other_at", "verdict"}``:
    'confirmed' (the registered offset matches best by more than
    REGISTRATION_SCL_TOL), 'contradicted' (another offset matches better by
    more than that: the 20 m layer sits elsewhere) or 'tie' (an SCL without
    structure at that scale)."""
    H, W = (int(v) for v in frame_shape)
    Hr, Wr = scl.shape
    dy, dx = (int(v) for v in off)
    okr = _valid_px(ref, ref_fill)
    oka = _valid_px(scl, scl_fill)
    shares = {}
    for oy in range(dy - px, dy + px + 1):
        for ox in range(dx - px, dx + px + 1):
            if oy < 0 or ox < 0 or oy + H > Hr or ox + W > Wr:
                continue
            both = okr & oka[oy:oy + H, ox:ox + W]
            n = int(both.sum())
            if n:
                shares[(oy, ox)] = float((scl[oy:oy + H, ox:ox + W][both] == ref[both]).sum()) / n
    s0 = shares.get((dy, dx))
    out = {"at_offset": None if s0 is None else round(s0, 6), "best_other": None,
           "best_other_at": None, "verdict": "tie"}
    if s0 is None:
        return dict(out, verdict="no pixel valid in both")
    others = {k: v for k, v in shares.items() if k != (dy, dx)}
    if others:
        k = max(others, key=others.get)
        out.update(best_other=round(others[k], 6), best_other_at=[int(k[0]), int(k[1])])
        if others[k] - s0 > REGISTRATION_SCL_TOL:
            out["verdict"] = "contradicted"
        elif s0 - others[k] > REGISTRATION_SCL_TOL:
            out["verdict"] = "confirmed"
    return out


def check_20m_layer(interp, off, frame_shape, dates):
    """The 20 m layer against the 10 m one INSIDE the delivery (no reference
    needed): on each date of ``dates`` [(date, position)], the delivery's own
    B08 frame at ``off`` (the template) is registered in its full B8A arrays
    (REGISTRATION_20M_BAND, nearest-resampled from 20 m; ``offset_of``, r >=
    REGISTRATION_20M_MIN_R). Returns the record: per date, the dates whose
    peak passes AT ``off`` (``confirmed``) and those whose peak passes
    ELSEWHERE (``contradicted``: the 20 m layer is displaced against the 10 m
    one). A date that does not measure (r too low, a flat peak) is neither."""
    b20, b10 = REGISTRATION_20M_BAND, REGISTRATION_BANDS[0]
    rec = {"band": b20, "against": b10, "checked": False, "confirmed": [], "contradicted": {},
           "per_date": {}}
    if b20 not in interp.bands or b10 not in interp.bands:
        return dict(rec, why=f"{b20} or {b10} not delivered")
    if not dates:
        return dict(rec, why="no usable date")
    H, W = (int(v) for v in frame_shape)
    dy, dx = (int(v) for v in off)
    for d, i in dates:
        img = np.asarray(interp.bands[b20].data[int(i)])
        tpl = np.asarray(interp.bands[b10].data[int(i)])[dy:dy + H, dx:dx + W]
        v = offset_of(img, _valid_px(img, interp.fills.get(b20)), tpl,
                      _valid_px(tpl, interp.fills.get(b10)), REGISTRATION_20M_MIN_R)
        del img, tpl
        rec["per_date"][d] = v
        if v.get("pass") and v["offset"] == [dy, dx]:
            rec["confirmed"].append(d)
        elif v.get("pass"):
            rec["contradicted"][d] = v["offset"]
    return dict(rec, checked=True)


def layer_20m_line(rec):
    """A line on ``check_20m_layer``'s record."""
    if not rec.get("checked"):
        return f"20 m layer: not checked ({rec.get('why')})"
    n = len(rec["per_date"])
    if rec["contradicted"]:
        where = "; ".join(f"{d} at {tuple(o)}" for d, o in rec["contradicted"].items())
        return (f"20 m layer DISPLACED: {rec['band']} against the delivery's own {rec['against']} "
                f"frame peaks elsewhere ({where})")
    if rec["confirmed"]:
        rs = [rec["per_date"][d]["r"] for d in rec["confirmed"]]
        return (f"20 m layer confirmed: {rec['band']} against the delivery's own {rec['against']} "
                f"frame peaks at the offset on {len(rec['confirmed'])} of {n} date(s) (r "
                f"{min(rs):.3f}..{max(rs):.3f})")
    why = "; ".join(f"{d}: {v.get('why') or 'no offset'}"
                    for d, v in list(rec["per_date"].items())[:3])
    return (f"20 m layer: {rec['band']} against {rec['against']} does not measure on {n} date(s) "
            f"({why})")


def phase_check(interp, idx):
    """The 20 m PHASE of the placed frame (REGISTRATION_PHASE_*): per 20 m
    spectral band, the share of valid pixel pairs inside one 20 m cell of
    the UTM lattice holding equal values (``block_equal_counts``) at the
    four phases of that lattice ("00"; "01": x shifted by 10 m; "10": y;
    "11": both), on the dates ``idx`` (positions). A band whose phase (0, 0)
    is beaten by REGISTRATION_PHASE_MIN_GAIN (with at least
    REGISTRATION_PHASE_MIN_PAIRS pairs) is listed in ``contradicted``: its
    20 m cells sit an odd number of pixels away from where they should."""
    x = np.asarray(interp.x, dtype="float64")
    y = np.asarray(interp.y, dtype="float64")
    idx = [int(i) for i in idx]
    dts = zc._dates_of(interp.t)
    out = {"dates": [dts[i] for i in idx], "bands": {}, "contradicted": []}
    for b in REGISTRATION_PHASE_BANDS:
        if b not in interp.bands or not idx:
            continue
        a = np.asarray(interp.bands[b].isel(t=idx).values)
        ok = _valid(a, interp.fills.get(b))
        shares, pairs = {}, 0
        for py, px in ((0, 0), (0, 1), (1, 0), (1, 1)):
            eq, n = block_equal_counts(a, x + 10.0 * px, y - 10.0 * py, ok)
            shares[f"{py}{px}"] = round(eq / n, 4) if n else None
            if (py, px) == (0, 0):
                pairs = n
        del a, ok
        out["bands"][b] = {"shares": shares, "pairs": pairs}
        s0 = shares["00"]
        other = [v for k, v in shares.items() if k != "00" and v is not None]
        if (s0 is not None and other and pairs >= REGISTRATION_PHASE_MIN_PAIRS
                and max(other) - s0 >= REGISTRATION_PHASE_MIN_GAIN):
            out["contradicted"].append(b)
    return out


def scl_phase_check(interp, idx):
    """The 20 m PHASE of a band subset's SCL, its only 20 m band without B8A /
    B11 / B12 (SUBSET_SCL_PHASE_*): the share of valid pixel pairs (SCL
    neither the fill nor 0) inside one 20 m cell of the UTM lattice holding
    the same class, at the four phases of that lattice (as ``phase_check``),
    pooled over the dates ``idx`` (positions). Nearest neighbour from the
    native grid makes phase (0, 0) the best (1.0 on a frame from tiles of its
    own UTM zone). ``verdict``: 'contradicted' when another phase beats (0, 0)
    by more than SUBSET_SCL_PHASE_TOL on at least REGISTRATION_PHASE_MIN_PAIRS
    pairs (the SCL sits an odd number of pixels away), 'confirmed' when (0, 0)
    beats every other phase by at least SUBSET_SCL_PHASE_MIN_GAP, 'not
    confirmed' otherwise ('not checked' without SCL, dates or pairs)."""
    x = np.asarray(interp.x, dtype="float64")
    y = np.asarray(interp.y, dtype="float64")
    idx = [int(i) for i in idx]
    dts = zc._dates_of(interp.t)
    out = {"band": "SCL", "dates": [dts[i] for i in idx], "shares": {}, "pairs": 0,
           "verdict": "not checked", "best_other": None, "tol": SUBSET_SCL_PHASE_TOL,
           "min_gap": SUBSET_SCL_PHASE_MIN_GAP}
    if "SCL" not in interp.bands or not idx:
        return dict(out, why="no SCL or no date")
    a = np.asarray(interp.bands["SCL"].isel(t=idx).values)
    ok = _valid_px(a, interp.fills.get("SCL"))
    shares, pairs = {}, 0
    for py, px in ((0, 0), (0, 1), (1, 0), (1, 1)):
        eq, n = block_equal_counts(a, x + 10.0 * px, y - 10.0 * py, ok)
        shares[f"{py}{px}"] = round(eq / n, 6) if n else None
        if (py, px) == (0, 0):
            pairs = n
    del a, ok
    out.update(shares=shares, pairs=int(pairs))
    s0 = shares["00"]
    other = {k: v for k, v in shares.items() if k != "00" and v is not None}
    if s0 is None or not other or pairs < REGISTRATION_PHASE_MIN_PAIRS:
        return dict(out, why=f"{pairs} valid pair(s) at phase 00 (at least "
                             f"{REGISTRATION_PHASE_MIN_PAIRS} needed)")
    best = max(other, key=other.get)
    out["best_other"] = best
    if other[best] - s0 > SUBSET_SCL_PHASE_TOL:
        return dict(out, verdict="contradicted")
    if s0 - other[best] >= SUBSET_SCL_PHASE_MIN_GAP:
        return dict(out, verdict="confirmed")
    return dict(out, verdict="not confirmed",
                why=f"phase 00 beats phase {best} by {s0 - other[best]:.4f} (< "
                    f"{SUBSET_SCL_PHASE_MIN_GAP})")


def scl_phase_line(chk):
    """A line on ``scl_phase_check``'s record."""
    shares = " ".join(f"{k} {'-' if v is None else f'{v:.4f}'}" for k, v in chk["shares"].items())
    head = {"contradicted": "SCL 20 m phase CONTRADICTED", "confirmed": "SCL 20 m phase confirmed",
            "not confirmed": "SCL 20 m phase NOT confirmed"}.get(chk["verdict"],
                                                                 "SCL 20 m phase: not checked")
    return (f"{head} (band subset without B8A / B11 / B12): share of equal SCL pairs inside one "
            f"20 m cell of the lattice at the phases 00 (the frame's) / 01 / 10 / 11 on "
            f"{chk['dates']}: {shares or '-'} ({chk['pairs']} pairs at 00)"
            + (f"; {chk['why']}" if chk.get("why") else ""))


def phase_line(chk):
    """A line on ``phase_check``'s record."""
    if not chk["bands"]:
        return "20 m phase: not checked (no 20 m spectral band or no date)"
    parts = "; ".join(f"{b} " + " ".join(f"{k} {'-' if v is None else f'{v:.3f}'}"
                                         for k, v in s["shares"].items())
                      for b, s in chk["bands"].items())
    head = ("20 m phase CONTRADICTED for " + ", ".join(chk["contradicted"]) if chk["contradicted"]
            else "20 m phase check passed")
    return (f"{head}: share of equal pixel pairs inside one 20 m cell of the lattice at the phases "
            f"00 (the frame's) / 01 / 10 / 11 on {chk['dates']}: {parts}")


def median_composite(band, idx, fill, rows=None, cols=None, block=REGISTRATION_COMPOSITE_ROWS):
    """The per-pixel MEDIAN (float32; NaN where no date has data) of the (t,
    y, x) ``band`` (DataArray, dask or numpy) over the dates ``idx``
    (positions), in the window ``rows`` x ``cols`` (default: everything),
    computed in blocks of ``block`` rows that read the dates together (a
    chunk is decoded once per block). Invalid values (the fill, 0, NaN) are
    left out of the median."""
    import warnings

    data = band.data if hasattr(band, "dims") else band
    _, Hr, Wr = (int(v) for v in data.shape)
    r0, r1 = (0, Hr) if rows is None else (int(rows.start), int(rows.stop))
    c0, c1 = (0, Wr) if cols is None else (int(cols.start), int(cols.stop))
    idx = sorted(int(i) for i in idx)
    out = np.full((r1 - r0, c1 - c0), np.nan, "float32")
    for a in range(r0, r1, int(block)):
        b = min(r1, a + int(block))
        raw = np.asarray(data[idx, a:b, c0:c1])
        stack = raw.astype("float32")
        stack[~_valid_px(raw, fill)] = np.nan
        del raw
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # pixels without any date
            out[a - r0:b - r0] = np.nanmedian(stack, axis=0)
        del stack
    return out


def register_with_netcdf(interp, frame, plan, log=print):
    """Measure the offset of ``frame`` (its axes ``x``, ``y``) in the
    delivered arrays against the reference netCDF ``plan.reference``.

    The common dates (within the period the reference was requested for)
    whose DELIVERED SCL is usable (``usable_dates``, counted in the
    always-inside core of the arrays, ``core_window``) are tried clearest
    first, at most REGISTRATION_MAX_TRIED, until REGISTRATION_DATES COUNT: per
    date, ``offset_of`` of the reference's frame in the full delivered array
    for every band of REGISTRATION_BANDS (B08 measures, B03 confirms), r >=
    REGISTRATION_MIN_R_ANY. A date counts when every band passes; a date that
    does not measure (its reference frame without data, r too low, a peak
    not unique or not centred) is SET ASIDE with the reason. Passed when no
    two passing results disagree on the offset, at least
    REGISTRATION_MIN_DATES dates count, at least REGISTRATION_MIN_DATES of
    them reach B08 r >= REGISTRATION_MIN_R, and the delivered SCL (20 m) does
    not match the reference's better at another offset within
    REGISTRATION_SCL_SEARCH_PX (``scl_offset_check``) on any counted date.
    Returns the record (``measured``: the offsets of the passing results;
    ``positions``: the counted dates' positions in the delivery;
    ``usable_positions``: the clearest usable ones)."""
    import xarray as xr

    fx = np.asarray(frame["x"], dtype="float64")
    fy = np.asarray(frame["y"], dtype="float64")
    H, W = len(fy), len(fx)
    _, Hr, Wr = (int(v) for v in next(iter(interp.bands.values())).shape)
    name = Path(plan.reference).name
    b0 = REGISTRATION_BANDS[0]
    rec = {"reference": str(plan.reference), "passed": False, "offset": None, "measured": [],
           "dates": [], "positions": [], "usable_positions": []}
    with xr.open_dataset(plan.reference, mask_and_scale=False) as r:
        tdim = zc._time_dim(r)
        ox = zc._overlap(fx, np.asarray(r["x"].values))
        oy = zc._overlap(fy, np.asarray(r["y"].values))
        if ox is None or oy is None or tuple(ox[:2]) != (0, W) or tuple(oy[:2]) != (0, H):
            return dict(rec, why=f"the frame ({H} x {W} px) is not inside the grid of {name}")
        c0, r0 = int(ox[2]), int(oy[2])
        missing = [b for b in REGISTRATION_BANDS + ("SCL",) if b not in interp.bands
                   or (b != "SCL" and b not in r.data_vars)]
        if missing:
            return dict(rec, why=f"bands {missing} missing from the delivery or from {name}")
        p0, p1 = plan.reference_period or ("0000", "9999")
        rpos = {}
        for j, d in enumerate(zc._dates_of(r[tdim].values)):
            if p0 <= d < p1:
                rpos.setdefault(d, j)
        common = [(d, i, rpos[d]) for i, d in enumerate(zc._dates_of(interp.t)) if d in rpos]
        rec["common_dates"] = len(common)
        if not common:
            return dict(rec, why=f"no date in common with {name}")
        rows, cols = core_window((Hr, Wr), (H, W))
        nv, nc = clear_counts(interp.bands["SCL"], [i for _, i, _ in common],
                              interp.fills.get("SCL"), rows, cols)
        cands, n_ok = usable_dates([d for d, _, _ in common], nv, nc,
                                   _window_px(rows, cols, (H, W)), n=None)
        rec.update(usable_dates=n_ok,
                   clear_window=[[rows.start, rows.stop], [cols.start, cols.stop]],
                   usable_positions=[[d, int(common[k][1])] for d, k in cands[:REGISTRATION_DATES]])
        if n_ok < REGISTRATION_MIN_DATES:
            return dict(rec, why=(
                f"{n_ok} usable date(s) among the {len(common)} in common with {name} (delivered "
                f"SCL clear on >= {REGISTRATION_MIN_CLEAR:.0%} of its valid pixels, valid on >= "
                f"{REGISTRATION_MIN_COVER:.0%} of the always-inside core of the arrays, rows "
                f"{rows.start}:{rows.stop} cols {cols.start}:{cols.stop}); {REGISTRATION_MIN_DATES} "
                f"needed: a cloudy delivery is refused, not guessed"))

        def frame_of(b, j):
            return np.asarray(r[b].isel({tdim: j, "y": slice(r0, r0 + H),
                                         "x": slice(c0, c0 + W)}).values)

        per, counted, aside, tried = {}, [], {}, 0
        for d, k in cands:
            if len(counted) >= REGISTRATION_DATES or tried >= REGISTRATION_MAX_TRIED:
                break
            _, i, j = common[k]
            tpls = {b: frame_of(b, j) for b in REGISTRATION_BANDS}
            cover = min(float(_valid_px(t, zc.fill_of(r[b])).mean()) for b, t in tpls.items())
            if cover < REGISTRATION_MIN_COVER:
                aside[d] = (f"the reference frame holds data on {cover:.0%} of its pixels (< "
                            f"{REGISTRATION_MIN_COVER:.0%})")
                continue
            tried += 1
            per[d] = {}
            for b in REGISTRATION_BANDS:
                img = np.asarray(interp.bands[b].data[i])
                per[d][b] = offset_of(img, _valid_px(img, interp.fills.get(b)), tpls[b],
                                      _valid_px(tpls[b], zc.fill_of(r[b])), REGISTRATION_MIN_R_ANY)
                del img
            del tpls
            bad = [f"{b} {v.get('why') or 'no offset'}" for b, v in per[d].items()
                   if not v.get("pass")]
            if bad:
                aside[d] = "; ".join(bad)
            else:
                counted.append((d, k))
        measured = sorted({tuple(v["offset"]) for res in per.values() for v in res.values()
                           if v.get("pass")})
        rec.update(dates=[d for d, _ in counted],
                   positions=[[d, int(common[k][1])] for d, k in counted],
                   clear={d: [int(nv[k]), int(nc[k])] for d, k in counted}, dates_tried=tried,
                   set_aside=aside, per_date=per, r=_r_table({d: per[d] for d, _ in counted}),
                   measured=[list(o) for o in measured])
        if len(measured) > 1:
            listed = "; ".join(
                f"{d} " + ", ".join(f"{b} {tuple(v['offset']) if v.get('offset') else '-'}"
                                    for b, v in res.items())
                for d, res in per.items())
            return dict(rec, why=f"the dates / bands disagree on the offset ({listed})")
        if len(counted) < REGISTRATION_MIN_DATES:
            return dict(rec, why=(
                f"{len(counted)} of the {tried} date(s) tried measure the offset "
                f"({REGISTRATION_MIN_DATES} needed; the usable dates are tried clearest first, at "
                f"most {REGISTRATION_MAX_TRIED}): "
                + "; ".join(f"{d}: {w}" for d, w in aside.items())))
        off = list(measured[0])
        strong = [d for d, _ in counted if (per[d][b0].get("r") or 0.0) >= REGISTRATION_MIN_R]
        rec["dates_min_r"] = strong
        if len(strong) < REGISTRATION_MIN_DATES:
            return dict(rec, why=(
                f"the {len(counted)} counted dates agree on {tuple(off)}, but only {len(strong)} "
                f"reach {b0} r >= {REGISTRATION_MIN_R} ({REGISTRATION_MIN_DATES} needed): "
                + ", ".join(f"{d} r {per[d][b0].get('r')}" for d, _ in counted)))
        if "SCL" in r.data_vars:
            scl = {}
            for d, k in counted:
                _, i, j = common[k]
                scl[d] = scl_offset_check(np.asarray(interp.bands["SCL"].data[i]),
                                          interp.fills.get("SCL"), frame_of("SCL", j),
                                          zc.fill_of(r["SCL"]), off, (H, W))
            rec.update(scl=scl, scl_equal={d: v["at_offset"] for d, v in scl.items()},
                       scl_confirmed=[d for d, v in scl.items() if v["verdict"] == "confirmed"])
            bad = {d: v for d, v in scl.items() if v["verdict"] == "contradicted"}
            if bad:
                return dict(rec, why=(
                    f"the delivered SCL (20 m) matches the reference's better elsewhere than at "
                    f"{tuple(off)}: " + "; ".join(
                        f"{d} equal on {v['at_offset']:.4f} at the offset, {v['best_other']:.4f} at "
                        f"{tuple(v['best_other_at'])}" for d, v in bad.items())
                    + " (the 20 m layer is displaced against the 10 m one)"))
    return dict(rec, passed=True, offset=off, why="")


def _json_attr(v):
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v) if v else None
    except (TypeError, ValueError):
        return None


def backend_version(results_meta):
    """The backend's version in a job's results metadata ('' when absent)."""
    m = results_meta or {}
    return str(m.get("openeo:version", "") or (m.get("properties") or {}).get("openeo:version", ""))


def input_tiles(meta):
    """The MGRS tiles of the Sentinel-2 L2A products a job read, sorted, from
    the product names in its results metadata (recorded with the
    registration: a year read from other tiles may be laid out otherwise)."""
    blob = json.dumps(meta or {}, default=str)
    return sorted(set(re.findall(r"S2[A-D]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T(\w{5})_\d{8}T\d{6}",
                                 blob)))


def sibling_candidates(req, frame, raw_shape, version, root, same_request=True):
    """Complete main stores of ``req.site`` under ``root`` (other years)
    whose own registration against a reference netCDF passed (``registration``
    attribute: method "netcdf", verified), with the same requested extent and
    the same grid (the frame's axes); ``same_request`` (TAKING a sibling's
    offset): also the same delivered array shape. Without it (CHECKING an
    --offset-px, or a frame-sized delivery) any delivered shape: the
    composites search the frame in THIS delivery's arrays. The backend
    version (``version``) is recorded by the caller, never compared: it
    changes every week or two (0.73.0a16.dev20260805 ... 0.73.0a17.
    dev20260915 in the legacy cubes) and only the correlation tells whether
    its layout changed. Returns ``(candidates nearest year first, [(year, why
    skipped)])``."""
    out, skipped = [], []
    want = _json_attr(json.dumps(req.spatial_extent))
    fx = np.asarray(frame["x"], dtype="float64")
    fy = np.asarray(frame["y"], dtype="float64")
    for year in zc.list_years(req.site, root):
        if year == req.year:
            continue
        path = zc.store_path(req.site, year, root)
        attrs = zc.store_attrs(path)
        rec = _json_attr(attrs.get("registration"))
        why = None
        if not isinstance(rec, dict) or rec.get("method") != "netcdf" or not rec.get("verified"):
            why = "not registered against a reference netCDF"
        elif _json_attr(attrs.get("spatial_extent")) != want:
            why = f"another requested extent ({attrs.get('spatial_extent')})"
        elif same_request and list(rec.get("delivered_shape") or []) != list(raw_shape):
            why = (f"its delivery was {rec.get('delivered_shape')} px, this one "
                   f"{list(raw_shape)}")
        else:
            ds = zc.open_store(path, bands=["SCL"])
            try:
                same = (np.array_equal(np.asarray(ds["x"].values, dtype="float64"), fx)
                        and np.array_equal(np.asarray(ds["y"].values, dtype="float64"), fy))
            finally:
                ds.close()
            if not same:
                why = "another grid"
        if why:
            skipped.append((year, why))
        else:
            out.append({"year": year, "path": str(path), "offset": list(rec["offset"]),
                        "delivered_shape": list(rec.get("delivered_shape") or []),
                        "backend_version": str(attrs.get("backend_version") or "")})
    out.sort(key=lambda c: (abs(c["year"] - req.year), c["year"]))
    return out, skipped


def recorded_delivered_shapes(req, root):
    """``{(rows, cols): [years]}``: the delivered array shapes recorded by the
    complete main stores of ``req.site`` under ``root`` (other years) with
    the same requested extent."""
    out = {}
    if root is None:
        return out
    want = _json_attr(json.dumps(req.spatial_extent))
    for year in zc.list_years(req.site, root):
        if year == req.year:
            continue
        attrs = zc.store_attrs(zc.store_path(req.site, year, root))
        shp = _json_attr(attrs.get("delivered_shape_px"))
        if (_json_attr(attrs.get("spatial_extent")) == want and isinstance(shp, list)
                and len(shp) == 2):
            out.setdefault((int(shp[0]), int(shp[1])), []).append(year)
    return out


def register_with_sibling(interp, frame, cand, log=print, want=None, pad=0):
    """Check an offset (``want``; default the sibling's own) on this delivery
    against a registered store of the site (``cand``: another year,
    registered against its reference netCDF): the per-pixel MEDIAN
    COMPOSITES of the REGISTRATION_SIBLING_DATES clearest usable dates of
    this delivery (its full arrays; dates chosen in the always-inside core)
    and of the sibling store (its frame), per band of REGISTRATION_BANDS,
    registered with ``offset_of`` (the peak rules; r >=
    REGISTRATION_SIBLING_MIN_R). ``pad``: the delivery's composite gets an
    invalid margin of ``pad`` pixels (a frame-sized delivery, searched within
    ``pad`` px of (0, 0)); the offsets reported are those of the unpadded
    arrays. Returns the record; ``outcome``: 'confirmed' (every band passes
    at ``want``), 'contradicted' (a band passes ELSEWHERE: ``measured``) or
    'unmeasured' (``why``); ``positions``: this delivery's clearest dates
    (for the 20 m check)."""
    H, W = len(frame["y"]), len(frame["x"])
    want = [int(v) for v in (cand["offset"] if want is None else want)]
    rec = {"reference": cand["path"], "sibling_year": cand["year"],
           "sibling_offset": cand["offset"], "want": want, "pad": int(pad), "passed": False,
           "offset": None, "outcome": "unmeasured", "measured": None, "dates": [],
           "sibling_dates": [], "positions": []}
    if "SCL" not in interp.bands or any(b not in interp.bands for b in REGISTRATION_BANDS):
        return dict(rec, why="the delivery lacks SCL or a registration band")
    _, Hr, Wr = (int(v) for v in next(iter(interp.bands.values())).shape)
    rows, cols = core_window((Hr, Wr), (H, W))
    mine_dates = zc._dates_of(interp.t)
    nv, nc = clear_counts(interp.bands["SCL"], range(len(mine_dates)), interp.fills.get("SCL"),
                          rows, cols)
    mine, n_mine = usable_dates(mine_dates, nv, nc, _window_px(rows, cols, (H, W)),
                                n=REGISTRATION_SIBLING_DATES)
    ds = zc.open_store(cand["path"], bands=list(REGISTRATION_BANDS) + ["SCL"])
    try:
        sib_dates = zc.dates(ds)
        sv, sc = clear_counts(ds["SCL"], range(len(sib_dates)), zc.fill_of(ds["SCL"]))
        theirs, n_theirs = usable_dates(sib_dates, sv, sc, H * W, n=REGISTRATION_SIBLING_DATES)
        rec.update(usable_dates=n_mine, sibling_usable_dates=n_theirs,
                   dates=[d for d, _ in mine], sibling_dates=[d for d, _ in theirs],
                   positions=[[d, int(i)] for d, i in mine[:REGISTRATION_DATES]])
        if len(mine) < REGISTRATION_MIN_DATES or len(theirs) < REGISTRATION_MIN_DATES:
            return dict(rec, why=(f"{len(mine)} usable date(s) in this delivery and {len(theirs)} "
                                  f"in the {cand['year']} store ({REGISTRATION_MIN_DATES} needed in "
                                  f"each)"))
        per = {}
        for b in REGISTRATION_BANDS:
            img = median_composite(interp.bands[b], [i for _, i in mine], interp.fills.get(b))
            tpl = median_composite(ds[b], [j for _, j in theirs], zc.fill_of(ds[b]))
            ok_i, ok_t = np.isfinite(img), np.isfinite(tpl)
            img, tpl = np.nan_to_num(img), np.nan_to_num(tpl)
            if pad:
                img, ok_i = np.pad(img, int(pad)), np.pad(ok_i, int(pad))
            v = offset_of(img, ok_i, tpl, ok_t, REGISTRATION_SIBLING_MIN_R)
            del img, tpl, ok_i, ok_t
            if pad:
                for k in ("offset", "second_at"):
                    if v.get(k) is not None:
                        v[k] = [v[k][0] - int(pad), v[k][1] - int(pad)]
            per[b] = v
    finally:
        ds.close()
    rec["per_band"] = per
    rec["r"] = {b: {"composite": v.get("r")} for b, v in per.items()}
    passing = {b: tuple(v["offset"]) for b, v in per.items() if v.get("pass")}
    if len(passing) == len(per) and all(o == tuple(want) for o in passing.values()):
        return dict(rec, passed=True, offset=want, outcome="confirmed", why="")
    other = {b: o for b, o in passing.items() if o != tuple(want)}
    if other:
        return dict(rec, outcome="contradicted", measured=list(next(iter(other.values()))),
                    why="; ".join(f"the {b} composites peak at {o} (r {per[b]['r']}, margin "
                                  f"{per[b]['margin']})" for b, o in other.items()))
    return dict(rec, why="; ".join(f"{b}: {v.get('why') or 'no offset'}"
                                   + (f" (peak at {tuple(v['offset'])})" if v.get("offset") else "")
                                   for b, v in per.items() if not v.get("pass")))


def _composite_line(got):
    """``B08 r 0.97 (margin 0.08), B03 ...`` of a sibling record."""
    return ", ".join(f"{b} r {v.get('r')} (margin {v.get('margin')})"
                     for b, v in (got.get("per_band") or {}).items())


def _frame_of(target, interp):
    if target is not None:
        return target
    return {"x": interp.x, "y": interp.y, "path": "the delivered grid"}


class _Refused(Exception):
    """Internal: a registration rule says no (``register_delivery``)."""


def register_delivery(interp, req, target, plan, results_meta=None, log=print):
    """Where the frame sits in the delivered (padded) arrays: the record kept
    in the store's attributes (``offset`` [rows, cols] of the frame's first
    pixel, ``method``, the dates and r values, the reference used, the 20 m
    layer's check). The frame is ``target`` (``placement_target``), or the
    delivered grid itself when its extent is the frame. The rules (module
    docstring):

    * arrays smaller than the frame: no offset puts the frame inside them;
      the coordinates place the pixels (method "none"; the padding rules of
      ``put_on_target`` apply);
    * arrays of the frame's shape whose x/y put the frame at (0, 0): the only
      offset there is (method "frame-sized"); in a year without reference it
      is checked within REGISTRATION_FRAME_SIZED_PX of (0, 0) against a
      registered store when there is one (refused when the composites peak
      elsewhere);
    * otherwise the offset must be established: against the reference netCDF
      covering the window (``register_with_netcdf``; method "netcdf";
      --offset-px must agree with it, and stands unconfirmed when the
      registration cannot measure without contradicting it, the verification
      after writing deciding); without one, --offset-px checked against a
      registered store of the site when there is one (method "expert
      --offset-px": refused when the composites peak elsewhere; unconfirmed
      when they cannot measure; refused, unless --accept-unverified-offset,
      when nothing checks it and the delivered shape is not the one the
      site's stores were delivered in), else the offset of a registered store
      of the same request confirmed by the composites (method "sibling
      store");
    * with an offset established, the 20 m layer must not contradict it
      (``check_20m_layer``; against the reference also ``scl_offset_check``);
      unconfirmed, it is a warning.

    What the rules do not establish raises RegistrationRefused (production,
    nothing written) or, for --test, is a warning and the coordinates place
    the pixels (offset None)."""
    first = next(iter(interp.bands.values()))
    _, Hr, Wr = (int(v) for v in first.shape)
    frame = _frame_of(target, interp)
    fx = np.asarray(frame["x"], dtype="float64")
    fy = np.asarray(frame["y"], dtype="float64")
    H, W = len(fy), len(fx)
    res = float(req.spec.resolution)
    cy = int(round((float(interp.y[0]) - float(fy[0])) / res)) if len(interp.y) and H else 0
    cx = int(round((float(fx[0]) - float(interp.x[0])) / res)) if len(interp.x) and W else 0
    rec = {"method": None, "offset": None, "verified": False, "checked": None,
           "delivered_shape": [Hr, Wr], "frame_shape": [H, W],
           "frame_first_centre": [float(fx[0]), float(fy[0])], "coordinate_offset": [cy, cx],
           "offsets_searched": [max(0, Hr - H + 1), max(0, Wr - W + 1)],
           "expert": list(plan.expert) if plan.expert else None, "reference": None,
           "backend_version": backend_version(results_meta), "input_tiles": input_tiles(results_meta),
           "positions": [], "rule": registration_rule(), "utc": utcnow().isoformat(timespec="seconds")}
    hint = (f" (the measurements are in {req.sidecar('format.txt').name}; --offset-px ROWS,COLS "
            f"gives the offset by hand, for one site: recorded, and checked against the reference "
            f"netCDF or a registered store of the site when there is one)")

    def done(method, offset, verified, why):
        rec.update(method=method, offset=None if offset is None else [int(v) for v in offset],
                   verified=bool(verified), why=why)
        o = rec["offset"]
        log(f"registration ({method}): " + (
            f"the frame ({H} x {W} px) starts at row {o[0]}, col {o[1]} of the delivered {Hr} x "
            f"{Wr} px arrays (their x/y put it at row {cy}, col {cx}); {why}" if o is not None
            else why))
        return rec

    def refuse(why):
        done(rec["method"] or "none", None, False, why)
        if plan.strict:
            raise RegistrationRefused(
                f"{req.tag}: the place of the frame ({H} x {W} px) in the delivered {Hr} x {Wr} px "
                f"arrays is not established: {why}; nothing written" + hint)
        interp.warnings.append(f"NOT REGISTERED: {why}; the delivered x/y place the pixels "
                               f"(--test only; a production year is refused)")
        return rec

    def clear_positions():
        """The delivery's clearest usable dates [(date, position)] (the 20 m check)."""
        if "SCL" not in interp.bands:
            return []
        rows, cols = core_window((Hr, Wr), (H, W))
        dts = zc._dates_of(interp.t)
        nv, nc = clear_counts(interp.bands["SCL"], range(len(dts)), interp.fills.get("SCL"),
                              rows, cols)
        return usable_dates(dts, nv, nc, _window_px(rows, cols, (H, W)))[0]

    def layer(off, positions, confirmed_otherwise=False):
        """The 20 m check at ``off``: refuses on a contradiction, warns when
        nothing confirms the 20 m layer; returns its line."""
        positions = [(str(d), int(i)) for d, i in positions]
        rec["positions"] = [[d, i] for d, i in positions]
        chk = check_20m_layer(interp, off, (H, W), positions)
        rec["layer_20m"] = chk
        line = layer_20m_line(chk)
        if chk["contradicted"]:
            raise _Refused(line)
        if (not chk["confirmed"] and not confirmed_otherwise and req.is_subset
                and REGISTRATION_20M_BAND not in interp.bands):
            # a band subset without B8A: its SCL is checked at the 20 m phases (phase_gate)
            interp.notes.append(f"{line}; this band subset's 20 m layer (SCL) is checked at the "
                                f"phases of the 20 m lattice after the placement")
        elif not chk["confirmed"] and not confirmed_otherwise:
            interp.warnings.append(f"the 20 m layer is not confirmed at the offset {tuple(off)} ("
                                   f"{line}; nothing else checks it in this year)")
        return line

    def siblings(same_request):
        if plan.root is None:
            return [], []
        cands, skipped = sibling_candidates(req, frame, (Hr, Wr), rec["backend_version"],
                                            plan.root, same_request=same_request)
        for extra in plan.sibling_roots:      # a band subset: the nine-band stores, read only
            more, why = sibling_candidates(req, frame, (Hr, Wr), rec["backend_version"],
                                           extra, same_request=same_request)
            cands += more
            skipped += [(y, f"{w} (under {extra})") for y, w in why]
        if plan.sibling_roots:
            cands.sort(key=lambda c: (abs(c["year"] - req.year), c["year"]))
        rec["siblings_skipped"] = [[y, w] for y, w in skipped]
        return cands, skipped

    try:
        if Hr < H or Wr < W:
            return done("none", None, False,
                        f"the delivered arrays ({Hr} x {Wr} px) are smaller than the frame ({H} x "
                        f"{W} px): no offset puts the frame inside them; the coordinates place the "
                        f"pixels and the padding rules apply")
        ex = plan.expert
        if ex is not None and not (0 <= ex[0] <= Hr - H and 0 <= ex[1] <= Wr - W):
            raise _Refused(f"--offset-px {ex[0]},{ex[1]} puts the frame outside the delivered "
                           f"arrays (offsets 0..{Hr - H} x 0..{Wr - W})")
        if (Hr, Wr) == (H, W) and (cy, cx) == (0, 0) and ex in (None, (0, 0)):
            why = ("the delivered arrays have the frame's shape and their x/y put the frame at "
                   "(0, 0): there is no other offset")
            cands = [] if plan.reference else siblings(False)[0]
            if cands:
                cand, pad = cands[0], REGISTRATION_FRAME_SIZED_PX
                got = register_with_sibling(interp, frame, cand, log, want=(0, 0), pad=pad)
                rec.update(sibling=got, reference=cand["path"])
                if got["outcome"] == "contradicted":
                    m = got["measured"]
                    raise _Refused(f"a delivery of the frame's shape whose composites against the "
                                   f"registered {cand['year']} store peak at ({m[0]}, {m[1]}), not "
                                   f"(0, 0) (searched within {pad} px): {got['why']}")
                if got["outcome"] == "confirmed":
                    rec["checked"] = (f"(0, 0) confirmed within {pad} px by the median composites "
                                      f"against the registered {cand['year']} store")
                    why += f"; {rec['checked']} ({_composite_line(got)})"
                else:
                    why += (f"; not confirmed by the composites against the registered "
                            f"{cand['year']} store ({got['why']})")
            return done("frame-sized", (0, 0), False, why)
        if plan.reference:
            name = Path(plan.reference).name
            rec["reference"] = plan.reference
            got = register_with_netcdf(interp, frame, plan, log)
            rec["netcdf"] = got
            rec["dates"], rec["r"] = got.get("dates"), got.get("r")
            if got.get("passed"):
                off = got["offset"]
                if ex is not None and list(ex) != list(off):
                    raise _Refused(f"--offset-px {ex[0]},{ex[1]} contradicts the registration "
                                   f"against {name}: ({off[0]}, {off[1]})")
                line = layer(off, got["positions"],
                             confirmed_otherwise=bool(got.get("scl_confirmed")))
                rs = ", ".join(f"{b} r >= {min(v for v in t.values() if v is not None):.5f}"
                               for b, t in (got.get("r") or {}).items())
                aside = "; ".join(f"{d}: {w}" for d, w in (got.get("set_aside") or {}).items())
                scl = (f"SCL best at the offset on {len(got.get('scl_confirmed') or [])} of "
                       f"{len(got.get('scl') or {})} date(s)" if "scl" in got else "no reference SCL")
                return done("netcdf", off, True,
                            f"registered against {name} on {len(got['dates'])} clear common dates "
                            f"{got['dates']}: {rs}"
                            + (f"; {len(got['set_aside'])} date(s) set aside ({aside[:600]})"
                               if aside else "")
                            + f"; {scl}; {line}"
                            + (f"; --offset-px {ex[0]},{ex[1]} confirmed" if ex is not None else ""))
            if ex is not None:
                other = [tuple(o) for o in got.get("measured") or [] if list(o) != list(ex)]
                if other:
                    raise _Refused(f"--offset-px {ex[0]},{ex[1]} contradicts the registration "
                                   f"against {name}, whose passing dates put the frame at {other} "
                                   f"({got.get('why')})")
                line = layer(ex, got.get("usable_positions") or [])
                interp.warnings.append(
                    f"--offset-px {ex[0]},{ex[1]} NOT confirmed by the registration against {name} "
                    f"({got.get('why')}): the verification after writing decides")
                return done("expert --offset-px", ex, False,
                            f"--offset-px, not confirmed by the registration against {name}: "
                            f"{got.get('why')}; {line}")
            raise _Refused(f"the registration against {name}: {got.get('why')}")
        if ex is not None:
            cands, skipped = siblings(False)
            if cands:
                cand = cands[0]
                rec["reference"] = cand["path"]
                got = register_with_sibling(interp, frame, cand, log, want=ex)
                rec["sibling"] = got
                rec["dates"], rec["r"] = got.get("dates"), got.get("r")
                if got["outcome"] == "contradicted":
                    m = got["measured"]
                    raise _Refused(f"--offset-px {ex[0]},{ex[1]} contradicts the median composites "
                                   f"against the registered {cand['year']} store, which put the "
                                   f"frame at ({m[0]}, {m[1]}): {got['why']}")
                same_shape = list(cand.get("delivered_shape") or []) == [Hr, Wr]
                if got["outcome"] == "confirmed":
                    if plan.require_confirmed and not same_shape and not plan.accept_unverified:
                        raise _Refused(
                            f"--offset-px {ex[0]},{ex[1]} is confirmed by the median composites "
                            f"against the registered {cand['year']} store, but this delivery's "
                            f"arrays ({Hr} x {Wr} px) are not the shape that store was delivered "
                            f"in ({cand.get('delivered_shape')}): a band subset takes an offset "
                            f"only from a delivery of the same layout (--accept-unverified-offset "
                            f"takes it anyway, recorded)")
                    line = layer(ex, got["positions"])
                    rec["checked"] = (f"confirmed by the median composites against the registered "
                                      f"{cand['year']} store"
                                      + ("" if same_shape or not plan.require_confirmed else
                                         "; delivered in another array shape: "
                                         "--accept-unverified-offset"))
                    return done("expert --offset-px", ex, False,
                                f"--offset-px {ex[0]},{ex[1]}: confirmed by the median composites "
                                f"of {len(got['dates'])} clear dates of this delivery and "
                                f"{len(got['sibling_dates'])} of the registered {cand['year']} "
                                f"store ({_composite_line(got)}); not verified against a reference "
                                f"netCDF ({plan.no_reference}); {line}")
                if plan.require_confirmed and not plan.accept_unverified:
                    raise _Refused(f"--offset-px {ex[0]},{ex[1]} is not confirmed by the median "
                                   f"composites against the registered {cand['year']} store "
                                   f"({got['why']}): a band subset takes an offset only when they "
                                   f"confirm it (--accept-unverified-offset takes it anyway, "
                                   f"recorded)")
                if not same_shape and not plan.accept_unverified:
                    raise _Refused(f"--offset-px {ex[0]},{ex[1]} cannot be checked against the "
                                   f"registered {cand['year']} store ({got['why']}) and this "
                                   f"delivery's arrays ({Hr} x {Wr} px) are not that store's "
                                   f"({cand.get('delivered_shape')}): the backend's layout may have "
                                   f"changed, and its offset with it (--accept-unverified-offset "
                                   f"takes it anyway, recorded)")
                line = layer(ex, got.get("positions") or clear_positions())
                rec["checked"] = (f"NOT confirmed by the composites against the registered "
                                  f"{cand['year']} store ({got['why']})"
                                  + ("" if same_shape else "; --accept-unverified-offset"))
                interp.warnings.append(f"--offset-px {ex[0]},{ex[1]} {rec['checked']}: taken "
                                       f"unverified")
                return done("expert --offset-px", ex, False,
                            f"--offset-px, {rec['checked']}; not verified ({plan.no_reference}); "
                            f"{line}")
            if plan.require_confirmed and not plan.accept_unverified:
                raise _Refused(f"--offset-px {ex[0]},{ex[1]} cannot be checked: no store of "
                               f"{req.site} registered against a reference netCDF with this extent "
                               f"and grid under {[str(r) for r in (plan.root,) + plan.sibling_roots]}"
                               + (f" ({'; '.join(f'{y}: {w}' for y, w in skipped)})" if skipped
                                  else "")
                               + ": a band subset takes an offset only when the median composites "
                                 "of such a store confirm it (--accept-unverified-offset takes it "
                                 "anyway, recorded)")
            shapes = recorded_delivered_shapes(req, plan.root)
            for extra in plan.sibling_roots:
                for s_, ys_ in recorded_delivered_shapes(req, extra).items():
                    shapes.setdefault(s_, []).extend(ys_)
            if shapes and (Hr, Wr) not in shapes and not plan.accept_unverified:
                raise _Refused(f"--offset-px {ex[0]},{ex[1]} cannot be checked (no store of "
                               f"{req.site} registered against a reference netCDF with this extent "
                               f"and grid) and this delivery's arrays ({Hr} x {Wr} px) are not "
                               f"those the site's stores were delivered in ("
                               + "; ".join(f"{list(s)}: {y}" for s, y in shapes.items())
                               + "): the backend's layout may have changed, and its offset with it "
                                 "(--accept-unverified-offset takes it anyway, recorded)")
            line = layer(ex, clear_positions())
            rec["checked"] = ("nothing to check it against"
                              + ("; --accept-unverified-offset" if shapes and (Hr, Wr) not in shapes
                                 else ""))
            return done("expert --offset-px", ex, False,
                        f"--offset-px, not verified ({plan.no_reference}; no store of {req.site} "
                        f"registered against a reference netCDF to check it against"
                        + (f": {'; '.join(f'{y}: {w}' for y, w in skipped)}" if skipped else "")
                        + f"); {line}")
        if plan.root is None:
            raise _Refused(f"no reference netCDF ({plan.no_reference}) and no registered store to "
                           f"take the offset from")
        cands, skipped = siblings(True)
        if not cands:
            raise _Refused(f"no reference netCDF ({plan.no_reference}) and no registered store of "
                           f"{req.site} from the same request"
                           + (f" (skipped: {'; '.join(f'{y}: {w}' for y, w in skipped)})" if skipped
                              else " (none complete)"))
        cand = cands[0]
        rec["reference"] = cand["path"]
        got = register_with_sibling(interp, frame, cand, log)
        rec["sibling"] = got
        rec["dates"], rec["r"] = got.get("dates"), got.get("r")
        if got["outcome"] != "confirmed":
            m = got.get("measured")
            raise _Refused(f"no reference netCDF ({plan.no_reference}); the offset "
                           f"({cand['offset'][0]}, {cand['offset'][1]}) of the registered "
                           f"{cand['year']} store is not confirmed on this delivery: {got['why']}"
                           + (f" (the composites put the frame at ({m[0]}, {m[1]}); --offset-px "
                              f"{m[0]},{m[1]} would be checked against them the same way)" if m
                              else ""))
        line = layer(got["offset"], got["positions"])
        return done("sibling store", got["offset"], False,
                    f"no reference netCDF ({plan.no_reference}): the offset of the {cand['year']} "
                    f"store ({Path(cand['path']).name}, registered against its reference), "
                    f"confirmed by the median composites of {len(got['dates'])} clear dates of "
                    f"this delivery and {len(got['sibling_dates'])} of that store "
                    f"({_composite_line(got)}); {line}")
    except _Refused as exc:
        return refuse(str(exc))


def angles_data_extent(interp, band="sunZenithAngles", n=REPORT_DATES):
    """``[[first row, last row], [first col, last col]]`` of the DELIVERED
    cells holding a valid ``band`` value on up to ``n`` dates spread over the
    year (before the placement; None without such a cell or band)."""
    a = interp.bands.get(band)
    if a is None or not a.shape[0]:
        return None
    T = int(a.shape[0])
    idx = [int(i) for i in np.unique(np.linspace(0, T - 1, min(T, n)).round())]
    v = np.asarray(a.isel(t=idx).values)
    ok = _valid(v, interp.fills.get(band))
    if v.dtype.kind == "i":
        ok &= v != zc.FILL
    rows, cols = np.flatnonzero(ok.any(axis=(0, 2))), np.flatnonzero(ok.any(axis=(0, 1)))
    if not rows.size:
        return None
    return [[int(rows[0]), int(rows[-1])], [int(cols[0]), int(cols[-1])]]


def angles_placement(interp, req, delivered, log=print):
    """How an angles delivery was placed (they carry no band to register them
    with; module docstring, "Registration"): ``interp.placement``, written to
    the store as ``delivered_shape_cells`` and ``placement``. ``delivered``:
    the delivered ``(shape, x0, y0, data extent)`` before the placement
    (``angles_data_extent``). A delivery larger than the frame (padded, like
    the main product's, whose x/y were wrong by up to (288, 271) px in
    2026-10) is placed by its x/y and NOT registered: a WARNING, and
    ``verify_angles`` lists it as a problem (not for the sensing-time
    inversion until registered). Where the sampled sun zeniths show the data
    SHIFTED against the x/y (starting after the frame's first row / column
    and running past its last one: a padded, displaced block), a production
    year stops (FormatError); --test warns."""
    (Hr, Wr), x0, y0, extent = delivered
    H, W = len(interp.y), len(interp.x)
    res = float(req.spec.resolution)
    cy = int(round((y0 - float(interp.y[0])) / res)) if H else 0
    cx = int(round((float(interp.x[0]) - x0) / res)) if W else 0
    rec = {"method": "coordinates", "registered": False, "delivered_shape_cells": [Hr, Wr],
           "frame_shape_cells": [H, W], "coordinate_offset_cells": [cy, cx],
           "data_extent_cells": extent, "padded": bool(Hr > H or Wr > W)}
    interp.placement = rec
    if not rec["padded"]:
        rec["why"] = ("the delivered arrays are not larger than the frame: placed by their x/y "
                      "(no other offset)")
        return rec
    if extent is None:
        seen = "no valid sun zenith on the sampled dates: where the data sit is unknown"
    else:
        (r0, r1), (c0, c1) = extent
        shifted = ((r0 > cy and r1 > cy + H - 1) or (c0 > cx and c1 > cx + W - 1))
        if shifted:
            msg = (f"the sampled sun zeniths of the padded angles delivery ({Hr} x {Wr} cells) "
                   f"occupy rows {r0}..{r1}, cols {c0}..{c1}, but its x/y put the {H} x {W} cell "
                   f"frame at rows {cy}..{cy + H - 1}, cols {cx}..{cx + W - 1}: the data are "
                   f"SHIFTED against the x/y, as the main product's were")
            rec["why"] = msg
            if not req.test:
                raise FormatError(msg + "; nothing written")
            interp.warnings.append(msg + " (--test only; a production year stops)")
            return rec
        if (r0, c0) == (0, 0) and (r1, c1) == (Hr - 1, Wr - 1):
            seen = ("the sampled sun zeniths fill the whole delivered arrays, so they cannot tell "
                    "where the frame sits")
        elif r0 > cy or c0 > cx or r1 < cy + H - 1 or c1 < cx + W - 1:
            seen = (f"the sampled sun zeniths (rows {r0}..{r1}, cols {c0}..{c1}) do not cover the "
                    f"whole frame where the x/y put it (rows {cy}..{cy + H - 1}, cols "
                    f"{cx}..{cx + W - 1})")
        else:
            seen = (f"the sampled sun zeniths (rows {r0}..{r1}, cols {c0}..{c1}) cover the frame "
                    f"where the x/y put it and nothing past it: consistent with the x/y")
    rec["why"] = (f"padded angles delivery ({Hr} x {Wr} cells for the {H} x {W} cell frame) placed "
                  f"by its x/y (the frame at row {cy}, col {cx}), NOT registered (the angles carry "
                  f"no band to register; the main product's padded deliveries were displaced "
                  f"against their x/y in 2026-10); {seen}")
    interp.warnings.append(rec["why"] + ": not for the sensing-time inversion until registered")
    log(f"angles placement: {rec['why']}")
    return rec


def phase_gate(interp, req, plan, log=print):
    """After the placement: ``phase_check`` of the placed frame on the dates
    the registration used (else up to REPORT_DATES spread over the year),
    recorded in ``interp.registration['phase_20m']`` and noted. A 20 m band
    at another phase of the 20 m lattice refuses a production year
    (RegistrationRefused, nothing written); --test warns."""
    reg = interp.registration
    T = len(interp.t)
    pos = [int(i) for _, i in (reg.get("positions") or [])]
    if not pos and T:
        pos = [int(i) for i in np.unique(np.linspace(0, T - 1, min(T, REPORT_DATES)).round())]
    chk = phase_check(interp, pos)
    reg["phase_20m"] = chk
    line = phase_line(chk)
    log(line)
    if req.is_subset and not chk["bands"]:          # a band subset without B8A / B11 / B12
        scl = scl_phase_check(interp, pos)
        reg["phase_20m_scl"] = scl
        sline = scl_phase_line(scl)
        log(sline)
        if scl["verdict"] == "contradicted":
            msg = (f"{sline}: the SCL is displaced by an odd number of pixels against the frame "
                   f"(the offset {reg.get('offset')} of the {reg.get('method')} registration does "
                   f"not hold for it)")
            if plan.strict:
                raise RegistrationRefused(f"{req.tag}: {msg}; nothing written")
            interp.warnings.append(f"{msg} (--test only; a production year is refused)")
        elif scl["verdict"] == "confirmed":
            interp.notes.append(sline)
        else:
            interp.warnings.append(f"{sline}: this band subset's 20 m layer is not confirmed")
    if not chk["contradicted"]:
        interp.notes.append(line)
        return chk
    msg = (f"{line}: the 20 m layer is displaced by an odd number of pixels against the frame "
           f"(the offset {reg.get('offset')} of the {reg.get('method')} registration does not hold "
           f"for it)")
    if plan.strict:
        raise RegistrationRefused(f"{req.tag}: {msg}; nothing written")
    interp.warnings.append(f"{msg} (--test only; a production year is refused)")
    return chk


def place_at_offset(interp, target, rec, why=None):
    """Crop every band (the 10 m ones, the nearest-resampled 20 m ones, SCL,
    CLD) at the registered offset onto the frame: ``raw[dy:dy + H, dx:dx +
    W]`` with the frame's axes (an index shift; values untouched); the
    cropped-away margin is summarised as a note."""
    dy, dx = (int(v) for v in rec["offset"])
    first = next(iter(interp.bands.values()))
    _, Hr, Wr = (int(v) for v in first.shape)
    frame = _frame_of(target, interp)
    fx = np.asarray(frame["x"], dtype="float64").copy()
    fy = np.asarray(frame["y"], dtype="float64").copy()
    H, W = len(fy), len(fx)
    cy, cx = rec.get("coordinate_offset", [dy, dx])
    lead = f"{why}; " if why else ""
    if (dy, dx, H, W) == (0, 0, Hr, Wr):
        interp.notes.append(f"{lead}the delivered grid is the frame ({H} x {W} px): offset (0, 0) "
                            f"({rec['method']})")
        interp.x, interp.y = fx, fy
        return None
    margin = margin_summary(interp, dy, dy + H, dx, dx + W)
    for b, a in list(interp.bands.items()):
        interp.bands[b] = a.isel(y=slice(dy, dy + H), x=slice(dx, dx + W))
    interp.x, interp.y = fx, fy
    interp.notes.append(
        f"{lead}delivered grid {Hr} x {Wr} px put on the frame {H} x {W} px from ({fx[0]}, {fy[0]}) "
        f"at the registered offset ({rec['method']}): cropped rows {dy}:{dy + H} cols {dx}:{dx + W}"
        + ("" if (cy, cx) == (dy, dx) else
           f"; the delivered x/y put the frame at row {cy}, col {cx}: not used"))
    if margin:
        interp.notes.append(format_margin(margin, interp.fills, how="the registered offset")[0])
    return margin


def _store_date_hint(part, store_path, files):
    """The one datetime the assets of a delivered store declare (STAC), or
    None: the date of a store without a time dimension."""
    part, sp = Path(part), Path(store_path).resolve()
    seen = set()
    for rel, md in (files or {}).items():
        when = md.get("datetime")
        if not when:
            continue
        cands = [(part / "assets" / rel).resolve(),
                 (part / "extracted" / (rel[:-4] if rel.lower().endswith(".zip")
                                        else rel + ".d")).resolve()]
        if any(c == sp or sp in c.parents or c in sp.parents for c in cands):
            seen.add(when)
    return seen.pop() if len(seen) == 1 else None


def _write_format_report(req, lines):
    path = Path(req.sidecar("format.txt"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def largest_chunk(raws):
    """``(bytes, description)`` of the largest delivered chunk of any array
    with two or more dimensions (what one decode costs in memory)."""
    best = (0, None)
    for r in raws:
        for name, i in r.info.items():
            ch = i.get("chunks")
            if not ch or len(i.get("shape") or []) < 2:
                continue
            try:
                nb = int(np.prod([int(c) for c in ch])) * np.dtype(i["dtype"]).itemsize
            except (TypeError, ValueError):
                continue
            if nb > best[0]:
                best = (nb, f"{name} chunks {list(ch)} {i['dtype']}")
    return best


def normalise(req, log, job_info=None, results_meta=None, ref_grid=None, report_all=False,
              backend_format=BACKEND_FORMAT, allow_offset=False, nodata=None,
              allow_padding=False, registration=None, dry_run=False):
    """Everything between the download and the store: extract, find, read,
    report, interpret. Returns ``(canonical dataset, interpretation, report
    lines)``; the report is also written to ``<year>.format.txt``.

    ``registration``: a ``RegistrationPlan`` (``registration_plan``; every
    main request of ``run_request`` and --renormalise has one): after the
    lattice check the frame's place in the delivered arrays is established by
    ``register_delivery`` and every band cropped at that offset
    (``place_at_offset``); the record travels in ``interp.registration``.
    None (the layout tests): the coordinates place the pixels, as before
    2026-10-19. ``dry_run``: stop once the pixels are placed; no report
    written, no dataset (``(None, interp, lines)``).

    Whatever goes wrong (an unknown codec, an unreadable time coordinate, a
    bad zip, a layout the interpreter does not know), the report gathered
    so far is written and logged with the reason (and the traceback when it
    is not a FormatError), and a FormatError is raised. x/y delivered as
    float32 are put on the reference lattice when every value is the float32
    rounding of a lattice value (``snap_to_lattice``: the lattice's exact
    float64 values). A grid off the reference lattice (``lattice_check``)
    stops here unless ``allow_offset``; the same lattice with another extent
    (the Zarr writer pads its arrays to whole chunks) is cropped onto the
    reference grid, or for --test onto the requested window
    (``put_on_target``), and the report says what the cropped-away margin
    held (``margin_summary``); a delivery that does not cover the reference
    grid of a production year stops unless ``allow_padding``.
    A --test of the lon/lat box (``--lonlat-extent``) keeps the delivered
    window. ``nodata``: the no-data of bands that declare none
    (``--nodata``); without it an undecidable no-data stops a production
    year (AMBIGUOUS NO-DATA) and is only reported by --test."""
    from collections import Counter

    spec = req.spec
    main = not req.is_angles
    part = Path(req.part)
    where = (f"extent {describe_extent(req.extent, spec.resolution)} ({req.extent_source}); "
             f"lon/lat box {req.bbox}" if req.extent is not None else
             f"bbox {req.bbox}" + (" (--lonlat-extent)" if req.extent_source else ""))
    lines = ["=" * 30 + f" FORMAT REPORT {req.tag} " + "=" * 30,
             f"request: {COLLECTION} bands {list(req.bands)} window [{req.start}, {req.end}) {where}"
             + ("" if main else f"; resample_spatial({spec.resolution} m, {spec.method!r})")]
    files = {}
    found = {"stores": [], "netcdf": [], "geotiff": [], "other": []}
    raws, problems = [], []
    try:
        if (part / ASSETS_DONE).is_file():
            files = json.loads((part / ASSETS_DONE).read_text(encoding="utf-8"))["files"]
        if (part / OUTPUT_FORMAT).is_file():
            lines += report_output_format(json.loads((part / OUTPUT_FORMAT).read_text(encoding="utf-8")))
        lines += report_job(job_info, results_meta, files)
        prods = input_products(results_meta)
        if prods:
            bl = Counter(b for v in prods.values() for b in v)
            lines.append(f"input products: {len(prods)} date(s) named in the results metadata, "
                         f"processing baselines {dict(sorted(bl.items()))}")
        else:
            lines.append("input products: none named in the results metadata (processing "
                         "baselines unknown)")
        extracted = extract_zips(part, log, dry_run=dry_run) if files else []
        for d in extracted:
            lines.append(f"zip extracted to {d}:")
            lines += report_tree(d)
        if not extracted and (part / "assets").is_dir():
            lines.append(f"downloaded files under {part / 'assets'} (no zip):")
            lines += report_tree(part / "assets")
        found = discover([part / "assets", part / "extracted"])
        lines.append(f"found: {len(found['stores'])} Zarr store(s), {len(found['netcdf'])} netCDF, "
                     f"{len(found['geotiff'])} GeoTIFF, {len(found['other'])} other file(s)")
        for s in found["stores"]:
            try:
                r = read_zarr(s)
                raws.append(r)
                lines += report_raw(r)
                if s["kind"] == "zarr" and report_all:
                    lines += ["  " + x for x in report_xarray_direct(s["path"])]
            except Exception as exc:          # unknown codec, dtype, broken metadata ...
                msg = f"{type(exc).__name__}: {exc}"
                problems.append(f"{s['path']}: {msg}")
                lines.append(f"  store {s['path']}: zarr cannot read it ({msg})")
                lines += report_raw_metadata(s["path"])
        if not found["stores"]:
            for p in found["netcdf"]:
                try:
                    r = read_netcdf(p)
                    raws.append(r)
                    lines += report_raw(r)
                except Exception as exc:
                    problems.append(f"{p}: {type(exc).__name__}: {exc}")
                    lines.append(f"  netCDF {p}: cannot be read ({type(exc).__name__}: {exc})")
            if found["netcdf"] and str(backend_format).lower() == "zarr":
                lines.append("  WARNING: Zarr was requested but the backend delivered netCDF: the "
                             "store is normalised from it")
        nb, what = largest_chunk(raws)
        if nb:
            lines.append(f"largest delivered chunk: {what} = {nb / 1e6:.1f} MB decoded"
                         + (f"  WARNING: above {BIG_CHUNK_BYTES / 1e9:.0f} GB, every block written "
                            f"decodes it whole (memory)" if nb > BIG_CHUNK_BYTES else ""))
        meta = results_meta or {}
        crs_hints = [("job results metadata", meta.get("properties") or {})]
        crs_hints += [(f"job results asset {k!r}", a) for k, a in (meta.get("assets") or {}).items()
                      if isinstance(a, dict)]
        if not raws:
            hint = ""
            if any("zarr" in str(v.get("type")).lower() for v in files.values()):
                hint = (" The backend announced a Zarr asset that did not download as a store "
                        "(a directory behind one href?): see the parameters of the Zarr format "
                        "in the report of --test and pass one that zips the store with "
                        "--format-options.")
            raise FormatError(f"nothing readable among the results: {found['geotiff'][:3]} "
                              f"{found['other'][:5]} {problems}.{hint}")
        kw = dict(crs_hints=crs_hints, nodata=nodata, strict_nodata=not req.test,
                  count_nodata=req.test,
                  ref_grid=ref_grid if ref_grid is not None else request_grid(req))
        if not main:                # --nodata is the main bands'; the angles' grid is 300 m
            kw.update(nodata=None, resolution=spec.resolution, product=spec)
        if len(raws) == 1:
            parts = [interpret(raws[0], req.bands,
                               date_hint=_store_date_hint(part, raws[0].source, files), **kw)]
        else:                       # several assets: each may hold a subset of the bands
            parts = []
            for r in raws:
                try:
                    parts.append(interpret(r, req.bands, require_all=False,
                                           date_hint=_store_date_hint(part, r.source, files), **kw))
                except AmbiguousNoData:
                    raise
                except Exception as exc:
                    problems.append(f"{type(exc).__name__}: {exc}")
                    lines.append(f"  asset skipped: {type(exc).__name__}: {exc}")
            if not parts:
                raise FormatError(f"none of the {len(raws)} delivered stores/files holds the "
                                  f"requested bands: {problems}")
        interp = combine(parts)
        missing = [b for b in req.bands if b not in interp.bands]
        if missing:
            raise FormatError(f"requested bands {missing} not found in any of the {len(raws)} "
                              f"delivered stores/files (found {sorted(interp.bands)}); "
                              f"skipped: {problems}")
        rgrid = request_grid(req)               # the UTM extent asked for (None: lon/lat)
        guard = ref_grid if ref_grid is not None else rgrid
        if guard is None:
            interp.warnings.append("no reference grid: CRS and 10 m lattice NOT checked")
        elif ref_grid is None:
            interp.warnings.append("no reference grid: CRS and lattice checked against the "
                                   "requested extent only")
        ext_note = delivered_extent_note(interp.attrs, req)
        if ext_note:
            interp.notes.append(ext_note)
        interp.notes.extend(snap_to_lattice(interp, guard))
        if main:
            err, extent = lattice_check(interp, guard, whole=not req.test)
        else:
            err, extent, note = angles_grid_check(interp, guard, whole=not req.test)
            if note:
                interp.warnings.append(note)
        if err and allow_offset:
            interp.warnings.append(err + " (written anyway: --allow-grid-offset)")
        elif err:
            raise FormatError(err + "; nothing written (run again with --allow-grid-offset to "
                              "write the store anyway)")
        else:
            reg = None
            if registration is not None and main:
                target, _ = placement_target(interp, req, guard, rgrid, extent)
                reg = register_delivery(interp, req, target, registration, results_meta, log)
                interp.registration = reg
            delivered = (tuple(int(v) for v in next(iter(interp.bands.values())).shape[1:]),
                         float(interp.x[0]), float(interp.y[0]),
                         None if main else angles_data_extent(interp))
            put_on_target(interp, req, guard, rgrid, extent, ref_grid, allow_padding=allow_padding,
                          registered=reg)
            if reg is not None:             # the 20 m layer's phase on the placed frame
                phase_gate(interp, req, registration, log)
            if not main:                    # the angles are placed by their x/y: say so
                angles_placement(interp, req, delivered, log)
        if dry_run:
            lines.append("=> dry run: placed, nothing written")
            return None, interp, lines
        if main:
            ds = canonical_dataset(interp, req.bands)
            lines += report_interp(interp, ds)
            idx, dline = report_dates(ds, interp.fills)
            lines += report_values(ds, interp.fills, idx, dline)
            lines += report_resampling(ds, interp.fills, idx)
            lines += report_summary(files, found, raws, interp)
        else:
            ds = canonical_dataset(interp, req.bands, product=spec)
            lines += report_interp(interp, ds)
            idx, dline = report_dates(ds, store_fills(interp, spec))
            lines += report_angle_values(ds, idx, dline)
            chk = angle_checks(ds)
            lines += report_angle_checks(chk)
            bad = sun_out_of_range(chk)
            if bad and not req.test:
                raise FormatError("sun angles outside their physical range (not degrees?): "
                                  + "; ".join(bad))
            if bad:
                msg = "sun angles outside their physical range: " + "; ".join(bad)
                interp.warnings.append(msg)
                lines.append("  WARNING: " + msg)
            zeros = sun_zeros(chk)
            if zeros:
                msg = ("sun angles exactly 0 (impossible at these sites: a no-data written as 0 "
                       "without being declared?): " + "; ".join(zeros) + " - kept as delivered")
                interp.warnings.append(msg)
                lines.append("  WARNING: " + msg)
            lines += report_summary(files, found, raws, interp, product=spec)
        lines.append("=> normalisable: " + ("YES" if not interp.warnings else
                                             "YES, with the warnings above"))
    except Exception as exc:
        is_fmt = isinstance(exc, FormatError)
        lines.append(f"=> NOT NORMALISABLE: {'' if is_fmt else type(exc).__name__ + ': '}{exc}")
        if not is_fmt:
            ours = [f for f in traceback.extract_tb(exc.__traceback__)
                    if Path(f.filename).name in ("download_cube_zarr.py", "zarr_cubes.py")]
            if ours:
                lines.append(f"   raised in {ours[-1].name} ({Path(ours[-1].filename).name}:"
                             f"{ours[-1].lineno}): {ours[-1].line}")
            lines.append("   traceback (last lines):")
            lines += ["   " + ln for ln in traceback.format_exc().splitlines()[-20:]]
        lines.append(f"   the backend's files stay in {req.part}; nothing was written to {req.store}")
        lines.append("=" * 76)
        path = req.sidecar("format.txt") if dry_run else _write_format_report(req, lines)
        for ln in lines:
            log(ln)
        if is_fmt:
            raise
        raise FormatError(f"{type(exc).__name__} while reading the delivered output: {exc} "
                          + ("(dry run: no FORMAT REPORT written)" if dry_run else
                             f"(FORMAT REPORT in {path})")) from exc
    lines.append("=" * 76)
    _write_format_report(req, lines)
    return ds, interp, lines


def provenance(req, job_info, results_meta, interp, ctx, flat_graph, graph_source="submitted",
               ref_grid=None, job_options=None):
    import openeo
    import xarray as xr
    import zarr
    import numcodecs

    attrs = {
        "process_graph_source": graph_source,
        "input_baselines": json.dumps(input_products(results_meta)),
        "grid_reference": (f"{ref_grid.get('kind', 'netcdf')}: {ref_grid.get('path')}" if ref_grid
                           else "none (grid not checked)"),
        "nodata_decision": json.dumps(interp.nodata_lines)[:4000],
        "title": f"Sentinel-2 L2A cube, {req.site} {req.year}" + (" (format test)" if req.test else ""),
        "Conventions": "CF-1.9",
        "site": req.site, "year": int(req.year),
        "temporal_extent": json.dumps([req.start, req.end]),
        "temporal_extent_note": "end-exclusive, as openEO",
        "spatial_extent": json.dumps(req.spatial_extent),
        "collection": COLLECTION, "backend": f"https://{BACKEND}",
        "job_id": (job_info or {}).get("id", ""), "job_title": (job_info or {}).get("title", ""),
        "job_created": str((job_info or {}).get("created", "")),
        "job_updated": str((job_info or {}).get("updated", "")),
        "job_costs": json.dumps((job_info or {}).get("costs")),
        "job_usage": json.dumps((job_info or {}).get("usage"))[:4000],
        "job_options": json.dumps(job_options or ctx.job_options),
        "backend_format": ctx.backend_format,
        "backend_format_options": json.dumps(ctx.format_options or {}),
        "backend_version": backend_version(results_meta),
        "processing_date": utcnow().isoformat(timespec="seconds") + "Z",
        "bands": json.dumps(list(req.bands)),
        "resampling": RESAMPLING,
        "max_cloud_cover": MAX_CLOUD_COVER,
        "process_graph": json.dumps(flat_graph),
        "delivered": interp.source,
        "delivered_attributes": json.dumps(interp.attrs, default=str)[:4000],
        "delivered_notes": json.dumps(interp.notes),
        "delivered_warnings": json.dumps(interp.warnings),
        "crs_source": interp.crs_source,
        "fill_sources": json.dumps(interp.fill_sources),
        "chunks": json.dumps(zc.CHUNKS), "compressor": json.dumps(zc.COMPRESSOR),
        "grid_epsg": int(zc._epsg_from_wkt(interp.crs_wkt) or 0),
        "software": json.dumps({"python": sys.version.split()[0], "xarray": xr.__version__,
                                "zarr": zarr.__version__, "numcodecs": numcodecs.__version__,
                                "openeo": openeo.__version__}),
        "written_by": "experiments/download_cube_zarr.py",
    }
    if req.extent is not None:      # the UTM extent sent; the old lon/lat box kept beside it
        attrs["spatial_extent_lonlat"] = json.dumps(req.bbox)
    if req.extent_source:
        attrs["spatial_extent_source"] = req.extent_source
    reg = interp.registration
    if reg:                         # where the frame sat in the delivered arrays (register_delivery)
        attrs.update({
            "delivered_offset_px": json.dumps(reg.get("offset")),
            "delivered_shape_px": json.dumps(reg.get("delivered_shape")),
            "registration_method": str(reg.get("method")),
            "registration_reference": str(reg.get("reference") or ""),
            "registration_dates": json.dumps(reg.get("dates") or []),
            "registration_r": json.dumps(reg.get("r") or {}),
            "registration": json.dumps(reg, default=str),
        })
    if req.is_subset:               # a band subset (--bands): what it is, where its nine bands are
        attrs.update({
            "title": (f"Sentinel-2 L2A cube, {req.site} {req.year} (bands {' '.join(req.bands)})"
                      + (" (format test)" if req.test else "")),
            "resampling": subset_resampling(req.bands),
            "band_subset": json.dumps(list(req.bands)),
            "band_subset_of": json.dumps(list(MAIN.bands)),
            "nine_band_root": str(req.main_root or ""),
            "light_gate": json.dumps({
                "passed_by": light_gate_passed(req.root, req.bands),
                "passed_on_common_dates": light_gate_common_dates(req.root, req.bands),
                "own_nine_band_store": bool(req.main_root is not None and zc.is_complete(
                    zc.store_path(req.site, req.year, req.main_root))),
                "without_light_gate": bool(getattr(ctx, "without_light_gate", False)),
                "extraction_gate": (extraction_gate_status(req.site, ctx)[1]
                                    if req.site in EXTRACTION_GATE_SITES else None),
                "without_extraction_gate": bool(getattr(ctx, "without_extraction_gate", False))}),
            "band_subset_note": (
                "a band subset of the main product (download_cube_zarr --bands): the process, "
                "extent, window, resampling and registration of the nine-band stores with fewer "
                "bands, in a store root of its own (one band set per root); the nine-band stores "
                "under nine_band_root were only read"),
        })
    if req.is_angles:
        attrs.update({
            "title": (f"Sentinel-2 L2A sun and view angles, {req.site} {req.year}"
                      + (" (format test)" if req.test else "")),
            "product": ANGLES.name,
            "resampling": ANGLES_RESAMPLING,
            "chunks": json.dumps(zc.ANGLES_CHUNKS),
            "grid_resolution_m": int(ANGLES_RES_M),
            "units": "degree",
            "nodata": "NaN (float32); the delivered no-data value, if any, mapped to NaN",
            "delivered_nodata": json.dumps({b: repr(f) for b, f in interp.fills.items()}),
            "main_store": str(zc.store_path(req.site, req.year, req.root)),
            "purpose": ("sun angles at the site, to recompute the time the satellite passed over "
                        "it (the scene labels are datatake starts); view angles for diagnostics"),
        })
        if interp.placement:        # placed by the x/y, not registered (angles_placement)
            attrs.update({
                "delivered_shape_cells": json.dumps(interp.placement.get("delivered_shape_cells")),
                "placement": json.dumps(interp.placement, default=str),
            })
    return attrs


def recorded_title(part):
    """The job title a part folder was made for (ledger), or None."""
    p = Path(part) / LEDGER
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("title")
    except ValueError:
        return None


def set_aside(part, why, log, test=False):
    """Move a part folder out of the way, never deleting it here: a
    production run removes it once the year's store is complete (unless
    --keep-raw); ``test`` (--test) never removes it."""
    part = Path(part)
    stamp = f"{part.name}.superseded-{utcnow():%Y%m%dT%H%M%S}"
    dest, k = part.with_name(stamp), 1
    while dest.exists():                      # two in the same second
        dest, k = part.with_name(f"{stamp}-{k}"), k + 1
    os.replace(part, dest)
    log(f"{why}: the previous job and download folder moved to {dest} ("
        + ("kept: --test never deletes it" if test else
           "deleted after this year's store is complete, unless --keep-raw")
        + "); the job of the current request is made (or reattached by its title)")
    return dest


def superseded_parts(req):
    part = Path(req.part)
    return sorted(part.parent.glob(f"{part.name}.superseded-*")) if part.parent.is_dir() else []


def seed_lonlat_part(req, title, log):
    """--test --lonlat-extent: when its part folder (under <root>/_test_lonlat)
    does not exist, copy into it the newest folder under <root>/_test made for
    the same job title (the download of 2026-10-13, which the first --test
    of the UTM extent set aside as ``<year>.part.superseded-<time>``), so the
    old job's results are neither downloaded again nor, if they expired on
    the backend, paid for again. The source is kept. Returns it, or None."""
    part = Path(req.part)
    if not req.test or req.extent is not None or part.exists():
        return None
    base = zc.site_dir(req.site, Path(req.root).parent / TEST_ROOT)
    if not base.is_dir():
        return None
    cands = [base / part.name] + sorted(base.glob(f"{part.name}.superseded-*"), reverse=True)
    for src in cands:
        if src.is_dir() and recorded_title(src) == title:
            part.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, part)
            log(f"{part.name}: copied from {src} (made for the same job title {title!r}; the "
                f"source is kept): its job is reattached and its download reused, no new job")
            return src
    return None


def refuse_snapped(req):
    """The message refusing a production request whose frame is the snapped
    lon/lat box (``snapped``), or None."""
    if req.test or not snapped(req):
        return None
    ref = ROOT / SITE_CONFIG[req.site]["reference"]
    return (f"{req.tag}: the extent {describe_extent(req.extent, req.spec.resolution)} is the "
            f"lon/lat box snapped outward to {FRAME_SNAP_M} m ({req.extent_origin}), not the frame "
            f"of the site's reference netCDF {ref} (missing or not a {RESOLUTION_M} m grid) nor of "
            f"a complete store: the store would not be on that frame, and would become the "
            f"reference of the site's other years. Put the netCDF there, or pass "
            f"--allow-snapped-frame to accept the snapped frame")


def run_request(ctx, req, log):
    """Download, normalise, verify and time one store (of either product).
    Returns a summary."""
    t0 = time.time()
    angles = req.is_angles
    if req.is_subset:               # a band subset never writes into a root of other bands
        why = subset_root_problem(req.root, req.bands, req.main_root)
        if why:
            raise FormatError(f"{req.tag}: {why}; nothing done")
    if not req.test and req.complete():                 # also with --new-job: never paid twice
        log(f"{req.store} is complete: nothing to download")
        if angles:                  # cheap; the main store of the year may have come since
            rep = verify_angles(req, log)
            return {"status": "skipped", "dates": rep["dates"]["angles"], "verdict": rep["verdict"]}
        ds = zc.open_store(req.store)
        dates = zc.dates(ds)
        ds.close()
        old = pre_registration_warning(req)
        if old:
            log(old)
        if not req.sidecar("verify.json").is_file():
            verify_request(req, log)
        still = update_overpass(req, dates, log, network=ctx.network)
        return dict({"status": "skipped", "dates": len(dates), "missing_overpass": still},
                    **({"pre_registration": True} if old else {}))
    why = refuse_snapped(req)
    if why and not ctx.allow_snapped:
        raise FrameRefused(why)
    part = Path(req.part)
    if req.is_subset and not (part / ASSETS_DONE).is_file():     # a job may be bought
        why = light_wait_problem(req, ctx)
        if why:
            raise LightGateMissing(f"{req.tag}: {why}; nothing started (no job, no download)")
        why = extraction_wait_problem(req, ctx)
        if why:
            raise ExtractionGateMissing(f"{req.tag}: {why}; nothing started (no job, no download)")
    flat = build_process(req, None, ctx.backend_format, ctx.format_options).flat_graph()
    title = job_title(req, graph_hash(flat))
    if not ctx.new_job:
        seed_lonlat_part(req, title, log)
    if part.exists():
        old = recorded_title(part)
        if ctx.new_job:
            set_aside(part, "--new-job", log, test=req.test)
        elif old and old != title:
            set_aside(part, f"the request changed since {part.name} was made (job title {old!r}, "
                            f"now {title!r}: other extent, format, format options or bands)", log,
                      test=req.test)
        elif not old and any((part / n).exists() for n in (ASSETS_DONE, "assets", DOWNLOAD_MARKER)):
            set_aside(part, f"{part.name} holds a download but no job ledger, so its request is "
                            f"unknown", log, test=req.test)
    part.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(part).free
    log(f"start: window [{req.start}, {req.end}) "
        + (f"extent {describe_extent(req.extent, req.spec.resolution)} ({req.extent_source})"
           if req.extent is not None else f"bbox {req.bbox} (lon/lat)")
        + f"; job '{title}'; {free / 1e9:.0f} GB free")
    led_path = part / LEDGER
    if (part / ASSETS_DONE).is_file():
        led = Ledger(part)
        cur = led.current() or {}
        info = cur.get("info", {})
        log(f"results of job {info.get('id', cur.get('job_id'))} already downloaded in {part}")
        meta = json.loads((part / "job-results.json").read_text(encoding="utf-8")) \
            if (part / "job-results.json").is_file() else {}
        reuse, _ = reusable_pending(req, info.get("id") or cur.get("job_id"))
        if reuse:                   # e.g. the rerun with --accept-differences after a refusal
            return finish_pending(ctx, req, log, part, info, t0)
    else:
        if req.test:
            describe_output_format(ctx.connection(), ctx.backend_format, part, log)
        job, info = ensure_finished_job(ctx, req, title, log)
        info = job_summary(info)
        try:
            meta = download_results(job, req, log)
        except Exception:
            if results_gone(job):
                led = Ledger(part)
                cur = led.current()
                if cur is not None and cur.get("job_id") == job.job_id:
                    led.update(cur, status="expired")
                log(f"the results of job {job.job_id} are no longer on the backend: the next run "
                    f"makes a new job")
            raise
    cur = (Ledger(part).current() or {}) if led_path.is_file() else {}
    if cur.get("process_graph"):
        graph, gsrc = cur["process_graph"], f"submitted with job {cur.get('job_id')}"
    else:
        graph, gsrc = flat, "built offline (the submitted graph was not recorded)"
    ref_root = _grid_root(req)
    ref = (angles_reference_grid(req.site, root=ref_root, log=log) if angles else
           reference_grid(req.site, root=ref_root, log=log))
    plan = None if angles else registration_plan(req, ctx, ref)
    ds, interp, lines = normalise(req, log, job_info=info, results_meta=meta, ref_grid=ref,
                                  allow_offset=ctx.allow_offset, nodata=ctx.nodata,
                                  allow_padding=ctx.allow_padding,
                                  report_all=req.test, backend_format=ctx.backend_format,
                                  registration=plan)
    log_report(req, lines, log)
    attrs = provenance(req, info, meta, interp, ctx, graph, graph_source=gsrc, ref_grid=ref,
                       job_options=cur.get("job_options"))
    log(f"writing {req.store} ({ds.sizes['t']} dates, {ds.sizes['y']} x {ds.sizes['x']} "
        + (f"cells of {req.spec.resolution} m, {len(req.bands)} angles as float32)" if angles else
           f"px, {len(req.bands)} bands)"))
    # the main store is completed only once its verification accepts it (verify_and_complete)
    write_store(ds, store_fills(interp, req.spec), req.store, attrs, log=log, groups=interp.groups,
                chunks=req.spec.chunks, format_id=req.spec.format_id, complete=angles,
                **_subset_write_kw(req))
    job_side = {"ledger": json.loads(led_path.read_text(encoding="utf-8"))
                if led_path.is_file() else {}, "job": info, "results_metadata": meta}
    zc.write_json(job_side, req.sidecar("job.json"))
    if not angles:              # raises VerificationRefused (store not complete, raw kept)
        rep = verify_and_complete(ctx, req, log)
    size = sum(f.stat().st_size for f in Path(req.store).rglob("*") if f.is_file())
    log(f"store complete: {req.store} ({size / 1e9:.2f} GB on disk)" if not angles else
        f"store complete: {req.store} ({size / 1e6:.2f} MB on disk)")
    if req.test and angles:
        cube = zc.open_angles(req.site, req.start, req.end, root=req.root)
        names = zc.band_names(cube)
        log(f"open_angles({req.site!r}, {req.start!r}, {req.end!r}, root={str(req.root)!r}): "
            f"{dict(cube.sizes)}; dates {zc.dates(cube)}")
        log(f"  angles {names}; dtypes {sorted({str(cube[b].dtype) for b in names})}; chunks "
            f"{cube[names[0]].encoding.get('chunks') if names else None}; CRS EPSG "
            f"{zc.grid_info(cube)['epsg']}; cells {zc.grid_info(cube)['dx']:g} m")
        cube.close()
        lon, lat = TEST_CENTRE
        try:
            at = zc.angles_at(req.site, lon, lat, req.start, req.end, root=req.root)
            log(f"angles_at({req.site!r}, {lon}, {lat}, {req.start!r}, {req.end!r}) at x "
                f"{at.attrs['x']:.1f}, y {at.attrs['y']:.1f} (bilinear):")
            for ln in at.round(4).to_string().splitlines():
                log(f"  {ln}")
        except ValueError as exc:             # informative: the point may fall outside
            log(f"angles_at({req.site!r}, {lon}, {lat}, ...): {exc}")
    elif req.test:
        cube = zc.open_cube(req.site, req.start, req.end, root=req.root)
        log(f"open_cube({req.site!r}, {req.start!r}, {req.end!r}, root={str(req.root)!r}): "
            f"{dict(cube.sizes)}; dates {zc.dates(cube)}")
        log(f"  bands {zc.band_names(cube)}; dtypes "
            f"{sorted({str(cube[b].dtype) for b in zc.band_names(cube)})}; fill "
            f"{sorted({str(zc.fill_of(cube[b])) for b in zc.band_names(cube)})}; chunks "
            f"{cube['B03'].encoding.get('chunks')}; CRS EPSG {zc.grid_info(cube)['epsg']}")
        cube.close()
    if angles:                      # the overpass file belongs to the main store's dates
        rep = verify_angles(req, log)
        dates, still = zc.dates(ds), []
    else:
        dates = zc.dates(ds)
        still = update_overpass(req, dates, log, network=ctx.network)
    raw_after_store(ctx, req, part, None if angles else rep, log)
    log(f"done in {(time.time() - t0) / 60:.0f} min")
    if angles:
        return {"status": "done", "dates": len(dates), "verdict": rep.get("verdict"),
                "mb": round(size / 1e6, 2), "job": info.get("id")}
    reg = interp.registration or {}
    out = {"status": "done", "dates": len(dates), "verdict": rep.get("verdict"),
           "accepted": (rep.get("acceptance") or {}).get("how"),
           "registration": f"{reg.get('method')} {reg.get('offset')}",
           "missing_overpass": still, "gb": round(size / 1e9, 3), "job": info.get("id"),
           "radiometry": (rep.get("radiometry") or {}).get("offset_state")}
    if req.is_subset:               # a band subset: its light-path gate, and what its job cost
        gate = rep.get("nine_band_store")
        out.update(light_path_gate=None if gate is None else
                   "passed" if gate.get("passed") else
                   "passed up to the no-data encoding" if gate.get("passed_up_to_nodata_encoding")
                   else "FAILED", job_costs=info.get("costs"))
    return out


def raw_after_store(ctx, req, part, rep, log):
    """Remove the raw download (and the superseded part folders) once a
    store is complete, unless --keep-raw or --test; a MAIN store accepted as
    anything but 'identical' or 'near-identical' (not verified: a sibling's
    or an expert offset, no reference; or --accept-differences) keeps it:
    it is the only way to register that year again without a new job. A
    band subset (--bands) always keeps it.
    ``rep``: the main store's verification (None for the angles)."""
    keep = "--keep-raw" if ctx.keep_raw else "--test" if req.test else None
    if keep is None and req.is_subset:      # a band subset never deletes its raw delivery
        keep = "a band subset keeps its raw delivery (--renormalise can rebuild it offline)"
    how = ((rep or {}).get("acceptance") or {}).get("how")
    if keep is None and rep is not None and how not in ("identical", "near-identical"):
        keep = (f"the store was accepted as {how!r}: the raw delivery is the only way to register "
                f"it again without a new job")
    if keep is None:
        shutil.rmtree(part, ignore_errors=True)
        log(f"raw download removed ({part}); the store keeps the provenance")
        for old in superseded_parts(req):
            shutil.rmtree(old, ignore_errors=True)
            log(f"superseded folder removed ({old})")
        return False
    log(f"raw download kept in {part} ({keep})"
        + (f"; superseded folders kept: {[p.name for p in superseded_parts(req)]}"
           if superseded_parts(req) else ""))
    return True


def reusable_pending(req, job_id):
    """``(True, why)`` when the MAIN store at ``req.store`` was written whole
    from this very delivery but not completed (its verification refused it,
    or the run stopped before deciding): not complete, its ``verification``
    pending, written from job ``job_id`` under the current registration rule
    (``registration_rule``). It is then verified again (``finish_pending``:
    --accept-differences completes it in minutes and no second copy is
    written); move it aside to have it written again."""
    store = Path(req.store)
    if req.is_angles or req.test or not store.exists() or req.complete():
        return False, "no pending store"
    attrs = zc.store_attrs(store)
    if not str(attrs.get("verification", "")).startswith("pending"):
        return False, f"{store.name} is not pending its verification"
    if not job_id or str(attrs.get("job_id")) != str(job_id):
        return False, f"{store.name} was written from job {attrs.get('job_id')!r}, not {job_id!r}"
    rec = _json_attr(attrs.get("registration"))
    if not isinstance(rec, dict) or rec.get("rule") != _json_attr(json.dumps(registration_rule())):
        return False, f"{store.name} was written under another registration rule"
    return True, (f"{store.name} was written whole from this delivery (job {job_id}) and is pending "
                  f"its verification")


def finish_pending(ctx, req, log, part, info, t0, note=None):
    """Verify again and complete (``verify_and_complete``) a pending main
    store of this delivery (``reusable_pending``) instead of writing it
    again; then the overpass times and the raw download as after a write.
    ``note``: a ledger note to fill (--renormalise). Returns the summary."""
    log(f"{req.store.name} was written whole from this delivery (job {info.get('id')}) and is "
        f"pending its verification: it is verified again, NOT rewritten (move it aside to have it "
        f"written again)")
    rep = verify_and_complete(ctx, req, log)
    if note is not None:
        note["verification"] = rep.get("acceptance")
    ds = zc.open_store(req.store)
    dates = zc.dates(ds)
    ds.close()
    # --renormalise (a note to fill) makes no network call: the overpass file offline
    still = update_overpass(req, dates, log, network=ctx.network and note is None)
    size = sum(f.stat().st_size for f in Path(req.store).rglob("*") if f.is_file())
    log(f"store complete: {req.store} ({size / 1e9:.2f} GB on disk)")
    if note is None:
        raw_after_store(ctx, req, part, rep, log)
    log(f"done in {(time.time() - t0) / 60:.0f} min")
    attrs = zc.store_attrs(req.store)
    off = _json_attr(attrs.get("delivered_offset_px"))
    return {"status": "verified again", "dates": len(dates), "verdict": rep.get("verdict"),
            "accepted": (rep.get("acceptance") or {}).get("how"),
            "registration": f"{attrs.get('registration_method')} {off}",
            "missing_overpass": still, "gb": round(size / 1e9, 3), "job": info.get("id"),
            "radiometry": (rep.get("radiometry") or {}).get("offset_state")}


def pre_registration_warning(req):
    """The warning for a complete MAIN store written before the registration
    (2026-10-19: no ``registration_method`` attribute), or None: its pixels
    may be displaced against the reference grid."""
    if req.is_angles or "registration_method" in zc.store_attrs(req.store):
        return None
    return (f"WARNING: {req.store.name} was written before the registration (2026-10-19): its "
            f"pixels may be displaced against the reference grid (padded deliveries were cut by "
            f"their x/y); rebuild it with --renormalise from its kept raw delivery, or move it aside "
            f"and download the year again")


#: the report lines a production run logs (--test logs them all)
LOGGED_REPORT = ("=", "found:", "store ", "  WARNING", "interpretation", "  CRS", "  registration",
                 "  time:", "  grid:", "input products", "largest delivered", "resampling check",
                 "  no-data", "  AMBIGUOUS")


def log_report(req, lines, log):
    """The FORMAT REPORT in the log: whole for --test, its main lines otherwise."""
    for ln in lines:
        if req.test or ln.startswith(LOGGED_REPORT):
            log(ln)


# ─────────────────────────────────────────────────────────────────────────────
#  band subsets (--bands; module docstring, "Band subsets"): off by default,
#  nothing here is used by a request of the nine bands
# ─────────────────────────────────────────────────────────────────────────────

#: the ``acceptance`` of a band subset that passed the light-path gate
NINE_BAND_HOW = "identical to the nine-band store"
#: the ``acceptance`` of a band subset that passed it up to the no-data encoding
#: (--l2a-nodata-equal: every difference a pixel-date that is 0 or the fill on every band in
#: both stores; ``nodata_encoding_check``)
NINE_BAND_NODATA_HOW = "identical to the nine-band store up to the L2A no-data encoding (0 / fill)"
#: the acceptances by the light-path gate (``light_gate_passed``)
LIGHT_GATE_HOWS = (NINE_BAND_HOW, NINE_BAND_NODATA_HOW)


def _grid_root(req):
    """Where the reference grid of ``req`` falls back to complete stores
    (``reference_grid``): the cubes root of a --test (the parent of
    ``<root>/_test``), the request's root, or for a band subset the
    nine-band root (``main_root``), whose stores are on the frame the
    subset requests."""
    if req.test:
        return req.root.parent
    if req.is_subset and req.main_root is not None:
        return Path(req.main_root)
    return req.root


def _subset_write_kw(req):
    """``write_store`` keywords: none for the nine bands; a band subset's
    radiometry indicator (SUBSET_RADIOMETRY_BANDS)."""
    return {"radiometry_bands": SUBSET_RADIOMETRY_BANDS} if req.is_subset else {}


def subset_resampling(bands):
    """The ``resampling`` attribute of a band subset's store (RESAMPLING for
    its own bands)."""
    ten = [b for b in bands if zc.NATIVE_RES_M.get(b) == RESOLUTION_M]
    twenty = [b for b in bands if zc.NATIVE_RES_M.get(b) != RESOLUTION_M]
    return (f"resample_spatial(resolution={RESOLUTION_M}, method='{RESAMPLE_METHOD}') on the "
            f"whole cube, as the existing netCDF cubes: identity for the native 10 m bands "
            f"({' '.join(ten)})"
            + (f"; nearest neighbour 20 m -> 10 m for {' '.join(twenty)}" if twenty else ""))


def subset_root_problem(store_root, bands, main_root):
    """Why stores of the band subset ``bands`` may NOT go under
    ``store_root`` (None when they may): it is the nine-band root
    ``main_root``, lies inside it or holds it, or holds a main store
    (complete or pending its verification) of another band set: one band set
    per root, so ``open_cube`` and the registered-store searches of each root
    never mix them."""
    s = Path(store_root).resolve()
    if main_root is not None:
        m = Path(main_root).resolve()
        if s == m:
            return (f"the store root {s} is the root of the nine-band stores (--root): a band "
                    f"subset never goes there (--store-root, default {SUBSET_ROOT})")
        for a, b, how in ((s, m, "lies inside"), (m, s, "holds")):
            try:
                a.relative_to(b)
            except ValueError:
                continue
            return (f"the store root {s} {how} the nine-band root {m}: keep the two apart "
                    f"(--store-root, default {SUBSET_ROOT})")
    if not s.is_dir():
        return None
    want = list(bands)
    for d in sorted(s.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        for p in sorted(d.iterdir()):
            if not (p.is_dir() and re.fullmatch(r"\d{4}\.zarr", p.name)):
                continue
            attrs = zc.store_attrs(p)
            if not attrs:
                continue
            got = _json_attr(attrs.get("bands"))
            if list(got or []) != want:
                return (f"{p} holds the bands {got}, not {want}: one band set per store root "
                        f"(open_cube refuses a window whose years differ in bands); give another "
                        f"--store-root")
    return None


def nine_band_root_problem(root):
    """Why NINE-BAND stores may not go under ``root`` (None when they may): it
    holds a band-subset store (the ``band_subset`` attribute every --bands
    store carries). One band set per root, as ``subset_root_problem`` for the
    subsets; the nine-band root ``cubes/`` holds none, so a default run is
    unchanged."""
    base = Path(root)
    if not base.is_dir():
        return None
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        for p in sorted(d.iterdir()):
            if not (p.is_dir() and re.fullmatch(r"\d{4}\.zarr", p.name)):
                continue
            attrs = zc.store_attrs(p)
            if attrs.get("band_subset"):
                got = _json_attr(attrs.get("band_subset"))
                return (f"{p} is a band-subset store (bands {got}): the nine-band stores never go "
                        f"into a root of band subsets (--root; the subsets' --store-root is "
                        f"{SUBSET_ROOT})")
    return None


def compare_stores(store, other, bands, log=None, t_block=None, row_block=None):
    """Two canonical main stores of one site and year compared BITWISE on
    ``bands``: their dates, their grids (axes and CRS) and, per band over the
    common dates, every pixel as stored (the fill included), with the dtype
    and the no-data value. Reads blocks of ``t_block`` dates x ``row_block``
    rows (the store chunks). Returns a JSON-able report; ``verdict``:
    'identical' (the same dates and grid, every band equal), 'differs' or
    'not comparable' (a band missing, another grid, no common date)."""
    say = log or (lambda m: None)
    t0 = time.time()
    a, b = zc.open_store(store), zc.open_store(other)
    try:
        da_, db_ = zc.dates(a), zc.dates(b)
        sa, sb = set(da_), set(db_)
        common = sorted(sa & sb)
        rep = {"store": str(store), "other": str(other), "bands": list(bands),
               "dates": {"store": len(da_), "other": len(db_), "common": len(common),
                         "only_store": sorted(sa - sb), "only_other": sorted(sb - sa)},
               "grid_identical": bool(zc._same_grid(a, b)), "per_band": {}}
        missing = [x for x in bands if x not in a.data_vars or x not in b.data_vars]
        if missing or not common or not rep["grid_identical"]:
            why = (f"band(s) {missing} missing" if missing else
                   "no common date" if not common else "another grid")
            return dict(rep, verdict="not comparable", reason=why,
                        seconds=round(time.time() - t0, 1))
        ia = {d: i for i, d in enumerate(da_)}
        ib = {d: i for i, d in enumerate(db_)}
        H = int(a.sizes["y"])
        tb, rb = int(t_block or zc.CHUNKS["t"]), int(row_block or zc.CHUNKS["y"])
        for x in bands:
            fa, fb = zc.fill_of(a[x]), zc.fill_of(b[x])
            s = {"dtype": [str(a[x].dtype), str(b[x].dtype)], "fill": [fa, fb], "pixels": 0,
                 "differing": 0, "dates_differing": {}}
            for k in range(0, len(common), tb):
                sel = common[k:k + tb]
                for r0 in range(0, H, rb):
                    va = np.asarray(a[x].isel(t=[ia[d] for d in sel], y=slice(r0, r0 + rb)).values)
                    vb = np.asarray(b[x].isel(t=[ib[d] for d in sel], y=slice(r0, r0 + rb)).values)
                    ne = va != vb
                    if va.dtype.kind == "f" and vb.dtype.kind == "f":
                        ne &= ~(np.isnan(va) & np.isnan(vb))
                    per = ne.reshape(len(sel), -1).sum(axis=1)
                    s["pixels"] += int(va.size)
                    s["differing"] += int(per.sum())
                    for d, c in zip(sel, per):
                        if c:
                            s["dates_differing"][d] = s["dates_differing"].get(d, 0) + int(c)
                    del va, vb, ne
            s["identical"] = bool(s["differing"] == 0 and s["dtype"][0] == s["dtype"][1]
                                  and _same_value(fa, fb))
            say(f"{x}: {s['differing']} of {s['pixels']} pixel-dates differ on {len(common)} "
                f"common dates (dtype {s['dtype'][0]} / {s['dtype'][1]}, fill {fa} / {fb})")
            rep["per_band"][x] = s
        same = all(s["identical"] for s in rep["per_band"].values())
        rep["bitwise_on_common_dates"] = same
        rep["verdict"] = ("identical" if same and not rep["dates"]["only_store"]
                          and not rep["dates"]["only_other"] else "differs")
        rep["seconds"] = round(time.time() - t0, 1)
        return rep
    finally:
        a.close()
        b.close()


def nodata_encoding_check(store, other, bands, t_block=None, row_block=None, max_dates=20):
    """The pixel-dates where two canonical stores of one site and year differ
    (any of ``bands``, bitwise, on their common dates and identical grid),
    split into the NO-DATA ENCODING ones (--l2a-nodata-equal): in BOTH stores
    every band of ``bands`` is 0 (the Sentinel-2 L2A no-data DN; SCL 0 =
    NO_DATA) or the store's fill (a chunk the backend did not write) at that
    pixel-date, and the others (a value, or a real 0 with a valid SCL). Needs
    SCL among ``bands`` (what says the pixel is no data). Returns a JSON-able
    record; ``equal_up_to_nodata_encoding``: every difference is encoding."""
    t0 = time.time()
    rule = ("a differing pixel-date is no-data ENCODING when, in both stores, every band of "
            f"{list(bands)} is 0 (L2A no data; SCL 0 = NO_DATA) or the store's fill")
    out = {"rule": rule, "bands": list(bands)}
    if "SCL" not in bands:
        return dict(out, equal_up_to_nodata_encoding=False,
                    reason="SCL is not compared: no-data cannot be told from a real 0")
    a, b = zc.open_store(store), zc.open_store(other)
    try:
        common = sorted(set(zc.dates(a)) & set(zc.dates(b)))
        if not common or not zc._same_grid(a, b) or any(x not in a.data_vars or x not in b.data_vars
                                                         for x in bands):
            return dict(out, equal_up_to_nodata_encoding=False,
                        reason="no common date, another grid or a band missing")
        ia = {d: i for i, d in enumerate(zc.dates(a))}
        ib = {d: i for i, d in enumerate(zc.dates(b))}
        fa = {x: zc.fill_of(a[x]) for x in bands}
        fb = {x: zc.fill_of(b[x]) for x in bands}
        H = int(a.sizes["y"])
        tb, rb = int(t_block or zc.CHUNKS["t"]), int(row_block or zc.CHUNKS["y"])
        n = {"pixel_dates": 0, "differing": 0, "nodata_encoding": 0, "other": 0,
             "fill_in_store_zero_in_other": 0, "zero_in_store_fill_in_other": 0}
        dates_other, dates_enc = {}, {}
        for k in range(0, len(common), tb):
            sel = common[k:k + tb]
            for r0 in range(0, H, rb):
                differ = nod_a = nod_b = None
                all_fill_a = all_zero_b = all_zero_a = all_fill_b = None
                for x in bands:
                    va = np.asarray(a[x].isel(t=[ia[d] for d in sel], y=slice(r0, r0 + rb)).values)
                    vb = np.asarray(b[x].isel(t=[ib[d] for d in sel], y=slice(r0, r0 + rb)).values)
                    ne = va != vb
                    ga = (va == 0) | ((va == fa[x]) if fa[x] is not None else False)
                    gb = (vb == 0) | ((vb == fb[x]) if fb[x] is not None else False)
                    fla = (va == fa[x]) if fa[x] is not None else np.zeros(va.shape, bool)
                    flb = (vb == fb[x]) if fb[x] is not None else np.zeros(vb.shape, bool)
                    if differ is None:
                        differ, nod_a, nod_b = ne, ga, gb
                        all_fill_a, all_zero_b = fla, vb == 0
                        all_zero_a, all_fill_b = va == 0, flb
                    else:
                        differ, nod_a, nod_b = differ | ne, nod_a & ga, nod_b & gb
                        all_fill_a, all_zero_b = all_fill_a & fla, all_zero_b & (vb == 0)
                        all_zero_a, all_fill_b = all_zero_a & (va == 0), all_fill_b & flb
                enc = differ & nod_a & nod_b
                oth = differ & ~(nod_a & nod_b)
                n["pixel_dates"] += int(differ.size)
                n["differing"] += int(differ.sum())
                n["nodata_encoding"] += int(enc.sum())
                n["other"] += int(oth.sum())
                n["fill_in_store_zero_in_other"] += int((differ & all_fill_a & all_zero_b).sum())
                n["zero_in_store_fill_in_other"] += int((differ & all_zero_a & all_fill_b).sum())
                for target, m in ((dates_enc, enc), (dates_other, oth)):
                    per = m.reshape(len(sel), -1).sum(axis=1)
                    for d, c in zip(sel, per):
                        if c:
                            target[d] = target.get(d, 0) + int(c)
        worst = lambda dd: dict(sorted(dd.items(), key=lambda kv: -kv[1])[:max_dates])  # noqa: E731
        return dict(out, **n, dates_common=len(common), dates_with_encoding=len(dates_enc),
                    dates_with_other=len(dates_other), worst_dates_encoding=worst(dates_enc),
                    worst_dates_other=worst(dates_other),
                    equal_up_to_nodata_encoding=bool(n["other"] == 0),
                    seconds=round(time.time() - t0, 1))
    finally:
        a.close()
        b.close()


def nine_band_gate(req, log=print, l2a_nodata=False):
    """The LIGHT-PATH GATE of a band subset (module docstring): its store
    against the complete NINE-BAND store of the same site and year under
    ``req.main_root``: ``compare_stores`` on the subset's bands ('identical':
    the same dates and grid, every pixel of every date equal, the fill
    included), the same registration offset and delivered array shape
    (``delivered_offset_px`` / ``delivered_shape_px``) and the same input
    baselines per date (where both recorded them). The whole record goes to
    ``<year>.vs_nine_band.json``; returns its summary (``passed``, ``why``),
    or None when there is no such store (nothing to compare: a year without
    a nine-band store). ``l2a_nodata`` (--l2a-nodata-equal, off by default):
    a gate that fails on the values alone is checked by
    ``nodata_encoding_check`` and, when every difference is the encoding of
    pixel-dates that are no data in both stores, recorded and returned as
    ``passed_up_to_nodata_encoding`` (``passed`` stays False)."""
    if not req.is_subset or req.main_root is None:
        return None
    other = zc.store_path(req.site, req.year, req.main_root)
    if not zc.is_complete(other):
        return None
    log(f"light-path gate: {req.store.name} against the nine-band store {other} "
        f"({' '.join(req.bands)}, bitwise)")
    cmp_ = compare_stores(req.store, other, req.bands, log=lambda m: log(f"  {m}"))
    mine, theirs = zc.store_attrs(req.store), zc.store_attrs(other)
    checks = {}
    for k in ("delivered_offset_px", "delivered_shape_px"):
        va, vb = _json_attr(mine.get(k)), _json_attr(theirs.get(k))
        checks[k] = {"subset": va, "nine_band": vb, "equal": va is not None and va == vb}
    pa = (zc.store_baselines(mine) or {}).get("per_date") or {}
    pb = (zc.store_baselines(theirs) or {}).get("per_date") or {}
    both = sorted(set(pa) & set(pb))
    differ = [d for d in both if sorted(pa[d]) != sorted(pb[d])]
    checks["input_baselines"] = {
        "dates_subset": len(pa), "dates_nine_band": len(pb), "dates_compared": len(both),
        "differing": differ[:40],
        "equal": (None if not pa or not pb else not differ and set(pa) == set(pb)),
        "equal_on_common_dates": None if not pa or not pb else not differ}
    layout_bad = [k for k in ("delivered_offset_px", "delivered_shape_px") if not checks[k]["equal"]]
    bad = []
    if cmp_["verdict"] != "identical":
        d = cmp_["dates"]
        bad.append(("the values are bitwise equal on all the common dates, but the DATES differ"
                    if cmp_.get("bitwise_on_common_dates") else
                    "the two stores differ" if cmp_["verdict"] == "differs" else
                    f"the two stores are not comparable ({cmp_.get('reason')})")
                   + f": dates {d['store']} / {d['other']}, only in the subset {d['only_store'][:6]}"
                   f", only in the nine bands {d['only_other'][:6]}"
                   + "".join(f"; {x} {s['differing']} differing pixel-dates"
                             + ("" if s["identical"] or s["differing"] else
                                f" (dtype {s['dtype']}, fill {s['fill']})")
                             for x, s in cmp_["per_band"].items()))
    for k in ("delivered_offset_px", "delivered_shape_px"):
        if not checks[k]["equal"]:
            bad.append(f"{k} {checks[k]['subset']} != the nine bands' {checks[k]['nine_band']}")
    if checks["input_baselines"]["equal"] is False:
        bad.append(f"input baselines differ ({len(differ)} of {len(both)} common dates, dates "
                   f"{len(pa)} / {len(pb)})")
    passed = not bad
    # the dates alone differ (the CDSE catalogue changed since the nine-band job): every common
    # date bitwise, the same layout and the same input baselines on them. Not an acceptance of
    # this store (that stays NINE_BAND_HOW, strict), but enough to lift the wait of the years
    # without a nine-band store (subset_gate_problem, recorded in their light_gate)
    common_only = bool(not passed and cmp_.get("bitwise_on_common_dates") and not layout_bad
                       and checks["input_baselines"]["equal_on_common_dates"] is not False
                       and cmp_["dates"]["common"] > 0)
    was = _json_attr(theirs.get("verification"))
    checks["nine_band_acceptance"] = was.get("how") if isinstance(was, dict) else None
    # --l2a-nodata-equal (off by default): the values differ, nothing else (the same dates, grid,
    # offset, delivered shape and input baselines): are the differences only the encoding of
    # pixel-dates that are no data in both stores (L2A 0 against the fill)? (nodata_encoding_check)
    enc, nodata_ok = None, False
    if l2a_nodata and not passed:
        d = cmp_["dates"]
        only_values = bool(cmp_["verdict"] == "differs" and cmp_.get("grid_identical")
                           and not d["only_store"] and not d["only_other"] and not layout_bad
                           and checks["input_baselines"]["equal"] is not False)
        if only_values:
            enc = nodata_encoding_check(req.store, other, req.bands)
            nodata_ok = bool(enc.get("equal_up_to_nodata_encoding"))
        else:
            enc = {"checked": False, "reason": "not only the values differ (dates, grid, offset, "
                                               "delivered shape or input baselines)"}
    why = ("; ".join(bad) if bad else
           f"{cmp_['dates']['common']} = {cmp_['dates']['other']} dates, "
           f"{' '.join(req.bands)} bitwise equal on every pixel of every date, offset "
           f"{checks['delivered_offset_px']['subset']}, delivered shape "
           f"{checks['delivered_shape_px']['subset']}, input baselines "
           + ("equal" if checks["input_baselines"]["equal"] else "not recorded on one side")
           + (f"; the nine-band store was accepted as '{checks['nine_band_acceptance']}'"
              if checks["nine_band_acceptance"] else ""))
    if nodata_ok:
        why = (f"{cmp_['dates']['common']} = {cmp_['dates']['other']} dates, the same offset "
               f"{checks['delivered_offset_px']['subset']}, delivered shape "
               f"{checks['delivered_shape_px']['subset']} and input baselines; "
               f"{' '.join(req.bands)} bitwise equal except {enc['differing']} pixel-dates (on "
               f"{enc['dates_with_encoding']} dates) that are no data in both stores, encoded as "
               f"0 in one and the fill in the other ({enc['fill_in_store_zero_in_other']} fill in "
               f"the subset / 0 in the nine bands, {enc['zero_in_store_fill_in_other']} the reverse); "
               f"no other difference"
               + (f"; the nine-band store was accepted as '{checks['nine_band_acceptance']}'"
                  if checks["nine_band_acceptance"] else ""))
    path = req.sidecar("vs_nine_band.json")
    extra = ({"passed_up_to_nodata_encoding": nodata_ok, "nodata_encoding": enc,
              "l2a_nodata_equal": True} if l2a_nodata else {})
    zc.write_json(dict({"site": req.site, "year": req.year, "store": str(req.store),
                        "nine_band_store": str(other), "bands": list(req.bands), "passed": passed,
                        "passed_on_common_dates": passed or common_only,
                        "dates_differ_only": common_only, "layout_differs": layout_bad, "why": why,
                        "checks": checks, "comparison": cmp_,
                        "checked_utc": utcnow().isoformat(timespec="seconds")}, **extra),
                  path)
    if passed:
        log(f"light-path gate PASSED: {why} ({path.name})")
    elif nodata_ok:
        log(f"light-path gate PASSED UP TO THE NO-DATA ENCODING (--l2a-nodata-equal): {why} "
            f"({path.name})")
    elif common_only:
        log(f"LIGHT-PATH GATE FAILED on the dates only: {why}. Every common date is bitwise equal "
            f"with the same layout and input baselines, which lifts the wait of the years without a "
            f"nine-band store (recorded in their light_gate); this store is decided by the rule "
            f"({path.name})")
    elif layout_bad:
        log(f"LIGHT-PATH GATE FAILED: {why}. The nine-band registration offsets / delivered shapes do "
            f"NOT hold for deliveries of these bands: each site needs a subset job of a year its "
            f"reference netCDF covers (registered against it) to measure its own offset before any "
            f"year without reference; those stay refused ({path.name})")
    else:
        log(f"LIGHT-PATH GATE FAILED: {why}. Do not download years without reference with this band "
            f"subset before this is explained ({path.name})")
    if l2a_nodata and not passed and not nodata_ok:
        log("  --l2a-nodata-equal: " + (
            f"{enc.get('other')} differing pixel-date(s) are NOT no data in both stores (worst dates "
            f"{enc.get('worst_dates_other')}); {enc.get('nodata_encoding')} are encoding only"
            if enc.get("checked", True) and "other" in enc else
            f"not applied ({enc.get('reason')})"))
    out = {"store": str(other), "passed": passed, "why": why, "verdict": cmp_["verdict"],
           "passed_on_common_dates": passed or common_only, "layout_differs": layout_bad,
           "dates": {k: cmp_["dates"][k] for k in ("store", "other", "common")},
           "checks": checks, "report": str(path)}
    if l2a_nodata:
        out["passed_up_to_nodata_encoding"] = nodata_ok
    return out


def light_gate_passed(root, bands):
    """The complete stores of ``bands`` under ``root`` (any site) accepted by
    the light-path gate (``verification`` how NINE_BAND_HOW, or
    NINE_BAND_NODATA_HOW with --l2a-nodata-equal): ``[path]``."""
    out = []
    for site, years in zc.available(root).items():
        for y in years:
            p = zc.store_path(site, y, root)
            attrs = zc.store_attrs(p)
            acc = _json_attr(attrs.get("verification"))
            if (isinstance(acc, dict) and acc.get("how") in LIGHT_GATE_HOWS
                    and list(_json_attr(attrs.get("bands")) or []) == list(bands)):
                out.append(str(p))
    return out


def light_gate_reports(root, bands):
    """The light-path gate reports (``<year>.vs_nine_band.json``) of stores of
    ``bands`` under ``root`` (any site) whose store is still there:
    ``[(path of the report, report)]``."""
    base = Path(root)
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        for p in sorted(d.glob("*.vs_nine_band.json")):
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            store = Path(str(rep.get("store") or ""))
            got = rep.get("bands") or _json_attr(zc.store_attrs(store).get("bands"))
            if store.is_dir() and list(got or []) == list(bands):
                out.append((p, rep))
    return out


def light_gate_common_dates(root, bands):
    """The stores of ``bands`` under ``root`` whose light-path gate failed on
    the DATES only (``passed_on_common_dates``: every common date bitwise, the
    same offset, delivered shape and input baselines on them; the dates differ,
    e.g. the CDSE catalogue changed since the nine-band job): ``[path]``."""
    return [str(rep.get("store")) for _, rep in light_gate_reports(root, bands)
            if not rep.get("passed") and rep.get("passed_on_common_dates")
            and not rep.get("layout_differs")
            and (rep.get("comparison") or {}).get("bitwise_on_common_dates")]


def subset_gate_problem(req, ctx=None):
    """Why the job of a band-subset year may not be bought yet (None when it
    may): a year without a complete nine-band store of its own under
    ``req.main_root`` (nothing to compare it with) waits until a store of the
    same bands under the store root, of any site, has passed the light-path
    gate (``light_gate_passed``: the cheap verification year comes first;
    or ``light_gate_common_dates``: it failed on the dates alone), unless
    ``ctx.without_light_gate`` (--without-light-gate, recorded); and, at the
    EXTRACTION_GATE_SITES, until the converter's 2023 extraction gate of the
    site passed or the user accepted it (``extraction_gate_problem``)."""
    return light_wait_problem(req, ctx) or extraction_wait_problem(req, ctx)


def _own_nine_band_store(req):
    return req.main_root is not None and zc.is_complete(zc.store_path(req.site, req.year,
                                                                      req.main_root))


def extraction_wait_problem(req, ctx=None):
    """``extraction_gate_problem`` for a year without a nine-band store of
    its own (a gate year never waits)."""
    if not req.is_subset or _own_nine_band_store(req):
        return None
    return extraction_gate_problem(req, ctx)


def light_wait_problem(req, ctx=None):
    """The light-path part of ``subset_gate_problem``."""
    if not req.is_subset or _own_nine_band_store(req):
        return None                          # a gate year itself
    if not getattr(ctx, "without_light_gate", False) and not (
            light_gate_passed(req.root, req.bands) or light_gate_common_dates(req.root, req.bands)):
        layout = [str(rep.get("store")) for _, rep in light_gate_reports(req.root, req.bands)
                  if rep.get("layout_differs")]
        return (f"no store of the bands {' '.join(req.bands)} under {req.root} has passed the "
                f"light-path gate yet: first download a year that has a complete nine-band store "
                f"under {req.main_root} (the cheap one: escalda --years 2023 --bands "
                f"{','.join(req.bands)}), which must be accepted as '{NINE_BAND_HOW}'"
                + (f" (the gate of {layout} FAILED on the layout: the nine-band offsets do not hold "
                   f"for these bands; each site needs a subset year its reference covers first)"
                   if layout else "")
                + "; --without-light-gate buys this year anyway (recorded)")
    return None


def extraction_gate_path(site, ctx=None):
    """``<EXTRACTION_GATE_DIR>/<site>_gate_2023.json`` (``ctx.extraction_gate_dir``,
    relative to the repository root unless absolute)."""
    d = Path(getattr(ctx, "extraction_gate_dir", None) or EXTRACTION_GATE_DIR)
    return (d if d.is_absolute() else ROOT / d) / f"{site}_gate_2023.json"


def extraction_gate_status(site, ctx=None):
    """``(ok, summary)`` of the converter's 2023 extraction gate of ``site``
    (experiments/light_to_extraction.py ``gate``): ok when it passed, or when
    it failed on extraction values only (the same dates, screening and static
    mask) and the user's acceptance is recorded in it (``accepted``)."""
    p = extraction_gate_path(site, ctx)
    if not p.is_file():
        return False, {"path": str(p), "present": False}
    try:
        g = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, {"path": str(p), "present": True, "readable": False, "why": str(exc)}
    only_values = bool((g.get("dates") or {}).get("equal")
                       and (g.get("screening") or {}).get("equal_record_dates")
                       and (g.get("extraction") or {}).get("keep_px_in_static_mask") == 0)
    accepted = g.get("accepted") if isinstance(g.get("accepted"), dict) else None
    dn = g.get("legacy_dn") or {}
    s = {"path": str(p), "present": True, "ok": g.get("ok"), "strict_ok": g.get("strict_ok"),
         "only_extraction_differences": only_values, "accepted_by": (accepted or {}).get("by"),
         "store": (g.get("store") or {}).get("path"), "utc": g.get("utc"),
         "legacy_dn_differ_pct": dn.get("differ_pct_of_valid_both"),
         "decision": g.get("decision")}
    return bool(g.get("ok") is True or (only_values and accepted)), s


def extraction_gate_problem(req, ctx=None):
    """Why a band-subset year without a nine-band store of its own, at an
    EXTRACTION_GATE_SITES site, may not be bought yet (None when it may): the
    converter's free 2023 extraction gate of the site has not passed and the
    user has not accepted its extraction-value differences
    (``extraction_gate_status``), unless --without-extraction-gate
    (recorded). Credits are spent on a site only once its 2023 extraction is
    shown reproducible (the Ems 9-band 2023 store differs from the legacy cube
    on 0.74 % of its valid B03 / B08 pixel-dates)."""
    if (not req.is_subset or req.site not in EXTRACTION_GATE_SITES
            or getattr(ctx, "without_extraction_gate", False)):
        return None
    ok, s = extraction_gate_status(req.site, ctx)
    if ok:
        return None
    if not s["present"] or s.get("readable") is False:
        what = f"no readable 2023 extraction gate of {req.site} at {s['path']}"
    elif s["only_extraction_differences"]:
        what = (f"the 2023 extraction gate of {req.site} ({s['path']}) FAILED on extraction values "
                f"only (B03 / B08 vs the legacy cube: {s.get('legacy_dn_differ_pct')} % of the valid "
                f"pixel-dates differ): accepting them is the user's decision, `python -X utf8 -m "
                f"experiments.light_to_extraction gate {req.site} --accept-gate-differences` "
                f"(recorded)")
    else:
        what = (f"the 2023 extraction gate of {req.site} ({s['path']}) FAILED on the dates, the "
                f"screening or the static mask: no flag accepts it")
    return (f"{what}. Run the free steps first (`light_to_extraction reference {req.site}`, then "
            f"`gate {req.site}`); --without-extraction-gate buys this year anyway (recorded)")


def decide(req, rep):
    """``(accepted, how, why)`` of a written main store: the rule
    (``verification_decision``); a band subset whose year has a complete
    nine-band store is accepted as NINE_BAND_HOW when its light-path gate
    passed (``rep['nine_band_store']``, ``nine_band_gate``: it holds what an
    accepted store holds), as NINE_BAND_NODATA_HOW when it passed up to the
    no-data encoding (only with --l2a-nodata-equal), and otherwise decided by
    the rule, the gate's failure said."""
    ok, how, why = verification_decision(rep)
    gate = rep.get("nine_band_store") if req.is_subset else None
    if not gate:
        return ok, how, why
    if gate.get("passed"):
        return True, NINE_BAND_HOW, (f"the light-path gate passed against {gate.get('store')} "
                                     f"({gate.get('why')}); against the reference: {how} ({why})")
    if gate.get("passed_up_to_nodata_encoding"):        # only with --l2a-nodata-equal
        return True, NINE_BAND_NODATA_HOW, (
            f"--l2a-nodata-equal: the light-path gate passed up to the no-data encoding against "
            f"{gate.get('store')} ({gate.get('why')}); against the reference: {how} ({why})")
    return ok, how, f"{why}; the LIGHT-PATH GATE FAILED ({gate.get('why')})"


def band_weight(bands, native=False):
    """The input weight of ``bands`` in a cost estimate: their number (each
    counted on the 10 m output grid), or (``native``) the 20 m bands at their
    native resolution, a quarter of a 10 m band each (nine: 5.25)."""
    if not native:
        return float(len(bands))
    return float(sum((RESOLUTION_M / float(zc.NATIVE_RES_M.get(b, RESOLUTION_M))) ** 2
                     for b in bands))


def subset_bytes_per_px_date(bands):
    """Bytes on disk per pixel and date of ``bands`` (BYTES_PER_BAND_PIXEL_DATE)."""
    return float(sum(BYTES_PER_BAND_PIXEL_DATE.get(b, 2.0) for b in bands))


def _input_mpx(usage):
    """A job's measured input in megapixels (``usage.input_pixel``: a value
    with its unit, mega-pixel on the CDSE backend), or None."""
    v = usage.get("input_pixel") if isinstance(usage, dict) else None
    unit = "mega-pixel"
    if isinstance(v, dict):
        unit, v = str(v.get("unit") or unit).lower(), v.get("value")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v / 1e6 if unit in ("pixel", "pixels", "px") else v


def measured_costs(site, root):
    """The measured jobs of ``site`` under ``root``: per complete main store
    whose sidecar ``<year>.job.json`` holds the job's ``costs``, ``{"site",
    "year", "credits", "dates", "delivered_shape" (the store's
    ``delivered_shape_px``, else its frame), "bands", "input_mpx" (None
    unless the job recorded ``usage.input_pixel``), "job", "source"}``."""
    out = []
    if root is None:
        return out
    for year in zc.list_years(site, root):
        side = zc.sidecar_path(site, year, "job.json", root)
        try:
            job = json.loads(Path(side).read_text(encoding="utf-8")).get("job") or {}
            credits = float(job.get("costs"))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if not credits > 0:
            continue
        meta = (zc._consolidated(zc.store_path(site, year, root)) or {}).get("metadata", {})
        attrs = meta.get(".zattrs", {})
        bands = _json_attr(attrs.get("bands")) or list(BANDS)
        tyx = (meta.get(f"{bands[0]}/.zarray") or {}).get("shape") or []
        shape = _json_attr(attrs.get("delivered_shape_px"))
        if not (isinstance(shape, list) and len(shape) == 2):
            shape = tyx[1:3]
        if len(tyx) != 3 or len(shape) != 2:
            continue
        out.append({"site": site, "year": int(year), "credits": credits, "dates": int(tyx[0]),
                    "delivered_shape": [int(shape[0]), int(shape[1])], "bands": list(bands),
                    "input_mpx": _input_mpx(job.get("usage")), "job": job.get("id"),
                    "source": str(side)})
    return out


def cost_calibration(req, roots):
    """The jobs whose cost per megapixel estimates ``req``'s: ``(entries,
    kind)``. The site's measured jobs (``measured_costs`` under ``roots``) of
    the same bands, else of other bands (the nine), else QUOTED_COSTS of the
    site, else QUOTED_COSTS of every site (a site never measured)."""
    measured = [e for r in roots if r is not None for e in measured_costs(req.site, r)]
    same = [e for e in measured if list(e["bands"]) == list(req.bands)]
    if same:
        return same, "measured on the site's jobs of the same bands"
    if measured:
        return measured, "measured on the site's jobs of other bands"
    quoted = [dict(e, bands=list(BANDS), input_mpx=None, source="QUOTED_COSTS")
              for e in QUOTED_COSTS if e["site"] == req.site]
    if quoted:
        return quoted, "quoted for the site's nine-band jobs (QUOTED_COSTS)"
    return ([dict(e, bands=list(BANDS), input_mpx=None, source="QUOTED_COSTS")
             for e in QUOTED_COSTS],
            f"no job of {req.site} measured or quoted: the other sites' (QUOTED_COSTS)")


def _credits_per_px_date(e):
    """Credits per delivered pixel-date of a calibration entry (0 when empty)."""
    pe = float(e["delivered_shape"][0]) * float(e["delivered_shape"][1]) * float(e["dates"])
    return e["credits"] / pe if pe > 0 else 0.0


def measured_share(bands, root, main_root):
    """The cost of a job of ``bands`` against a nine-band job, MEASURED where a
    site has both: its jobs of ``bands`` (``measured_costs`` under ``root``)
    and its nine-band jobs (under ``main_root``), the ratio of their median
    credits per delivered pixel-date. ``[(site, share)]`` (empty when no site
    has both; e.g. the Westerschelde 2023 job of the light-path gate gives
    the share the other sites' estimates then use)."""
    out = []
    if root is None or main_root is None:
        return out
    for site in SITES:
        mine = [e for e in measured_costs(site, root) if list(e["bands"]) == list(bands)]
        nine = [e for e in measured_costs(site, main_root) if list(e["bands"]) == list(BANDS)]
        if mine and nine:
            a = float(np.median([_credits_per_px_date(e) for e in mine]))
            b = float(np.median([_credits_per_px_date(e) for e in nine]))
            if a > 0 and b > 0:
                out.append((site, round(a / b, 4)))
    return out


def credit_estimate(bands, px, n_dates, entries, share=None):
    """The credits of a job of ``bands`` delivering ``px`` pixels on
    ``n_dates`` dates, at the median cost per input megapixel of the
    calibration ``entries`` (``cost_calibration``), the input counted two
    ways: every band on the 10 m output grid (``credits_grid``), the 20 m
    bands at their native resolution (``credits_native``); ``credits_lo`` /
    ``credits_hi`` the two (one number when the calibration has the same
    bands). ``best``: with a calibration of other bands (the nine), the
    ``share`` of their cost MEASURED at another site (``measured_share``)
    when given, else the way a calibration job's measured input
    (``usage.input_pixel``, per pixel-date of its delivery: ``measured_k``)
    points to, within COST_CONVENTION_TOL; else None."""
    in_g = px * n_dates * band_weight(bands) / 1e6
    in_n = px * n_dates * band_weight(bands, native=True) / 1e6
    out = {"input_grid_mpx": in_g, "input_native_mpx": in_n, "credits_lo": None,
           "credits_hi": None, "credits_grid": None, "credits_native": None, "best": None,
           "convention": None, "rate_grid": None, "rate_native": None, "measured_k": None}
    rg, rn, ks = [], [], []
    for e in entries:
        pe = float(e["delivered_shape"][0]) * float(e["delivered_shape"][1]) * float(e["dates"])
        if pe <= 0:
            continue
        rg.append(e["credits"] / (pe * band_weight(e["bands"]) / 1e6))
        rn.append(e["credits"] / (pe * band_weight(e["bands"], native=True) / 1e6))
        if e.get("input_mpx"):
            ks.append((e["input_mpx"] / (pe / 1e6), band_weight(e["bands"]),
                       band_weight(e["bands"], native=True)))
    if not rg:
        return out
    rate_g, rate_n = float(np.median(rg)), float(np.median(rn))
    cg, cn = rate_g * in_g, rate_n * in_n
    out.update(rate_grid=rate_g, rate_native=rate_n, credits_grid=cg, credits_native=cn,
               credits_lo=min(cg, cn), credits_hi=max(cg, cn))
    other = [e for e in entries if list(e["bands"]) != list(bands)]
    if share is not None and other:
        rate = float(np.median([_credits_per_px_date(e) for e in other]))
        out.update(best=rate * px * n_dates * float(share), share=float(share),
                   convention=f"the cost share {float(share):.3f} of the nine bands measured on "
                              f"a job of these bands")
        return out
    if ks:
        k = float(np.median([v[0] for v in ks]))
        wg, wn = ks[0][1], ks[0][2]
        out["measured_k"] = round(k, 4)
        if abs(k - wg) <= COST_CONVENTION_TOL * wg:
            out.update(convention="every band on the 10 m grid", best=cg)
        elif abs(k - wn) <= COST_CONVENTION_TOL * wn:
            out.update(convention="the 20 m bands at their native resolution", best=cn)
    return out


def expected_delivered_shape(req, frame_shape):
    """``([rows, cols], source)``: the (padded) arrays a job of ``req`` should
    deliver, for its cost: the shape the site's stores of the same extent
    recorded (``recorded_delivered_shapes``: the store root, else the
    nine-band root; the most frequent), else QUOTED_COSTS's, else the frame."""
    for r in (req.root, req.main_root):
        shapes = recorded_delivered_shapes(req, r) if r is not None else {}
        if shapes:
            s = max(shapes, key=lambda k: len(shapes[k]))
            return [int(s[0]), int(s[1])], f"as delivered to the {r} stores {sorted(shapes[s])}"
    quoted = [e for e in QUOTED_COSTS if e["site"] == req.site]
    if quoted:
        return ([int(v) for v in quoted[0]["delivered_shape"]],
                "as delivered to the site's nine-band jobs (QUOTED_COSTS)")
    return [int(v) for v in frame_shape], "the frame: no delivered shape recorded"


def _calibration_line(entries, kind):
    return (kind + ": " + "; ".join(
        f"{e['site']} {e.get('year') or '?'} {e['credits']:g} credits, {e['dates']} dates x "
        f"{e['delivered_shape'][0]} x {e['delivered_shape'][1]} px x {len(e['bands'])} bands"
        + (f", input_pixel {e['input_mpx']:.0f} Mpx" if e.get("input_mpx") else "")
        for e in entries[:6]) + (" ..." if len(entries) > 6 else ""))


def subset_dry_run(req, ctx, frame_shape, n_dates, say):
    """The dry run of a band-subset request (no network): the registration it
    will meet (the reference netCDF covering the year; else the registered
    stores of the site under the store root and the nine-band root that will
    check its offset), the light-path gate, the expected input megapixels and
    the credits (``credit_estimate``). Returns the estimate."""
    cfg = SITE_CONFIG[req.site]
    ref = ROOT / cfg["reference"]
    p0, p1 = cfg["reference_period"]
    roots = [Path(req.root)] + ([Path(req.main_root)] if req.main_root is not None else [])
    if ref.is_file() and req.start < p1 and req.end > p0:
        say(f"registration: against the reference netCDF {ref.name}, which covers the year (as "
            f"the nine bands)")
    else:
        why = (f"{ref.name} is not on this machine" if not ref.is_file() else
               f"{ref.name} covers [{p0}, {p1})")
        ex = getattr(ctx, "offset_px", None)
        grid = request_grid(req)
        found = []
        for r in roots if grid is not None else []:
            found += [c for c in sibling_candidates(req, grid, (0, 0), "", r,
                                                    same_request=False)[0]]
        say(f"registration: no reference for [{req.start}, {req.end}) ({why}): "
            + (f"--offset-px {ex[0]},{ex[1]}" if ex else
               "the offset of a registered store delivered in the same array shape")
            + ", taken only when the median composites of a registered store CONFIRM it on a "
              "delivery of that store's array shape")
        for c in found:
            say(f"  registered store {c['path']}: offset {c['offset']}, delivered "
                f"{c['delivered_shape']}")
        offs = sorted({tuple(c["offset"]) for c in found})
        if not found:
            say(f"  NO registered store of {req.site} with this extent and grid under "
                f"{[str(r) for r in roots]}: the year would be refused")
        elif ex and tuple(ex) not in offs:
            say(f"  WARNING: --offset-px {ex[0]},{ex[1]} is not the offset of the site's "
                f"registered stores {offs}")
    if req.main_root is not None and zc.is_complete(zc.store_path(req.site, req.year,
                                                                  req.main_root)):
        say(f"light-path gate: after writing, against the complete nine-band store "
            f"{zc.store_path(req.site, req.year, req.main_root)} ({' '.join(req.bands)} bitwise)")
    else:
        passed = light_gate_passed(req.root, req.bands)
        common = light_gate_common_dates(req.root, req.bands)
        if passed:
            say(f"light-path gate: passed by {passed}: this year may be bought")
        elif common:
            say(f"light-path gate: failed on the dates only by {common} (every common date "
                f"bitwise, the same layout): this year may be bought (recorded)")
        elif getattr(ctx, "without_light_gate", False):
            say("light-path gate: NOT passed yet by any store of these bands; "
                "--without-light-gate: this year would be bought anyway (recorded)")
        else:
            say(f"light-path gate: NOT passed yet by any store of {' '.join(req.bands)} under "
                f"{req.root}: this year would be REFUSED before any job (the gate year first, "
                f"e.g. escalda 2023; or --without-light-gate)")
        if req.site in EXTRACTION_GATE_SITES:
            ok, s = extraction_gate_status(req.site, ctx)
            if ok:
                say(f"extraction gate (experiments.light_to_extraction): {s['path']} "
                    + ("passed" if s.get("strict_ok") else f"accepted by the user ({s['accepted_by']})")
                    + ": this year may be bought")
            elif getattr(ctx, "without_extraction_gate", False):
                say(f"extraction gate: not passed ({s['path']}); --without-extraction-gate: this "
                    f"year would be bought anyway (recorded)")
            else:
                say(f"extraction gate: NOT passed ({extraction_gate_problem(req, ctx)}): this year "
                    f"would be REFUSED before any job")
    shape, ssrc = expected_delivered_shape(req, frame_shape)
    entries, kind = cost_calibration(req, roots)
    shares = measured_share(req.bands, req.root, req.main_root)
    share = float(np.median([s for _, s in shares])) if shares else None
    est = credit_estimate(req.bands, shape[0] * shape[1], n_dates, entries, share=share)
    est["delivered_shape"] = shape
    say(f"input: {shape[0]} x {shape[1]} px ({ssrc}) x ~{n_dates:.0f} dates x "
        f"{len(req.bands)} bands = ~{est['input_grid_mpx']:,.0f} Mpx counted on the 10 m grid "
        f"(~{est['input_native_mpx']:,.0f} Mpx with the 20 m bands at their native resolution)")
    if est["credits_lo"] is None:
        say("credits: no calibration (no measured or quoted job)")
        return est
    rng = (f"~{est['credits_lo']:,.0f}" if round(est["credits_lo"]) == round(est["credits_hi"])
           else f"~{est['credits_lo']:,.0f}-{est['credits_hi']:,.0f}")
    if est.get("share") is not None:
        best = (f"; best ~{est['best']:,.0f}: {est['convention']} ("
                + ", ".join(f"{s} {v}" for s, v in shares) + ")")
    elif est["best"] is not None:
        best = (f"; best ~{est['best']:,.0f}: the calibration jobs' measured input is "
                f"{est['measured_k']} Mpx per Mpx-date, {est['convention']}")
    elif est["measured_k"]:
        best = (f"; measured input of the calibration jobs {est['measured_k']} Mpx per Mpx-date, "
                f"neither way")
    else:
        best = ""
    say(f"credits: {rng} (cost per Mpx {est['rate_grid']:.4f} on the 10 m grid / "
        f"{est['rate_native']:.4f} native{best}; plus any fixed cost per job)")
    say(f"  calibration {_calibration_line(entries, kind)}")
    return est


# ─────────────────────────────────────────────────────────────────────────────
#  --renormalise: a store rebuilt from its kept raw delivery (no network, no job)
# ─────────────────────────────────────────────────────────────────────────────

def _raw_of(ctx, req):
    """The kept raw delivery of ``req``, checked: ``(part, title, ledger,
    current job entry, job info, results metadata, process graph, its
    source)``. RawMissing when ``<year>.part`` holds no complete download;
    FormatError when it was made for another request (another job title:
    extent, bands, format or its options)."""
    part = Path(req.part)
    if not (part / ASSETS_DONE).is_file():
        raise RawMissing(
            f"{req.tag}: no raw delivery to rebuild {req.store.name} from: {part} "
            + ("holds no complete download" if part.exists() else "does not exist")
            + f" (it is removed once a store is complete unless --keep-raw); a new job is "
              f"needed: move {req.store.name} aside and run the download again")
    flat = build_process(req, None, ctx.backend_format, ctx.format_options).flat_graph()
    title = job_title(req, graph_hash(flat))
    made = recorded_title(part)
    if made != title:
        raise FormatError(f"{req.tag}: {part} was made for the job title {made!r}, the current "
                          f"request is {title!r} (another extent, bands, format or format "
                          f"options): it is not this request's delivery; nothing done")
    led = Ledger(part)
    cur = led.current() or {}
    info = cur.get("info") or {"id": cur.get("job_id")}
    meta_path = part / "job-results.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if cur.get("process_graph"):
        graph, gsrc = cur["process_graph"], f"submitted with job {cur.get('job_id')}"
    else:
        graph, gsrc = flat, "built offline (the submitted graph was not recorded)"
    return part, title, led, cur, info, meta, graph, gsrc


def _tree_size(path):
    """Bytes of the files under ``path`` (0 when absent)."""
    path = Path(path)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def keep_sidecars(req, aside, log=print):
    """Copy the sidecars of the store about to be superseded (``verify.json``,
    ``format.txt``, ``job.json``) beside them as ``<name>.superseded-<stamp>``
    with the stamp of ``aside`` (the store's own superseded name), before
    anything rewrites them: the evidence of the old store (e.g. its verify.json
    saying 'differs') survives the rebuild. Returns the copies."""
    stamp = Path(aside).name.split(".superseded-", 1)[1]
    out = []
    for k in ("verify.json", "format.txt", "job.json"):
        p = Path(req.sidecar(k))
        if p.is_file():
            dst = p.with_name(f"{p.name}.superseded-{stamp}")
            shutil.copy2(p, dst)
            out.append(dst)
    if out:
        log(f"sidecars of {req.store.name} kept as {[p.name for p in out]}")
    return out


def _ledger_note(led, note, req, info, meta):
    """Append ``note`` to the part's ledger and rewrite ``<year>.job.json``."""
    led.data.setdefault("notes", []).append(note)
    led.save()
    zc.write_json({"ledger": led.data, "job": info, "results_metadata": meta},
                  req.sidecar("job.json"))


def renormalise_request(ctx, req, log, dry_run=False):
    """--renormalise one main store: rebuild it from its kept raw delivery
    (``<year>.part``) with the registration, WITHOUT any network call or job.

    The raw is normalised again (``normalise`` with ``registration_plan``)
    and written; the store there (complete or not) is moved aside to
    ``<year>.zarr.superseded-<UTC>`` (never deleted) only once the new one is
    written whole (``write_store``): a refusal, an error or an interruption
    before leaves it in place. Its sidecars (``verify.json``, ``format.txt``,
    ``job.json``) are first copied with the same suffix (``keep_sidecars``).
    The new store is verified (``verify_and_complete``: refused -> not
    complete) and ``<year>.verify.json``, ``<year>.format.txt``, a note in the
    part's ledger (``job.json``: ``notes``, also on failure) and
    ``<year>.job.json`` updated. A store written whole from this delivery but
    pending its verification (``reusable_pending``: e.g. refused before, and
    now --accept-differences) is verified again instead of written again.
    The free disk space must hold 1.2 times the store there (it is kept). The
    raw is kept. ``dry_run``: only the measurement (the offset, its dates and
    r, the 20 m checks) and what would be done; nothing written. Refused when
    the raw is missing (RawMissing) or was made for another request."""
    t0 = time.time()
    if req.is_angles:
        log("--renormalise: the angles stores are not registered (no B08): skipped")
        return {"status": "skipped", "why": "angles"}
    if req.is_subset:               # a band subset never writes into a root of other bands
        why = subset_root_problem(req.root, req.bands, req.main_root)
        if why:
            raise FormatError(f"{req.tag}: {why}; nothing done")
    part, title, led, cur, info, meta, graph, gsrc = _raw_of(ctx, req)
    log(f"--renormalise{' --dry-run' if dry_run else ''}: rebuilding {req.store.name} from the raw "
        f"delivery of job {info.get('id')} in {part} (no network, no job)")
    old = Path(req.store) if Path(req.store).exists() else None
    state = ("complete" if old is not None and req.complete() else
             "NOT complete" if old is not None else None)
    reuse, reuse_why = reusable_pending(req, info.get("id") or cur.get("job_id"))
    need = _tree_size(old) if old is not None else 0
    free = shutil.disk_usage(zc.site_dir(req.site, req.root)).free
    space = f"~{_size(need)} needed beside the kept store, {_size(free)} free"
    short = old is not None and not reuse and free < 1.2 * need
    if dry_run and reuse:
        log(f"DRY RUN: {reuse_why}: it would be verified again (not rewritten) and completed if "
            f"accepted; nothing done")
        return {"status": "dry run", "would": "verify the pending store again"}
    ref = reference_grid(req.site, root=req.main_root if req.is_subset and req.main_root is not None
                         else req.root, log=log)
    plan = registration_plan(req, ctx, ref)
    kw = dict(job_info=info, results_meta=meta, ref_grid=ref, allow_offset=ctx.allow_offset,
              nodata=ctx.nodata, allow_padding=ctx.allow_padding,
              backend_format=ctx.backend_format, registration=plan)
    if dry_run:
        _, interp, _ = normalise(req, log, dry_run=True, **kw)
        reg = interp.registration or {}
        log(f"DRY RUN: registration {reg.get('method')}: offset {reg.get('offset')} of the frame "
            f"{reg.get('frame_shape')} in the delivered {reg.get('delivered_shape')} px (their x/y: "
            f"{reg.get('coordinate_offset')})" + (f"; dates {reg.get('dates')}" if reg.get('dates')
                                                  else "")
            + (f"; r {reg.get('r')}" if reg.get("r") else ""))
        log(f"DRY RUN: {layer_20m_line(reg.get('layer_20m') or {'why': 'no offset measured'})}; "
            + (phase_line(reg["phase_20m"]) if reg.get("phase_20m") else "20 m phase: not checked"))
        for w in interp.warnings:
            log(f"DRY RUN: WARNING: {w}")
        log("DRY RUN: would move " + (f"{old} ({state}) to {old.name}.superseded-<UTC> once a new "
                                      f"{req.store.name} is written whole from {part}" if old
                                      else f"nothing ({req.store.name} absent); would write a new "
                                           f"{req.store.name} from {part}")
            + f" ({space}" + ("; NOT ENOUGH: it would be refused" if short else "")
            + f"), verify it against the reference and update {req.sidecar('verify.json').name}, "
              f"{req.sidecar('format.txt').name}, {req.sidecar('job.json').name} (the old ones "
              f"kept as <name>.superseded-<UTC>) and the ledger's notes; nothing done")
        return {"status": "dry run", "offset": reg.get("offset"), "method": reg.get("method"),
                "would_move": str(old) if old else None}
    note = {"utc": utcnow().isoformat(timespec="seconds"), "action": "--renormalise",
            "superseded_store": None}
    if reuse:
        note["action"] = "--renormalise (the pending store verified again, not rewritten)"
        try:
            res = finish_pending(ctx, req, log, part, info, t0, note=note)
        except BaseException as exc:
            note["failed"] = f"{type(exc).__name__}: {exc}"[:2000]
            raise
        finally:
            _ledger_note(led, note, req, info, meta)
        return dict(res, status="renormalised (pending store verified again)")
    if short:
        raise RuntimeError(f"{req.tag}: {space}: not enough to write the new store beside the kept "
                           f"one; nothing done")
    aside = superseded_name(req.store) if old is not None else None
    if aside is not None:
        note["sidecars_kept"] = [str(p) for p in keep_sidecars(req, aside, log)]
    try:
        ds, interp, lines = normalise(req, log, **kw)
        log_report(req, lines, log)
        reg = interp.registration or {}
        note.update(registration_method=reg.get("method"), delivered_offset_px=reg.get("offset"))
        attrs = provenance(req, info, meta, interp, ctx, graph, graph_source=gsrc, ref_grid=ref,
                           job_options=cur.get("job_options"))
        attrs.update({"renormalised_utc": note["utc"], "renormalised_from": str(part),
                      "superseded_store": str(aside) if aside else ""})
        log(f"writing {req.store} ({ds.sizes['t']} dates, {ds.sizes['y']} x {ds.sizes['x']} px, "
            f"{len(req.bands)} bands) from the raw delivery"
            + (f"; {old.name} stays in place until the new one is written whole, then moves to "
               f"{aside.name}" if aside else ""))
        write_store(ds, store_fills(interp, req.spec), req.store, attrs, log=log,
                    groups=interp.groups, chunks=req.spec.chunks, format_id=req.spec.format_id,
                    complete=False, supersede_to=aside, **_subset_write_kw(req))
        if aside is not None and aside.exists():
            note["superseded_store"] = str(aside)
        rep = verify_and_complete(ctx, req, log)
        note["verification"] = rep.get("acceptance")
    except BaseException as exc:             # KeyboardInterrupt too: the note says it failed
        note["failed"] = f"{type(exc).__name__}: {exc}"[:2000]
        raise
    finally:
        _ledger_note(led, note, req, info, meta)
    dates = zc.dates(ds)
    still = update_overpass(req, dates, log, network=False)
    size = _tree_size(req.store)
    log(f"store complete: {req.store} ({size / 1e9:.2f} GB on disk); the raw delivery stays in "
        f"{part}; done in {(time.time() - t0) / 60:.0f} min")
    return {"status": "renormalised", "dates": len(dates), "verdict": rep.get("verdict"),
            "accepted": (rep.get("acceptance") or {}).get("how"),
            "registration": f"{reg.get('method')} {reg.get('offset')}",
            "superseded": note["superseded_store"], "missing_overpass": still}


def renormalise_run(reqs, ctx, dry_run=False):
    """--renormalise [--dry-run] over the requests (main products; one site
    lock each, as a download run): the summary, exit code 0 / 1."""
    results, failed = {}, {}
    by_site = {}
    for r in reqs:
        by_site.setdefault(r.site, []).append(r)
    for site, site_reqs in by_site.items():
        lock = None
        if not dry_run:
            try:
                lock = SiteLock(zc.site_dir(site, site_reqs[0].root))
                lock.__enter__()
            except LockedError as exc:
                make_log(site_reqs[0])(str(exc))
                for r in site_reqs:
                    failed[r.tag] = str(exc)
                continue
        try:
            for req in site_reqs:
                log = make_log(req) if not dry_run else (lambda m, _t=req.tag: print(
                    f"[{_t}] {m}", flush=True))
                try:
                    results[req.tag] = renormalise_request(ctx, req, log, dry_run=dry_run)
                except Exception as exc:          # one year never stops the others
                    failed[req.tag] = f"{type(exc).__name__}: {exc}"
                    log(f"FAILED: {type(exc).__name__}: {exc}")
        finally:
            if lock is not None:
                lock.__exit__(None, None, None)
    print("\nSUMMARY (--renormalise" + (" --dry-run)" if dry_run else ")"), flush=True)
    for tag, r in results.items():
        print(f"  {tag}: {r}", flush=True)
    for tag, e in failed.items():
        print(f"  {tag}: FAILED {e}", flush=True)
    return 1 if failed else 0


# ─────────────────────────────────────────────────────────────────────────────
#  dry run and verify-only
# ─────────────────────────────────────────────────────────────────────────────

def _size(nbytes):
    """Bytes as a short human string (MB below 1 GB)."""
    return f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.1f} MB"


def _size_small(nbytes):
    """``_size``, with kB below 1 MB (the angles of a short window)."""
    return _size(nbytes) if nbytes >= 1e6 else f"{nbytes / 1e3:.1f} kB"


def extent_line(req):
    """``<EPSG, edges, size>  <- <origin>`` of a request (the dry run)."""
    if req.extent is None:
        return f"lon/lat box {req.bbox} (EPSG:4326)  <- {req.extent_origin or 'the lon/lat box'}"
    return f"{describe_extent(req.extent, req.spec.resolution)}  <- {req.extent_origin}"


def dry_run(reqs, ctx):
    """Process graphs, outputs, their state and a size estimate per
    product, and the spatial extent each job requests (EPSG, edges, size,
    and whether it is the frame of a reference netCDF or comes from
    SNAPPING the lon/lat box); no network."""
    totals = {}
    shown, guarded = set(), set()
    refs, angle_refs = {}, {}
    ext_seen = {}
    credits = {}                    # band subsets: site -> [lo, hi, best-or-None, site-years]

    def say(m):
        print(f"   {m}", flush=True)

    for req in reqs:
        spec, angles = req.spec, req.is_angles
        sr = build_process(req, None, ctx.backend_format, ctx.format_options)
        flat = sr.flat_graph()
        title = job_title(req, graph_hash(flat))
        first = (req.site, spec.name) not in shown
        print(f"\n== {req.tag}: window [{req.start}, {req.end})", flush=True)
        if req.extent is not None:
            print(f"   spatial extent: {describe_extent(req.extent, spec.resolution)}", flush=True)
            print(f"     origin: {req.extent_origin}", flush=True)
            if first:
                print(f"     source: {req.extent_source}", flush=True)
            print(f"     lon/lat box (kept in the provenance, not requested): {req.bbox}", flush=True)
        else:
            print(f"   spatial extent: the lon/lat box {req.bbox} (EPSG:4326)", flush=True)
            print(f"     origin: {req.extent_origin or 'the lon/lat box'}", flush=True)
        ext_seen.setdefault((req.site, spec.name), req)
        print(f"   job '{title}', format {ctx.backend_format} {ctx.format_options or {}}, "
              f"job options {ctx.job_options}", flush=True)
        if (req.site, spec.name) not in shown:
            print("   process graph:\n" + "\n".join("     " + ln for ln in
                                                     json.dumps(flat, indent=2).splitlines()),
                  flush=True)
            shown.add((req.site, spec.name))
        else:
            print("   process graph: as above, temporal_extent "
                  f"{flat['loadcollection1']['arguments']['temporal_extent']}", flush=True)
        complete = (not req.test) and req.complete()
        state = ("COMPLETE (skipped)" if complete else
                 "partial download present" if Path(req.part).exists() else "to download")
        led = Ledger(req.part).current() if (Path(req.part) / LEDGER).is_file() else None
        print(f"   outputs: {req.store} [{state}]" + (f"; ledger job {led['job_id']} "
                                                       f"({led.get('status')})" if led else ""),
              flush=True)
        print(f"            {req.sidecar('verify.json').name}, {req.sidecar('format.txt').name}, "
              f"{req.sidecar('job.json').name}"
              + ("" if angles else f", {zc.overpass_path(req.site, req.root)}"), flush=True)
        if req.site not in refs:
            refs[req.site] = None if req.test else reference_summary(req.site)
        if (req.site, spec.name) not in guarded:
            guarded.add((req.site, spec.name))
            ref_root = _grid_root(req)
            try:
                if angles:
                    angle_refs[req.site] = angles_reference_grid(req.site, root=ref_root, log=say)
                else:
                    reference_grid(req.site, root=ref_root, log=say)
            except Exception as exc:          # informative only
                print(f"   grid guard: could not prepare the reference grid ({exc})", flush=True)
        ref = refs[req.site]
        if req.extent is not None:         # the store is the requested extent (cropped to it)
            shape, gsrc = frame_shape(req.extent, spec.resolution), "the requested extent"
        elif angles:
            ag = angle_refs.get(req.site)
            if ag is not None and not req.test:
                shape, gsrc = [len(ag["y"]), len(ag["x"])], ag["path"]
            else:
                g = predict_grid(req.bbox, res=spec.resolution)
                shape, gsrc = g["shape"], (f"bbox envelope on the {spec.resolution} m lattice, "
                                           f"EPSG:{g['epsg']}")
        elif ref and not req.test:
            shape, gsrc = ref["shape"], f"existing cube {Path(ref['path']).name}"
        else:
            g = predict_grid(req.bbox)
            shape, gsrc = g["shape"], f"bbox envelope on the 10 m lattice, EPSG:{g['epsg']}"
        days = int((np.datetime64(req.end) - np.datetime64(req.start)).astype(int))
        if ref and req.year in ref["per_year"] and not req.test and days >= 365:
            n = ref["per_year"][req.year]
            dsrc = "dates of that year in the existing cube"
        else:
            n = TYPICAL_DATES.get(req.year, 145) * min(1.0, days / 365.0)
            dsrc = "typical count for the year"
        px = shape[0] * shape[1]
        raw = px * n * spec.raw_itemsize * len(req.bands)
        bpd = subset_bytes_per_px_date(req.bands) if req.is_subset else spec.disk_per_px_date
        disk = px * n * bpd
        total = totals.setdefault(spec.name, {"px_dates": 0.0, "raw": 0.0, "disk": 0.0,
                                              "biggest_raw": 0.0, "todo": 0, "of": 0})
        total["of"] += 1
        if not complete:
            total["px_dates"] += px * n
            total["raw"] += raw
            total["disk"] += disk
            total["biggest_raw"] = max(total["biggest_raw"], raw)
            total["todo"] += 1
            if req.is_subset:               # a band subset KEEPS its raw delivery (raw_after_store)
                total["raw_kept"] = total.get("raw_kept", 0.0) + raw
        if angles:
            print(f"   grid {shape[0]} x {shape[1]} cells of {spec.resolution} m ({gsrc}); "
                  f"~{n:.0f} dates ({dsrc})", flush=True)
            print(f"   size: {px * n:.0f} cell-dates; {_size_small(raw)} as float32 "
                  f"({len(req.bands)} angles), ~{_size_small(disk)} on disk (estimate "
                  f"{spec.disk_per_px_date} B per cell-date)", flush=True)
            continue
        print(f"   grid {shape[0]} x {shape[1]} px ({gsrc}); ~{n:.0f} dates ({dsrc})", flush=True)
        if not req.is_subset:
            print(f"   size: {px * n / 1e9:.3f} G pixel-dates; {_size(raw)} as int16, "
                  f"~{_size(disk)} on disk (estimate {BYTES_PER_PIXEL_DATE} B per pixel-date)",
                  flush=True)
            continue
        print(f"   size: {px * n / 1e9:.3f} G pixel-dates; {_size(raw)} as int16 "
              f"({len(req.bands)} bands), ~{_size(disk)} on disk (estimate {bpd:.2f} B per "
              f"pixel-date for {' '.join(req.bands)})", flush=True)
        est = subset_dry_run(req, ctx, shape, n, say)
        if not complete and est.get("credits_lo") is not None:
            c = credits.setdefault(req.site, [0.0, 0.0, 0.0, 0, 0])
            c[0] += est["credits_lo"]
            c[1] += est["credits_hi"]
            c[2] += est["best"] if est.get("best") is not None else 0.0
            c[3] += 1 if est.get("best") is not None else 0
            c[4] += 1
    root = Path(reqs[0].root) if reqs else zc.cubes_root()
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print("", flush=True)
    if ext_seen:
        print("SPATIAL EXTENTS requested (spatial_extent of load_collection; west south east north "
              "in the CRS given; origin):", flush=True)
        refused = []
        for (site, pname), r in ext_seen.items():
            no = refuse_snapped(r) is not None and not ctx.allow_snapped
            if no:
                refused.append(r)
            print(f"  {site:<12} {pname:<6}{' TEST' if r.test else ''}  {extent_line(r)}"
                  + ("  [REFUSED: no reference netCDF; --allow-snapped-frame]" if no else ""),
                  flush=True)
        if refused:
            print("REFUSED before any job (production lands only on a reference frame): "
                  + ", ".join(f"{r.site} {r.product}" for r in refused) + ". Missing here: "
                  + ", ".join(sorted({str(ROOT / SITE_CONFIG[r.site]['reference'])
                                      for r in refused}))
                  + " (or pass --allow-snapped-frame)", flush=True)
    for name, t in totals.items():
        if name == MAIN.name and t.get("raw_kept"):     # band subsets: the raw deliveries stay
            need = t["disk"] + t["raw_kept"]
            print(f"TOTAL main: {t['todo']} store(s) to make (of {t['of']}): "
                  f"{t['px_dates'] / 1e9:.2f} G pixel-dates, {_size(t['raw'])} as int16, "
                  f"~{_size(t['disk'])} on disk; a band subset KEEPS its raw download (zip + "
                  f"extracted, ~{_size(t['raw_kept'])} for these site-years, cumulative): ~"
                  f"{_size(need)} in all" + ("" if need < free else
                                             f"; WARNING: more than the {_size(free)} free"),
                  flush=True)
        elif name == MAIN.name:
            print(f"TOTAL main: {t['todo']} store(s) to make (of {t['of']}): "
                  f"{t['px_dates'] / 1e9:.2f} G pixel-dates, {_size(t['raw'])} as int16, "
                  f"~{_size(t['disk'])} on disk; while a year is processed its raw download also "
                  f"needs up to ~{_size(t['biggest_raw'])} (zip + extracted, removed after it)",
                  flush=True)
        else:
            print(f"TOTAL {name}: {t['todo']} store(s) to make (of {t['of']}): "
                  f"{t['px_dates'] / 1e6:.3f} M cell-dates, {_size_small(t['raw'])} as "
                  f"{PRODUCTS[name].store_dtype}, ~{_size_small(t['disk'])} on disk; one small "
                  f"job each", flush=True)
    if credits:                     # band subsets only
        lo, hi = sum(c[0] for c in credits.values()), sum(c[1] for c in credits.values())
        n = sum(c[4] for c in credits.values())
        print(f"TOTAL credits of the band-subset stores to make: ~{lo:,.0f}-{hi:,.0f} for {n} "
              f"site-year(s), plus any fixed cost per job (per site: "
              + "; ".join(f"{s} ~{c[0]:,.0f}-{c[1]:,.0f} ({c[4]})"
                          + (f", best ~{c[2]:,.0f} on {c[3]}" if c[3] else "")
                          for s, c in credits.items())
              + "); an account holds ~10,000 a month", flush=True)
    if getattr(ctx, "expect_account", None):
        print(f"--expect-account {ctx.expect_account}: not checked (a dry run makes no network call)",
              flush=True)
    print(f"free under {probe}: {_size(free)}", flush=True)
    return 0


def verify_only(reqs, ctx=None):
    bad = 0
    for req in reqs:
        log = make_log(req)
        if not req.complete():
            log(f"{req.store}: not downloaded (or not complete)")
            bad += 1
            continue
        if req.is_angles:
            if verify_angles(req, log)["problems"]:
                bad += 1
            continue
        old = pre_registration_warning(req)
        if old:
            log(old)
        rep = verify_request(req, log, ctx)
        ok, how, why = decide(req, rep)
        was = _json_attr(zc.store_attrs(req.store).get("verification"))
        flagged = (not ok and isinstance(was, dict)
                   and was.get("how") in ("accepted with --accept-differences",
                                          NINE_BAND_NODATA_HOW))
        # the decision goes into verify.json again, with the one made at writing
        rep["acceptance"] = {"accepted": ok or flagged, "how": was.get("how") if flagged else how,
                             "why": why, "rule": dict(VERIFY_MIN_EQUAL),
                             "verdict": rep.get("verdict"), "checked_utc": rep.get("checked_utc"),
                             "at_writing": was, "by": "--verify-only"}
        zc.write_json(rep, req.sidecar("verify.json"))
        log(f"verification decision: {how} ({why})"
            + (f"; the store was accepted at writing: {was.get('how')}"
               if not ok and isinstance(was, dict) else "")
            + (("; accepted with --accept-differences, as recorded: not counted as a failure"
                if was.get("how") == "accepted with --accept-differences" else
                f"; accepted at writing as '{was.get('how')}' (--l2a-nodata-equal), as recorded: not "
                f"counted as a failure") if flagged else ""))
        ds = zc.open_store(req.store)
        dates = zc.dates(ds)
        ds.close()
        have = zc.read_overpass(req.site, req.root)
        miss = [d for d in dates if d not in have]
        log(f"overpass times: {len(dates) - len(miss)} of {len(dates)} dates"
            + (f"; missing {miss[:10]}" if miss else ""))
        if (not ok and not flagged) or miss:
            bad += 1
    return 1 if bad else 0


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sites", nargs="*", help=f"any of {', '.join(SITES)} (default: all)")
    ap.add_argument("--years", nargs="+", default=None,
                    help="years or ranges: 2023-2025 | 2017-2022 2025 | 2023,2025 "
                         "(default 2023-2025)")
    ap.add_argument("--test", action="store_true",
                    help=f"format test: a {TEST_SIZE_M} x {TEST_SIZE_M} m window "
                         f"({TEST_SIZE_M // RESOLUTION_M} x {TEST_SIZE_M // RESOLUTION_M} px) on the "
                         "Villaviciosa lattice, 2025-06-01..15, all bands and the angles (the "
                         "--products), into <root>/_test; prints a FORMAT REPORT per product")
    ap.add_argument("--odd-window", action="store_true",
                    help=f"with --test: the main product only, on a {TEST_ODD_SIZE_M} x "
                         f"{TEST_ODD_SIZE_M} m window ({TEST_ODD_SIZE_M // RESOLUTION_M} x "
                         f"{TEST_ODD_SIZE_M // RESOLUTION_M} px, odd) whose west and south edges "
                         "are lines 10 m past a multiple of 20 m (as the production frames' "
                         "east / south edges), under <root>/_test_odd: one more small job")
    ap.add_argument("--lonlat-extent", action="store_true",
                    help="request the site's (or the --test) lon/lat box as before 2026-10-13, "
                         "instead of the UTM extent on the reference lattice: for diagnosis only "
                         "(its grid is off the lattice and is refused unless --allow-grid-offset); "
                         "with --test it works under <root>/_test_lonlat")
    ap.add_argument("--products", default=None,
                    help="the stores to make, comma-separated: main (<year>.zarr: the nine bands "
                         f"at {RESOLUTION_M} m, int16) and/or angles (<year>.angles.zarr: sun and "
                         f"view angles on {ANGLES_RES_M} m cells, float32 degrees; one small "
                         "separate job). Default: both (with --bands: main, the only one). A "
                         "complete store is skipped, so a year whose main store exists only "
                         "fetches its angles")
    ap.add_argument("--bands", default=None, metavar="B03,B08,SCL",
                    help="opt-in BAND SUBSET of the main product (default: the nine bands): any "
                         f"of {' '.join(BANDS)} holding {' '.join(SUBSET_REQUIRED)}; the same "
                         "process, extent, window and registration with fewer bands (fewer "
                         "credits), its own job title, its stores under --store-root, never "
                         "beside the nine-band ones (whose --root is then only read: frame, "
                         "registered stores that check an offset, the light-path gate against "
                         "the same year's nine-band store, the cost calibration of --dry-run). "
                         "Not with --test, --odd-window, --lonlat-extent or the angles")
    ap.add_argument("--store-root", default=None,
                    help=f"with --bands: the root of the band-subset stores (default "
                         f"<repo>/{SUBSET_ROOT}); refused when it is the nine-band --root, lies "
                         f"inside it or holds it, or holds a store of another band set")
    ap.add_argument("--without-light-gate", action="store_true",
                    help="with --bands: buy a year that has no nine-band store of its own although "
                         "no store of these bands has passed the light-path gate yet (default: "
                         "refused before any job; the cheap gate year comes first, e.g. escalda "
                         "2023); recorded in the store")
    ap.add_argument("--l2a-nodata-equal", action="store_true",
                    help="with --bands: the light-path gate may pass UP TO THE NO-DATA ENCODING: "
                         "every pixel-date where the subset and the nine-band store differ is 0 (L2A "
                         "no data; SCL 0 = NO_DATA) or the fill on every compared band in both "
                         "stores (a chunk one delivery wrote as zeros, the other did not write); the "
                         "store is then accepted as '" + NINE_BAND_NODATA_HOW + "' (recorded, with "
                         "the counts in <year>.vs_nine_band.json) and lifts the wait of the years "
                         "without a nine-band store. Default: bitwise, the fill included")
    ap.add_argument("--extraction-gate-dir", default=None,
                    help=f"with --bands: where experiments.light_to_extraction writes its 2023 "
                         f"extraction gates <site>_gate_2023.json (default {EXTRACTION_GATE_DIR}); a "
                         f"year without a nine-band store of its own at {', '.join(EXTRACTION_GATE_SITES)} "
                         "is bought only once that gate passed or the user accepted it there")
    ap.add_argument("--without-extraction-gate", action="store_true",
                    help="with --bands: buy such a year although the site's 2023 extraction gate has "
                         "not passed nor been accepted (recorded in the store)")
    ap.add_argument("--expect-account", default=None, metavar="E-MAIL",
                    help="before any job is submitted or looked up, describe the openEO account "
                         "(connection.describe_account()), print its e-mail and stop if it is not this "
                         "one (the refresh token in use is chosen by OPENEO_CONFIG_HOME); default: not "
                         "checked")
    ap.add_argument("--login", action="store_true",
                    help="only authenticate with the Copernicus Data Space (interactive, once)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the process graphs, the outputs and a size estimate; no network")
    ap.add_argument("--verify-only", action="store_true",
                    help="compare existing stores with the netCDF cubes again; no network")
    ap.add_argument("--root", default=None,
                    help=f"cubes root (default: ${zc.ENV_ROOT} or <repo>/cubes)")
    ap.add_argument("--allow-grid-offset", action="store_true",
                    help="write a store whose grid is off the existing cube's 10 m lattice or in "
                         "another CRS (default: stop with an error)")
    ap.add_argument("--allow-padding", action="store_true",
                    help="write a production store whose delivery does not cover the whole "
                         "reference grid, padded with the fill (default: stop with an error)")
    ap.add_argument("--allow-snapped-frame", action="store_true",
                    help="download a site whose reference netCDF is not on this machine, on its "
                         f"lon/lat box snapped outward to {FRAME_SNAP_M} m (default: refused "
                         "before any job, since the store would not be on the netCDF frame)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep <year>.part (the backend's files) after a store is complete")
    ap.add_argument("--nodata", type=int, default=None,
                    help="no-data value of the main bands whose delivered data declare none (no "
                         "_FillValue / nodata attribute); needed only when a run stops with "
                         "AMBIGUOUS NO-DATA (see the FORMAT REPORT of --test). Not used for the "
                         "angles (their Zarr fill_value or NaN is the no-data)")
    ap.add_argument("--new-job", action="store_true",
                    help="pay for a fresh job: move <year>.part aside and do not reattach a job "
                         "of the same title (use once; drop it when resuming). Complete stores "
                         "are still skipped")
    ap.add_argument("--retries", type=int, default=1,
                    help="new jobs tried after a failed one, per year and run, each with twice "
                         f"the executor memory up to {MAX_RETRY_MEMORY_GB}G (default 1)")
    ap.add_argument("--max-wait-hours", type=float, default=24.0,
                    help="give up waiting for one job after this long (it keeps running)")
    ap.add_argument("--job-options", default=None,
                    help=f"JSON job options (default {json.dumps(JOB_OPTIONS)})")
    ap.add_argument("--backend-format", default=BACKEND_FORMAT, choices=("Zarr", "netCDF"),
                    help="what the backend writes (netCDF: fallback; the store is the same)")
    ap.add_argument("--format-options", default=None,
                    help="JSON save_result options for the format (the --test report lists "
                         "the ones the backend declares for Zarr)")
    ap.add_argument("--renormalise", action="store_true",
                    help="rebuild the main stores of the sites / years from their kept raw "
                         "delivery (<year>.part, see --keep-raw) with the registration, without "
                         "any network call or job: the store there is moved aside to "
                         "<year>.zarr.superseded-<UTC> (never deleted) once the new one is written "
                         "whole, the new one verified; refused when the raw is missing or the disk "
                         "cannot hold a second copy. A store of the same delivery pending its "
                         "verification is verified again, not rewritten. With --dry-run: the "
                         "measured offsets, the 20 m checks and what would be moved, nothing done")
    ap.add_argument("--offset-px", default=None, metavar="ROWS,COLS",
                    help="expert: the row and column of the frame's first pixel in the delivered "
                         "(padded) arrays, for ONE site. Recorded in the store; with a reference "
                         "netCDF covering the year the registration must agree with it (and the "
                         "verification after writing still decides); without one it is checked "
                         "against the median composites of a store of the site registered against "
                         "its reference (refused when they put the frame elsewhere), else taken "
                         "unverified")
    ap.add_argument("--accept-unverified-offset", action="store_true",
                    help="with --offset-px: take it although nothing can check it and the "
                         "delivered arrays are not the shape the site's stores were delivered in "
                         "(recorded; default: refused, the backend's layout may have changed)")
    ap.add_argument("--accept-differences", action="store_true",
                    help="complete a main store although its verification against the reference "
                         "netCDF refuses it (VERIFY_MIN_EQUAL): differing VALUES (another mosaic, a "
                         "reprocessed product), never a store whose geometry is not the "
                         "reference's; recorded in the store and in <year>.verify.json. A store "
                         "written from the same delivery and refused is verified again, not "
                         "rewritten")
    a = ap.parse_args(argv)
    a.sites = [s.lower() for s in a.sites] or list(SITES)
    unknown = sorted(set(a.sites) - set(SITES))
    if unknown:
        ap.error(f"unknown site(s) {unknown}; choose from {list(SITES)}")
    try:
        a.years = parse_years(a.years) if a.years else list(DEFAULT_YEARS)
    except ValueError as exc:
        ap.error(str(exc))
    if a.bands is not None:
        try:
            a.bands = parse_bands(a.bands)
        except ValueError as exc:
            ap.error(f"--bands: {exc}")
        if a.bands == tuple(BANDS):          # the nine bands: the default product
            a.bands = None
    if (a.store_root is not None or a.without_light_gate) and a.bands is None:
        ap.error("--store-root and --without-light-gate concern a band subset's stores: they go "
                 "with --bands (a subset of the nine bands)")
    if a.l2a_nodata_equal and a.bands is None:
        ap.error("--l2a-nodata-equal concerns a band subset's light-path gate: it goes with --bands")
    if (a.extraction_gate_dir is not None or a.without_extraction_gate) and a.bands is None:
        ap.error("--extraction-gate-dir and --without-extraction-gate concern a band subset's "
                 "purchases: they go with --bands")
    if a.expect_account is not None and "@" not in str(a.expect_account):
        ap.error(f"--expect-account {a.expect_account!r}: an e-mail address expected")
    if a.bands is not None:
        if a.test or a.odd_window or a.lonlat_extent:
            ap.error("--bands (a band subset of the production stores) goes without --test, "
                     "--odd-window and --lonlat-extent")
        try:
            given = parse_products(a.products) if a.products is not None else [MAIN.name]
        except ValueError:
            given = [MAIN.name]                  # reported below as for the nine bands
        if given != [MAIN.name]:
            ap.error("--bands is a subset of the main product: --products main (or nothing); "
                     "the angles do not depend on the bands")
    try:
        a.products = parse_products(a.products if a.products is not None else
                                    MAIN.name if a.bands is not None else ",".join(PRODUCTS))
    except ValueError as exc:
        ap.error(f"--products: {exc}")
    for opt in ("job_options", "format_options"):
        v = getattr(a, opt)
        if v:
            try:
                json.loads(v)
            except ValueError as exc:
                ap.error(f"--{opt.replace('_', '-')} is not JSON: {exc}")
    if sum([a.test, a.verify_only, a.login, a.renormalise]) > 1:
        ap.error("--test, --verify-only, --login and --renormalise exclude each other")
    if a.odd_window and (not a.test or a.lonlat_extent):
        ap.error("--odd-window goes with --test (the UTM extent), not with --lonlat-extent")
    if a.renormalise and (a.new_job or a.lonlat_extent):
        ap.error("--renormalise uses the kept raw delivery: no --new-job, no --lonlat-extent")
    if a.offset_px is not None:
        m = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", str(a.offset_px))
        if not m:
            ap.error(f"--offset-px {a.offset_px!r}: two non-negative integers ROWS,COLS expected")
        a.offset_px = (int(m.group(1)), int(m.group(2)))
        if a.test or a.verify_only or len(a.sites) != 1:
            ap.error("--offset-px is the offset of ONE site's production deliveries: give exactly "
                     "one site, without --test or --verify-only")
    if a.accept_unverified_offset and a.offset_px is None:
        ap.error("--accept-unverified-offset goes with --offset-px")
    return a


def main(argv=None):
    a = parse_args(argv)
    os.chdir(ROOT)
    if a.login:
        import pyintertidal as pit

        conn = pit.scenes.connect(interactive=True)
        print("authenticated; the refresh token is cached for unattended downloads", flush=True)
        if getattr(a, "expect_account", None):
            try:
                check_account(conn, a.expect_account)
            except AccountMismatch as exc:
                print(f"REFUSED: {exc}", flush=True)
                return 2
        return 0
    root = zc.cubes_root(a.root)
    ctx = Context(a)
    # --bands: a band subset; its stores under --store-root, the nine-band root only read
    bands = getattr(a, "bands", None)
    store_root = zc.cubes_root(getattr(a, "store_root", None) or SUBSET_ROOT) if bands else None
    sub_kw = {"bands": bands, "main_root": root} if bands else {}
    if bands:
        print(f"band subset {' '.join(bands)}: stores under {store_root}; the nine-band stores "
              f"under {root} are only read", flush=True)
        why = subset_root_problem(store_root, bands, root)
        if why:
            print(f"REFUSED, nothing started (no job, no download): {why}", flush=True)
            return 2
    else:                           # the nine bands never go into a root of band subsets
        why = nine_band_root_problem(root)
        if why:
            print(f"REFUSED, nothing started (no job, no download): {why}", flush=True)
            return 2
    # --renormalise: the main stores only (the angles carry no B08 to register), so the angles
    # requests (and their reference grids) are never built
    products = [MAIN.name] if a.renormalise else a.products
    reqs = (test_requests(root, products, lonlat=a.lonlat_extent, odd=a.odd_window) if a.test
            else year_requests(a.sites, a.years, store_root or root, products,
                               lonlat=a.lonlat_extent, extents=not a.verify_only, **sub_kw))
    if a.renormalise:
        return renormalise_run(reqs, ctx, dry_run=a.dry_run)
    if a.dry_run:
        return dry_run(reqs, ctx)
    if a.verify_only:
        return verify_only(reqs, ctx)
    refused = [m for m in (refuse_snapped(r) for r in reqs if not r.complete()) if m]
    if refused and not ctx.allow_snapped:
        print("REFUSED, nothing started (no job, no download):", flush=True)
        for m in refused:
            print(f"  {m}", flush=True)
        return 2
    if ctx.expect_account:          # the account first: before any job is submitted or looked up
        try:
            ctx.connection()
        except Exception as exc:  # noqa: BLE001 - AccountMismatch, or no connection to check it on
            print(f"REFUSED, nothing started (no job, no download): {exc}", flush=True)
            return 2
    ensure_gitignore(store_root or root)
    results, failed = {}, {}
    by_site = {}
    for r in reqs:
        by_site.setdefault((r.site, str(r.root)), []).append(r)
    try:
        for (site, _), site_reqs in by_site.items():
            log0 = make_log(site_reqs[0])
            try:
                lock = SiteLock(zc.site_dir(site, site_reqs[0].root))
                lock.__enter__()
            except LockedError as exc:
                log0(str(exc))
                for r in site_reqs:
                    failed[r.tag] = str(exc)
                continue
            try:
                for req in site_reqs:
                    log = make_log(req)
                    try:
                        results[req.tag] = run_request(ctx, req, log)
                    except Exception as exc:          # one year never stops the others
                        failed[req.tag] = f"{type(exc).__name__}: {exc}"
                        log(f"FAILED: {type(exc).__name__}: {exc}")
                        for ln in traceback.format_exc().splitlines()[-12:]:
                            log(f"  {ln}")
                        log(f"run the same command again to resume; finished stores are skipped, "
                            f"the raw download in {req.part} is reused")
            finally:
                lock.__exit__(None, None, None)
    except KeyboardInterrupt:
        print("\ninterrupted. A job that was running keeps running on the backend; run the same "
              "command again to reattach to it (its id is in <year>.part/job.json, or "
              "<year>.angles.part/job.json for the angles).", flush=True)
        return 130
    print("\nSUMMARY", flush=True)
    if ctx.expect_account:
        print(f"  openEO account: {(ctx.account or {}).get('email')} (--expect-account)", flush=True)
    for tag, r in results.items():
        print(f"  {tag}: {r}", flush=True)
    for tag, e in failed.items():
        print(f"  {tag}: FAILED {e}", flush=True)
    missing = {t: r["missing_overpass"] for t, r in results.items() if r.get("missing_overpass")}
    if missing:
        print(f"  overpass times missing for {missing}: those scenes would be dropped", flush=True)
    odd = {t: r.get("verdict") for t, r in results.items()
           if " angles" in t and not str(r.get("verdict", "")).startswith("ok")}
    if odd:
        print(f"  angles to look at (see <year>.angles.verify.json): {odd}", flush=True)
    return 1 if failed or missing else 0


if __name__ == "__main__":
    sys.exit(main())
