# Glossary

*The terms a reader meets on these pages, in plain words.* Where a term has
a formula, the link leads to it in [`METHOD.md`](METHOD.md) or
[`VALIDATION_DESIGN.md`](VALIDATION_DESIGN.md).

## Satellite images and data

**Sentinel-2.** A pair (since 2025, a trio: 2A, 2B and 2C) of European
Copernicus satellites that photograph the land and the coast every few days
in 13 spectral bands, at 10 to 60 m. MAREA uses its 10 m green (B03) and
near-infrared (B08) bands.

**L2A.** The Sentinel-2 product corrected for the atmosphere ("bottom of
atmosphere" reflectance), delivered with a scene classification (SCL).

**SCL (scene classification).** A map that comes with every Sentinel-2 L2A
image and labels each pixel as cloud, cloud shadow, water, vegetation, bare
soil, snow and so on. MAREA uses it as its cloud mask: a pixel counts only
if its class is vegetation, bare soil, water or unclassified (classes 4-7).
The full table is in [`METHOD.md`](METHOD.md#4-looks-ndwi-clear-sky-and-wet-or-dry).

**Copernicus Data Space Ecosystem (CDSE).** The European platform that
hosts the Sentinel archive and lets users process it in the cloud. A free
account is enough to download data cubes.

**openEO.** An open standard and interface for asking a cloud platform to
process Earth-observation data ("load these bands over this area for these
dates, and save the result"). MAREA uses it to download its data cubes from
the CDSE.

**Data cube.** The satellite archive of one area as a single array: bands ×
time × y × x. MAREA's cubes hold B03, B08 and SCL for every Sentinel-2 scene.
The validation uses Zarr cubes (one file store per site and year) with nine
bands.

**STAC catalog.** The searchable index of the satellite products. MAREA
reads from it the acquisition time of every scene.

**Scene.** One Sentinel-2 image of the area, on one date.

**Look.** One cloud-free view of one pixel in one scene. A pixel seen in 400
scenes, of which 150 were clear over it, has 150 looks.

**NDWI (Normalized Difference Water Index).** A number computed from the
green and near-infrared light of each pixel, (B03 - B08) / (B03 + B08).
Water absorbs near-infrared light, so NDWI is high over water and low over
dry ground. MAREA calls a look *wet* when its NDWI is above 0.

**Wet / dry.** The two states of a look: wet if its NDWI is above 0, dry
otherwise.

**Water frequency.** The share of a pixel's looks that were wet. Always-wet
pixels are sea or channel; never-wet pixels are land; the intertidal flats
lie in between. MAREA maps the pixels wet in 10 % to 90 % of their looks.

## The tide

**Tide.** The rise and fall of the sea, driven by the Moon and the Sun.
Along these coasts it rises and falls about twice a day.

**Tidal constituent (M2, S2, K1, ...).** The tide is a sum of regular waves,
each with a fixed period set by astronomy. M2, the main lunar wave, has a
period of 12.42 hours; S2, the main solar wave, exactly 12 hours.

**Spring and neap tides.** The largest and smallest tides of the
fortnightly cycle, when the solar and lunar waves add up or partly cancel.

**Sun-synchronous orbit; aliasing.** Sentinel-2 passes each place at nearly
the same time of day. It therefore sees the tide at a slowly drifting
phase: M2 appears as a 14.8-day wave, and S2 is always seen at the same
phase. Some tide levels, such as the extreme low tides of one site, may never
be photographed.

**Tide model; EOT20.** A global model that predicts the tide anywhere at
any time from its constituents. EOT20 (DGFI-TUM; Hart-Davis et al. 2021) is
built from satellite altimetry. MAREA reads it through the Python packages
eo-tides and pyTMD.

**Tide gauge.** An instrument in a harbor that records the water level every
few minutes. The IOC Sea Level Station Monitoring Facility shares the
records of gauges worldwide.

**Mouth.** Where the estuary meets the open sea. MAREA finds it as the
permanent sea that touches the edge of the image.

**Boundary tide, $h_B(t)$.** The water level at the mouth, at the time of
each image: EOT20, averaged with the nearest tide gauges where they exist.
Every elevation MAREA makes is relative to it.

**Tidal delay (lag), $\tau$.** How many minutes later the tide reaches a
point inside the estuary than the mouth. It grows inland, because the tide
travels up the channel as a wave. MAREA measures it from the images.

**Gain (amplification).** How much larger or smaller the tide is inside the
estuary than at the mouth. Wet/dry images cannot measure it
([`METHOD.md`](METHOD.md#12-what-is-provably-not-estimable)).

**Along-channel distance, $s$.** The distance from the mouth measured
through the water, the path the tide travels, not as the crow flies.

**Section (band).** A 3-km stretch of along-channel distance. MAREA fits
one delay per section; each pixel then reads its own delay from a smooth
curve through the section delays.

**Vertical datum.** The zero from which heights are measured. MAREA's
elevations are relative to the boundary tide (about mean sea level), not to
a national datum such as the Alicante datum (Spain) or NAP (Normaal
Amsterdams Peil, the Netherlands). Comparisons with surveys remove the
median offset first and report it as the bias.

## The method

**Intertidal zone; tidal flat.** The ground that is flooded at high tide and
exposed at low tide: mud and sand flats, the habitat of *Zostera noltei*
seagrass meadows, and the feeding grounds of wading birds.

**Estuary; ría.** An estuary is a river mouth where fresh and sea water mix
and the tide reaches inland. A ría is a drowned river valley, typical of the
Galician and Cantabrian coasts of Spain.

**Scene screening.** Choosing which scenes to use. MAREA keeps a scene if
at most 5 % of the whole image is cloudy or, failing that, if at most 10 %
of the transition zone is cloudy.

**Reference map; transition zone.** A map, built from the clearest scenes,
of where the ground is always water, always land, or sometimes either: the
transition zone, where the tidal flats are.

**Flooding curve.** How a pixel's NDWI changes as the water rises: low while
the pixel is dry, high once it is under water, with an S-shaped step in
between, NDWI = a + b Φ((h - z)/σ).

**Elevation, $z$.** The height of a pixel's ground: the water level at the
midpoint of its flooding curve, where half of the pixel is under water.

**Sigma, $\sigma$.** The width of the flooding curve: how gradually the
pixel goes from dry to wet. MAREA's maps show it as the uncertainty of the
elevation; it also reflects the roughness of the ground inside the pixel.

**Quality rule (bracket).** An elevation is kept only if the pixel was seen
at least three times with the water below it and three times above it, so
that the elevation is interpolated, not extrapolated.

**Profiled likelihood.** A way to test one parameter (here, the delay) when
many others are unknown (each pixel's elevation and width): for every
candidate delay, every pixel first takes its own best elevation and width,
and only then are the candidates compared.

**Bernoulli likelihood.** The probability of a series of yes/no outcomes
(here, wet/dry looks) under a model; the delay that makes the observed looks
most probable is the estimate.

**Bootstrap.** Repeating an estimate on resampled data (here, 30 random
resamples of the scenes) to see how much it varies; the spread gives an
interval.

**Emersion time.** The share of time a point is out of the water. A pixel
with an emersion time of 0.6 is exposed to the air 60 % of the time (about
14 hours a day). It organizes intertidal life: seagrass, marsh plants and
shellfish each live in their own band of emersion time.

**Hydroperiod.** The opposite of emersion time: the share of time a point is
under water.

**Hypsometric curve; hypsometric integral.** The hypsometric curve gives the
area of ground below each level. The hypsometric integral (Strahler's)
summarizes its shape in one number between 0 and 1: high for a flat that
sits mostly high, low for one that sits mostly low.

## Validation

**Independent check (truth).** A measurement of the same quantity that the
method never used: a survey of the ground, or a tide gauge inside the
estuary.

**Vaklodingen.** Rijkswaterstaat's Vaklodingen depth surveys of the Dutch
coast: grids of the sea bed and the tidal flats at 20 m, compiled per map
sheet and year from echo-sounding and, where it exists, airborne LiDAR.
Rijkswaterstaat is the Dutch agency for water and infrastructure.

**PNOA LiDAR (MDT02).** The airborne laser surveys of Spain's national
aerial orthophoto plan (Plan Nacional de Ortofotografía Aérea), published by
the Instituto Geográfico Nacional as 2 m terrain models (MDT02). The laser
does not reach the ground under water, so water surfaces are masked.

**LiDAR.** Light detection and ranging: a laser on an aircraft that
measures the height of the ground.

**RTK-GNSS.** Real-time kinematic satellite positioning: a field receiver
that measures the position and height of a point to about a centimeter.

**Judge (in-estuary gauge).** A tide gauge inside an estuary whose record
gives the real delay of the tide at that point; it judges the delay that
each method estimates.

**NIDEM; ITEM.** The Intertidal Extents Model and the National Intertidal
Digital Elevation Model of Australia (Sagar et al. 2017; Bishop-Taylor et
al. 2019), which map elevation from waterlines at different tide levels.
The main competitor of MAREA in the validation.

**DEA Intertidal.** Digital Earth Australia's intertidal elevation product
and code (Bishop-Taylor et al.), which reads each pixel's elevation as the
lowest tide at which it looks wet.

**Competitor; method comparison.** Other published methods for mapping
intertidal elevation from satellite images, run on the same inputs as
MAREA. They are listed in
[`VALIDATION_DESIGN.md`](VALIDATION_DESIGN.md#8-the-methods-compared).

**Fixed configuration.** One set of inputs and settings (scenes, cloud mask,
NDWI threshold, pixels, tide) given identically to every method, so that
only the method differs.

**Window of n images.** The n most recent screened images before a survey
date, used as the input of every method in the fixed-configuration
comparison (n = 40 to 125).

**RMSE; bias; centered RMSE.** The RMSE (root mean square error) is the
typical size of the error. The bias is the median error: a constant offset,
often a difference of vertical datum. The centered RMSE removes that offset
first and measures the error of the shape of the ground.

**Slope.** The slope of a straight line fitted to the mapped heights
against the true heights: 1 means undistorted relief; below 1, flattened
relief.

**r (Pearson correlation).** How closely the mapped and true heights follow
the same pattern, from -1 to 1.

**Coverage.** The share of the scored ground for which a method gives an
elevation.

**NOT COMPUTABLE.** Reported when a method's own rules stop it at a site (for
example, too few images). The rules are never relaxed to obtain a number.

**Pre-registration; lock.** Writing down the settings, tests and decision
rules before seeing any result (`LOCK.json`), so that the results cannot
steer the choices.

**Home advantage.** MAREA was developed on the same sites, gauges and
surveys it is validated on. The validation declares this and lists every
check that influenced a setting.

**Emersion-time error.** A proposed validation metric that turns each
elevation error into hours per day of emersion time, the unit that matters
to ecologists.
