# Paper draft — MAREA

Working file for the manuscript. Sections are added in order as they are
agreed; every factual claim must trace to a notebook, a script or a
`results/*.json` verdict, and every literature claim to
`literature_quotes.md` (verbatim passages with page numbers) and
`marea.bib`. Citation keys are the BibTeX keys.

Status: abstract in draft (2026-09-21).

---

## Title (candidates)

1. **Timing the tide from space: intertidal topography from Sentinel-2 with the estuarine tide measured from the imagery itself**
2. MAREA: measuring the interior tide of estuaries from the Sentinel-2 archive to map intertidal topography
3. The pixel as a tide gauge: interior tidal lag and intertidal elevation from a decade of Sentinel-2

## Abstract (v2, 2026-09-21)

Tidal flats are among the most extensive and most threatened coastal
ecosystems, and satellite archives now allow their topography to be mapped
by reading, for every pixel, the water level at which it floods. Every
published implementation assigns each scene a single water level, taken
from an ocean tide model or a tide gauge at one point. Inside estuaries
that level is wrong: the tide arrives later and deformed the farther it
travels up the channel, so the error grows inland and is systematic. We
present MAREA, a method that measures this interior lag from the imagery
itself. Each intertidal pixel is treated as a binary tide gauge, and for
each band of along-channel distance the lag that best explains its
ten-year wet/dry record is found by maximising a profiled Bernoulli
likelihood anchored at the mouth. A binary archive can measure the phase
of the interior tide but not its amplitude or datum, and a lag is applied
only where it exceeds what a null world without interior tide would
produce. Tested on four European estuaries against held-out tide gauges,
MAREA recovered lags of 71 and 128 min on the Ems-Dollard and the Wadden
Sea (gauges: 65 and 95), applied none where none exists, and reduced the
water-level error inside those estuaries by more than 40 %. Against the
official Dutch bathymetry the elevation error fell by 25–33 %, and the
relief at the head of the Ems-Dollard, absent from the uncorrected map, was
recovered. Against an RTK-GNSS survey in the Ría de Villaviciosa (Spain)
the estimator reached 0.16 m, half the error of the published logistic
alternative. The method needs no instrument inside the estuary; code,
notebooks and negative results are released.

*(Abstracts carry no citations by convention; the citation density the
paper needs starts in the Introduction. v1, 360 words, is in the git
history of this file.)*

## Keywords

intertidal topography; tidal flats; Sentinel-2; tidal propagation;
estuaries; waterline method; tide models; profile likelihood; digital
elevation model; validation

## Highlights (journal style, ≤ 85 characters each)

- The tide inside an estuary is measured from Sentinel-2 imagery, with no instrument.
- Each intertidal pixel acts as a binary tide gauge; the lag is a likelihood maximum.
- A binary archive can measure tidal phase but provably not gain or datum.
- Lags of 71 and 128 min recovered on the Ems and Wadden; gauges say 65 and 95.
- Elevation error against official bathymetry falls by 25–33 % where the lag is large.

---

## Introduction

*(next)*
