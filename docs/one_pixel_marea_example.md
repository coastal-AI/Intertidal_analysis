# One pixel through MAREA: clocks, likelihood, decision, inversion

A worked example on a single Sentinel-2 pixel of the Ría de Villaviciosa, taken
from `pixel_walkthrough_villaviciosa.ipynb` (every number below is printed by
that notebook; every figure is one of its outputs). The point is to make the
tide-clock question concrete: the pixel has one true elevation, the ocean
model gives the water level at the mouth, and the question is *which clock*
that level should be read on before the pixel's wet/dry record is inverted.

## The pixel and its record

* Pixel at 7.95 km along the water from the mouth, in band 5 of 6 (band
  centre 7.84 km, 5 004 pixels).
* 1 379 scenes over 2016-2025; 305 pass the cloud screen over the transition
  zone; at this pixel 199 visits survive the per-pixel cloud mask.
* Water frequency 0.613 → intertidal. Threshold NDWI > 0 splits its 288 clear
  visits into 167 wet and 121 dry.
* Each visit carries the EOT20 level at the overpass time (range −2.27 to
  +1.22 m).

**What the cube holds for the pixel.** Three numbers per visit, once per
scene, for ten years: the green (B03) and near-infrared (B08) reflectance,
and the scene class (SCL) that Sen2Cor gave the pixel. Figure 1 shows them
three ways. The top row is what one visit *is*: the NDWI image of the 1.2 km
around the pixel (red square) on four dates chosen by rule, a firmly dry
look, a firmly wet look, a cloud and a cloud shadow. On the dry date the
channel is a thread and the flats are brown; on the wet date the estuary is
full and the pixel is under water; under cloud the whole window is one flat
value, no information at all; under cloud shadow the flats turn pale cyan,
because a shadow reads like water in these two bands (the pixel's NDWI is
+0.10 there), which is why the SCL screen drops shadows as well as clouds.
The middle row is the pixel's near-infrared at every one of its 1 379
visits, on a log axis: the clear-class visits (dark) fall in two crowds,
around 100 when the pixel is under water and around 1 000 when it is bare
ground, and the cloud and shadow classes (grey) sit above both, mostly
between 3 000 and 12 000. The 156 visits Sen2Cor left unclassified (orange)
have ground-like values but no clear class, and the pipeline does not use
them. The bottom left zooms into the year with most clear visits and shows
both bands at each visit, joined by a segment: blue when green is above
near-infrared (water), brown when near-infrared is above green (ground).
That ordering is the whole detector: NDWI is the normalised difference of
the two, and its sign says which one wins. The bottom right counts the
classes: of 1 379 visits, 935 are cloud or shadow, 156 unclassified, and 288
carry a clear class (190 bare ground, 97 water, 1 vegetation). Those 288 are
the pixel's usable looks before the scene-level screen of section 3.

![The pixel's visits](figures/onepixel/cell05_0.png)
*Figure 1. Top, NDWI of the 1.2 km window around the pixel (red square) on four visits chosen by rule: a firmly dry look (lower-quartile NDWI of the bare-ground visits), a firmly wet look (upper quartile of the water visits), the median cloud and the median cloud shadow; each title gives the pixel's own three numbers on that date. Middle, the pixel's near-infrared reflectance at all 1 379 visits (log axis), coloured by SCL group: clear (dark), unclassified (orange), cloud or shadow (grey); letters mark the four visits above. Bottom left, the year with most clear visits (2025) with both bands at every visit, the segment blue when green exceeds near-infrared (water) and brown otherwise (ground), grey for cloudy classes. Bottom right, the number of visits in each SCL class.*

**The threshold, and where the pixel sits against it.** The histogram is
the site calibration: NDWI of the clear-sky pixels of the twelve clearest
scenes, exposed ground in brown (its mode near −0.75) and water in blue
(the long tail above +0.3). Otsu's split of this sample lands at +0.25,
pinned at the top of its allowed range — a warning that the sample is not
clearly bimodal, not a result — and the study adopts 0 instead, because
turbid estuarine water reads lower than clear water. The red ticks are this
pixel's own 288 clear visits laid on the same axis: a cluster just below 0
(dry) and a spread from 0 to 1.8 (wet), with almost nothing in between.
The cut at 0 falls in the gap; the cut at +0.25 would have called some
wet visits dry.

![The threshold on the pixel](figures/onepixel/cell07_0.png)
*Figure 2. NDWI histogram of the clear-sky pixels of the site's 12 clearest scenes (brown ground, blue water); black dashed, Otsu's split (+0.25, pinned at the cap); red line, the adopted threshold (0); red ticks, this pixel's 288 clear visits.*

**The cloud screen, as it lands on the pixel.** Two filters, at two scales.
First the scene: a date survives if at most 10 % of the *transition zone*
is cloudy — the fraction is measured over the flat, not over the whole
frame, which is what keeps scenes that are cloudy at sea but clear over the
estuary; 305 of 1 379 dates pass. Then the pixel: on a kept scene, the SCL
class at this pixel must be a clear one (4-7, 12), otherwise that visit is
dropped for this pixel alone. In the figure, grey points are visits on
rejected scenes (1 074 of them), orange are kept scenes where this pixel
itself was under cloud or shadow (106), and the blue and brown points are
the 199 observations that survive, already labelled wet or dry by the
threshold. Note how many orange points sit at NDWI ≈ 0.3-0.6 among the wet
cluster: thin cloud over water still reads "wet" on the index, and only the
SCL class catches it.

![The cloud screen on the pixel](figures/onepixel/cell10_0.png)
*Figure 3. The pixel's NDWI against time, coloured by what the cloud screen did with each visit: grey, scene rejected; orange, scene kept but pixel cloudy per SCL; blue and brown, the 199 surviving observations, wet and dry.*

**Where the pixel lives.** Water frequency in a 1.2 km window around the
pixel (red square), 7.9 km up the ría. Dark brown is land that is never
wet, dark green the channel that is always wet, and the pale band between
them is the flat: pixels wet on some visits and dry on others, ordered by
height — the closer to the channel, the wetter. The pixel sits on the
inner edge of that band, on the west bank of the meander, with WF = 0.61:
wet on six visits out of ten. That is why it is a good witness — it spends
enough time on both sides of the waterline for its wet/dry record to
locate the water level, and it is far enough upstream for the clock to
matter.

![Where the pixel sits: water frequency around it](figures/onepixel/cell13_0.png)
*Figure 4. Water frequency in a 1.2 km window around the pixel (red square): brown never wet, green always wet, the pale band is the intertidal flat.*

**Two cuts, three classes: the intertidal decision.** The histogram is the
water frequency of every pixel of the transition class. It has three
populations: a wall at 0 (ground that is almost never wet: supratidal
marsh, noise at the halo's edge), a wall at 1 (the channel, almost always
wet) and a low, wide hump in between — the intertidal flat, whose water
frequency is an ordering by height. A three-class Otsu split finds the two
valleys that separate them, at 0.21 and 0.66 (dashed). The study does not
adopt them: they would cut off the lowest flats (wet on 70-90 % of visits)
and the highest (wet on 5-20 %), which are real intertidal ground that the
fit can still resolve. So the window is opened to 0.05-0.95 (red) and the
per-pixel quality control downstream is left to reject what does not fit.
This pixel, at 0.61, is inside both windows: intertidal by any reading.

![Two cuts, three classes](figures/onepixel/cell12_0.png)
*Figure 5. Water-frequency histogram of the site's transition class; dashed, the two valleys of the three-class Otsu (0.21 and 0.66); red, the adopted window (0.05-0.95); the thick line, this pixel (0.61).*

**The same 199 observations, seen two ways.** Left, NDWI against time:
wet visits (blue) and dry visits (brown) alternate with no pattern a
calendar could explain. Right, the same points against the ocean tide at
the moment of each visit: the record becomes a staircase — dry below about
−0.5 m, wet above 0 m, mixed in between. The step is the pixel's elevation
and the width of the mixed zone is its sub-pixel relief plus the noise of
the level assigned to each visit. Everything that follows is about reading
that step correctly: MAREA asks whether a different clock on the tide axis
makes the step sharper, and the inversion fits a sigmoid to it.

![The tide at every visit](figures/onepixel/cell15_0.png)
*Figure 6. The pixel's 199 observations: left, NDWI against date; right, NDWI against the model tide level at the instant of each visit; blue wet, brown dry.*

## 1 · The same pixel under five clocks

The boundary level is read τ minutes earlier for each candidate clock
(τ = −30, 0, +24, +60, +90 min). Nothing about the pixel changes; only the
x-coordinate of every visit moves along the tide curve. On each clock the
NDWI sigmoid is refitted (closed form: elevation z, sub-pixel spread σ,
offset a, gain b).

![NDWI curves under five clocks](figures/onepixel/cell19_0.png)
*Figure 7. The same 199 observations under five clocks (τ = −30, 0, +24, +60, +90 min): each panel shifts every visit's level and refits the NDWI sigmoid (red curve); the dotted line is the fitted elevation z. Each fit's parameters are in the panel title.*

| τ (min) | z (m) | σ (m) | a | b | rms |
|---|---|---|---|---|---|
| −30 | −0.305 | 0.45 | −0.106 | 0.745 | 0.2744 |
| 0 | −0.217 | 0.32 | −0.103 | 0.771 | 0.2338 |
| **+24** | −0.246 | 0.45 | −0.135 | 0.835 | **0.2228** |
| +60 | −0.217 | 0.45 | −0.118 | 0.810 | 0.2508 |
| +90 | −0.334 | 0.85 | −0.173 | 0.887 | 0.3000 |

Reading: the wrong clock does not just shift z, it *blurs* the transition
(σ grows, the fit residual grows). The clock that makes the wet and dry
visits separate most cleanly is the one under which the pixel behaves like a
threshold sensor.

## 2 · The likelihood, step by step

To choose the clock, MAREA does not look at the NDWI values. It looks only
at whether the pixel came out **wet or dry** on each visit: $w_t = 1$ or
$w_t = 0$. That binary record does not change if the tide level is rescaled
or shifted (the affine theorem), so the only thing it can yield is the
*phase*, which is the clock.

The pixel model is simple. The pixel has a mean elevation $z$ and an
internal relief $\sigma$ (inside 10 m there is not one height but many).
With the water well above $z$ it reads wet; well below, dry; near $z$,
sometimes one and sometimes the other. The probability of reading wet when
the water stands at level $h$ is the cumulative bell $\Phi\big((h -
z)/\sigma\big)$.

With that, for a candidate clock $\tau$ and a candidate pair $(z, \sigma)$,
five operations are done visit by visit:

1. **The level under the clock.** Read the boundary $\tau$ minutes before
   the overpass:
   $$h_t(\tau) = h_{\mathrm{EOT20}}\!\left(t_{\mathrm{overpass}} - \tau\right).$$
2. **How much water stood over the pixel**, in units of its relief:
   $$u_t = \frac{h_t(\tau) - z}{\sigma}.$$
   Positive, water above the mean height; negative, below; $|u_t| \gg 1$,
   the model is sure of what it should see.
3. **The probability of wet:**
   $$P_t = \Phi(u_t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{u_t} e^{-x^2/2}\,\mathrm{d}x,$$
   clipped to $[10^{-6},\ 1 - 10^{-6}]$ so that one impossible visit cannot
   send the sum to $-\infty$ (the product clips at $10^{-4}$).
4. **The cost of the visit:**
   $$\ell_t = w_t \ln P_t + (1 - w_t)\ln(1 - P_t)
   = \begin{cases}\ln P_t & \text{if it came out wet},\\ \ln(1 - P_t) & \text{if it came out dry.}\end{cases}$$
   Never positive: close to 0 when the model was confidently right, very
   negative when it was confidently wrong.
5. **The sum over the 199 visits:**
   $$\ell(\tau; z, \sigma) = \sum_{t=1}^{199} \ell_t .$$

On the ocean clock ($\tau = 0$) the best pair is $z = -0.66$ m,
$\sigma = 0.65$ m, and the sum is $\ell(0) = -57.23$: 0.29 per visit on
average. The first twelve terms, with operations 2, 3 and 4 as columns:

| date | $h_t$ (m) | $w_t$ | $u_t$ | $P_t$ | $\ell_t$ |
|---|---|---|---|---|---|
| 2016-01-01 | −0.119 | 1 | 0.83 | 0.7959 | $\ln 0.7959 = -0.2283$ |
| 2016-01-11 | −1.554 | 0 | −1.38 | 0.0837 | $\ln(1-0.0837) = -0.0874$ |
| 2016-03-18 | 0.726 | 1 | 2.13 | 0.9833 | −0.0168 |
| 2016-07-09 | −1.006 | 0 | −0.54 | 0.2955 | $\ln(1-0.2955) = -0.3503$ |
| 2016-07-16 | 0.562 | 1 | 1.88 | 0.9696 | −0.0309 |
| 2016-08-15 | 0.519 | 1 | 1.81 | 0.9648 | −0.0359 |
| 2016-10-07 | −0.646 | 1 | 0.02 | 0.5068 | $\ln 0.5068 = -0.6797$ |
| 2016-10-14 | 0.484 | 1 | 1.76 | 0.9604 | −0.0404 |
| 2016-11-16 | −1.861 | 0 | −1.85 | 0.0320 | −0.0325 |
| 2016-12-03 | −1.374 | 0 | −1.10 | 0.1350 | −0.1451 |
| 2016-12-13 | −0.331 | 0 | 0.50 | 0.6919 | $\ln(1-0.6919) = -1.1772$ |
| 2017-01-05 | −0.018 | 1 | 0.98 | 0.8372 | −0.1777 |

Three lines are enough to read the table. On 2016-03-18 the water stood
2.1 reliefs above the pixel, the model expected wet at 98 % and the pixel
came out wet: cost 0.017, almost free. On 2016-10-07 the water stood right
at the pixel's height ($u_t = 0.02$), the model did not know (51 %) and the
pixel came out wet: cost 0.68, the price of ambiguity. On 2016-12-13 the
water stood half a relief above (69 % wet expected) and the pixel came out
**dry**: cost 1.18, a contradiction. Over the whole record the cheapest
visits cost 0.003 and the dearest 5.79, 3.65 and 2.72, all contradictions:
dry with the water well above, or wet with the water well below. A wrong
clock assigns wrong levels to the visits, and a wrong level turns cheap
visits into dear ones. That is what the clock search measures.

**Profiling.** $z$ and $\sigma$ are not what we are after here, and the
clock must not be blamed for a bad pair. So, for each clock, a grid of pairs
is tried and only the best value is kept:

$$
\ell^{\ast}(\tau) = \max_{z \in Z,\ \sigma \in S}\ \ell(\tau; z, \sigma).
$$

$Z$ is 120 levels between the lowest and highest observed tide (60 over the
tidal range ± 0.3 m in the product); $S$ is the archive's set of relief
atoms, $\{0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.65\}$ m (extended
here to 1.10 m only to see the whole surface). The figure is
$\ell(0; z, \sigma)$ for the ocean clock, with the maximum marked.

![Profiling surface for the ocean clock](figures/onepixel/cell22_0.png)
*Figure 8. Log-likelihood surface ℓ(0; z, σ) of the ocean clock over the grid of candidate elevations (x) and widths (y); the circle marks the maximum, z = −0.66 m, σ = 0.65 m, the profiled pair.*

**The same thing, drawn.** Three panels, all on this pixel's 199 visits.
Left, the pixel's rule: the probability of reading wet against the water
level, $\Phi((h - z)/\sigma)$ with the ocean-clock pair $z = -0.66$ m,
$\sigma = 0.65$ m; every visit sits on the axis at its level, blue if it
read wet, brown if dry, and the three visits of the text are numbered on
the curve. Centre, what a visit costs as a function of $u$: the blue curve
is the cost of a wet reading, the brown one of a dry reading; a wet visit
far above the pixel (1) costs nothing, a visit at the pixel's own height (2)
costs 0.68 whatever it reads, a dry visit half a relief above (3) costs
1.18. Right, the bill: the 199 costs sorted from dearest to cheapest, in
grey on the ocean clock (sum 57.2) and in red on the band clock +18 min
(sum 54.8). The two clocks agree on the cheap visits; they differ on the
dear tail, which is where the contradictions live — 15 visits cost more
than 1 on the ocean clock, 11 on the band clock. That tail is what the
clock search reads.

![The likelihood, drawn](figures/onepixel/likelihood_explained.png)
*Figure 9. A, the pixel's rule P(wet | h) with the 199 visits on it and three numbered visits; B, the cost of a visit as a function of u = (h − z)/σ, blue curve if it read wet and brown if dry; C, the 199 costs sorted from dearest to cheapest on the ocean clock (grey) and on the band clock +18 min (red).*

## 3 · The same sum for every clock, and the optimum

Repeat the profiled sum for 31 candidate clocks. Two curves: the pixel alone
(199 visits) and its whole band (300 sampled pixels, scaled ÷25 to share the
axis).

![Log-likelihood against the candidate clock](figures/onepixel/cell24_0.png)
*Figure 10. Top, profiled log-likelihood relative to τ = 0 for 31 candidate clocks, for the pixel alone (blue) and for its 300-pixel band (red, ÷25); the grey band is the ±10-min threshold and the vertical red line the band optimum (+18). Below, the profiled elevation z and width σ for each clock.*

* Pixel alone: best τ = **+24 min**. The curve is flat near the top: one
  pixel with 199 visits barely resolves 20 minutes.
* Its band: best τ\* = **+18 min**, with a sharp peak — 300 pixels sharing one
  clock make the estimate precise (about ±4 min in the synthetic calibration).
* The lower panels show how z and σ react as the clock moves: z steps on the
  grid; σ is smallest near the optimum and inflates on both sides. σ is the
  first casualty of a wrong clock.

## 4 · The decision and what it implies

* The band clock τ\* = +18 min exceeds the detection threshold of ±10 min, so
  it is **applied** (in Villaviciosa only this band clears it; the other five
  keep the ocean clock).
* Physical scale: when the water crosses z = −0.66 m the tide moves at
  0.41 m/h, i.e. **0.069 m per 10 min** of clock error. If every visit moved
  the same way, τ\* = +18 min would be worth ~12 cm of level.
* It does not, and that is the subtle part: rising passes move −0.13 m and
  falling passes +0.12 m — opposite signs — so the **net** median level change
  per visit is −0.01 m. A clock error does not bias the level; it *spreads*
  it, which is why σ grows and the fit blurs. The gain from the right clock
  is a sharper transition, not a shifted one.

## 5 · The inversion with and without the clock

The elevation estimator (`marea.invert_series`, the same closed-form sigmoid
used by every product) is run twice on the same 199 visits:

![Inversion on the ocean clock vs the band clock](figures/onepixel/cell29_0.png)
*Figure 11. The elevation inversion (`invert_series`) on the same 199 observations with the ocean clock (left, z = −0.204 m) and with the band clock +18 min (right, z = −0.241 m); red curve, the fitted sigmoid; dotted, the elevation.*

| clock | z (m) | σ (m) |
|---|---|---|
| ocean clock (τ = 0) | −0.204 | 0.32 |
| band clock (τ\* = +18 min) | −0.241 | 0.45 |

The clock moved this pixel's elevation by −3.7 cm. For reference, the shipped
HSR product (2023-2025, ocean clock) reads −0.232 m at the pixel and the
shipped MAREA product −0.266 m (band clocks applied: [0, 0, 0, 0, 0, +15]).

## What the pixel teaches

1. The clock is measured from the **binary** record, per band, because that
   record cannot see gain or datum and therefore cannot be fooled by them.
2. One pixel cannot fix a clock (flat likelihood); a band can (sharp peak).
   That is why MAREA bands the flat by distance from the mouth.
3. A wrong clock inflates σ and spreads the transition rather than biasing z.
   In a short deep ría the effect on z is centimetres (−3.7 cm here). In a
   long shallow estuary it is decimetres: on the Ems-Dollard the measured
   clock reaches +104 min and moves the map by 0.19 m (median) and up to
   0.8 m at the head — see `tide_adjustment_comparison.md`.
4. The decision is gated: below ±10 min the ocean clock is kept, and a
   profile with a negative above-threshold lag is vetoed as physically
   impossible. The product says which clock it used, band by band.
