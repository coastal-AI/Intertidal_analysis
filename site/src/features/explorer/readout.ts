/**
 * Legend ticks, height labels and readout lines of the explorer (pure functions, no DOM).
 *
 * Heights: the images are coloured on the map's zero (the tide model's mean sea level). Where a site has C, heights
 * are shown above NMMA (Alicante mean sea level) = map value + C, so a legend tick for the NMMA height t sits at
 * (t − C − d0) / (d1 − d0) along a colour bar whose domain is [d0, d1].
 */
import type { ColorBarTick } from '@/components/ui'
import { fmtMinutes, fmtNumber, fmtPercent, NBSP } from '@/lib/format'
import type { ClassScale, LinearScale, MapHeights, MapLayer, TideLevel } from './types'

/** What is added to map values to show them: C, or 0 where the site has no NMMA link. */
export function heightOffset(h: MapHeights): number {
  return h.datum === 'NMMA' ? h.C : 0
}

/** Units of a height: "m NMMA", or "m" above the tide model's mean. */
export function heightUnits(h: MapHeights): string {
  return h.datum === 'NMMA' ? 'm NMMA' : 'm'
}

/** The one line that says how heights are given at the site, with C and where it comes from. */
export function heightNote(h: MapHeights): string {
  if (h.datum === 'NMMA') {
    return `Heights in m above NMMA (Alicante mean sea level) = tide-model level + C, with C = ${fmtNumber(h.C, { decimals: 2 })}${NBSP}m from the ${h.gauge} tide gauge.`
  }
  return 'Heights in m above the tide model’s mean sea level; no NMMA link here.'
}

/** A height as shown: "+0.10 m NMMA". */
export function formatHeight(mapValue: number, h: MapHeights): string {
  return `${fmtNumber(mapValue + heightOffset(h), { decimals: 2, sign: true })}${NBSP}${heightUnits(h)}`
}

/** A value with its units: "0.22 m", "42 %", "7 min"; ``signed`` writes "+" before positive values ("+0.12 m"). */
export function formatValue(value: number, units: string, decimals: number, signed = false): string {
  if (!signed && units === '%') return fmtPercent(value, decimals)
  if (!signed && units === 'min') return fmtMinutes(value, decimals)
  return `${fmtNumber(value, { decimals, sign: signed })}${NBSP}${units}`
}

/** The smallest round step that gives at most ``maxTicks`` ticks over ``span``. */
export function niceStep(span: number, maxTicks = 7): number {
  for (const step of [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 15, 20, 25, 50, 100]) {
    if (Math.floor(span / step + 1e-9) + 1 <= maxTicks) return step
  }
  return span
}

function decimalsOf(step: number): number {
  const s = String(step)
  return s.includes('.') ? s.split('.')[1].length : 0
}

/** A tick label: integers without decimals, the others with the step's decimals ("−1.5", "0", "1"). */
function tickLabel(t: number, step: number): string {
  const rounded = Math.round(t * 1e6) / 1e6
  return fmtNumber(rounded, { decimals: Number.isInteger(rounded) ? 0 : decimalsOf(step) })
}

/**
 * Ticks of a linear colour bar. Height layers get round values of the shown height (value + offset) at the
 * matching position; other layers get the scale's own ticks.
 */
export function linearTicks(scale: LinearScale, offset = 0, roundShown = false): ColorBarTick[] {
  const [d0, d1] = scale.domain
  const span = d1 - d0
  if (!roundShown) {
    const step = scale.ticks.length > 1 ? Math.abs(scale.ticks[1] - scale.ticks[0]) : span
    return scale.ticks.map((t) => ({ at: (t - d0) / span, label: tickLabel(t, step) }))
  }
  const lo = d0 + offset
  const hi = d1 + offset
  const step = niceStep(span)
  const out: ColorBarTick[] = []
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + 1e-9; k++) {
    const t = k * step
    out.push({ at: (t - offset - d0) / span, label: tickLabel(t, step) })
  }
  return out
}

/** How a colour scale runs from its low end to its high end: "light to dark" or "dark to light" (hex colours). */
export function shadeDirection(colours: readonly string[]): string {
  const luma = (hex: string) => {
    const n = Number.parseInt(hex.slice(1, 7), 16)
    return 0.2126 * ((n >> 16) & 255) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255)
  }
  return luma(colours[0]) >= luma(colours[colours.length - 1]) ? 'light to dark' : 'dark to light'
}

/** Labels of a classed bar: the first, middle and last classes, at the centres of their steps. */
export function classTicks(scale: ClassScale, decimals = 2): ColorBarTick[] {
  const n = scale.classes.length
  const picks = n <= 3 ? scale.classes.map((_, i) => i) : [0, Math.floor((n - 1) / 2), n - 1]
  return picks.map((i) => ({ at: (i + 0.5) / n, label: fmtNumber(scale.classes[i].value, { decimals }) }))
}

/** Where a height falls among the tidal levels: "between MLWN and MSL", "below LAT", "above HAT". */
export function datumZone(levels: readonly TideLevel[], mapValue: number): string {
  const n = levels.length
  if (n === 0) return ''
  if (mapValue < levels[0].level) return `below ${levels[0].id}`
  if (mapValue >= levels[n - 1].level) return `above ${levels[n - 1].id}`
  for (let i = 0; i < n - 1; i++) {
    if (mapValue < levels[i + 1].level) return `between ${levels[i].id} and ${levels[i + 1].id}`
  }
  return `above ${levels[n - 1].id}`
}

export const NO_VALUE = 'No MAREA value here'

/** What the readout knows about the pointed place. */
export interface ReadoutValues {
  /** Height on the map's zero, or null. */
  height: number | null
  /** Value of the active raster layer, or null. */
  value: number | null
  /** Properties of the point under the pointer (points layers), or null. */
  point?: Record<string, unknown> | null
}

/** The readout line for a layer: "+0.10 m NMMA", "0.22 m · elevation +0.10 m NMMA", "+0.10 m NMMA · between MSL and MHWN". */
export function readoutLine(layer: MapLayer, heights: MapHeights, v: ReadoutValues): string {
  const elev = v.height === null ? null : formatHeight(v.height, heights)
  if (layer.kind === 'raster') {
    if (layer.heights) return elev ?? NO_VALUE
    if (v.value === null) return NO_VALUE
    const shown = formatValue(v.value, layer.units, layer.decimals, layer.signed === true)
    return elev ? `${shown} · elevation ${elev}` : shown
  }
  if (layer.kind === 'contours') {
    if (v.height === null || elev === null) return NO_VALUE
    return `${elev} · ${datumZone(layer.levels, v.height)}`
  }
  const p = v.point
  if (p) {
    const parts = layer.fields
      .filter((f) => typeof p[f.property] === 'number')
      .map((f) => `${f.label} ${fmtNumber(p[f.property] as number, { decimals: layer.decimals, sign: true })}${NBSP}${layer.units}`)
    if (parts.length) return parts.join(' · ')
  }
  return elev ?? NO_VALUE
}
