/**
 * The one legend of the explorer: the active layer's colour bar (or tide-level ladder), its units, one sentence and,
 * for heights, how they are given. ``compact`` is the phone's collapsed panel: the bar and any warning only.
 */
import { clsx } from 'clsx'
import { ColorBar } from '@/components/ui'
import { fmtNumber, NBSP } from '@/lib/format'
import { classTicks, heightNote, heightOffset, heightUnits, linearTicks, shadeDirection } from './readout'
import { CONTOUR_COLOUR, contourDash, EQUAL_DELAY_COLOUR, EQUAL_DELAY_DASH } from './lineStyles'
import type { ContourLayer, MapHeights, MapLayer, RasterLayer } from './types'

/** A line as the map draws it: a white casing under the coloured (possibly dashed) line. */
export function LineSwatch({ color, dash, width = 1.5 }: { color: string; dash: number[] | null; width?: number }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 28 8" className="h-2 w-7 shrink-0 overflow-visible">
      <line x1="1" y1="4" x2="27" y2="4" stroke="#FFFFFF" strokeWidth="3.5" strokeLinecap="round" />
      <line
        x1="1"
        y1="4"
        x2="27"
        y2="4"
        stroke={color}
        strokeWidth={width}
        strokeLinecap={dash ? 'butt' : 'round'}
        strokeDasharray={dash ? dash.map((d) => d * width).join(' ') : undefined}
      />
    </svg>
  )
}

function Title({ title, units }: { title: string; units: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-sm font-semibold text-ink">{title}</span>
      <span className="text-xs text-muted">{units}</span>
    </div>
  )
}

function rasterBar(layer: RasterLayer, heights: MapHeights) {
  const units = layer.heights ? heightUnits(heights) : layer.units
  if (layer.scale.type === 'classes') {
    const cls = layer.scale.classes
    return (
      <ColorBar
        stops={cls.map((c) => c.color)}
        stepped
        ticks={classTicks(layer.scale)}
        label={`${layer.title}: ${cls.length} classes from ${fmtNumber(cls[0].value)} to ${fmtNumber(cls[cls.length - 1].value)}${NBSP}${units}, ${shadeDirection(cls.map((c) => c.color))}`}
      />
    )
  }
  const offset = layer.heights ? heightOffset(heights) : 0
  const [d0, d1] = layer.scale.domain
  const dec = layer.heights ? 2 : layer.decimals
  return (
    <ColorBar
      stops={layer.scale.stops}
      ticks={linearTicks(layer.scale, offset, !!layer.heights)}
      label={`${layer.title} from ${fmtNumber(d0 + offset, { decimals: dec })} to ${fmtNumber(d1 + offset, { decimals: dec })}${NBSP}${units}, ${shadeDirection(layer.scale.stops)}`}
    />
  )
}

/** The tidal levels, highest first; levels outside the mapped heights are faded. */
export function Ladder({ layer, heights, compact }: { layer: ContourLayer; heights: MapHeights; compact: boolean }) {
  const shown = (v: { level: number; nmma: number | null }) =>
    fmtNumber(heights.datum === 'NMMA' && v.nmma !== null ? v.nmma : v.level + heightOffset(heights), { decimals: 2, sign: true })
  const levels = [...layer.levels].reverse()
  if (compact) {
    // The phone's collapsed panel: the drawn levels only (hidden while the panel is open, which lists them all).
    return (
      <ul className="cg26-collapsed-only flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink">
        {levels
          .filter((l) => l.drawn)
          .map((l) => (
            <li key={l.id} className="flex items-center gap-1.5 tabular-nums">
              <LineSwatch color={CONTOUR_COLOUR} dash={contourDash(l.id)} />
              <span className="font-medium">{l.id}</span>
              <span className="text-muted">{shown(l)}</span>
            </li>
          ))}
      </ul>
    )
  }
  return (
    <>
      <ul className="grid min-w-0 grid-cols-1 gap-1 text-sm">
        {levels.map((l) => (
          <li key={l.id} className={clsx('flex min-w-0 items-center gap-2', !l.drawn && 'opacity-50')}>
            {l.drawn ? <LineSwatch color={CONTOUR_COLOUR} dash={contourDash(l.id)} /> : <span className="w-7 shrink-0" aria-hidden="true" />}
            <span className="w-11 shrink-0 font-medium text-ink">{l.id}</span>
            <span className="min-w-0 flex-1 truncate text-xs text-muted" title={l.name}>
              {l.name}
            </span>
            <span className="tabular-nums text-ink">{shown(l)}</span>
            {!l.drawn && <span className="sr-only">(outside the mapped heights)</span>}
          </li>
        ))}
      </ul>
      <p className="text-xs text-muted">Faded levels are outside the mapped heights.</p>
    </>
  )
}

export interface LegendProps {
  layer: MapLayer
  heights: MapHeights
  compact?: boolean
  className?: string
}

export function Legend({ layer, heights, compact = false, className }: LegendProps) {
  const units = layer.heights ? heightUnits(heights) : layer.units
  return (
    <div className={clsx('grid min-w-0 grid-cols-1', compact ? 'gap-1.5' : 'gap-2', className)}>
      <Title title={layer.title} units={units} />
      {layer.kind === 'raster' && rasterBar(layer, heights)}
      {layer.kind === 'points' && (
        <ColorBar
          stops={layer.scale.stops}
          ticks={linearTicks(layer.scale)}
          label={`${layer.title}, coloured from ${fmtNumber(layer.scale.domain[0])} to ${fmtNumber(layer.scale.domain[1])}${NBSP}${units}`}
        />
      )}
      {layer.kind === 'contours' && <Ladder layer={layer} heights={heights} compact={compact} />}
      {layer.kind === 'raster' && layer.lines && (
        <div className="flex items-center gap-2 text-xs text-ink">
          <LineSwatch color={EQUAL_DELAY_COLOUR} dash={EQUAL_DELAY_DASH} width={1.8} />
          <span>{layer.lines.legend}</span>
        </div>
      )}
      {layer.kind === 'raster' && layer.note && <p className="text-xs font-medium text-ink">{layer.note}</p>}
      {!compact && <p className="text-sm text-muted">{layer.about}</p>}
      {!compact && layer.heights && <p className="text-xs text-muted">{heightNote(heights)}</p>}
    </div>
  )
}
