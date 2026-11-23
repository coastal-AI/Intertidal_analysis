import { clsx } from 'clsx'
import type { CSSProperties } from 'react'

export interface ColorBarTick {
  /** Position along the bar, 0 (left) to 1 (right). */
  at: number
  label: string
}

/** CSS gradient through ``stops`` (hex colours): continuous, or ``stepped`` into equal classes. */
export function gradientCss(stops: readonly string[], stepped = false, space: 'srgb' | 'oklab' = 'srgb'): string {
  if (stops.length === 0) return 'none'
  if (stops.length === 1) return `linear-gradient(to right, ${stops[0]}, ${stops[0]})`
  if (stepped) {
    const n = stops.length
    const parts = stops.map((c, i) => `${c} ${((i / n) * 100).toFixed(3)}% ${(((i + 1) / n) * 100).toFixed(3)}%`)
    return `linear-gradient(to right, ${parts.join(', ')})`
  }
  const how = space === 'oklab' ? 'to right in oklab' : 'to right'
  return `linear-gradient(${how}, ${stops.join(', ')})`
}

export interface ColorBarProps {
  /** Colours from left (low) to right (high), e.g. a layer's anchors. */
  stops: readonly string[]
  /** Labels under the bar. */
  ticks?: readonly ColorBarTick[]
  /** Equal classes instead of a smooth ramp (e.g. the eight uncertainty classes). */
  stepped?: boolean
  /** Accessible description of the whole scale, e.g. "Elevation from −2 m (light) to +1 m (dark)". */
  label?: string
  className?: string
}

/**
 * The colour bar of a legend: a smooth ramp interpolated in OKLab where the browser can (the map images are coloured
 * the same way), sRGB elsewhere, with labels positioned under it.
 */
export function ColorBar({ stops, ticks = [], stepped = false, label, className }: ColorBarProps) {
  const style = {
    '--cb-srgb': gradientCss(stops, stepped, 'srgb'),
    '--cb-oklab': gradientCss(stops, stepped, stepped ? 'srgb' : 'oklab'),
  } as CSSProperties
  return (
    <div className={clsx('w-full', className)} role={label ? 'img' : undefined} aria-label={label}>
      <div className="ui-colorbar h-3 w-full rounded-full ring-1 ring-inset ring-black/10 dark:ring-white/15" style={style} />
      {ticks.length > 0 && (
        <div className="relative mt-1.5 h-4 text-xs text-muted tabular-nums" aria-hidden="true">
          {ticks.map((t) => {
            const at = Math.min(1, Math.max(0, t.at))
            const shift = at < 0.04 ? '0%' : at > 0.96 ? '-100%' : '-50%'
            return (
              <span
                key={`${t.at}-${t.label}`}
                className="absolute top-0 leading-4 whitespace-nowrap"
                style={{ left: `${at * 100}%`, transform: `translateX(${shift})` }}
              >
                {t.label}
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}
