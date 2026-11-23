/**
 * The explorer's controls: site switcher, layer list, readout line and the map toggles (basemap, 2D/3D).
 * Wide screens get a select and a vertical list in a floating card; phones get rows of chips. A site flagged
 * preliminary in the site list gets that word next to its name.
 */
import { clsx } from 'clsx'
import { ChevronDown } from 'lucide-react'
import { useId, type ReactNode } from 'react'
import { Chip, gradientCss, Segmented } from '@/components/ui'
import type { Site, SiteId } from '@/lib/sites'
import { BASEMAPS } from './basemaps'
import type { BaseMap } from './hashState'
import { CONTOUR_COLOUR } from './lineStyles'
import type { MapLayer } from './types'

/** The word shown next to the name of a site whose map is drawn inside an automatic mask (index.json: preliminary). */
export const PRELIMINARY = 'preliminary'

/** A small picture of a layer: its colours, or a line for tide lines. */
function LayerIcon({ layer }: { layer: MapLayer }) {
  if (layer.kind === 'contours') {
    return (
      <svg aria-hidden="true" viewBox="0 0 16 16" className="size-4 shrink-0 text-ink">
        <path d="M1 11c3-4 5 1 8-3s5-2 6-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <path d="M1 14c3-3 6 0 9-3s4-2 5-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeDasharray="2.5 2" />
      </svg>
    )
  }
  const stops =
    layer.kind === 'raster' && layer.scale.type === 'classes' ? layer.scale.classes.map((c) => c.color) : layer.scale.type === 'linear' ? layer.scale.stops : [CONTOUR_COLOUR]
  return (
    <span
      aria-hidden="true"
      className={clsx('size-4 shrink-0 ring-1 ring-inset ring-black/10 dark:ring-white/15', layer.kind === 'points' ? 'rounded-full' : 'rounded-[5px]')}
      style={{ backgroundImage: gradientCss(stops, layer.kind === 'raster' && layer.scale.type === 'classes') }}
    />
  )
}

export function SiteSelect({ sites, value, onChange }: { sites: Site[]; value: SiteId; onChange: (id: SiteId) => void }) {
  const id = useId()
  return (
    <div className="relative">
      <label htmlFor={id} className="sr-only">
        Site
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as SiteId)}
        className="h-11 w-full cursor-pointer appearance-none rounded-lg border border-control bg-surface pr-10 pl-3 text-base font-semibold text-ink hover:bg-sunken"
      >
        {sites.map((s) => (
          <option key={s.id} value={s.id}>
            {s.preliminary ? `${s.longName} · ${PRELIMINARY}` : s.longName}
          </option>
        ))}
      </select>
      <ChevronDown aria-hidden="true" className="pointer-events-none absolute top-1/2 right-3 size-5 -translate-y-1/2 text-muted" />
    </div>
  )
}

/** A row of chips that scrolls sideways (phones). */
function ChipRow({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div
      role="group"
      aria-label={label}
      className={clsx('flex gap-2 overflow-x-auto overscroll-x-contain [scrollbar-width:none] [&::-webkit-scrollbar]:hidden', className)}
    >
      {children}
    </div>
  )
}

export function SiteChips({ sites, value, onChange, className }: { sites: Site[]; value: SiteId; onChange: (id: SiteId) => void; className?: string }) {
  return (
    <ChipRow label="Site" className={className}>
      {sites.map((s) => (
        <Chip key={s.id} size="sm" selected={s.id === value} onClick={() => onChange(s.id)} className="shadow-card">
          {s.name}
          {s.preliminary && <span className="text-xs font-normal opacity-80">{PRELIMINARY}</span>}
        </Chip>
      ))}
    </ChipRow>
  )
}

export function LayerChips({ layers, value, onChange }: { layers: MapLayer[]; value: string; onChange: (id: string) => void }) {
  return (
    <ChipRow label="Layer" className="-mx-1 px-1 py-0.5">
      {layers.map((l) => (
        <Chip key={l.id} size="sm" selected={l.id === value} onClick={() => onChange(l.id)}>
          {l.title}
        </Chip>
      ))}
    </ChipRow>
  )
}

export function LayerList({ layers, value, onChange }: { layers: MapLayer[]; value: string; onChange: (id: string) => void }) {
  const name = useId()
  return (
    <fieldset className="min-w-0">
      <legend className="sr-only">Layer</legend>
      <div className="grid gap-0.5">
        {layers.map((l) => (
          <label
            key={l.id}
            className={clsx(
              'flex min-h-10 cursor-pointer items-center gap-3 rounded-lg px-3 text-sm text-ink transition-colors duration-150',
              'has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-focus',
              l.id === value ? 'bg-primary-soft font-semibold' : 'hover:bg-sunken',
            )}
          >
            <input type="radio" className="sr-only" name={name} value={l.id} checked={l.id === value} onChange={() => onChange(l.id)} />
            <LayerIcon layer={l} />
            <span>{l.title}</span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}

export function Readout({ text, className }: { text: string; className?: string }) {
  return (
    <p aria-live="polite" className={clsx('min-h-5 text-sm leading-5 text-ink tabular-nums', className)}>
      {text}
    </p>
  )
}

export interface MapTogglesProps {
  base: BaseMap
  onBase: (b: BaseMap) => void
  /** Offer the 2D/3D toggle. */
  can3d: boolean
  threeD: boolean
  on3d: (on: boolean) => void
  className?: string
}

export function MapToggles({ base, onBase, can3d, threeD, on3d, className }: MapTogglesProps) {
  return (
    <div className={clsx('flex gap-2', className)}>
      <div className="rounded-full shadow-float">
        <Segmented<BaseMap>
          name="basemap"
          legend="Basemap"
          size="sm"
          value={base}
          onChange={onBase}
          options={BASEMAPS.map((b) => ({ value: b.id, label: b.label }))}
        />
      </div>
      {can3d && (
        <div className="rounded-full shadow-float">
          <Segmented<'2d' | '3d'>
            name="view"
            legend="View"
            size="sm"
            value={threeD ? '3d' : '2d'}
            onChange={(v) => on3d(v === '3d')}
            options={[
              { value: '2d', label: '2D' },
              { value: '3d', label: '3D' },
            ]}
          />
        </div>
      )}
    </div>
  )
}
