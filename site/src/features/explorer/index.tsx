/**
 * The map viewer, the whole site: one site and one layer at a time over an orthophoto or a grey map, one legend, a
 * readout of the pixel under the pointer (or the tap, or the map centre with the keyboard), where allowed a 3D view,
 * and an info panel (About.tsx). State lives in the URL hash (#site=..&layer=..&base=photo|map&3d=0|1), so views can
 * be shared.
 *
 * Data (see scripts/data_layers.py): data/sites/index.json for the site list, data/sites/<id>/manifest.json for a
 * site's grid, layers and legends, and the images and lines it names. The code draws whatever layers a manifest
 * lists, so a build with other data needs no code change.
 */
import './explorer.css'
import { clsx } from 'clsx'
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { ButtonAnchor, Notice, Sheet, Skeleton, Slider } from '@/components/ui'
import { config } from '@/config'
import type { ExplorerProps } from '@/features/contracts'
import { dataUrl } from '@/lib/assets'
import { fmtInt } from '@/lib/format'
import { useSitesIndex, type Site, type SiteId } from '@/lib/sites'
import { useTheme } from '@/lib/theme'
import { AboutDialog, BrandPill, BrandRow } from './About'
import { pixelOf } from './geo'
import { formatHash, normaliseState, parseHash, type BaseMap, type ExplorerState } from './hashState'
import { siteFileUrl, useLocationHash, useMediaQuery, useSiteManifest, useValueGrid } from './hooks'
import { Ladder, Legend } from './Legend'
import { heightNote, heightOffset, readoutLine } from './readout'
import { ExplorerMap, type MapInit, type PointerInfo } from './mapController'
import { LayerChips, LayerList, MapToggles, PRELIMINARY, Readout, SiteChips, SiteSelect } from './Panel'
import type { TerrainHeights } from './terrarium'
import { heightLayer, rasterById, type MapLayer, type SiteManifest } from './types'
import { valueAt } from './valueGrid'

const WIDE = '(min-width: 768px)'
const FINE_POINTER = '(hover: hover) and (pointer: fine)'
/** Opacity of the coloured layer when the page opens (the basemap shows a little through it). */
const DEFAULT_OPACITY = 0.9

function hasWebGL2(): boolean {
  try {
    const gl = document.createElement('canvas').getContext('webgl2')
    gl?.getExtension('WEBGL_lose_context')?.loseContext()
    return gl !== null
  } catch {
    return false
  }
}

/** "97 Sentinel-2 images, 2023–2025", and "· preliminary" for a site whose map is drawn inside an automatic mask. */
function imagesLine(site: Pick<SiteManifest, 'images' | 'years'>, preliminary = false): string {
  return `${fmtInt(site.images)} Sentinel-2 images, ${site.years}${preliminary ? ` · ${PRELIMINARY}` : ''}`
}

/** The map centre's target while the keyboard moves the map. */
function Crosshair() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 32 32"
      className="pointer-events-none absolute top-1/2 left-1/2 z-10 size-8 -translate-x-1/2 -translate-y-1/2"
    >
      <circle cx="16" cy="16" r="6" fill="none" stroke="#FFFFFF" strokeWidth="4" />
      <path d="M16 1v9M16 22v9M1 16h9M22 16h9" stroke="#FFFFFF" strokeWidth="4" />
      <circle cx="16" cy="16" r="6" fill="none" stroke="#1F2A30" strokeWidth="2" />
      <path d="M16 1v9M16 22v9M1 16h9M22 16h9" stroke="#1F2A30" strokeWidth="2" />
    </svg>
  )
}

function NoWebGL({ site }: { site?: Site }) {
  return (
    <div className="flex h-full w-full items-center justify-center bg-sunken p-4">
      <div className="grid max-w-sm justify-items-center gap-4 rounded-xl border border-line bg-raised p-6 text-center shadow-float">
        {site && <img src={dataUrl(site.thumb)} alt={`Elevation map of ${site.longName}`} className="max-h-56 w-auto" />}
        <p className="text-ink">The interactive map needs WebGL 2, which this browser does not offer.</p>
        <ButtonAnchor href={config.repoUrl} variant="secondary">
          The repository
        </ButtonAnchor>
      </div>
    </div>
  )
}

interface Pointer {
  p: PointerInfo
  /** A tap or click: the probe marker shows where. */
  pinned: boolean
}

export default function Explorer({ defaultSite, enable3d }: ExplorerProps) {
  const [locationHash, replaceHash] = useLocationHash()
  const index = useSitesIndex()
  const sites = index.status === 'ready' ? index.data.sites : null
  const wide = useMediaQuery(WIDE)
  const finePointer = useMediaQuery(FINE_POINTER)
  const can3d = enable3d === 'on' || (enable3d === 'desktop' && wide && finePointer)
  const [theme] = useTheme()
  const [webgl] = useState(hasWebGL2)
  const [opacity, setOpacity] = useState(DEFAULT_OPACITY)
  const [about, setAbout] = useState(false)

  // ------------------------------------------------------------------------------------------ state in the hash
  const requested = useMemo(() => parseHash(locationHash, defaultSite), [locationHash, defaultSite])
  const manifest = useSiteManifest(requested.site)
  const site = manifest.status === 'ready' ? manifest.data : null
  // While another site loads, the map and legend keep showing the last one.
  const [shown, setShown] = useState<SiteManifest | null>(null)
  useEffect(() => {
    if (site) setShown(site)
  }, [site])
  const view = site ?? shown
  const state = useMemo(
    () => normaliseState(requested, site ? site.layers.map((l) => l.id) : null, can3d),
    [requested, site, can3d],
  )
  const hash = `#${formatHash(state)}`
  useEffect(() => {
    if (locationHash !== hash) replaceHash(hash)
  }, [hash, locationHash, replaceHash])
  const setState = useCallback(
    (patch: Partial<ExplorerState>) => replaceHash(`#${formatHash({ ...state, ...patch })}`),
    [replaceHash, state],
  )

  // ------------------------------------------------------------------------------------------ layers and values
  const layer: MapLayer | null = view ? (view.layers.find((l) => l.id === state.layer) ?? view.layers[0]) : null
  const under = view && layer?.kind === 'points' && layer.under ? (rasterById(view, layer.under) ?? null) : null
  const hLayer = view ? (heightLayer(view) ?? null) : null
  const valueLayer = layer?.kind === 'raster' && layer !== hLayer ? layer : null
  const heights = useValueGrid(view, hLayer)
  const values = useValueGrid(view, valueLayer)
  const terrain = useMemo<TerrainHeights | null>(
    () => (view && heights.grid ? { grid: view.grid, values: heights.grid, add: heightOffset(view.heights) } : null),
    [view, heights.grid],
  )

  // ------------------------------------------------------------------------------------------ readout
  const [pointer, setPointer] = useState<Pointer | null>(null)
  const [keyboard, setKeyboard] = useState(false)
  useEffect(() => setPointer(null), [view?.id])
  const readout = useMemo(() => {
    if (!view || !layer) return ''
    if (!pointer) return keyboard ? '' : finePointer ? 'Point at the map to read a value.' : 'Tap the map to read a value.'
    if (heights.failed || values.failed) return 'Values cannot be read on this map.'
    if (!heights.grid || (valueLayer && !values.grid)) return 'Reading values…'
    const px = pixelOf(view.grid, pointer.p.lng, pointer.p.lat)
    const height = px ? valueAt(heights.grid, px.col, px.row) : null
    const value = layer.kind !== 'raster' ? null : layer === hLayer ? height : px && values.grid ? valueAt(values.grid, px.col, px.row) : null
    return readoutLine(layer, view.heights, { height, value, point: pointer.p.point })
  }, [view, layer, pointer, keyboard, finePointer, heights, values, valueLayer, hLayer])

  // ------------------------------------------------------------------------------------------ the map
  const rootRef = useRef<HTMLDivElement>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const headRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<ExplorerMap | null>(null)
  const fitted = useRef<string | null>(null)
  const [sheetH, setSheetH] = useState(0)

  // Phones: the panel's collapsed height, so the map's own controls and the framing stay above it.
  useEffect(() => {
    const el = headRef.current
    if (wide || !el || typeof ResizeObserver === 'undefined') {
      setSheetH(0)
      return
    }
    const measure = () => setSheetH(Math.ceil(el.getBoundingClientRect().height) + 14)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [wide, view !== null])

  const fitPadding = useCallback((): MapInit['padding'] => {
    const w = rootRef.current?.clientWidth ?? 1024
    if (wide) return { top: 64, right: 24, bottom: 32, left: Math.min(352, Math.round(w * 0.42)) }
    const panel = headRef.current ? Math.ceil(headRef.current.getBoundingClientRect().height) + 14 : 160
    return { top: 60, right: 16, bottom: panel + 12, left: 16 }
  }, [wide])

  const mapLabel = view && layer ? `Map of ${view.longName}, ${layer.title}. Arrow keys move the map, plus and minus zoom.` : 'Map'

  useEffect(() => {
    if (!webgl || !view || mapRef.current || !boxRef.current) return
    mapRef.current = new ExplorerMap(
      boxRef.current,
      { base: state.base, theme, site: view, layer, under, opacity, threeD: state.threeD, terrain: null, probe: null },
      { bounds: view.grid.bounds, padding: fitPadding(), wide, hover: finePointer, urlOf: siteFileUrl, label: mapLabel },
      {
        onPointer: (p, how) => setPointer(p ? { p, pinned: how === 'tap' } : null),
        onCentre: (p) => setPointer({ p, pinned: false }),
        onKeyboard: (on) => {
          setKeyboard(on)
          if (!on) setPointer(null)
        },
      },
    )
    fitted.current = view.id
    // Created once, with the first site; later changes go through update() below.
  }, [webgl, view])

  useEffect(
    () => () => {
      mapRef.current?.destroy()
      mapRef.current = null
      fitted.current = null
    },
    [],
  )

  useEffect(() => {
    mapRef.current?.update({
      base: state.base,
      theme,
      site: view,
      layer,
      under,
      opacity,
      threeD: state.threeD,
      terrain,
      probe: pointer?.pinned ? [pointer.p.lng, pointer.p.lat] : null,
    })
  })

  useEffect(() => {
    const m = mapRef.current
    if (!m || !view || fitted.current === view.id) return
    m.fit(view.grid.bounds, fitPadding(), true)
    fitted.current = view.id
  }, [view, fitPadding])

  useEffect(() => {
    mapRef.current?.setWide(wide)
  }, [wide])
  useEffect(() => {
    mapRef.current?.setHover(finePointer)
  }, [finePointer])
  useEffect(() => {
    mapRef.current?.setLabel(mapLabel)
  }, [mapLabel])

  // ------------------------------------------------------------------------------------------ layout
  if (!webgl) return <NoWebGL site={sites?.find((s) => s.id === requested.site)} />

  const failed = index.status === 'error' || manifest.status === 'error'
  const pickSite = (id: SiteId) => setState({ site: id })
  const pickLayer = (id: string) => setState({ layer: id })
  const layers = view?.layers ?? []
  const preliminary = Boolean(view && sites?.find((s) => s.id === view.id)?.preliminary)
  // The opacity slider: only while a coloured image is on the map.
  const hasImage = layer?.kind === 'raster' || under !== null
  const opacityPct = Math.round(opacity * 100)
  const opacitySlider = hasImage ? (
    <Slider
      label="Opacity"
      min={0}
      max={100}
      step={5}
      value={opacityPct}
      valueText={`${opacityPct} %`}
      onChange={(v) => setOpacity(v / 100)}
    />
  ) : null

  const header = wide ? (
    <div className="grid grid-cols-1 gap-1 pt-1 pb-2 pr-1">
      <BrandRow onInfo={() => setAbout(true)} />
      {sites ? <SiteSelect sites={sites} value={requested.site} onChange={pickSite} /> : <Skeleton className="h-11 w-full" />}
      <p className="text-sm text-muted">{view ? imagesLine(view, preliminary) : ' '}</p>
    </div>
  ) : (
    <div ref={headRef} className="grid min-w-0 grid-cols-1 gap-2 py-1.5">
      {layers.length > 0 && <LayerChips layers={layers} value={layer?.id ?? ''} onChange={pickLayer} />}
      {view && layer ? <Legend layer={layer} heights={view.heights} compact /> : <Skeleton className="h-10 w-full" />}
      <Readout text={readout} />
    </div>
  )

  const body = wide ? (
    <div className="grid grid-cols-1 gap-4 pt-1">
      {failed && <Notice tone="warn">The map data could not be loaded.</Notice>}
      {layers.length > 0 && <LayerList layers={layers} value={layer?.id ?? ''} onChange={pickLayer} />}
      {view && layer && <Legend layer={layer} heights={view.heights} className="border-t border-line pt-4" />}
      {opacitySlider}
    </div>
  ) : (
    <div className="grid grid-cols-1 gap-3 pt-1">
      {failed && <Notice tone="warn">The map data could not be loaded.</Notice>}
      {view && (
        <p className="text-sm font-medium text-ink">
          {view.longName} · {imagesLine(view, preliminary)}
        </p>
      )}
      {layer && <p className="text-sm text-muted">{layer.about}</p>}
      {view && layer?.kind === 'contours' && <Ladder layer={layer} heights={view.heights} compact={false} />}
      {view && layer?.heights && <p className="text-xs text-muted">{heightNote(view.heights)}</p>}
      {opacitySlider}
    </div>
  )

  return (
    <div
      ref={rootRef}
      className="cg26-map relative h-full w-full overflow-hidden bg-sunken"
      style={{ '--sheet-h': `${wide ? 0 : sheetH}px` } as CSSProperties}
    >
      {/* MapLibre makes its container position: relative, so it fills the frame by size, not by inset. */}
      <div ref={boxRef} className="h-full w-full" />
      {!view && !failed && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center" role="status">
          <span className="sr-only">Loading the map</span>
          <Skeleton className="h-24 w-64" />
        </div>
      )}
      {keyboard && <Crosshair />}
      {!wide && sites && (
        <SiteChips sites={sites} value={requested.site} onChange={pickSite} className="absolute inset-x-0 top-0 z-10 px-3 py-2" />
      )}
      {!wide && <BrandPill onInfo={() => setAbout(true)} className="absolute top-14 left-3 z-10" />}
      <MapToggles
        className={clsx('absolute z-10', wide ? 'top-4 right-4' : 'top-14 right-3')}
        base={state.base}
        onBase={(b: BaseMap) => setState({ base: b })}
        can3d={can3d}
        threeD={state.threeD}
        on3d={(on) => setState({ threeD: on })}
      />
      <Sheet label="Map controls" header={header} footer={wide ? <Readout text={readout} /> : undefined}>
        {body}
      </Sheet>
      <AboutDialog open={about} onClose={() => setAbout(false)} with3d={can3d} />
    </div>
  )
}
