/**
 * A thin wrapper around MapLibre GL JS for the explorer: one map, its basemaps, the site's coloured layer (an image,
 * or tiles of one, over the site's grid, drawn with linear resampling), tide lines, points, a probe marker and the
 * optional 3D terrain.
 *
 * React keeps the wanted view (update()); the wrapper applies only what changed, once the style has loaded.
 * MapLibre is imported here and in the files this one imports, so it stays in the map page's own chunk.
 */
import 'maplibre-gl/dist/maplibre-gl.css'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url'
import {
  AttributionControl,
  Map as MapLibreMap,
  NavigationControl,
  ScaleControl,
  addProtocol,
  setWorkerUrl,
  type GeoJSONSource,
  type ImageSource,
  type LayerSpecification,
  type MapMouseEvent,
  type RasterDEMTileSource,
  type SkySpecification,
  type StyleSpecification,
} from 'maplibre-gl'
import type { Theme } from '@/lib/theme'
import { BASEMAPS, TERRAIN } from './basemaps'
import { tileTouchesGrid } from './geo'
import type { BaseMap } from './hashState'
import { CONTOUR_COLOUR, CONTOUR_STYLES, EQUAL_DELAY_COLOUR, EQUAL_DELAY_DASH } from './lineStyles'
import { compositeTile, type TerrainHeights } from './terrarium'
import type { LinearScale, MapLayer, RasterLayer, SiteManifest } from './types'

// Bundled, MapLibre would look for its worker next to this chunk: give it the worker file's own URL.
setWorkerUrl(workerUrl)

/** Top-left, top-right, bottom-right and bottom-left [lon, lat] of an image. */
type Corners = [[number, number], [number, number], [number, number], [number, number]]

const TEXT_FONT = ['Arial', 'Helvetica', 'sans-serif']
const EMPTY = { type: 'FeatureCollection' as const, features: [] }
const DEM_PROTOCOL = 'cg26dem'
const PITCH_3D = 60
const EXAGGERATION = 1.5

/** What the map should show; React passes it to update(). */
export interface MapView {
  base: BaseMap
  theme: Theme
  site: SiteManifest | null
  /** The active layer of the site. */
  layer: MapLayer | null
  /** A raster drawn under a points layer (its "under" layer), else null. */
  under: RasterLayer | null
  /** Opacity of the coloured image, 0 to 1. */
  opacity: number
  threeD: boolean
  /** Heights for the 3D terrain (the site's elevation values), once loaded. */
  terrain: TerrainHeights | null
  /** [lon, lat] of the probe marker, or null. */
  probe: [number, number] | null
}

export interface PointerInfo {
  lng: number
  lat: number
  /** Properties of the point feature under the pointer (points layers), else null. */
  point: Record<string, unknown> | null
}

export interface MapCallbacks {
  /** Pointer over the map (hover), or a tap or click; null when the pointer leaves the map. */
  onPointer: (p: PointerInfo | null, how: 'hover' | 'tap') => void
  /** The map centre while the map has keyboard focus. */
  onCentre: (p: PointerInfo) => void
  /** Keyboard focus on the map (true) or not (false). */
  onKeyboard: (active: boolean) => void
}

export interface MapInit {
  /** Initial [w, s, e, n] to fit, and the padding around it. */
  bounds: [number, number, number, number]
  padding: { top: number; right: number; bottom: number; left: number }
  /** Wide screen: zoom buttons are shown. */
  wide: boolean
  /** The pointer can hover (a mouse): the readout follows it. */
  hover: boolean
  /** Absolute URL of a file of a site's folder. */
  urlOf: (siteId: string, file: string) => string
  /** Accessible name of the map. */
  label: string
}

// ---------------------------------------------------------------------------------------------------- 3D terrain
let terrainHeights: TerrainHeights | null = null

function makeCanvas(size: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(size, size)
  const c = document.createElement('canvas')
  c.width = size
  c.height = size
  return c
}

async function canvasPng(canvas: OffscreenCanvas | HTMLCanvasElement): Promise<Blob> {
  if ('convertToBlob' in canvas) return canvas.convertToBlob({ type: 'image/png' })
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('terrain: PNG encoding failed'))), 'image/png'),
  )
}

/** One terrain tile: the open DEM tile, with MAREA's heights written in where the site's map has them. */
async function demTile(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
  const m = /^[a-z0-9]+:\/\/(\d+)\/(\d+)\/(\d+)/.exec(url)
  if (!m) throw new Error(`terrain: bad tile URL ${url}`)
  const [z, x, y] = m.slice(1).map(Number)
  const res = await fetch(TERRAIN.tiles.replace('{z}', String(z)).replace('{x}', String(x)).replace('{y}', String(y)), {
    signal,
  })
  if (!res.ok) return new ArrayBuffer(0) // an empty tile: no elevation there
  const blob = await res.blob()
  const t = terrainHeights
  if (!t || !tileTouchesGrid(t.grid, z, x, y)) return blob.arrayBuffer()
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  const size = bitmap.width
  const canvas = makeCanvas(size)
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null
  if (!ctx) {
    bitmap.close()
    return blob.arrayBuffer()
  }
  ctx.drawImage(bitmap, 0, 0)
  bitmap.close()
  const img = ctx.getImageData(0, 0, size, size)
  if (compositeTile(img.data, size, z, x, y, t) === 0) return blob.arrayBuffer()
  ctx.putImageData(img, 0, 0)
  return (await canvasPng(canvas)).arrayBuffer()
}

addProtocol(DEM_PROTOCOL, async (params, abortController) => ({
  data: await demTile(params.url, abortController.signal),
}))

// ---------------------------------------------------------------------------------------------------- style
function sky(theme: Theme): SkySpecification {
  return theme === 'dark'
    ? { 'sky-color': '#13212A', 'horizon-color': '#24404B', 'fog-color': '#13212A', 'sky-horizon-blend': 0.6,
        'horizon-fog-blend': 0.6, 'fog-ground-blend': 0.7, 'atmosphere-blend': 0 }
    : { 'sky-color': '#BFD9E0', 'horizon-color': '#F2F6F7', 'fog-color': '#F2F6F7', 'sky-horizon-blend': 0.6,
        'horizon-fog-blend': 0.6, 'fog-ground-blend': 0.7, 'atmosphere-blend': 0 }
}

const BACKGROUND: Record<Theme, string> = { light: '#E9EFF1', dark: '#0A1215' }

/** Line casing (white, under every line set) and labels along the lines, from zoom 13. */
const casing = (id: string, source: string): LayerSpecification => ({
  id,
  type: 'line',
  source,
  layout: { 'line-join': 'round', 'line-cap': 'round', visibility: 'none' },
  paint: { 'line-color': '#FFFFFF', 'line-width': 3.5, 'line-opacity': 0.85 },
})
const label = (id: string, source: string, color: string): LayerSpecification => ({
  id,
  type: 'symbol',
  source,
  minzoom: 13,
  layout: {
    'symbol-placement': 'line-center',
    'text-field': ['get', 'label'],
    'text-font': TEXT_FONT,
    'text-size': 12,
    'text-max-angle': 80,
    visibility: 'none',
  },
  paint: { 'text-color': color, 'text-halo-color': '#FFFFFF', 'text-halo-width': 1.5 },
})

const INK = CONTOUR_COLOUR
const DELAY_LINE = EQUAL_DELAY_COLOUR

/** The tide-line layers: one per dash, each filtered to its levels. */
const CONTOUR_LINES = CONTOUR_STYLES.map((c) => ({
  id: c.id,
  filter: ['in', ['get', 'datum'], ['literal', [...c.datums]]] as unknown[],
  dash: c.dash ?? undefined,
}))

function baseStyle(base: BaseMap, theme: Theme): StyleSpecification {
  const sources: StyleSpecification['sources'] = {
    contours: { type: 'geojson', data: EMPTY },
    equal: { type: 'geojson', data: EMPTY },
    points: { type: 'geojson', data: EMPTY },
    probe: { type: 'geojson', data: EMPTY },
  }
  const layers: LayerSpecification[] = [{ id: 'background', type: 'background', paint: { 'background-color': BACKGROUND[theme] } }]
  for (const b of BASEMAPS) {
    sources[`base-${b.id}`] = { type: 'raster', tiles: b.tiles, tileSize: 256, maxzoom: b.maxzoom, attribution: b.attribution }
    layers.push({ id: `base-${b.id}`, type: 'raster', source: `base-${b.id}`, layout: { visibility: b.id === base ? 'visible' : 'none' } })
  }
  layers.push(casing('contours-casing', 'contours'))
  for (const c of CONTOUR_LINES) {
    layers.push({
      id: c.id,
      type: 'line',
      source: 'contours',
      filter: c.filter as never,
      layout: { 'line-join': 'round', 'line-cap': c.dash ? 'butt' : 'round', visibility: 'none' },
      paint: { 'line-color': INK, 'line-width': 1.5, ...(c.dash ? { 'line-dasharray': c.dash } : {}) },
    })
  }
  layers.push(casing('equal-casing', 'equal'))
  layers.push({
    id: 'equal-line',
    type: 'line',
    source: 'equal',
    layout: { 'line-join': 'round', 'line-cap': 'butt', visibility: 'none' },
    paint: { 'line-color': DELAY_LINE, 'line-width': 1.8, 'line-dasharray': EQUAL_DELAY_DASH },
  })
  layers.push({
    id: 'points-circle',
    type: 'circle',
    source: 'points',
    layout: { visibility: 'none' },
    paint: { 'circle-radius': 5, 'circle-color': INK, 'circle-stroke-color': '#FFFFFF', 'circle-stroke-width': 1.5 },
  })
  layers.push(label('contours-label', 'contours', INK))
  layers.push(label('equal-label', 'equal', DELAY_LINE))
  layers.push({
    id: 'probe-halo',
    type: 'circle',
    source: 'probe',
    paint: { 'circle-radius': 8, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': '#FFFFFF', 'circle-stroke-width': 4 },
  })
  layers.push({
    id: 'probe-ring',
    type: 'circle',
    source: 'probe',
    paint: { 'circle-radius': 8, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': INK, 'circle-stroke-width': 2 },
  })
  return { version: 8, sources, layers }
}

/** Colour of points: the scale's stops spread evenly over its domain. */
function pointColour(scale: LinearScale, property: string): unknown {
  const [d0, d1] = scale.domain
  const n = scale.stops.length
  const stops = scale.stops.flatMap((c, i) => [d0 + ((d1 - d0) * i) / (n - 1), c])
  return ['interpolate', ['linear'], ['to-number', ['get', property]], ...stops]
}

const reducedMotion = () => typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

// ---------------------------------------------------------------------------------------------------- the map
export class ExplorerMap {
  readonly map: MapLibreMap
  private readonly init: MapInit
  private readonly cb: MapCallbacks
  private ready = false
  private want: MapView
  private have: Record<string, string | boolean | null> = {}
  /** Ids of the overlay tiles on the map (each an image source and its raster layer). */
  private rasterIds: string[] = []
  private nav: NavigationControl | null = null
  private hoverCapable: boolean
  private frame = 0
  private centreFrame = 0
  private keyboard = false
  private terrainKey = 0
  private wide = true

  constructor(container: HTMLElement, view: MapView, init: MapInit, cb: MapCallbacks) {
    this.want = view
    this.init = init
    this.cb = cb
    this.hoverCapable = init.hover
    this.map = new MapLibreMap({
      container,
      style: baseStyle(view.base, view.theme),
      bounds: init.bounds,
      fitBoundsOptions: { padding: init.padding },
      maxZoom: 18,
      minZoom: 5,
      maxPitch: 70,
      attributionControl: false,
      pixelRatio: Math.min(window.devicePixelRatio || 1, 2),
    })
    this.have.base = view.base
    this.have.theme = view.theme
    this.map.addControl(new AttributionControl({ compact: true }), 'bottom-right')
    this.map.addControl(new ScaleControl({ unit: 'metric', maxWidth: 96 }), 'bottom-left')
    this.setWide(init.wide)
    this.setRotation(false)
    const canvas = this.map.getCanvas()
    canvas.setAttribute('aria-label', init.label)
    canvas.addEventListener('focus', this.onFocus)
    canvas.addEventListener('blur', this.onBlur)
    this.map.on('mousemove', this.onMove)
    this.map.on('mouseout', this.onOut)
    this.map.on('click', this.onClick)
    this.map.on('move', this.onCameraMove)
    this.map.on('error', (e) => console.warn('map:', e.error?.message ?? e))
    this.map.once('load', () => {
      this.ready = true
      this.apply()
      this.foldAttribution()
    })
    this.map.once('idle', () => this.foldAttribution())
    if (new URLSearchParams(window.location.search).has('debug')) {
      ;(window as unknown as { __mareaMap?: MapLibreMap }).__mareaMap = this.map
    }
  }

  /** Change what the map shows (only the given fields). */
  update(next: Partial<MapView>): void {
    this.want = { ...this.want, ...next }
    if (this.ready) this.apply()
  }

  /** Fit [w, s, e, n] in the map, with padding; animated unless the visitor prefers reduced motion. */
  fit(bounds: [number, number, number, number], padding: MapInit['padding'], animate: boolean): void {
    const bearing = this.want.threeD ? this.map.getBearing() : 0
    this.map.fitBounds(bounds, { padding, bearing, duration: animate && !reducedMotion() ? 900 : 0 })
  }

  /** Show the zoom buttons on wide screens only. */
  setWide(wide: boolean): void {
    this.wide = wide
    if (wide && !this.nav) {
      this.nav = new NavigationControl({ showCompass: true, visualizePitch: true })
      this.map.addControl(this.nav, 'bottom-right')
    } else if (!wide && this.nav) {
      this.map.removeControl(this.nav)
      this.nav = null
    }
  }

  setHover(hover: boolean): void {
    this.hoverCapable = hover
  }

  setLabel(text: string): void {
    this.map.getCanvas().setAttribute('aria-label', text)
  }

  resize(): void {
    this.map.resize()
  }

  destroy(): void {
    cancelAnimationFrame(this.frame)
    cancelAnimationFrame(this.centreFrame)
    const canvas = this.map.getCanvas()
    canvas.removeEventListener('focus', this.onFocus)
    canvas.removeEventListener('blur', this.onBlur)
    if (terrainHeights === this.want.terrain) terrainHeights = null
    this.map.remove()
  }

  // ------------------------------------------------------------------------------------------------ applying a view
  private vis(ids: string[], on: boolean): void {
    for (const id of ids) if (this.map.getLayer(id)) this.map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none')
  }

  private setGeoJSON(source: string, url: string | null): void {
    const key = `geo:${source}`
    if (this.have[key] === url) return
    this.map.getSource<GeoJSONSource>(source)?.setData(url ?? EMPTY)
    this.have[key] = url
  }

  /** The coloured image: one image source and raster layer per overlay tile (marea-raster-0, -1, …). */
  private setRaster(layer: RasterLayer | null): void {
    const site = this.want.site
    const tiles =
      layer && site
        ? layer.overlay.tiles.map((t) => ({ url: this.init.urlOf(site.id, t.file), coordinates: t.corners as Corners }))
        : []
    const key = tiles.length ? tiles.map((t) => t.url).join(' ') : null
    if (this.have.raster === key) return
    this.have.raster = key
    if (!tiles.length) {
      this.vis(this.rasterIds, false)
      return
    }
    // A site with fewer tiles than the last one: drop the extra tiles.
    while (this.rasterIds.length > tiles.length) {
      const id = this.rasterIds.pop() as string
      this.map.removeLayer(id)
      this.map.removeSource(id)
    }
    tiles.forEach(({ url, coordinates }, i) => {
      const id = `marea-raster-${i}`
      const src = this.map.getSource<ImageSource>(id)
      if (src) src.updateImage({ url, coordinates })
      else {
        this.map.addSource(id, { type: 'image', url, coordinates })
        this.map.addLayer(
          {
            id,
            type: 'raster',
            source: id,
            // Linear: the overlay is drawn at twice the grid's resolution from the values; its colours and its faded
            // edge blend smoothly between pixels instead of showing them as blocks.
            paint: { 'raster-opacity': this.want.opacity, 'raster-resampling': 'linear', 'raster-fade-duration': 0 },
          },
          'contours-casing',
        )
        this.rasterIds.push(id)
      }
    })
    this.have.opacity = null // new tiles take the current opacity in setOpacity()
    this.vis(this.rasterIds, true)
  }

  private setOpacity(opacity: number): void {
    const key = String(opacity)
    if (this.have.opacity === key || this.rasterIds.length === 0) return
    for (const id of this.rasterIds) this.map.setPaintProperty(id, 'raster-opacity', opacity)
    this.have.opacity = key
  }

  /** Phones: start with the attribution folded into its (i) button, as MapLibre folds it after the first drag. */
  private foldAttribution(): void {
    if (this.wide) return
    this.map.getContainer().querySelector('.maplibregl-ctrl-attrib')?.classList.remove('maplibregl-compact-show')
  }

  private setCredit(text: string | null): void {
    if (this.have.credit === text) return
    if (this.map.getLayer('credit')) this.map.removeLayer('credit')
    if (this.map.getSource('credit')) this.map.removeSource('credit')
    if (text) {
      // An empty source whose only job is to carry the site's credit line into the attribution control.
      this.map.addSource('credit', { type: 'geojson', data: EMPTY, attribution: text })
      this.map.addLayer({ id: 'credit', type: 'circle', source: 'credit', paint: { 'circle-radius': 0 } })
    }
    this.have.credit = text
  }

  private setRotation(on: boolean): void {
    if (on) {
      this.map.dragRotate.enable()
      this.map.touchZoomRotate.enableRotation()
      this.map.touchPitch.enable()
      this.map.keyboard.enableRotation()
    } else {
      this.map.dragRotate.disable()
      this.map.touchZoomRotate.disableRotation()
      this.map.touchPitch.disable()
      this.map.keyboard.disableRotation()
    }
  }

  private setTerrain(): void {
    const { threeD, terrain, site } = this.want
    terrainHeights = threeD ? terrain : null
    const key = threeD ? `${site?.id ?? ''}:${terrain ? 'heights' : 'open'}` : null
    if (this.have.terrain === key) return
    const wasOn = typeof this.have.terrain === 'string'
    this.have.terrain = key
    if (!threeD) {
      this.map.setTerrain(null)
      this.setRotation(false)
      if (wasOn) this.map.easeTo({ pitch: 0, bearing: 0, duration: reducedMotion() ? 0 : 700 })
      return
    }
    this.terrainKey += 1
    const tiles = [`${DEM_PROTOCOL}://{z}/{x}/{y}?v=${this.terrainKey}`]
    const src = this.map.getSource<RasterDEMTileSource>('terrain')
    if (src) src.setTiles(tiles)
    else {
      this.map.addSource('terrain', {
        type: 'raster-dem',
        tiles,
        tileSize: TERRAIN.tileSize,
        maxzoom: TERRAIN.maxzoom,
        encoding: 'terrarium',
        attribution: TERRAIN.attribution,
      })
    }
    this.map.setTerrain({ source: 'terrain', exaggeration: EXAGGERATION })
    this.map.setSky(sky(this.want.theme))
    this.setRotation(true)
    if (!wasOn) {
      const bearing = this.map.getBearing() || -20
      this.map.easeTo({ pitch: PITCH_3D, bearing, duration: reducedMotion() ? 0 : 900 })
    }
  }

  private apply(): void {
    const w = this.want
    const map = this.map
    if (this.have.base !== w.base) {
      for (const b of BASEMAPS) this.vis([`base-${b.id}`], b.id === w.base)
      this.have.base = w.base
    }
    if (this.have.theme !== w.theme) {
      map.setPaintProperty('background', 'background-color', BACKGROUND[w.theme])
      if (w.threeD) map.setSky(sky(w.theme))
      this.have.theme = w.theme
    }
    const site = w.site
    const layer = w.layer
    this.setCredit(site?.attribution ?? null)
    // The coloured image: the raster layer itself, or the raster under a points layer.
    this.setRaster(layer?.kind === 'raster' ? layer : layer?.kind === 'points' ? w.under : null)
    this.setOpacity(w.opacity)
    // Tide lines.
    const contours = layer?.kind === 'contours' && site ? this.init.urlOf(site.id, layer.file) : null
    this.setGeoJSON('contours', contours)
    this.vis(['contours-casing', 'contours-label', ...CONTOUR_LINES.map((c) => c.id)], contours !== null)
    // Lines drawn with a raster (equal delay).
    const equal = layer?.kind === 'raster' && layer.lines && site ? this.init.urlOf(site.id, layer.lines.file) : null
    this.setGeoJSON('equal', equal)
    this.vis(['equal-casing', 'equal-line', 'equal-label'], equal !== null)
    // Points.
    const points = layer?.kind === 'points' && site ? this.init.urlOf(site.id, layer.file) : null
    this.setGeoJSON('points', points)
    if (layer?.kind === 'points' && this.have.pointsColour !== `${site?.id}:${layer.id}`) {
      map.setPaintProperty('points-circle', 'circle-color', pointColour(layer.scale, layer.colorBy) as never)
      this.have.pointsColour = `${site?.id}:${layer.id}`
    }
    this.vis(['points-circle'], points !== null)
    // Probe marker.
    const probeKey = w.probe ? w.probe.join(',') : null
    if (this.have.probe !== probeKey) {
      map.getSource<GeoJSONSource>('probe')?.setData(
        w.probe ? { type: 'Feature', geometry: { type: 'Point', coordinates: w.probe }, properties: {} } : EMPTY,
      )
      this.have.probe = probeKey
    }
    this.setTerrain()
  }

  // ------------------------------------------------------------------------------------------------ pointer and keyboard
  private pointAt(e: MapMouseEvent): Record<string, unknown> | null {
    if (!this.map.getLayer('points-circle') || this.map.getLayoutProperty('points-circle', 'visibility') !== 'visible') return null
    const { x, y } = e.point
    const hits = this.map.queryRenderedFeatures(
      [
        [x - 7, y - 7],
        [x + 7, y + 7],
      ],
      { layers: ['points-circle'] },
    )
    return hits.length ? (hits[0].properties as Record<string, unknown>) : null
  }

  private onMove = (e: MapMouseEvent): void => {
    if (!this.hoverCapable || this.keyboard) return
    cancelAnimationFrame(this.frame)
    this.frame = requestAnimationFrame(() => this.cb.onPointer({ lng: e.lngLat.lng, lat: e.lngLat.lat, point: this.pointAt(e) }, 'hover'))
  }

  private onOut = (): void => {
    if (!this.hoverCapable || this.keyboard) return
    cancelAnimationFrame(this.frame)
    this.cb.onPointer(null, 'hover')
  }

  private onClick = (e: MapMouseEvent): void => {
    cancelAnimationFrame(this.frame)
    this.cb.onPointer({ lng: e.lngLat.lng, lat: e.lngLat.lat, point: this.pointAt(e) }, 'tap')
  }

  private onCameraMove = (): void => {
    if (!this.keyboard) return
    cancelAnimationFrame(this.centreFrame)
    this.centreFrame = requestAnimationFrame(() => {
      const c = this.map.getCenter()
      this.cb.onCentre({ lng: c.lng, lat: c.lat, point: null })
    })
  }

  private onFocus = (): void => {
    const canvas = this.map.getCanvas()
    let keyboard = true
    try {
      keyboard = canvas.matches(':focus-visible')
    } catch {
      keyboard = true
    }
    if (!keyboard) return
    this.keyboard = true
    this.cb.onKeyboard(true)
    const c = this.map.getCenter()
    this.cb.onCentre({ lng: c.lng, lat: c.lat, point: null })
  }

  private onBlur = (): void => {
    if (!this.keyboard) return
    this.keyboard = false
    this.cb.onKeyboard(false)
  }
}
