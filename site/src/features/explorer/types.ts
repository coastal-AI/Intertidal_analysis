/**
 * A site's map: data/sites/<id>/manifest.json, schema "cg26-explorer-site/2", written by scripts/data_layers.py.
 *
 * Paths inside it are relative to the manifest's folder. The explorer is generic: it draws whatever layers the
 * manifest lists, by kind ("raster": a coloured image, in one or more tiles, with a value image for the readout;
 * "contours": lines at the tidal levels; "points": a GeoJSON of points coloured by one property), so a build with
 * other data needs no code.
 */
import { isSiteId, type SiteId } from '@/lib/sites'

export const SITE_SCHEMA = 'cg26-explorer-site/2'

/** A continuous colour scale: ``stops`` evenly spaced over ``domain`` (interpolated in OKLab, as the images are). */
export interface LinearScale {
  type: 'linear'
  domain: [number, number]
  ticks: number[]
  stops: string[]
}

/** A classed colour scale (e.g. the eight uncertainty widths). */
export interface ClassScale {
  type: 'classes'
  classes: { value: number; color: string }[]
}

export type ColourScale = LinearScale | ClassScale

/** A value image: code = 256 R + G, value = offset + scale × code, code 0 = no data. */
export interface ValueFile {
  file: string
  scale: number
  offset: number
  /** [col, row, code] probes the browser compares with what it decodes. */
  check: [number, number, number][]
}

/** One image of an overlay, placed by its own corners. */
export interface OverlayTile {
  file: string
  /** Size in pixels. */
  width: number
  height: number
  /** [lon, lat] of the top-left, top-right, bottom-right and bottom-left corners. */
  corners: [number, number][]
}

/**
 * The coloured image of a raster layer, drawn from its values at twice the grid's resolution over the grid's extent:
 * one image, or tiles side by side where one would be too large for a texture (row by row from the top left, in rows
 * and columns that line up).
 */
export interface Overlay {
  /** Size in pixels of the whole overlay. */
  width: number
  height: number
  tiles: OverlayTile[]
}

/** Lines drawn with a raster layer (e.g. lines of equal delay over the delay map). */
export interface LineOverlay {
  file: string
  title: string
  /** One legend line, e.g. "Lines of equal delay, every 10 min". */
  legend: string
}

interface LayerBase {
  id: string
  title: string
  /** One sentence about the layer, shown under its legend. */
  about: string
  /** Units of the values, e.g. "m", "%", "min". */
  units: string
  /** Values are heights in the site's height system (NMMA = value + C where the site has C). */
  heights?: boolean
}

export interface RasterLayer extends LayerBase {
  kind: 'raster'
  /** Digits after the decimal point in the readout. */
  decimals: number
  /** The coloured RGBA image(s) over the site's grid. */
  overlay: Overlay
  values: ValueFile
  scale: ColourScale
  lines?: LineOverlay
  /** One line under the legend, e.g. "No delay here: every section is 0 min." */
  note?: string
  /** The sign of a value matters (e.g. a difference between two maps): the readout writes "+" before positive values. */
  signed?: boolean
}

export interface TideLevel {
  id: string
  name: string
  /** Height on the map's zero (the tide model's mean sea level). */
  level: number
  /** Height above NMMA, or null where the site has no C. */
  nmma: number | null
  /** Whether the map has a line at this level. */
  drawn: boolean
}

export interface ContourLayer extends LayerBase {
  kind: 'contours'
  /** GeoJSON lines with properties "datum" and "label". */
  file: string
  /** LAT to HAT, increasing. */
  levels: TideLevel[]
}

export interface PointsLayer extends LayerBase {
  kind: 'points'
  /** GeoJSON points. */
  file: string
  /** A raster layer of the same site drawn under the points (optional). */
  under?: string
  /** Property that colours the points, on ``scale`` (values as shown, in ``units``). */
  colorBy: string
  scale: LinearScale
  decimals: number
  /** Properties shown by the readout over a point, in this order. */
  fields: { property: string; label: string }[]
}

export type MapLayer = RasterLayer | ContourLayer | PointsLayer

/** Heights above NMMA: NMMA height = map value + C. */
export interface HeightsNMMA {
  datum: 'NMMA'
  C: number
  sigma: number
  source: string
  /** Short gauge name, e.g. "Ferrol". */
  gauge: string
}

/** Heights above the tide model's mean sea level (no NMMA link at the site). */
export interface HeightsModel {
  datum: 'model'
  C: null
  note: string
}

export type MapHeights = HeightsNMMA | HeightsModel

/** The site's Web Mercator grid, shared by all its raster layers. */
export interface SiteGrid {
  width: number
  height: number
  /** Top-left corner (EPSG:3857, m). */
  x0: number
  y0: number
  /** Pixel size (EPSG:3857, m). */
  res: number
  /** [lon, lat] of the top-left, top-right, bottom-right and bottom-left corners. */
  corners: [number, number][]
  /** [west, south, east, north]. */
  bounds: [number, number, number, number]
  centre: [number, number]
}

export interface SiteManifest {
  schema: typeof SITE_SCHEMA
  id: SiteId
  name: string
  longName: string
  images: number
  firstImage: string
  lastImage: string
  years: string
  heights: MapHeights
  grid: SiteGrid
  layers: MapLayer[]
  /** Credit line of the map data. */
  attribution: string
}

function fail(id: string, reason: string): never {
  throw new Error(`sites/${id}/manifest.json: ${reason}`)
}

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const isStr = (v: unknown): v is string => typeof v === 'string'
const isLonLat = (v: unknown): boolean => Array.isArray(v) && v.length === 2 && v.every(isNum)

function checkOverlay(id: string, layer: string, o: unknown): void {
  const v = o as Record<string, unknown> | undefined
  const tiles = v?.tiles as Record<string, unknown>[] | undefined
  if (!v || !isNum(v.width) || !isNum(v.height) || !Array.isArray(tiles) || tiles.length === 0) fail(id, `${layer}: bad overlay`)
  for (const t of tiles) {
    const c = t.corners as unknown[]
    if (!isStr(t.file) || !isNum(t.width) || !isNum(t.height) || !Array.isArray(c) || c.length !== 4 || !c.every(isLonLat)) {
      fail(id, `${layer}: bad overlay tile`)
    }
  }
}

function checkScale(id: string, layer: string, s: unknown, linearOnly = false): void {
  const o = s as Record<string, unknown> | undefined
  if (o?.type === 'linear') {
    const d = o.domain as unknown[]
    if (!Array.isArray(d) || d.length !== 2 || !d.every(isNum) || (d[1] as number) <= (d[0] as number)) {
      fail(id, `${layer}: bad domain`)
    }
    if (!Array.isArray(o.stops) || o.stops.length < 2 || !o.stops.every(isStr)) fail(id, `${layer}: bad stops`)
    if (!Array.isArray(o.ticks) || !o.ticks.every(isNum)) fail(id, `${layer}: bad ticks`)
    return
  }
  if (!linearOnly && o?.type === 'classes') {
    const c = o.classes as { value: unknown; color: unknown }[]
    if (!Array.isArray(c) || c.length === 0 || !c.every((k) => isNum(k.value) && isStr(k.color))) {
      fail(id, `${layer}: bad classes`)
    }
    return
  }
  fail(id, `${layer}: unknown colour scale`)
}

/** Check the shape of a parsed manifest and return it typed; throws with a short reason otherwise. */
export function parseSiteManifest(json: unknown, expectedId?: string): SiteManifest {
  const o = (json ?? {}) as Record<string, unknown>
  const id = isStr(o.id) ? o.id : '?'
  if (o.schema !== SITE_SCHEMA) fail(id, `schema is ${JSON.stringify(o.schema)}, expected "${SITE_SCHEMA}"`)
  if (!isSiteId(o.id)) fail(id, 'unknown site id')
  if (expectedId && o.id !== expectedId) fail(id, `names site ${o.id}, expected ${expectedId}`)
  for (const k of ['name', 'longName', 'firstImage', 'lastImage', 'years', 'attribution']) {
    if (!isStr(o[k])) fail(id, `${k} is not a string`)
  }
  if (!isNum(o.images)) fail(id, 'images is not a number')
  const h = o.heights as Record<string, unknown> | undefined
  if (!h || (h.datum === 'NMMA' ? !isNum(h.C) : !(h.datum === 'model' && h.C === null))) fail(id, 'bad heights')
  const g = o.grid as Record<string, unknown> | undefined
  if (!g || !['width', 'height', 'x0', 'y0', 'res'].every((k) => isNum(g[k])) || (g.res as number) <= 0) {
    fail(id, 'bad grid')
  }
  if (!Array.isArray(g.corners) || g.corners.length !== 4) fail(id, 'grid needs four corners')
  if (!Array.isArray(g.bounds) || g.bounds.length !== 4 || !g.bounds.every(isNum)) fail(id, 'bad grid bounds')
  if (!Array.isArray(o.layers) || o.layers.length === 0) fail(id, 'no layers')
  const seen = new Set<string>()
  for (const raw of o.layers as Record<string, unknown>[]) {
    const lid = isStr(raw.id) ? raw.id : '?'
    if (seen.has(lid)) fail(id, `layer ${lid} listed twice`)
    seen.add(lid)
    if (!isStr(raw.title) || !isStr(raw.units)) fail(id, `${lid}: title and units must be strings`)
    if (raw.kind === 'raster') {
      const v = raw.values as Record<string, unknown> | undefined
      if (!v || !isStr(v.file) || !isNum(v.scale) || !isNum(v.offset)) fail(id, `${lid}: bad raster files`)
      checkOverlay(id, lid, raw.overlay)
      if (!Array.isArray(v.check)) fail(id, `${lid}: no check values`)
      checkScale(id, lid, raw.scale)
    } else if (raw.kind === 'contours') {
      const lv = raw.levels as Record<string, unknown>[]
      if (!isStr(raw.file) || !Array.isArray(lv) || lv.length === 0) fail(id, `${lid}: bad contour layer`)
      for (let i = 0; i < lv.length; i++) {
        if (!isStr(lv[i].id) || !isNum(lv[i].level)) fail(id, `${lid}: bad level`)
        if (i > 0 && (lv[i].level as number) <= (lv[i - 1].level as number)) fail(id, `${lid}: levels must increase`)
      }
    } else if (raw.kind === 'points') {
      if (!isStr(raw.file) || !isStr(raw.colorBy) || !Array.isArray(raw.fields)) fail(id, `${lid}: bad points layer`)
      checkScale(id, lid, raw.scale, true)
    } else {
      fail(id, `${lid}: unknown kind ${JSON.stringify(raw.kind)}`)
    }
  }
  for (const raw of o.layers as Record<string, unknown>[]) {
    if (raw.kind === 'points' && isStr(raw.under) && !(o.layers as { id: string; kind: string }[]).some((l) => l.id === raw.under && l.kind === 'raster')) {
      fail(id, `${String(raw.id)}: "under" names no raster layer`)
    }
  }
  return o as unknown as SiteManifest
}

/** The site's raster layers by id. */
export function rasterById(site: SiteManifest, id: string | undefined): RasterLayer | undefined {
  const l = site.layers.find((x) => x.id === id)
  return l && l.kind === 'raster' ? l : undefined
}

/** The layer that gives heights to the readout: the first raster layer flagged as heights. */
export function heightLayer(site: SiteManifest): RasterLayer | undefined {
  return site.layers.find((l): l is RasterLayer => l.kind === 'raster' && l.heights === true)
}
