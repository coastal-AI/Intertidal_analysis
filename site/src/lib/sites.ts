/**
 * The site list: data/sites/index.json, schema "cg26-sites/1".
 *
 * The data script (scripts/data_layers.py) writes the file; the map viewer reads it with loadSitesIndex() or
 * useSitesIndex(). Every path inside it is relative to the data folder (BASE_URL + "data/"): use dataUrl(site.thumb).
 */
import { dataUrl, loadJSON } from './assets'
import { useResource, type Resource } from './useResource'

export const SITES_SCHEMA = 'cg26-sites/1'
/** Where the site list lives, relative to the data folder. */
export const SITES_INDEX_PATH = 'sites/index.json'

/** The six sites, west to east. */
export const SITE_IDS = ['foz', 'eo', 'villaviciosa', 'santander', 'santona', 'mundaka'] as const
export type SiteId = (typeof SITE_IDS)[number]

export function isSiteId(value: unknown): value is SiteId {
  return typeof value === 'string' && (SITE_IDS as readonly string[]).includes(value)
}

/** Heights above NMMA (mean sea level at Alicante): NMMA height = tide-model level + C. */
export interface SiteHeightsNMMA {
  datum: 'NMMA'
  /** Constant added to the tide-model level, in metres. */
  C: number
  /** Uncertainty of C, in metres. */
  sigma: number
  /** Short source text, e.g. the tide gauge and the number of images. */
  source: string
}

/** Heights above the tide model's mean sea level (no link to NMMA at this site). */
export interface SiteHeightsModel {
  datum: 'model'
  C: null
  /** One-line note shown next to the heights. */
  note: string
}

export type SiteHeights = SiteHeightsNMMA | SiteHeightsModel

export interface Site {
  id: SiteId
  name: string
  longName: string
  region: string
  /** First and last Sentinel-2 image of the product (YYYY-MM-DD). */
  firstImage: string
  lastImage: string
  /** Years of the product, e.g. "2023–2025". */
  years: string
  /** Number of Sentinel-2 images in the product. */
  images: number
  /** [lon, lat] of the map's centre. */
  centre: [number, number]
  /** [west, south, east, north] in degrees. */
  bounds: [number, number, number, number]
  heights: SiteHeights
  /** Thumbnail, relative to the data folder ("sites/<id>/thumb.webp"). */
  thumb: string
  /** The site's layer manifest, relative to the data folder ("sites/<id>/manifest.json"). */
  manifest: string
  /** Layer ids the map offers at this site. */
  layers: string[]
}

export interface SitesIndex {
  schema: typeof SITES_SCHEMA
  licence: string
  attribution: string
  /** West to east. */
  sites: Site[]
}

function fail(reason: string): never {
  throw new Error(`data/${SITES_INDEX_PATH}: ${reason}`)
}

function isNumberArray(v: unknown, n: number): boolean {
  return Array.isArray(v) && v.length === n && v.every((x) => typeof x === 'number' && Number.isFinite(x))
}

/** Check the shape of a parsed index.json and return it typed; throws with a short reason otherwise. */
export function parseSitesIndex(json: unknown): SitesIndex {
  if (typeof json !== 'object' || json === null) fail('not an object')
  const o = json as Record<string, unknown>
  if (o.schema !== SITES_SCHEMA) fail(`schema is ${JSON.stringify(o.schema)}, expected "${SITES_SCHEMA}"`)
  if (!Array.isArray(o.sites) || o.sites.length === 0) fail('no sites')
  for (const s of o.sites as Record<string, unknown>[]) {
    if (!isSiteId(s.id)) fail(`unknown site id ${JSON.stringify(s.id)}`)
    for (const k of ['name', 'longName', 'region', 'firstImage', 'lastImage', 'years', 'thumb', 'manifest']) {
      if (typeof s[k] !== 'string') fail(`${s.id}: ${k} is not a string`)
    }
    if (typeof s.images !== 'number') fail(`${s.id}: images is not a number`)
    if (!isNumberArray(s.centre, 2)) fail(`${s.id}: centre is not [lon, lat]`)
    if (!isNumberArray(s.bounds, 4)) fail(`${s.id}: bounds is not [w, s, e, n]`)
    const h = s.heights as Record<string, unknown> | undefined
    if (!h || (h.datum === 'NMMA' ? typeof h.C !== 'number' : h.datum !== 'model')) fail(`${s.id}: bad heights`)
    if (!Array.isArray(s.layers)) fail(`${s.id}: layers is not a list`)
  }
  return o as unknown as SitesIndex
}

/** Load data/sites/index.json once (cached) and check it. */
export function loadSitesIndex(): Promise<SitesIndex> {
  return loadJSON<unknown>(dataUrl(SITES_INDEX_PATH)).then(parseSitesIndex)
}

/** React hook: the site list, with loading and error states. */
export function useSitesIndex(): Resource<SitesIndex> {
  return useResource(SITES_INDEX_PATH, loadSitesIndex)
}
