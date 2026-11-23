/**
 * The explorer's state in the URL hash, so every view can be shared: #site=<id>&layer=<id>&base=photo|map&3d=0|1.
 * Unknown or missing values fall back to the defaults; the layer is checked against the site's layers once its
 * manifest is loaded.
 */
import { isSiteId, type SiteId } from '@/lib/sites'

export type BaseMap = 'photo' | 'map'

export interface ExplorerState {
  site: SiteId
  layer: string
  base: BaseMap
  threeD: boolean
}

export const DEFAULT_LAYER = 'elevation'

/** Read the hash ("#site=eo&layer=delay" or "site=eo..."). Values that are not understood become the defaults. */
export function parseHash(hash: string, defaultSite: SiteId): ExplorerState {
  const p = new URLSearchParams(hash.replace(/^#/, ''))
  const site = p.get('site')
  const layer = p.get('layer') ?? ''
  const base = p.get('base')
  return {
    site: isSiteId(site) ? site : defaultSite,
    layer: /^[a-z][a-z0-9_]{0,31}$/.test(layer) ? layer : DEFAULT_LAYER,
    base: base === 'map' ? 'map' : 'photo',
    threeD: p.get('3d') === '1',
  }
}

/** The hash of a state, without the leading "#", always in the same order. */
export function formatHash(s: ExplorerState): string {
  return `site=${s.site}&layer=${s.layer}&base=${s.base}&3d=${s.threeD ? 1 : 0}`
}

/**
 * The state as shown: the layer replaced by the default when the site lacks it, and 3D off where it may not be
 * offered.
 */
export function normaliseState(s: ExplorerState, layerIds: readonly string[] | null, allow3d: boolean): ExplorerState {
  const layer = layerIds === null || layerIds.includes(s.layer) ? s.layer : layerIds.includes(DEFAULT_LAYER) ? DEFAULT_LAYER : layerIds[0]
  return { ...s, layer, threeD: allow3d && s.threeD }
}
