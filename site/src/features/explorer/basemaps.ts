/**
 * Basemaps and terrain, loaded from their services at run time (nothing is copied into the repository).
 *
 *   photo  PNOA orthophoto, Instituto Geográfico Nacional (IGN), WMTS, CC BY 4.0 scne.es
 *   map    IGN base map in grey, WMTS, CC BY 4.0 scne.es
 *   3D     Mapterhorn terrain tiles (Terrarium encoding; in Spain from the IGN digital terrain models), no key
 */
import type { BaseMap } from './hashState'

const IGN_ATTRIBUTION = '© <a href="https://www.ign.es" target="_blank" rel="noopener">Instituto Geográfico Nacional</a>, CC BY 4.0 scne.es'

/** A WMTS GetTile URL of an IGN service in the Web Mercator tile grid. */
function ign(service: string, layer: string, format: string): string {
  return (
    `https://www.ign.es/wmts/${service}?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=${layer}&STYLE=default` +
    `&TILEMATRIXSET=GoogleMapsCompatible&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=${format}`
  )
}

export interface BaseMapDef {
  id: BaseMap
  /** Label of the toggle. */
  label: string
  tiles: string[]
  maxzoom: number
  attribution: string
}

export const BASEMAPS: readonly BaseMapDef[] = [
  {
    id: 'photo',
    label: 'Photo',
    tiles: [ign('pnoa-ma', 'OI.OrthoimageCoverage', 'image/jpeg')],
    maxzoom: 18,
    attribution: `Orthophoto PNOA ${IGN_ATTRIBUTION}`,
  },
  {
    id: 'map',
    label: 'Map',
    tiles: [ign('ign-base', 'IGNBase-gris', 'image/jpeg')],
    maxzoom: 18,
    attribution: `Base map ${IGN_ATTRIBUTION}`,
  },
]

/** Terrain tiles: 512 px, Terrarium encoding. */
export const TERRAIN = {
  tiles: 'https://tiles.mapterhorn.com/{z}/{x}/{y}.webp',
  tileSize: 512,
  maxzoom: 14,
  attribution:
    'Terrain <a href="https://mapterhorn.com/attribution" target="_blank" rel="noopener">© Mapterhorn</a> (IGN, CC BY 4.0 scne.es; Copernicus GLO-30) with MAREA heights',
} as const
