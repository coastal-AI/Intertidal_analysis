/**
 * Web Mercator (EPSG:3857) arithmetic: where a longitude and latitude fall on a site's grid, and where a map tile is.
 */
import type { SiteGrid } from './types'

/** Radius of the Web Mercator sphere (m). */
export const R_EARTH = 6378137

/** EPSG:3857 coordinates (m) of a longitude and latitude (degrees). */
export function lonLatToMercator(lon: number, lat: number): [number, number] {
  const x = (R_EARTH * lon * Math.PI) / 180
  const y = R_EARTH * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))
  return [x, y]
}

/** Longitude and latitude (degrees) of EPSG:3857 coordinates (m). */
export function mercatorToLonLat(x: number, y: number): [number, number] {
  const lon = ((x / R_EARTH) * 180) / Math.PI
  const lat = ((2 * Math.atan(Math.exp(y / R_EARTH)) - Math.PI / 2) * 180) / Math.PI
  return [lon, lat]
}

/** Column and row of the grid pixel that holds a point, or null outside the grid. */
export function pixelOf(grid: SiteGrid, lon: number, lat: number): { col: number; row: number } | null {
  const [x, y] = lonLatToMercator(lon, lat)
  const col = Math.floor((x - grid.x0) / grid.res)
  const row = Math.floor((grid.y0 - y) / grid.res)
  if (col < 0 || row < 0 || col >= grid.width || row >= grid.height) return null
  return { col, row }
}

/** EPSG:3857 extent of tile z/x/y: left, top and side (m). */
export function tileExtent(z: number, x: number, y: number): { left: number; top: number; size: number } {
  const world = 2 * Math.PI * R_EARTH
  const size = world / 2 ** z
  return { left: -world / 2 + x * size, top: world / 2 - y * size, size }
}

/** Whether tile z/x/y overlaps the grid. */
export function tileTouchesGrid(grid: SiteGrid, z: number, x: number, y: number): boolean {
  const t = tileExtent(z, x, y)
  const right = grid.x0 + grid.width * grid.res
  const bottom = grid.y0 - grid.height * grid.res
  return t.left < right && t.left + t.size > grid.x0 && t.top > bottom && t.top - t.size < grid.y0
}
