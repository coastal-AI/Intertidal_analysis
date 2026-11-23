/**
 * The 3D view's terrain: open elevation tiles in Terrarium encoding (height + 32768 = 256 R + G + B / 256), with
 * MAREA's own heights written into the pixels where the site's elevation map has a value.
 */
import { tileExtent } from './geo'
import type { SiteGrid } from './types'
import type { ValueGrid } from './valueGrid'

/** Terrarium bytes [R, G, B] of a height in metres. */
export function encodeTerrarium(height: number): [number, number, number] {
  const v = Math.min(65535.996, Math.max(0, height + 32768))
  const r = Math.floor(v / 256)
  const g = Math.floor(v) - r * 256
  const b = Math.floor((v - Math.floor(v)) * 256)
  return [r, g, b]
}

/** Height in metres of Terrarium bytes. */
export function decodeTerrarium(r: number, g: number, b: number): number {
  return r * 256 + g + b / 256 - 32768
}

/** What the terrain writes into its tiles: the site's grid, its elevation values and the offset to add (C). */
export interface TerrainHeights {
  grid: SiteGrid
  values: ValueGrid
  /** Added to the map values, so the heights match the land terrain (NMMA). */
  add: number
}

/**
 * Write MAREA's heights into the RGBA pixels of tile z/x/y (``size`` pixels a side) in place; returns how many
 * pixels changed. Pixels are sampled at their centres; pixels outside the grid or without a value keep the tile's.
 */
export function compositeTile(
  rgba: Uint8ClampedArray | Uint8Array,
  size: number,
  z: number,
  x: number,
  y: number,
  t: TerrainHeights,
): number {
  const { grid, values, add } = t
  const ext = tileExtent(z, x, y)
  const step = ext.size / size
  const right = grid.x0 + grid.width * grid.res
  const bottom = grid.y0 - grid.height * grid.res
  const i0 = Math.max(0, Math.floor((grid.x0 - ext.left) / step))
  const i1 = Math.min(size, Math.ceil((right - ext.left) / step))
  const j0 = Math.max(0, Math.floor((ext.top - grid.y0) / step))
  const j1 = Math.min(size, Math.ceil((ext.top - bottom) / step))
  let n = 0
  for (let j = j0; j < j1; j++) {
    const row = Math.floor((grid.y0 - (ext.top - (j + 0.5) * step)) / grid.res)
    if (row < 0 || row >= grid.height) continue
    const rowOffset = row * grid.width
    for (let i = i0; i < i1; i++) {
      const col = Math.floor((ext.left + (i + 0.5) * step - grid.x0) / grid.res)
      if (col < 0 || col >= grid.width) continue
      const code = values.codes[rowOffset + col]
      if (code === 0) continue
      const [r, g, b] = encodeTerrarium(values.offset + values.scale * code + add)
      const k = 4 * (j * size + i)
      rgba[k] = r
      rgba[k + 1] = g
      rgba[k + 2] = b
      rgba[k + 3] = 255
      n++
    }
  }
  return n
}
