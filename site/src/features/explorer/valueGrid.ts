/**
 * Value images: the numbers behind a raster layer, for the pixel readout and the 3D terrain.
 *
 * Each is an 8-bit RGB PNG on the site's grid with code = 256 R + G (B = 0), value = offset + scale × code, and code
 * 0 for no data. The PNGs carry no colour profile, so a browser decodes them exactly when it is asked not to convert
 * colours. Only the codes are kept (2 bytes a pixel); the RGBA buffer of the decode is dropped at once.
 */
import type { SiteGrid, ValueFile } from './types'

export interface ValueGrid {
  width: number
  height: number
  codes: Uint16Array
  scale: number
  offset: number
}

/** The uint16 codes of RGBA pixels (code = 256 R + G). */
export function codesFromRgba(rgba: ArrayLike<number>, width: number, height: number): Uint16Array {
  const n = width * height
  if (rgba.length < 4 * n) throw new Error(`value image: ${rgba.length} bytes for ${width} × ${height} pixels`)
  const codes = new Uint16Array(n)
  for (let i = 0, k = 0; i < n; i++, k += 4) codes[i] = (rgba[k] << 8) | rgba[k + 1]
  return codes
}

/** The value of pixel (col, row), or null for no data or outside the grid. */
export function valueAt(grid: ValueGrid, col: number, row: number): number | null {
  if (col < 0 || row < 0 || col >= grid.width || row >= grid.height) return null
  const code = grid.codes[row * grid.width + col]
  return code === 0 ? null : grid.offset + grid.scale * code
}

/** The [col, row, code] probes that do not match the decoded codes (empty when the decode is right). */
export function failedChecks(grid: ValueGrid, check: readonly (readonly [number, number, number])[]): number[][] {
  return check.filter(([c, r, code]) => grid.codes[r * grid.width + c] !== code).map((x) => [...x])
}

/** A canvas to read pixels from: OffscreenCanvas where there is one. */
function makeCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height)
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  return c
}

/** Fetch and decode a value image; throws if it fails, has another size or does not pass its checks. */
export async function loadValueGrid(url: string, file: ValueFile, grid: SiteGrid, signal?: AbortSignal): Promise<ValueGrid> {
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`value image: HTTP ${res.status} for ${url}`)
  const blob = await res.blob()
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  } catch {
    bitmap = await createImageBitmap(blob)
  }
  const { width, height } = grid
  if (bitmap.width !== width || bitmap.height !== height) {
    bitmap.close()
    throw new Error(`value image: ${bitmap.width} × ${bitmap.height}, expected ${width} × ${height}`)
  }
  const canvas = makeCanvas(width, height)
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null
  if (!ctx) throw new Error('value image: no 2D canvas')
  ctx.drawImage(bitmap, 0, 0)
  bitmap.close()
  const codes = codesFromRgba(ctx.getImageData(0, 0, width, height).data, width, height)
  canvas.width = 0 // release the pixels now (Safari keeps canvas memory until then)
  canvas.height = 0
  const out: ValueGrid = { width, height, codes, scale: file.scale, offset: file.offset }
  const bad = failedChecks(out, file.check)
  if (bad.length) throw new Error(`value image: ${bad.length} of ${file.check.length} check pixels differ in ${url}`)
  return out
}
