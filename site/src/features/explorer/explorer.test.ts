// Unit tests of the map explorer's pure parts: the URL hash, Web Mercator pixels, value-image decoding, legend
// ticks in NMMA, the tide-level zone, readout lines, the 3D terrain encoding, and the generated data files
// (public/data/sites, written by scripts/data_layers.py) against both data contracts, the colour scales of
// ramps.json and the overlay tiling.
import { describe, expect, it } from 'vitest'
import { parseSitesIndex, SITE_IDS } from '@/lib/sites'
import indexJson from '../../../public/data/sites/index.json'
import fozJson from '../../../public/data/sites/foz/manifest.json'
import eoJson from '../../../public/data/sites/eo/manifest.json'
import villaJson from '../../../public/data/sites/villaviciosa/manifest.json'
import santanderJson from '../../../public/data/sites/santander/manifest.json'
import santonaJson from '../../../public/data/sites/santona/manifest.json'
import mundakaJson from '../../../public/data/sites/mundaka/manifest.json'
import { lonLatToMercator, mercatorToLonLat, pixelOf, tileExtent, tileTouchesGrid } from './geo'
import { DEFAULT_LAYER, formatHash, normaliseState, parseHash } from './hashState'
import { contourDash } from './lineStyles'
import ramps from './ramps.json'
import {
  classTicks,
  datumZone,
  formatHeight,
  heightNote,
  linearTicks,
  niceStep,
  NO_VALUE,
  readoutLine,
  shadeDirection,
} from './readout'
import { compositeTile, decodeTerrarium, encodeTerrarium } from './terrarium'
import { heightLayer, parseSiteManifest, type ContourLayer, type RasterLayer, type SiteManifest } from './types'
import { codesFromRgba, failedChecks, valueAt, type ValueGrid } from './valueGrid'

const MANIFESTS: Record<string, unknown> = {
  foz: fozJson,
  eo: eoJson,
  villaviciosa: villaJson,
  santander: santanderJson,
  santona: santonaJson,
  mundaka: mundakaJson,
}
const villa = parseSiteManifest(villaJson, 'villaviciosa')
const santander = parseSiteManifest(santanderJson, 'santander')
const layerOf = (s: SiteManifest, id: string) => s.layers.find((l) => l.id === id)!
/** Formatted text with its no-break spaces (between numbers and units) written as plain spaces. */
const plain = (text: string) => text.replace(/ /g, ' ')

describe('URL hash state', () => {
  it('reads a full hash', () => {
    expect(parseHash('#site=eo&layer=delay&base=map&3d=1', 'villaviciosa')).toEqual({
      site: 'eo',
      layer: 'delay',
      base: 'map',
      threeD: true,
    })
  })

  it('falls back to the defaults for missing or unknown values', () => {
    expect(parseHash('', 'villaviciosa')).toEqual({ site: 'villaviciosa', layer: DEFAULT_LAYER, base: 'photo', threeD: false })
    expect(parseHash('#site=atlantis&layer=../x&base=globe&3d=yes', 'foz')).toEqual({
      site: 'foz',
      layer: DEFAULT_LAYER,
      base: 'photo',
      threeD: false,
    })
  })

  it('takes a link that names the site alone (#site=<id>)', () => {
    expect(parseHash('#site=mundaka', 'villaviciosa').site).toBe('mundaka')
  })

  it('writes the hash in one order and reads it back', () => {
    const s = { site: 'santona' as const, layer: 'tide_lines', base: 'map' as const, threeD: false }
    expect(formatHash(s)).toBe('site=santona&layer=tide_lines&base=map&3d=0')
    expect(parseHash(`#${formatHash(s)}`, 'foz')).toEqual(s)
  })

  it('replaces a layer the site lacks and turns 3D off where it is not offered', () => {
    const s = parseHash('#site=eo&layer=other&3d=1', 'foz')
    expect(normaliseState(s, ['elevation', 'sigma'], true)).toMatchObject({ layer: 'elevation', threeD: true })
    expect(normaliseState(s, ['sigma'], false)).toMatchObject({ layer: 'sigma', threeD: false })
    expect(normaliseState(s, null, true).layer).toBe('other') // manifest not loaded yet: keep what was asked
  })
})

describe('Web Mercator and the site grid', () => {
  it('projects and inverts', () => {
    const [x, y] = lonLatToMercator(-5.39555, 43.51387)
    expect(x).toBeCloseTo(-600629.8785, 3)
    expect(y).toBeCloseTo(5390518.0753, 3)
    const [lon, lat] = mercatorToLonLat(x, y)
    expect(lon).toBeCloseTo(-5.39555, 9)
    expect(lat).toBeCloseTo(43.51387, 9)
  })

  it('finds the pixel of a point (same cell as a NumPy decode of the value image)', () => {
    expect(pixelOf(villa.grid, -5.39555, 43.51387)).toEqual({ col: 1057, row: 857 })
    expect(pixelOf(villa.grid, -5.395546248, 43.513906728)).toEqual({ col: 1057, row: 856 })
  })

  it('gives the corner pixels and nothing outside', () => {
    const [tlLon, tlLat] = villa.grid.corners[0]
    const [brLon, brLat] = villa.grid.corners[2]
    expect(pixelOf(villa.grid, tlLon + 1e-6, tlLat - 1e-6)).toEqual({ col: 0, row: 0 })
    expect(pixelOf(villa.grid, brLon - 1e-6, brLat + 1e-6)).toEqual({ col: villa.grid.width - 1, row: villa.grid.height - 1 })
    expect(pixelOf(villa.grid, tlLon - 1e-4, tlLat)).toBeNull()
    expect(pixelOf(villa.grid, brLon, brLat - 1e-4)).toBeNull()
  })

  it('places map tiles', () => {
    const t = tileExtent(0, 0, 0)
    expect(t.left).toBeCloseTo(-20037508.34, 1)
    expect(t.top).toBeCloseTo(20037508.34, 1)
    expect(tileTouchesGrid(villa.grid, 14, 7946, 5988)).toBe(true)
    expect(tileTouchesGrid(villa.grid, 14, 7000, 5988)).toBe(false)
  })
})

describe('value images', () => {
  const rgba = new Uint8ClampedArray([
    // 3 x 2 pixels: codes 0, 1, 256 / 65535, 31950, 0
    0, 0, 0, 255, 0, 1, 0, 255, 1, 0, 0, 255,
    255, 255, 0, 255, 124, 206, 0, 255, 0, 0, 0, 255,
  ])
  const grid: ValueGrid = { width: 3, height: 2, codes: codesFromRgba(rgba, 3, 2), scale: 1e-4, offset: -3.0001 }

  it('decodes code = 256 R + G', () => {
    expect([...grid.codes]).toEqual([0, 1, 256, 65535, 31950, 0])
  })

  it('reads values, with code 0 and outside the grid as no data', () => {
    expect(valueAt(grid, 0, 0)).toBeNull()
    expect(valueAt(grid, 1, 0)).toBeCloseTo(-3.0, 9)
    expect(valueAt(grid, 1, 1)).toBeCloseTo(0.1949, 9)
    expect(valueAt(grid, 3, 0)).toBeNull()
    expect(valueAt(grid, 0, -1)).toBeNull()
  })

  it('reports check pixels that differ', () => {
    expect(failedChecks(grid, [[1, 1, 31950], [0, 0, 0]])).toEqual([])
    expect(failedChecks(grid, [[1, 1, 31951]])).toEqual([[1, 1, 31951]])
  })

  it('refuses a short buffer', () => {
    expect(() => codesFromRgba(new Uint8Array(8), 3, 2)).toThrow()
  })
})

describe('legend ticks and heights', () => {
  const elevation = layerOf(villa, 'elevation') as RasterLayer

  it('puts round NMMA heights at (t − C − d0) / (d1 − d0)', () => {
    if (elevation.scale.type !== 'linear') throw new Error('elevation is linear')
    const C = 0.3778
    const ticks = linearTicks(elevation.scale, C, true)
    expect(ticks.map((t) => t.label)).toEqual(['−1.5', '−1', '−0.5', '0', '0.5', '1', '1.5'])
    const [d0, d1] = elevation.scale.domain
    ticks.forEach((t, i) => expect(t.at).toBeCloseTo((-1.5 + 0.5 * i - C - d0) / (d1 - d0), 12))
    expect(ticks[0].at).toBeGreaterThan(0)
    expect(ticks[ticks.length - 1].at).toBeLessThan(1)
  })

  it('uses the same rule without C (heights above the model mean)', () => {
    if (elevation.scale.type !== 'linear') throw new Error('elevation is linear')
    const ticks = linearTicks(elevation.scale, 0, true)
    expect(ticks.map((t) => t.label)).toEqual(['−2', '−1.5', '−1', '−0.5', '0', '0.5', '1'])
    expect(ticks[4].at).toBeCloseTo(2.2 / 3.4, 12)
  })

  it('keeps a layer’s own ticks when it is not a height', () => {
    const delay = layerOf(villa, 'delay') as RasterLayer
    if (delay.scale.type !== 'linear') throw new Error('delay is linear')
    expect(linearTicks(delay.scale).map((t) => t.label)).toEqual(['0', '5', '10', '15', '20'])
  })

  it('labels the first, middle and last uncertainty classes', () => {
    const sigma = layerOf(villa, 'sigma') as RasterLayer
    if (sigma.scale.type !== 'classes') throw new Error('sigma is classed')
    const ticks = classTicks(sigma.scale)
    expect(ticks.map((t) => t.label)).toEqual(['0.03', '0.15', '0.65'])
    expect(ticks.map((t) => t.at)).toEqual([0.5 / 8, 3.5 / 8, 7.5 / 8])
  })

  it('picks round steps', () => {
    expect(niceStep(3.4)).toBe(0.5)
    expect(niceStep(90)).toBe(15)
    expect(niceStep(10)).toBe(2)
  })

  it('says which way a scale runs (dark blue low ground to light yellow high ground; the others light to dark)', () => {
    expect(shadeDirection(ramps.elevation.stops)).toBe('dark to light')
    expect(shadeDirection(ramps.sigma.classes)).toBe('light to dark')
    expect(shadeDirection(ramps.emersion.stops)).toBe('light to dark')
    expect(shadeDirection(ramps.delay.stops)).toBe('light to dark')
  })

  it('writes heights in NMMA where the site has C, else above the model mean', () => {
    expect(plain(formatHeight(-0.276, villa.heights))).toBe('+0.10 m NMMA')
    expect(plain(formatHeight(-0.5, santander.heights))).toBe('−0.50 m')
    expect(plain(heightNote(villa.heights))).toContain('tide-model level + C, with C = 0.38 m')
    expect(heightNote(villa.heights)).toContain('Ferrol')
    expect(heightNote(santander.heights)).toContain('no NMMA link')
  })
})

describe('tide levels', () => {
  const lines = layerOf(villa, 'tide_lines') as ContourLayer

  it('runs from LAT to HAT with NMMA = level + C', () => {
    expect(lines.levels.map((l) => l.id)).toEqual(['LAT', 'MLWS', 'MLWN', 'MSL', 'MHWN', 'MHWS', 'HAT'])
    for (const l of lines.levels) expect(l.nmma! - l.level).toBeCloseTo(0.3778, 3)
    expect(lines.levels.filter((l) => l.drawn).map((l) => l.id)).toEqual(['MLWS', 'MLWN', 'MSL', 'MHWN'])
  })

  it('names the zone of a height', () => {
    expect(datumZone(lines.levels, -0.276)).toBe('between MLWN and MSL')
    expect(datumZone(lines.levels, 0)).toBe('between MSL and MHWN')
    expect(datumZone(lines.levels, -3)).toBe('below LAT')
    expect(datumZone(lines.levels, 3)).toBe('above HAT')
  })

  it('draws mean sea level solid, neaps dashed, springs dotted', () => {
    expect(contourDash('MSL')).toBeNull()
    expect(contourDash('MLWN')).toEqual([3, 2])
    expect(contourDash('MHWS')).toEqual([1, 1.5])
  })
})

describe('readout lines', () => {
  const heights = villa.heights

  it('gives the height on the elevation layer', () => {
    expect(plain(readoutLine(layerOf(villa, 'elevation'), heights, { height: 0.2657, value: 0.2657 }))).toBe('+0.64 m NMMA')
  })

  it('gives the layer value and the height on other layers', () => {
    expect(plain(readoutLine(layerOf(villa, 'sigma'), heights, { height: 0.2657, value: 0.65 }))).toBe('0.65 m · elevation +0.64 m NMMA')
    expect(plain(readoutLine(layerOf(villa, 'emersion'), heights, { height: 0.2657, value: 57.08 }))).toBe('57 % · elevation +0.64 m NMMA')
    expect(plain(readoutLine(layerOf(villa, 'delay'), heights, { height: null, value: 7 }))).toBe('7 min')
  })

  it('places the height among the tidal levels on the tide-line layer', () => {
    expect(plain(readoutLine(layerOf(villa, 'tide_lines'), heights, { height: -0.276, value: null }))).toBe(
      '+0.10 m NMMA · between MLWN and MSL',
    )
  })

  it('writes the sign of a signed layer (generic flag of the manifest)', () => {
    const sigma = layerOf(villa, 'sigma') as RasterLayer
    const diff: RasterLayer = { ...sigma, id: 'diff', title: 'Difference', signed: true }
    expect(plain(readoutLine(diff, heights, { height: 0.2657, value: 0.12 }))).toBe('+0.12 m · elevation +0.64 m NMMA')
    expect(plain(readoutLine(diff, heights, { height: null, value: -0.3 }))).toBe('−0.30 m')
  })

  it('says so where the map has no value', () => {
    expect(plain(readoutLine(layerOf(villa, 'elevation'), heights, { height: null, value: null }))).toBe(NO_VALUE)
    expect(plain(readoutLine(layerOf(villa, 'sigma'), heights, { height: 0.1, value: null }))).toBe(NO_VALUE)
  })

  it('reads the fields of a point on a points layer (generic layer kind)', () => {
    const points = {
      id: 'pts',
      kind: 'points' as const,
      title: 'Points',
      about: '',
      units: 'm NMMA',
      file: 'p.geojson',
      colorBy: 'a',
      scale: { type: 'linear' as const, domain: [-2, 2] as [number, number], ticks: [-2, 0, 2], stops: ['#000000', '#FFFFFF'] },
      decimals: 2,
      fields: [
        { property: 'a', label: 'One' },
        { property: 'b', label: 'Two' },
      ],
    }
    expect(plain(readoutLine(points, heights, { height: 0, value: null, point: { a: 0.52, b: -0.1 } }))).toBe(
      'One +0.52 m NMMA · Two −0.10 m NMMA',
    )
    expect(plain(readoutLine(points, heights, { height: -0.276, value: null, point: null }))).toBe('+0.10 m NMMA')
  })
})

describe('3D terrain (Terrarium)', () => {
  it('encodes and decodes heights', () => {
    for (const h of [-2.2, -0.276, 0, 0.64, 12.5, 1234.567]) {
      const [r, g, b] = encodeTerrarium(h)
      expect(decodeTerrarium(r, g, b)).toBeCloseTo(h, 2)
      for (const v of [r, g, b]) expect(Number.isInteger(v) && v >= 0 && v <= 255).toBe(true)
    }
  })

  it('writes MAREA heights (+C) into the tile pixels over the grid only', () => {
    const grid = { width: 2, height: 2, x0: -1000, y0: 1000, res: 1000, corners: [], bounds: [0, 0, 0, 0], centre: [0, 0] } as never
    const values: ValueGrid = { width: 2, height: 2, codes: new Uint16Array([30001, 0, 30001, 30001]), scale: 1e-4, offset: -3.0001 }
    // Tile 0/0/0 covers the world; a 4096-px tile has ~9.8 km pixels, so test a fine tile over the grid instead.
    const z = 15
    const size = 256
    const rgba = new Uint8ClampedArray(size * size * 4)
    const n = compositeTile(rgba, size, z, 2 ** 14 - 1, 2 ** 14 - 1, { grid, values, add: 0.5 })
    expect(n).toBeGreaterThan(0)
    const k = rgba.findIndex((v, i) => i % 4 === 3 && v === 255) - 3
    expect(decodeTerrarium(rgba[k], rgba[k + 1], rgba[k + 2])).toBeCloseTo(0.5, 2)
    const far = new Uint8ClampedArray(size * size * 4)
    expect(compositeTile(far, size, z, 0, 0, { grid, values, add: 0 })).toBe(0)
    expect(far.every((v) => v === 0)).toBe(true)
  })
})

describe('generated data (scripts/data_layers.py)', () => {
  it('writes a site list that passes the shell’s contract, west to east', () => {
    const index = parseSitesIndex(indexJson)
    expect(index.sites.map((s) => s.id)).toEqual([...SITE_IDS])
    for (const s of index.sites) {
      expect(s.layers).toEqual(['elevation', 'sigma', 'emersion', 'delay', 'tide_lines'])
      expect(s.thumb).toBe(`sites/${s.id}/thumb.webp`)
    }
  })

  it('writes six manifests that pass the explorer’s contract', () => {
    for (const id of SITE_IDS) {
      const site = parseSiteManifest(MANIFESTS[id], id)
      expect(heightLayer(site)?.id).toBe('elevation')
      expect(site.grid.corners).toHaveLength(4)
      for (const l of site.layers) {
        if (l.kind === 'raster') expect(l.values.check.length).toBeGreaterThanOrEqual(5)
      }
    }
  })

  it('gives NMMA heights at four sites and model heights at Santander and Santoña', () => {
    const C = Object.fromEntries(SITE_IDS.map((id) => [id, parseSiteManifest(MANIFESTS[id], id).heights.C]))
    expect(C).toEqual({ foz: 0.3151, eo: 0.324, villaviciosa: 0.3778, santander: null, santona: null, mundaka: 0.3742 })
  })

  it('draws lines of equal delay at Mundaka only, and notes where there is no delay', () => {
    for (const id of SITE_IDS) {
      const delay = layerOf(parseSiteManifest(MANIFESTS[id], id), 'delay') as RasterLayer
      expect(Boolean(delay.lines)).toBe(id === 'mundaka')
      expect(Boolean(delay.note)).toBe(['eo', 'santona', 'santander'].includes(id))
    }
  })

  it('colours every map and legend with the scales of ramps.json', () => {
    for (const id of SITE_IDS) {
      for (const l of parseSiteManifest(MANIFESTS[id], id).layers) {
        if (l.kind !== 'raster') continue
        const ramp = ramps[l.id as keyof typeof ramps] as { stops?: string[]; classes?: string[] }
        if (l.scale.type === 'linear') expect(l.scale.stops).toEqual(ramp.stops)
        else expect(l.scale.classes.map((c) => c.color)).toEqual(ramp.classes)
      }
    }
  })

  it('draws each overlay at twice the grid, in tiles of at most 4096 px that cover the grid edge to edge', () => {
    for (const id of SITE_IDS) {
      const site = parseSiteManifest(MANIFESTS[id], id)
      for (const l of site.layers) {
        if (l.kind !== 'raster') continue
        const o = l.overlay
        expect([o.width, o.height]).toEqual([2 * site.grid.width, 2 * site.grid.height])
        // Tiles come row by row: the first row's widths add up to the overlay's width.
        let nx = 0
        for (let w = 0; w < o.width; nx++) w += o.tiles[nx].width
        const ny = o.tiles.length / nx
        expect(Number.isInteger(ny)).toBe(true)
        let height = 0
        o.tiles.forEach((t, k) => {
          expect(Math.max(t.width, t.height)).toBeLessThanOrEqual(4096)
          expect(t.width).toBe(o.tiles[k % nx].width) // columns line up
          expect(t.height).toBe(o.tiles[k - (k % nx)].height) // a row has one height
          if (k % nx === 0) height += t.height
          const right = (k % nx) + 1 < nx ? o.tiles[k + 1] : null
          const below = k + nx < o.tiles.length ? o.tiles[k + nx] : null
          if (right) expect([right.corners[0], right.corners[3]]).toEqual([t.corners[1], t.corners[2]])
          if (below) expect([below.corners[0], below.corners[1]]).toEqual([t.corners[3], t.corners[2]])
        })
        expect(height).toBe(o.height)
        expect(o.tiles[0].corners[0]).toEqual(site.grid.corners[0])
        expect(o.tiles[o.tiles.length - 1].corners[2]).toEqual(site.grid.corners[2])
      }
    }
  })
})
