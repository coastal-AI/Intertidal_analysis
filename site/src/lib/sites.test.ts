import { describe, expect, it } from 'vitest'
import indexJson from '../../public/data/sites/index.json'
import { SITE_IDS, isSiteId, parseSitesIndex } from './sites'

describe('the site list contract (cg26-sites/1)', () => {
  const index = parseSitesIndex(indexJson)

  it('accepts the generated site list', () => {
    expect(index.schema).toBe('cg26-sites/1')
    expect(index.sites).toHaveLength(4)
  })

  it('lists the sites west to east', () => {
    expect(index.sites.map((s) => s.id)).toEqual([...SITE_IDS])
    const lons = index.sites.map((s) => s.centre[0])
    expect([...lons].sort((a, b) => a - b)).toEqual(lons)
  })

  it('gives NMMA heights with C, or model heights with a note', () => {
    for (const s of index.sites) {
      if (s.heights.datum === 'NMMA') {
        expect(s.heights.C).toBeGreaterThan(0.2)
        expect(s.heights.C).toBeLessThan(0.5)
        expect(s.heights.source.length).toBeGreaterThan(0)
      } else {
        expect(s.heights.C).toBeNull()
        expect(s.heights.note.length).toBeGreaterThan(0)
      }
    }
    expect(index.sites.filter((s) => s.heights.datum === 'model').map((s) => s.id)).toEqual([])
  })

  it('keeps every path relative to the data folder', () => {
    for (const s of index.sites) {
      for (const p of [s.thumb, s.manifest]) {
        expect(p.startsWith('/')).toBe(false)
        expect(p).not.toMatch(/^[a-z]+:/i)
      }
    }
  })

  it('rejects another schema or an unknown site', () => {
    expect(() => parseSitesIndex({ ...indexJson, schema: 'cg26-sites/0' })).toThrow(/schema/)
    const bad = { ...indexJson, sites: [{ ...indexJson.sites[0], id: 'nowhere' }] }
    expect(() => parseSitesIndex(bad)).toThrow(/unknown site id/)
    expect(() => parseSitesIndex(null)).toThrow()
  })

  it('recognises site ids', () => {
    expect(isSiteId('villaviciosa')).toBe(true)
    expect(isSiteId('urdaibai')).toBe(false)
    expect(isSiteId('santander')).toBe(false)
  })

  it('needs a preliminary flag on every site', () => {
    expect(index.sites.every((s) => s.preliminary === false)).toBe(true)
    const bad = { ...indexJson, sites: [{ ...indexJson.sites[0], preliminary: 'no' }] }
    expect(() => parseSitesIndex(bad)).toThrow(/preliminary/)
  })
})
