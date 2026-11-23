import { afterEach, describe, expect, it, vi } from 'vitest'
import { FetchError, assetUrl, clearJSONCache, dataUrl, loadJSON } from './assets'

describe('loadJSON', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    clearJSONCache()
  })

  it('fetches each file once, at its URL under the base path', async () => {
    const fetchMock = vi.fn(async (_url: RequestInfo | URL) => new Response('{"schema":"x"}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const url = dataUrl('sites/index.json')
    expect(await loadJSON(url)).toEqual({ schema: 'x' })
    expect(await loadJSON('data/sites/index.json')).toEqual({ schema: 'x' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe(url)
  })

  it('does not cache a failure', async () => {
    const fetchMock = vi.fn(async () => new Response('missing', { status: 404 }))
    vi.stubGlobal('fetch', fetchMock)
    await expect(loadJSON('data/sites/eo/manifest.json')).rejects.toBeInstanceOf(FetchError)
    await expect(loadJSON('data/sites/eo/manifest.json')).rejects.toMatchObject({ status: 404 })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})

describe('assetUrl', () => {
  it('puts public files under the base path', () => {
    expect(assetUrl('figures/elevation_map.webp', '/Intertidal_analysis/')).toBe('/Intertidal_analysis/figures/elevation_map.webp')
    expect(assetUrl('/figures/x.webp', '/Intertidal_analysis/')).toBe('/Intertidal_analysis/figures/x.webp')
  })
  it('works for a root base and a base without its trailing slash', () => {
    expect(assetUrl('favicon.svg', '/')).toBe('/favicon.svg')
    expect(assetUrl('favicon.svg', '/marea')).toBe('/marea/favicon.svg')
  })
  it('is idempotent: a URL already under the base is not prefixed again', () => {
    const once = assetUrl('data/sites/index.json', '/Intertidal_analysis/')
    expect(assetUrl(once, '/Intertidal_analysis/')).toBe(once)
    expect(assetUrl(dataUrl('sites/eo/manifest.json', '/x/'), '/x/')).toBe('/x/data/sites/eo/manifest.json')
    expect(assetUrl(assetUrl('favicon.svg', '/'), '/')).toBe('/favicon.svg')
  })
  it('leaves absolute URLs alone', () => {
    expect(assetUrl('https://www.ign.es/wmts/pnoa-ma', '/x/')).toBe('https://www.ign.es/wmts/pnoa-ma')
    expect(assetUrl('//cdn.example.org/a.js', '/x/')).toBe('//cdn.example.org/a.js')
    expect(assetUrl('data:image/png;base64,AAAA', '/x/')).toBe('data:image/png;base64,AAAA')
  })
})

describe('dataUrl', () => {
  it('points into the data folder', () => {
    expect(dataUrl('sites/index.json', '/Intertidal_analysis/')).toBe('/Intertidal_analysis/data/sites/index.json')
    expect(dataUrl('/sites/eo/manifest.json', '/')).toBe('/data/sites/eo/manifest.json')
  })
})
