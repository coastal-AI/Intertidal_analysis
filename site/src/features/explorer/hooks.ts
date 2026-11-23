/**
 * React hooks of the explorer: the URL hash, media queries, the site's manifest and its value images.
 */
import { useCallback, useEffect, useState } from 'react'
import { dataUrl, loadJSON } from '@/lib/assets'
import type { SiteId } from '@/lib/sites'
import { useResource, type Resource } from '@/lib/useResource'
import { parseSiteManifest, type RasterLayer, type SiteManifest } from './types'
import { loadValueGrid, type ValueGrid } from './valueGrid'

/**
 * The URL hash ("#site=..", or "" when there is none), following the address bar, and a setter that replaces it
 * without adding a history entry (the explorer's state changes do not fill the Back button).
 */
export function useLocationHash(): [string, (hash: string) => void] {
  const [hash, setHash] = useState(() => window.location.hash)
  useEffect(() => {
    const on = () => setHash(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const replace = useCallback((next: string) => {
    const h = next === '' || next.startsWith('#') ? next : `#${next}`
    if (window.location.hash !== h) {
      const { pathname, search } = window.location
      window.history.replaceState(window.history.state, '', `${pathname}${search}${h}`)
    }
    setHash(h)
  }, [])
  return [hash, replace]
}

/** Whether a media query matches, following changes. */
export function useMediaQuery(query: string): boolean {
  const get = () => typeof window.matchMedia === 'function' && window.matchMedia(query).matches
  const [matches, setMatches] = useState(get)
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const mq = window.matchMedia(query)
    const on = () => setMatches(mq.matches)
    on()
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [query])
  return matches
}

/** Absolute URL of a file in a site's data folder (MapLibre fetches some files in a worker, so URLs are absolute). */
export function siteFileUrl(siteId: string, file: string): string {
  return new URL(dataUrl(`sites/${siteId}/${file}`), window.location.href).href
}

/** The site's manifest (cached per site). */
export function useSiteManifest(id: SiteId): Resource<SiteManifest> {
  return useResource(`site:${id}`, () => loadJSON<unknown>(dataUrl(`sites/${id}/manifest.json`)).then((j) => parseSiteManifest(j, id)))
}

// Value images are large (up to 18 MB of codes each), so only those in use stay in memory: each entry counts the
// components that hold it and is dropped, its download cancelled, when the last one lets it go.
const grids = new Map<string, { promise: Promise<ValueGrid>; holders: number; abort: AbortController }>()

function hold(url: string, layer: RasterLayer, site: SiteManifest): Promise<ValueGrid> {
  let entry = grids.get(url)
  if (!entry) {
    const abort = new AbortController()
    entry = { promise: loadValueGrid(url, layer.values, site.grid, abort.signal), holders: 0, abort }
    grids.set(url, entry)
  }
  entry.holders += 1
  return entry.promise
}

function letGo(url: string): void {
  const entry = grids.get(url)
  if (!entry) return
  entry.holders -= 1
  if (entry.holders <= 0) {
    entry.abort.abort()
    grids.delete(url)
  }
}

export interface GridState {
  grid: ValueGrid | null
  /** The image could not be read or did not pass its checks: the readout is off for this layer. */
  failed: boolean
}

/** The decoded value image of a raster layer of the site (null while it loads, or for no layer). */
export function useValueGrid(site: SiteManifest | null, layer: RasterLayer | null | undefined): GridState {
  const url = site && layer ? siteFileUrl(site.id, layer.values.file) : null
  const [state, setState] = useState<{ url: string | null } & GridState>({ url: null, grid: null, failed: false })
  useEffect(() => {
    if (!url || !site || !layer) return
    let alive = true
    hold(url, layer, site).then(
      (grid) => {
        if (alive) setState({ url, grid, failed: false })
      },
      (err: unknown) => {
        if (!alive) return
        if (!(err instanceof DOMException && err.name === 'AbortError')) console.warn('readout off:', err)
        setState({ url, grid: null, failed: true })
      },
    )
    return () => {
      alive = false
      letGo(url)
    }
    // The URL names the site and the layer.
  }, [url])
  return state.url === url ? { grid: state.grid, failed: state.failed } : { grid: null, failed: false }
}
