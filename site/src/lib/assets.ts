/**
 * URLs of the site's own files and a cached JSON loader.
 *
 * Every file under public/ is served under the base path (SITE_BASE), so code never writes "/data/..." by hand:
 * assetUrl("figures/x.webp") and dataUrl("sites/index.json") give the right URL for any base.
 */

/**
 * URL of a file of the public folder, under the base path. Absolute URLs, and paths that already start with the
 * base, are returned unchanged, so assetUrl(assetUrl(p)) === assetUrl(p).
 */
export function assetUrl(path: string, base: string = import.meta.env.BASE_URL): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith('//')) return path
  const b = base.endsWith('/') ? base : `${base}/`
  if (b !== '/' && path.startsWith(b)) return path
  return b + path.replace(/^\/+/, '')
}

/** URL of a file of the data folder (public/data/), e.g. dataUrl("sites/index.json"). */
export function dataUrl(path: string, base: string = import.meta.env.BASE_URL): string {
  return assetUrl(`data/${path.replace(/^\/+/, '')}`, base)
}

/** A failed request, with its HTTP status (0 when the network failed). */
export class FetchError extends Error {
  readonly status: number
  readonly url: string
  constructor(url: string, status: number, message?: string) {
    super(message ?? `Request failed (${status || 'network'}): ${url}`)
    this.name = 'FetchError'
    this.status = status
    this.url = url
  }
}

const jsonCache = new Map<string, Promise<unknown>>()

/**
 * Fetch and parse a JSON file once per URL: later calls get the same promise. A failed request is not cached, so
 * a retry fetches again. ``path`` is a URL or a path of the public folder (see assetUrl).
 */
export function loadJSON<T = unknown>(path: string): Promise<T> {
  const url = assetUrl(path)
  const hit = jsonCache.get(url)
  if (hit) return hit as Promise<T>
  const p = fetch(url, { headers: { Accept: 'application/json' } })
    .then(async (res) => {
      if (!res.ok) throw new FetchError(url, res.status)
      return (await res.json()) as T
    })
    .catch((err: unknown) => {
      jsonCache.delete(url)
      if (err instanceof FetchError) throw err
      throw new FetchError(url, 0, err instanceof Error ? err.message : String(err))
    })
  jsonCache.set(url, p)
  return p
}

/** Forget cached JSON (tests, or after a deliberate reload). */
export function clearJSONCache(): void {
  jsonCache.clear()
}
