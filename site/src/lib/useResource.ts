import { useEffect, useState } from 'react'

/** State of an asynchronous resource. */
export type Resource<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  | { status: 'error'; error: Error }

const LOADING = { status: 'loading' } as const

/**
 * Run ``load`` when ``key`` changes and follow its promise. ``load`` should be cached (e.g. loadJSON), so pages that
 * mount again get the data at once. A null key means "nothing to load" and stays in "loading". The state is kept with
 * its key, so a new key reads as "loading" from its first render, never as the last key's data or error.
 */
export function useResource<T>(key: string | null, load: () => Promise<T>): Resource<T> {
  const [state, setState] = useState<{ key: string | null; resource: Resource<T> }>({ key: null, resource: LOADING })
  useEffect(() => {
    if (key === null) return
    let alive = true
    load().then(
      (data) => alive && setState({ key, resource: { status: 'ready', data } }),
      (error: unknown) =>
        alive && setState({ key, resource: { status: 'error', error: error instanceof Error ? error : new Error(String(error)) } }),
    )
    return () => {
      alive = false
    }
    // Only ``key`` re-runs the effect: ``load`` is expected to change with it.
  }, [key])
  return state.key === key ? state.resource : LOADING
}
