import { useEffect, useState } from 'react'

/** State of an asynchronous resource. */
export type Resource<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  | { status: 'error'; error: Error }

/**
 * Run ``load`` when ``key`` changes and follow its promise. ``load`` should be cached (e.g. loadJSON), so pages that
 * mount again get the data at once. A null key means "nothing to load" and stays in "loading".
 */
export function useResource<T>(key: string | null, load: () => Promise<T>): Resource<T> {
  const [state, setState] = useState<Resource<T>>({ status: 'loading' })
  useEffect(() => {
    if (key === null) return
    let alive = true
    setState({ status: 'loading' })
    load().then(
      (data) => alive && setState({ status: 'ready', data }),
      (error: unknown) => alive && setState({ status: 'error', error: error instanceof Error ? error : new Error(String(error)) }),
    )
    return () => {
      alive = false
    }
    // Only ``key`` re-runs the effect: ``load`` is expected to change with it.
  }, [key])
  return state
}
