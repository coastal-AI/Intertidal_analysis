import { lazy, Suspense } from 'react'
import { ErrorBoundary } from '@/components/layout'
import { Skeleton } from '@/components/ui'
import { config } from '@/config'

// The map viewer is the whole site. It loads in its own chunk (with MapLibre GL JS), after a light first paint.
const Explorer = lazy(() => import('@/features/explorer'))

function MapLoading() {
  return (
    <div className="flex h-full w-full items-center justify-center bg-sunken" role="status">
      <span className="sr-only">Loading the map</span>
      <Skeleton className="h-24 w-64" />
    </div>
  )
}

/** The site: one full-screen map viewer, its state in the URL hash. */
export default function App() {
  return (
    <main className="relative h-dvh w-full overflow-hidden">
      <h1 className="sr-only">MAREA: maps of intertidal elevation and emersion time</h1>
      <ErrorBoundary>
        <Suspense fallback={<MapLoading />}>
          <Explorer defaultSite="villaviciosa" enable3d={config.enable3d} />
        </Suspense>
      </ErrorBoundary>
    </main>
  )
}
