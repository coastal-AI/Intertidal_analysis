import { Component, type ErrorInfo, type ReactNode } from 'react'
import { Button } from '@/components/ui'

interface Props {
  children: ReactNode
  /** What to show instead of the broken part (default: one line and a Reload button). */
  fallback?: ReactNode
}

interface State {
  error: Error | null
}

/**
 * Keeps the rest of the site working when a part fails, e.g. a page chunk that cannot be fetched after the site
 * was published again (a reload fetches the new one).
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('MAREA: a part of the page failed', error, info.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      this.props.fallback ?? (
        <div role="alert" className="mx-auto grid max-w-md justify-items-center gap-3 px-4 py-16 text-center">
          <p className="text-ink">This part of the page could not be loaded.</p>
          <Button variant="secondary" onClick={() => window.location.reload()}>
            Reload
          </Button>
        </div>
      )
    )
  }
}
