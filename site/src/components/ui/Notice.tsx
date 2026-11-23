import { clsx } from 'clsx'
import { Info, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'

export interface NoticeProps {
  /** "info" (teal) or "warn" (orange). */
  tone?: 'info' | 'warn'
  /** Announce changes to screen readers (for messages that appear after an action). */
  live?: boolean
  children: ReactNode
  className?: string
}

/** One short line of information or warning. */
export function Notice({ tone = 'info', live = false, children, className }: NoticeProps) {
  const Icon = tone === 'warn' ? TriangleAlert : Info
  return (
    <div
      role={live ? 'status' : undefined}
      aria-live={live ? 'polite' : undefined}
      className={clsx(
        'flex items-start gap-2 rounded-lg px-3 py-2 text-sm text-ink',
        tone === 'warn' ? 'bg-signal/10' : 'bg-primary-soft',
        className,
      )}
    >
      <Icon aria-hidden="true" className={clsx('mt-0.5 size-4 shrink-0', tone === 'warn' ? 'text-signal' : 'text-primary')} />
      <div className="min-w-0">{children}</div>
    </div>
  )
}
