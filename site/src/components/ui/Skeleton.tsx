import { clsx } from 'clsx'

/** A grey placeholder block shown while content loads (hidden from screen readers). */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={clsx('animate-pulse rounded-lg bg-line/70', className)} />
}
