import { clsx } from 'clsx'
import type { ReactNode } from 'react'

export interface BadgeProps {
  children: ReactNode
  /** "brand" (teal tint), "neutral" (grey tint) or "outline". */
  tone?: 'brand' | 'neutral' | 'outline'
  /** "sm" (12 px text, default) or "md" (14 px text, 36 px tall). */
  size?: 'sm' | 'md'
  /** Layout classes only (margins, visibility through a wrapper): size and colour come from the props. */
  className?: string
}

/** A small label, e.g. "CARTOGRASS 2026". */
export function Badge({ children, tone = 'neutral', size = 'sm', className }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full font-semibold whitespace-nowrap',
        size === 'sm' ? 'px-2.5 py-0.5 text-xs [&_svg]:size-3.5' : 'min-h-9 px-3.5 text-sm [&_svg]:size-4',
        tone === 'brand' && 'bg-primary-soft text-primary',
        tone === 'neutral' && 'bg-sunken text-muted',
        tone === 'outline' && 'text-muted ring-1 ring-inset ring-line-strong',
        className,
      )}
    >
      {children}
    </span>
  )
}
