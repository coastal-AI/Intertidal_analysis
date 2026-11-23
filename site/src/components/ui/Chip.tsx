import { clsx } from 'clsx'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

export interface ChipProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Selected state: aria-pressed for a toggle, aria-selected with role="tab", aria-checked with role="radio". */
  selected?: boolean
  icon?: ReactNode
  size?: 'md' | 'sm'
}

/** A pill: a toggle, a tab or a small action (e.g. a reset chip). */
export function Chip({ selected, icon, size = 'md', role, className, children, type = 'button', ...rest }: ChipProps) {
  const state =
    selected === undefined
      ? {}
      : role === 'tab'
        ? { 'aria-selected': selected }
        : role === 'radio'
          ? { 'aria-checked': selected }
          : { 'aria-pressed': selected }
  return (
    <button
      type={type}
      role={role}
      {...state}
      className={clsx(
        'inline-flex shrink-0 items-center gap-1.5 rounded-full border font-medium whitespace-nowrap',
        'transition-colors duration-150 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4',
        size === 'md' ? 'min-h-11 px-4 text-sm' : 'min-h-8 px-3 text-xs',
        selected
          ? 'border-transparent bg-primary text-primary-ink'
          : 'border-line-strong bg-surface text-ink hover:border-control hover:bg-sunken',
        className,
      )}
      {...rest}
    >
      {icon && <span aria-hidden="true" className="inline-flex">{icon}</span>}
      {children}
    </button>
  )
}
