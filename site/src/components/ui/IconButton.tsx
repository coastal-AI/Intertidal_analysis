import { clsx } from 'clsx'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Accessible name (also the tooltip). */
  label: string
  /** The icon. */
  children: ReactNode
  /** "ghost" on page backgrounds, "surface" over maps. */
  variant?: 'ghost' | 'surface'
}

/** A 44 px square button with one icon. */
export function IconButton({ label, children, variant = 'ghost', className, type = 'button', ...rest }: IconButtonProps) {
  return (
    <button
      type={type}
      aria-label={label}
      title={label}
      className={clsx(
        'inline-flex size-11 shrink-0 items-center justify-center rounded-full text-ink transition-colors duration-150',
        'disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-5',
        variant === 'ghost' ? 'hover:bg-sunken' : 'bg-raised shadow-float ring-1 ring-line hover:bg-sunken',
        className,
      )}
      {...rest}
    >
      <span aria-hidden="true" className="inline-flex">{children}</span>
    </button>
  )
}
