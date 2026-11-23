import { clsx } from 'clsx'

/** The MAREA mark: two waves on a rounded square in the theme's primary colour (decorative). */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" focusable="false" className={clsx('shrink-0', className)}>
      <rect width="32" height="32" rx="8" className="fill-primary" />
      <path
        d="M6 12.5c2.5 0 2.5-2.5 5-2.5s2.5 2.5 5 2.5 2.5-2.5 5-2.5 2.5 2.5 5 2.5"
        fill="none"
        strokeWidth="2.4"
        strokeLinecap="round"
        className="stroke-primary-ink"
      />
      <path
        d="M6 20.5c2.5 0 2.5-2.5 5-2.5s2.5 2.5 5 2.5 2.5-2.5 5-2.5 2.5 2.5 5 2.5"
        fill="none"
        strokeWidth="2.4"
        strokeLinecap="round"
        className="stroke-primary-ink opacity-60"
      />
    </svg>
  )
}
