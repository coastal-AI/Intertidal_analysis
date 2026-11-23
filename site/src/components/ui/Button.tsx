import { clsx } from 'clsx'
import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost'
export type ButtonSize = 'md' | 'sm'

const BASE =
  'inline-flex items-center justify-center gap-2 rounded-full font-semibold whitespace-nowrap select-none ' +
  'transition-colors duration-150 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-[1.125em] [&_svg]:shrink-0'

const SIZES: Record<ButtonSize, string> = {
  md: 'min-h-11 px-5 text-base', // 44 px touch target
  sm: 'min-h-9 px-4 text-sm',
}

const VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-primary text-primary-ink shadow-card hover:bg-primary-hover',
  secondary: 'bg-surface text-ink ring-1 ring-inset ring-control hover:bg-sunken',
  ghost: 'text-primary hover:bg-primary-soft',
}

export interface ButtonStyle {
  variant?: ButtonVariant
  size?: ButtonSize
  className?: string
}

/** Class names of a button, for elements that are not one of the components below. */
export function buttonClass({ variant = 'primary', size = 'md', className }: ButtonStyle = {}): string {
  return clsx(BASE, SIZES[size], VARIANTS[variant], className)
}

interface ButtonContent {
  variant?: ButtonVariant
  size?: ButtonSize
  /** Icon before the label (aria-hidden is added). */
  icon?: ReactNode
  /** Icon after the label. */
  iconRight?: ReactNode
}

function Content({ icon, iconRight, children }: { icon?: ReactNode; iconRight?: ReactNode; children?: ReactNode }) {
  return (
    <>
      {icon && <span aria-hidden="true" className="inline-flex">{icon}</span>}
      {children}
      {iconRight && <span aria-hidden="true" className="inline-flex">{iconRight}</span>}
    </>
  )
}

/** A button. */
export function Button({
  variant,
  size,
  icon,
  iconRight,
  className,
  children,
  type = 'button',
  ...rest
}: ButtonContent & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button type={type} className={buttonClass({ variant, size, className })} {...rest}>
      <Content icon={icon} iconRight={iconRight}>{children}</Content>
    </button>
  )
}

/** A link to another site (new tab by default) or to a file, styled as a button. */
export function ButtonAnchor({
  variant,
  size,
  icon,
  iconRight,
  className,
  children,
  external = true,
  ...rest
}: ButtonContent & AnchorHTMLAttributes<HTMLAnchorElement> & { external?: boolean }) {
  const ext = external ? { target: '_blank', rel: 'noopener noreferrer' } : {}
  return (
    <a className={buttonClass({ variant, size, className })} {...ext} {...rest}>
      <Content icon={icon} iconRight={iconRight}>{children}</Content>
      {external && <span className="sr-only"> (opens in a new tab)</span>}
    </a>
  )
}
