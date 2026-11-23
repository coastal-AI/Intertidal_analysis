import { clsx } from 'clsx'
import { ArrowUpRight } from 'lucide-react'
import type { AnchorHTMLAttributes, ReactNode } from 'react'

export interface ExternalLinkProps extends AnchorHTMLAttributes<HTMLAnchorElement> {
  href: string
  children: ReactNode
  /** Show the arrow icon (default true). */
  icon?: boolean
}

/** A link to another site, opened in a new tab and announced as such. Works inside running text. */
export function ExternalLink({ href, children, icon = true, className, ...rest }: ExternalLinkProps) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={clsx(
        'font-medium text-primary underline decoration-1 underline-offset-[3px] hover:decoration-2',
        className,
      )}
      {...rest}
    >
      {children}
      {icon && <ArrowUpRight aria-hidden="true" className="ml-0.5 inline-block size-[0.95em] align-[-0.1em]" />}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  )
}
