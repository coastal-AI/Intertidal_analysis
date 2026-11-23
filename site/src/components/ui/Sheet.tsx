import { clsx } from 'clsx'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { useId, useState, type ReactNode } from 'react'

export interface SheetProps {
  /** Accessible name of the panel, e.g. "Map controls". */
  label: string
  /** Always visible: on phones it is the bar of the collapsed panel (e.g. the site name and the readout). */
  header: ReactNode
  /** The body: always shown on wide screens, shown on phones when the panel is expanded. */
  children: ReactNode
  /** Phones only: start expanded. */
  defaultOpen?: boolean
  className?: string
}

/**
 * The map's control panel: a bottom sheet on phones (a bar that expands), a floating card at the top left on wide
 * screens. Place it inside a positioned box that fills the map area (the viewer's root is one).
 */
export function Sheet({ label, header, children, defaultOpen = false, className }: SheetProps) {
  const [open, setOpen] = useState(defaultOpen)
  const bodyId = useId()
  return (
    <section
      aria-label={label}
      className={clsx(
        'absolute inset-x-0 bottom-0 z-20 flex max-h-[70%] flex-col overflow-hidden rounded-t-2xl border-t border-line bg-raised shadow-float',
        'md:inset-x-auto md:top-4 md:bottom-auto md:left-4 md:max-h-[calc(100%-2rem)] md:w-80 md:rounded-xl md:border',
        className,
      )}
    >
      <div className="flex items-center gap-2 py-1.5 pr-1.5 pl-4">
        <div className="min-w-0 flex-1">{header}</div>
        <button
          type="button"
          className="inline-flex size-11 shrink-0 items-center justify-center rounded-full text-ink hover:bg-sunken md:hidden [&_svg]:size-5"
          aria-expanded={open}
          aria-controls={bodyId}
          aria-label={open ? 'Hide the panel' : 'Show the panel'}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <ChevronDown aria-hidden="true" /> : <ChevronUp aria-hidden="true" />}
        </button>
      </div>
      <div id={bodyId} className={clsx('min-h-0 overflow-y-auto overscroll-contain px-4 pb-4', open ? 'block' : 'hidden md:block')}>
        {children}
      </div>
    </section>
  )
}
