/**
 * The viewer's name, its info button and the short panel it opens: the poster's title, the authors and their
 * affiliations, the repository, the data sources with their licences and attributions, and how to cite the
 * repository. Nothing else: the method and the data are described in the repository.
 */
import { clsx } from 'clsx'
import { ArrowUpRight, Info, Moon, Sun, X } from 'lucide-react'
import { useEffect, useId, useRef } from 'react'
import { ButtonAnchor, ExternalLink, IconButton } from '@/components/ui'
import { config, repoFile } from '@/config'
import { conference, dataSources, poster, repositoryCitation } from '@/content/site'
import { assetUrl } from '@/lib/assets'
import { useTheme } from '@/lib/theme'

const ABOUT = 'About MAREA'

export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <IconButton label={`Switch to the ${next} theme`} onClick={() => setTheme(next)} className={className}>
      {theme === 'dark' ? <Sun /> : <Moon />}
    </IconButton>
  )
}

/** Wide screens: the top row of the control panel (name, congress, info button, theme). */
export function BrandRow({ onInfo }: { onInfo: () => void }) {
  return (
    <div className="flex items-center gap-2">
      <div className="min-w-0 flex-1 leading-tight">
        <p className="text-base font-semibold tracking-tight text-ink">MAREA</p>
        <p className="text-[11px] font-semibold tracking-wide text-primary uppercase">{conference.name}</p>
      </div>
      <IconButton label={ABOUT} aria-haspopup="dialog" onClick={onInfo}>
        <Info />
      </IconButton>
      <ThemeToggle />
    </div>
  )
}

/** Phones: the name over the map, as one button that opens the info panel. */
export function BrandPill({ onInfo, className }: { onInfo: () => void; className?: string }) {
  return (
    <button
      type="button"
      aria-label={ABOUT}
      aria-haspopup="dialog"
      title={ABOUT}
      onClick={onInfo}
      className={clsx(
        'inline-flex h-11 items-center gap-2 rounded-full bg-raised pr-3.5 pl-4 text-ink shadow-float ring-1 ring-line',
        'transition-colors duration-150 hover:bg-sunken',
        className,
      )}
    >
      <span className="text-base font-semibold tracking-tight">MAREA</span>
      <Info aria-hidden="true" className="size-4 text-muted" />
    </button>
  )
}

function Sup({ children }: { children: number[] }) {
  return <sup className="ml-px text-[0.7em]">{children.join(',')}</sup>
}

export interface AboutDialogProps {
  open: boolean
  onClose: () => void
  /** List the 3D terrain among the sources (the 3D view is offered here). */
  with3d: boolean
}

/** The info panel: a modal dialog (Escape, the close button or a click outside close it). */
export function AboutDialog({ open, onClose, with3d }: AboutDialogProps) {
  const ref = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  const sourcesId = useId()
  const citeId = useId()

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])

  const sources = dataSources.filter((s) => with3d || !s.only3d)
  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(e) => {
        // A click on the backdrop lands on the dialog itself (its content sits in an inner box).
        if (e.target === e.currentTarget) e.currentTarget.close()
      }}
      className={clsx(
        'm-auto max-h-[calc(100dvh-2rem)] w-[calc(100%-1.5rem)] max-w-lg overflow-y-auto overscroll-contain',
        'rounded-2xl border border-line bg-raised p-0 text-ink shadow-float backdrop:bg-black/50',
      )}
    >
      <div className="grid grid-cols-1 gap-5 p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-primary">{conference.line}</p>
            <h2 id={titleId} className="mt-1.5 text-lg leading-snug">
              {poster.title}
            </h2>
          </div>
          <IconButton label="Close" onClick={() => ref.current?.close()} className="-mt-1.5 -mr-2">
            <X />
          </IconButton>
        </div>

        <p className="text-sm leading-relaxed">
          {poster.authors.map((a, i) => (
            <span key={a.name}>
              {i > 0 && ', '}
              {a.name}
              <Sup>{a.affiliations}</Sup>
            </span>
          ))}
          <br />
          <span className="text-muted">
            {poster.affiliations.map((af, i) => (
              <span key={af.id}>
                {i > 0 && ' · '}
                <Sup>{[af.id]}</Sup> {af.name}
              </span>
            ))}
          </span>
        </p>

        <ButtonAnchor href={config.repoUrl} size="sm" variant="primary" iconRight={<ArrowUpRight />} className="justify-self-start">
          The repository on GitHub
        </ButtonAnchor>

        <section aria-labelledby={sourcesId} className="grid grid-cols-1 gap-2">
          <h3 id={sourcesId} className="text-sm">
            Data sources
          </h3>
          <ul className="grid gap-1.5 text-sm leading-snug">
            {sources.map((s) => (
              <li key={s.label}>
                <span className="font-semibold">{s.label}:</span>{' '}
                <ExternalLink href={s.href} icon={false} className="font-normal">
                  {s.credit}
                </ExternalLink>
                <span className="text-muted"> · {s.licence}</span>
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby={citeId} className="grid grid-cols-1 gap-2">
          <h3 id={citeId} className="text-sm">
            How to cite
          </h3>
          <p className="rounded-lg bg-sunken px-3 py-2 text-sm leading-relaxed break-words select-all">
            {repositoryCitation(config.repoUrl)}
          </p>
        </section>

        <p className="text-xs text-muted">
          Code: <ExternalLink href={repoFile('LICENSE')} icon={false} className="font-normal">MIT</ExternalLink> ·
          bundled software:{' '}
          <a
            href={assetUrl('licenses/THIRD-PARTY-NOTICES.txt')}
            className="text-primary underline decoration-1 underline-offset-[3px] hover:decoration-2"
          >
            licences
          </a>
        </p>
      </div>
    </dialog>
  )
}
