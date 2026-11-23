import { clsx } from 'clsx'
import type { ReactNode } from 'react'

export interface SegmentedOption<T extends string> {
  value: T
  label: ReactNode
  disabled?: boolean
}

export interface SegmentedProps<T extends string> {
  /** Name of the radio group (unique on the page). */
  name: string
  /** What is being chosen, e.g. "Basemap". */
  legend: string
  /** Show the legend above the control (otherwise it is read by screen readers only). */
  showLegend?: boolean
  options: readonly SegmentedOption<T>[]
  value: T
  onChange: (value: T) => void
  size?: 'md' | 'sm'
  /** Stretch to the width of the parent, options sharing it equally. */
  fullWidth?: boolean
  className?: string
}

/**
 * A segmented control: native radio buttons (arrow keys move the choice, Tab leaves the group) drawn as one pill.
 */
export function Segmented<T extends string>({
  name,
  legend,
  showLegend = false,
  options,
  value,
  onChange,
  size = 'md',
  fullWidth = false,
  className,
}: SegmentedProps<T>) {
  return (
    <fieldset className={clsx('min-w-0', className)}>
      <legend className={showLegend ? 'mb-1.5 text-sm font-medium text-muted' : 'sr-only'}>{legend}</legend>
      <div className={clsx('gap-1 rounded-full bg-sunken p-1', fullWidth ? 'flex w-full' : 'inline-flex')}>
        {options.map((o) => (
          <label
            key={o.value}
            className={clsx(
              'relative flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-full px-3.5 text-center font-medium',
              'text-muted transition-colors duration-150 hover:text-ink [&_svg]:size-4',
              'has-[:checked]:bg-surface has-[:checked]:text-ink has-[:checked]:shadow-card',
              'has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-focus',
              'has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-50',
              size === 'md' ? 'min-h-11 text-sm' : 'min-h-9 text-sm',
            )}
          >
            <input
              type="radio"
              className="sr-only"
              name={name}
              value={o.value}
              checked={value === o.value}
              disabled={o.disabled}
              onChange={() => onChange(o.value)}
            />
            {o.label}
          </label>
        ))}
      </div>
    </fieldset>
  )
}
