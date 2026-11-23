import { clsx } from 'clsx'
import { useId, type ReactNode } from 'react'

export interface SliderProps {
  id?: string
  label: ReactNode
  min: number
  max: number
  step?: number
  value: number
  onChange: (value: number) => void
  /** Value shown next to the label and read by screen readers, e.g. "120 of 305". */
  valueText?: string
  /** One short line under the slider. */
  hint?: ReactNode
  disabled?: boolean
  className?: string
}

/** A native range input with its label and value (keyboard: arrows, Page Up/Down, Home/End). */
export function Slider({ id, label, min, max, step = 1, value, onChange, valueText, hint, disabled, className }: SliderProps) {
  const autoId = useId()
  const inputId = id ?? autoId
  return (
    <div className={clsx('grid gap-1', className)}>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={inputId} className="text-sm font-medium text-ink">
          {label}
        </label>
        {valueText !== undefined && (
          <output htmlFor={inputId} className="text-sm text-muted tabular-nums">
            {valueText}
          </output>
        )}
      </div>
      <input
        id={inputId}
        type="range"
        className="ui-range h-11 w-full cursor-pointer disabled:cursor-not-allowed"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-valuetext={valueText}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <p className="text-xs text-muted">{hint}</p>}
    </div>
  )
}
