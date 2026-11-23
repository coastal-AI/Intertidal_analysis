/**
 * Number and unit formats of the map viewer: English, decimal point, the real minus sign (U+2212), a no-break space
 * before units.
 */

export const MINUS = '−'
/** No-break space, between a number and its unit. */
export const NBSP = ' '
/** En dash, for ranges ("2023–2025"). */
export const EN_DASH = '–'

export interface NumberFormat {
  /** Digits after the decimal point (default 2). */
  decimals?: number
  /** Write "+" before positive values (default false). */
  sign?: boolean
}

/** A number with a fixed number of decimals and the real minus sign: −0.46, +0.10, 1,234.5. */
export function fmtNumber(value: number, { decimals = 2, sign = false }: NumberFormat = {}): string {
  if (!Number.isFinite(value)) return EN_DASH
  const abs = Math.abs(value).toLocaleString('en-GB', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
  const zero = Number(abs.replace(/,/g, '')) === 0
  if (value < 0 && !zero) return MINUS + abs
  return sign && !zero ? `+${abs}` : abs
}

/** A percentage: "42 %" (no decimals by default). */
export function fmtPercent(value: number, decimals = 0): string {
  return `${fmtNumber(value, { decimals })}${NBSP}%`
}

/** Minutes: "7 min". */
export function fmtMinutes(value: number, decimals = 0): string {
  return `${fmtNumber(value, { decimals })}${NBSP}min`
}

/** A whole number with thousands separators: "29,728". */
export function fmtInt(value: number): string {
  return Number.isFinite(value) ? Math.round(value).toLocaleString('en-GB') : EN_DASH
}
