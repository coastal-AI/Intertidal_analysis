// WCAG 2.2 contrast of the design tokens, in both themes: text at least 4.5:1, controls and chart marks at least 3:1.
// The tokens are read from index.css itself (the :root block is the light theme, the .dark block the dark theme).
import { describe, expect, it } from 'vitest'
import css from './index.css?raw'

type Tokens = Record<string, string>

function hexTokens(block: RegExp): Tokens {
  const m = block.exec(css)
  if (!m) throw new Error(`no block ${block} in index.css`)
  const out: Tokens = {}
  for (const d of m[1].matchAll(/--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\b/g)) out[d[1]] = d[2]
  return out
}

const light = hexTokens(/(?:^|\n):root\s*\{([^}]*)\}/)
const darkOnly = hexTokens(/(?:^|\n)\s*\.dark\s*\{([^}]*)\}/)
const dark: Tokens = { ...light, ...darkOnly }

function channel(c: number): number {
  const s = c / 255
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
}

function luminance(hex: string): number {
  const n = Number.parseInt(hex.slice(1), 16)
  return 0.2126 * channel((n >> 16) & 255) + 0.7152 * channel((n >> 8) & 255) + 0.0722 * channel(n & 255)
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((p, q) => q - p)
  return (hi + 0.05) / (lo + 0.05)
}

const TEXT = 4.5
const UI = 3

/** [foreground token, background token, minimum ratio]. */
const PAIRS: [string, string, number][] = [
  // body and secondary text on every surface
  ['ink', 'canvas', TEXT], ['ink', 'surface', TEXT], ['ink', 'raised', TEXT], ['ink', 'sunken', TEXT],
  ['muted', 'canvas', TEXT], ['muted', 'surface', TEXT], ['muted', 'raised', TEXT], ['muted', 'sunken', TEXT],
  // links, buttons and selected chips
  ['primary', 'canvas', TEXT], ['primary', 'surface', TEXT], ['primary', 'raised', TEXT], ['primary', 'sunken', TEXT],
  ['primary', 'primary-soft', TEXT], ['ink', 'primary-soft', TEXT],
  ['primary-ink', 'primary', TEXT], ['primary-ink', 'primary-hover', TEXT],
  // form control borders, focus ring, accents and chart marks
  ['control', 'canvas', UI], ['control', 'surface', UI], ['control', 'raised', UI],
  ['focus', 'canvas', UI], ['focus', 'surface', UI], ['focus', 'raised', UI],
  ['accent', 'surface', UI], ['signal', 'surface', UI],
]

describe('design tokens', () => {
  it('the dark theme defines every colour token of the light theme', () => {
    expect(Object.keys(darkOnly).sort()).toEqual(Object.keys(light).sort())
  })

  it.each([
    ['light', light],
    ['dark', dark],
  ] as const)('%s theme: every pair meets its contrast minimum', (_name, tokens) => {
    const failures: string[] = []
    for (const [fg, bg, min] of PAIRS) {
      expect(tokens[fg], `token --${fg}`).toMatch(/^#[0-9a-fA-F]{6}$/)
      expect(tokens[bg], `token --${bg}`).toMatch(/^#[0-9a-fA-F]{6}$/)
      const ratio = contrast(tokens[fg], tokens[bg])
      if (ratio < min) failures.push(`--${fg} on --${bg}: ${ratio.toFixed(2)} < ${min}`)
    }
    expect(failures).toEqual([])
  })

  it('computes the WCAG ratio (black on white is 21:1)', () => {
    expect(contrast('#000000', '#FFFFFF')).toBeCloseTo(21, 5)
  })
})
